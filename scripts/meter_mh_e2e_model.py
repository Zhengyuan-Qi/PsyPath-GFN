#!/usr/bin/env python3
"""Trainable METER-MH-E2E measurement network.

The HumanOmniV2 backbone produces online event representations.  This module
contains only the participant-level clinical measurement operator, so it can
be unit-tested without loading the foundation model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


TASK_DEPRESSION = 0
TASK_PTSD = 1
TASK_NAMES = {TASK_DEPRESSION: "depression", TASK_PTSD: "ptsd"}
TASK_MAX_SCORE = {TASK_DEPRESSION: 24.0, TASK_PTSD: 85.0}
TASK_SCREENING_CUTOFF = {TASK_DEPRESSION: 10.0, TASK_PTSD: 44.0}
M1_ANSWER_DYNAMIC_ABLATIONS = {"m1_answer_dynamic", "m1_time_shuffle"}
M1_STATIC_CONTROLS = {"m1_static_continue"}


@dataclass(frozen=True)
class LossWeights:
    binary: float = 1.0
    severity: float = 0.30
    process: float = 0.12
    evidence: float = 0.12
    calibration: float = 0.08
    teacher: float = 0.35
    noninferiority: float = 0.20
    uncertainty: float = 0.02


class MeterE2EHead(nn.Module):
    """Shared response-measurement operator for all clinical tasks."""

    def __init__(
        self,
        hidden_size: int = 3584,
        visual_size: int = 1280,
        model_size: int = 256,
        transformer_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.visual_size = int(visual_size)
        self.model_size = int(model_size)

        self.view_norm = nn.LayerNorm(hidden_size)
        self.shared_view_projection = nn.Linear(hidden_size, model_size)
        self.view_identity = nn.Parameter(torch.zeros(3, model_size))
        nn.init.normal_(self.view_identity, std=0.02)

        relation_size = model_size * 7
        self.response_relation = nn.Sequential(
            nn.LayerNorm(relation_size),
            nn.Linear(relation_size, model_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_size * 2, model_size),
        )
        self.visual_adapter = nn.Sequential(
            nn.LayerNorm(visual_size),
            nn.Linear(visual_size, model_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_size * 2, model_size),
        )
        self.visual_reliability = nn.Sequential(
            nn.LayerNorm(model_size * 3 + 1),
            nn.Linear(model_size * 3 + 1, model_size),
            nn.GELU(),
            nn.Linear(model_size, 1),
        )
        self.timing_adapter = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, model_size),
            nn.GELU(),
            nn.Linear(model_size, model_size),
        )
        self.task_embedding = nn.Embedding(2, model_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_size,
            nhead=8,
            dim_feedforward=model_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.dynamic_adapter = nn.Sequential(
            nn.LayerNorm(model_size * 3),
            nn.Linear(model_size * 3, model_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_size * 2, model_size),
        )
        # M1 is a residual extension of matched_tav. A zero gate makes its
        # initial screening predictions exactly equal to the M0 TAV path.
        self.answer_dynamic_scale = nn.Parameter(torch.ones(()))
        # Reconstruct the complete five-channel timing/availability vector
        # used by the runner.  Keeping this contract explicit prevents a
        # silent broadcast or a late loss-shape failure during training.
        self.process_reconstruction = nn.Linear(model_size, 5)
        self.evidence_head = nn.Sequential(
            nn.LayerNorm(model_size),
            nn.Linear(model_size, model_size // 2),
            nn.GELU(),
            nn.Linear(model_size // 2, 1),
        )
        pooled_size = model_size * 4
        self.participant_projection = nn.Sequential(
            nn.LayerNorm(pooled_size),
            nn.Linear(pooled_size, model_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_size * 2, model_size),
        )
        self.binary_head = nn.Linear(model_size, 1)
        self.severity_head = nn.Linear(model_size, 1)
        self.uncertainty_head = nn.Linear(model_size, 1)

    def initialize_answer_dynamic_residual(self) -> None:
        """Make the M1 residual exactly zero while keeping first-step gradients."""
        final = self.dynamic_adapter[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("dynamic_adapter must end in a linear residual projection")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            self.answer_dynamic_scale.fill_(1.0)

    @staticmethod
    def _temporal_position_encoding(
        length: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, width, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(width, 1))
        )
        encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequency)
        if width > 1:
            encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=dtype)

    def _project_views(
        self,
        clinical: torch.Tensor,
        answer: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = []
        for index, value in enumerate((clinical, answer, context)):
            values.append(
                self.shared_view_projection(self.view_norm(value))
                + self.view_identity[index]
            )
        return values[0], values[1], values[2]

    def encode_events(
        self,
        clinical: torch.Tensor,
        answer: torch.Tensor,
        context: torch.Tensor,
        visual: torch.Tensor,
        timing: torch.Tensor,
        task_id: int,
        ablation: str = "none",
        temporal_permutation: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if clinical.ndim != 2 or answer.shape != clinical.shape or context.shape != clinical.shape:
            raise ValueError("clinical, answer, and context must be aligned [events, hidden]")
        if visual.ndim != 2 or len(visual) != len(clinical):
            raise ValueError("visual must be aligned [events, visual_hidden]")
        if timing.shape != (len(clinical), 5):
            raise ValueError("timing must be [events, 5]")

        clinical_p, answer_p, context_p = self._project_views(clinical, answer, context)
        if ablation in {"matched_ta", "matched_tav", *M1_STATIC_CONTROLS}:
            # Capacity-matched ordinary controls: retain the same online TA
            # event inputs and, for TAV, the same audited visual stream, but do
            # not use temporal differences, evidence weighting, or timing.
            response = clinical_p
            if ablation in {"matched_tav", *M1_STATIC_CONTROLS}:
                visual_state = self.visual_adapter(visual)
                visual_available = visual.abs().sum(dim=-1, keepdim=True).gt(0).to(visual.dtype)
                response = response + visual_available * visual_state
            return {
                "event_states": response,
                "evidence_logits": torch.zeros(
                    len(response), device=response.device, dtype=response.dtype
                ),
                "process_prediction": torch.zeros(
                    len(response), 5, device=response.device, dtype=response.dtype
                ),
                "visual_reliability": torch.zeros(
                    len(response), device=response.device, dtype=response.dtype
                ),
            }
        if ablation in M1_ANSWER_DYNAMIC_ABLATIONS:
            # Preserve the complete M0 TAV path as an explicit anchor. The
            # Answer-Dynamic branch can only modify it through a bounded,
            # zero-initialized residual.
            visual_available = visual.abs().sum(dim=-1, keepdim=True).gt(0).to(visual.dtype)
            visual_state = self.visual_adapter(visual)
            anchor = clinical_p + visual_available * visual_state

            # Keep the extra stochastic branch from advancing the global RNG.
            # This preserves the exact matched_tav dropout stream when the
            # residual scale is zero, including in training mode.
            fork_devices = [anchor.device.index or 0] if anchor.is_cuda else []
            with torch.random.fork_rng(devices=fork_devices, enabled=self.training):
                relation = torch.cat(
                    (
                        clinical_p,
                        answer_p,
                        context_p,
                        clinical_p - answer_p,
                        clinical_p - context_p,
                        (clinical_p - answer_p).abs(),
                        (clinical_p - context_p).abs(),
                    ),
                    dim=-1,
                )
                response = self.response_relation(relation)
                task = self.task_embedding(
                    torch.full((len(anchor),), int(task_id), device=anchor.device, dtype=torch.long)
                )
                dynamic_input = response + visual_available * visual_state + self.timing_adapter(timing) + task

                inverse_permutation = None
                if temporal_permutation is not None:
                    if temporal_permutation.shape != (len(anchor),):
                        raise ValueError("temporal_permutation must contain one index per event")
                    temporal_permutation = temporal_permutation.to(device=anchor.device, dtype=torch.long)
                    if not torch.equal(
                        torch.sort(temporal_permutation).values,
                        torch.arange(len(anchor), device=anchor.device),
                    ):
                        raise ValueError("temporal_permutation must be a valid event permutation")
                    inverse_permutation = torch.argsort(temporal_permutation)
                    dynamic_input = dynamic_input[temporal_permutation]

                dynamic_input = dynamic_input + self._temporal_position_encoding(
                    len(dynamic_input),
                    dynamic_input.shape[-1],
                    device=dynamic_input.device,
                    dtype=dynamic_input.dtype,
                )
                states = self.temporal_encoder(dynamic_input.unsqueeze(0)).squeeze(0)
                delta = torch.zeros_like(states)
                if len(states) > 1:
                    delta[1:] = states[1:] - states[:-1]
                dynamic = self.dynamic_adapter(torch.cat((states, delta, delta.abs()), dim=-1))
                if inverse_permutation is not None:
                    dynamic = dynamic[inverse_permutation]
            event_states = anchor + torch.tanh(self.answer_dynamic_scale) * dynamic
            return {
                "event_states": event_states,
                "evidence_logits": torch.zeros(
                    len(event_states), device=event_states.device, dtype=event_states.dtype
                ),
                "process_prediction": self.process_reconstruction(dynamic),
                "visual_reliability": visual_available.squeeze(-1),
            }
        if ablation == "no_context":
            context_p = torch.zeros_like(context_p)
        relation = torch.cat(
            (
                clinical_p,
                answer_p,
                context_p,
                clinical_p - answer_p,
                clinical_p - context_p,
                (clinical_p - answer_p).abs(),
                (clinical_p - context_p).abs(),
            ),
            dim=-1,
        )
        response = self.response_relation(relation)

        visual_available = visual.abs().sum(dim=-1, keepdim=True).gt(0).to(visual.dtype)
        visual_state = self.visual_adapter(visual)
        reliability_input = torch.cat(
            (response, visual_state, (response - visual_state).abs(), visual_available),
            dim=-1,
        )
        reliability = torch.sigmoid(self.visual_reliability(reliability_input))
        if ablation == "no_visual":
            reliability = torch.zeros_like(reliability)
        fused = response + reliability * visual_available * visual_state
        fused = fused + self.timing_adapter(timing)

        task = self.task_embedding(
            torch.full((len(fused),), int(task_id), device=fused.device, dtype=torch.long)
        )
        states = self.temporal_encoder((fused + task).unsqueeze(0)).squeeze(0)
        delta = torch.zeros_like(states)
        if len(states) > 1:
            delta[1:] = states[1:] - states[:-1]
        if ablation == "no_answer_dynamics":
            delta = torch.zeros_like(delta)
        dynamic = self.dynamic_adapter(torch.cat((states, delta, delta.abs()), dim=-1))
        event_states = states + dynamic
        return {
            "event_states": event_states,
            "evidence_logits": self.evidence_head(event_states).squeeze(-1),
            "process_prediction": self.process_reconstruction(event_states),
            "visual_reliability": reliability.squeeze(-1),
        }

    def participant_readout(
        self,
        event_states: torch.Tensor,
        evidence_logits: torch.Tensor,
        keep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if keep is None:
            keep = torch.ones(len(event_states), dtype=torch.bool, device=event_states.device)
        if keep.ndim != 1 or len(keep) != len(event_states):
            raise ValueError("keep must be an event mask")
        if not bool(keep.any()):
            keep = keep.clone()
            keep[torch.argmax(evidence_logits)] = True
        ordinary_pool = bool(torch.all(evidence_logits == 0))
        masked_logits = evidence_logits.masked_fill(~keep, torch.finfo(evidence_logits.dtype).min)
        selected = event_states[keep]
        if ordinary_pool:
            evidence_weight = keep.to(event_states.dtype) / keep.to(event_states.dtype).sum().clamp_min(1.0)
            weighted = selected.mean(dim=0)
        else:
            evidence_weight = torch.softmax(masked_logits, dim=0)
            weighted = (evidence_weight.unsqueeze(-1) * event_states).sum(dim=0)
        mean = selected.mean(dim=0)
        std = selected.std(dim=0, unbiased=False)
        boundary = torch.zeros_like(mean) if ordinary_pool else selected[-1] - selected[0]
        participant = self.participant_projection(torch.cat((weighted, mean, std, boundary)))
        return {
            "participant": participant,
            "logit": self.binary_head(participant).squeeze(-1),
            "severity_unit": torch.sigmoid(self.severity_head(participant).squeeze(-1)),
            "log_variance": self.uncertainty_head(participant).squeeze(-1).clamp(-4.0, 4.0),
            "evidence_weight": evidence_weight,
        }

    def forward(
        self,
        clinical: torch.Tensor,
        answer: torch.Tensor,
        context: torch.Tensor,
        visual: torch.Tensor,
        timing: torch.Tensor,
        task_id: int,
        ablation: str = "none",
        temporal_permutation: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_events(
            clinical,
            answer,
            context,
            visual,
            timing,
            task_id,
            ablation=ablation,
            temporal_permutation=temporal_permutation,
        )
        readout = self.participant_readout(
            encoded["event_states"], encoded["evidence_logits"]
        )
        return {**encoded, **readout}

    def intervention_readouts(
        self,
        output: dict[str, torch.Tensor],
        random_seed: int,
    ) -> dict[str, torch.Tensor]:
        states = output["event_states"]
        evidence = output["evidence_logits"]
        count = max(1, int(math.ceil(0.20 * len(states))))
        selected = torch.topk(evidence.detach(), k=min(count, len(states))).indices
        generator = torch.Generator(device=states.device)
        generator.manual_seed(int(random_seed))
        random = torch.randperm(len(states), generator=generator, device=states.device)[:count]
        selected_delete = torch.ones(len(states), dtype=torch.bool, device=states.device)
        selected_delete[selected] = False
        random_delete = torch.ones(len(states), dtype=torch.bool, device=states.device)
        random_delete[random] = False
        selected_only = torch.zeros(len(states), dtype=torch.bool, device=states.device)
        selected_only[selected] = True
        return {
            "selected_delete": self.participant_readout(states, evidence, selected_delete)["logit"],
            "random_delete": self.participant_readout(states, evidence, random_delete)["logit"],
            "selected_only": self.participant_readout(states, evidence, selected_only)["logit"],
        }


def measurement_loss(
    model: MeterE2EHead,
    output: dict[str, torch.Tensor],
    *,
    label: float,
    task_id: int,
    score: float | None,
    teacher_probability: float | None,
    timing_target: torch.Tensor,
    timing_mask: torch.Tensor,
    positive_weight: float,
    random_seed: int,
    ablation: str = "none",
    weights: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, float]]:
    device = output["logit"].device
    dtype = output["logit"].dtype
    target = torch.tensor(float(label), device=device, dtype=dtype)
    pos_weight = torch.tensor(float(positive_weight), device=device, dtype=dtype)
    binary_vector = F.binary_cross_entropy_with_logits(
        output["logit"], target, pos_weight=pos_weight, reduction="none"
    )
    uncertainty = torch.exp(-output["log_variance"]) * binary_vector + output["log_variance"]
    binary = binary_vector
    probability = torch.sigmoid(output["logit"])
    calibration = (probability - target).square()

    max_score = TASK_MAX_SCORE[int(task_id)]
    predicted_score = output["severity_unit"] * max_score
    severity = torch.zeros((), device=device, dtype=dtype)
    pim = torch.zeros((), device=device, dtype=dtype)
    matched_control = ablation in {"matched_ta", "matched_tav", *M1_STATIC_CONTROLS}
    m1_answer_dynamic = ablation in M1_ANSWER_DYNAMIC_ABLATIONS
    if (
        score is not None
        and math.isfinite(float(score))
        and ablation != "no_pim"
        and not matched_control
        and not m1_answer_dynamic
    ):
        score_target = torch.tensor(float(score), device=device, dtype=dtype)
        severity = F.smooth_l1_loss(predicted_score / max_score, score_target / max_score)
        cutoff = TASK_SCREENING_CUTOFF[int(task_id)]
        score_risk = torch.sigmoid((predicted_score - cutoff) / 2.0)
        pim = (probability - score_risk).square()

    process = torch.zeros((), device=device, dtype=dtype)
    if ablation != "no_answer_dynamics" and not matched_control and bool(timing_mask.any()):
        process = F.smooth_l1_loss(
            output["process_prediction"][timing_mask], timing_target[timing_mask]
        )

    evidence = torch.zeros((), device=device, dtype=dtype)
    if (
        ablation != "no_v7"
        and ablation != "no_ensr"
        and not matched_control
        and not m1_answer_dynamic
    ):
        interventions = model.intervention_readouts(output, random_seed)
        direction = target * 2.0 - 1.0
        selected_drop = direction * (output["logit"] - interventions["selected_delete"])
        random_drop = direction * (output["logit"] - interventions["random_delete"])
        ranking = F.relu(torch.tensor(0.05, device=device, dtype=dtype) - selected_drop + random_drop)
        sufficiency = F.binary_cross_entropy_with_logits(interventions["selected_only"], target)
        evidence = ranking + 0.5 * sufficiency

    teacher = torch.zeros((), device=device, dtype=dtype)
    noninferiority = torch.zeros((), device=device, dtype=dtype)
    if (
        teacher_probability is not None
        and ablation != "no_teacher"
        and not matched_control
        and not m1_answer_dynamic
    ):
        teacher_target = torch.tensor(
            float(teacher_probability), device=device, dtype=dtype
        ).clamp(1e-4, 1.0 - 1e-4)
        teacher = F.binary_cross_entropy_with_logits(output["logit"], teacher_target)
        teacher_label_loss = F.binary_cross_entropy(teacher_target, target)
        noninferiority = F.relu(binary_vector - teacher_label_loss - 0.01)

    total = (
        weights.binary * binary
        + weights.severity * (severity + pim)
        + weights.process * process
        + weights.evidence * evidence
        + weights.calibration * calibration
        + weights.teacher * teacher
        + weights.noninferiority * noninferiority
        + weights.uncertainty * uncertainty
    )
    values = {
        "total": float(total.detach()),
        "binary": float(binary.detach()),
        "severity": float(severity.detach()),
        "pim": float(pim.detach()),
        "process": float(process.detach()),
        "evidence": float(evidence.detach()),
        "calibration": float(calibration.detach()),
        "teacher": float(teacher.detach()),
        "noninferiority": float(noninferiority.detach()),
        "probability": float(probability.detach()),
        "predicted_score": float(predicted_score.detach()),
    }
    return total, values


def trainable_parameter_summary(module: nn.Module) -> dict[str, Any]:
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in module.parameters())
    return {
        "trainable": int(trainable),
        "total": int(total),
        "fraction": float(trainable / max(total, 1)),
    }
