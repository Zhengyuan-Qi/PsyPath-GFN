#!/usr/bin/env python3
"""Online migration and end-to-end extension for PsyPath-GFN."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from scripts.meter_gfn import EvidenceTrajectory, HierarchicalClinicalGFlowNet
from scripts.meter_mh_online_operator import FrozenStandardPCA, VIEW_NAMES, summarize
from scripts.psypath_gfn import (
    EvidencePath,
    MonotonicPsychometricGFlowNet,
    MultimodalMonotonicPsychometricGFlowNet,
    trajectory_balance_loss,
)
from scripts.psypath_measurement import (
    CrossFittedPsychometricMeasurementBank,
    FrozenAlternatingPsychometricOperator,
    InstrumentConditionedPsychometricMeasurement,
    PsychometricLossWeights,
    psychometric_log_reward,
    psychometric_measurement_loss,
)


@dataclass(frozen=True)
class PsyPathE2ELossWeights:
    measurement: float = 1.0
    trajectory_balance: float = 1.0
    anchor: float = 0.20
    anchor_severity: float = 0.20


def summarize_subset(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if mask.ndim != 1 or len(mask) != len(values):
        raise ValueError("subset mask must contain one value per event")
    if not bool(mask.any()):
        return torch.zeros(
            values.shape[1] * 8, dtype=values.dtype, device=values.device
        )
    return summarize(values[mask])


def evidence_features_torch(
    projected: Mapping[str, torch.Tensor], evidence_mask: torch.Tensor
) -> torch.Tensor:
    """Differentiable equivalent of cached ``evidence_features``."""

    length = len(projected["answer"])
    if evidence_mask.shape != (length, len(VIEW_NAMES)):
        raise ValueError(f"invalid evidence mask: {tuple(evidence_mask.shape)}")
    mask = evidence_mask.to(dtype=torch.bool, device=projected["answer"].device)
    blocks = [
        summarize_subset(projected[name], mask[:, index])
        for index, name in enumerate(VIEW_NAMES)
    ]
    for left, right in (
        ("clinical", "answer"),
        ("clinical", "context"),
        ("answer", "context"),
    ):
        joint = mask[:, VIEW_NAMES.index(left)] & mask[:, VIEW_NAMES.index(right)]
        blocks.append(summarize_subset(projected[left] - projected[right], joint))
    dtype = projected["answer"].dtype
    modality_fraction = mask.to(dtype).mean(dim=0)
    event_fraction = mask.any(dim=1).to(dtype).mean().reshape(1)
    modality_count = (
        mask.sum(dim=1).to(dtype).mean() / float(len(VIEW_NAMES))
    ).reshape(1)
    return torch.nan_to_num(
        torch.cat((*blocks, modality_fraction, event_fraction, modality_count))
    )


def projected_event_nodes(
    projected: Mapping[str, torch.Tensor], temporal: torch.Tensor
) -> torch.Tensor:
    """Build channel-preserving online evidence nodes.

    The first three channels are audited HumanOmniV2 evidence views and the
    fourth is the official visual stream. They are evidence channels, not a
    claim that every HumanOmniV2 view is a physically independent modality.
    """

    values = [projected[name] for name in VIEW_NAMES]
    if len({tuple(value.shape) for value in values}) != 1:
        raise ValueError("projected evidence channels must be aligned")
    if temporal.ndim != 2 or len(temporal) != len(values[0]):
        raise ValueError("temporal features must align with evidence events")
    temporal = temporal.to(device=values[0].device, dtype=values[0].dtype)
    return torch.stack(
        [torch.cat((value, temporal), dim=-1) for value in values], dim=1
    )


def compatibility_event_features(
    projected: Mapping[str, torch.Tensor], temporal: torch.Tensor
) -> torch.Tensor:
    clinical = projected["clinical"]
    answer = projected["answer"]
    context = projected["context"]
    visual = projected["visual"]
    temporal = temporal.to(device=answer.device, dtype=answer.dtype)
    return torch.cat(
        (
            clinical,
            answer,
            context,
            visual,
            clinical - answer,
            clinical - context,
            answer - context,
            temporal,
        ),
        dim=-1,
    )


class FrozenPsyPathAnchor(nn.Module):
    """Frozen online graph reproducing a locked cached PsyPath checkpoint."""

    def __init__(
        self,
        *,
        shared_projection: FrozenStandardPCA,
        visual_projection: FrozenStandardPCA,
        policy: MonotonicPsychometricGFlowNet | HierarchicalClinicalGFlowNet,
        measurement: FrozenAlternatingPsychometricOperator,
        instrument_id: int,
        minimum_events: int,
        maximum_events: int,
        maximum_fraction: float,
    ) -> None:
        super().__init__()
        self.shared_projection = shared_projection
        self.visual_projection = visual_projection
        self.policy = policy
        self.measurement = measurement
        self.instrument_id = int(instrument_id)
        self.minimum_events = int(minimum_events)
        self.maximum_events = int(maximum_events)
        self.maximum_fraction = float(maximum_fraction)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        instrument_id: int,
        dtype: torch.dtype = torch.float64,
    ) -> "FrozenPsyPathAnchor":
        projections = payload["event_projections"]
        shared = projections["shared"]
        visual = projections["visual"]
        shared_scaler, shared_reducer = (
            shared
            if isinstance(shared, tuple)
            else (shared.scaler, shared.reducer)
        )
        visual_scaler, visual_reducer = (
            visual
            if isinstance(visual, tuple)
            else (visual.scaler, visual.reducer)
        )
        state = payload["gflownet_state"]
        input_dim = int(state["event_encoder.0.weight"].shape[1])
        hidden_dim = int(state["event_encoder.0.weight"].shape[0])
        if "phenotype_head.weight" in state:
            phenotype_count = int(state["phenotype_head.weight"].shape[0])
            policy = HierarchicalClinicalGFlowNet(
                input_dim,
                hidden_dim=hidden_dim,
                phenotype_count=phenotype_count,
            )
        elif "bundle_encoder.0.weight" in state:
            view_count = int(state["bundle_encoder.0.weight"].shape[1])
            instrument_count = int(state["instrument_embedding.weight"].shape[0])
            policy = MonotonicPsychometricGFlowNet(
                input_dim,
                hidden_dim=hidden_dim,
                view_count=view_count,
                instrument_count=instrument_count,
                instrument_id=instrument_id,
            )
        else:
            keys = sorted(state)[:20]
            raise ValueError(
                "unsupported PsyPath checkpoint policy; expected phenotype_head "
                f"or bundle_encoder keys, observed={keys}"
            )
        policy.load_state_dict(state, strict=True)
        config = dict(payload.get("config", {}))
        return cls(
            shared_projection=FrozenStandardPCA(
                shared_scaler, shared_reducer, dtype=dtype
            ),
            visual_projection=FrozenStandardPCA(
                visual_scaler, visual_reducer, dtype=dtype
            ),
            policy=policy.to(dtype=torch.float32),
            measurement=FrozenAlternatingPsychometricOperator(
                payload["psychometric_operator"], dtype=dtype
            ),
            instrument_id=instrument_id,
            minimum_events=int(config.get("minimum_events", 2)),
            maximum_events=int(config.get("maximum_events", 16)),
            maximum_fraction=float(config.get("maximum_fraction", 0.30)),
        )

    def project_views(
        self, raw: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            "clinical": self.shared_projection(raw["clinical"]),
            "answer": self.shared_projection(raw["answer"]),
            "context": self.shared_projection(raw["context"]),
            "visual": self.visual_projection(raw["visual"]),
        }

    def event_limits(self, count: int) -> tuple[int, int]:
        minimum = min(count, max(1, self.minimum_events))
        maximum = min(
            count,
            max(minimum, int(math.ceil(self.maximum_fraction * count))),
            self.maximum_events,
        )
        return minimum, max(minimum, maximum)

    def forward(
        self,
        raw: Mapping[str, torch.Tensor],
        temporal: torch.Tensor,
        *,
        samples: int,
        seed: int,
        sample_seeds: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        if int(samples) < 1:
            raise ValueError("samples must be positive")
        if sample_seeds is not None and len(sample_seeds) != int(samples):
            raise ValueError(
                f"sample_seeds length {len(sample_seeds)} != samples {int(samples)}"
            )
        projected = self.project_views(raw)
        events = compatibility_event_features(projected, temporal).to(torch.float32)
        encoded, context = self.policy.encode(events)
        minimum, maximum = self.event_limits(len(events))
        paths: list[EvidencePath | EvidenceTrajectory] = []
        predictions = []
        for sample in range(int(samples)):
            generator = torch.Generator(device=events.device.type)
            path_seed = (
                int(sample_seeds[sample])
                if sample_seeds is not None
                else int(seed) + sample * 1009
            )
            generator.manual_seed(path_seed)
            path = self.policy.trajectory_from_encoding(
                encoded,
                context,
                minimum_events=minimum,
                maximum_events=maximum,
                generator=generator,
            )
            features = evidence_features_torch(projected, path.mask)
            predictions.append(self.measurement(features))
            paths.append(path)
        probability = torch.stack(
            [prediction["probability"].squeeze(0) for prediction in predictions]
        )
        severity = torch.stack(
            [prediction["severity"].squeeze(0) for prediction in predictions]
        )
        mean_probability = probability.mean()
        return {
            "probability": mean_probability,
            "logit": torch.logit(mean_probability.clamp(1e-6, 1.0 - 1e-6)),
            "severity": severity.mean(),
            "path_probability_std": probability.std(correction=0),
            "path_severity_std": severity.std(correction=0),
            "projected": projected,
            "paths": paths,
        }


class PsyPathEndToEndModel(nn.Module):
    """Frozen exact anchor plus a zero-impact trainable PsyPath extension."""

    def __init__(
        self,
        anchor: FrozenPsyPathAnchor,
        *,
        temporal_dim: int,
        score_maxima: tuple[float, ...],
        hidden_dim: int = 96,
        measurement_folds: int = 4,
    ) -> None:
        super().__init__()
        self.anchor = anchor
        projected_dim = int(anchor.shared_projection.components.shape[0])
        node_dim = projected_dim + int(temporal_dim)
        self.policy = MultimodalMonotonicPsychometricGFlowNet(
            node_dim,
            hidden_dim=hidden_dim,
            channel_count=len(VIEW_NAMES),
            instrument_count=len(score_maxima),
        )
        prototype = InstrumentConditionedPsychometricMeasurement(
            node_dim,
            hidden_dim=hidden_dim,
            channel_count=len(VIEW_NAMES),
            temporal_dim=temporal_dim,
            score_maxima=score_maxima,
        )
        self.measurement_bank = CrossFittedPsychometricMeasurementBank(
            prototype, folds=measurement_folds
        )
        self.risk_scale = nn.Parameter(torch.zeros(()))
        self.severity_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        raw: Mapping[str, torch.Tensor],
        temporal: torch.Tensor,
        *,
        instrument_id: int,
        samples: int,
        seed: int,
        anchor_samples: int | None = None,
        anchor_sample_seeds: Sequence[int] | None = None,
        reward_fold: int | None = None,
    ) -> dict[str, Any]:
        anchor = self.anchor(
            raw,
            temporal,
            samples=int(anchor_samples or samples),
            seed=seed,
            sample_seeds=anchor_sample_seeds,
        )
        projected = anchor["projected"]
        nodes = projected_event_nodes(projected, temporal).to(torch.float32)
        minimum, maximum = self.anchor.event_limits(len(nodes))
        encoded, context = self.policy.encode(nodes)
        paths = []
        measurements = []
        for sample in range(int(samples)):
            generator = torch.Generator(device=nodes.device.type)
            generator.manual_seed(int(seed) + sample * 2017)
            path = self.policy.trajectory_from_encoding(
                encoded,
                context,
                instrument_id=instrument_id,
                minimum_events=minimum,
                maximum_events=maximum,
                generator=generator,
            )
            if reward_fold is None:
                measured = self.measurement_bank.posterior(
                    nodes,
                    temporal.to(nodes.dtype),
                    path.mask,
                    instrument_id=instrument_id,
                )
                measured = {
                    "probability": measured["probability"],
                    "risk_logit": measured["risk_logit"],
                    "severity": measured["severity"],
                    "severity_unit": measured["severity"]
                    / self.measurement_bank.models[0].score_maxima[
                        int(instrument_id)
                    ].to(measured["severity"].dtype),
                    "log_variance": torch.log(
                        measured["severity_std"].square().clamp_min(1e-6)
                    ),
                    "active_fraction": path.mask.to(nodes.dtype).mean(),
                    "posterior_members": measured["members"],
                }
            else:
                measured = self.measurement_bank.held_out(
                    reward_fold,
                    nodes,
                    temporal.to(nodes.dtype),
                    path.mask,
                    instrument_id=instrument_id,
                )
                measured["fit_outputs"] = self.measurement_bank.fit_outputs(
                    reward_fold,
                    nodes,
                    temporal.to(nodes.dtype),
                    path.mask,
                    instrument_id=instrument_id,
                )
                measured["prediction_probability"] = torch.stack(
                    [item["probability"] for item in measured["fit_outputs"]]
                ).mean()
                measured["prediction_severity"] = torch.stack(
                    [item["severity"] for item in measured["fit_outputs"]]
                ).mean()
            measurements.append(measured)
            paths.append(path)
        neural_probability = torch.stack(
            [item.get("prediction_probability", item["probability"]) for item in measurements]
        ).mean()
        path_probability = torch.stack(
            [item.get("prediction_probability", item["probability"]) for item in measurements]
        )
        neural_logit = torch.logit(neural_probability.clamp(1e-6, 1.0 - 1e-6))
        path_severity = torch.stack(
            [item.get("prediction_severity", item["severity"]) for item in measurements]
        )
        neural_severity = path_severity.mean()
        online_logit = anchor["logit"] + torch.tanh(self.risk_scale) * (
            neural_logit - anchor["logit"].detach()
        )
        online_severity = anchor["severity"] + torch.tanh(self.severity_scale) * (
            neural_severity - anchor["severity"].detach()
        )
        return {
            "anchor": anchor,
            "nodes": nodes,
            "context": context,
            "paths": paths,
            "measurements": measurements,
            "neural_probability": neural_probability,
            "neural_probability_std": path_probability.std(correction=0),
            "neural_severity": neural_severity,
            "neural_severity_std": path_severity.std(correction=0),
            "online_logit": online_logit,
            "online_probability": torch.sigmoid(online_logit),
            "online_severity": online_severity,
        }

    def training_loss(
        self,
        output: Mapping[str, Any],
        *,
        instrument_id: int,
        label: float,
        score: float | None,
        score_max: float,
        reward_temperature: float,
        reward_fold: int,
        loss_weights: PsyPathE2ELossWeights = PsyPathE2ELossWeights(),
        measurement_weights: PsychometricLossWeights = PsychometricLossWeights(),
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        measurement_losses = []
        rewards = []
        for measured in output["measurements"]:
            fit_outputs = measured.get("fit_outputs")
            if fit_outputs is None:
                raise ValueError("training output is missing cross-fitted fit outputs")
            for fit_output in fit_outputs:
                loss, _ = psychometric_measurement_loss(
                    fit_output,
                    label=label,
                    score=score,
                    score_max=score_max,
                    weights=measurement_weights,
                )
                measurement_losses.append(loss)
            rewards.append(
                psychometric_log_reward(
                    measured,
                    label=label,
                    score=score,
                    score_max=score_max,
                    temperature=reward_temperature,
                    weights=measurement_weights,
                )
            )
        if not measurement_losses:
            raise RuntimeError(f"no fit-side measurement losses for fold {reward_fold}")
        measurement_loss = torch.stack(measurement_losses).mean()
        tb_loss = trajectory_balance_loss(
            self.policy,
            list(output["paths"]),
            rewards,
            context=output["context"],
            instrument_id=instrument_id,
        )
        label_tensor = output["online_logit"].new_tensor(float(label))
        anchor_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["online_logit"], label_tensor
        )
        anchor_severity_loss = output["online_logit"].new_zeros(())
        if score is not None:
            score_tensor = output["online_severity"].new_tensor(
                float(score) / float(score_max)
            )
            anchor_severity_loss = torch.nn.functional.smooth_l1_loss(
                output["online_severity"] / float(score_max),
                score_tensor,
            )
        total = (
            float(loss_weights.measurement) * measurement_loss
            + float(loss_weights.trajectory_balance) * tb_loss
            + float(loss_weights.anchor) * anchor_loss
            + float(loss_weights.anchor_severity) * anchor_severity_loss
        )
        return total, {
            "measurement": measurement_loss,
            "trajectory_balance": tb_loss,
            "anchor": anchor_loss,
            "anchor_severity": anchor_severity_loss,
        }


class MultiTaskPsyPathEndToEndModel(nn.Module):
    """One shared PsyPath policy/measurement model with task-specific anchors."""

    def __init__(
        self,
        anchors: Mapping[str, FrozenPsyPathAnchor],
        *,
        temporal_dim: int,
        score_maxima: tuple[float, ...],
        hidden_dim: int = 96,
        measurement_folds: int = 4,
    ) -> None:
        super().__init__()
        if not anchors:
            raise ValueError("at least one frozen task anchor is required")
        self.anchors = nn.ModuleDict(dict(anchors))
        projected_dims = {
            int(anchor.shared_projection.components.shape[0])
            for anchor in anchors.values()
        }
        if len(projected_dims) != 1:
            raise ValueError(f"task anchor dimensions differ: {sorted(projected_dims)}")
        projected_dim = next(iter(projected_dims))
        node_dim = projected_dim + int(temporal_dim)
        self.task_channel_alignment = nn.ModuleDict(
            {
                task_name: nn.ModuleList(
                    [nn.Linear(projected_dim, projected_dim) for _ in VIEW_NAMES]
                )
                for task_name in anchors
            }
        )
        with torch.no_grad():
            identity = torch.eye(projected_dim)
            for adapters in self.task_channel_alignment.values():
                for adapter in adapters:
                    adapter.weight.copy_(identity)
                    adapter.bias.zero_()
        self.policy = MultimodalMonotonicPsychometricGFlowNet(
            node_dim,
            hidden_dim=hidden_dim,
            channel_count=len(VIEW_NAMES),
            instrument_count=len(score_maxima),
        )
        prototype = InstrumentConditionedPsychometricMeasurement(
            node_dim,
            hidden_dim=hidden_dim,
            channel_count=len(VIEW_NAMES),
            temporal_dim=temporal_dim,
            score_maxima=score_maxima,
        )
        self.measurement_bank = CrossFittedPsychometricMeasurementBank(
            prototype, folds=measurement_folds
        )
        self.risk_scale = nn.Parameter(torch.zeros(len(anchors)))
        self.severity_scale = nn.Parameter(torch.zeros(len(anchors)))
        self.task_index = {name: index for index, name in enumerate(anchors)}

    def aligned_views(
        self,
        task_name: str,
        projected: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        adapters = self.task_channel_alignment[task_name]
        return {
            name: adapters[index](projected[name].to(adapters[index].weight.dtype))
            for index, name in enumerate(VIEW_NAMES)
        }

    def forward(
        self,
        raw: Mapping[str, torch.Tensor],
        temporal: torch.Tensor,
        *,
        task_name: str,
        instrument_id: int,
        samples: int,
        seed: int,
        reward_fold: int | None = None,
        anchor_samples: int | None = None,
        anchor_sample_seeds: Sequence[int] | None = None,
        adapted_raw: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        if task_name not in self.anchors:
            raise KeyError(f"unknown PsyPath task: {task_name}")
        anchor_module = self.anchors[task_name]
        anchor = anchor_module(
            raw,
            temporal,
            samples=int(anchor_samples or samples),
            seed=seed,
            sample_seeds=anchor_sample_seeds,
        )
        policy_projected = (
            anchor_module.project_views(adapted_raw)
            if adapted_raw is not None
            else anchor["projected"]
        )
        policy_projected = self.aligned_views(task_name, policy_projected)
        nodes = projected_event_nodes(policy_projected, temporal).to(torch.float32)
        minimum, maximum = anchor_module.event_limits(len(nodes))
        encoded, context = self.policy.encode(nodes)
        paths = []
        measurements = []
        for sample in range(int(samples)):
            generator = torch.Generator(device=nodes.device.type)
            generator.manual_seed(int(seed) + sample * 2017)
            path = self.policy.trajectory_from_encoding(
                encoded,
                context,
                instrument_id=instrument_id,
                minimum_events=minimum,
                maximum_events=maximum,
                generator=generator,
            )
            if reward_fold is None:
                posterior = self.measurement_bank.posterior(
                    nodes,
                    temporal.to(nodes.dtype),
                    path.mask,
                    instrument_id=instrument_id,
                )
                score_max = self.measurement_bank.models[0].score_maxima[
                    int(instrument_id)
                ].to(posterior["severity"].dtype)
                measured = {
                    "probability": posterior["probability"],
                    "risk_logit": posterior["risk_logit"],
                    "severity": posterior["severity"],
                    "severity_unit": posterior["severity"] / score_max,
                    "log_variance": torch.log(
                        posterior["severity_std"].square().clamp_min(1e-6)
                    ),
                    "active_fraction": path.mask.to(nodes.dtype).mean(),
                    "posterior_members": posterior["members"],
                }
            else:
                measured = self.measurement_bank.held_out(
                    reward_fold,
                    nodes,
                    temporal.to(nodes.dtype),
                    path.mask,
                    instrument_id=instrument_id,
                )
                measured["fit_outputs"] = self.measurement_bank.fit_outputs(
                    reward_fold,
                    nodes,
                    temporal.to(nodes.dtype),
                    path.mask,
                    instrument_id=instrument_id,
                )
                measured["prediction_probability"] = torch.stack(
                    [item["probability"] for item in measured["fit_outputs"]]
                ).mean()
                measured["prediction_severity"] = torch.stack(
                    [item["severity"] for item in measured["fit_outputs"]]
                ).mean()
            paths.append(path)
            measurements.append(measured)

        path_probability = torch.stack(
            [item.get("prediction_probability", item["probability"]) for item in measurements]
        )
        neural_probability = path_probability.mean()
        neural_logit = torch.logit(neural_probability.clamp(1e-6, 1.0 - 1e-6))
        path_severity = torch.stack(
            [item.get("prediction_severity", item["severity"]) for item in measurements]
        )
        neural_severity = path_severity.mean()
        task_index = self.task_index[task_name]
        online_logit = anchor["logit"] + torch.tanh(self.risk_scale[task_index]) * (
            neural_logit - anchor["logit"].detach()
        )
        online_severity = anchor["severity"] + torch.tanh(
            self.severity_scale[task_index]
        ) * (neural_severity - anchor["severity"].detach())
        return {
            "anchor": anchor,
            "nodes": nodes,
            "context": context,
            "paths": paths,
            "measurements": measurements,
            "neural_probability": neural_probability,
            "neural_probability_std": path_probability.std(correction=0),
            "neural_severity": neural_severity,
            "neural_severity_std": path_severity.std(correction=0),
            "online_logit": online_logit,
            "online_probability": torch.sigmoid(online_logit),
            "online_severity": online_severity,
        }

    def training_loss(
        self,
        output: Mapping[str, Any],
        *,
        instrument_id: int,
        label: float,
        score: float | None,
        score_max: float,
        reward_temperature: float,
        reward_fold: int,
        loss_weights: PsyPathE2ELossWeights = PsyPathE2ELossWeights(),
        measurement_weights: PsychometricLossWeights = PsychometricLossWeights(),
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        measurement_losses = []
        rewards = []
        for measured in output["measurements"]:
            fit_outputs = measured.get("fit_outputs")
            if fit_outputs is None:
                raise ValueError("training output is missing cross-fitted fit outputs")
            for fit_output in fit_outputs:
                fit_loss, _ = psychometric_measurement_loss(
                    fit_output,
                    label=label,
                    score=score,
                    score_max=score_max,
                    weights=measurement_weights,
                )
                measurement_losses.append(fit_loss)
            rewards.append(
                psychometric_log_reward(
                    measured,
                    label=label,
                    score=score,
                    score_max=score_max,
                    temperature=reward_temperature,
                    weights=measurement_weights,
                )
            )
        if not measurement_losses:
            raise RuntimeError(f"no fit-side measurement losses for fold {reward_fold}")
        measurement_loss = torch.stack(measurement_losses).mean()
        tb_loss = trajectory_balance_loss(
            self.policy,
            list(output["paths"]),
            rewards,
            context=output["context"],
            instrument_id=instrument_id,
        )
        label_tensor = output["online_logit"].new_tensor(float(label))
        anchor_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["online_logit"], label_tensor
        )
        anchor_severity_loss = output["online_logit"].new_zeros(())
        if score is not None:
            score_tensor = output["online_severity"].new_tensor(
                float(score) / float(score_max)
            )
            anchor_severity_loss = torch.nn.functional.smooth_l1_loss(
                output["online_severity"] / float(score_max),
                score_tensor,
            )
        total = (
            float(loss_weights.measurement) * measurement_loss
            + float(loss_weights.trajectory_balance) * tb_loss
            + float(loss_weights.anchor) * anchor_loss
            + float(loss_weights.anchor_severity) * anchor_severity_loss
        )
        return total, {
            "measurement": measurement_loss,
            "trajectory_balance": tb_loss,
            "anchor": anchor_loss,
            "anchor_severity": anchor_severity_loss,
        }
