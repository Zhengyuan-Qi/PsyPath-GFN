#!/usr/bin/env python3
"""Task-specific end-to-end extension above a locked formal PsyPath anchor."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from scripts.meter_gfn import (
    HierarchicalClinicalGFlowNet,
    RewardTerms,
    trajectory_balance_loss,
)
from scripts.meter_mh_online_operator import VIEW_NAMES
from scripts.psypath_measurement import FrozenAlternatingPsychometricOperator
from scripts.psypath_online_model import (
    FrozenPsyPathAnchor,
    compatibility_event_features,
    evidence_features_torch,
)


class TrainableFormalPsychometricOperator(nn.Module):
    """Trainable copy initialized exactly from a locked formal PIM operator."""

    def __init__(self, frozen: FrozenAlternatingPsychometricOperator) -> None:
        super().__init__()
        if frozen.mode != "pim":
            raise ValueError("formal E2E requires a PIM psychometric operator")
        if frozen.severity_pipeline is None or frozen.calibrator is None:
            raise ValueError("incomplete frozen PIM operator")
        self.score_max = float(frozen.score_max)
        self.severity_pipeline = copy.deepcopy(frozen.severity_pipeline)
        self.calibrator = copy.deepcopy(frozen.calibrator)
        for name in ("coefficient", "intercept"):
            value = getattr(self.severity_pipeline, name)
            del self.severity_pipeline._buffers[name]
            self.severity_pipeline.register_parameter(
                name, nn.Parameter(value.detach().clone())
            )
        for name in ("coefficient", "intercept"):
            value = getattr(self.calibrator, name)
            del self.calibrator._buffers[name]
            self.calibrator.register_parameter(name, nn.Parameter(value.detach().clone()))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim == 1:
            features = features[None, :]
        raw_severity = self.severity_pipeline(features)
        logit = self.calibrator.logits(raw_severity)
        return {
            "probability": torch.sigmoid(logit).clamp(1e-6, 1.0 - 1e-6),
            "logit": logit,
            "severity": raw_severity.clamp(0.0, self.score_max),
        }


class FormalPsyPathEndToEndModel(nn.Module):
    """Exact formal anchor plus task-specific policy and psychometric increments."""

    def __init__(self, anchor: FrozenPsyPathAnchor) -> None:
        super().__init__()
        if not isinstance(anchor.policy, HierarchicalClinicalGFlowNet):
            raise TypeError("formal E2E requires a hierarchical clinical GFlowNet anchor")
        self.anchor = anchor
        self.policy_shared_projection = copy.deepcopy(anchor.shared_projection)
        self.policy_visual_projection = copy.deepcopy(anchor.visual_projection)
        self.policy = copy.deepcopy(anchor.policy)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(True)
        self.measurement = TrainableFormalPsychometricOperator(anchor.measurement)
        self.risk_scale = nn.Parameter(torch.zeros(()))
        self.severity_scale = nn.Parameter(torch.zeros(()))
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

    def move_trainable(self, device: torch.device | str) -> "FormalPsyPathEndToEndModel":
        """Move the trainable graph while retaining the exact CPU anchor."""

        self.policy.to(device)
        self.measurement.to(device)
        self.policy_shared_projection.to(device)
        self.policy_visual_projection.to(device)
        self.risk_scale.data = self.risk_scale.data.to(device)
        self.severity_scale.data = self.severity_scale.data.to(device)
        return self

    def forward(
        self,
        base_raw: Mapping[str, torch.Tensor],
        temporal: torch.Tensor,
        *,
        samples: int,
        policy_seeds: Sequence[int],
        anchor_seeds: Sequence[int],
        adapted_raw: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        if len(policy_seeds) != int(samples) or len(anchor_seeds) != int(samples):
            raise ValueError("policy and anchor seed lists must match samples")
        anchor_device = next(self.anchor.parameters()).device
        anchor_raw = {
            name: value.detach().to(device=anchor_device)
            for name, value in base_raw.items()
        }
        anchor = self.anchor(
            anchor_raw,
            temporal.detach().to(device=anchor_device),
            samples=samples,
            seed=0,
            sample_seeds=anchor_seeds,
        )
        policy_device = next(self.policy.parameters()).device
        anchor_logit = anchor["logit"].to(device=policy_device)
        anchor_severity = anchor["severity"].to(device=policy_device)
        policy_raw = adapted_raw if adapted_raw is not None else base_raw
        projected = {
            "clinical": self.policy_shared_projection(policy_raw["clinical"]),
            "answer": self.policy_shared_projection(policy_raw["answer"]),
            "context": self.policy_shared_projection(policy_raw["context"]),
            "visual": self.policy_visual_projection(policy_raw["visual"]),
        }
        events = compatibility_event_features(projected, temporal).to(torch.float32)
        encoded, context = self.policy.encode(events)
        minimum, maximum = self.anchor.event_limits(len(events))
        paths = []
        predictions = []
        for path_seed in policy_seeds:
            generator = torch.Generator(device=events.device.type)
            generator.manual_seed(int(path_seed))
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
        neural_probability = probability.mean()
        neural_logit = torch.logit(neural_probability.clamp(1e-6, 1.0 - 1e-6))
        neural_severity = severity.mean()
        online_logit = anchor_logit + torch.tanh(self.risk_scale) * (
            neural_logit - anchor_logit.detach()
        )
        online_severity = anchor_severity + torch.tanh(self.severity_scale) * (
            neural_severity - anchor_severity.detach()
        )
        return {
            "anchor_probability": anchor["probability"].to(device=policy_device),
            "anchor_logit": anchor_logit,
            "anchor_severity": anchor_severity,
            "anchor_path_probability_std": anchor["path_probability_std"].to(
                device=policy_device
            ),
            "anchor_path_severity_std": anchor["path_severity_std"].to(
                device=policy_device
            ),
            "events": events,
            "context": context,
            "paths": paths,
            "predictions": predictions,
            "neural_probability": neural_probability,
            "neural_severity": neural_severity,
            "online_logit": online_logit,
            "online_probability": torch.sigmoid(online_logit),
            "online_severity": online_severity,
            "path_probability_std": probability.std(correction=0),
            "path_severity_std": severity.std(correction=0),
        }

    def training_loss(
        self,
        output: Mapping[str, Any],
        *,
        label: float,
        score: float | None,
        score_max: float,
        reward_temperature: float,
        severity_weight: float = 0.30,
        anchor_weight: float = 0.20,
        anchor_severity_weight: float = 0.20,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        label_tensor = output["online_logit"].new_tensor(float(label))
        measurement_losses = []
        rewards = []
        for prediction in output["predictions"]:
            classification = F.binary_cross_entropy_with_logits(
                prediction["logit"].squeeze(0), label_tensor
            )
            severity = classification.new_zeros(())
            if score is not None:
                severity_target = prediction["severity"].new_tensor(
                    float(score) / float(score_max)
                )
                severity = F.smooth_l1_loss(
                    prediction["severity"].squeeze(0) / float(score_max),
                    severity_target,
                )
            measurement = classification + float(severity_weight) * severity
            measurement_losses.append(measurement)
            rewards.append(
                RewardTerms(
                    log_reward=-measurement.detach()
                    / max(float(reward_temperature), 1e-6),
                    classification=classification.detach(),
                    severity=severity.detach(),
                    necessity=classification.new_zeros(()),
                    specificity=classification.new_zeros(()),
                    compactness=classification.new_zeros(()),
                )
            )
        measurement_loss = torch.stack(measurement_losses).mean()
        tb_loss = trajectory_balance_loss(
            self.policy,
            output["events"],
            list(output["paths"]),
            rewards,
            log_z=self.policy.partition_from_context(output["context"]),
        )
        anchor_loss = F.binary_cross_entropy_with_logits(
            output["online_logit"], label_tensor
        )
        anchor_severity_loss = anchor_loss.new_zeros(())
        if score is not None:
            anchor_severity_loss = F.smooth_l1_loss(
                output["online_severity"] / float(score_max),
                output["online_severity"].new_tensor(float(score) / float(score_max)),
            )
        total = (
            measurement_loss
            + tb_loss
            + float(anchor_weight) * anchor_loss
            + float(anchor_severity_weight) * anchor_severity_loss
        )
        return total, {
            "measurement": measurement_loss,
            "trajectory_balance": tb_loss,
            "anchor": anchor_loss,
            "anchor_severity": anchor_severity_loss,
        }
