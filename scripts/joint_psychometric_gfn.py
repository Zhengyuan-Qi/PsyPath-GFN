#!/usr/bin/env python3
"""Joint psychometric measurement and counterfactual evidence GFlowNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from scripts.meter_gfn import (
    EvidenceTrajectory,
    HierarchicalClinicalGFlowNet,
    RewardTerms,
    random_matched_mask,
)
from scripts.meter_mh_online_operator import VIEW_NAMES


@dataclass
class JointRewardTerms:
    log_reward: torch.Tensor
    classification: torch.Tensor
    severity: torch.Tensor
    sufficiency: torch.Tensor
    necessity: torch.Tensor
    specificity: torch.Tensor
    compactness: torch.Tensor


def joint_event_features(
    projected: Mapping[str, torch.Tensor], temporal: torch.Tensor
) -> torch.Tensor:
    clinical = projected["clinical"]
    answer = projected["answer"]
    context = projected["context"]
    visual = projected["visual"]
    if temporal.shape != (len(answer), 4):
        raise ValueError(f"invalid temporal features: {tuple(temporal.shape)}")
    return torch.cat(
        (
            clinical,
            answer,
            context,
            visual,
            clinical - answer,
            clinical - context,
            answer - context,
            temporal.to(answer.dtype),
        ),
        dim=-1,
    )


class JointPsychometricMeasurement(nn.Module):
    """Measure binary risk and continuous severity from an evidence subgraph."""

    def __init__(
        self,
        event_dim: int,
        visual_dim: int,
        *,
        hidden_dim: int,
        phenotype_count: int,
        score_max: float,
    ) -> None:
        super().__init__()
        self.score_max = float(score_max)
        self.view_encoders = nn.ModuleDict(
            {
                "clinical": nn.Linear(event_dim, hidden_dim),
                "answer": nn.Linear(event_dim, hidden_dim),
                "context": nn.Linear(event_dim, hidden_dim),
                "visual": nn.Linear(visual_dim, hidden_dim),
            }
        )
        self.temporal_encoder = nn.Linear(4, hidden_dim)
        self.sequence = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.phenotype_embedding = nn.Embedding(phenotype_count, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.10),
        )
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.severity_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        projected: Mapping[str, torch.Tensor],
        temporal: torch.Tensor,
        evidence_mask: torch.Tensor,
        *,
        phenotype: int | None,
    ) -> dict[str, torch.Tensor]:
        if evidence_mask.shape != (len(projected["answer"]), len(VIEW_NAMES)):
            raise ValueError(f"invalid evidence mask: {tuple(evidence_mask.shape)}")
        active = evidence_mask.any(dim=1)
        if not bool(active.any()):
            raise ValueError("measurement requires at least one active evidence event")
        encoded = torch.zeros(
            (len(active), self.risk_head.in_features),
            dtype=projected["answer"].dtype,
            device=projected["answer"].device,
        )
        count = evidence_mask.sum(dim=1).clamp_min(1).to(encoded.dtype)
        for index, name in enumerate(VIEW_NAMES):
            encoded = encoded + self.view_encoders[name](projected[name]) * evidence_mask[:, index, None]
        encoded = encoded / count[:, None]
        encoded = encoded + self.temporal_encoder(temporal.to(encoded.dtype))
        encoded = encoded * active[:, None]
        sequence, _ = self.sequence(encoded[None, :, :])
        sequence = sequence.squeeze(0)
        logits = self.attention(sequence).squeeze(-1).masked_fill(~active, -torch.inf)
        weights = torch.softmax(logits, dim=0)
        pooled = (weights[:, None] * sequence).sum(dim=0)
        if phenotype is None:
            route = torch.zeros(
                self.phenotype_embedding.embedding_dim,
                dtype=pooled.dtype,
                device=pooled.device,
            )
        else:
            route = self.phenotype_embedding.weight[int(phenotype)]
        hidden = self.fusion(torch.cat((pooled, route)))
        risk_logit = self.risk_head(hidden).squeeze(-1)
        severity = torch.sigmoid(self.severity_head(hidden).squeeze(-1)) * self.score_max
        return {
            "logit": risk_logit,
            "probability": torch.sigmoid(risk_logit),
            "severity": severity,
            "attention": weights,
        }


def all_evidence_mask(projected: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.ones(
        (len(projected["answer"]), len(VIEW_NAMES)),
        dtype=torch.bool,
        device=projected["answer"].device,
    )


def complement_mask(mask: torch.Tensor) -> torch.Tensor:
    complement = ~mask
    if not bool(complement.any()):
        complement = torch.ones_like(mask)
    return complement


def target_cross_entropy(
    logit: torch.Tensor, label: torch.Tensor, positive_weight: float
) -> torch.Tensor:
    weight = torch.tensor(
        float(positive_weight), dtype=logit.dtype, device=logit.device
    )
    return F.binary_cross_entropy_with_logits(logit, label, pos_weight=weight)


def psychometric_losses(
    measurement: JointPsychometricMeasurement,
    projected: Mapping[str, torch.Tensor],
    temporal: torch.Tensor,
    trajectory: EvidenceTrajectory,
    *,
    label: int,
    score: float | None,
    random_seed: int,
    pim_weight: float,
    ensr_weight: float,
    sufficiency_weight: float,
    necessity_margin: float,
    specificity_margin: float,
    positive_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dtype = projected["answer"].dtype
    device = projected["answer"].device
    target = torch.tensor(float(label), dtype=dtype, device=device)
    full = measurement(
        projected,
        temporal,
        all_evidence_mask(projected),
        phenotype=None,
    )
    selected = measurement(
        projected,
        temporal,
        trajectory.mask,
        phenotype=trajectory.phenotype,
    )
    full_classification = target_cross_entropy(full["logit"], target, positive_weight)
    sufficiency = target_cross_entropy(selected["logit"], target, positive_weight)
    severity = torch.zeros_like(full_classification)
    if score is not None and pim_weight > 0:
        score_target = torch.tensor(float(score), dtype=dtype, device=device)
        severity = F.smooth_l1_loss(
            full["severity"] / measurement.score_max,
            score_target / measurement.score_max,
        ) + F.smooth_l1_loss(
            selected["severity"] / measurement.score_max,
            score_target / measurement.score_max,
        )
    necessity = torch.zeros_like(full_classification)
    specificity = torch.zeros_like(full_classification)
    selected_delete = None
    random_delete = None
    if ensr_weight > 0:
        selected_delete = measurement(
            projected,
            temporal,
            complement_mask(trajectory.mask),
            phenotype=trajectory.phenotype,
        )
        random_mask = random_matched_mask(trajectory.mask, random_seed)
        random_delete = measurement(
            projected,
            temporal,
            complement_mask(random_mask),
            phenotype=trajectory.phenotype,
        )
        selected_deletion_cost = target_cross_entropy(
            selected_delete["logit"], target, positive_weight
        ) - full_classification
        random_deletion_cost = target_cross_entropy(
            random_delete["logit"], target, positive_weight
        ) - full_classification
        necessity = F.relu(float(necessity_margin) - selected_deletion_cost)
        specificity = F.relu(
            random_deletion_cost - selected_deletion_cost + float(specificity_margin)
        )
    total = (
        full_classification
        + float(sufficiency_weight) * sufficiency
        + float(pim_weight) * severity
        + float(ensr_weight) * (necessity + specificity)
    )
    return total, {
        "full": full,
        "selected": selected,
        "selected_delete": selected_delete,
        "random_delete": random_delete,
        "full_classification": full_classification,
        "sufficiency": sufficiency,
        "severity": severity,
        "necessity": necessity,
        "specificity": specificity,
    }


@torch.no_grad()
def joint_clinical_log_reward(
    measurement: JointPsychometricMeasurement,
    projected: Mapping[str, torch.Tensor],
    temporal: torch.Tensor,
    trajectory: EvidenceTrajectory,
    *,
    label: int,
    score: float | None,
    random_seed: int,
    pim_weight: float,
    ensr_weight: float,
    compactness_weight: float,
    sufficiency_weight: float,
    necessity_margin: float,
    specificity_margin: float,
    reward_temperature: float,
    positive_weight: float = 1.0,
) -> JointRewardTerms:
    _, blocks = psychometric_losses(
        measurement,
        projected,
        temporal,
        trajectory,
        label=label,
        score=score,
        random_seed=random_seed,
        pim_weight=pim_weight,
        ensr_weight=ensr_weight,
        sufficiency_weight=sufficiency_weight,
        necessity_margin=necessity_margin,
        specificity_margin=specificity_margin,
        positive_weight=positive_weight,
    )
    compactness = trajectory.mask.any(dim=1).to(projected["answer"].dtype).mean()
    total = (
        blocks["full_classification"]
        + float(sufficiency_weight) * blocks["sufficiency"]
        + float(pim_weight) * blocks["severity"]
        + float(ensr_weight) * (blocks["necessity"] + blocks["specificity"])
        + float(compactness_weight) * compactness
    )
    return JointRewardTerms(
        log_reward=-total / max(float(reward_temperature), 1e-6),
        classification=blocks["full_classification"],
        severity=blocks["severity"],
        sufficiency=blocks["sufficiency"],
        necessity=blocks["necessity"],
        specificity=blocks["specificity"],
        compactness=compactness,
    )


def joint_trajectory_balance_loss(
    gflownet: HierarchicalClinicalGFlowNet,
    events: torch.Tensor,
    trajectories: list[EvidenceTrajectory],
    rewards: list[JointRewardTerms],
) -> torch.Tensor:
    log_z = gflownet.partition(events)
    residual = torch.stack(
        [
            log_z + trajectory.log_forward - trajectory.log_backward - reward.log_reward
            for trajectory, reward in zip(trajectories, rewards)
        ]
    )
    return residual.square().mean()
