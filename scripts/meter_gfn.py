#!/usr/bin/env python3
"""Core components for METER-GFN cached clinical evidence inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from scripts.meter_mh_online_operator import (
    VIEW_NAMES,
    FrozenMeterPIM,
    event_matrix,
    summarize,
    summarize_subset,
)


MODALITY_BUNDLES = torch.tensor(
    [[bool(code & (1 << index)) for index in range(len(VIEW_NAMES))] for code in range(1, 16)],
    dtype=torch.bool,
)


@dataclass
class EvidenceTrajectory:
    phenotype: int
    mask: torch.Tensor
    log_forward: torch.Tensor
    log_backward: torch.Tensor
    action_count: int
    event_order: torch.Tensor | None = None
    bundle_order: torch.Tensor | None = None


@dataclass
class RewardTerms:
    log_reward: torch.Tensor
    classification: torch.Tensor
    severity: torch.Tensor
    necessity: torch.Tensor
    specificity: torch.Tensor
    compactness: torch.Tensor


def _safe_subset(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if len(indices) == 0:
        return torch.zeros_like(summarize(value[:1]))
    return summarize_subset(value, indices)


def masked_meter_features(
    anchor: FrozenMeterPIM,
    projected: Mapping[str, torch.Tensor],
    selected_mask: torch.Tensor,
    *,
    intervention: str,
) -> torch.Tensor:
    """Build the original METER feature layout under event-modality intervention."""
    if selected_mask.ndim != 2 or selected_mask.shape[1] != len(VIEW_NAMES):
        raise ValueError(f"invalid event-modality mask: {tuple(selected_mask.shape)}")
    if selected_mask.shape[0] != len(projected["answer"]):
        raise ValueError("event-modality mask length does not match projected events")
    if intervention == "selected_only":
        keep = selected_mask
    elif intervention == "selected_delete":
        keep = ~selected_mask
    else:
        raise ValueError(f"unsupported intervention: {intervention}")

    masked = {
        name: projected[name] * keep[:, index, None].to(projected[name].dtype)
        for index, name in enumerate(VIEW_NAMES)
    }
    clinical = masked["clinical"]
    answer = masked["answer"]
    context = masked["context"]
    visual = masked["visual"]
    process = torch.cat(
        (
            summarize(answer),
            summarize(clinical),
            summarize(context),
            summarize(clinical - answer),
            summarize(clinical - context),
        )
    )
    process_visual = torch.cat((process, summarize(visual)))
    selected_blocks = []
    complement_blocks = []
    for index, name in enumerate(VIEW_NAMES):
        selected = torch.where(selected_mask[:, index])[0]
        complement = torch.where(~selected_mask[:, index])[0]
        selected_blocks.append(_safe_subset(masked[name], selected))
        complement_blocks.append(_safe_subset(masked[name], complement))
    selected_cat = torch.cat(selected_blocks)
    complement_cat = torch.cat(complement_blocks)
    return torch.cat((process_visual, selected_cat, complement_cat, selected_cat - complement_cat))


def measure_mask(
    anchor: FrozenMeterPIM,
    projected: Mapping[str, torch.Tensor],
    selected_mask: torch.Tensor,
    *,
    intervention: str,
) -> dict[str, torch.Tensor]:
    features = masked_meter_features(anchor, projected, selected_mask, intervention=intervention)
    return anchor.measure_features(features)


def fixed_topk_mask(anchor: FrozenMeterPIM, projected: Mapping[str, torch.Tensor]) -> torch.Tensor:
    selected = anchor.evidence_scorer.indices(event_matrix(projected))
    mask = torch.zeros(
        (len(projected["answer"]), len(VIEW_NAMES)),
        dtype=torch.bool,
        device=projected["answer"].device,
    )
    mask[selected] = True
    return mask


def random_matched_mask(selected_mask: torch.Tensor, seed: int) -> torch.Tensor:
    """Randomize selected event locations while preserving modality bundles."""
    chosen_events = torch.where(selected_mask.any(dim=1))[0]
    bundles = selected_mask[chosen_events]
    generator = torch.Generator(device=selected_mask.device)
    generator.manual_seed(int(seed))
    target_events = torch.randperm(
        len(selected_mask), generator=generator, device=selected_mask.device
    )[: len(chosen_events)]
    if len(bundles) > 1:
        bundle_order = torch.randperm(len(bundles), generator=generator, device=selected_mask.device)
        bundles = bundles[bundle_order]
    result = torch.zeros_like(selected_mask)
    result[target_events] = bundles
    return result


def shuffled_evidence_mask(selected_mask: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=selected_mask.device)
    generator.manual_seed(int(seed))
    order = torch.randperm(len(selected_mask), generator=generator, device=selected_mask.device)
    return selected_mask[order]


def build_event_features(
    anchor: FrozenMeterPIM,
    projected: Mapping[str, torch.Tensor],
    temporal: torch.Tensor,
) -> torch.Tensor:
    base = torch.cat((event_matrix(projected), projected["visual"]), dim=-1)
    evidence = anchor.evidence_scorer.evidence(event_matrix(projected))
    evidence = (evidence - evidence.mean()) / evidence.std(correction=0).clamp_min(1e-6)
    if temporal.shape != (len(base), 4):
        raise ValueError(f"invalid temporal features: {tuple(temporal.shape)}")
    return torch.cat((base, temporal.to(base.dtype), evidence[:, None]), dim=-1)


class HierarchicalClinicalGFlowNet(nn.Module):
    """Phenotype -> response event -> modality-bundle conditional GFlowNet."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 96,
        phenotype_count: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.phenotype_count = int(phenotype_count)
        self.event_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.phenotype_head = nn.Linear(hidden_dim, phenotype_count)
        self.phenotype_embedding = nn.Embedding(phenotype_count, hidden_dim)
        self.event_query = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.modality_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(MODALITY_BUNDLES)),
        )
        self.log_z = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.event_encoder(events)
        context = self.context_encoder(torch.cat((encoded.mean(dim=0), encoded.std(dim=0, correction=0))))
        return encoded, context

    def partition(self, events: torch.Tensor) -> torch.Tensor:
        _, context = self.encode(events)
        return self.partition_from_context(context)

    def partition_from_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.log_z(context).squeeze(-1)

    def _state(
        self,
        encoded: torch.Tensor,
        context: torch.Tensor,
        phenotype: int,
        selected_events: torch.Tensor,
        maximum_events: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_mask = selected_events.clone()
        if state_mask.any():
            selected_summary = encoded[state_mask].mean(dim=0)
        else:
            selected_summary = torch.zeros_like(context)
        route = self.phenotype_embedding.weight[int(phenotype)]
        progress = torch.tensor(
            [float(state_mask.sum().item()) / max(1, maximum_events)],
            dtype=context.dtype,
            device=context.device,
        )
        query = self.event_query(
            torch.cat((context, route, selected_summary, context * route, progress))
        )
        event_logits = (encoded * query).sum(dim=-1) / math.sqrt(self.hidden_dim)
        stop_logit = self.stop_head(torch.cat((context, route, selected_summary, progress))).squeeze(-1)
        return event_logits, stop_logit, route

    def trajectory(
        self,
        events: torch.Tensor,
        *,
        minimum_events: int,
        maximum_events: int,
        generator: torch.Generator | None = None,
        greedy: bool = False,
        exploration_epsilon: float = 0.0,
    ) -> EvidenceTrajectory:
        encoded, context = self.encode(events)
        return self.trajectory_from_encoding(
            encoded,
            context,
            minimum_events=minimum_events,
            maximum_events=maximum_events,
            generator=generator,
            greedy=greedy,
            exploration_epsilon=exploration_epsilon,
        )

    def trajectory_from_encoding(
        self,
        encoded: torch.Tensor,
        context: torch.Tensor,
        *,
        minimum_events: int,
        maximum_events: int,
        generator: torch.Generator | None = None,
        greedy: bool = False,
        exploration_epsilon: float = 0.0,
    ) -> EvidenceTrajectory:
        if not 0.0 <= float(exploration_epsilon) <= 1.0:
            raise ValueError("exploration_epsilon must be in [0, 1]")

        def sample_action(logits: torch.Tensor, log_probability: torch.Tensor) -> torch.Tensor:
            if greedy:
                return logits.argmax()
            explore = float(exploration_epsilon) > 0.0 and bool(
                torch.rand((), generator=generator, device=logits.device)
                < float(exploration_epsilon)
            )
            if explore:
                valid = torch.nonzero(torch.isfinite(logits), as_tuple=False).flatten()
                selected = torch.randint(
                    len(valid), (1,), generator=generator, device=logits.device
                ).squeeze(0)
                return valid[selected]
            return torch.multinomial(log_probability.exp(), 1, generator=generator).squeeze(0)

        phenotype_logits = self.phenotype_head(context)
        phenotype_log_probability = F.log_softmax(phenotype_logits, dim=0)
        phenotype_tensor = sample_action(phenotype_logits, phenotype_log_probability)
        phenotype = int(phenotype_tensor.item())
        log_forward = phenotype_log_probability[phenotype_tensor]
        selected_events = torch.zeros(len(encoded), dtype=torch.bool, device=encoded.device)
        selected_mask = torch.zeros(
            (len(encoded), len(VIEW_NAMES)), dtype=torch.bool, device=encoded.device
        )
        modality_bundles = MODALITY_BUNDLES.to(encoded.device)
        count = 0
        event_order: list[int] = []
        bundle_order: list[int] = []
        while count < maximum_events:
            event_logits, stop_logit, route = self._state(
                encoded, context, phenotype, selected_events, maximum_events
            )
            # Autograd saves the boolean mask for the backward pass.  Clone it
            # because the trajectory state is updated in-place after sampling.
            event_logits = event_logits.masked_fill(selected_events.clone(), -torch.inf)
            if count >= minimum_events:
                choice_logits = torch.cat((event_logits, stop_logit[None]))
            else:
                choice_logits = event_logits
            choice_log_probability = F.log_softmax(choice_logits, dim=0)
            choice = sample_action(choice_logits, choice_log_probability)
            log_forward = log_forward + choice_log_probability[choice]
            if count >= minimum_events and int(choice.item()) == len(encoded):
                break
            event_index = int(choice.item())
            modality_logits = self.modality_head(
                torch.cat((encoded[event_index], context, route))
            )
            modality_log_probability = F.log_softmax(modality_logits, dim=0)
            bundle_index = sample_action(modality_logits, modality_log_probability)
            log_forward = log_forward + modality_log_probability[bundle_index]
            selected_events[event_index] = True
            selected_mask[event_index] = modality_bundles[bundle_index]
            event_order.append(event_index)
            bundle_order.append(int(bundle_index.item()))
            count += 1
        log_backward = -torch.lgamma(
            torch.tensor(float(count + 1), dtype=log_forward.dtype, device=log_forward.device)
        )
        return EvidenceTrajectory(
            phenotype=phenotype,
            mask=selected_mask,
            log_forward=log_forward,
            log_backward=log_backward,
            action_count=count,
            event_order=torch.tensor(event_order, dtype=torch.long, device=encoded.device),
            bundle_order=torch.tensor(bundle_order, dtype=torch.long, device=encoded.device),
        )

    def replay_trajectory_from_encoding(
        self,
        encoded: torch.Tensor,
        context: torch.Tensor,
        *,
        phenotype: int,
        event_order: torch.Tensor | list[int],
        bundle_order: torch.Tensor | list[int],
        minimum_events: int,
        maximum_events: int,
    ) -> EvidenceTrajectory:
        """Evaluate the exact forward probability of a prescribed trajectory."""
        events = torch.as_tensor(event_order, dtype=torch.long, device=encoded.device)
        bundles = torch.as_tensor(bundle_order, dtype=torch.long, device=encoded.device)
        if events.ndim != 1 or bundles.ndim != 1 or len(events) != len(bundles):
            raise ValueError("event_order and bundle_order must be equally sized vectors")
        if not minimum_events <= len(events) <= maximum_events:
            raise ValueError("trajectory action count is outside the configured limits")
        if len(torch.unique(events)) != len(events):
            raise ValueError("trajectory event_order contains duplicates")
        if bool(((events < 0) | (events >= len(encoded))).any()):
            raise ValueError("trajectory event_order contains an invalid event index")
        if bool(((bundles < 0) | (bundles >= len(MODALITY_BUNDLES))).any()):
            raise ValueError("trajectory bundle_order contains an invalid bundle index")
        if not 0 <= int(phenotype) < self.phenotype_count:
            raise ValueError("trajectory phenotype is invalid")

        phenotype_logits = self.phenotype_head(context)
        log_forward = F.log_softmax(phenotype_logits, dim=0)[int(phenotype)]
        selected_events = torch.zeros(len(encoded), dtype=torch.bool, device=encoded.device)
        selected_mask = torch.zeros(
            (len(encoded), len(VIEW_NAMES)), dtype=torch.bool, device=encoded.device
        )
        modality_bundles = MODALITY_BUNDLES.to(encoded.device)
        for count, (event_tensor, bundle_tensor) in enumerate(zip(events, bundles)):
            event_index = int(event_tensor.item())
            bundle_index = int(bundle_tensor.item())
            event_logits, stop_logit, route = self._state(
                encoded, context, int(phenotype), selected_events, maximum_events
            )
            event_logits = event_logits.masked_fill(selected_events.clone(), -torch.inf)
            choice_logits = (
                torch.cat((event_logits, stop_logit[None]))
                if count >= minimum_events
                else event_logits
            )
            log_forward = log_forward + F.log_softmax(choice_logits, dim=0)[event_index]
            modality_logits = self.modality_head(
                torch.cat((encoded[event_index], context, route))
            )
            log_forward = log_forward + F.log_softmax(modality_logits, dim=0)[bundle_index]
            selected_events[event_index] = True
            selected_mask[event_index] = modality_bundles[bundle_index]

        action_count = len(events)
        if action_count < maximum_events:
            event_logits, stop_logit, _ = self._state(
                encoded, context, int(phenotype), selected_events, maximum_events
            )
            event_logits = event_logits.masked_fill(selected_events.clone(), -torch.inf)
            stop_logits = torch.cat((event_logits, stop_logit[None]))
            log_forward = log_forward + F.log_softmax(stop_logits, dim=0)[-1]
        log_backward = -torch.lgamma(
            torch.tensor(
                float(action_count + 1),
                dtype=log_forward.dtype,
                device=log_forward.device,
            )
        )
        return EvidenceTrajectory(
            phenotype=int(phenotype),
            mask=selected_mask,
            log_forward=log_forward,
            log_backward=log_backward,
            action_count=action_count,
            event_order=events.clone(),
            bundle_order=bundles.clone(),
        )


def clinical_log_reward(
    anchor: FrozenMeterPIM,
    projected: Mapping[str, torch.Tensor],
    selected_mask: torch.Tensor,
    *,
    label: int,
    score: float | None,
    score_max: float,
    random_seed: int,
    pim_weight: float,
    ensr_weight: float,
    compactness_weight: float,
    temperature: float,
) -> RewardTerms:
    with torch.no_grad():
        clean_features = anchor.meter_v7(projected)
        clean = anchor.measure_features(clean_features)
        selected_only = measure_mask(anchor, projected, selected_mask, intervention="selected_only")
        selected_delete = measure_mask(anchor, projected, selected_mask, intervention="selected_delete")
        random_mask = random_matched_mask(selected_mask, random_seed)
        random_delete = measure_mask(anchor, projected, random_mask, intervention="selected_delete")
        target = torch.tensor(float(label), dtype=clean["logit"].dtype, device=clean["logit"].device)
        classification = F.binary_cross_entropy_with_logits(selected_only["logit"].squeeze(0), target)
        if score is None or pim_weight <= 0:
            severity = torch.zeros_like(classification)
        else:
            severity_target = torch.tensor(
                float(score) / float(score_max), dtype=clean["severity"].dtype, device=clean["severity"].device
            )
            severity = F.smooth_l1_loss(
                selected_only["severity"].squeeze(0) / float(score_max), severity_target
            )
        direction = target.mul(2.0).sub(1.0)
        necessity_margin = direction * (
            clean["logit"].squeeze(0) - selected_delete["logit"].squeeze(0)
        )
        random_margin = direction * (
            clean["logit"].squeeze(0) - random_delete["logit"].squeeze(0)
        )
        necessity = F.relu(0.05 - necessity_margin)
        specificity = F.relu(random_margin - necessity_margin + 0.02)
        compactness = selected_mask.any(dim=1).to(clean["logit"].dtype).mean()
        total = (
            classification
            + float(pim_weight) * severity
            + float(ensr_weight) * (necessity + specificity)
            + float(compactness_weight) * compactness
        )
        log_reward = -total / max(float(temperature), 1e-6)
    return RewardTerms(
        log_reward=log_reward,
        classification=classification,
        severity=severity,
        necessity=necessity,
        specificity=specificity,
        compactness=compactness,
    )


def trajectory_balance_loss(
    model: HierarchicalClinicalGFlowNet,
    events: torch.Tensor,
    trajectories: list[EvidenceTrajectory],
    rewards: list[RewardTerms],
    *,
    log_z: torch.Tensor | None = None,
    relative_weight: float = 0.0,
) -> torch.Tensor:
    if log_z is None:
        log_z = model.partition(events)
    residuals = torch.stack([
        log_z + trajectory.log_forward - trajectory.log_backward - reward.log_reward
        for trajectory, reward in zip(trajectories, rewards)
    ])
    loss = residuals.square().mean()
    if float(relative_weight) > 0.0 and len(residuals) > 1:
        # Subtracting two trajectory-balance equations cancels log Z. This
        # directly calibrates within-participant reward differences to flow
        # differences while retaining the standard TB objective above.
        centered = residuals - residuals.mean()
        loss = loss + float(relative_weight) * centered.square().mean()
    return loss


def reinforce_loss(
    trajectories: list[EvidenceTrajectory],
    rewards: list[RewardTerms],
    *,
    entropy_weight: float,
) -> torch.Tensor:
    log_probability = torch.stack([trajectory.log_forward for trajectory in trajectories])
    reward = torch.stack([item.log_reward for item in rewards]).detach()
    advantage = reward - reward.mean()
    return -(advantage * log_probability).mean() + float(entropy_weight) * log_probability.mean()


def jaccard_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.bool()
    right = right.bool()
    union = torch.logical_or(left, right).sum().item()
    if union == 0:
        return 0.0
    intersection = torch.logical_and(left, right).sum().item()
    return float(1.0 - intersection / union)


def posterior_diversity(masks: list[torch.Tensor]) -> float:
    if len(masks) < 2:
        return 0.0
    values = [jaccard_distance(masks[i], masks[j]) for i in range(len(masks)) for j in range(i)]
    return float(np.mean(values)) if values else 0.0
