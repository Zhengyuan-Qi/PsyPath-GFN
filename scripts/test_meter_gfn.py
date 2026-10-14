#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.meter_gfn import (
    EvidenceTrajectory,
    HierarchicalClinicalGFlowNet,
    MODALITY_BUNDLES,
    posterior_diversity,
    random_matched_mask,
    trajectory_balance_loss,
)


def test_trajectory_contract() -> None:
    torch.manual_seed(7)
    model = HierarchicalClinicalGFlowNet(17, hidden_dim=24, phenotype_count=3)
    events = torch.randn(11, 17)
    generator = torch.Generator().manual_seed(11)
    trajectory = model.trajectory(
        events,
        minimum_events=2,
        maximum_events=4,
        generator=generator,
    )
    assert 2 <= trajectory.action_count <= 4
    assert trajectory.mask.shape == (11, 4)
    assert int(trajectory.mask.any(dim=1).sum()) == trajectory.action_count
    assert bool(torch.isfinite(trajectory.log_forward))
    assert bool(torch.isfinite(trajectory.log_backward))
    assert 0 <= trajectory.phenotype < 3


def test_random_mask_preserves_bundles() -> None:
    mask = torch.zeros(9, 4, dtype=torch.bool)
    mask[1] = MODALITY_BUNDLES[2]
    mask[5] = MODALITY_BUNDLES[10]
    randomized = random_matched_mask(mask, 19)
    assert int(randomized.any(dim=1).sum()) == 2
    assert sorted(randomized[randomized.any(dim=1)].sum(dim=1).tolist()) == sorted(
        mask[mask.any(dim=1)].sum(dim=1).tolist()
    )


def test_diversity() -> None:
    left = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
    right = torch.tensor([[0, 1], [0, 0]], dtype=torch.bool)
    assert posterior_diversity([left, left]) == 0.0
    assert posterior_diversity([left, right]) == 1.0


def test_relative_trajectory_balance_penalizes_misaligned_differences() -> None:
    model = HierarchicalClinicalGFlowNet(3, hidden_dim=8, phenotype_count=2)
    events = torch.randn(4, 3)
    zero_mask = torch.zeros(4, 4, dtype=torch.bool)
    trajectories = [
        EvidenceTrajectory(0, zero_mask, torch.tensor(1.0), torch.tensor(0.0), 0),
        EvidenceTrajectory(0, zero_mask, torch.tensor(-1.0), torch.tensor(0.0), 0),
    ]
    rewards = [
        type("Reward", (), {"log_reward": torch.tensor(0.0)})(),
        type("Reward", (), {"log_reward": torch.tensor(0.0)})(),
    ]
    standard = trajectory_balance_loss(
        model, events, trajectories, rewards, log_z=torch.tensor(0.0)
    )
    calibrated = trajectory_balance_loss(
        model,
        events,
        trajectories,
        rewards,
        log_z=torch.tensor(0.0),
        relative_weight=2.0,
    )
    assert torch.isclose(standard, torch.tensor(1.0))
    assert torch.isclose(calibrated, torch.tensor(3.0))


def test_replayed_trajectory_matches_sample_probability() -> None:
    torch.manual_seed(31)
    model = HierarchicalClinicalGFlowNet(9, hidden_dim=16, phenotype_count=3)
    events = torch.randn(8, 9)
    encoded, context = model.encode(events)
    sampled = model.trajectory_from_encoding(
        encoded,
        context,
        minimum_events=2,
        maximum_events=4,
        generator=torch.Generator().manual_seed(47),
    )
    replayed = model.replay_trajectory_from_encoding(
        encoded,
        context,
        phenotype=sampled.phenotype,
        event_order=sampled.event_order,
        bundle_order=sampled.bundle_order,
        minimum_events=2,
        maximum_events=4,
    )
    assert torch.equal(replayed.mask, sampled.mask)
    assert torch.allclose(replayed.log_forward, sampled.log_forward, atol=1e-6, rtol=1e-6)
    assert torch.allclose(replayed.log_backward, sampled.log_backward, atol=1e-7, rtol=0.0)


if __name__ == "__main__":
    test_trajectory_contract()
    test_random_mask_preserves_bundles()
    test_diversity()
    test_relative_trajectory_balance_penalizes_misaligned_differences()
    test_replayed_trajectory_matches_sample_probability()
    print("meter_gfn synthetic tests passed")
