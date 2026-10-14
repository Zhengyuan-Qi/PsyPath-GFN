#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.alternating_psychometric_gfn import (
    evidence_features,
    fit_direct_operator,
    fit_pim_operator,
    full_mask,
    operator_log_reward,
    operator_log_rewards,
    random_sparse_mask,
)
from scripts.meter_gfn import HierarchicalClinicalGFlowNet, trajectory_balance_loss
from scripts.meter_mh_online_operator import VIEW_NAMES


def main() -> None:
    rng = np.random.default_rng(17)
    participant_count = 24
    labels = np.asarray([index % 2 for index in range(participant_count)], dtype=np.int64)
    scores = 4.0 + labels * 11.0 + rng.normal(0.0, 0.8, participant_count)
    participant_ids = np.asarray([f"P{index:03d}" for index in range(participant_count)])
    feature_sets = []
    full_features = []
    augmented_labels = []
    augmented_scores = []
    augmented_groups = []
    projected_records = []
    for index, (label, score, pid) in enumerate(zip(labels, scores, participant_ids)):
        length = 8 + index % 5
        projected = {
            name: rng.normal(label * 0.8 + score * 0.03, 1.0, (length, 6)).astype(np.float32)
            for name in VIEW_NAMES
        }
        projected_records.append(projected)
        full_features.append(evidence_features(projected, full_mask(length)))
        for mask_index in range(3):
            mask = (
                full_mask(length)
                if mask_index == 0
                else random_sparse_mask(length, 2, 5, 1000 + index * 10 + mask_index)
            )
            feature_sets.append(evidence_features(projected, mask))
            augmented_labels.append(label)
            augmented_scores.append(score)
            augmented_groups.append(pid)
    augmented = np.stack(feature_sets)
    full = np.stack(full_features)
    common = {
        "augmented_features": augmented,
        "augmented_groups": np.asarray(augmented_groups),
        "full_features": full,
        "labels": labels,
        "participant_ids": participant_ids,
        "score_max": 24.0,
        "seed": 31,
        "components": 12,
    }
    pim = fit_pim_operator(
        augmented_scores=np.asarray(augmented_scores),
        scores=scores,
        **common,
    )
    direct = fit_direct_operator(
        augmented_labels=np.asarray(augmented_labels),
        **common,
    )
    for operator in (pim, direct):
        prediction = operator.predict(full[:3])
        assert prediction["probability"].shape == (3,)
        assert np.isfinite(prediction["probability"]).all()

    events = torch.randn(len(projected_records[0]["answer"]), 20, dtype=torch.float32)
    model = HierarchicalClinicalGFlowNet(20, hidden_dim=16, phenotype_count=3)
    trajectory = model.trajectory(
        events,
        minimum_events=2,
        maximum_events=4,
        generator=torch.Generator().manual_seed(7),
    )
    encoded, context = model.encode(events)
    shared_trajectory = model.trajectory_from_encoding(
        encoded,
        context,
        minimum_events=2,
        maximum_events=4,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(trajectory.mask, shared_trajectory.mask)
    reward = operator_log_reward(
        pim,
        projected_records[0],
        trajectory,
        label=int(labels[0]),
        score=float(scores[0]),
        random_seed=41,
        pim_weight=0.3,
        ensr_weight=0.2,
        sufficiency_weight=1.0,
        compactness_weight=0.05,
        necessity_margin=0.05,
        specificity_margin=0.02,
        temperature=0.5,
        path_process_weight=0.15,
        path_monotonic_weight=0.10,
        path_order_weight=0.15,
        path_order_margin=0.02,
        path_duplicate_weight=0.01,
    )
    batch_reward = operator_log_rewards(
        pim,
        projected_records[0],
        [trajectory],
        label=int(labels[0]),
        score=float(scores[0]),
        random_seeds=[41],
        pim_weight=0.3,
        ensr_weight=0.2,
        sufficiency_weight=1.0,
        compactness_weight=0.05,
        necessity_margin=0.05,
        specificity_margin=0.02,
        temperature=0.5,
        path_process_weight=0.15,
        path_monotonic_weight=0.10,
        path_order_weight=0.15,
        path_order_margin=0.02,
        path_duplicate_weight=0.01,
    )[0]
    for key in reward:
        assert np.isclose(reward[key], batch_reward[key])
    for key in ("path_process", "path_monotonic", "path_order", "forward_path_loss", "reverse_path_loss"):
        assert np.isfinite(reward[key])
    reward_tensor = type("Reward", (), {"log_reward": torch.tensor(reward["log_reward"])})()
    loss = trajectory_balance_loss(
        model,
        events,
        [trajectory],
        [reward_tensor],
        log_z=model.partition_from_context(context),
    )
    loss.backward()
    assert torch.isfinite(loss)

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "operator.pt"
        torch.save({"pim": pim, "direct": direct}, checkpoint)
        assert checkpoint.stat().st_size > 0
    print("alternating psychometric GFlowNet tests passed")


if __name__ == "__main__":
    main()
