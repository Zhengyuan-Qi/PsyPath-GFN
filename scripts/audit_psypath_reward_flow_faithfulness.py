#!/usr/bin/env python3
"""Frozen-checkpoint audits for PsyPath-GFN reward flow and label-free faithfulness."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

from scripts import validate_alternating_psychometric_gfn_cached as formal
from scripts.alternating_psychometric_gfn import evidence_features, operator_log_rewards
from scripts.meter_mh_online_operator import VIEW_NAMES


SCHEMA = "psypath_reward_flow_label_free_audit_v1"
TASK_NAMES = ("daic_depression", "edaic_depression", "edaic_ptsd")
DEFAULT_SEEDS = (7, 17, 27, 41, 53)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def mean_std(values: Iterable[float]) -> dict[str, float | int | None]:
    finite_values = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            finite_values.append(numeric)
    array = np.asarray(finite_values, dtype=np.float64)
    if len(array) == 0:
        return {"n": 0, "mean": None, "std": None}
    return {"n": int(len(array)), "mean": float(array.mean()), "std": float(array.std(ddof=0))}


def rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 3 or float(left.std()) <= 1e-12 or float(right.std()) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return None
    return pearson(rank_average(left[valid]), rank_average(right[valid]))


def fixed_effect_slope(x_values: list[np.ndarray], y_values: list[np.ndarray]) -> float | None:
    centered_x = []
    centered_y = []
    for x_value, y_value in zip(x_values, y_values):
        x_value = np.asarray(x_value, dtype=np.float64)
        y_value = np.asarray(y_value, dtype=np.float64)
        centered_x.append(x_value - x_value.mean())
        centered_y.append(y_value - y_value.mean())
    x = np.concatenate(centered_x)
    y = np.concatenate(centered_y)
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else None


def binary_cross_entropy(probability: float, label: int) -> float:
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(-(int(label) * np.log(probability) + (1 - int(label)) * np.log(1.0 - probability)))


def checkpoint_path(formal_root: Path, task_name: str, seed: int) -> Path:
    return formal_root / task_name / "checkpoints" / task_name / "gflownet" / f"seed_{seed}.pt"


def load_checkpoint(formal_root: Path, task_name: str, seed: int, device: torch.device) -> tuple[Path, dict[str, Any]]:
    path = checkpoint_path(formal_root, task_name, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = formal.load_trusted_checkpoint(path, device)
    required = {
        "task": task_name,
        "variant": "full",
        "method": "gflownet",
        "seed": int(seed),
    }
    observed = {key: checkpoint.get(key) for key in required}
    if observed != required:
        raise RuntimeError(f"checkpoint contract mismatch: expected={required}, observed={observed}")
    return path, checkpoint


def checkpoint_args(checkpoint: dict[str, Any]) -> SimpleNamespace:
    config = dict(checkpoint.get("config", {}))
    defaults = {
        "hidden_dim": 64,
        "phenotype_count": 4,
        "policy": "legacy",
        "minimum_events": 2,
        "maximum_events": 16,
        "maximum_fraction": 0.30,
        "pim_reward_weight": 0.30,
        "ensr_reward_weight": 0.05,
        "sufficiency_weight": 1.0,
        "compactness_weight": 0.03,
        "necessity_margin": 0.05,
        "specificity_margin": 0.02,
        "reward_temperature": 0.50,
        "path_process": False,
        "path_process_weight": 0.0,
        "path_monotonic_weight": 0.0,
        "path_order_weight": 0.0,
        "path_order_margin": 0.02,
        "path_duplicate_weight": 0.0,
    }
    for name, value in defaults.items():
        config.setdefault(name, value)
    return SimpleNamespace(**config)


def reward_kwargs(config: SimpleNamespace, record: dict[str, Any], random_seeds: list[int]) -> dict[str, Any]:
    variant = str(getattr(config, "variant", "full"))
    variant_config = formal.VARIANTS.get(variant, formal.VARIANTS["full"])
    path_process = bool(getattr(config, "path_process", False))
    return {
        "label": int(record["label"]),
        "score": record.get("score"),
        "random_seeds": random_seeds,
        "pim_weight": float(config.pim_reward_weight) if variant_config["pim"] else 0.0,
        "ensr_weight": float(config.ensr_reward_weight) if variant_config["ensr"] else 0.0,
        "sufficiency_weight": float(config.sufficiency_weight),
        "compactness_weight": float(config.compactness_weight),
        "necessity_margin": float(config.necessity_margin),
        "specificity_margin": float(config.specificity_margin),
        "temperature": float(config.reward_temperature),
        "path_process_weight": float(config.path_process_weight) if path_process else 0.0,
        "path_monotonic_weight": float(config.path_monotonic_weight) if path_process else 0.0,
        "path_order_weight": float(config.path_order_weight) if path_process else 0.0,
        "path_order_margin": float(config.path_order_margin),
        "path_duplicate_weight": float(config.path_duplicate_weight),
    }


def trajectory_signature(trajectory: Any) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([int(trajectory.phenotype)], dtype=np.int64).tobytes())
    digest.update(trajectory.mask.detach().cpu().numpy().astype(np.uint8).tobytes())
    digest.update(trajectory.event_order.detach().cpu().numpy().astype(np.int64).tobytes())
    digest.update(trajectory.bundle_order.detach().cpu().numpy().astype(np.int64).tobytes())
    return digest.hexdigest()[:20]


def posterior_prediction(operator: Any, projected: dict[str, np.ndarray], masks: np.ndarray) -> tuple[float, float | None]:
    features = np.stack([evidence_features(projected, mask) for mask in masks])
    prediction = operator.predict(features)
    probability = float(np.mean(prediction["probability"]))
    severity_values = np.asarray(prediction["severity"], dtype=np.float64)
    severity = float(np.nanmean(severity_values)) if np.isfinite(severity_values).any() else None
    return probability, severity


def available_evidence(raw_record: dict[str, Any]) -> np.ndarray:
    return np.stack(
        [
            np.linalg.norm(np.asarray(raw_record["raw"][name], dtype=np.float64), axis=1) > 1e-10
            for name in VIEW_NAMES
        ],
        axis=1,
    )


def matched_candidates(raw_record: dict[str, Any], critical_event: int) -> tuple[np.ndarray, str]:
    availability = available_evidence(raw_record)
    question = np.asarray(raw_record["temporal"], dtype=np.float64)[:, 3] > 0.5
    event_count = len(availability)
    all_indices = np.asarray([index for index in range(event_count) if index != critical_event], dtype=np.int64)
    if len(all_indices) == 0:
        return all_indices, "none"

    critical_pattern = availability[critical_event]
    critical_count = int(critical_pattern.sum())
    critical_question = bool(question[critical_event])
    tiers = [
        (
            "exact_availability_and_question",
            np.asarray(
                [
                    index
                    for index in all_indices
                    if np.array_equal(availability[index], critical_pattern)
                    and bool(question[index]) == critical_question
                ],
                dtype=np.int64,
            ),
        ),
        (
            "exact_availability",
            np.asarray(
                [index for index in all_indices if np.array_equal(availability[index], critical_pattern)],
                dtype=np.int64,
            ),
        ),
        (
            "availability_count_and_question",
            np.asarray(
                [
                    index
                    for index in all_indices
                    if int(availability[index].sum()) == critical_count
                    and bool(question[index]) == critical_question
                ],
                dtype=np.int64,
            ),
        ),
        (
            "availability_count",
            np.asarray(
                [index for index in all_indices if int(availability[index].sum()) == critical_count],
                dtype=np.int64,
            ),
        ),
        ("unrestricted", all_indices),
    ]
    for tier_name, candidates in tiers:
        if len(candidates) == 0:
            continue
        distance = np.abs(candidates - int(critical_event))
        order = np.argsort(distance, kind="mergesort")
        nearest_count = min(len(candidates), max(3, int(math.ceil(0.25 * len(candidates)))))
        return candidates[order[:nearest_count]], tier_name
    raise AssertionError("unreachable matching state")


def audit_label_free_event(
    operator: Any,
    record: dict[str, Any],
    raw_record: dict[str, Any],
    masks: np.ndarray,
    *,
    random_controls: int,
    seed: int,
) -> dict[str, Any]:
    event_inclusion = masks.any(axis=2).mean(axis=0)
    critical_event = int(np.argmax(event_inclusion))
    candidates, match_tier = matched_candidates(raw_record, critical_event)
    if len(candidates) == 0:
        return {
            "eligible": False,
            "exclusion_reason": "fewer than two response events",
            "selector": "maximum posterior event-inclusion probability",
            "selector_uses_ground_truth": False,
            "primary_metric_uses_ground_truth": False,
            "critical_event": critical_event,
            "critical_event_position": 0.0,
            "critical_inclusion_probability": float(event_inclusion[critical_event]),
            "matched_random_inclusion_probability": None,
            "match_tier": match_tier,
            "candidate_count": 0,
            "random_control_draws": 0,
            "baseline_probability": None,
            "critical_deleted_probability": None,
            "critical_probability_change": None,
            "matched_random_probability_change": None,
            "probability_change_specificity_gap": None,
            "critical_severity_change": None,
            "matched_random_severity_change": None,
            "severity_change_specificity_gap": None,
            "critical_bce_cost": None,
            "matched_random_bce_cost": None,
            "bce_specificity_gap": None,
            "critical_severity_error_cost": None,
            "matched_random_severity_error_cost": None,
        }

    rng = np.random.default_rng(int(seed))
    random_events = rng.choice(candidates, size=max(1, int(random_controls)), replace=True)
    baseline_probability, baseline_severity = posterior_prediction(operator, record["projected_np"], masks)
    baseline_bce = binary_cross_entropy(baseline_probability, int(record["label"]))

    critical_masks = masks.copy()
    critical_masks[:, critical_event, :] = False
    critical_probability, critical_severity = posterior_prediction(
        operator, record["projected_np"], critical_masks
    )
    critical_probability_change = abs(critical_probability - baseline_probability)
    critical_bce_cost = binary_cross_entropy(critical_probability, int(record["label"])) - baseline_bce

    random_bce_costs = []
    random_probability_changes = []
    random_severity_costs = []
    prediction_cache: dict[int, tuple[float, float | None]] = {}
    for event in np.unique(random_events):
        random_masks = masks.copy()
        random_masks[:, int(event), :] = False
        prediction_cache[int(event)] = posterior_prediction(operator, record["projected_np"], random_masks)
    for event in random_events:
        probability, severity = prediction_cache[int(event)]
        random_bce_costs.append(binary_cross_entropy(probability, int(record["label"])) - baseline_bce)
        random_probability_changes.append(abs(probability - baseline_probability))
        if record.get("score") is not None and severity is not None and baseline_severity is not None:
            score = float(record["score"])
            random_severity_costs.append(abs(severity - score) - abs(baseline_severity - score))

    critical_severity_cost = None
    if record.get("score") is not None and critical_severity is not None and baseline_severity is not None:
        score = float(record["score"])
        critical_severity_cost = abs(critical_severity - score) - abs(baseline_severity - score)

    random_bce_cost = float(np.mean(random_bce_costs))
    matched_random_probability_change = float(np.mean(random_probability_changes))
    critical_severity_change = (
        abs(float(critical_severity) - float(baseline_severity))
        if critical_severity is not None and baseline_severity is not None
        else None
    )
    matched_random_severity_change = None
    if baseline_severity is not None:
        random_severity_changes = [
            abs(float(prediction_cache[int(event)][1]) - float(baseline_severity))
            for event in random_events
            if prediction_cache[int(event)][1] is not None
        ]
        if random_severity_changes:
            matched_random_severity_change = float(np.mean(random_severity_changes))
    return {
        "eligible": True,
        "exclusion_reason": None,
        "selector": "maximum posterior event-inclusion probability",
        "selector_uses_ground_truth": False,
        "primary_metric_uses_ground_truth": False,
        "critical_event": critical_event,
        "critical_event_position": float(critical_event / max(1, len(event_inclusion) - 1)),
        "critical_inclusion_probability": float(event_inclusion[critical_event]),
        "matched_random_inclusion_probability": float(np.mean(event_inclusion[random_events])),
        "match_tier": match_tier,
        "candidate_count": int(len(candidates)),
        "random_control_draws": int(len(random_events)),
        "baseline_probability": baseline_probability,
        "critical_deleted_probability": critical_probability,
        "critical_probability_change": critical_probability_change,
        "matched_random_probability_change": matched_random_probability_change,
        "probability_change_specificity_gap": (
            critical_probability_change - matched_random_probability_change
        ),
        "critical_severity_change": critical_severity_change,
        "matched_random_severity_change": matched_random_severity_change,
        "severity_change_specificity_gap": (
            critical_severity_change - matched_random_severity_change
            if critical_severity_change is not None and matched_random_severity_change is not None
            else None
        ),
        "critical_bce_cost": critical_bce_cost,
        "matched_random_bce_cost": random_bce_cost,
        "bce_specificity_gap": critical_bce_cost - random_bce_cost,
        "critical_severity_error_cost": critical_severity_cost,
        "matched_random_severity_error_cost": (
            float(np.mean(random_severity_costs)) if random_severity_costs else None
        ),
    }


def reward_bin_summary(log_reward: np.ndarray, flow_score: np.ndarray, bins: int = 5) -> list[dict[str, float | int]]:
    order = np.argsort(log_reward, kind="mergesort")
    assignments = np.empty(len(order), dtype=np.int64)
    assignments[order] = np.minimum(bins - 1, np.arange(len(order)) * bins // max(1, len(order)))
    centered_flow = flow_score - flow_score.mean()
    result = []
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            result.append(
                {
                    "reward_quintile": index + 1,
                    "n": int(selected.sum()),
                    "mean_log_reward": float(log_reward[selected].mean()),
                    "mean_centered_flow_score": float(centered_flow[selected].mean()),
                }
            )
    return result


def audit_seed(
    args: argparse.Namespace,
    task_name: str,
    seed: int,
    raw_test: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    started = time.time()
    path, checkpoint = load_checkpoint(args.formal_root, task_name, seed, device)
    config = checkpoint_args(checkpoint)
    test = formal.prepare_records(raw_test, checkpoint["event_projections"], device)
    raw_by_pid = {str(record["pid"]): record for record in raw_test}
    model = formal.new_model(test[0], config, device, task_name=task_name)
    model.load_state_dict(checkpoint["gflownet_state"], strict=True)
    model.eval()
    operator = checkpoint["psychometric_operator"]

    trajectory_dir = args.output / "trajectory_records"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = trajectory_dir / f"{task_name}__seed_{seed}.csv.gz"
    temporary_path = trajectory_path.with_suffix(trajectory_path.suffix + ".tmp")
    participant_records = []
    reward_arrays: list[np.ndarray] = []
    flow_arrays: list[np.ndarray] = []
    predicted_reward_arrays: list[np.ndarray] = []
    participant_rhos = []
    bin_rows = []

    with gzip.open(temporary_path, "wt", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "task",
            "seed",
            "pid",
            "path_index",
            "trajectory_signature",
            "phenotype",
            "action_count",
            "log_forward",
            "log_backward",
            "flow_score",
            "log_z",
            "log_reward",
            "tb_residual",
            "classification_cost",
            "severity_cost",
            "necessity_cost",
            "specificity_cost",
            "compactness_cost",
            "path_process_cost",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for participant_index, record in enumerate(test):
            encoding = model.encode(record["events"])
            trajectories = [
                formal.sample_trajectory(
                    model,
                    record,
                    config,
                    formal.stable_seed(SCHEMA, task_name, seed, record["pid"], sample),
                    gradient=False,
                    encoding=encoding,
                )
                for sample in range(args.samples)
            ]
            random_seeds = [
                formal.stable_seed(SCHEMA, "reward_control", task_name, seed, record["pid"], sample)
                for sample in range(args.samples)
            ]
            rewards = operator_log_rewards(
                operator,
                record["projected_np"],
                trajectories,
                **reward_kwargs(config, record, random_seeds),
            )
            log_forward = np.asarray([float(item.log_forward.detach().cpu()) for item in trajectories])
            log_backward = np.asarray([float(item.log_backward.detach().cpu()) for item in trajectories])
            flow_score = log_forward - log_backward
            log_reward = np.asarray([float(item["log_reward"]) for item in rewards], dtype=np.float64)
            log_z = float(model.partition_from_context(encoding[1]).detach().cpu())
            predicted_log_reward = log_z + flow_score
            residual = predicted_log_reward - log_reward
            masks = np.stack(
                [item.mask.detach().cpu().numpy().astype(bool) for item in trajectories], axis=0
            )

            label_free = audit_label_free_event(
                operator,
                record,
                raw_by_pid[str(record["pid"])],
                masks,
                random_controls=args.random_controls,
                seed=formal.stable_seed(SCHEMA, "label_free", task_name, seed, record["pid"]),
            )
            participant_rho = spearman(log_reward, flow_score)
            if participant_rho is not None:
                participant_rhos.append(participant_rho)
            reward_arrays.append(log_reward)
            flow_arrays.append(flow_score)
            predicted_reward_arrays.append(predicted_log_reward)
            for row in reward_bin_summary(log_reward, flow_score):
                bin_rows.append({"pid": str(record["pid"]), **row})

            signatures = [trajectory_signature(item) for item in trajectories]
            participant_records.append(
                {
                    "pid": str(record["pid"]),
                    "trajectory_count": int(len(trajectories)),
                    "unique_trajectory_fraction": float(len(set(signatures)) / len(signatures)),
                    "reward_flow": {
                        "spearman": participant_rho,
                        "pearson": pearson(log_reward, flow_score),
                        "tb_residual_mean": float(residual.mean()),
                        "tb_residual_mae": float(np.abs(residual).mean()),
                        "tb_residual_rmse": float(np.sqrt(np.square(residual).mean())),
                        "log_reward_std": float(log_reward.std(ddof=0)),
                        "flow_score_std": float(flow_score.std(ddof=0)),
                    },
                    "label_free_faithfulness": label_free,
                }
            )
            for path_index, (trajectory, reward, signature) in enumerate(
                zip(trajectories, rewards, signatures)
            ):
                writer.writerow(
                    {
                        "task": task_name,
                        "seed": seed,
                        "pid": record["pid"],
                        "path_index": path_index,
                        "trajectory_signature": signature,
                        "phenotype": trajectory.phenotype,
                        "action_count": trajectory.action_count,
                        "log_forward": float(trajectory.log_forward.detach().cpu()),
                        "log_backward": float(trajectory.log_backward.detach().cpu()),
                        "flow_score": float(flow_score[path_index]),
                        "log_z": log_z,
                        "log_reward": float(log_reward[path_index]),
                        "tb_residual": float(residual[path_index]),
                        "classification_cost": reward["classification"],
                        "severity_cost": reward["severity"],
                        "necessity_cost": reward["necessity"],
                        "specificity_cost": reward["specificity"],
                        "compactness_cost": reward["compactness"],
                        "path_process_cost": reward["path_process"],
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "figure2_audit_participant",
                        "task": task_name,
                        "seed": seed,
                        "completed": participant_index + 1,
                        "total": len(test),
                    }
                ),
                flush=True,
            )
    os.replace(temporary_path, trajectory_path)

    centered_reward = np.concatenate([value - value.mean() for value in reward_arrays])
    centered_flow = np.concatenate([value - value.mean() for value in flow_arrays])
    observed_spearman = spearman(centered_reward, centered_flow)
    observed_slope = fixed_effect_slope(reward_arrays, flow_arrays)
    observed_residual = np.concatenate(
        [predicted - reward for predicted, reward in zip(predicted_reward_arrays, reward_arrays)]
    )

    null_spearman = []
    null_rmse = []
    for permutation in range(args.null_permutations):
        shuffled_rewards = []
        shuffled_centered = []
        rng = np.random.default_rng(formal.stable_seed(SCHEMA, "null", task_name, seed, permutation))
        for reward in reward_arrays:
            shuffled = rng.permutation(reward)
            shuffled_rewards.append(shuffled)
            shuffled_centered.append(shuffled - shuffled.mean())
        correlation = spearman(np.concatenate(shuffled_centered), centered_flow)
        if correlation is not None:
            null_spearman.append(correlation)
        shuffled_residual = np.concatenate(
            [predicted - reward for predicted, reward in zip(predicted_reward_arrays, shuffled_rewards)]
        )
        null_rmse.append(float(np.sqrt(np.square(shuffled_residual).mean())))

    observed_value = float(observed_spearman) if observed_spearman is not None else float("nan")
    permutation_p = float(
        (1 + sum(value >= observed_value for value in null_spearman)) / (1 + len(null_spearman))
    )
    bce_gaps = [
        float(item["label_free_faithfulness"]["bce_specificity_gap"])
        for item in participant_records
        if item["label_free_faithfulness"]["bce_specificity_gap"] is not None
    ]
    seed_result = {
        "schema": SCHEMA,
        "task": task_name,
        "seed": int(seed),
        "checkpoint": str(path),
        "checkpoint_sha256": formal.sha256_file(path),
        "official_test_n": int(len(test)),
        "path_samples_per_participant": int(args.samples),
        "reward_flow": {
            "primary_statistic": "within-participant-centered Spearman(log reward, log PF - log PB)",
            "centered_spearman": observed_spearman,
            "participant_spearman": mean_std(participant_rhos),
            "participant_positive_fraction": float(np.mean(np.asarray(participant_rhos) > 0.0)),
            "fixed_effect_slope": observed_slope,
            "tb_residual": {
                "mean": float(observed_residual.mean()),
                "mae": float(np.abs(observed_residual).mean()),
                "rmse": float(np.sqrt(np.square(observed_residual).mean())),
            },
            "within_participant_reward_permutation": {
                "permutations": int(args.null_permutations),
                "spearman": mean_std(null_spearman),
                "one_sided_p": permutation_p,
                "tb_rmse": mean_std(null_rmse),
            },
            "reward_quintiles": [
                {
                    "reward_quintile": quintile,
                    "mean_centered_flow_score": float(
                        np.mean(
                            [
                                row["mean_centered_flow_score"]
                                for row in bin_rows
                                if row["reward_quintile"] == quintile
                            ]
                        )
                    ),
                }
                for quintile in range(1, 6)
            ],
        },
        "label_free_faithfulness": {
            "selector": "maximum posterior event-inclusion probability",
            "selector_uses_ground_truth": False,
            "primary_metric": "absolute risk-probability change after evidence deletion",
            "primary_metric_uses_ground_truth": False,
            "test_labels_used_only_for_post_selection_scoring": True,
            "eligible_participants": int(
                sum(item["label_free_faithfulness"]["eligible"] for item in participant_records)
            ),
            "excluded_participants": int(
                sum(not item["label_free_faithfulness"]["eligible"] for item in participant_records)
            ),
            "critical_probability_change": mean_std(
                item["label_free_faithfulness"]["critical_probability_change"]
                for item in participant_records
            ),
            "matched_random_probability_change": mean_std(
                item["label_free_faithfulness"]["matched_random_probability_change"]
                for item in participant_records
            ),
            "probability_change_specificity_gap": mean_std(
                item["label_free_faithfulness"]["probability_change_specificity_gap"]
                for item in participant_records
            ),
            "critical_severity_change": mean_std(
                item["label_free_faithfulness"]["critical_severity_change"]
                for item in participant_records
            ),
            "matched_random_severity_change": mean_std(
                item["label_free_faithfulness"]["matched_random_severity_change"]
                for item in participant_records
            ),
            "severity_change_specificity_gap": mean_std(
                item["label_free_faithfulness"]["severity_change_specificity_gap"]
                for item in participant_records
            ),
            "critical_bce_cost": mean_std(
                item["label_free_faithfulness"]["critical_bce_cost"]
                for item in participant_records
            ),
            "matched_random_bce_cost": mean_std(
                item["label_free_faithfulness"]["matched_random_bce_cost"]
                for item in participant_records
            ),
            "bce_specificity_gap": mean_std(bce_gaps),
            "positive_participant_fraction": (
                float(np.mean(np.asarray(bce_gaps) > 0.0)) if bce_gaps else None
            ),
            "critical_inclusion_probability": mean_std(
                item["label_free_faithfulness"]["critical_inclusion_probability"]
                for item in participant_records
            ),
            "matched_random_inclusion_probability": mean_std(
                item["label_free_faithfulness"]["matched_random_inclusion_probability"]
                for item in participant_records
            ),
            "match_tier_counts": {
                tier: sum(
                    item["label_free_faithfulness"]["match_tier"] == tier
                    for item in participant_records
                )
                for tier in sorted(
                    {
                        item["label_free_faithfulness"]["match_tier"]
                        for item in participant_records
                    }
                )
            },
        },
        "participants": participant_records,
        "trajectory_records": str(trajectory_path),
        "elapsed_s": float(time.time() - started),
    }
    return seed_result


def nested_bootstrap_gap(
    seed_results: list[dict[str, Any]],
    *,
    metric: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    by_seed = []
    for result in seed_results:
        by_seed.append(
            {
                item["pid"]: float(item["label_free_faithfulness"][metric])
                for item in result["participants"]
                if item["label_free_faithfulness"][metric] is not None
            }
        )
    common = sorted(set.intersection(*(set(item) for item in by_seed)))
    matrix = np.asarray([[item[pid] for pid in common] for item in by_seed], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        sampled_seeds = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        sampled_participants = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        values[index] = float(matrix[np.ix_(sampled_seeds, sampled_participants)].mean())
    return {
        "draws": int(draws),
        "participant_clusters": int(len(common)),
        "mean": float(matrix.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_positive": float(np.mean(values > 0.0)),
    }


def run_task(args: argparse.Namespace) -> dict[str, Any]:
    formal.configure_runtime(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    task = formal.TASKS[args.task]
    raw_test = formal.load_records(
        args.meter_root,
        args.contract_dir,
        args.task,
        allowed_splits=("test",),
        split_policy="official",
    )["test"]
    formal.attach_temporal({"test": raw_test}, args.meter_root, task["dataset"])
    if args.max_participants > 0:
        raw_test = raw_test[: args.max_participants]

    seed_results = []
    for seed in args.seeds:
        progress_path = args.output / args.task / "progress" / f"seed_{seed}.json"
        result = audit_seed(args, args.task, int(seed), raw_test, device)
        seed_results.append(result)
        atomic_json(progress_path, result)

    reward_correlations = [
        result["reward_flow"]["centered_spearman"]
        for result in seed_results
        if result["reward_flow"]["centered_spearman"] is not None
    ]
    reward_slopes = [
        result["reward_flow"]["fixed_effect_slope"]
        for result in seed_results
        if result["reward_flow"]["fixed_effect_slope"] is not None
    ]
    probability_gaps = [
        result["label_free_faithfulness"]["probability_change_specificity_gap"]["mean"]
        for result in seed_results
    ]
    bce_gaps = [
        result["label_free_faithfulness"]["bce_specificity_gap"]["mean"]
        for result in seed_results
    ]
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "task": args.task,
        "dataset": task["dataset"],
        "split_policy": "official",
        "official_splits": formal.EXPECTED_SPLITS[task["dataset"]],
        "label_profile": task["binary_profile"],
        "score_profile": task["score_profile"],
        "cache_signature": formal.cache_signature(args.meter_root, task["dataset"]),
        "frozen_checkpoint_audit": True,
        "model_selection_or_training_performed": False,
        "test_labels_used_for_reward_and_final_audit_metrics_only": True,
        "seeds": list(map(int, args.seeds)),
        "summary": {
            "reward_flow": {
                "centered_spearman": mean_std(reward_correlations),
                "fixed_effect_slope": mean_std(reward_slopes),
                "positive_seed_count": int(sum(value > 0 for value in reward_correlations)),
                "seed_count": int(len(reward_correlations)),
            },
            "label_free_faithfulness": {
                "primary_metric": "absolute risk-probability change after evidence deletion",
                "primary_metric_uses_ground_truth": False,
                "critical_probability_change": mean_std(
                    result["label_free_faithfulness"]["critical_probability_change"]["mean"]
                    for result in seed_results
                ),
                "matched_random_probability_change": mean_std(
                    result["label_free_faithfulness"]["matched_random_probability_change"]["mean"]
                    for result in seed_results
                ),
                "probability_change_specificity_gap": mean_std(probability_gaps),
                "positive_probability_gap_seed_count": int(
                    sum(value > 0 for value in probability_gaps)
                ),
                "probability_gap_nested_bootstrap": nested_bootstrap_gap(
                    seed_results,
                    metric="probability_change_specificity_gap",
                    draws=args.bootstrap_draws,
                    seed=formal.stable_seed(SCHEMA, args.task, "probability_gap_bootstrap"),
                ),
                "critical_severity_change": mean_std(
                    result["label_free_faithfulness"]["critical_severity_change"]["mean"]
                    for result in seed_results
                ),
                "matched_random_severity_change": mean_std(
                    result["label_free_faithfulness"]["matched_random_severity_change"]["mean"]
                    for result in seed_results
                ),
                "severity_change_specificity_gap": mean_std(
                    result["label_free_faithfulness"]["severity_change_specificity_gap"]["mean"]
                    for result in seed_results
                ),
                "critical_bce_cost": mean_std(
                    result["label_free_faithfulness"]["critical_bce_cost"]["mean"]
                    for result in seed_results
                ),
                "matched_random_bce_cost": mean_std(
                    result["label_free_faithfulness"]["matched_random_bce_cost"]["mean"]
                    for result in seed_results
                ),
                "bce_specificity_gap": mean_std(bce_gaps),
                "positive_bce_gap_seed_count": int(sum(value > 0 for value in bce_gaps)),
                "seed_count": int(len(bce_gaps)),
                "bce_gap_nested_bootstrap": nested_bootstrap_gap(
                    seed_results,
                    metric="bce_specificity_gap",
                    draws=args.bootstrap_draws,
                    seed=formal.stable_seed(SCHEMA, args.task, "bce_gap_bootstrap"),
                ),
            },
        },
        "runs": seed_results,
    }
    atomic_json(args.output / args.task / "FINAL_AUDIT.json", result)
    return result


def combine(args: argparse.Namespace) -> dict[str, Any]:
    tasks = {}
    for task_name in TASK_NAMES:
        path = args.output / task_name / "FINAL_AUDIT.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        tasks[task_name] = json.loads(path.read_text(encoding="utf-8"))

    figure_rows = []
    for task_name, result in tasks.items():
        for run in result["runs"]:
            figure_rows.append(
                {
                    "task": task_name,
                    "seed": run["seed"],
                    "reward_flow_spearman": run["reward_flow"]["centered_spearman"],
                    "reward_flow_slope": run["reward_flow"]["fixed_effect_slope"],
                    "tb_residual_rmse": run["reward_flow"]["tb_residual"]["rmse"],
                    "shuffled_tb_residual_rmse": run["reward_flow"]["within_participant_reward_permutation"]["tb_rmse"]["mean"],
                    "critical_bce_cost": run["label_free_faithfulness"]["critical_bce_cost"]["mean"],
                    "matched_random_bce_cost": run["label_free_faithfulness"]["matched_random_bce_cost"]["mean"],
                    "label_free_specificity_gap": run["label_free_faithfulness"]["bce_specificity_gap"]["mean"],
                    "critical_probability_change": run["label_free_faithfulness"]["critical_probability_change"]["mean"],
                    "matched_random_probability_change": run["label_free_faithfulness"]["matched_random_probability_change"]["mean"],
                    "label_free_probability_gap": run["label_free_faithfulness"]["probability_change_specificity_gap"]["mean"],
                    "critical_severity_change": run["label_free_faithfulness"]["critical_severity_change"]["mean"],
                    "matched_random_severity_change": run["label_free_faithfulness"]["matched_random_severity_change"]["mean"],
                    "label_free_severity_gap": run["label_free_faithfulness"]["severity_change_specificity_gap"]["mean"],
                }
            )
    csv_path = args.output / "FIGURE2_MECHANISM_DATA.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(figure_rows[0]))
        writer.writeheader()
        writer.writerows(figure_rows)

    result = {
        "schema": SCHEMA,
        "status": "complete",
        "protocol": {
            "split_policy": "official",
            "seeds": list(map(int, args.seeds)),
            "frozen_checkpoint_audit": True,
            "training_or_model_selection": False,
            "label_free_selector": "maximum posterior event-inclusion probability",
            "selector_uses_test_label_or_score": False,
        },
        "tasks": {task_name: value["summary"] for task_name, value in tasks.items()},
        "figure2_data": str(csv_path),
    }
    atomic_json(args.output / "FINAL_METRICS.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(*TASK_NAMES, "combine"), required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--meter-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--random-controls", type=int, default=32)
    parser.add_argument("--null-permutations", type=int, default=200)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--max-participants", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    if args.task == "combine":
        result = combine(args)
    else:
        result = run_task(args)
    print(json.dumps({"event": "figure2_mechanism_audit_complete", "task": args.task, "summary": result.get("summary")}, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
