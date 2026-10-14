#!/usr/bin/env python3
"""Fast, preregistered audit of PsyPath-GFN path utility and faithfulness."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audit_psypath_reward_flow_faithfulness as prior_audit
from scripts import validate_alternating_psychometric_gfn_cached as formal
from scripts.alternating_psychometric_gfn import operator_log_rewards
from scripts.meter_gfn import EvidenceTrajectory, random_matched_mask


SCHEMA = "psypath_path_utility_crossfit_v1"


def mean_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"n": 0, "mean": None, "std": None, "positive_fraction": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def bootstrap_mean(values: Iterable[float], *, draws: int, seed: int) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"draws": int(draws), "mean": None, "ci95": [None, None], "probability_positive": None}
    rng = np.random.default_rng(int(seed))
    sampled = array[rng.integers(0, len(array), size=(int(draws), len(array)))].mean(axis=1)
    return {
        "draws": int(draws),
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "probability_positive": float(np.mean(sampled > 0.0)),
    }


def random_trajectory(trajectory: EvidenceTrajectory, seed: int) -> EvidenceTrajectory:
    mask = random_matched_mask(trajectory.mask, int(seed))
    event_order = torch.nonzero(mask.any(dim=1), as_tuple=False).flatten()
    return EvidenceTrajectory(
        phenotype=int(trajectory.phenotype),
        mask=mask,
        log_forward=torch.zeros_like(trajectory.log_forward),
        log_backward=torch.zeros_like(trajectory.log_backward),
        action_count=int(trajectory.action_count),
        event_order=event_order,
        bundle_order=None,
    )


def terminal_reward_kwargs(
    config: SimpleNamespace, record: dict[str, Any], random_seeds: list[int]
) -> dict[str, Any]:
    values = prior_audit.reward_kwargs(config, record, random_seeds)
    values.update(
        {
            "path_process_weight": 0.0,
            "path_monotonic_weight": 0.0,
            "path_order_weight": 0.0,
            "path_duplicate_weight": 0.0,
        }
    )
    return values


def model_only_event_score(
    operator: Any, record: dict[str, Any], masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    baseline_probability, baseline_severity = prior_audit.posterior_prediction(
        operator, record["projected_np"], masks
    )
    inclusion = masks.any(axis=2).mean(axis=0)
    scores = np.full(masks.shape[1], -np.inf, dtype=np.float64)
    for event in range(masks.shape[1]):
        if inclusion[event] <= 0.0:
            continue
        deleted = masks.copy()
        deleted[:, event, :] = False
        probability, severity = prior_audit.posterior_prediction(
            operator, record["projected_np"], deleted
        )
        score = abs(probability - baseline_probability)
        if severity is not None and baseline_severity is not None:
            score += 0.25 * abs(severity - baseline_severity) / max(float(operator.score_max), 1.0)
        scores[event] = score
    return scores, inclusion


def matched_control_events(
    raw_record: dict[str, Any], critical: int, inclusion: np.ndarray, *, count: int, seed: int
) -> tuple[np.ndarray, str]:
    candidates, tier = prior_audit.matched_candidates(raw_record, critical)
    if len(candidates) == 0:
        return candidates, tier
    distance = np.abs(inclusion[candidates] - inclusion[critical])
    order = np.argsort(distance, kind="mergesort")
    nearest = candidates[order[: min(len(candidates), max(3, int(math.ceil(len(candidates) * 0.25))))]]
    rng = np.random.default_rng(int(seed))
    return rng.choice(nearest, size=max(1, int(count)), replace=True), tier


def heldout_deletion(
    operator: Any,
    record: dict[str, Any],
    raw_record: dict[str, Any],
    discovery_masks: np.ndarray,
    evaluation_masks: np.ndarray,
    *,
    random_controls: int,
    seed: int,
) -> dict[str, Any] | None:
    if discovery_masks.shape[1] < 2:
        return None
    scores, inclusion = model_only_event_score(operator, record, discovery_masks)
    if not np.isfinite(scores).any():
        return None
    critical = int(np.nanargmax(scores))
    controls, tier = matched_control_events(
        raw_record,
        critical,
        inclusion,
        count=random_controls,
        seed=seed,
    )
    if len(controls) == 0:
        return None

    baseline_probability, baseline_severity = prior_audit.posterior_prediction(
        operator, record["projected_np"], evaluation_masks
    )
    deleted = evaluation_masks.copy()
    deleted[:, critical, :] = False
    critical_probability, critical_severity = prior_audit.posterior_prediction(
        operator, record["projected_np"], deleted
    )
    critical_probability_change = abs(critical_probability - baseline_probability)
    critical_severity_change = (
        abs(critical_severity - baseline_severity)
        if critical_severity is not None and baseline_severity is not None
        else None
    )

    random_probability_changes = []
    random_severity_changes = []
    for event in controls:
        control_masks = evaluation_masks.copy()
        control_masks[:, int(event), :] = False
        probability, severity = prior_audit.posterior_prediction(
            operator, record["projected_np"], control_masks
        )
        random_probability_changes.append(abs(probability - baseline_probability))
        if severity is not None and baseline_severity is not None:
            random_severity_changes.append(abs(severity - baseline_severity))
    random_probability_change = float(np.mean(random_probability_changes))
    random_severity_change = (
        float(np.mean(random_severity_changes)) if random_severity_changes else None
    )
    return {
        "critical_event": critical,
        "match_tier": tier,
        "critical_discovery_score": float(scores[critical]),
        "critical_inclusion": float(inclusion[critical]),
        "matched_random_inclusion": float(np.mean(inclusion[controls])),
        "critical_probability_change": critical_probability_change,
        "matched_random_probability_change": random_probability_change,
        "probability_change_gap": critical_probability_change - random_probability_change,
        "critical_severity_change": critical_severity_change,
        "matched_random_severity_change": random_severity_change,
        "severity_change_gap": (
            critical_severity_change - random_severity_change
            if critical_severity_change is not None and random_severity_change is not None
            else None
        ),
    }


def audit_participant(
    model: Any,
    operator: Any,
    config: SimpleNamespace,
    record: dict[str, Any],
    raw_record: dict[str, Any],
    *,
    task: str,
    seed: int,
    samples: int,
    random_controls: int,
    evaluate_path_utility: bool,
) -> dict[str, Any]:
    encoding = model.encode(record["events"])
    trajectories = [
        formal.sample_trajectory(
            model,
            record,
            config,
            formal.stable_seed(SCHEMA, task, seed, record["pid"], "gfn", index),
            gradient=False,
            encoding=encoding,
        )
        for index in range(int(samples))
    ]
    path_utility = {
        "evaluated": False,
        "paired_paths": 0,
        "mean_log_reward_gain": None,
        "positive_pair_fraction": None,
        "component_cost_reduction": None,
    }
    if evaluate_path_utility:
        random_trajectories = [
            random_trajectory(
                trajectory,
                formal.stable_seed(SCHEMA, task, seed, record["pid"], "matched", index),
            )
            for index, trajectory in enumerate(trajectories)
        ]
        reward_seeds = [
            formal.stable_seed(SCHEMA, task, seed, record["pid"], "reward", index)
            for index in range(int(samples))
        ]
        gfn_rewards = operator_log_rewards(
            operator,
            record["projected_np"],
            trajectories,
            **terminal_reward_kwargs(config, record, reward_seeds),
        )
        random_rewards = operator_log_rewards(
            operator,
            record["projected_np"],
            random_trajectories,
            **terminal_reward_kwargs(config, record, reward_seeds),
        )
        log_reward_gain = np.asarray(
            [left["log_reward"] - right["log_reward"] for left, right in zip(gfn_rewards, random_rewards)],
            dtype=np.float64,
        )
        path_utility = {
            "evaluated": True,
            "paired_paths": int(len(log_reward_gain)),
            "mean_log_reward_gain": float(log_reward_gain.mean()),
            "positive_pair_fraction": float(np.mean(log_reward_gain > 0.0)),
            "component_cost_reduction": {
                name: float(
                    np.mean(
                        [right[name] - left[name] for left, right in zip(gfn_rewards, random_rewards)]
                    )
                )
                for name in ("classification", "severity", "necessity", "specificity")
            },
        }

    masks = np.stack(
        [trajectory.mask.detach().cpu().numpy().astype(bool) for trajectory in trajectories]
    )
    midpoint = len(masks) // 2
    folds = []
    for fold, (discovery, evaluation) in enumerate(
        ((masks[:midpoint], masks[midpoint:]), (masks[midpoint:], masks[:midpoint]))
    ):
        result = heldout_deletion(
            operator,
            record,
            raw_record,
            discovery,
            evaluation,
            random_controls=random_controls,
            seed=formal.stable_seed(SCHEMA, task, seed, record["pid"], "deletion", fold),
        )
        if result is not None:
            folds.append(result)

    return {
        "pid": str(record["pid"]),
        "path_utility": path_utility,
        "crossfit_deletion": {
            "eligible": bool(folds),
            "folds": folds,
            "probability_change_gap": (
                float(np.mean([item["probability_change_gap"] for item in folds])) if folds else None
            ),
            "severity_change_gap": (
                float(
                    np.mean(
                        [item["severity_change_gap"] for item in folds if item["severity_change_gap"] is not None]
                    )
                )
                if any(item["severity_change_gap"] is not None for item in folds)
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meter-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(formal.TASKS), required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--max-participants", type=int, default=0)
    parser.add_argument("--skip-path-utility", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.samples < 4 or args.samples % 2:
        raise ValueError("samples must be an even integer >= 4")
    started = time.time()
    formal.configure_runtime(args.threads)
    formal.validate_contract(args.contract_dir)
    device = torch.device(args.device)
    checkpoint = formal.load_trusted_checkpoint(args.checkpoint, device)
    if checkpoint.get("task") != args.task or checkpoint.get("method") != "gflownet":
        raise RuntimeError("checkpoint task/method contract mismatch")
    config = prior_audit.checkpoint_args(checkpoint)
    setattr(config, "exploration_epsilon", 0.0)

    raw = formal.load_records(
        args.meter_root,
        args.contract_dir,
        args.task,
        allowed_splits=(args.split,),
        split_policy="official",
    )[args.split]
    formal.attach_temporal({args.split: raw}, args.meter_root, formal.TASKS[args.task]["dataset"])
    if args.max_participants > 0:
        raw = raw[: args.max_participants]
    records = formal.prepare_records(raw, checkpoint["event_projections"], device)
    raw_by_pid = {str(item["pid"]): item for item in raw}
    model = formal.new_model(records[0], config, device, task_name=args.task)
    model.load_state_dict(checkpoint["gflownet_state"], strict=True)
    model.eval()
    operator = checkpoint["psychometric_operator"]

    participants = []
    for index, record in enumerate(records, 1):
        participants.append(
            audit_participant(
                model,
                operator,
                config,
                record,
                raw_by_pid[str(record["pid"])],
                task=args.task,
                seed=int(checkpoint["seed"]),
                samples=args.samples,
                random_controls=args.random_controls,
                evaluate_path_utility=not args.skip_path_utility,
            )
        )
        print(
            json.dumps(
                {
                    "event": "participant_complete",
                    "index": index,
                    "total": len(records),
                    "pid": str(record["pid"]),
                }
            ),
            flush=True,
        )

    reward_gains = [
        item["path_utility"]["mean_log_reward_gain"]
        for item in participants
        if item["path_utility"]["mean_log_reward_gain"] is not None
    ]
    probability_gaps = [
        item["crossfit_deletion"]["probability_change_gap"]
        for item in participants
        if item["crossfit_deletion"]["probability_change_gap"] is not None
    ]
    severity_gaps = [
        item["crossfit_deletion"]["severity_change_gap"]
        for item in participants
        if item["crossfit_deletion"]["severity_change_gap"] is not None
    ]
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "task": args.task,
        "split": args.split,
        "split_policy": "official",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": formal.sha256_file(args.checkpoint),
        "seed": int(checkpoint["seed"]),
        "participants": int(len(records)),
        "samples_per_participant": int(args.samples),
        "protocol": {
            "path_control": "event-count and modality-bundle matched random paths",
            "path_utility_uses_labels_or_scores_for_final_evaluation": True,
            "crossfit_selector": "model-only prediction influence on disjoint path samples",
            "crossfit_selector_uses_labels_or_scores": False,
            "crossfit_primary_metric_uses_labels_or_scores": False,
            "test_used_for_training_or_selection": False,
        },
        "path_utility": {
            "primary_metric": "GFN minus matched-random terminal log psychometric reward",
            "participant_summary": mean_summary(reward_gains),
            "participant_bootstrap": bootstrap_mean(
                reward_gains,
                draws=args.bootstrap_draws,
                seed=formal.stable_seed(SCHEMA, args.task, args.split, "reward"),
            ),
        },
        "crossfit_faithfulness": {
            "primary_metric": "held-out absolute probability-change gap: selected minus matched random",
            "probability_gap": mean_summary(probability_gaps),
            "probability_gap_bootstrap": bootstrap_mean(
                probability_gaps,
                draws=args.bootstrap_draws,
                seed=formal.stable_seed(SCHEMA, args.task, args.split, "probability"),
            ),
            "severity_gap": mean_summary(severity_gaps),
            "severity_gap_bootstrap": bootstrap_mean(
                severity_gaps,
                draws=args.bootstrap_draws,
                seed=formal.stable_seed(SCHEMA, args.task, args.split, "severity"),
            ),
        },
        "gate": {
            "path_utility_positive": (
                bool(np.mean(reward_gains) > 0.0) if reward_gains else None
            ),
            "crossfit_probability_gap_positive": bool(np.mean(probability_gaps) > 0.0),
            "passed": bool(
                np.mean(probability_gaps) > 0.0
                and (not reward_gains or np.mean(reward_gains) > 0.0)
            ),
        },
        "participant_records": participants,
        "elapsed_s": float(time.time() - started),
    }
    formal.atomic_json(args.output, result)
    print(json.dumps({"event": "audit_complete", "gate": result["gate"]}), flush=True)


if __name__ == "__main__":
    main()
