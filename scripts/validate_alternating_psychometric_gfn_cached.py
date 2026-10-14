#!/usr/bin/env python3
"""Strict alternating psychometric GFlowNet validation without anchor mixing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_adyme_mh_final_gate_v2 as gate
from scripts.alternating_psychometric_gfn import (
    AlternatingPsychometricOperator,
    complement,
    evidence_features,
    fit_direct_operator,
    fit_pim_operator,
    full_mask,
    operator_log_rewards,
    random_sparse_mask,
)
from scripts.clinical_label_contract import validate_contract
from scripts.joint_psychometric_gfn import joint_event_features
from scripts.meter_gfn import (
    HierarchicalClinicalGFlowNet,
    posterior_diversity,
    random_matched_mask,
    reinforce_loss,
    shuffled_evidence_mask,
    trajectory_balance_loss,
)
from scripts.psypath_gfn import MonotonicPsychometricGFlowNet
from scripts.psypath_split_policy import SPLIT_POLICY as PSYPATH_SPLIT_POLICY
from scripts.run_meter_mh_online_operator_gate import EXPECTED_SPLITS, TASKS, cache_signature
from scripts.train_meter_mh_sequential_increment import load_records


SCHEMA = "psychometric_evidence_path_gfn_cached_v1"
SCORE_MAX = {
    "daic_depression": 24.0,
    "edaic_depression": 24.0,
    "edaic_ptsd": 85.0,
    "cmdc_phq9": 27.0,
    "mpdd_phq9": 27.0,
}
VARIANTS = {
    "full": {"pim": True, "ensr": True, "score_permutation": False},
    "no_pim": {"pim": False, "ensr": True, "score_permutation": False},
    "no_ensr": {"pim": True, "ensr": False, "score_permutation": False},
    "score_permutation": {"pim": True, "ensr": True, "score_permutation": True},
    "no_pim_no_ensr": {"pim": False, "ensr": False, "score_permutation": False},
    "score_permutation_no_ensr": {"pim": True, "ensr": False, "score_permutation": True},
}


def configure_runtime(threads: int) -> None:
    threads = max(1, min(32, int(threads)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**31)


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def finite_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    valid = array[np.isfinite(array)]
    return float(valid.mean()) if len(valid) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trusted_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def metric_block(labels: Iterable[int], probability: Iterable[float], threshold: float) -> dict[str, Any]:
    target = np.asarray(list(labels), dtype=np.int64)
    probability = np.clip(np.asarray(list(probability), dtype=np.float64), 1e-6, 1.0 - 1e-6)
    prediction = (probability >= float(threshold)).astype(np.int64)
    result = {
        "n": int(len(target)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, prediction)),
        "weighted_f1": float(f1_score(target, prediction, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(target, prediction, zero_division=0)),
        "brier": float(np.mean(np.square(probability - target))),
    }
    result["auroc"] = float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None
    return result


def choose_threshold(labels: list[int], probability: list[float]) -> float:
    choices = []
    for threshold in np.arange(0.15, 0.851, 0.025):
        block = metric_block(labels, probability, float(threshold))
        choices.append(
            (
                block["weighted_f1"],
                block["macro_f1"],
                block.get("auroc") or 0.0,
                -block["brier"],
                -abs(float(threshold) - 0.5),
                float(threshold),
            )
        )
    return float(max(choices)[-1])


def severity_block(records: list[dict[str, Any]], prediction: Iterable[float]) -> dict[str, Any]:
    pairs = [
        (float(record["score"]), float(value))
        for record, value in zip(records, prediction)
        if record.get("score") is not None and np.isfinite(float(value))
    ]
    if len(pairs) < 2:
        return {"n": len(pairs), "ccc": None, "mae": None, "rmse": None}
    target = np.asarray([item[0] for item in pairs], dtype=np.float64)
    predicted = np.asarray([item[1] for item in pairs], dtype=np.float64)
    covariance = float(np.mean((target - target.mean()) * (predicted - predicted.mean())))
    denominator = float(target.var() + predicted.var() + (target.mean() - predicted.mean()) ** 2)
    return {
        "n": int(len(target)),
        "ccc": 2.0 * covariance / denominator if denominator > 1e-12 else None,
        "mae": float(mean_absolute_error(target, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(target, predicted))),
    }


def temporal_features(turns: list[dict[str, Any]]) -> np.ndarray:
    count = len(turns)
    position = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    duration = np.asarray(
        [max(0.0, float(turn.get("stop", 0.0)) - float(turn.get("start", 0.0))) for turn in turns],
        dtype=np.float32,
    )
    latency = []
    previous_stop = None
    for turn in turns:
        observed = float(turn.get("latency", 0.0) or 0.0)
        if observed <= 0.0 and previous_stop is not None:
            observed = max(0.0, float(turn.get("start", 0.0)) - previous_stop)
        latency.append(observed)
        previous_stop = float(turn.get("stop", 0.0))
    question = np.asarray([float(bool(turn.get("question_available", False))) for turn in turns], dtype=np.float32)

    def standardized(values: np.ndarray) -> np.ndarray:
        values = np.log1p(np.maximum(values, 0.0))
        return ((values - values.mean()) / max(float(values.std()), 1e-6)).astype(np.float32)

    return np.stack((position, standardized(duration), standardized(np.asarray(latency)), question), axis=1)


def attach_temporal(records: dict[str, list[dict[str, Any]]], root: Path, dataset: str) -> None:
    for split_records in records.values():
        for record in split_records:
            path = root / dataset / "cache" / f"{record['pid']}.npz"
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
            turns = list(metadata.get("turns", []))
            if len(turns) != len(record["raw"]["answer"]):
                raise RuntimeError(f"turn/event mismatch: {dataset}/{record['pid']}")
            record["temporal"] = temporal_features(turns)


def fit_event_projections(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    shared_names = ("clinical", "answer", "context")
    widths = {
        int(record["raw"][name].shape[1])
        for record in records
        for name in shared_names
    }
    if len(widths) == 1:
        shared = gate.fit_projection(
            np.concatenate(
                [record["raw"][name] for record in records for name in shared_names],
                axis=0,
            ),
            seed + 11,
            gate.EVENT_COMPONENTS,
        )
        shared_by_view = None
    else:
        shared = None
        shared_by_view = {
            name: gate.fit_projection(
                np.concatenate([record["raw"][name] for record in records], axis=0),
                seed + 11 + index,
                gate.EVENT_COMPONENTS,
            )
            for index, name in enumerate(shared_names)
        }
    visual = gate.fit_projection(
        np.concatenate([record["raw"]["visual"] for record in records], axis=0),
        seed + 13,
        gate.EVENT_COMPONENTS,
    )
    return {"shared": shared, "shared_by_view": shared_by_view, "visual": visual}


def prepare_records(
    records: list[dict[str, Any]], projections: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    prepared = []
    for record in records:
        projected_np = {}
        for name in ("clinical", "answer", "context", "visual"):
            if name == "visual":
                projection = projections["visual"]
            elif projections.get("shared_by_view") is not None:
                projection = projections["shared_by_view"][name]
            else:
                projection = projections["shared"]
            projected_np[name] = gate.transform(record["raw"][name], projection)
        projected_torch = {
            name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in projected_np.items()
        }
        temporal = torch.as_tensor(record["temporal"], dtype=torch.float32, device=device)
        prepared.append(
            {
                "pid": str(record["pid"]),
                "label": int(record["label"]),
                "score": record.get("score"),
                "projected_np": projected_np,
                "events": joint_event_features(projected_torch, temporal),
            }
        )
    return prepared


def score_targets(records: list[dict[str, Any]], variant: str, seed: int) -> dict[str, float]:
    if not VARIANTS[variant]["score_permutation"]:
        return {}
    observed = [(record["pid"], float(record["score"])) for record in records if record.get("score") is not None]
    values = np.asarray([score for _, score in observed], dtype=np.float64)
    shuffled = np.random.default_rng(stable_seed("alternating_score", seed)).permutation(values)
    return {pid: float(value) for (pid, _), value in zip(observed, shuffled)}


def event_limits(count: int, args: argparse.Namespace) -> tuple[int, int]:
    minimum = min(count, max(1, int(args.minimum_events)))
    maximum = min(count, max(minimum, int(math.ceil(float(args.maximum_fraction) * count))))
    maximum = min(maximum, int(args.maximum_events))
    return minimum, max(minimum, maximum)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    return generator


def sample_trajectory(
    model: HierarchicalClinicalGFlowNet,
    record: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
    *,
    gradient: bool,
    encoding: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    minimum, maximum = event_limits(len(record["events"]), args)
    generator = make_generator(record["events"].device, seed)

    def draw():
        if encoding is None:
            return model.trajectory(
                record["events"],
                minimum_events=minimum,
                maximum_events=maximum,
                generator=generator,
                exploration_epsilon=(float(args.exploration_epsilon) if gradient else 0.0),
            )
        return model.trajectory_from_encoding(
            encoding[0],
            encoding[1],
            minimum_events=minimum,
            maximum_events=maximum,
            generator=generator,
            exploration_epsilon=(float(args.exploration_epsilon) if gradient else 0.0),
        )

    if gradient:
        return draw()
    with torch.no_grad():
        return draw()


def augmentation_masks(
    model: HierarchicalClinicalGFlowNet,
    record: dict[str, Any],
    args: argparse.Namespace,
    *,
    cycle: int,
    seed: int,
) -> list[np.ndarray]:
    length = len(record["events"])
    minimum, maximum = event_limits(length, args)
    masks = [full_mask(length)]
    encoding = None
    for index in range(args.operator_trajectories):
        if cycle == 0 or index == args.operator_trajectories - 1:
            mask = random_sparse_mask(
                length,
                minimum,
                maximum,
                stable_seed("operator_random", seed, cycle, record["pid"], index),
            )
        else:
            if encoding is None:
                with torch.no_grad():
                    encoding = model.encode(record["events"])
            trajectory = sample_trajectory(
                model,
                record,
                args,
                stable_seed("operator_gfn", seed, cycle, record["pid"], index),
                gradient=False,
                encoding=encoding,
            )
            mask = trajectory.mask.cpu().numpy().astype(bool)
        masks.append(mask)
    return masks


def fit_operator(
    model: HierarchicalClinicalGFlowNet,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    task_name: str,
    variant: str,
    cycle: int,
    seed: int,
    permuted_scores: dict[str, float],
) -> AlternatingPsychometricOperator:
    augmented_features = []
    augmented_labels = []
    augmented_scores = []
    augmented_groups = []
    full_features = []
    labels = []
    scores = []
    participant_ids = []
    for record in records:
        participant_ids.append(record["pid"])
        labels.append(record["label"])
        score = permuted_scores.get(record["pid"], record.get("score"))
        scores.append(float(score) if score is not None else np.nan)
        full_features.append(
            evidence_features(record["projected_np"], full_mask(len(record["events"])))
        )
        for mask in augmentation_masks(model, record, args, cycle=cycle, seed=seed):
            augmented_features.append(evidence_features(record["projected_np"], mask))
            augmented_labels.append(record["label"])
            augmented_scores.append(float(score) if score is not None else np.nan)
            augmented_groups.append(record["pid"])
    augmented_features_array = np.stack(augmented_features)
    full_features_array = np.stack(full_features)
    common = {
        "augmented_features": augmented_features_array,
        "augmented_groups": np.asarray(augmented_groups),
        "full_features": full_features_array,
        "labels": np.asarray(labels, dtype=np.int64),
        "participant_ids": np.asarray(participant_ids),
        "score_max": SCORE_MAX[task_name],
        "seed": stable_seed(task_name, seed, cycle, "operator"),
        "components": args.measurement_components,
    }
    if VARIANTS[variant]["pim"]:
        return fit_pim_operator(
            augmented_scores=np.asarray(augmented_scores, dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float64),
            **common,
        )
    return fit_direct_operator(
        augmented_labels=np.asarray(augmented_labels, dtype=np.int64),
        **common,
    )


def fit_full_evidence_operator(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    task_name: str,
    variant: str,
    seed: int,
    permuted_scores: dict[str, float],
) -> AlternatingPsychometricOperator:
    features = np.stack(
        [
            evidence_features(record["projected_np"], full_mask(len(record["events"])))
            for record in records
        ]
    )
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    scores = np.asarray(
        [
            float(permuted_scores.get(record["pid"], record.get("score")))
            if permuted_scores.get(record["pid"], record.get("score")) is not None
            else np.nan
            for record in records
        ],
        dtype=np.float64,
    )
    participant_ids = np.asarray([record["pid"] for record in records])
    common = {
        "augmented_features": features,
        "augmented_groups": participant_ids,
        "full_features": features,
        "labels": labels,
        "participant_ids": participant_ids,
        "score_max": SCORE_MAX[task_name],
        "seed": stable_seed(task_name, seed, "independent_full_operator"),
        "components": args.measurement_components,
    }
    if VARIANTS[variant]["pim"]:
        return fit_pim_operator(
            augmented_scores=scores,
            scores=scores,
            **common,
        )
    return fit_direct_operator(
        augmented_labels=labels,
        **common,
    )


def train_selector_epoch(
    model: HierarchicalClinicalGFlowNet | MonotonicPsychometricGFlowNet,
    operator: AlternatingPsychometricOperator | dict[str, AlternatingPsychometricOperator],
    records: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    task_name: str,
    variant: str,
    method: str,
    seed: int,
    cycle: int,
    epoch: int,
    permuted_scores: dict[str, float],
) -> float:
    model.train()
    order = np.random.default_rng(stable_seed("selector_order", seed, cycle, epoch)).permutation(len(records))
    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, index in enumerate(order, 1):
        record = records[int(index)]
        encoding = model.encode(record["events"])
        trajectories = []
        random_seeds = []
        for sample in range(args.selector_trajectories):
            trajectory = sample_trajectory(
                model,
                record,
                args,
                stable_seed("selector", seed, cycle, epoch, record["pid"], sample),
                gradient=True,
                encoding=encoding,
            )
            random_seeds.append(
                stable_seed("selector_random", seed, cycle, epoch, record["pid"], sample)
            )
            trajectories.append(trajectory)
        target_score = permuted_scores.get(record["pid"], record.get("score"))
        record_operator = operator[record["pid"]] if isinstance(operator, dict) else operator
        reward_values_list = operator_log_rewards(
            record_operator,
            record["projected_np"],
            trajectories,
            label=record["label"],
            score=target_score,
            random_seeds=random_seeds,
            pim_weight=args.pim_reward_weight if VARIANTS[variant]["pim"] else 0.0,
            ensr_weight=args.ensr_reward_weight if VARIANTS[variant]["ensr"] else 0.0,
            sufficiency_weight=args.sufficiency_weight,
            compactness_weight=args.compactness_weight,
            necessity_margin=args.necessity_margin,
            specificity_margin=args.specificity_margin,
            temperature=args.reward_temperature,
            path_process_weight=args.path_process_weight if args.path_process else 0.0,
            path_monotonic_weight=args.path_monotonic_weight if args.path_process else 0.0,
            path_order_weight=args.path_order_weight if args.path_process else 0.0,
            path_order_margin=args.path_order_margin,
            path_duplicate_weight=args.path_duplicate_weight,
        )
        rewards = []
        for trajectory, reward_values in zip(trajectories, reward_values_list):
            dtype = trajectory.log_forward.dtype
            device = trajectory.log_forward.device
            rewards.append(
                SimpleNamespace(
                    **{
                        key: torch.tensor(float(value), dtype=dtype, device=device)
                        for key, value in reward_values.items()
                        if key in (
                            "log_reward",
                            "classification",
                            "severity",
                            "necessity",
                            "specificity",
                            "compactness",
                            "path_process",
                            "path_monotonic",
                            "path_order",
                            "path_duplicate",
                        )
                    }
                )
            )
        if method == "gflownet":
            loss = trajectory_balance_loss(
                model,
                record["events"],
                trajectories,
                rewards,
                log_z=model.partition_from_context(encoding[1]),
                relative_weight=args.relative_tb_weight,
            )
        elif method == "reinforce":
            loss = reinforce_loss(trajectories, rewards, entropy_weight=args.entropy_weight)
        else:
            raise ValueError(method)
        (loss / args.gradient_accumulation).backward()
        total += float(loss.detach().cpu())
        if step % args.gradient_accumulation == 0 or step == len(order):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return total / max(1, len(records))


@torch.no_grad()
def posterior_prediction(
    model: HierarchicalClinicalGFlowNet,
    operator: AlternatingPsychometricOperator,
    full_operator: AlternatingPsychometricOperator,
    record: dict[str, Any],
    args: argparse.Namespace,
    *,
    samples: int,
    seed: int,
    include_controls: bool,
) -> dict[str, Any]:
    model.eval()
    probability = []
    severity = []
    masks = []
    phenotypes = []
    selected_delete_cost = []
    random_delete_cost = []
    reference_full = operator.predict(
        evidence_features(record["projected_np"], full_mask(len(record["events"])))
    )
    independent_full = full_operator.predict(
        evidence_features(record["projected_np"], full_mask(len(record["events"])))
    )

    def ce(value: float) -> float:
        value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
        label = record["label"]
        return float(-(label * np.log(value) + (1 - label) * np.log(1.0 - value)))

    full_ce = ce(float(reference_full["probability"][0]))
    encoding = model.encode(record["events"])
    random_masks = []
    shuffled_masks = []
    forward_path_probability = []
    reverse_path_probability = []
    shuffled_path_probability = []
    forward_path_severity = []
    reverse_path_severity = []
    shuffled_path_severity = []
    critical_delete_cost = []
    random_step_delete_cost = []

    def prefix_measurements(trajectory, order: list[int]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        terminal = trajectory.mask.detach().cpu().numpy().astype(bool)
        prefix = np.zeros_like(terminal)
        prefix_masks = []
        for event in order:
            prefix[int(event)] = terminal[int(event)]
            prefix_masks.append(prefix.copy())
        if not prefix_masks:
            prefix_masks.append(terminal.copy())
        measured = operator.predict(
            np.stack(
                [evidence_features(record["projected_np"], item) for item in prefix_masks]
            )
        )
        return (
            np.asarray(measured["probability"], dtype=np.float64),
            np.asarray(measured["severity"], dtype=np.float64),
            prefix_masks,
        )
    for sample in range(samples):
        trajectory = sample_trajectory(
            model,
            record,
            args,
            stable_seed(seed, record["pid"], sample),
            gradient=False,
            encoding=encoding,
        )
        mask = trajectory.mask.cpu().numpy().astype(bool)
        masks.append(trajectory.mask.detach().clone())
        phenotypes.append(trajectory.phenotype)
        if args.path_process:
            event_order = trajectory.event_order.detach().cpu().tolist()
            forward_p, forward_s, prefix_masks = prefix_measurements(trajectory, event_order)
            reverse_p, reverse_s, _ = prefix_measurements(
                trajectory, list(reversed(event_order))
            )
            shuffled_order = list(event_order)
            np.random.default_rng(
                stable_seed(seed, record["pid"], sample, "path_order_shuffle")
            ).shuffle(shuffled_order)
            shuffled_p, shuffled_s, _ = prefix_measurements(trajectory, shuffled_order)
            forward_path_probability.append(float(np.mean(forward_p)))
            reverse_path_probability.append(float(np.mean(reverse_p)))
            shuffled_path_probability.append(float(np.mean(shuffled_p)))
            forward_path_severity.append(float(np.nanmean(forward_s)))
            reverse_path_severity.append(float(np.nanmean(reverse_s)))
            shuffled_path_severity.append(float(np.nanmean(shuffled_s)))

            losses = []
            for probability_value, severity_value in zip(forward_p, forward_s):
                loss = ce(float(probability_value))
                if operator.mode == "pim" and record.get("score") is not None:
                    loss += args.pim_reward_weight * abs(
                        float(severity_value) - float(record["score"])
                    ) / max(operator.score_max, 1.0)
                losses.append(loss)
            gains = np.asarray(
                [0.0] + [losses[index - 1] - losses[index] for index in range(1, len(losses))],
                dtype=np.float64,
            )
            critical_step = int(np.argmax(gains))
            random_step = int(
                np.random.default_rng(
                    stable_seed(seed, record["pid"], sample, "random_path_step")
                ).integers(0, max(1, len(event_order)))
            )
            terminal = trajectory.mask.detach().cpu().numpy().astype(bool)
            critical_mask = terminal.copy()
            random_step_mask = terminal.copy()
            critical_mask[int(event_order[critical_step])] = False
            random_step_mask[int(event_order[random_step])] = False
            critical_prediction = operator.predict(
                evidence_features(record["projected_np"], critical_mask)
            )
            random_step_prediction = operator.predict(
                evidence_features(record["projected_np"], random_step_mask)
            )
            terminal_prediction = operator.predict(
                evidence_features(record["projected_np"], terminal)
            )
            terminal_ce = ce(float(terminal_prediction["probability"][0]))
            critical_delete_cost.append(
                ce(float(critical_prediction["probability"][0])) - terminal_ce
            )
            random_step_delete_cost.append(
                ce(float(random_step_prediction["probability"][0])) - terminal_ce
            )
        if include_controls:
            random_mask = random_matched_mask(
                trajectory.mask, stable_seed(seed, record["pid"], sample, "random")
            ).cpu().numpy().astype(bool)
            shuffled_mask = shuffled_evidence_mask(
                trajectory.mask, stable_seed(seed, record["pid"], sample, "shuffle")
            ).cpu().numpy().astype(bool)
            random_masks.append(random_mask)
            shuffled_masks.append(shuffled_mask)
    mask_arrays = [mask.cpu().numpy().astype(bool) for mask in masks]
    selected = operator.predict(
        np.stack([evidence_features(record["projected_np"], mask) for mask in mask_arrays])
    )
    probability = selected["probability"].astype(np.float64).tolist()
    severity = selected["severity"].astype(np.float64).tolist()
    shuffled_probability = []
    if include_controls:
        selected_delete = operator.predict(
            np.stack(
                [evidence_features(record["projected_np"], complement(mask)) for mask in mask_arrays]
            )
        )
        random_delete = operator.predict(
            np.stack(
                [evidence_features(record["projected_np"], complement(mask)) for mask in random_masks]
            )
        )
        shuffled = operator.predict(
            np.stack(
                [evidence_features(record["projected_np"], mask) for mask in shuffled_masks]
            )
        )
        selected_delete_cost = [ce(float(value)) - full_ce for value in selected_delete["probability"]]
        random_delete_cost = [ce(float(value)) - full_ce for value in random_delete["probability"]]
        shuffled_probability = shuffled["probability"].astype(np.float64).tolist()
    counts = np.bincount(phenotypes, minlength=args.phenotype_count).astype(np.float64)
    distribution = counts / max(1.0, counts.sum())
    nonzero = distribution[distribution > 0]
    result = {
        "probability": float(np.mean(probability)),
        "severity": float(np.nanmean(severity)) if np.isfinite(severity).any() else np.nan,
        "probability_std": float(np.std(probability)),
        "severity_std": float(np.nanstd(severity)) if np.isfinite(severity).any() else np.nan,
        "diversity": posterior_diversity(masks),
        "phenotype_entropy": float(-(nonzero * np.log(nonzero)).sum()),
        "phenotype_coverage": int((counts > 0).sum()),
        "selected_fraction": float(np.mean([mask.any(dim=1).float().mean().item() for mask in masks])),
        "full_probability": float(independent_full["probability"][0]),
        "full_severity": float(independent_full["severity"][0]),
    }
    if include_controls:
        result.update(
            {
                "selected_delete_cost": float(np.mean(selected_delete_cost)),
                "random_delete_cost": float(np.mean(random_delete_cost)),
                "shuffled_probability": float(np.mean(shuffled_probability)),
            }
        )
        if args.path_process:
            result.update(
                {
                    "forward_path_probability": float(np.mean(forward_path_probability)),
                    "reverse_path_probability": float(np.mean(reverse_path_probability)),
                    "shuffled_path_probability": float(np.mean(shuffled_path_probability)),
                    "forward_path_severity": float(np.mean(forward_path_severity)),
                    "reverse_path_severity": float(np.mean(reverse_path_severity)),
                    "shuffled_path_severity": float(np.mean(shuffled_path_severity)),
                    "critical_step_delete_cost": float(np.mean(critical_delete_cost)),
                    "random_step_delete_cost": float(np.mean(random_step_delete_cost)),
                }
            )
    return result


@torch.no_grad()
def evaluate(
    model: HierarchicalClinicalGFlowNet,
    operator: AlternatingPsychometricOperator,
    full_operator: AlternatingPsychometricOperator,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    samples: int,
    seed: int,
    threshold: float | None,
    full_threshold: float | None,
    include_controls: bool,
) -> dict[str, Any]:
    predictions = [
        posterior_prediction(
            model,
            operator,
            full_operator,
            record,
            args,
            samples=samples,
            seed=stable_seed(seed, "posterior"),
            include_controls=include_controls,
        )
        for record in records
    ]
    labels = [record["label"] for record in records]
    posterior_probability = [item["probability"] for item in predictions]
    full_probability = [item["full_probability"] for item in predictions]
    if threshold is None:
        threshold = choose_threshold(labels, posterior_probability)
    if full_threshold is None:
        full_threshold = choose_threshold(labels, full_probability)
    result = {
        "threshold": threshold,
        "full_threshold": full_threshold,
        "posterior": metric_block(labels, posterior_probability, threshold),
        "full_evidence": metric_block(labels, full_probability, full_threshold),
        "severity": {
            "posterior": severity_block(records, [item["severity"] for item in predictions]),
            "full_evidence": severity_block(records, [item["full_severity"] for item in predictions]),
        },
        "uncertainty": {
            "mean_probability_std": float(np.mean([item["probability_std"] for item in predictions])),
            "mean_severity_std": finite_mean([item["severity_std"] for item in predictions]),
            "mean_pairwise_jaccard_distance": float(np.mean([item["diversity"] for item in predictions])),
            "mean_phenotype_entropy": float(np.mean([item["phenotype_entropy"] for item in predictions])),
            "mean_phenotype_coverage": float(np.mean([item["phenotype_coverage"] for item in predictions])),
            "mean_selected_fraction": float(np.mean([item["selected_fraction"] for item in predictions])),
        },
        "probability": {"posterior": posterior_probability, "full_evidence": full_probability},
        "severity_predictions": {
            "posterior": [item["severity"] for item in predictions],
            "full_evidence": [item["full_severity"] for item in predictions],
        },
        "labels": labels,
        "pids": [record["pid"] for record in records],
    }
    if include_controls:
        shuffled_probability = [item["shuffled_probability"] for item in predictions]
        selected_cost = [item["selected_delete_cost"] for item in predictions]
        random_cost = [item["random_delete_cost"] for item in predictions]
        result["controls"] = {
            "evidence_shuffle": metric_block(labels, shuffled_probability, threshold),
            "selected_delete_cost": float(np.mean(selected_cost)),
            "random_delete_cost": float(np.mean(random_cost)),
            "specificity_gap": float(np.mean(np.asarray(selected_cost) - np.asarray(random_cost))),
            "relative_specificity_gap": float(
                np.mean(np.asarray(selected_cost) - np.asarray(random_cost))
                / max(
                    float(
                        np.median(
                            np.abs(np.concatenate((np.asarray(selected_cost), np.asarray(random_cost))))
                        )
                    ),
                    1e-4,
                )
            ),
        }
        if args.path_process:
            forward_probability = [item["forward_path_probability"] for item in predictions]
            reverse_probability = [item["reverse_path_probability"] for item in predictions]
            shuffled_path_probability = [item["shuffled_path_probability"] for item in predictions]
            forward_severity = [item["forward_path_severity"] for item in predictions]
            reverse_severity = [item["reverse_path_severity"] for item in predictions]
            shuffled_severity = [item["shuffled_path_severity"] for item in predictions]
            critical_cost = [item["critical_step_delete_cost"] for item in predictions]
            random_step_cost = [item["random_step_delete_cost"] for item in predictions]
            result["controls"].update(
                {
                    "path_forward": metric_block(labels, forward_probability, threshold),
                    "path_reverse": metric_block(labels, reverse_probability, threshold),
                    "path_shuffle": metric_block(labels, shuffled_path_probability, threshold),
                    "path_forward_severity": severity_block(records, forward_severity),
                    "path_reverse_severity": severity_block(records, reverse_severity),
                    "path_shuffle_severity": severity_block(records, shuffled_severity),
                    "critical_step_delete_cost": float(np.mean(critical_cost)),
                    "random_step_delete_cost": float(np.mean(random_step_cost)),
                    "critical_step_specificity_gap": float(
                        np.mean(np.asarray(critical_cost) - np.asarray(random_step_cost))
                    ),
                }
            )
    return result


def new_model(
    record: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    *,
    task_name: str,
) -> HierarchicalClinicalGFlowNet | MonotonicPsychometricGFlowNet:
    if args.policy == "monotonic":
        instrument_id = {
            "daic_depression": 0,
            "edaic_depression": 0,
            "edaic_ptsd": 1,
        }[task_name]
        return MonotonicPsychometricGFlowNet(
            int(record["events"].shape[1]),
            hidden_dim=args.hidden_dim,
            view_count=len(record["projected_np"]),
            instrument_count=2,
            instrument_id=instrument_id,
        ).to(device=device, dtype=torch.float32)
    return HierarchicalClinicalGFlowNet(
        int(record["events"].shape[1]),
        hidden_dim=args.hidden_dim,
        phenotype_count=args.phenotype_count,
    ).to(device=device, dtype=torch.float32)


def fit_cross_fitted_operator_map(
    model: HierarchicalClinicalGFlowNet | MonotonicPsychometricGFlowNet,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    task_name: str,
    variant: str,
    cycle: int,
    seed: int,
    permuted_scores: dict[str, float],
) -> dict[str, AlternatingPsychometricOperator]:
    """Fit one nuisance measurement operator without each held-out participant."""

    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    nonzero = counts[counts > 0]
    if len(nonzero) < 2 or int(nonzero.min()) < 2:
        raise RuntimeError(f"insufficient classes for reward cross-fitting: {counts.tolist()}")
    folds = min(int(args.reward_folds), int(nonzero.min()))
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=stable_seed(task_name, seed, cycle, "reward_cross_fit"),
    )
    result: dict[str, AlternatingPsychometricOperator] = {}
    indices = np.arange(len(records))
    for fold, (fit_indices, held_indices) in enumerate(splitter.split(indices, labels)):
        fit_records = [records[int(index)] for index in fit_indices]
        held_records = [records[int(index)] for index in held_indices]
        operator = fit_operator(
            model,
            fit_records,
            args,
            task_name=task_name,
            variant=variant,
            cycle=cycle,
            seed=stable_seed(seed, cycle, "reward_fold", fold),
            permuted_scores=permuted_scores,
        )
        for record in held_records:
            if record["pid"] in result:
                raise RuntimeError(f"duplicate reward operator for {record['pid']}")
            result[record["pid"]] = operator
    expected = {record["pid"] for record in records}
    if set(result) != expected:
        missing = sorted(expected - set(result))
        raise RuntimeError(f"incomplete reward cross-fitting: {missing[:10]}")
    return result


def development_selection_key(development: dict[str, Any]) -> tuple[float, ...]:
    metrics = development["posterior"]
    severity = development["severity"]["posterior"]
    return (
        float(metrics["weighted_f1"]),
        float(metrics["macro_f1"]),
        float(metrics.get("auroc") or 0.0),
        float(severity.get("ccc") if severity.get("ccc") is not None else -1.0),
        -float(metrics["brier"]),
    )


def _rank_average(values: np.ndarray) -> np.ndarray:
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


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return None
    left_rank = _rank_average(left[valid])
    right_rank = _rank_average(right[valid])
    if float(left_rank.std()) <= 1e-12 or float(right_rank.std()) <= 1e-12:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


@torch.no_grad()
def development_mechanism_audit(
    model: HierarchicalClinicalGFlowNet,
    operator: AlternatingPsychometricOperator,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    task_name: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    """Score mechanism quality on Dev without using labels for evidence selection."""
    model.eval()
    reward_arrays = []
    flow_arrays = []
    residual_arrays = []
    probability_gaps = []
    participant_rhos = []
    for record in records:
        encoding = model.encode(record["events"])
        trajectories = [
            sample_trajectory(
                model,
                record,
                args,
                stable_seed("mechanism_dev", seed, record["pid"], sample),
                gradient=False,
                encoding=encoding,
            )
            for sample in range(args.mechanism_development_samples)
        ]
        rewards = operator_log_rewards(
            operator,
            record["projected_np"],
            trajectories,
            label=record["label"],
            score=record.get("score"),
            random_seeds=[
                stable_seed("mechanism_dev_random", seed, record["pid"], sample)
                for sample in range(args.mechanism_development_samples)
            ],
            pim_weight=args.pim_reward_weight if VARIANTS[variant]["pim"] else 0.0,
            ensr_weight=args.ensr_reward_weight if VARIANTS[variant]["ensr"] else 0.0,
            sufficiency_weight=args.sufficiency_weight,
            compactness_weight=args.compactness_weight,
            necessity_margin=args.necessity_margin,
            specificity_margin=args.specificity_margin,
            temperature=args.reward_temperature,
            path_process_weight=args.path_process_weight if args.path_process else 0.0,
            path_monotonic_weight=args.path_monotonic_weight if args.path_process else 0.0,
            path_order_weight=args.path_order_weight if args.path_process else 0.0,
            path_order_margin=args.path_order_margin,
            path_duplicate_weight=args.path_duplicate_weight,
        )
        log_reward = np.asarray([float(item["log_reward"]) for item in rewards])
        log_forward = np.asarray([float(item.log_forward.detach().cpu()) for item in trajectories])
        log_backward = np.asarray([float(item.log_backward.detach().cpu()) for item in trajectories])
        flow = log_forward - log_backward
        log_z = float(model.partition_from_context(encoding[1]).detach().cpu())
        reward_arrays.append(log_reward)
        flow_arrays.append(flow)
        residual_arrays.append(log_z + flow - log_reward)
        rho = _spearman(log_reward, flow)
        if rho is not None:
            participant_rhos.append(rho)

        masks = np.stack(
            [item.mask.detach().cpu().numpy().astype(bool) for item in trajectories], axis=0
        )
        event_count = masks.shape[1]
        if event_count < 2:
            continue
        inclusion = masks.any(axis=2).mean(axis=0)
        critical = int(np.argmax(inclusion))
        candidates = np.asarray([index for index in range(event_count) if index != critical])
        rng = np.random.default_rng(stable_seed("mechanism_dev_controls", seed, record["pid"]))
        if len(candidates) > args.mechanism_random_controls:
            candidates = rng.choice(
                candidates, size=args.mechanism_random_controls, replace=False
            )
        base_features = np.stack(
            [evidence_features(record["projected_np"], mask) for mask in masks]
        )
        base_probability = float(np.mean(operator.predict(base_features)["probability"]))
        critical_masks = masks.copy()
        critical_masks[:, critical, :] = False
        critical_probability = float(
            np.mean(
                operator.predict(
                    np.stack(
                        [
                            evidence_features(record["projected_np"], mask)
                            for mask in critical_masks
                        ]
                    )
                )["probability"]
            )
        )
        random_changes = []
        for candidate in candidates:
            candidate_masks = masks.copy()
            candidate_masks[:, int(candidate), :] = False
            probability = float(
                np.mean(
                    operator.predict(
                        np.stack(
                            [
                                evidence_features(record["projected_np"], mask)
                                for mask in candidate_masks
                            ]
                        )
                    )["probability"]
                )
            )
            random_changes.append(abs(probability - base_probability))
        probability_gaps.append(
            abs(critical_probability - base_probability) - float(np.mean(random_changes))
        )

    centered_reward = np.concatenate([value - value.mean() for value in reward_arrays])
    centered_flow = np.concatenate([value - value.mean() for value in flow_arrays])
    residual = np.concatenate(residual_arrays)
    return {
        "reward_flow_spearman": _spearman(centered_reward, centered_flow),
        "participant_reward_flow_spearman_mean": (
            float(np.mean(participant_rhos)) if participant_rhos else None
        ),
        "tb_residual_rmse": float(np.sqrt(np.square(residual).mean())),
        "label_free_probability_gap": (
            float(np.mean(probability_gaps)) if probability_gaps else None
        ),
        "label_free_selector": "maximum posterior event-inclusion probability",
        "selector_uses_label_or_score": False,
        "participants": int(len(records)),
        "counterfactual_eligible_participants": int(len(probability_gaps)),
        "path_samples_per_participant": int(args.mechanism_development_samples),
    }


def _mechanism_selection_key(snapshot: dict[str, Any]) -> tuple[float, ...]:
    mechanism = snapshot["mechanism_development"]
    rho = mechanism.get("reward_flow_spearman")
    gap = mechanism.get("label_free_probability_gap")
    rho = float(rho) if rho is not None else -1.0
    gap = float(gap) if gap is not None else -1.0
    metrics = snapshot["development"]["posterior"]
    return (
        float(rho > 0.0 and gap > 0.0),
        float(rho > 0.0) + float(gap > 0.0),
        rho,
        gap,
        float(metrics["weighted_f1"]),
        -float(mechanism["tb_residual_rmse"]),
    )


def run_schedule(
    model: HierarchicalClinicalGFlowNet | MonotonicPsychometricGFlowNet,
    records: list[dict[str, Any]],
    full_operator: AlternatingPsychometricOperator,
    args: argparse.Namespace,
    *,
    task_name: str,
    variant: str,
    method: str,
    seed: int,
    cycles: int,
    last_cycle_epochs: int,
    evaluate_records: list[dict[str, Any]] | None,
) -> tuple[AlternatingPsychometricOperator, list[dict[str, Any]], dict[str, Any] | None]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    permutation = score_targets(records, variant, seed)
    history = []
    operator = None
    best_snapshot = None
    snapshots = []
    for cycle in range(cycles):
        operator = fit_operator(
            model,
            records,
            args,
            task_name=task_name,
            variant=variant,
            cycle=cycle,
            seed=seed,
            permuted_scores=permutation,
        )
        reward_operators = (
            fit_cross_fitted_operator_map(
                model,
                records,
                args,
                task_name=task_name,
                variant=variant,
                cycle=cycle,
                seed=seed,
                permuted_scores=permutation,
            )
            if args.cross_fitted_rewards
            else operator
        )
        epoch_count = last_cycle_epochs if cycle == cycles - 1 else args.epochs_per_cycle
        for epoch in range(1, epoch_count + 1):
            loss = train_selector_epoch(
                model,
                reward_operators,
                records,
                optimizer,
                args,
                task_name=task_name,
                variant=variant,
                method=method,
                seed=seed,
                cycle=cycle,
                epoch=epoch,
                permuted_scores=permutation,
            )
            item = {
                "cycle": cycle + 1,
                "epoch": epoch,
                "selector_loss": loss,
                "operator_mode": operator.mode,
                "operator_hyperparameter": operator.selected_hyperparameter,
                "reward_cross_fitted": bool(args.cross_fitted_rewards),
            }
            if evaluate_records is not None:
                item["development"] = evaluate(
                    model,
                    operator,
                    full_operator,
                    evaluate_records,
                    args,
                    samples=args.development_samples,
                    seed=stable_seed(task_name, seed, cycle, epoch, "dev"),
                    threshold=None,
                    full_threshold=None,
                    include_controls=args.path_process,
                )
                key = development_selection_key(item["development"])
                snapshot = {
                        "selection_key": key,
                        "cycle": cycle + 1,
                        "epoch": epoch,
                        "selector_loss": loss,
                        "development": copy.deepcopy(item["development"]),
                        "gflownet_state": {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        },
                        "psychometric_operator": copy.deepcopy(operator),
                        "independent_full_evidence_operator": copy.deepcopy(full_operator),
                }
                if args.selection_mode == "mechanism_aligned":
                    snapshots.append(snapshot)
                elif best_snapshot is None or key > best_snapshot["selection_key"]:
                    best_snapshot = snapshot
            history.append(item)
            if evaluate_records is not None:
                metrics = item["development"]["posterior"]
                severity = item["development"]["severity"]["posterior"]
                print(
                    json.dumps(
                        {
                            "event": "alternating_psychometric_gfn_epoch",
                            "task": task_name,
                            "variant": variant,
                            "method": method,
                            "seed": seed,
                            "cycle": item["cycle"],
                            "epoch": item["epoch"],
                            "selector_loss": item["selector_loss"],
                            "dev_f1": metrics["weighted_f1"],
                            "dev_auroc": metrics.get("auroc"),
                            "dev_ccc": severity.get("ccc"),
                        }
                    ),
                    flush=True,
                )
    if operator is None:
        raise RuntimeError("schedule produced no operator")
    if evaluate_records is not None and args.selection_mode == "mechanism_aligned":
        best_f1 = max(
            float(snapshot["development"]["posterior"]["weighted_f1"])
            for snapshot in snapshots
        )
        eligible = [
            snapshot
            for snapshot in snapshots
            if float(snapshot["development"]["posterior"]["weighted_f1"])
            >= best_f1 - float(args.selection_f1_margin)
        ]
        ordered_groups = (
            sorted(eligible, key=lambda item: item["selection_key"], reverse=True),
            sorted(eligible, key=lambda item: float(item["selector_loss"])),
            sorted(eligible, key=lambda item: (item["cycle"], item["epoch"]), reverse=True),
        )
        shortlist = []
        seen = set()
        for rank in range(max(len(group) for group in ordered_groups)):
            for group in ordered_groups:
                if rank >= len(group):
                    continue
                snapshot = group[rank]
                signature = (snapshot["cycle"], snapshot["epoch"])
                if signature in seen:
                    continue
                seen.add(signature)
                shortlist.append(snapshot)
                if len(shortlist) >= int(args.mechanism_candidate_count):
                    break
            if len(shortlist) >= int(args.mechanism_candidate_count):
                break
        for candidate_index, snapshot in enumerate(shortlist):
            model.load_state_dict(snapshot["gflownet_state"], strict=True)
            snapshot["mechanism_development"] = development_mechanism_audit(
                model,
                snapshot["psychometric_operator"],
                evaluate_records,
                args,
                task_name=task_name,
                variant=variant,
                seed=stable_seed("mechanism_candidate", seed),
            )
            snapshot["mechanism_candidate_index"] = candidate_index
        best_snapshot = max(shortlist, key=_mechanism_selection_key)
        best_snapshot["selection_key"] = _mechanism_selection_key(best_snapshot)
        best_snapshot["selection_policy"] = {
            "name": "performance-constrained mechanism alignment",
            "best_development_f1": best_f1,
            "f1_noninferiority_margin": float(args.selection_f1_margin),
            "eligible_checkpoints": int(len(eligible)),
            "audited_shortlist": int(len(shortlist)),
            "test_used_for_selection": False,
            "candidates": [
                {
                    "cycle": int(candidate["cycle"]),
                    "epoch": int(candidate["epoch"]),
                    "selector_loss": float(candidate["selector_loss"]),
                    "weighted_f1": float(
                        candidate["development"]["posterior"]["weighted_f1"]
                    ),
                    "mechanism": copy.deepcopy(candidate["mechanism_development"]),
                    "selected": candidate is best_snapshot,
                }
                for candidate in shortlist
            ],
        }
    return operator, history, best_snapshot


def develop_seed(
    args: argparse.Namespace,
    task_name: str,
    seed: int,
    method: str,
    raw_records: dict[str, list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    projection_seed = stable_seed(task_name, seed, "event_projection")
    warm_start = None
    warm_start_sha256 = None
    if args.warm_start_checkpoint is not None:
        warm_start = load_trusted_checkpoint(args.warm_start_checkpoint, device)
        if warm_start.get("task") != task_name:
            raise RuntimeError("warm-start task does not match the requested task")
        if warm_start.get("method") != method:
            raise RuntimeError("warm-start method does not match the requested method")
        if int(warm_start.get("seed", -1)) != int(seed):
            raise RuntimeError("warm-start seed does not match the requested seed")
        projections = warm_start["event_projections"]
        warm_start_sha256 = sha256_file(args.warm_start_checkpoint)
    else:
        projections = fit_event_projections(raw_records["train"], projection_seed)
    train = prepare_records(raw_records["train"], projections, device)
    dev = prepare_records(raw_records["dev"], projections, device)
    seed_all(stable_seed(task_name, seed, "paired_model"))
    model = new_model(train[0], args, device, task_name=task_name)
    if warm_start is not None:
        model.load_state_dict(warm_start["gflownet_state"], strict=True)
    permutation = score_targets(train, args.variant, seed)
    full_operator = fit_full_evidence_operator(
        train,
        args,
        task_name=task_name,
        variant=args.variant,
        seed=seed,
        permuted_scores=permutation,
    )
    _, history, best_snapshot = run_schedule(
        model,
        train,
        full_operator,
        args,
        task_name=task_name,
        variant=args.variant,
        method=method,
        seed=seed,
        cycles=args.cycles,
        last_cycle_epochs=(
            args.last_cycle_epochs
            if args.last_cycle_epochs is not None
            else args.epochs_per_cycle
        ),
        evaluate_records=dev,
    )
    if best_snapshot is None:
        raise RuntimeError("development schedule produced no selectable checkpoint")
    checkpoint = args.output / "development_checkpoints" / task_name / method / f"seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "stage": "development_locked",
            "task": task_name,
            "variant": args.variant,
            "method": method,
            "seed": seed,
            "selected_cycle": int(best_snapshot["cycle"]),
            "selected_epoch": int(best_snapshot["epoch"]),
            "threshold": float(best_snapshot["development"]["threshold"]),
            "full_threshold": float(best_snapshot["development"]["full_threshold"]),
            "event_projections": projections,
            "psychometric_operator": best_snapshot["psychometric_operator"],
            "independent_full_evidence_operator": best_snapshot[
                "independent_full_evidence_operator"
            ],
            "gflownet_state": best_snapshot["gflownet_state"],
            "mechanism_development": best_snapshot.get("mechanism_development"),
            "selection_policy": best_snapshot.get("selection_policy"),
            "warm_start_checkpoint": (
                str(args.warm_start_checkpoint)
                if args.warm_start_checkpoint is not None
                else None
            ),
            "warm_start_checkpoint_sha256": warm_start_sha256,
            "config": vars(args),
        },
        checkpoint,
    )
    return {
        "seed": seed,
        "method": method,
        "selected_cycle": int(best_snapshot["cycle"]),
        "selected_epoch": int(best_snapshot["epoch"]),
        "threshold": float(best_snapshot["development"]["threshold"]),
        "full_threshold": float(best_snapshot["development"]["full_threshold"]),
        "development": best_snapshot["development"],
        "mechanism_development": best_snapshot.get("mechanism_development"),
        "selection_policy": best_snapshot.get("selection_policy"),
        "projection_seed": projection_seed,
        "warm_start_checkpoint": (
            str(args.warm_start_checkpoint)
            if args.warm_start_checkpoint is not None
            else None
        ),
        "warm_start_checkpoint_sha256": warm_start_sha256,
        "development_checkpoint": str(checkpoint),
        "development_checkpoint_sha256": sha256_file(checkpoint),
        "history": history,
    }


def locked_test_seed(
    args: argparse.Namespace,
    task_name: str,
    development: dict[str, Any],
    method: str,
    raw_records: dict[str, list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    seed = int(development["seed"])
    development_checkpoint = Path(development["development_checkpoint"])
    if sha256_file(development_checkpoint) != development["development_checkpoint_sha256"]:
        raise RuntimeError(f"development checkpoint hash mismatch: {development_checkpoint}")
    locked = load_trusted_checkpoint(development_checkpoint, device)
    projections = locked["event_projections"]
    test = prepare_records(raw_records["test"], projections, device)
    model = new_model(test[0], args, device, task_name=task_name)
    model.load_state_dict(locked["gflownet_state"], strict=True)
    operator = locked["psychometric_operator"]
    full_operator = locked["independent_full_evidence_operator"]
    test_result = evaluate(
        model,
        operator,
        full_operator,
        test,
        args,
        samples=args.test_samples,
        seed=stable_seed(task_name, seed, "test"),
        threshold=float(development["threshold"]),
        full_threshold=float(development["full_threshold"]),
        include_controls=True,
    )
    checkpoint = args.output / "checkpoints" / task_name / method / f"seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "task": task_name,
            "variant": args.variant,
            "method": method,
            "seed": seed,
            "selected_cycle": development["selected_cycle"],
            "selected_epoch": development["selected_epoch"],
            "threshold": development["threshold"],
            "full_threshold": development["full_threshold"],
            "development_checkpoint": str(development_checkpoint),
            "development_checkpoint_sha256": development["development_checkpoint_sha256"],
            "event_projections": projections,
            "psychometric_operator": operator,
            "independent_full_evidence_operator": full_operator,
            "gflownet_state": model.state_dict(),
            "mechanism_development": locked.get("mechanism_development"),
            "selection_policy": locked.get("selection_policy"),
            "config": vars(args),
        },
        checkpoint,
    )
    return {
        "seed": seed,
        "method": method,
        "selected_cycle": development["selected_cycle"],
        "selected_epoch": development["selected_epoch"],
        "development_checkpoint": str(development_checkpoint),
        "development_checkpoint_sha256": development["development_checkpoint_sha256"],
        "checkpoint": str(checkpoint),
        "development": development["development"],
        "mechanism_development": development.get("mechanism_development"),
        "selection_policy": development.get("selection_policy"),
        "test": test_result,
    }


def aggregate(runs: list[dict[str, Any]], method: str, branch: str) -> dict[str, Any]:
    metrics = [run["methods"][method]["test"][branch] for run in runs]
    result = {}
    for key in ("weighted_f1", "macro_f1", "positive_f1", "accuracy", "auroc", "brier"):
        values = [item[key] for item in metrics if item.get(key) is not None]
        result[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
    return result


def aggregate_section(runs: list[dict[str, Any]], method: str, section: str, branch: str | None = None) -> dict[str, Any]:
    blocks = [run["methods"][method]["test"][section] for run in runs]
    if branch is not None:
        blocks = [block[branch] for block in blocks]
    result = {}
    for key in blocks[0]:
        if isinstance(blocks[0][key], dict):
            continue
        values = [block[key] for block in blocks if block.get(key) is not None and np.isfinite(block[key])]
        result[key] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=0)) if values else None,
        }
    return result


def aggregate_controls(runs: list[dict[str, Any]], method: str) -> dict[str, Any]:
    blocks = [run["methods"][method]["test"]["controls"] for run in runs]
    result = {}
    for key in (
        "selected_delete_cost",
        "random_delete_cost",
        "specificity_gap",
        "relative_specificity_gap",
        "critical_step_delete_cost",
        "random_step_delete_cost",
        "critical_step_specificity_gap",
    ):
        values = [block[key] for block in blocks if key in block]
        if not values:
            continue
        result[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
    for control_name in ("evidence_shuffle", "path_forward", "path_reverse", "path_shuffle"):
        metrics = [block[control_name] for block in blocks if control_name in block]
        if not metrics:
            continue
        result[control_name] = {}
        for key in ("weighted_f1", "macro_f1", "positive_f1", "accuracy", "auroc", "brier"):
            values = [item[key] for item in metrics if item.get(key) is not None]
            result[control_name][key] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=0)) if values else None,
            }
    for control_name in ("path_forward_severity", "path_reverse_severity", "path_shuffle_severity"):
        metrics = [block[control_name] for block in blocks if control_name in block]
        if not metrics:
            continue
        result[control_name] = {}
        for key in metrics[0]:
            values = [item[key] for item in metrics if item.get(key) is not None]
            result[control_name][key] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=0)) if values else None,
            }
    return result


def run_task(args: argparse.Namespace, task_name: str, device: torch.device) -> dict[str, Any]:
    started = time.time()
    task = TASKS[task_name]
    raw_records = load_records(
        args.meter_root,
        args.contract_dir,
        task_name,
        allowed_splits=("train", "dev"),
        split_policy=args.split_policy,
    )
    attach_temporal(raw_records, args.meter_root, task["dataset"])
    if args.selector_method == "auto":
        methods = (
            ("gflownet", "reinforce")
            if args.variant == "full" and not args.gflownet_only
            else ("gflownet",)
        )
    elif args.selector_method == "both":
        methods = ("gflownet", "reinforce")
    else:
        methods = (args.selector_method,)
    development_runs = []
    for seed in args.seeds:
        method_results = {
            method: develop_seed(args, task_name, seed, method, raw_records, device)
            for method in methods
        }
        development_runs.append({"seed": seed, "methods": method_results})
        atomic_json(
            args.output / "development" / f"{task_name}.progress.json",
            {"schema": SCHEMA, "status": "development", "runs": development_runs},
        )
    atomic_json(
        args.output / "locks" / f"{task_name}.json",
        {
            "schema": SCHEMA,
            "status": "locked_before_test_read",
            "task": task_name,
            "variant": args.variant,
            "split_policy": args.split_policy,
            "official_splits": EXPECTED_SPLITS[task["dataset"]],
            "cache_signature": cache_signature(args.meter_root, task["dataset"]),
            "seeds": [
                {
                    "seed": run["seed"],
                    "methods": {
                        method: {
                            "selected_cycle": result["selected_cycle"],
                            "selected_epoch": result["selected_epoch"],
                            "threshold": result["threshold"],
                            "full_threshold": result["full_threshold"],
                            "development_checkpoint": result["development_checkpoint"],
                            "development_checkpoint_sha256": result[
                                "development_checkpoint_sha256"
                            ],
                        }
                        for method, result in run["methods"].items()
                    },
                }
                for run in development_runs
            ],
        },
    )
    if args.development_only:
        result = {
            "schema": SCHEMA,
            "status": "development_locked",
            "task": task_name,
            "variant": args.variant,
            "split_policy": args.split_policy,
            "dataset": task["dataset"],
            "official_splits": EXPECTED_SPLITS[task["dataset"]],
            "label_profile": task["binary_profile"],
            "score_profile": task["score_profile"],
            "cache_signature": cache_signature(args.meter_root, task["dataset"]),
            "test_opened_after_lock": False,
            "methods": list(methods),
            "development_runs": development_runs,
            "elapsed_s": time.time() - started,
        }
        atomic_json(args.output / "DEVELOPMENT_METRICS.json", result)
        return result
    raw_records["test"] = load_records(
        args.meter_root,
        args.contract_dir,
        task_name,
        allowed_splits=("test",),
        split_policy=args.split_policy,
    )["test"]
    attach_temporal({"test": raw_records["test"]}, args.meter_root, task["dataset"])
    runs = []
    for development in development_runs:
        method_results = {
            method: locked_test_seed(
                args, task_name, development["methods"][method], method, raw_records, device
            )
            for method in methods
        }
        runs.append({"seed": development["seed"], "methods": method_results})
        atomic_json(
            args.output / "tasks" / f"{task_name}.progress.json",
            {"schema": SCHEMA, "status": "locked_test", "runs": runs},
        )
    summary = {}
    for method in methods:
        summary[method] = {
            "posterior": aggregate(runs, method, "posterior"),
            "full_evidence": aggregate(runs, method, "full_evidence"),
            "severity": {
                "posterior": aggregate_section(runs, method, "severity", "posterior"),
                "full_evidence": aggregate_section(runs, method, "severity", "full_evidence"),
            },
            "uncertainty": aggregate_section(runs, method, "uncertainty"),
            "controls": aggregate_controls(runs, method),
        }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "task": task_name,
        "variant": args.variant,
        "split_policy": args.split_policy,
        "dataset": task["dataset"],
        "official_splits": EXPECTED_SPLITS[task["dataset"]],
        "label_profile": task["binary_profile"],
        "score_profile": task["score_profile"],
        "cache_signature": cache_signature(args.meter_root, task["dataset"]),
        "frozen_prediction_anchor": False,
        "historical_prediction_used": False,
        "psychometric_operator_refit_from_current_evidence": True,
        "final_prediction": "posterior marginalization over alternating psychometric GFlowNet evidence paths",
        "test_opened_after_lock": True,
        "methods": list(methods),
        "runs": runs,
        "summary": summary,
        "elapsed_s": time.time() - started,
    }
    atomic_json(args.output / "tasks" / f"{task_name}.json", result)
    atomic_json(args.output / "FINAL_METRICS.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meter-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument(
        "--split-policy",
        choices=("official", PSYPATH_SPLIT_POLICY),
        default="official",
    )
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27, 41, 53))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--epochs-per-cycle", type=int, default=6)
    parser.add_argument("--last-cycle-epochs", type=int)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--phenotype-count", type=int, default=4)
    parser.add_argument("--policy", choices=("legacy", "monotonic"), default="legacy")
    parser.add_argument("--cross-fitted-rewards", action="store_true")
    parser.add_argument("--reward-folds", type=int, default=4)
    parser.add_argument("--minimum-events", type=int, default=2)
    parser.add_argument("--maximum-events", type=int, default=16)
    parser.add_argument("--maximum-fraction", type=float, default=0.30)
    parser.add_argument("--operator-trajectories", type=int, default=4)
    parser.add_argument("--selector-trajectories", type=int, default=4)
    parser.add_argument("--exploration-epsilon", type=float, default=0.0)
    parser.add_argument("--development-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=48)
    parser.add_argument(
        "--selection-mode",
        choices=("performance", "mechanism_aligned"),
        default="performance",
    )
    parser.add_argument("--selection-f1-margin", type=float, default=0.02)
    parser.add_argument("--mechanism-development-samples", type=int, default=24)
    parser.add_argument("--mechanism-random-controls", type=int, default=8)
    parser.add_argument("--mechanism-candidate-count", type=int, default=6)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--measurement-components", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--relative-tb-weight", type=float, default=0.0)
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--pim-reward-weight", type=float, default=0.30)
    parser.add_argument("--ensr-reward-weight", type=float, default=0.20)
    parser.add_argument("--sufficiency-weight", type=float, default=1.0)
    parser.add_argument("--compactness-weight", type=float, default=0.05)
    parser.add_argument("--necessity-margin", type=float, default=0.05)
    parser.add_argument("--specificity-margin", type=float, default=0.02)
    parser.add_argument("--reward-temperature", type=float, default=0.50)
    parser.add_argument("--path-process", action="store_true")
    parser.add_argument("--path-process-weight", type=float, default=0.15)
    parser.add_argument("--path-monotonic-weight", type=float, default=0.10)
    parser.add_argument("--path-order-weight", type=float, default=0.15)
    parser.add_argument("--path-order-margin", type=float, default=0.02)
    parser.add_argument("--path-duplicate-weight", type=float, default=0.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--gflownet-only", action="store_true")
    parser.add_argument(
        "--selector-method",
        choices=("auto", "gflownet", "reinforce", "both"),
        default="auto",
    )
    args = parser.parse_args()

    configure_runtime(args.threads)
    validate_contract(args.contract_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output / "RUN_CONTRACT.json",
        {
            "schema": SCHEMA,
            "status": "running",
            "task": args.task,
            "variant": args.variant,
            "frozen_prediction_anchor": False,
            "historical_prediction_used": False,
            "test_opened_after_development_lock": True,
            "split_policy": args.split_policy,
            "config": vars(args),
        },
    )
    result = run_task(args, args.task, device)
    print(
        json.dumps(
            {
                "event": "alternating_psychometric_gfn_complete",
                "task": args.task,
                "status": result.get("status"),
                "summary": result.get("summary"),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
