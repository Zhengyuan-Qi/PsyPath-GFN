#!/usr/bin/env python3
"""Train bounded Answer-Dynamic/PIM/ENSR increments above online METER-PIM.

HumanOmniV2 and the exported psychometric operator remain frozen.  The only
trainable component estimates the expected answer from interview context and
models the sequential deviation of the observed response.  Development-set
non-inferiority selects either the learned increment or the exact zero-delta
METER-PIM anchor; the official Test split is evaluated only after that choice.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clinical_label_contract import load_label_profile, validate_contract
from scripts.meter_mh_online_operator import (
    BoundedSequentialIncrement,
    FrozenMeterPIM,
    OnlineSequentialPsychometricModel,
)
from scripts.psypath_split_policy import (
    SPLIT_POLICY as PSYPATH_SPLIT_POLICY,
    load_joint_partition,
    score_map_by_pid,
)


SCHEMA = "meter_mh_sequential_increment_ablation_v2"
TASKS = {
    "daic_depression": {
        "dataset": "daic",
        "binary_profile": "depression_phq8_screening",
        "score_profile": "depression_phq_score",
        "score_max": 24.0,
    },
    "edaic_depression": {
        "dataset": "edaic",
        "binary_profile": "depression_phq8_screening",
        "score_profile": "depression_phq_score",
        "score_max": 24.0,
    },
    "edaic_ptsd": {
        "dataset": "edaic",
        "binary_profile": "ptsd_dsm_iv_binary",
        "score_profile": "ptsd_score_repaired",
        "score_max": 85.0,
    },
}
EXPECTED_SPLITS = {
    "daic": {"train": 107, "dev": 35, "test": 47},
    "edaic": {"train": 163, "dev": 56, "test": 56},
}
VIEWS = ("clinical", "answer", "context", "visual")
VARIANTS = {
    "full": {
        "answer_dynamic": True,
        "pim_supervision": True,
        "ensr_supervision": True,
        "time_shuffle": False,
        "score_permutation": False,
        "definition": "complete Answer-Dynamic + PIM + ENSR increment",
    },
    "no_answer_dynamic": {
        "answer_dynamic": False,
        "pim_supervision": True,
        "ensr_supervision": True,
        "time_shuffle": False,
        "score_permutation": False,
        "definition": "replace each event view by its participant mean and remove observed-minus-expected deviations while preserving architecture and parameter count",
    },
    "no_pim": {
        "answer_dynamic": True,
        "pim_supervision": False,
        "ensr_supervision": True,
        "time_shuffle": False,
        "score_permutation": False,
        "definition": "retain the frozen PIM anchor but remove continuous severity supervision from the increment",
    },
    "no_ensr": {
        "answer_dynamic": True,
        "pim_supervision": True,
        "ensr_supervision": False,
        "time_shuffle": False,
        "score_permutation": False,
        "definition": "remove evidence necessity, sufficiency, and random-deletion specificity losses",
    },
    "time_shuffle": {
        "answer_dynamic": True,
        "pim_supervision": True,
        "ensr_supervision": True,
        "time_shuffle": True,
        "score_permutation": False,
        "definition": "apply a participant- and seed-locked permutation to the increment event sequence",
    },
    "score_permutation": {
        "answer_dynamic": True,
        "pim_supervision": True,
        "ensr_supervision": True,
        "time_shuffle": False,
        "score_permutation": True,
        "definition": "permute observed continuous severity targets within the training partition",
    },
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


def variant_contract(name: str) -> dict[str, Any]:
    if name not in VARIANTS:
        raise ValueError(f"unknown protocol variant: {name}")
    return VARIANTS[name]


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def event_order(
    record: dict[str, Any],
    variant: str,
    seed: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not variant_contract(variant)["time_shuffle"]:
        return None
    count = len(record["raw"]["answer"])
    permutation = np.random.default_rng(
        stable_seed("time_shuffle", variant, seed, record["pid"])
    ).permutation(count)
    return torch.as_tensor(permutation, dtype=torch.long, device=device)


def forward_variant(
    model: OnlineSequentialPsychometricModel,
    record: dict[str, Any],
    device: torch.device,
    variant: str,
    seed: int,
    *,
    intervention: str = "clean",
) -> dict[str, torch.Tensor]:
    spec = variant_contract(variant)
    return model(
        raw_tensors(record, device),
        answer_dynamic=bool(spec["answer_dynamic"]),
        order=event_order(record, variant, seed, device),
        intervention=intervention,
        intervention_seed=stable_seed("intervention", variant, seed, record["pid"], intervention),
    )


def permuted_score_targets(
    records: list[dict[str, Any]],
    variant: str,
    seed: int,
) -> dict[str, float]:
    if not variant_contract(variant)["score_permutation"]:
        return {}
    observed = [(str(record["pid"]), float(record["score"])) for record in records if record.get("score") is not None]
    values = np.asarray([score for _, score in observed], dtype=np.float64)
    shuffled = np.random.default_rng(
        stable_seed("score_permutation", variant, seed, len(records))
    ).permutation(values)
    return {pid: float(score) for (pid, _), score in zip(observed, shuffled)}


def normalize_pid(value: Any) -> str:
    return str(int(float(str(value).strip())))


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
    if isinstance(value, torch.device):
        return str(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(type(value).__name__)


def cache_signature(root: Path, dataset: str) -> str:
    payload = json.loads((root / dataset / "RUN_CONTRACT.json").read_text(encoding="utf-8"))
    signature = str(payload.get("run_signature", ""))
    if not signature:
        raise RuntimeError(f"missing cache signature for {dataset}")
    return signature


def score_map(contract_dir: Path, task: dict[str, Any]) -> dict[tuple[str, str], float | None]:
    frame, profile = load_label_profile(
        contract_dir, task["dataset"], task["score_profile"], include_unlabeled=True
    )
    result: dict[tuple[str, str], float | None] = {}
    for row in frame.to_dict("records"):
        value, mask = row.get(profile.target_column), row.get(profile.mask_column)
        try:
            score = float(value) if int(mask) == 1 and np.isfinite(float(value)) else None
        except (TypeError, ValueError):
            score = None
        result[(normalize_pid(row["pid"]), str(row["split"]))] = score
    return result


def load_records(
    cache_root: Path,
    contract_dir: Path,
    task_name: str,
    allowed_splits: tuple[str, ...] = ("train", "dev", "test"),
    *,
    split_policy: str = "official",
) -> dict[str, list[dict[str, Any]]]:
    task = TASKS[task_name]
    dataset = task["dataset"]
    signature = cache_signature(cache_root, dataset)
    if split_policy not in {"official", PSYPATH_SPLIT_POLICY}:
        raise ValueError(f"unknown split policy: {split_policy}")
    official_scores = score_map(contract_dir, task) if split_policy == "official" else None
    result: dict[str, list[dict[str, Any]]] = {}
    for split in allowed_splits:
        if split_policy == PSYPATH_SPLIT_POLICY:
            frame, profile, split_metadata = load_joint_partition(
                contract_dir,
                dataset,
                task["binary_profile"],
                split,
                include_unlabeled=False,
            )
            expected = int(split_metadata["rows"])
            scores = score_map_by_pid(
                contract_dir,
                dataset,
                task["score_profile"],
                split,
            )
        else:
            expected = EXPECTED_SPLITS[dataset][split]
            frame, profile = load_label_profile(
                contract_dir,
                dataset,
                task["binary_profile"],
                split=split,
                include_unlabeled=False,
            )
            scores = official_scores
        records = []
        for row in frame.to_dict("records"):
            source_split = str(row["split"])
            canonical_split = str(row.get("canonical_split", source_split))
            if split_policy == "official" and source_split != split:
                continue
            if split_policy == "official" and dataset == "edaic" and canonical_split != source_split:
                raise RuntimeError(
                    f"E-DAIC official/canonical split mismatch pid={row['pid']}: "
                    f"official={source_split} canonical={canonical_split}"
                )
            pid = normalize_pid(row["pid"])
            path = cache_root / dataset / "cache" / f"{pid}.npz"
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
                raw = {
                    name: np.nan_to_num(np.asarray(archive[name], dtype=np.float32))
                    for name in VIEWS
                }
            if metadata.get("run_signature") != signature:
                raise RuntimeError(f"cache signature mismatch: {path}")
            if len({len(value) for value in raw.values()}) != 1:
                raise ValueError(f"unaligned event views: {task_name}/{pid}")
            records.append(
                {
                    "pid": pid,
                    "split": split,
                    "source_split": source_split,
                    "canonical_split": canonical_split,
                    "label": int(row[profile.target_column]),
                    "score": (
                        scores.get(pid)
                        if split_policy == PSYPATH_SPLIT_POLICY
                        else scores.get((pid, split))
                    ),
                    "raw": raw,
                }
            )
        # Preserve the governed contract order.  Formal METER OOF teachers
        # were emitted in this exact participant order.
        if len(records) != expected:
            raise RuntimeError(f"{dataset} {split} count mismatch: {len(records)} != {expected}")
        result[split] = records
    return result


def raw_tensors(record: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(record["raw"][name], dtype=torch.float64, device=device)
        for name in VIEWS
    }


def metric_block(labels: Iterable[int], probability: Iterable[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=np.int64)
    probability = np.clip(np.asarray(list(probability), dtype=np.float64), 1e-6, 1.0 - 1e-6)
    prediction = (probability >= float(threshold)).astype(np.int64)
    result = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y, prediction, zero_division=0)),
        "brier": float(np.mean(np.square(probability - y))),
    }
    result["auroc"] = float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None
    return result


def choose_threshold(labels: list[int], probability: list[float]) -> float:
    choices = []
    for threshold in np.arange(0.30, 0.701, 0.025):
        metrics = metric_block(labels, probability, float(threshold))
        choices.append(
            (
                metrics["weighted_f1"],
                metrics["macro_f1"],
                -metrics["brier"],
                -abs(float(threshold) - 0.5),
                float(threshold),
            )
        )
    return float(max(choices)[-1])


def severity_block(records: list[dict[str, Any]], prediction: list[float]) -> dict[str, Any]:
    pairs = [
        (float(row["score"]), float(value))
        for row, value in zip(records, prediction)
        if row.get("score") is not None
    ]
    if len(pairs) < 2:
        return {"n": len(pairs), "ccc": None, "mae": None, "rmse": None}
    y = np.asarray([item[0] for item in pairs], dtype=np.float64)
    p = np.asarray([item[1] for item in pairs], dtype=np.float64)
    covariance = float(np.mean((y - y.mean()) * (p - p.mean())))
    denominator = float(y.var() + p.var() + (y.mean() - p.mean()) ** 2)
    ccc = (2.0 * covariance / denominator) if denominator > 1e-12 else None
    return {
        "n": int(len(y)),
        "ccc": ccc,
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
    }


def ensr_losses(
    model: OnlineSequentialPsychometricModel,
    raw: dict[str, torch.Tensor],
    clean_logit: torch.Tensor,
    label: torch.Tensor,
    random_seed: int,
    *,
    answer_dynamic: bool,
    order: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    projected = model.anchor.project_views(raw)
    selected = model.anchor.evidence_scorer.indices(
        torch.cat(
            (
                projected["clinical"],
                projected["answer"],
                projected["context"],
                projected["clinical"] - projected["answer"],
                projected["clinical"] - projected["context"],
            ),
            dim=-1,
        )
    )
    generator = torch.Generator(device=selected.device)
    generator.manual_seed(int(random_seed))
    random_indices = torch.sort(
        torch.randperm(len(projected["answer"]), generator=generator, device=selected.device)[: len(selected)]
    ).values
    selected_delete_keep = torch.ones(len(projected["answer"]), dtype=torch.bool, device=selected.device)
    selected_delete_keep[selected] = False
    selected_only_keep = ~selected_delete_keep
    random_delete_keep = torch.ones_like(selected_delete_keep)
    random_delete_keep[random_indices] = False

    def counterfactual_logit(keep: torch.Tensor) -> torch.Tensor:
        counterfactual_projected = model.anchor._apply_keep(projected, keep)
        features = model.anchor.meter_v7(counterfactual_projected, selected)
        anchor_logit = model.anchor.measure_features(features)["logit"].squeeze(0)
        dynamic_logit = model.increment(
            counterfactual_projected,
            answer_dynamic=answer_dynamic,
            order=order,
        )["risk_delta"]
        return anchor_logit + dynamic_logit

    selected_delete = counterfactual_logit(selected_delete_keep)
    selected_only = counterfactual_logit(selected_only_keep)
    random_delete = counterfactual_logit(random_delete_keep)

    direction = label.mul(2.0).sub(1.0)
    necessity_margin = direction * (clean_logit - selected_delete)
    random_margin = direction * (clean_logit - random_delete)
    necessity = F.relu(0.05 - necessity_margin)
    specificity = F.relu(random_margin - necessity_margin + 0.02)
    sufficiency = F.binary_cross_entropy_with_logits(selected_only, label)
    return {
        "necessity": necessity,
        "specificity": specificity,
        "sufficiency": sufficiency,
    }


def load_oof_teacher(
    teacher_root: Path | None,
    task_name: str,
    seed: int,
    expected: int,
) -> np.ndarray | None:
    if teacher_root is None:
        return None
    path = teacher_root / "tasks" / f"{task_name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    run = next(
        item for item in payload["result"]["oof_train"] if int(item["seed"]) == int(seed)
    )
    predictions = run["oof_predictions"]
    policy = payload["meter_mh"]["policy"]
    weights = policy["weights"]
    teacher = (
        float(weights["benchmark_weight"]) * np.asarray(predictions["benchmark_ta"], dtype=np.float64)
        + float(weights["answer_dynamic_weight"]) * np.asarray(predictions["answer_dynamic"], dtype=np.float64)
        + float(weights["candidate_weight"]) * np.asarray(predictions["meter_pim"], dtype=np.float64)
    )
    if len(teacher) != expected:
        raise RuntimeError(
            f"training OOF teacher length mismatch for {task_name}/seed={seed}: "
            f"{len(teacher)} != {expected}"
        )
    return np.clip(teacher, 1e-6, 1.0 - 1e-6)


@torch.no_grad()
def evaluate(
    model: OnlineSequentialPsychometricModel,
    records: list[dict[str, Any]],
    device: torch.device,
    variant: str,
    seed: int,
    threshold: float | None = None,
    anchor_threshold: float | None = None,
    intervention: str = "clean",
) -> dict[str, Any]:
    model.eval()
    labels, probability, anchor_probability, severity = [], [], [], []
    for record in records:
        output = forward_variant(
            model,
            record,
            device,
            variant,
            seed,
            intervention=intervention,
        )
        labels.append(int(record["label"]))
        probability.append(float(output["online_probability"].item()))
        anchor_probability.append(float(output["probability"].item()))
        severity.append(float(output["online_severity"].item()))
    if threshold is None:
        threshold = 0.5
    if anchor_threshold is None:
        anchor_threshold = 0.5
    return {
        "threshold": float(threshold),
        "variant": variant,
        "intervention": intervention,
        "model": metric_block(labels, probability, threshold),
        "anchor": metric_block(labels, anchor_probability, anchor_threshold),
        "severity": severity_block(records, severity),
        "probability": probability,
        "anchor_probability": anchor_probability,
        "severity_prediction": severity,
        "severity_targets": [
            float(record["score"]) if record.get("score") is not None else None
            for record in records
        ],
        "labels": labels,
        "pids": [str(record["pid"]) for record in records],
    }


def noninferior(candidate: dict[str, Any], anchor: dict[str, Any], margin: float) -> bool:
    if candidate["weighted_f1"] < anchor["weighted_f1"] - margin:
        return False
    if candidate.get("auroc") is not None and anchor.get("auroc") is not None:
        if candidate["auroc"] < anchor["auroc"] - margin:
            return False
    return candidate["brier"] <= anchor["brier"] + 0.015


def develop_seed(
    args: argparse.Namespace,
    task_name: str,
    seed: int,
    records: dict[str, list[dict[str, Any]]],
    operator_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    seed_all(seed)
    development_operator_path = operator_root / "operators_train" / task_name / f"seed_{seed}.pt"
    if not development_operator_path.is_file():
        raise FileNotFoundError(development_operator_path)
    payload = torch.load(development_operator_path, map_location=device, weights_only=False)
    anchor: FrozenMeterPIM = payload["module"].to(device).eval()
    event_dim = int(anchor.shared_projection.components.shape[0])
    increment = BoundedSequentialIncrement(event_dim=event_dim, hidden_dim=args.hidden_dim).to(
        device=device, dtype=torch.float64
    )
    model = OnlineSequentialPsychometricModel(anchor, increment).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    teacher_probability = load_oof_teacher(
        args.teacher_root, task_name, seed, len(records["train"])
    )
    variant = args.variant
    spec = variant_contract(variant)
    score_targets = permuted_score_targets(records["train"], variant, seed)

    initial_dev = evaluate(model, records["dev"], device, variant, seed)
    best_state = copy.deepcopy(increment.state_dict())
    best_epoch = 0
    best_key = (
        initial_dev["model"]["weighted_f1"],
        initial_dev["model"].get("auroc") or 0.0,
        -initial_dev["model"]["brier"],
    )
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch * 101).permutation(len(records["train"]))
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for step, index in enumerate(order, 1):
            record = records["train"][int(index)]
            raw = raw_tensors(record, device)
            dynamic_order = event_order(record, variant, seed, device)
            output = model(
                raw,
                answer_dynamic=bool(spec["answer_dynamic"]),
                order=dynamic_order,
            )
            label = torch.tensor(float(record["label"]), dtype=torch.float64, device=device)
            classification = F.binary_cross_entropy_with_logits(output["online_logit"], label)
            anchor_target = output["probability"].detach().squeeze(0)
            loss = classification + args.anchor_weight * F.binary_cross_entropy_with_logits(
                output["online_logit"], anchor_target
            )
            if teacher_probability is not None:
                teacher = torch.tensor(
                    float(teacher_probability[int(index)]), dtype=torch.float64, device=device
                )
                loss = loss + args.teacher_weight * F.binary_cross_entropy_with_logits(
                    output["online_logit"], teacher
                )
            score = score_targets.get(str(record["pid"]), record.get("score"))
            if spec["pim_supervision"] and score is not None:
                score_tensor = torch.tensor(float(score), dtype=torch.float64, device=device)
                severity = F.smooth_l1_loss(
                    output["online_severity"] / TASKS[task_name]["score_max"],
                    score_tensor / TASKS[task_name]["score_max"],
                )
                loss = loss + args.pim_weight * severity
            if int(record["label"]) == 0:
                # Learn the context-conditioned normative response manifold
                # only from screening-negative training participants.
                normative = F.smooth_l1_loss(
                    output["expected_answer"], output["answer_target"]
                )
                loss = loss + args.normative_weight * normative
            if spec["ensr_supervision"]:
                cf = ensr_losses(
                    model,
                    raw,
                    output["online_logit"],
                    label,
                    seed * 100000 + epoch * 1000 + step,
                    answer_dynamic=bool(spec["answer_dynamic"]),
                    order=dynamic_order,
                )
                loss = loss + args.ensr_weight * (
                    cf["necessity"] + cf["specificity"] + args.sufficiency_weight * cf["sufficiency"]
                )
            loss = loss / args.gradient_accumulation
            loss.backward()
            epoch_loss += float(loss.detach().cpu()) * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(increment.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        dev = evaluate(model, records["dev"], device, variant, seed)
        key = (
            dev["model"]["weighted_f1"],
            dev["model"].get("auroc") or 0.0,
            -dev["model"]["brier"],
        )
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / max(1, len(order)),
                "dev_model": dev["model"],
                "dev_anchor": dev["anchor"],
            }
        )
        print(
            json.dumps(
                {
                    "event": "cispm_development_epoch",
                    "task": task_name,
                    "variant": variant,
                    "seed": seed,
                    "epoch": epoch,
                    "loss": history[-1]["loss"],
                    "weighted_f1": dev["model"]["weighted_f1"],
                    "anchor_weighted_f1": dev["anchor"]["weighted_f1"],
                    "auroc": dev["model"].get("auroc"),
                }
            ),
            flush=True,
        )
        if key > best_key and noninferior(dev["model"], dev["anchor"], args.noninferiority_margin):
            best_key = key
            best_state = copy.deepcopy(increment.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break

    increment.load_state_dict(best_state)
    final_dev = evaluate(model, records["dev"], device, variant, seed)
    admitted = best_epoch > 0 and noninferior(
        final_dev["model"], final_dev["anchor"], args.noninferiority_margin
    )
    if not admitted:
        increment.risk_scale.data.zero_()
        increment.severity_scale.data.zero_()
        selected = "zero_increment_anchor"
        final_dev = evaluate(model, records["dev"], device, variant, seed)
    else:
        selected = "bounded_sequential_increment"
    locked_threshold = 0.5
    selected_state = copy.deepcopy(increment.state_dict())
    output_path = args.output / "development_checkpoints" / task_name / f"seed_{seed}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "task": task_name,
            "variant": variant,
            "variant_contract": spec,
            "seed": seed,
            "selected": selected,
            "threshold": locked_threshold,
            "development_operator": str(development_operator_path),
            "development_increment_state": selected_state,
            "config": {key: json_default(value) if isinstance(value, (Path, torch.device)) else value for key, value in vars(args).items()},
        },
        output_path,
    )
    print(
        json.dumps(
            {
                "event": "cispm_seed_development_locked",
                "task": task_name,
                "variant": variant,
                "seed": seed,
                "selected": selected,
                "best_epoch": best_epoch,
            }
        ),
        flush=True,
    )
    return {
        "seed": seed,
        "variant": variant,
        "selected": selected,
        "best_epoch": best_epoch,
        "checkpoint": str(output_path),
        "initial_dev": initial_dev,
        "final_dev": final_dev,
        "history": history,
    }


def locked_test_seed(
    args: argparse.Namespace,
    task_name: str,
    development: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    operator_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    seed = int(development["seed"])
    selected = str(development["selected"])
    best_epoch = int(development["best_epoch"])
    variant = args.variant
    spec = variant_contract(variant)
    locked_operator_path = operator_root / "operators" / task_name / f"seed_{seed}.pt"
    locked_payload = torch.load(locked_operator_path, map_location=device, weights_only=False)
    locked_anchor: FrozenMeterPIM = locked_payload["module"].to(device).eval()
    seed_all(seed + 900000)
    locked_increment = BoundedSequentialIncrement(
        event_dim=int(locked_anchor.shared_projection.components.shape[0]),
        hidden_dim=args.hidden_dim,
    ).to(device=device, dtype=torch.float64)
    locked_model = OnlineSequentialPsychometricModel(locked_anchor, locked_increment).to(device)
    if selected == "bounded_sequential_increment":
        optimizer = torch.optim.AdamW(
            [parameter for parameter in locked_model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        fit_records = records["train"] + records["dev"]
        score_targets = permuted_score_targets(fit_records, variant, seed)
        for epoch in range(1, best_epoch + 1):
            locked_model.train()
            order = np.random.default_rng(seed + 500000 + epoch * 101).permutation(len(fit_records))
            optimizer.zero_grad(set_to_none=True)
            for step, index in enumerate(order, 1):
                record = fit_records[int(index)]
                raw = raw_tensors(record, device)
                dynamic_order = event_order(record, variant, seed, device)
                output = locked_model(
                    raw,
                    answer_dynamic=bool(spec["answer_dynamic"]),
                    order=dynamic_order,
                )
                label = torch.tensor(float(record["label"]), dtype=torch.float64, device=device)
                anchor_target = output["probability"].detach().squeeze(0)
                loss = F.binary_cross_entropy_with_logits(output["online_logit"], label)
                loss = loss + args.anchor_weight * F.binary_cross_entropy_with_logits(
                    output["online_logit"], anchor_target
                )
                score = score_targets.get(str(record["pid"]), record.get("score"))
                if spec["pim_supervision"] and score is not None:
                    score_tensor = torch.tensor(float(score), dtype=torch.float64, device=device)
                    loss = loss + args.pim_weight * F.smooth_l1_loss(
                        output["online_severity"] / TASKS[task_name]["score_max"],
                        score_tensor / TASKS[task_name]["score_max"],
                    )
                if int(record["label"]) == 0:
                    loss = loss + args.normative_weight * F.smooth_l1_loss(
                        output["expected_answer"], output["answer_target"]
                    )
                if spec["ensr_supervision"]:
                    cf = ensr_losses(
                        locked_model,
                        raw,
                        output["online_logit"],
                        label,
                        seed * 1000000 + epoch * 1000 + step,
                        answer_dynamic=bool(spec["answer_dynamic"]),
                        order=dynamic_order,
                    )
                    loss = loss + args.ensr_weight * (
                        cf["necessity"] + cf["specificity"] + args.sufficiency_weight * cf["sufficiency"]
                    )
                loss = loss / args.gradient_accumulation
                loss.backward()
                if step % args.gradient_accumulation == 0 or step == len(order):
                    torch.nn.utils.clip_grad_norm_(locked_increment.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
    else:
        locked_increment.risk_scale.data.zero_()
        locked_increment.severity_scale.data.zero_()
    test = evaluate(
        locked_model,
        records["test"],
        device,
        variant,
        seed,
        threshold=0.5,
        anchor_threshold=0.5,
    )
    controls = {
        intervention: evaluate(
            locked_model,
            records["test"],
            device,
            variant,
            seed,
            threshold=0.5,
            anchor_threshold=0.5,
            intervention=intervention,
        )
        for intervention in ("selected_delete", "random_delete", "selected_only")
    }
    output_path = args.output / "checkpoints" / task_name / f"seed_{seed}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "task": task_name,
            "variant": variant,
            "variant_contract": spec,
            "seed": seed,
            "selected": selected,
            "best_epoch": best_epoch,
            "threshold": 0.5,
            "locked_operator": str(locked_operator_path),
            "locked_increment_state": locked_increment.state_dict(),
        },
        output_path,
    )
    print(
        json.dumps(
            {
                "event": "cispm_locked_test_complete",
                "task": task_name,
                "seed": seed,
                "selected": selected,
                "weighted_f1": test["model"]["weighted_f1"],
                "anchor_weighted_f1": test["anchor"]["weighted_f1"],
            }
        ),
        flush=True,
    )
    return {
        "seed": seed,
        "variant": variant,
        "selected": selected,
        "best_epoch": best_epoch,
        "checkpoint": str(output_path),
        "locked_test": test,
        "counterfactual_test": controls,
    }


def aggregate(runs: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    metrics = [run["locked_test"][branch] for run in runs]
    result = {}
    for key in ("weighted_f1", "macro_f1", "positive_f1", "accuracy", "auroc", "brier"):
        values = np.asarray([item[key] for item in metrics if item.get(key) is not None], dtype=np.float64)
        result[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return result


def aggregate_counterfactual(
    runs: list[dict[str, Any]], intervention: str, branch: str
) -> dict[str, Any]:
    metrics = [run["counterfactual_test"][intervention][branch] for run in runs]
    result = {}
    for key in ("weighted_f1", "macro_f1", "positive_f1", "accuracy", "auroc", "brier"):
        values = np.asarray(
            [item[key] for item in metrics if item.get(key) is not None], dtype=np.float64
        )
        result[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return result


def run_task(args: argparse.Namespace, task_name: str, device: torch.device) -> dict[str, Any]:
    started = time.time()
    records = load_records(
        args.meter_root, args.contract_dir, task_name, allowed_splits=("train", "dev")
    )
    development_runs = []
    for seed in args.seeds:
        development_runs.append(
            develop_seed(args, task_name, seed, records, args.operator_root, device)
        )
        atomic_json(
            args.output / "tasks" / f"{task_name}.progress.json",
            {"schema": SCHEMA, "status": "development", "runs": development_runs},
        )
    atomic_json(
        args.output / "locks" / f"{task_name}.development_locked.json",
        {
            "schema": SCHEMA,
            "status": "locked_before_test_read",
            "task": task_name,
            "threshold": 0.5,
            "seeds": [
                {
                    "seed": run["seed"],
                    "selected": run["selected"],
                    "best_epoch": run["best_epoch"],
                }
                for run in development_runs
            ],
        },
    )
    records["test"] = load_records(
        args.meter_root, args.contract_dir, task_name, allowed_splits=("test",)
    )["test"]
    runs = [
        locked_test_seed(args, task_name, development, records, args.operator_root, device)
        for development in development_runs
    ]
    result = {
        "status": "complete",
        "task": task_name,
        "variant": args.variant,
        "variant_contract": variant_contract(args.variant),
        "dataset": TASKS[task_name]["dataset"],
        "official_splits": EXPECTED_SPLITS[TASKS[task_name]["dataset"]],
        "runs": runs,
        "development_runs": development_runs,
        "selected_counts": {
            name: sum(run["selected"] == name for run in runs)
            for name in ("bounded_sequential_increment", "zero_increment_anchor")
        },
        "test_summary": {
            "model": aggregate(runs, "model"),
            "anchor": aggregate(runs, "anchor"),
        },
        "counterfactual_test_summary": {
            intervention: {
                "model": aggregate_counterfactual(runs, intervention, "model"),
                "anchor": aggregate_counterfactual(runs, intervention, "anchor"),
            }
            for intervention in ("selected_delete", "random_delete", "selected_only")
        },
    }
    atomic_json(args.output / "tasks" / f"{task_name}.json", result)
    print(
        json.dumps(
            {
                "event": "cispm_task_complete",
                "task": task_name,
                "variant": args.variant,
                "elapsed_s": time.time() - started,
                "model_weighted_f1": result["test_summary"]["model"]["weighted_f1"],
                "anchor_weighted_f1": result["test_summary"]["anchor"]["weighted_f1"],
            }
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meter-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--operator-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASKS), action="append")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27, 41, 53))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--pim-weight", type=float, default=0.25)
    parser.add_argument("--teacher-weight", type=float, default=0.50)
    parser.add_argument("--anchor-weight", type=float, default=0.50)
    parser.add_argument("--normative-weight", type=float, default=0.05)
    parser.add_argument("--ensr-weight", type=float, default=0.10)
    parser.add_argument("--sufficiency-weight", type=float, default=0.25)
    parser.add_argument("--noninferiority-margin", type=float, default=0.0)
    args = parser.parse_args()

    configure_runtime(args.threads)
    validate_contract(args.contract_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = tuple(args.task or TASKS)
    final = {
        "schema": SCHEMA,
        "status": "running",
        "variant": args.variant,
        "variant_contract": variant_contract(args.variant),
        "model": "METER-MH online sequential psychometric model",
        "humanomniv2_training": False,
        "frozen_anchor": "online differentiable METER-PIM",
        "trainable_increment": "context-expected answer deviation + Answer-Dynamic + ENSR",
        "tasks": {},
    }
    atomic_json(args.output / "PROGRESS.json", final)
    for task_name in tasks:
        final["tasks"][task_name] = run_task(args, task_name, device)
        atomic_json(args.output / "PROGRESS.json", final)
    final["status"] = "complete"
    atomic_json(args.output / "FINAL_METRICS.json", final)
    print(json.dumps({"event": "meter_sequential_increment_complete", "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
