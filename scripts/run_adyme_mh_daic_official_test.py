#!/usr/bin/env python3
"""Strict DAIC-WOZ official train/dev/test evaluation for A-DyME-MH.

This is a separate protocol from the joint DAIC/E-DAIC canonical evaluation.
It uses the raw official DAIC split field from the audited label contract:
official train -> OOF policy lock, official dev -> refit, official test ->
final retrospective evaluation. It never edits raw labels or caches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.clinical_label_contract import load_label_profile, validate_contract
from scripts.run_adyme_mh_final_gate_v2 import (
    ANCHOR_NAMES,
    BOOTSTRAP_SAMPLES,
    CONTROL_NAMES,
    F1_MARGIN,
    MIN_NONNEGATIVE_SEEDS,
    SEEDS,
    VARIANT_NAMES,
    aggregate_metric_blocks,
    atomic_json,
    cache_signature_audit,
    classification_metrics,
    configure_threads,
    finite,
    fit_all_variants,
    load_meld_train,
    mix_predictions,
    noninferior_to_both,
    paired_bootstrap_delta,
    safe_rule,
    score_array,
    severity_metrics,
)


SCHEMA = "adyme_mh_daic_official_test_v1"


def normalize_pid(value: Any) -> str:
    text = str(value).strip()
    return str(int(float(text)))


def load_records_official(
    split: str,
    adyme_root: Path,
    benchmark_root: Path,
    contract_dir: Path,
) -> list[dict[str, Any]]:
    """Load DAIC records by raw official split, never canonical_split."""

    frame, binary_spec = load_label_profile(
        contract_dir,
        "daic",
        "depression_phq8_screening",
        include_unlabeled=False,
    )
    score_frame, score_spec = load_label_profile(
        contract_dir,
        "daic",
        "depression_phq_score",
        include_unlabeled=True,
    )
    scores: dict[str, float | None] = {}
    for row in score_frame.to_dict("records"):
        pid = normalize_pid(row["pid"])
        value, mask = row.get(score_spec.target_column), row.get(score_spec.mask_column)
        try:
            scores[pid] = (
                float(value)
                if int(mask) == 1 and value is not None and np.isfinite(float(value))
                else None
            )
        except (TypeError, ValueError):
            scores[pid] = None

    contract_path = adyme_root / "daic" / "RUN_CONTRACT.json"
    expected_signature = None
    if contract_path.is_file():
        expected_signature = json.loads(contract_path.read_text(encoding="utf-8")).get("run_signature")

    records: list[dict[str, Any]] = []
    excluded_signatures: dict[str, int] = {}
    for row in frame.to_dict("records"):
        if str(row["split"]) != split:
            continue
        pid = normalize_pid(row["pid"])
        adyme_path = adyme_root / "daic" / "cache" / f"{pid}.npz"
        benchmark_path = benchmark_root / "daic" / "depression" / "text_audio" / f"{pid}_all.npy"
        if not adyme_path.is_file() or not benchmark_path.is_file():
            raise FileNotFoundError(f"missing official DAIC cache for PID {pid}")
        with np.load(adyme_path, allow_pickle=False) as archive:
            raw = {name: finite(archive[name]) for name in ("answer", "joint", "context", "visual")}
            metadata = json.loads(str(archive["metadata_json"].item()))
        actual_signature = metadata.get("run_signature")
        if expected_signature and actual_signature != expected_signature:
            key = str(actual_signature or "MISSING")
            excluded_signatures[key] = excluded_signatures.get(key, 0) + 1
            continue
        benchmark = finite(np.load(benchmark_path, mmap_mode="r"))
        if any(value.ndim != 2 or len(value) == 0 for value in raw.values()):
            raise ValueError(f"invalid official DAIC shapes for PID {pid}")
        records.append(
            {
                "pid": pid,
                "split": split,
                "label": int(row[binary_spec.target_column]),
                "score": scores.get(pid),
                "benchmark": benchmark,
                "raw": raw,
                "metadata": metadata,
                "alignment": {
                    "status": "participant_level_only",
                    "official_split_field": True,
                    "position_matching_used": False,
                    "padding_or_truncation_used": False,
                    "adyme_event_count": int(len(raw["answer"])),
                    "benchmark_event_count": int(len(benchmark)),
                },
            }
        )
    if not records:
        raise RuntimeError(f"official DAIC split is empty: {split}")
    return records


def lock_policies(train: list[dict[str, Any]], meld_train: np.ndarray, task_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Perform the training-only nested OOF policy lock."""

    y_train = np.asarray([record["label"] for record in train], dtype=np.int64)
    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        oof = {name: np.zeros(len(train), dtype=np.float64) for name in VARIANT_NAMES}
        thresholds = {name: np.full(len(train), 0.5, dtype=np.float64) for name in VARIANT_NAMES}
        severity_oof: dict[str, np.ndarray] = {}
        fold_rules: list[dict[str, Any]] = []
        protected_oof = {name: np.zeros(len(train), dtype=np.float64) for name in VARIANT_NAMES if name not in ANCHOR_NAMES}
        protected_thresholds = {name: np.full(len(train), 0.5, dtype=np.float64) for name in protected_oof}
        for fold, (fit_idx, held_idx) in enumerate(
            __import__("sklearn.model_selection", fromlist=["StratifiedKFold"]).StratifiedKFold(
                5, shuffle=True, random_state=seed
            ).split(np.zeros(len(train)), y_train)
        ):
            fit_rows = [train[int(i)] for i in fit_idx]
            held_rows = [train[int(i)] for i in held_idx]
            rule = safe_rule(fit_rows, meld_train, seed + fold * 1009)
            predictions, configs = fit_all_variants(fit_rows, held_rows, meld_train, None, seed + fold * 2003)
            for name, probability in predictions.items():
                oof[name][held_idx] = probability
                thresholds[name][held_idx] = configs[name]["threshold"]
            for name, value in configs.get("severity_predictions", {}).items():
                severity_oof.setdefault(name, np.zeros(len(train), dtype=np.float64))[held_idx] = value
            for candidate, details in rule["candidate_details"].items():
                selected = details["selected"]
                protected_oof[candidate][held_idx] = mix_predictions(
                    predictions["benchmark_ta"], predictions["answer_dynamic"], predictions[candidate], selected["weights"]
                )
                protected_thresholds[candidate][held_idx] = selected["threshold"]
            fold_rules.append(rule)
        metrics = {name: classification_metrics(y_train, oof[name], float(np.median(thresholds[name]))) for name in oof}
        anchor_metrics = {name: metrics[name] for name in ANCHOR_NAMES}
        protected_metrics = {
            candidate: classification_metrics(y_train, values, float(np.median(protected_thresholds[candidate])))
            for candidate, values in protected_oof.items()
        }
        seed_support = {}
        for candidate, candidate_metric in protected_metrics.items():
            admitted, checks = noninferior_to_both(candidate_metric, anchor_metrics)
            positive_mass = any(
                rule["candidate_details"].get(candidate, {}).get("selected", {}).get("weights", {}).get("candidate_weight", 0.0) > 0.0
                for rule in fold_rules
            )
            seed_support[candidate] = {"admitted": bool(admitted and positive_mass), "checks": checks, "positive_mass": positive_mass}
        run = {
            "seed": seed,
            "oof_metrics": metrics,
            "protected_metrics": protected_metrics,
            "oof_predictions": {name: values.tolist() for name, values in oof.items()},
            "oof_thresholds": {name: float(np.median(values)) for name, values in thresholds.items()},
            "severity_oof_metrics": {
                name: severity_metrics(
                    score_array(train)[np.isfinite(score_array(train))],
                    values[np.isfinite(score_array(train))],
                )
                for name, values in severity_oof.items()
            },
            "fold_rules": fold_rules,
            "seed_support": seed_support,
        }
        runs.append(run)
        atomic_json(task_dir / f"seed_{seed}.json", run)

    final_policies: dict[str, Any] = {}
    for candidate in VARIANT_NAMES:
        if candidate in ANCHOR_NAMES:
            continue
        supported = [run for run in runs if run["seed_support"].get(candidate, {}).get("admitted")]
        if len(supported) < MIN_NONNEGATIVE_SEEDS:
            final_policies[candidate] = {
                "status": "fallback_benchmark_insufficient_seed_support",
                "supporting_seeds": len(supported),
                "required_seeds": MIN_NONNEGATIVE_SEEDS,
                "weights": {"benchmark_weight": 1.0, "answer_dynamic_weight": 0.0, "candidate_weight": 0.0},
                "threshold": float(np.median([run["oof_metrics"]["benchmark_ta"]["threshold"] for run in runs])),
            }
            continue
        selected_items = []
        for run in supported:
            for fold_rule in run["fold_rules"]:
                detail = fold_rule["candidate_details"].get(candidate)
                if detail and detail["status"] == "dual_anchor_admitted_within_seed":
                    selected_items.append(detail["selected"])
        weights = {key: float(np.median([item["weights"][key] for item in selected_items])) for key in ("benchmark_weight", "answer_dynamic_weight", "candidate_weight")}
        total = sum(weights.values())
        final_policies[candidate] = {
            "status": "admitted_after_seed_stability",
            "supporting_seeds": len(supported),
            "required_seeds": MIN_NONNEGATIVE_SEEDS,
            "weights": {key: value / total for key, value in weights.items()},
            "threshold": float(np.median([item["threshold"] for item in selected_items])),
        }

    for candidate, policy in list(final_policies.items()):
        if policy["status"] != "admitted_after_seed_stability":
            continue
        policy_checks = []
        policy_support = 0
        for run in runs:
            oof = {name: np.asarray(values, dtype=np.float64) for name, values in run["oof_predictions"].items()}
            probability = mix_predictions(oof["benchmark_ta"], oof["answer_dynamic"], oof[candidate], policy["weights"])
            metrics = classification_metrics(y_train, probability, policy["threshold"])
            admissible, checks = noninferior_to_both(metrics, {name: run["oof_metrics"][name] for name in ANCHOR_NAMES})
            policy_checks.append({"seed": run["seed"], "admissible": admissible, "checks": checks, "metrics": metrics})
            policy_support += int(admissible)
        if policy_support < MIN_NONNEGATIVE_SEEDS:
            final_policies[candidate] = {
                "status": "fallback_benchmark_final_policy_not_represented",
                "supporting_seeds": policy_support,
                "required_seeds": MIN_NONNEGATIVE_SEEDS,
                "weights": {"benchmark_weight": 1.0, "answer_dynamic_weight": 0.0, "candidate_weight": 0.0},
                "threshold": float(np.median([run["oof_metrics"]["benchmark_ta"]["threshold"] for run in runs])),
            }
        else:
            policy["exact_seed_audit"] = policy_checks
    anchor_policy = __import__("scripts.run_adyme_mh_final_gate_v2", fromlist=["choose_anchor_only_policy"]).choose_anchor_only_policy(
        runs[0]["oof_predictions"]["benchmark_ta"], runs[0]["oof_predictions"]["answer_dynamic"], y_train,
        {name: runs[0]["oof_metrics"][name]["threshold"] for name in ANCHOR_NAMES},
    )
    return runs, final_policies, anchor_policy


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    train = load_records_official("train", args.adyme_root, args.benchmark_root, args.contract_dir)
    dev = load_records_official("dev", args.adyme_root, args.benchmark_root, args.contract_dir)
    test = load_records_official("test", args.adyme_root, args.benchmark_root, args.contract_dir)
    task_dir = args.output / "daic" / "depression"
    task_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(task_dir / "LOCKED_PROTOCOL.json", {
        "schema": SCHEMA,
        "dataset": "daic",
        "target": "depression",
        "label_profile": "depression_phq8_screening",
        "split_unit": "participant",
        "canonical_split": False,
        "official_split": True,
        "raw_split_source": "clinical_label_contract_v1 daic_labels.csv split field",
        "cohort_counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "seeds": list(SEEDS),
        "test_opened": False,
        "anchors": list(ANCHOR_NAMES),
        "dual_anchor_noninferiority": {"weighted_f1_margin": F1_MARGIN, "auroc_margin": 0.01, "brier_margin": 0.015, "min_supporting_seeds": MIN_NONNEGATIVE_SEEDS},
        "test_selection_forbidden": True,
        "raw_inputs_read_only": True,
    })
    atomic_json(task_dir / "LOCK_COMPLETE.json", {"status": "official_train_dev_policy_lock_complete", "schema": SCHEMA})
    meld_train = load_meld_train(args.adyme_root)
    runs, final_policies, anchor_policy = lock_policies(train, meld_train, task_dir)
    atomic_json(task_dir / "POLICY_LOCKED.json", {"status": "locked_before_official_test_read", "final_policies": final_policies, "anchor_only_policy": anchor_policy})

    fit_rows = train + dev
    y_test = np.asarray([record["label"] for record in test], dtype=np.int64)
    locked_runs = []
    for seed in SEEDS:
        predictions, configs = fit_all_variants(fit_rows, test, meld_train, None, seed + 180000)
        raw = {name: classification_metrics(y_test, probability, configs[name]["threshold"]) for name, probability in predictions.items()}
        scores = score_array(test)
        mask = np.isfinite(scores)
        raw_regression = {
            name: severity_metrics(scores[mask], values[mask])
            for name, values in configs.get("severity_predictions", {}).items()
            if int(mask.sum()) >= 2
        }
        protected = {}
        bootstrap = {}
        for index, (candidate, policy) in enumerate(final_policies.items()):
            probability = mix_predictions(predictions["benchmark_ta"], predictions["answer_dynamic"], predictions[candidate], policy["weights"])
            protected[candidate] = classification_metrics(y_test, probability, policy["threshold"])
            bootstrap[candidate] = {anchor: paired_bootstrap_delta(y_test, probability, policy["threshold"], predictions[anchor], configs[anchor]["threshold"], seed * 1000 + index) for anchor in ANCHOR_NAMES}
        controls = {}
        for control in CONTROL_NAMES:
            control_predictions, control_configs = fit_all_variants(fit_rows, test, meld_train, control, seed + 180000, held_control_only=True)
            controls[control] = {name: classification_metrics(y_test, probability, control_configs[name]["threshold"]) for name, probability in control_predictions.items()}
        locked_runs.append({"seed": seed, "raw": raw, "raw_regression": raw_regression, "protected": protected, "paired_bootstrap": bootstrap, "controls": controls, "policies": final_policies})
    return {
        "inventory": {"train": len(train), "dev": len(dev), "test": len(test)},
        "cache_signature_audit": cache_signature_audit(args.adyme_root, ("daic",)),
        "oof_train": runs,
        "final_policies": final_policies,
        "anchor_only_policy": anchor_policy,
        "locked": locked_runs,
        "locked_summary": {
            "raw": aggregate_metric_blocks([item["raw"] for item in locked_runs]),
            "protected": aggregate_metric_blocks([item["protected"] for item in locked_runs]),
            "raw_regression": aggregate_metric_blocks([item["raw_regression"] for item in locked_runs]),
            "controls": {control: aggregate_metric_blocks([item["controls"][control] for item in locked_runs]) for control in CONTROL_NAMES},
        },
        "protocol": {
            "schema": SCHEMA,
            "split_unit": "participant",
            "canonical_split": False,
            "official_split": True,
            "train_dev_policy_lock_before_test": True,
            "test_opened_after_policy_lock": True,
            "seeds": list(SEEDS),
            "outer_folds": 5,
            "dual_anchor_noninferiority": True,
            "anchor_names": list(ANCHOR_NAMES),
            "raw_inputs_read_only": True,
            "MELD_labels_used": False,
            "test_selection_forbidden": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adyme-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    configure_threads(args.threads)
    validate_contract(args.contract_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "status": "running"}
    atomic_json(args.output / "PROGRESS.json", payload)
    payload["dataset"] = evaluate(args)
    payload["status"] = "complete"
    atomic_json(args.output / "FINAL_METRICS.json", payload)
    print(json.dumps({"event": "daic_official_test_complete", "output": str(args.output / "FINAL_METRICS.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
