#!/usr/bin/env python3
"""Strict three-task METER-MH validation on the frozen v1 cache.

The runner keeps one task-conditioned model family across DAIC depression,
E-DAIC depression, and E-DAIC PTSD.  Candidate measurement paths are selected
only from participant-level training OOF predictions.  DAIC uses its official
107/35/47 split; E-DAIC uses the audited canonical train/dev/test split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_adyme_mh_daic_official_test as daic_official
from scripts import run_adyme_mh_final_gate_v2 as gate
from scripts import run_epcm_mh_clinical_gate as epcm
from scripts import run_epcm_mh_daic_official_test as epcm_daic
from scripts.clinical_label_contract import validate_contract


SCHEMA = "meter_mh_strict_v1"
VIEW_NAMES = ("clinical", "answer", "context", "visual")
ANCHOR_NAMES = ("benchmark_ta", "answer_dynamic")
BASE_VARIANTS = (
    "benchmark_ta",
    "answer_dynamic",
    "meter_answer",
    "context_only",
    "meter_process",
    "meter_process_visual",
    "meter_v7",
)
VARIANT_NAMES = BASE_VARIANTS + ("meter_pim", "meter_full")
METER_ROUTE_CANDIDATES = ("meter_pim", "meter_full")
CONTROL_NAMES = (
    "time_shuffle",
    "context_swap",
    "history_ablation",
    "visual_ablation",
    "selected_evidence_delete",
    "random_evidence_delete",
    "selected_evidence_only",
    "random_evidence_only",
)
TASKS = {
    "daic_depression": ("daic", "depression"),
    "edaic_depression": ("edaic", "depression"),
    "edaic_ptsd": ("edaic", "ptsd"),
}
EXTERNAL_TAV_BASELINES = {
    "daic_depression": {"weighted_f1_mean": 0.6963, "weighted_f1_std": 0.0224, "n_test": 47},
    "edaic_depression": {"weighted_f1_mean": 0.596, "weighted_f1_std": 0.012, "n_test": 56},
    "edaic_ptsd": {"weighted_f1_mean": 0.500, "weighted_f1_std": 0.010, "n_test": 56},
}


def configure_protocol() -> None:
    """Install the METER representation/model contract into audited engines."""

    epcm.SCHEMA = SCHEMA
    epcm.VIEW_NAMES = VIEW_NAMES
    epcm.VARIANT_NAMES = VARIANT_NAMES
    epcm.CONTROL_NAMES = CONTROL_NAMES

    gate.SCHEMA = SCHEMA
    gate.VIEW_NAMES = VIEW_NAMES
    gate.ANCHOR_NAMES = ANCHOR_NAMES
    gate.VARIANT_NAMES = VARIANT_NAMES
    gate.CONTROL_NAMES = CONTROL_NAMES
    gate.classification_metrics = epcm.classification_metrics
    gate.load_records = epcm.load_records
    gate.fit_all_variants = fit_all_variants

    daic_official.SCHEMA = SCHEMA
    daic_official.ANCHOR_NAMES = ANCHOR_NAMES
    daic_official.VARIANT_NAMES = VARIANT_NAMES
    daic_official.CONTROL_NAMES = CONTROL_NAMES
    daic_official.classification_metrics = epcm.classification_metrics
    daic_official.fit_all_variants = fit_all_variants
    daic_official.load_records_official = epcm_daic.load_records_official
    daic_official.load_meld_train = lambda _root: np.empty((0, 0), dtype=np.float32)


def make_features(
    records: list[dict[str, Any]],
    projections: dict[str, Any],
    scorer: tuple[Any, Any],
    control: str | None = None,
) -> dict[str, np.ndarray]:
    """Build matched TAV anchors and METER response-measurement features."""

    processed: list[dict[str, np.ndarray]] = []
    for index, record in enumerate(records):
        raw = {key: gate.finite(value).copy() for key, value in record["raw"].items()}
        rng = np.random.default_rng(20260811 + index * 97)
        if control == "time_shuffle":
            order = rng.permutation(len(raw["answer"]))
            for key in VIEW_NAMES:
                raw[key] = raw[key][order]
        elif control == "history_ablation":
            raw["context"] = np.zeros_like(raw["context"])
        elif control == "context_swap" and len(records) > 1:
            source = records[(index + 1) % len(records)]["raw"]["context"]
            if len(source) >= len(raw["context"]):
                raw["context"] = gate.finite(source[: len(raw["context"])])
            else:
                repeats = int(np.ceil(len(raw["context"]) / len(source)))
                raw["context"] = gate.finite(
                    np.tile(source, (repeats, 1))[: len(raw["context"])]
                )
        if control == "visual_ablation":
            raw["visual"] = np.zeros_like(raw["visual"])

        projected = {name: gate.transform(raw[name], projections[name]) for name in VIEW_NAMES}
        clean_events = epcm.event_matrix(projected)
        selected = epcm.evidence_indices(clean_events, scorer)
        all_indices = np.arange(len(clean_events))
        complement = np.setdiff1d(all_indices, selected, assume_unique=True)
        random_indices = np.sort(rng.choice(all_indices, size=len(selected), replace=False))

        evidence_controls = {
            "selected_evidence_delete",
            "random_evidence_delete",
            "selected_evidence_only",
            "random_evidence_only",
        }
        if control in evidence_controls:
            affected = selected if control.startswith("selected") else random_indices
            keep = np.ones(len(clean_events), dtype=bool)
            if control.endswith("delete"):
                keep[affected] = False
            else:
                keep[:] = False
                keep[affected] = True
            for name in VIEW_NAMES:
                projected[name] = projected[name].copy()
                projected[name][~keep] = 0.0

        clinical = projected["clinical"]
        answer = projected["answer"]
        context = projected["context"]
        visual = projected["visual"]
        benchmark_event = gate.transform(record["benchmark"], projections["benchmark"])

        benchmark_ta = gate.mean_std(record["benchmark"])
        benchmark_visual = gate.summarize(visual)
        benchmark_tav = gate.finite(np.concatenate((benchmark_ta, benchmark_visual)))
        answer_dynamic = gate.summarize(benchmark_event)
        meter_answer = gate.summarize(answer)
        clinical_dynamic = gate.summarize(clinical)
        context_dynamic = gate.summarize(context)
        conditional = gate.summarize(clinical - answer)
        history_response = gate.summarize(clinical - context)
        visual_dynamic = gate.summarize(visual)
        process = gate.finite(
            np.concatenate(
                (meter_answer, clinical_dynamic, context_dynamic, conditional, history_response)
            )
        )
        process_visual = gate.finite(np.concatenate((process, visual_dynamic)))

        selected_blocks = [
            epcm.summarize_subset(projected[name], selected) for name in VIEW_NAMES
        ]
        complement_blocks = [
            epcm.summarize_subset(projected[name], complement) for name in VIEW_NAMES
        ]
        evidence_block = gate.finite(
            np.concatenate(
                selected_blocks
                + complement_blocks
                + [np.concatenate(selected_blocks) - np.concatenate(complement_blocks)]
            )
        )
        processed.append(
            {
                # Historical engine key retained for compatibility; this
                # value is the matched Text+Audio plus audited visual anchor.
                "benchmark_ta": benchmark_tav,
                "answer_dynamic": answer_dynamic,
                "meter_answer": meter_answer,
                "context_only": context_dynamic,
                "meter_process": process,
                "meter_process_visual": process_visual,
                "meter_v7": gate.finite(np.concatenate((process_visual, evidence_block))),
            }
        )
    return {
        name: np.stack([item[name] for item in processed])
        for name in BASE_VARIANTS
    }


def fit_all_variants(
    train: list[dict[str, Any]],
    held: list[dict[str, Any]],
    _unused: np.ndarray,
    control: str | None,
    seed: int,
    *,
    held_control_only: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit all ablations and the unified METER measurement path."""

    projections = epcm.fit_view_projections(train, np.empty((0, 0), dtype=np.float32))
    scorer = epcm.fit_evidence_scorer(train, projections, seed + 17)
    train_features = make_features(
        train, projections, scorer, control=None if held_control_only else control
    )
    held_features = make_features(held, projections, scorer, control=control)
    predictions: dict[str, np.ndarray] = {}
    configs: dict[str, Any] = {}
    y = np.asarray([record["label"] for record in train], dtype=np.int64)
    severity = gate.score_array(train)
    severity_predictions: dict[str, np.ndarray] = {}

    for variant in BASE_VARIANTS:
        x_train = gate.finite(train_features[variant])
        x_held = gate.finite(held_features[variant])
        c_value, threshold = gate.choose_classifier(x_train, y, seed + len(variant))
        predictions[variant] = gate.predict_classifier(
            gate.fit_classifier(x_train, y, c_value, seed + 1000), x_held
        )
        configs[variant] = {"c": c_value, "threshold": threshold, "feature": variant}
        score_mask = np.isfinite(severity)
        if int(score_mask.sum()) >= 8:
            alpha = gate.choose_ridge_alpha(
                x_train[score_mask], severity[score_mask], seed + 2000
            )
            bundle = gate.fit_regressor(
                x_train[score_mask], severity[score_mask], alpha, seed + 3000
            )
            severity_predictions[variant] = gate.predict_regressor(bundle, x_held)
            configs[variant]["alpha"] = alpha

    pim_probability, pim_score, pim_config = epcm.measurement_head(
        train_features["meter_v7"],
        y,
        severity,
        held_features["meter_v7"],
        seed + 4000,
    )
    predictions["meter_pim"] = pim_probability
    severity_predictions["meter_pim"] = pim_score
    configs["meter_pim"] = {"feature": "meter_v7", **pim_config}

    predictions["meter_full"] = np.clip(
        0.5 * predictions["meter_v7"] + 0.5 * predictions["meter_pim"],
        1e-6,
        1.0 - 1e-6,
    )
    severity_predictions["meter_full"] = pim_score
    configs["meter_full"] = {
        "feature": "fixed meter_v7 + meter_pim measurement ensemble",
        "weights": {"meter_v7": 0.5, "meter_pim": 0.5},
        "threshold": 0.5,
    }
    configs["severity_predictions"] = severity_predictions
    return predictions, configs


def select_meter_route(result: dict[str, Any]) -> dict[str, Any]:
    """Select the task-conditioned METER path from training OOF only."""

    eligible: list[tuple[float, float, float, str]] = []
    policies = result["final_policies"]
    for candidate in METER_ROUTE_CANDIDATES:
        policy = policies.get(candidate, {})
        audits = policy.get("exact_seed_audit", [])
        if policy.get("status") != "admitted_after_seed_stability" or not audits:
            continue
        f1 = float(np.mean([item["metrics"]["weighted_f1"] for item in audits]))
        auroc = float(np.mean([item["metrics"].get("auroc", 0.0) for item in audits]))
        brier = float(np.mean([item["metrics"]["brier"] for item in audits]))
        eligible.append((f1, auroc, -brier, candidate))

    if not eligible:
        summary = result["locked_summary"]["raw"]["benchmark_ta"]
        return {
            "status": "fallback_to_matched_tav_anchor",
            "selected_path": "benchmark_ta",
            "selection_source": "training_oof_only",
            "test_metrics": summary,
        }

    _, _, _, selected = max(eligible)
    return {
        "status": "meter_path_admitted",
        "selected_path": selected,
        "selection_source": "training_oof_only",
        "candidate_set": list(METER_ROUTE_CANDIDATES),
        "policy": policies[selected],
        "test_metrics": result["locked_summary"]["protected"][selected],
        "paired_bootstrap": [
            run["paired_bootstrap"].get(selected, {}) for run in result["locked"]
        ],
    }


def evaluate_daic_strict(args: argparse.Namespace) -> dict[str, Any]:
    """Lock DAIC policies before the official 47-person test is loaded."""

    train = epcm_daic.load_records_official(
        "train", args.meter_root, args.benchmark_root, args.contract_dir
    )
    dev = epcm_daic.load_records_official(
        "dev", args.meter_root, args.benchmark_root, args.contract_dir
    )
    task_dir = args.output / "work" / "daic_depression" / "daic" / "depression"
    task_dir.mkdir(parents=True, exist_ok=True)
    gate.atomic_json(
        task_dir / "LOCKED_PROTOCOL.json",
        {
            "schema": SCHEMA,
            "dataset": "daic",
            "target": "depression",
            "label_profile": "depression_phq8_screening",
            "split_unit": "participant",
            "official_split": True,
            "cohort_counts_before_test_open": {"train": len(train), "dev": len(dev)},
            "seeds": list(gate.SEEDS),
            "test_opened": False,
            "anchors": list(ANCHOR_NAMES),
            "route_candidates": list(METER_ROUTE_CANDIDATES),
            "test_selection_forbidden": True,
            "raw_inputs_read_only": True,
        },
    )
    gate.atomic_json(
        task_dir / "LOCK_COMPLETE.json",
        {"status": "official_train_oof_lock_started_before_test_read", "schema": SCHEMA},
    )

    empty_source_prior = np.empty((0, 0), dtype=np.float32)
    runs, final_policies, anchor_policy = daic_official.lock_policies(
        train, empty_source_prior, task_dir
    )
    gate.atomic_json(
        task_dir / "POLICY_LOCKED.json",
        {
            "status": "locked_before_official_test_read",
            "final_policies": final_policies,
            "anchor_only_policy": anchor_policy,
        },
    )

    # This is the first point at which official DAIC test labels are loaded.
    test = epcm_daic.load_records_official(
        "test", args.meter_root, args.benchmark_root, args.contract_dir
    )
    fit_rows = train + dev
    y_test = np.asarray([record["label"] for record in test], dtype=np.int64)
    locked_runs = []
    for seed in gate.SEEDS:
        predictions, configs = fit_all_variants(
            fit_rows, test, empty_source_prior, None, seed + 180000
        )
        raw = {
            name: epcm.classification_metrics(y_test, probability, configs[name]["threshold"])
            for name, probability in predictions.items()
        }
        scores = gate.score_array(test)
        score_mask = np.isfinite(scores)
        raw_regression = {
            name: gate.severity_metrics(scores[score_mask], values[score_mask])
            for name, values in configs.get("severity_predictions", {}).items()
            if int(score_mask.sum()) >= 2
        }
        protected: dict[str, Any] = {}
        bootstrap: dict[str, Any] = {}
        for index, (candidate, policy) in enumerate(final_policies.items()):
            probability = gate.mix_predictions(
                predictions["benchmark_ta"],
                predictions["answer_dynamic"],
                predictions[candidate],
                policy["weights"],
            )
            protected[candidate] = epcm.classification_metrics(
                y_test, probability, policy["threshold"]
            )
            bootstrap[candidate] = {
                anchor: gate.paired_bootstrap_delta(
                    y_test,
                    probability,
                    policy["threshold"],
                    predictions[anchor],
                    configs[anchor]["threshold"],
                    seed * 1000 + index,
                )
                for anchor in ANCHOR_NAMES
            }
        controls: dict[str, Any] = {}
        for control in CONTROL_NAMES:
            control_predictions, control_configs = fit_all_variants(
                fit_rows,
                test,
                empty_source_prior,
                control,
                seed + 180000,
                held_control_only=True,
            )
            controls[control] = {
                name: epcm.classification_metrics(
                    y_test, probability, control_configs[name]["threshold"]
                )
                for name, probability in control_predictions.items()
            }
        locked_runs.append(
            {
                "seed": seed,
                "raw": raw,
                "raw_regression": raw_regression,
                "protected": protected,
                "paired_bootstrap": bootstrap,
                "controls": controls,
                "policies": final_policies,
            }
        )

    return {
        "inventory": {"train": len(train), "dev": len(dev), "test": len(test)},
        "cache_signature_audit": gate.cache_signature_audit(args.meter_root, ("daic",)),
        "oof_train": runs,
        "final_policies": final_policies,
        "anchor_only_policy": anchor_policy,
        "locked": locked_runs,
        "locked_summary": {
            "raw": gate.aggregate_metric_blocks([item["raw"] for item in locked_runs]),
            "protected": gate.aggregate_metric_blocks(
                [item["protected"] for item in locked_runs]
            ),
            "raw_regression": gate.aggregate_metric_blocks(
                [item["raw_regression"] for item in locked_runs]
            ),
            "controls": {
                control: gate.aggregate_metric_blocks(
                    [item["controls"][control] for item in locked_runs]
                )
                for control in CONTROL_NAMES
            },
        },
        "protocol": {
            "schema": SCHEMA,
            "split_unit": "participant",
            "official_split": True,
            "train_oof_policy_lock_before_test_read": True,
            "test_opened_after_policy_lock": True,
            "seeds": list(gate.SEEDS),
            "anchor_names": list(ANCHOR_NAMES),
            "route_candidates": list(METER_ROUTE_CANDIDATES),
            "test_selection_forbidden": True,
            "raw_inputs_read_only": True,
        },
    }


def evaluate_task(args: argparse.Namespace, task_name: str) -> dict[str, Any]:
    configure_protocol()
    dataset, target = TASKS[task_name]
    namespace = argparse.Namespace(
        adyme_root=args.meter_root,
        benchmark_root=args.benchmark_root,
        contract_dir=args.contract_dir,
        output=args.output / "work" / task_name,
        threads=args.threads,
    )
    namespace.output.mkdir(parents=True, exist_ok=True)
    if dataset == "daic":
        result = evaluate_daic_strict(args)
    else:
        result = gate.evaluate_dataset(
            namespace, dataset, target, np.empty((0, 0), dtype=np.float32)
        )
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "task": task_name,
        "dataset": dataset,
        "target": target,
        "result": result,
        "meter_mh": select_meter_route(result),
        "external_tav_baseline": EXTERNAL_TAV_BASELINES[task_name],
        "raw_inputs_read_only": True,
    }
    gate.atomic_json(args.output / "tasks" / f"{task_name}.json", payload)
    return payload


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    tasks = {}
    for task_name in TASKS:
        path = args.output / "tasks" / f"{task_name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed task result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise RuntimeError(f"task is not complete: {task_name}")
        tasks[task_name] = payload

    summary = {}
    for task_name, payload in tasks.items():
        baseline = payload["external_tav_baseline"]
        meter = payload["meter_mh"]
        metric = meter["test_metrics"]["weighted_f1"]
        summary[task_name] = {
            "selected_path": meter["selected_path"],
            "meter_weighted_f1": metric,
            "external_tav_baseline": baseline,
            "delta_weighted_f1": float(metric["mean"] - baseline["weighted_f1_mean"]),
        }

    final = {
        "schema": SCHEMA,
        "status": "complete",
        "model": "METER-MH",
        "definition": (
            "HumanOmniV2 clinical TA observations + answer dynamics + audited visual evidence + "
            "V7 evidence faithfulness + PIM continuous measurement + train-OOF task-conditioned route"
        ),
        "single_model_policy": {
            "architecture_shared_across_tasks": True,
            "task_conditioning": "instrument-specific clinical head and OOF-selected PIM/FULL path",
            "test_selection_used": False,
            "reported_primary_name": "METER-MH",
            "raw_protected_public_split": False,
        },
        "tasks": tasks,
        "primary_summary": summary,
        "protocol": {
            "daic": "official 107/35/47 participant split",
            "edaic": "audited canonical train/dev/test participant split",
            "label_contract": "clinical_label_contract_v1",
            "depression_profile": "depression_phq8_screening",
            "ptsd_profile": "ptsd_dsm_iv_binary",
            "seeds": list(gate.SEEDS),
            "route_candidates_frozen": list(METER_ROUTE_CANDIDATES),
            "route_selection": "training participant OOF only",
            "test_status": "retrospective locked evaluation; historically inspected by the project",
            "raw_inputs_read_only": True,
        },
    }
    gate.atomic_json(args.output / "FINAL_METRICS.json", final)
    gate.atomic_json(
        args.output / "COMPLETE.json",
        {"status": "complete", "schema": SCHEMA, "tasks": list(TASKS)},
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meter-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    gate.configure_threads(min(32, max(1, int(args.threads))))
    validate_contract(args.contract_dir)
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "meter_mh_strict_protocol_v1":
        raise ValueError("unexpected METER-MH protocol schema")
    for dataset, expected in (("daic", 189), ("edaic", 275)):
        final_cache = args.meter_root / dataset / "FINAL_CACHE.json"
        state = json.loads(final_cache.read_text(encoding="utf-8"))
        if state.get("status") != "complete" or int(state.get("total_items", -1)) != expected:
            raise RuntimeError(f"incomplete METER cache for {dataset}: {state}")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        final = finalize(args)
        print(json.dumps({"event": "meter_mh_complete", "summary": final["primary_summary"]}))
        return 0
    if args.task is None:
        parser.error("--task is required unless --finalize is used")
    gate.atomic_json(
        args.output / "tasks" / f"{args.task}.progress.json",
        {"status": "running", "schema": SCHEMA, "task": args.task},
    )
    payload = evaluate_task(args, args.task)
    gate.atomic_json(
        args.output / "tasks" / f"{args.task}.progress.json",
        {"status": "complete", "schema": SCHEMA, "task": args.task},
    )
    print(json.dumps({"event": "meter_mh_task_complete", "task": args.task, "meter_mh": payload["meter_mh"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
