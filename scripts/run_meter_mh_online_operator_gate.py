#!/usr/bin/env python3
"""Export and audit the cached METER-PIM path as an online PyTorch graph.

This gate performs no HumanOmniV2 training.  It fits the original strict
scikit-learn measurement operator on governed official Train+Dev records,
converts the fitted operator to a frozen PyTorch graph, and verifies the graph
from raw online event views through probability and continuous severity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_adyme_mh_final_gate_v2 as gate
from scripts import run_epcm_mh_clinical_gate as epcm
from scripts import run_meter_mh_strict as strict
from scripts.clinical_label_contract import load_label_profile, validate_contract
from scripts.meter_mh_online_operator import FrozenMeterPIM


SCHEMA = "meter_mh_online_operator_gate_v1"
TASKS = {
    "daic_depression": {
        "dataset": "daic",
        "binary_profile": "depression_phq8_screening",
        "score_profile": "depression_phq_score",
        "locked_seed_offset": 180000,
    },
    "edaic_depression": {
        "dataset": "edaic",
        "binary_profile": "depression_phq8_screening",
        "score_profile": "depression_phq_score",
        "locked_seed_offset": 80000,
    },
    "edaic_ptsd": {
        "dataset": "edaic",
        "binary_profile": "ptsd_dsm_iv_binary",
        "score_profile": "ptsd_score_repaired",
        "locked_seed_offset": 80000,
    },
}
EXPECTED_SPLITS = {
    "daic": {"train": 107, "dev": 35, "test": 47},
    "edaic": {"train": 163, "dev": 56, "test": 56},
}
VIEWS = ("clinical", "answer", "context", "visual")


def configure_threads(value: int) -> None:
    value = max(1, min(32, int(value)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(value)
    torch.set_num_threads(value)


def normalize_pid(value: Any) -> str:
    return str(int(float(str(value).strip())))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=gate.json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_signature(cache_root: Path, dataset: str) -> str:
    path = cache_root / dataset / "RUN_CONTRACT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = str(payload.get("run_signature", ""))
    if not signature:
        raise RuntimeError(f"missing run signature: {path}")
    return signature


def load_score_map(contract_dir: Path, task: dict[str, Any]) -> dict[tuple[str, str], float | None]:
    frame, profile = load_label_profile(
        contract_dir,
        task["dataset"],
        task["score_profile"],
        include_unlabeled=True,
    )
    result: dict[tuple[str, str], float | None] = {}
    for row in frame.to_dict("records"):
        pid, split = normalize_pid(row["pid"]), str(row["split"])
        value, mask = row.get(profile.target_column), row.get(profile.mask_column)
        try:
            result[(pid, split)] = (
                float(value)
                if int(mask) == 1 and value is not None and np.isfinite(float(value))
                else None
            )
        except (TypeError, ValueError):
            result[(pid, split)] = None
    return result


def load_task_records(
    cache_root: Path,
    contract_dir: Path,
    task_name: str,
) -> dict[str, list[dict[str, Any]]]:
    task = TASKS[task_name]
    dataset = task["dataset"]
    signature = cache_signature(cache_root, dataset)
    scores = load_score_map(contract_dir, task)
    result: dict[str, list[dict[str, Any]]] = {}
    for split, expected in EXPECTED_SPLITS[dataset].items():
        frame, profile = load_label_profile(
            contract_dir,
            dataset,
            task["binary_profile"],
            split=split,
            include_unlabeled=False,
        )
        records: list[dict[str, Any]] = []
        for row in frame.to_dict("records"):
            source_split = str(row["split"])
            canonical_split = str(row.get("canonical_split", source_split))
            if source_split != split:
                continue
            if dataset == "edaic" and canonical_split != source_split:
                raise RuntimeError(
                    f"E-DAIC official/canonical split mismatch pid={row['pid']}: "
                    f"official={source_split} canonical={canonical_split}"
                )
            pid = normalize_pid(row["pid"])
            path = cache_root / dataset / "cache" / f"{pid}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                raw = {name: gate.finite(archive[name]) for name in VIEWS}
                metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("run_signature") != signature:
                raise RuntimeError(f"cache signature mismatch: {path}")
            if any(value.ndim != 2 or len(value) == 0 for value in raw.values()):
                raise ValueError(f"invalid METER views {dataset}/{pid}")
            if len({len(value) for value in raw.values()}) != 1:
                raise ValueError(f"unaligned METER views {dataset}/{pid}")
            records.append(
                {
                    "pid": pid,
                    "split": split,
                    "label": int(row[profile.target_column]),
                    "score": scores.get((pid, split)),
                    "raw": raw,
                    "native_visual_dim": int(
                        metadata.get("visual_audit", {}).get(
                            "native_feature_dim", raw["visual"].shape[1]
                        )
                    ),
                }
            )
        records.sort(key=lambda item: int(item["pid"]))
        if len(records) != expected:
            raise RuntimeError(f"{dataset} {split} count mismatch: {len(records)} != {expected}")
        result[split] = records
    return result


def raw_view(record: dict[str, Any], name: str) -> np.ndarray:
    value = record["raw"][name]
    if name != "visual":
        return value
    native = int(record.get("native_visual_dim", value.shape[1]))
    if not 1 <= native <= value.shape[1]:
        raise ValueError(f"invalid native visual width: {native} for {value.shape}")
    tail = value[:, native:]
    if tail.size and np.max(np.abs(tail)) > 1e-8:
        raise ValueError("visual padding tail contains non-zero values")
    return value[:, :native]


def fit_projections(records: list[dict[str, Any]]) -> dict[str, Any]:
    shared = gate.fit_projection(
        np.concatenate(
            [raw_view(record, view) for record in records for view in ("clinical", "answer", "context")],
            axis=0,
        ),
        20260810,
        gate.EVENT_COMPONENTS,
    )
    visual = gate.fit_projection(
        np.concatenate([raw_view(record, "visual") for record in records], axis=0),
        20260812,
        gate.EVENT_COMPONENTS,
    )
    return {"clinical": shared, "answer": shared, "context": shared, "visual": visual}


def meter_v7_features(
    records: list[dict[str, Any]],
    projections: dict[str, Any],
    scorer: tuple[Any, Any],
) -> np.ndarray:
    features = []
    for record in records:
        projected = {
            name: gate.transform(raw_view(record, name), projections[name])
            for name in VIEWS
        }
        events = epcm.event_matrix(projected)
        selected = epcm.evidence_indices(events, scorer)
        all_indices = np.arange(len(events))
        complement = np.setdiff1d(all_indices, selected, assume_unique=True)
        clinical, answer = projected["clinical"], projected["answer"]
        context, visual = projected["context"], projected["visual"]
        process = gate.finite(
            np.concatenate(
                (
                    gate.summarize(answer),
                    gate.summarize(clinical),
                    gate.summarize(context),
                    gate.summarize(clinical - answer),
                    gate.summarize(clinical - context),
                )
            )
        )
        process_visual = gate.finite(np.concatenate((process, gate.summarize(visual))))
        selected_blocks = [epcm.summarize_subset(projected[name], selected) for name in VIEWS]
        complement_blocks = [epcm.summarize_subset(projected[name], complement) for name in VIEWS]
        selected_cat = np.concatenate(selected_blocks)
        complement_cat = np.concatenate(complement_blocks)
        features.append(
            gate.finite(
                np.concatenate((process_visual, selected_cat, complement_cat, selected_cat - complement_cat))
            )
        )
    return np.stack(features)


def fit_pim_bundle(
    x_train: np.ndarray,
    labels: np.ndarray,
    severity: np.ndarray,
    seed: int,
) -> tuple[tuple[Any, Any, Any], StandardScaler, LogisticRegression, float]:
    mask = np.isfinite(severity)
    x_score, score, y_score = x_train[mask], severity[mask], labels[mask]
    if len(score) < 8 or len(np.unique(y_score)) < 2:
        raise RuntimeError("insufficient continuous-label rows for METER-PIM")
    alpha = gate.choose_ridge_alpha(x_score, score, seed)
    folds = min(4, len(score))
    oof_score = np.zeros(len(score), dtype=np.float64)
    for fold, (fit_idx, held_idx) in enumerate(
        KFold(folds, shuffle=True, random_state=seed).split(x_score)
    ):
        bundle = gate.fit_regressor(x_score[fit_idx], score[fit_idx], alpha, seed + fold * 101)
        oof_score[held_idx] = gate.predict_regressor(bundle, x_score[held_idx])
    score_scaler = StandardScaler().fit(oof_score[:, None])
    calibrator = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    ).fit(score_scaler.transform(oof_score[:, None]), y_score)
    severity_bundle = gate.fit_regressor(x_score, score, alpha, seed + 1000)
    return severity_bundle, score_scaler, calibrator, float(alpha)


def fit_operator(
    fit_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[FrozenMeterPIM, float]:
    projections = fit_projections(fit_rows)
    scorer = epcm.fit_evidence_scorer(fit_rows, projections, seed + 17)
    features = meter_v7_features(fit_rows, projections, scorer)
    labels = np.asarray([row["label"] for row in fit_rows], dtype=np.int64)
    severity = gate.score_array(fit_rows)
    bundle, score_scaler, calibrator, alpha = fit_pim_bundle(
        features, labels, severity, seed + 4000
    )
    module = FrozenMeterPIM.from_sklearn(
        projections=projections,
        evidence_scorer=scorer,
        severity_bundle=bundle,
        score_scaler=score_scaler,
        calibrator=calibrator,
    ).eval()
    return module, alpha


def formal_raw_meter_pim(formal_results: Path, task_name: str) -> dict[str, float]:
    path = formal_results / "tasks" / f"{task_name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"]["locked_summary"]["raw"]["meter_pim"]["weighted_f1"]


def run_task(
    args: argparse.Namespace,
    task_name: str,
) -> dict[str, Any]:
    task = TASKS[task_name]
    rows = load_task_records(args.meter_root, args.contract_dir, task_name)
    fit_rows = rows["train"] + rows["dev"]
    held_rows = rows["test"]
    y_fit = np.asarray([row["label"] for row in fit_rows], dtype=np.int64)
    y_held = np.asarray([row["label"] for row in held_rows], dtype=np.int64)
    severity_fit = gate.score_array(fit_rows)
    seed_results = []
    for seed in gate.SEEDS:
        locked_seed = int(seed + task["locked_seed_offset"])
        development_seed = int(seed + 40000)
        development_module, development_alpha = fit_operator(rows["train"], development_seed)
        development_dir = args.output / "operators_train" / task_name
        development_dir.mkdir(parents=True, exist_ok=True)
        development_path = development_dir / f"seed_{seed}.pt"
        torch.save(
            {
                "schema": "meter_mh_online_operator_v1",
                "scope": "official_train_only_for_development",
                "task": task_name,
                "seed": int(seed),
                "fit_seed": development_seed,
                "module": development_module,
                "threshold": 0.5,
                "alpha": development_alpha,
                "cache_signature": cache_signature(args.meter_root, task["dataset"]),
                "label_profile": task["binary_profile"],
                "score_profile": task["score_profile"],
            },
            development_path,
        )
        projections = fit_projections(fit_rows)
        scorer = epcm.fit_evidence_scorer(fit_rows, projections, locked_seed + 17)
        fit_features = meter_v7_features(fit_rows, projections, scorer)
        held_features = meter_v7_features(held_rows, projections, scorer)
        bundle, score_scaler, calibrator, alpha = fit_pim_bundle(
            fit_features,
            y_fit,
            severity_fit,
            locked_seed + 4000,
        )
        sklearn_severity = gate.predict_regressor(bundle, held_features)
        sklearn_probability = calibrator.predict_proba(
            score_scaler.transform(sklearn_severity[:, None])
        )[:, 1]

        module = FrozenMeterPIM.from_sklearn(
            projections=projections,
            evidence_scorer=scorer,
            severity_bundle=bundle,
            score_scaler=score_scaler,
            calibrator=calibrator,
        ).eval()
        feature_delta = 0.0
        probability_delta = 0.0
        severity_delta = 0.0
        torch_probability = []
        torch_severity = []
        with torch.inference_mode():
            for index, record in enumerate(held_rows):
                raw = {
                    name: torch.as_tensor(record["raw"][name], dtype=torch.float64)
                    for name in VIEWS
                }
                output = module(raw)
                torch_probability.append(float(output["probability"].item()))
                torch_severity.append(float(output["severity"].item()))
                feature_delta = max(
                    feature_delta,
                    float(np.max(np.abs(output["meter_v7"].cpu().numpy() - held_features[index]))),
                )
        torch_probability_array = np.asarray(torch_probability, dtype=np.float64)
        torch_severity_array = np.asarray(torch_severity, dtype=np.float64)
        probability_delta = float(np.max(np.abs(torch_probability_array - sklearn_probability)))
        severity_delta = float(np.max(np.abs(torch_severity_array - sklearn_severity)))
        sklearn_metrics = epcm.classification_metrics(y_held, sklearn_probability, 0.5)
        torch_metrics = epcm.classification_metrics(y_held, torch_probability_array, 0.5)

        artifact_dir = args.output / "operators" / task_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"seed_{seed}.pt"
        torch.save(
            {
                "schema": "meter_mh_online_operator_v1",
                "task": task_name,
                "seed": int(seed),
                "locked_seed": locked_seed,
                "module": module,
                "threshold": 0.5,
                "alpha": alpha,
                "cache_signature": cache_signature(args.meter_root, task["dataset"]),
                "label_profile": task["binary_profile"],
                "score_profile": task["score_profile"],
            },
            artifact_path,
        )
        seed_results.append(
            {
                "seed": int(seed),
                "locked_seed": locked_seed,
                "alpha": alpha,
                "artifact": str(artifact_path),
                "development_artifact": str(development_path),
                "sklearn_metrics": sklearn_metrics,
                "torch_metrics": torch_metrics,
                "max_abs_delta": {
                    "meter_v7": feature_delta,
                    "probability": probability_delta,
                    "severity": severity_delta,
                },
                "passed": bool(
                    feature_delta <= args.feature_tolerance
                    and probability_delta <= args.probability_tolerance
                    and severity_delta <= args.severity_tolerance
                    and sklearn_metrics["weighted_f1"] == torch_metrics["weighted_f1"]
                ),
            }
        )

    observed = np.asarray(
        [item["torch_metrics"]["weighted_f1"] for item in seed_results], dtype=np.float64
    )
    formal = formal_raw_meter_pim(args.formal_results, task_name)
    observed_std = float(observed.std(ddof=1)) if len(observed) > 1 else 0.0
    formal_delta = {
        "mean": float(observed.mean() - float(formal["mean"])),
        "std": float(observed_std - float(formal["std"])),
    }
    operator_passed = all(item["passed"] for item in seed_results)
    historical_reproduced = abs(formal_delta["mean"]) <= args.metric_tolerance
    return {
        "status": "passed" if operator_passed else "failed",
        "operator_equivalence": "passed" if operator_passed else "failed",
        "historical_formal_reproduction": "passed" if historical_reproduced else "not_reproduced",
        "dataset": task["dataset"],
        "official_splits": EXPECTED_SPLITS[task["dataset"]],
        "cache_signature": cache_signature(args.meter_root, task["dataset"]),
        "seeds": seed_results,
        "torch_weighted_f1": {
            "mean": float(observed.mean()),
            "std": observed_std,
        },
        "formal_cached_raw_meter_pim": formal,
        "formal_delta": formal_delta,
        "historical_result_policy": (
            "historical result remains comparable"
            if historical_reproduced
            else "historical cache was deleted and is not used as the corrected online anchor"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meter-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--formal-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--task", choices=tuple(TASKS), action="append")
    parser.add_argument("--feature-tolerance", type=float, default=2e-5)
    parser.add_argument("--probability-tolerance", type=float, default=1e-6)
    parser.add_argument("--severity-tolerance", type=float, default=1e-5)
    parser.add_argument("--metric-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    configure_threads(args.threads)
    validate_contract(args.contract_dir)
    strict.configure_protocol()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = tuple(args.task or TASKS)
    progress = {
        "schema": SCHEMA,
        "status": "running",
        "online_humanomniv2_training": False,
        "random_measurement_head": False,
        "tasks": {},
    }
    atomic_json(args.output / "PROGRESS.json", progress)
    for task_name in tasks:
        progress["tasks"][task_name] = run_task(args, task_name)
        atomic_json(args.output / "PROGRESS.json", progress)
    passed = all(item["status"] == "passed" for item in progress["tasks"].values())
    progress["status"] = "passed" if passed else "failed"
    atomic_json(args.output / "FINAL_METRICS.json", progress)
    print(json.dumps({"event": "meter_online_operator_gate_complete", "status": progress["status"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
