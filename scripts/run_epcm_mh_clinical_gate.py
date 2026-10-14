#!/usr/bin/env python3
"""Strict clinical-only EPCM-MH validation on completed frozen caches.

MELD is deliberately absent from this gate.  It is an optional source-domain
prior and must not block the clinical mechanism test.  Model selection uses
participant-level nested OOF predictions only; DAIC dev and E-DAIC test are
opened after the policy has been locked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_adyme_mh_final_gate_v2 as gate


SCHEMA = "epcm_mh_clinical_gate_v1"
VIEW_NAMES = ("clinical", "answer", "context", "visual")
VARIANT_NAMES = (
    "benchmark_ta",
    "answer_dynamic",
    "epcm_answer",
    "context_only",
    "epcm_process",
    "epcm_process_visual",
    "epcm_v7",
    "epcm_pim",
    "epcm_full",
)
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
EVIDENCE_FRACTION = 0.25


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    total = max(1, len(y))
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probability >= lower) & (probability < upper if index < bins - 1 else probability <= upper)
        if not np.any(mask):
            continue
        value += float(mask.sum()) / total * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(value)


_classification_metrics = gate.classification_metrics


def classification_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    result = _classification_metrics(y, probability, threshold)
    result["ece_10bin"] = ece(y, np.clip(np.asarray(probability), 1e-6, 1.0 - 1e-6), 10)
    return result


def load_records(
    dataset: str,
    target: str,
    split: str,
    epcm_root: Path,
    benchmark_root: Path,
    contract_dir: Path,
) -> list[dict[str, Any]]:
    if target == "depression":
        binary_profile, score_profile = "depression_phq8_screening", "depression_phq_score"
    elif target == "ptsd":
        binary_profile, score_profile = "ptsd_dsm_iv_binary", "ptsd_score_repaired"
    else:
        raise ValueError(f"unsupported target: {target}")

    # Load only the requested participant split.  This keeps held-out labels
    # out of memory during the training/OOF policy-lock stage.
    frame, binary_spec = gate.load_label_profile(
        contract_dir, dataset, binary_profile, split=split, include_unlabeled=False
    )
    score_frame, score_spec = gate.load_label_profile(
        contract_dir, dataset, score_profile, split=split, include_unlabeled=True
    )
    scores: dict[str, float | None] = {}
    for row in score_frame.to_dict("records"):
        pid = gate.normalize_pid(row["pid"])
        value, mask = row.get(score_spec.target_column), row.get(score_spec.mask_column)
        try:
            scores[pid] = float(value) if int(mask) == 1 and value is not None and np.isfinite(float(value)) else None
        except (TypeError, ValueError):
            scores[pid] = None

    run_contract = epcm_root / dataset / "RUN_CONTRACT.json"
    expected_signature = json.loads(run_contract.read_text(encoding="utf-8")).get("run_signature")
    records = []
    for row in frame.to_dict("records"):
        if str(row["canonical_split"]) != split:
            continue
        pid = gate.normalize_pid(row["pid"])
        epcm_path = epcm_root / dataset / "cache" / f"{pid}.npz"
        benchmark_path = benchmark_root / dataset / "depression" / "text_audio" / f"{pid}_all.npy"
        if not epcm_path.is_file() or not benchmark_path.is_file():
            raise FileNotFoundError(f"missing {dataset}/{pid}: {epcm_path} {benchmark_path}")
        with np.load(epcm_path, allow_pickle=False) as archive:
            raw = {name: gate.finite(archive[name]) for name in VIEW_NAMES}
            metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("run_signature") != expected_signature:
            raise RuntimeError(f"cache signature mismatch: {epcm_path}")
        if any(value.ndim != 2 or len(value) == 0 for value in raw.values()):
            raise ValueError(f"invalid EPCM views {dataset}/{pid}: {[value.shape for value in raw.values()]}")
        if len({len(raw[name]) for name in VIEW_NAMES}) != 1:
            raise ValueError(f"unaligned EPCM views {dataset}/{pid}: {[len(raw[name]) for name in VIEW_NAMES]}")
        benchmark = gate.finite(np.load(benchmark_path, mmap_mode="r"))
        records.append({
            "pid": pid,
            "split": split,
            "label": int(row[binary_spec.target_column]),
            "score": scores.get(pid),
            "benchmark": benchmark,
            "raw": raw,
            "metadata": metadata,
            "alignment": {
                "status": "epcm_turn_aligned",
                "epcm_events": int(len(raw["clinical"])),
                "benchmark_events": int(len(benchmark)),
                "epcm_views_position_aligned": True,
                "benchmark_pooled_independently": True,
            },
        })
    return records


def fit_view_projections(records: list[dict[str, Any]], _unused: np.ndarray) -> dict[str, Any]:
    base_blocks = [record["raw"][view] for record in records for view in ("clinical", "answer", "context")]
    shared = gate.fit_projection(np.concatenate(base_blocks, axis=0), 20260810, gate.EVENT_COMPONENTS)
    benchmark = gate.fit_projection(
        np.concatenate([record["benchmark"] for record in records], axis=0), 20260811, gate.EVENT_COMPONENTS
    )
    visual_shapes = {record["raw"]["visual"].shape[1] for record in records}
    if len(visual_shapes) != 1:
        raise ValueError(f"visual dimensions differ within dataset: {sorted(visual_shapes)}")
    visual = gate.fit_projection(
        np.concatenate([record["raw"]["visual"] for record in records], axis=0), 20260812, gate.EVENT_COMPONENTS
    )
    return {"clinical": shared, "answer": shared, "context": shared, "benchmark": benchmark, "visual": visual}


def event_matrix(projected: dict[str, np.ndarray]) -> np.ndarray:
    length = min(len(projected["clinical"]), len(projected["answer"]), len(projected["context"]))
    clinical = projected["clinical"][:length]
    answer = projected["answer"][:length]
    context = projected["context"][:length]
    return gate.finite(np.concatenate((clinical, answer, context, clinical - answer, clinical - context), axis=1))


def fit_evidence_scorer(records: list[dict[str, Any]], projections: dict[str, Any], seed: int) -> tuple[StandardScaler, LogisticRegression]:
    blocks, labels, weights = [], [], []
    for record in records:
        projected = {name: gate.transform(record["raw"][name], projections[name]) for name in ("clinical", "answer", "context")}
        values = event_matrix(projected)
        blocks.append(values)
        labels.extend([record["label"]] * len(values))
        weights.extend([1.0 / len(values)] * len(values))
    matrix = np.concatenate(blocks, axis=0)
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(
        C=0.03,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=int(seed),
    ).fit(scaler.transform(matrix), np.asarray(labels), sample_weight=np.asarray(weights))
    return scaler, model


def evidence_indices(values: np.ndarray, scorer: tuple[StandardScaler, LogisticRegression]) -> np.ndarray:
    scaler, model = scorer
    evidence = np.abs(model.decision_function(scaler.transform(values)))
    count = max(1, int(np.ceil(EVIDENCE_FRACTION * len(values))))
    return np.argsort(evidence, kind="stable")[-count:]


def summarize_subset(value: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros(gate.summarize(value[:1]).shape, dtype=np.float32)
    return gate.summarize(value[np.sort(indices)])


def make_features(
    records: list[dict[str, Any]],
    projections: dict[str, Any],
    scorer: tuple[StandardScaler, LogisticRegression],
    control: str | None = None,
) -> dict[str, np.ndarray]:
    processed = []
    for index, record in enumerate(records):
        raw = {key: gate.finite(value).copy() for key, value in record["raw"].items()}
        rng = np.random.default_rng(20260810 + index * 97)
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
                raw["context"] = gate.finite(np.tile(source, (repeats, 1))[: len(raw["context"])])
        if control == "visual_ablation":
            raw["visual"] = np.zeros_like(raw["visual"])

        projected = {name: gate.transform(raw[name], projections[name]) for name in VIEW_NAMES}
        clean_events = event_matrix(projected)
        selected = evidence_indices(clean_events, scorer)
        all_indices = np.arange(len(clean_events))
        complement = np.setdiff1d(all_indices, selected, assume_unique=True)
        random_indices = np.sort(rng.choice(all_indices, size=len(selected), replace=False))

        if control in ("selected_evidence_delete", "random_evidence_delete", "selected_evidence_only", "random_evidence_only"):
            affected = selected if control.startswith("selected") else random_indices
            keep = np.ones(len(clean_events), dtype=bool)
            if control.endswith("delete"):
                keep[affected] = False
            else:
                keep[:] = False
                keep[affected] = True
            for name in ("clinical", "answer", "context", "visual"):
                projected[name] = projected[name].copy()
                projected[name][~keep] = 0.0
            clean_events = event_matrix(projected)

        clinical = projected["clinical"]
        answer = projected["answer"]
        context = projected["context"]
        visual = projected["visual"]
        benchmark_event = gate.transform(record["benchmark"], projections["benchmark"])
        answer_dynamic = gate.summarize(answer)
        clinical_dynamic = gate.summarize(clinical)
        context_dynamic = gate.summarize(context)
        conditional = gate.summarize(clinical - answer)
        history_response = gate.summarize(clinical - context)
        visual_dynamic = gate.summarize(visual)
        process = gate.finite(np.concatenate((answer_dynamic, clinical_dynamic, context_dynamic, conditional, history_response)))
        process_visual = gate.finite(np.concatenate((process, visual_dynamic)))

        selected_blocks = [summarize_subset(projected[name], selected) for name in ("clinical", "answer", "context", "visual")]
        complement_blocks = [summarize_subset(projected[name], complement) for name in ("clinical", "answer", "context", "visual")]
        evidence_block = gate.finite(np.concatenate(selected_blocks + complement_blocks + [
            np.concatenate(selected_blocks) - np.concatenate(complement_blocks)
        ]))
        processed.append({
            "benchmark_ta": gate.mean_std(record["benchmark"]),
            "answer_dynamic": gate.summarize(benchmark_event),
            "epcm_answer": answer_dynamic,
            "context_only": context_dynamic,
            "epcm_process": process,
            "epcm_process_visual": process_visual,
            "epcm_v7": gate.finite(np.concatenate((process_visual, evidence_block))),
        })
    names = tuple(processed[0])
    return {name: np.stack([item[name] for item in processed]) for name in names}


def measurement_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    severity: np.ndarray,
    x_held: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mask = np.isfinite(severity)
    if int(mask.sum()) < 8 or len(np.unique(y_train[mask])) < 2:
        raise RuntimeError("insufficient continuous-label rows for EPCM-PIM")
    x_score, score, y_score = x_train[mask], severity[mask], y_train[mask]
    alpha = gate.choose_ridge_alpha(x_score, score, seed)
    folds = min(4, len(score))
    oof_score = np.zeros(len(score), dtype=np.float64)
    for fold, (fit_idx, held_idx) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(x_score)):
        bundle = gate.fit_regressor(x_score[fit_idx], score[fit_idx], alpha, seed + fold * 101)
        oof_score[held_idx] = gate.predict_regressor(bundle, x_score[held_idx])
    score_scaler = StandardScaler().fit(oof_score[:, None])
    calibrator = LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=seed
    ).fit(score_scaler.transform(oof_score[:, None]), y_score)
    bundle = gate.fit_regressor(x_score, score, alpha, seed + 1000)
    held_score = gate.predict_regressor(bundle, x_held)
    probability = calibrator.predict_proba(score_scaler.transform(held_score[:, None]))[:, 1]
    return probability.astype(np.float64), held_score.astype(np.float64), {
        "alpha": float(alpha),
        "calibration": "OOF continuous-score Platt calibration",
        "threshold": 0.5,
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
    projections = fit_view_projections(train, np.empty((0, 0), dtype=np.float32))
    scorer = fit_evidence_scorer(train, projections, seed + 17)
    train_features = make_features(train, projections, scorer, control=None if held_control_only else control)
    held_features = make_features(held, projections, scorer, control=control)
    predictions: dict[str, np.ndarray] = {}
    configs: dict[str, Any] = {}
    y = np.asarray([record["label"] for record in train], dtype=np.int64)
    severity = gate.score_array(train)
    severity_predictions: dict[str, np.ndarray] = {}
    for variant in train_features:
        x_train, x_held = gate.finite(train_features[variant]), gate.finite(held_features[variant])
        c_value, threshold = gate.choose_classifier(x_train, y, seed + len(variant))
        predictions[variant] = gate.predict_classifier(gate.fit_classifier(x_train, y, c_value, seed + 1000), x_held)
        configs[variant] = {"c": c_value, "threshold": threshold, "feature": variant}
        score_mask = np.isfinite(severity)
        if int(score_mask.sum()) >= 8:
            alpha = gate.choose_ridge_alpha(x_train[score_mask], severity[score_mask], seed + 2000)
            bundle = gate.fit_regressor(x_train[score_mask], severity[score_mask], alpha, seed + 3000)
            severity_predictions[variant] = gate.predict_regressor(bundle, x_held)
            configs[variant]["alpha"] = alpha

    pim_probability, pim_score, pim_config = measurement_head(
        train_features["epcm_v7"], y, severity, held_features["epcm_v7"], seed + 4000
    )
    predictions["epcm_pim"] = pim_probability
    severity_predictions["epcm_pim"] = pim_score
    configs["epcm_pim"] = {"feature": "epcm_v7", **pim_config}
    predictions["epcm_full"] = np.clip(0.5 * predictions["epcm_v7"] + 0.5 * predictions["epcm_pim"], 1e-6, 1 - 1e-6)
    severity_predictions["epcm_full"] = pim_score
    configs["epcm_full"] = {
        "feature": "fixed epcm_v7 + epcm_pim",
        "weights": {"epcm_v7": 0.5, "epcm_pim": 0.5},
        "threshold": 0.5,
    }
    configs["severity_predictions"] = severity_predictions
    return predictions, configs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epcm-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--tasks",
        default="daic/depression,edaic/depression,edaic/ptsd",
        help="Comma-separated dataset/target tasks; used for CPU-bounded parallel execution.",
    )
    args = parser.parse_args()
    gate.configure_threads(args.threads)
    gate.validate_contract(args.contract_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    args.adyme_root = args.epcm_root

    gate.SCHEMA = SCHEMA
    gate.VIEW_NAMES = VIEW_NAMES
    gate.VARIANT_NAMES = VARIANT_NAMES
    gate.CONTROL_NAMES = CONTROL_NAMES
    gate.classification_metrics = classification_metrics
    gate.load_records = load_records
    gate.fit_all_variants = fit_all_variants

    payload = {
        "schema": SCHEMA,
        "status": "running",
        "datasets": {},
        "clinical_only_gate": True,
        "meld_required": False,
        "mechanism": {
            "response_process": True,
            "official_visual_as_rejectable_evidence": True,
            "v7_train_only_evidence_selection": True,
            "pim_continuous_measurement": True,
            "protected_noninferiority": True,
        },
    }
    payload["cache_signature_audit"] = gate.cache_signature_audit(args.epcm_root, ("daic", "edaic"))
    allowed_tasks = {("daic", "depression"), ("edaic", "depression"), ("edaic", "ptsd")}
    tasks = []
    for token in args.tasks.split(","):
        parts = tuple(part.strip() for part in token.strip().split("/"))
        if len(parts) != 2 or parts not in allowed_tasks:
            raise ValueError(f"unsupported EPCM task: {token}")
        if parts not in tasks:
            tasks.append(parts)
    if not tasks:
        raise ValueError("at least one EPCM task is required")
    payload["requested_tasks"] = [f"{dataset}/{target}" for dataset, target in tasks]
    task_suffix = "_".join(f"{dataset}_{target}" for dataset, target in tasks)
    progress_path = args.output / ("PROGRESS.json" if len(tasks) == 3 else f"PROGRESS_{task_suffix}.json")
    final_path = args.output / ("FINAL_METRICS.json" if len(tasks) == 3 else f"FINAL_METRICS_{task_suffix}.json")
    gate.atomic_json(progress_path, payload)
    for dataset, target in tasks:
        payload["datasets"][f"{dataset}/{target}"] = gate.evaluate_dataset(
            args, dataset, target, np.empty((0, 0), dtype=np.float32)
        )
        gate.atomic_json(progress_path, payload)
    payload["status"] = "complete"
    gate.atomic_json(final_path, payload)
    print(json.dumps({"event": "epcm_mh_clinical_gate_complete", "output": str(final_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
