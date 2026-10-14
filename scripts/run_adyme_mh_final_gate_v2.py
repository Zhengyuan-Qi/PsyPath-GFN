#!/usr/bin/env python3
"""Final, strict A-DyME-MH Gate after the HumanOmniV2 caches are complete.

This evaluator is deliberately downstream of frozen caches.  It never edits
raw datasets, label files, benchmark caches, or A-DyME caches.  The exact
HumanOmniV2 Text+Audio mean/std representation and the answer-only response
dynamics representation are protected anchors.  Context-conditioned A-DyME
variants are admitted only through participant-level nested OOF selection.

The safe selector searches a three-way convex mixture of Benchmark,
Answer-Dynamic, and one A-DyME candidate.  Every admitted mixture must be
non-inferior to both anchors in training-only OOF.  If that dual-anchor gate
cannot be established, the candidate is not admitted and both anchors remain
reported separately.  This protects the paper protocol from silently
replacing a strong baseline with a weaker mechanism.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scripts.clinical_label_contract import load_label_profile, validate_contract


SCHEMA = "adyme_mh_final_gate_v2"
SEEDS = (7, 17, 27, 41, 53)
C_VALUES = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
SAFE_WEIGHTS = tuple(np.linspace(0.0, 1.0, 11))
F1_MARGIN = 0.01
AUROC_MARGIN = 0.01
BRIER_MARGIN = 0.015
MIN_NONNEGATIVE_SEEDS = 4
BOOTSTRAP_SAMPLES = 2000
MAX_COMPONENTS = 32
EVENT_COMPONENTS = 24
VIEW_NAMES = ("answer", "joint", "context", "visual")
ANCHOR_NAMES = ("benchmark_ta", "answer_dynamic")
VARIANT_NAMES = (
    "benchmark_ta",
    "answer_dynamic",
    "adyme_answer",
    "context_only",
    "joint_response",
    "joint_response_meld",
    "joint_response_visual",
)
CONTROL_NAMES = ("time_shuffle", "context_swap", "history_ablation", "visual_ablation")


def configure_threads(value: int) -> None:
    value = max(1, min(32, int(value)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(value)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finite(value: Any) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_pid(value: Any) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def score_array(records: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for record in records:
        value = record.get("score")
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = np.nan
        values.append(value if np.isfinite(value) else np.nan)
    return np.asarray(values, dtype=np.float64)


def ccc(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if len(y) < 2:
        return 0.0
    covariance = float(np.mean((y - y.mean()) * (prediction - prediction.mean())))
    denominator = float(y.var() + prediction.var() + (y.mean() - prediction.mean()) ** 2)
    return float(2.0 * covariance / denominator) if denominator > 1e-12 else 0.0


def classification_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    prediction = (probability >= float(threshold)).astype(np.int64)
    result: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "threshold": float(threshold),
    }
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, probability))
        result["pr_auc"] = float(average_precision_score(y, probability))
    return result


def severity_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    return {
        "n": int(len(y)),
        "ccc": ccc(y, prediction),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
    }


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    choices = []
    for threshold in np.arange(0.30, 0.701, 0.025):
        block = classification_metrics(y, probability, float(threshold))
        choices.append((float(block["weighted_f1"]), float(block["macro_f1"]), -abs(float(threshold) - 0.5), float(threshold)))
    return float(max(choices)[-1])


def fit_projection(matrix: np.ndarray, seed: int, components: int = MAX_COMPONENTS) -> tuple[StandardScaler, PCA]:
    matrix = finite(matrix)
    scaler = StandardScaler().fit(matrix)
    n = max(1, min(int(components), matrix.shape[1], matrix.shape[0] - 1))
    reducer = PCA(n_components=n, whiten=True, svd_solver="randomized", random_state=int(seed)).fit(scaler.transform(matrix))
    return scaler, reducer


def transform(value: np.ndarray, projection: tuple[StandardScaler, PCA]) -> np.ndarray:
    scaler, reducer = projection
    return finite(reducer.transform(scaler.transform(finite(value))))


def fit_classifier(x: np.ndarray, y: np.ndarray, c_value: float, seed: int) -> tuple[StandardScaler, PCA, LogisticRegression]:
    projection = fit_projection(x, seed)
    model = LogisticRegression(
        C=float(c_value), class_weight="balanced", solver="liblinear", max_iter=5000, random_state=int(seed)
    ).fit(transform(x, projection), y)
    return projection[0], projection[1], model


def predict_classifier(bundle: tuple[StandardScaler, PCA, LogisticRegression], x: np.ndarray) -> np.ndarray:
    scaler, reducer, model = bundle
    return model.predict_proba(reducer.transform(scaler.transform(finite(x))))[:, 1].astype(np.float64)


def choose_classifier(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    minimum = int(np.bincount(y).min())
    folds = max(2, min(4, minimum))
    probabilities = {c_value: np.zeros(len(y), dtype=np.float64) for c_value in C_VALUES}
    for fold, (fit_idx, held_idx) in enumerate(
        StratifiedKFold(folds, shuffle=True, random_state=int(seed)).split(np.zeros(len(y)), y)
    ):
        projection = fit_projection(x[fit_idx], seed + fold * 101)
        fit_x = transform(x[fit_idx], projection)
        held_x = transform(x[held_idx], projection)
        for c_value in C_VALUES:
            model = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=int(seed + fold * 101),
            ).fit(fit_x, y[fit_idx])
            probabilities[c_value][held_idx] = model.predict_proba(held_x)[:, 1]
    choices = []
    for c_value in C_VALUES:
        threshold = choose_threshold(y, probabilities[c_value])
        block = classification_metrics(y, probabilities[c_value], threshold)
        choices.append((float(block["weighted_f1"]) - 0.05 * float(block["brier"]), float(block.get("auroc", 0.0)), -float(c_value), float(c_value), threshold))
    best = max(choices)
    return float(best[3]), float(best[4])


def choose_ridge_alpha(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    if len(y) < 6:
        return 10.0
    folds = min(4, len(y))
    errors = {alpha: [] for alpha in ALPHAS}
    for fold, (fit_idx, held_idx) in enumerate(KFold(folds, shuffle=True, random_state=int(seed)).split(x)):
        projection = fit_projection(x[fit_idx], seed + fold * 101)
        fit_x = transform(x[fit_idx], projection)
        held_x = transform(x[held_idx], projection)
        for alpha in ALPHAS:
            model = Ridge(alpha=float(alpha)).fit(fit_x, y[fit_idx])
            errors[alpha].append(mean_absolute_error(y[held_idx], model.predict(held_x)))
    choices = [(float(np.mean(errors[alpha])), -float(alpha), float(alpha)) for alpha in ALPHAS]
    return float(min(choices)[2])


def fit_regressor(x: np.ndarray, y: np.ndarray, alpha: float, seed: int) -> tuple[StandardScaler, PCA, Ridge]:
    projection = fit_projection(x, seed)
    model = Ridge(alpha=float(alpha)).fit(transform(x, projection), y)
    return projection[0], projection[1], model


def predict_regressor(bundle: tuple[StandardScaler, PCA, Ridge], x: np.ndarray) -> np.ndarray:
    scaler, reducer, model = bundle
    return model.predict(reducer.transform(scaler.transform(finite(x)))).astype(np.float64)


def summarize(sequence: np.ndarray) -> np.ndarray:
    value = finite(sequence)
    if value.ndim != 2 or len(value) == 0:
        raise ValueError(f"invalid event sequence {value.shape}")
    position = np.linspace(-1.0, 1.0, len(value), dtype=np.float32)
    delta = np.diff(value, axis=0) if len(value) > 1 else np.zeros((1, value.shape[1]), dtype=np.float32)
    slope = (position[:, None] * value).sum(axis=0) / max(float(np.square(position).sum()), 1e-6)
    return finite(np.concatenate((
        value.mean(axis=0), value.std(axis=0), np.quantile(value, 0.1, axis=0), np.quantile(value, 0.9, axis=0),
        slope, value[-1] - value[0], np.mean(np.abs(delta), axis=0), np.sqrt(np.mean(np.square(delta), axis=0)),
    )))


def mean_std(sequence: np.ndarray) -> np.ndarray:
    value = finite(sequence)
    return finite(np.concatenate((value.mean(axis=0), value.std(axis=0))))


def align_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    length = min(len(left), len(right))
    return finite(left[:length] - right[:length])


def load_meld_train(cache_root: Path) -> np.ndarray:
    contract_path = cache_root / "meld" / "RUN_CONTRACT.json"
    expected_signature = None
    if contract_path.is_file():
        expected_signature = json.loads(contract_path.read_text(encoding="utf-8")).get("run_signature")
    values = []
    for path in sorted((cache_root / "meld" / "cache").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if expected_signature and metadata.get("run_signature") != expected_signature:
                continue
            if str(metadata.get("split", "")) == "train":
                values.append(finite(archive["answer"]))
    if not values:
        raise RuntimeError("MELD train cache is empty")
    return np.concatenate(values, axis=0)


def cache_signature_audit(adyme_root: Path, datasets: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset in datasets:
        contract_path = adyme_root / dataset / "RUN_CONTRACT.json"
        expected = None
        if contract_path.is_file():
            expected = json.loads(contract_path.read_text(encoding="utf-8")).get("run_signature")
        counts: dict[str, int] = {}
        for path in sorted((adyme_root / dataset / "cache").glob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata_json"].item()))
                signature = str(metadata.get("run_signature") or "MISSING")
            except Exception:
                signature = "UNREADABLE"
            counts[signature] = counts.get(signature, 0) + 1
        result[dataset] = {
            "expected_signature": expected,
            "signature_counts": counts,
            "included_files": int(counts.get(str(expected), 0)) if expected else int(sum(counts.values())),
            "excluded_files": int(sum(counts.values()) - counts.get(str(expected), 0)) if expected else 0,
            "policy": "only RUN_CONTRACT signature is included; other files are preserved and excluded",
        }
    return result


def load_records(
    dataset: str,
    target: str,
    split: str,
    adyme_root: Path,
    benchmark_root: Path,
    contract_dir: Path,
) -> list[dict[str, Any]]:
    if target == "depression":
        binary_profile, score_profile = "depression_phq8_screening", "depression_phq_score"
    elif target == "ptsd":
        binary_profile, score_profile = "ptsd_dsm_iv_binary", "ptsd_score_repaired"
    else:
        raise ValueError(f"unsupported target: {target}")
    frame, binary_spec = load_label_profile(contract_dir, dataset, binary_profile, include_unlabeled=False)
    score_frame, score_spec = load_label_profile(contract_dir, dataset, score_profile, include_unlabeled=True)
    scores: dict[str, float | None] = {}
    for row in score_frame.to_dict("records"):
        pid = normalize_pid(row["pid"])
        value, mask = row.get(score_spec.target_column), row.get(score_spec.mask_column)
        try:
            scores[pid] = float(value) if int(mask) == 1 and value is not None and np.isfinite(float(value)) else None
        except (TypeError, ValueError):
            scores[pid] = None
    records = []
    contract_path = adyme_root / dataset / "RUN_CONTRACT.json"
    expected_signature = None
    if contract_path.is_file():
        expected_signature = json.loads(contract_path.read_text(encoding="utf-8")).get("run_signature")
    excluded_signatures: dict[str, int] = {}
    for row in frame.to_dict("records"):
        if str(row["canonical_split"]) != split:
            continue
        pid = normalize_pid(row["pid"])
        adyme_path = adyme_root / dataset / "cache" / f"{pid}.npz"
        benchmark_path = benchmark_root / dataset / "depression" / "text_audio" / f"{pid}_all.npy"
        if not adyme_path.is_file() or not benchmark_path.is_file():
            raise FileNotFoundError(f"missing {dataset}/{pid}: {adyme_path} {benchmark_path}")
        with np.load(adyme_path, allow_pickle=False) as archive:
            raw = {name: finite(archive[name]) for name in VIEW_NAMES}
            metadata = json.loads(str(archive["metadata_json"].item()))
        actual_signature = metadata.get("run_signature")
        if expected_signature and actual_signature != expected_signature:
            key = str(actual_signature or "MISSING")
            excluded_signatures[key] = excluded_signatures.get(key, 0) + 1
            continue
        benchmark = finite(np.load(benchmark_path, mmap_mode="r"))
        if any(value.ndim != 2 or len(value) == 0 for value in raw.values()):
            raise ValueError(f"invalid A-DyME shapes {dataset}/{pid}")
        # A-DyME and the historical Benchmark intentionally use different
        # event contracts.  A-DyME groups participant fragments under the
        # preceding interviewer question so that context-conditioned response
        # dynamics are observable; the Benchmark cache preserves its original
        # participant-fragment event sequence.  The clinical estimator is
        # participant-level, so the two event streams must be pooled separately
        # rather than position-matched, truncated, padded, or mixed.
        alignment = {
            "status": "participant_level_only",
            "event_alignment_required": False,
            "adyme_event_count": int(len(raw["answer"])),
            "benchmark_event_count": int(len(benchmark)),
            "event_count_equal": bool(len(raw["answer"]) == len(benchmark)),
            "adyme_event_contract": "question_conditioned_grouped_response_events",
            "benchmark_event_contract": "historical_participant_fragment_events",
            "position_matching_used": False,
            "padding_or_truncation_used": False,
        }
        records.append({
            "pid": pid,
            "split": split,
            "label": int(row[binary_spec.target_column]),
            "score": scores.get(pid),
            "benchmark": benchmark,
            "raw": raw,
            "metadata": metadata,
            "alignment": alignment,
        })
    if excluded_signatures:
        for record in records:
            record.setdefault("audit", {})["excluded_cache_signatures"] = excluded_signatures
    return records


def fit_view_projections(records: list[dict[str, Any]], meld_train: np.ndarray) -> dict[str, tuple[StandardScaler, PCA]]:
    # The three 3584-dimensional HumanOmni views share one event geometry.
    # Subtracting separately fitted PCA spaces would have no defined meaning.
    base_blocks = [
        record["raw"][view]
        for record in records
        for view in ("answer", "joint", "context")
        if record["raw"][view].shape[1] > 0
    ]
    benchmark_blocks = [record["benchmark"] for record in records if record["benchmark"].shape[1] > 0]
    if len(meld_train):
        base_blocks.append(meld_train)
    if not base_blocks:
        raise RuntimeError("no clinical event views available")
    shared = fit_projection(np.concatenate(base_blocks, axis=0), 20260808, EVENT_COMPONENTS)
    result = {"answer": shared, "joint": shared, "context": shared}
    if benchmark_blocks:
        if any(block.shape[1] != benchmark_blocks[0].shape[1] for block in benchmark_blocks):
            raise ValueError("benchmark event dimensions are inconsistent")
        result["benchmark"] = fit_projection(np.concatenate(benchmark_blocks, axis=0), 20260810, EVENT_COMPONENTS)
    visual_blocks = [record["raw"]["visual"] for record in records if record["raw"]["visual"].shape[1] > 0]
    if visual_blocks:
        result["visual"] = fit_projection(np.concatenate(visual_blocks, axis=0), 20260809, EVENT_COMPONENTS)
    return result


def make_features(records: list[dict[str, Any]], projections: dict[str, tuple[StandardScaler, PCA]], meld_mean: np.ndarray, meld_std: np.ndarray, control: str | None = None) -> dict[str, np.ndarray]:
    processed = []
    for index, record in enumerate(records):
        raw = {key: finite(value) for key, value in record["raw"].items()}
        rng = np.random.default_rng(20260808 + index * 97)
        if control == "time_shuffle":
            order = rng.permutation(len(raw["answer"]))
            for key in VIEW_NAMES:
                if len(raw[key]) == len(order):
                    raw[key] = raw[key][order]
        elif control == "history_ablation":
            raw["context"] = np.zeros_like(raw["context"])
        elif control == "context_swap" and len(records) > 1:
            source = records[(index + 1) % len(records)]
            raw["context"] = finite(source["raw"]["context"])
        if control == "visual_ablation":
            raw["visual"] = np.zeros((len(raw["answer"]), 0), dtype=np.float32)
        projected = {
            key: transform(raw[key], projections[key])
            for key in projections
            if key in raw and raw[key].shape[1] > 0
        }
        benchmark_event = transform(record["benchmark"], projections["benchmark"])
        answer = projected["answer"]
        joint = projected["joint"]
        context = projected["context"]
        # This is the protected Answer-Dynamic anchor: it is computed from
        # the same frozen Benchmark TA event sequence, never from a new
        # clinical label or from a test-tuned representation.
        answer_dynamic_anchor = summarize(benchmark_event)
        adyme_answer_dynamic = summarize(answer)
        joint_dynamic = summarize(joint)
        context_dynamic = summarize(context)
        conditional_dynamic = summarize(align_difference(joint, answer))
        context_response_dynamic = summarize(align_difference(joint, context))
        distance = np.linalg.norm((answer - meld_mean) / meld_std, axis=1)
        meld_distance = finite(np.asarray((distance.mean(), distance.std(), np.quantile(distance, .1), np.quantile(distance, .9), distance[-1] - distance[0], np.mean(np.abs(np.diff(distance))) if len(distance) > 1 else 0.0, distance.min(), distance.max())))
        values = {
            "benchmark_ta": mean_std(record["benchmark"]),
            "answer_dynamic": answer_dynamic_anchor,
            "adyme_answer": adyme_answer_dynamic,
            "context_only": context_dynamic,
            "joint_response": finite(np.concatenate((adyme_answer_dynamic, joint_dynamic, conditional_dynamic, context_response_dynamic))),
            "joint_response_meld": finite(np.concatenate((adyme_answer_dynamic, joint_dynamic, conditional_dynamic, context_response_dynamic, meld_distance))),
        }
        if "visual" in projected:
            visual_dynamic = summarize(projected["visual"])
            values["joint_response_visual"] = finite(np.concatenate((values["joint_response_meld"], visual_dynamic)))
        processed.append(values)
    names = tuple(processed[0].keys())
    return {name: np.stack([item[name] for item in processed]) for name in names}


def build_matrix(features: dict[str, np.ndarray], variant: str) -> np.ndarray:
    if variant not in features:
        raise KeyError(f"variant unavailable: {variant}")
    return finite(features[variant])


def fit_all_variants(
    train: list[dict[str, Any]],
    held: list[dict[str, Any]],
    meld_train: np.ndarray,
    control: str | None,
    seed: int,
    *,
    held_control_only: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    projections = fit_view_projections(train, meld_train)
    meld_projection = transform(meld_train, projections["answer"])
    meld_mean = finite(meld_projection.mean(axis=0))
    meld_std = np.maximum(finite(meld_projection.std(axis=0)), 1e-3)
    # Strict mechanism controls keep the fitted clinical model unchanged and
    # perturb only the held participant observations. This prevents a model
    # from relearning a shuffled or ablated representation and makes the
    # resulting performance change attributable to the tested evidence path.
    train_features = make_features(
        train,
        projections,
        meld_mean,
        meld_std,
        control=None if held_control_only else control,
    )
    held_features = make_features(held, projections, meld_mean, meld_std, control=control)
    variants = [name for name in VARIANT_NAMES if name in train_features and name in held_features]
    predictions: dict[str, np.ndarray] = {}
    configs: dict[str, Any] = {}
    y = np.asarray([record["label"] for record in train], dtype=np.int64)
    severity = np.asarray([
        np.nan if record["score"] is None else float(record["score"])
        for record in train
    ], dtype=np.float64)
    severity_predictions: dict[str, np.ndarray] = {}
    for variant in variants:
        x_train, x_held = build_matrix(train_features, variant), build_matrix(held_features, variant)
        c_value, threshold = choose_classifier(x_train, y, seed + len(variant))
        predictions[variant] = predict_classifier(fit_classifier(x_train, y, c_value, seed + 1000), x_held)
        config: dict[str, Any] = {"c": c_value, "threshold": threshold, "feature": variant}
        score_mask = np.isfinite(severity)
        if int(score_mask.sum()) >= 6:
            alpha = choose_ridge_alpha(x_train[score_mask], severity[score_mask], seed + 2000)
            bundle = fit_regressor(x_train[score_mask], severity[score_mask], alpha, seed + 3000)
            severity_predictions[variant] = predict_regressor(bundle, x_held)
            config["alpha"] = alpha
        configs[variant] = config
    configs["severity_predictions"] = severity_predictions
    return predictions, configs


def mixture_grid() -> list[dict[str, float]]:
    """Return a finite simplex with exact sum-one weights.

    The vertices are exact Benchmark and exact Answer-Dynamic fallbacks.  A
    candidate can only be admitted if it contributes positive mass and the
    resulting mixture passes both Anchor constraints.
    """
    grid: list[dict[str, float]] = []
    for benchmark_index in range(11):
        for answer_index in range(11 - benchmark_index):
            candidate_index = 10 - benchmark_index - answer_index
            grid.append({
                "benchmark_weight": benchmark_index / 10.0,
                "answer_dynamic_weight": answer_index / 10.0,
                "candidate_weight": candidate_index / 10.0,
            })
    return grid


def mix_predictions(
    benchmark: np.ndarray,
    answer_dynamic: np.ndarray,
    candidate: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    return np.clip(
        weights["benchmark_weight"] * benchmark
        + weights["answer_dynamic_weight"] * answer_dynamic
        + weights["candidate_weight"] * candidate,
        1e-6,
        1.0 - 1e-6,
    )


def noninferior_to_both(metrics: dict[str, Any], anchors: dict[str, dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, Any] = {}
    passed = True
    for name, anchor in anchors.items():
        current = {
            "weighted_f1": float(metrics["weighted_f1"] - anchor["weighted_f1"]),
            "auroc": float(metrics.get("auroc", 0.0) - anchor.get("auroc", 0.0)),
            "brier": float(metrics["brier"] - anchor["brier"]),
        }
        checks[name] = {
            "delta_weighted_f1": current["weighted_f1"],
            "delta_auroc": current["auroc"],
            "delta_brier": current["brier"],
            "weighted_f1_ok": current["weighted_f1"] >= -F1_MARGIN,
            "auroc_ok": current["auroc"] >= -AUROC_MARGIN,
            "brier_ok": current["brier"] <= BRIER_MARGIN,
        }
        passed = passed and checks[name]["weighted_f1_ok"] and checks[name]["auroc_ok"] and checks[name]["brier_ok"]
    return bool(passed), checks


def choose_anchor_only_policy(
    benchmark: np.ndarray,
    answer_dynamic: np.ndarray,
    y: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Choose an anchor-only policy only if it passes both-anchor checks."""
    benchmark = np.asarray(benchmark, dtype=np.float64)
    answer_dynamic = np.asarray(answer_dynamic, dtype=np.float64)
    anchor_metrics = {
        name: classification_metrics(y, values, thresholds[name])
        for name, values in (("benchmark_ta", benchmark), ("answer_dynamic", answer_dynamic))
    }
    choices = []
    for weights in mixture_grid():
        if weights["candidate_weight"] > 0.0:
            continue
        probability = mix_predictions(benchmark, answer_dynamic, benchmark, weights)
        threshold = choose_threshold(y, probability)
        metrics = classification_metrics(y, probability, threshold)
        admissible, checks = noninferior_to_both(metrics, anchor_metrics)
        choices.append({"weights": weights, "threshold": threshold, "metrics": metrics, "admissible": admissible, "checks": checks})
    admissible = [item for item in choices if item["admissible"]]
    if not admissible:
        return {
            "status": "no_single_anchor_policy_passes_both",
            "weights": {"benchmark_weight": 1.0, "answer_dynamic_weight": 0.0, "candidate_weight": 0.0},
            "threshold": thresholds["benchmark_ta"],
            "choices": choices,
        }
    selected = max(
        admissible,
        key=lambda item: (
            min(item["checks"][name]["delta_weighted_f1"] for name in ANCHOR_NAMES),
            min(item["checks"][name]["delta_auroc"] for name in ANCHOR_NAMES),
            -max(item["checks"][name]["delta_brier"] for name in ANCHOR_NAMES),
        ),
    )
    return {"status": "anchor_only_policy_passes_both", "selected": selected, "choices": choices}


def paired_bootstrap_delta(
    y: np.ndarray,
    candidate: np.ndarray,
    candidate_threshold: float,
    anchor: np.ndarray,
    anchor_threshold: float,
    seed: int,
) -> dict[str, Any]:
    """Participant-level paired bootstrap for a locked comparison."""
    y = np.asarray(y, dtype=np.int64)
    candidate = np.asarray(candidate, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if len(y) < 2:
        return {"n": int(len(y)), "samples": 0, "status": "too_few_participants"}
    rng = np.random.default_rng(int(seed))
    values: dict[str, list[float]] = {"weighted_f1": [], "macro_f1": [], "positive_f1": [], "auroc": [], "brier": []}
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, len(y), size=len(y))
        y_boot = y[indices]
        candidate_metrics = classification_metrics(y_boot, candidate[indices], candidate_threshold)
        anchor_metrics = classification_metrics(y_boot, anchor[indices], anchor_threshold)
        values["weighted_f1"].append(candidate_metrics["weighted_f1"] - anchor_metrics["weighted_f1"])
        values["macro_f1"].append(candidate_metrics["macro_f1"] - anchor_metrics["macro_f1"])
        values["positive_f1"].append(candidate_metrics["positive_f1"] - anchor_metrics["positive_f1"])
        values["brier"].append(candidate_metrics["brier"] - anchor_metrics["brier"])
        if "auroc" in candidate_metrics and "auroc" in anchor_metrics:
            values["auroc"].append(candidate_metrics["auroc"] - anchor_metrics["auroc"])
    output: dict[str, Any] = {"n": int(len(y)), "samples": int(BOOTSTRAP_SAMPLES), "seed": int(seed)}
    for name, samples in values.items():
        if not samples:
            continue
        array = np.asarray(samples, dtype=np.float64)
        output[name] = {
            "mean": float(np.mean(array)),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
        }
    return output


def aggregate_metric_blocks(items: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    names = sorted({name for item in items for name in item})
    for name in names:
        blocks = [item[name] for item in items if name in item]
        keys = sorted({key for block in blocks for key in block if key not in ("n", "threshold")})
        output[name] = {
            key: {
                "mean": float(np.mean([block[key] for block in blocks])),
                "std": float(np.std([block[key] for block in blocks], ddof=1)) if len(blocks) > 1 else 0.0,
            }
            for key in keys
        }
    return output


def final_seed_selection(runs: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    selected = []
    for run in runs:
        detail = run["safe_rule"]["candidate_details"].get(candidate)
        if detail and detail["status"] == "dual_anchor_admitted_within_seed":
            selected.append(detail["selected"])
    if len(selected) < MIN_NONNEGATIVE_SEEDS:
        return {
            "status": "fallback_benchmark_insufficient_seed_support",
            "supporting_seeds": len(selected),
            "required_seeds": MIN_NONNEGATIVE_SEEDS,
            "weights": {"benchmark_weight": 1.0, "answer_dynamic_weight": 0.0, "candidate_weight": 0.0},
        }
    keys = ("benchmark_weight", "answer_dynamic_weight", "candidate_weight")
    weights = {key: float(np.median([item["weights"][key] for item in selected])) for key in keys}
    total = sum(weights.values())
    weights = {key: value / total for key, value in weights.items()}
    threshold = float(np.median([item["threshold"] for item in selected]))
    return {
        "status": "admitted_after_seed_stability",
        "supporting_seeds": len(selected),
        "required_seeds": MIN_NONNEGATIVE_SEEDS,
        "weights": weights,
        "threshold": threshold,
    }


def safe_rule(train: list[dict[str, Any]], meld_train: np.ndarray, seed: int) -> dict[str, Any]:
    outer_predictions = {name: np.zeros(len(train), dtype=np.float64) for name in VARIANT_NAMES}
    for fold, (fit_idx, held_idx) in enumerate(
        StratifiedKFold(4, shuffle=True, random_state=seed).split(np.zeros(len(train)), [r["label"] for r in train])
    ):
        fit_rows = [train[int(i)] for i in fit_idx]
        held_rows = [train[int(i)] for i in held_idx]
        predictions, _ = fit_all_variants(fit_rows, held_rows, meld_train, None, seed + fold * 101)
        for name, value in predictions.items():
            outer_predictions[name][held_idx] = value
    y = np.asarray([record["label"] for record in train], dtype=np.int64)
    thresholds = {name: choose_threshold(y, outer_predictions[name]) for name in outer_predictions}
    anchor_metrics = {
        name: classification_metrics(y, outer_predictions[name], thresholds[name])
        for name in ANCHOR_NAMES
    }
    grid = mixture_grid()
    candidate_details: dict[str, Any] = {}
    for candidate in VARIANT_NAMES:
        if candidate in ANCHOR_NAMES or candidate not in outer_predictions:
            continue
        choices = []
        for weights in grid:
            probability = mix_predictions(
                outer_predictions["benchmark_ta"],
                outer_predictions["answer_dynamic"],
                outer_predictions[candidate],
                weights,
            )
            threshold = choose_threshold(y, probability)
            metrics = classification_metrics(y, probability, threshold)
            admissible, checks = noninferior_to_both(metrics, anchor_metrics)
            choices.append({
                "weights": weights,
                "threshold": threshold,
                "metrics": metrics,
                "admissible": admissible,
                "checks": checks,
            })
        admissible = [item for item in choices if item["admissible"] and item["weights"]["candidate_weight"] > 0.0]
        if not admissible:
            candidate_details[candidate] = {
                "status": "no_dual_anchor_admission",
                "selected": {
                    "weights": {"benchmark_weight": 1.0, "answer_dynamic_weight": 0.0, "candidate_weight": 0.0},
                    "threshold": thresholds["benchmark_ta"],
                    "fallback": "benchmark_ta",
                },
                "choices": choices,
            }
            continue
        selected = max(
            admissible,
            key=lambda item: (
                min(item["checks"][name]["delta_weighted_f1"] for name in ANCHOR_NAMES),
                min(item["checks"][name]["delta_auroc"] for name in ANCHOR_NAMES),
                -max(item["checks"][name]["delta_brier"] for name in ANCHOR_NAMES),
                item["weights"]["candidate_weight"],
                -item["threshold"],
            ),
        )
        candidate_details[candidate] = {
            "status": "dual_anchor_admitted_within_seed",
            "selected": {
                "weights": selected["weights"],
                "threshold": selected["threshold"],
                "fallback": "benchmark_ta",
            },
            "choices": choices,
        }
    return {
        "anchor_metrics": anchor_metrics,
        "thresholds": thresholds,
        "candidate_details": candidate_details,
        "grid_size": len(grid),
        "constraints": {
            "weighted_f1_margin": F1_MARGIN,
            "auroc_margin": AUROC_MARGIN,
            "brier_margin": BRIER_MARGIN,
            "positive_candidate_mass_required": True,
        },
    }


def evaluate_dataset(args: argparse.Namespace, dataset: str, target: str, meld_train: np.ndarray) -> dict[str, Any]:
    train = load_records(dataset, target, "train", args.adyme_root, args.benchmark_root, args.contract_dir)
    dev = load_records(dataset, target, "dev", args.adyme_root, args.benchmark_root, args.contract_dir)
    profile = "depression_phq8_screening" if target == "depression" else "ptsd_dsm_iv_binary"
    task_dir = args.output / dataset / target
    lock = {
        "schema": SCHEMA,
        "dataset": dataset,
        "target": target,
        "label_profile": profile,
        "split_unit": "participant",
        "canonical_split": True,
        "seeds": list(SEEDS),
        "test_opened": False,
        "anchors": list(ANCHOR_NAMES),
        "dual_anchor_noninferiority": {
            "weighted_f1_margin": F1_MARGIN,
            "auroc_margin": AUROC_MARGIN,
            "brier_margin": BRIER_MARGIN,
            "min_supporting_seeds": MIN_NONNEGATIVE_SEEDS,
        },
        "test_selection_forbidden": True,
        "raw_inputs_read_only": True,
    }
    atomic_json(task_dir / "LOCKED_PROTOCOL.json", lock)
    atomic_json(task_dir / "LOCK_COMPLETE.json", {"status": "locked_before_test_read", "schema": SCHEMA})

    runs: list[dict[str, Any]] = []
    y_train = np.asarray([record["label"] for record in train], dtype=np.int64)
    for seed in SEEDS:
        seed_path = task_dir / f"seed_{seed}.json"
        if seed_path.is_file():
            try:
                cached_run = json.loads(seed_path.read_text(encoding="utf-8"))
                required = ("seed", "oof_metrics", "protected_metrics", "oof_predictions", "oof_thresholds", "fold_rules", "seed_support")
                if int(cached_run.get("seed")) == int(seed) and all(key in cached_run for key in required):
                    runs.append(cached_run)
                    continue
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        oof = {name: np.zeros(len(train), dtype=np.float64) for name in VARIANT_NAMES}
        thresholds = {name: np.full(len(train), 0.5, dtype=np.float64) for name in VARIANT_NAMES}
        severity_oof: dict[str, np.ndarray] = {}
        fold_rules: list[dict[str, Any]] = []
        protected_oof = {name: np.zeros(len(train), dtype=np.float64) for name in VARIANT_NAMES if name not in ANCHOR_NAMES}
        protected_thresholds = {name: np.full(len(train), 0.5, dtype=np.float64) for name in protected_oof}
        for fold, (fit_idx, held_idx) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(train)), y_train)
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
                    predictions["benchmark_ta"],
                    predictions["answer_dynamic"],
                    predictions[candidate],
                    selected["weights"],
                )
                protected_thresholds[candidate][held_idx] = selected["threshold"]
            fold_rules.append(rule)
        metrics = {
            name: classification_metrics(y_train, oof[name], float(np.median(thresholds[name])))
            for name in oof
        }
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
                name: severity_metrics(score_array(train)[np.isfinite(score_array(train))], values[np.isfinite(score_array(train))])
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
        weights = {
            key: float(np.median([item["weights"][key] for item in selected_items]))
            for key in ("benchmark_weight", "answer_dynamic_weight", "candidate_weight")
        }
        total = sum(weights.values())
        weights = {key: value / total for key, value in weights.items()}
        final_policies[candidate] = {
            "status": "admitted_after_seed_stability",
            "supporting_seeds": len(supported),
            "required_seeds": MIN_NONNEGATIVE_SEEDS,
            "weights": weights,
            "threshold": float(np.median([item["threshold"] for item in selected_items])),
        }

    # Recheck the exact median policy on every seed's complete training OOF.
    # Fold-wise admissibility alone is insufficient for this final policy.
    for candidate, policy in list(final_policies.items()):
        if policy["status"] != "admitted_after_seed_stability":
            continue
        policy_support = 0
        policy_checks: list[dict[str, Any]] = []
        for run in runs:
            oof = {name: np.asarray(values, dtype=np.float64) for name, values in run["oof_predictions"].items()}
            y = y_train
            probability = mix_predictions(oof["benchmark_ta"], oof["answer_dynamic"], oof[candidate], policy["weights"])
            metrics = classification_metrics(y, probability, policy["threshold"])
            admissible, checks = noninferior_to_both(
                metrics,
                {name: run["oof_metrics"][name] for name in ANCHOR_NAMES},
            )
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

    anchor_policy = choose_anchor_only_policy(
        runs[0]["oof_predictions"]["benchmark_ta"],
        runs[0]["oof_predictions"]["answer_dynamic"],
        y_train,
        {name: runs[0]["oof_metrics"][name]["threshold"] for name in ANCHOR_NAMES},
    )

    # The test split is opened only after the training-only policy is fully written.
    test = load_records(dataset, target, "test", args.adyme_root, args.benchmark_root, args.contract_dir) if dataset == "edaic" else []
    fit_rows = train if dataset == "daic" else train + dev
    held_rows = dev if dataset == "daic" else test
    y_held = np.asarray([record["label"] for record in held_rows], dtype=np.int64)
    locked_runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        predictions, configs = fit_all_variants(fit_rows, held_rows, meld_train, None, seed + 80000)
        raw = {
            name: classification_metrics(y_held, probability, configs[name]["threshold"])
            for name, probability in predictions.items()
        }
        held_scores = score_array(held_rows)
        held_score_mask = np.isfinite(held_scores)
        raw_regression = {
            name: severity_metrics(
                held_scores[held_score_mask],
                values[held_score_mask],
            )
            for name, values in configs.get("severity_predictions", {}).items()
            if int(held_score_mask.sum()) >= 2
        }
        protected: dict[str, Any] = {}
        bootstrap: dict[str, Any] = {}
        for candidate, policy in final_policies.items():
            probability = mix_predictions(
                predictions["benchmark_ta"],
                predictions["answer_dynamic"],
                predictions[candidate],
                policy["weights"],
            )
            protected[candidate] = classification_metrics(y_held, probability, policy["threshold"])
            bootstrap[candidate] = {
                anchor: paired_bootstrap_delta(
                    y_held,
                    probability,
                    policy["threshold"],
                    predictions[anchor],
                    configs[anchor]["threshold"],
                    seed * 1000 + index,
                )
                for index, anchor in enumerate(ANCHOR_NAMES)
            }
        controls: dict[str, Any] = {}
        for control in CONTROL_NAMES:
            control_predictions, control_configs = fit_all_variants(
                fit_rows,
                held_rows,
                meld_train,
                control,
                seed + 80000,
                held_control_only=True,
            )
            controls[control] = {
                name: classification_metrics(
                    y_held,
                    probability,
                    control_configs[name]["threshold"],
                )
                for name, probability in control_predictions.items()
            }
        locked_runs.append({
            "seed": seed,
            "raw": raw,
            "raw_regression": raw_regression,
            "protected": protected,
            "paired_bootstrap": bootstrap,
            "controls": controls,
            "policies": final_policies,
        })

    return {
        "inventory": {"train": len(train), "dev": len(dev), "test": len(test)},
        "cache_signature_audit": cache_signature_audit(args.adyme_root, (dataset,)),
        "oof_train": runs,
        "final_policies": final_policies,
        "anchor_only_policy": anchor_policy,
        "locked": locked_runs,
        "locked_summary": {
            "raw": aggregate_metric_blocks([item["raw"] for item in locked_runs]),
            "protected": aggregate_metric_blocks([item["protected"] for item in locked_runs]),
            "raw_regression": aggregate_metric_blocks([item["raw_regression"] for item in locked_runs]),
            "controls": {
                control: aggregate_metric_blocks(
                    [item["controls"][control] for item in locked_runs]
                )
                for control in CONTROL_NAMES
            },
        },
        "protocol": {
            "schema": SCHEMA,
            "split_unit": "participant",
            "canonical_split": True,
            "seeds": list(SEEDS),
            "outer_folds": 5,
            "safe_weight_simplex_grid": len(mixture_grid()),
            "safe_weight_zero_is_exact_benchmark_fallback": True,
            "dual_anchor_noninferiority": True,
            "anchor_names": list(ANCHOR_NAMES),
            "test_opened_after_train_lock": dataset == "edaic",
            "raw_inputs_read_only": True,
            "MELD_labels_used": False,
            "mechanism_controls": {
                "names": list(CONTROL_NAMES),
                "training_representation": "clean",
                "held_representation": "controlled",
                "model_and_threshold_refit_with_same_clean_training_protocol": True,
                "test_selection_forbidden": True,
            },
            "historical_answer_dynamic_not_recomputed_from_new_prompt": True,
            "final_model_naming": "A-DyME only when a positive-candidate policy passes both anchors; otherwise report Benchmark and Answer-Dynamic separately",
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
    meld_train = load_meld_train(args.adyme_root)
    payload = {"schema": SCHEMA, "status": "running", "datasets": {}}
    payload["cache_signature_audit"] = cache_signature_audit(args.adyme_root, ("daic", "edaic", "meld"))
    atomic_json(args.output / "PROGRESS.json", payload)
    tasks = (("daic", "depression"), ("edaic", "depression"), ("edaic", "ptsd"))
    for dataset, target in tasks:
        payload["datasets"][f"{dataset}/{target}"] = evaluate_dataset(args, dataset, target, meld_train)
        atomic_json(args.output / "PROGRESS.json", payload)
    payload["status"] = "complete"
    atomic_json(args.output / "FINAL_METRICS.json", payload)
    print(json.dumps({"event": "adyme_mh_final_gate_v2_complete", "output": str(args.output / "FINAL_METRICS.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
