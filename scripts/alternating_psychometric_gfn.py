#!/usr/bin/env python3
"""Alternating psychometric operator and clinical evidence GFlowNet utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scripts import run_adyme_mh_final_gate_v2 as gate
from scripts.meter_gfn import EvidenceTrajectory, random_matched_mask
from scripts.meter_mh_online_operator import VIEW_NAMES


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
LOGISTIC_C = (0.01, 0.1, 1.0, 10.0)


def _summary(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(value, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if len(selected) == 0:
        return np.zeros(value.shape[1] * 8, dtype=np.float64)
    return gate.summarize(selected)


def evidence_features(
    projected: Mapping[str, np.ndarray], evidence_mask: np.ndarray
) -> np.ndarray:
    """Fixed-dimensional multimodal measurement features for an evidence graph."""
    mask = np.asarray(evidence_mask, dtype=bool)
    length = len(projected["answer"])
    if mask.shape != (length, len(VIEW_NAMES)):
        raise ValueError(f"invalid evidence mask: {mask.shape}")
    blocks = []
    for index, name in enumerate(VIEW_NAMES):
        blocks.append(_summary(projected[name], mask[:, index]))
    pairs = (("clinical", "answer"), ("clinical", "context"), ("answer", "context"))
    for left, right in pairs:
        left_index = VIEW_NAMES.index(left)
        right_index = VIEW_NAMES.index(right)
        joint = mask[:, left_index] & mask[:, right_index]
        blocks.append(_summary(projected[left] - projected[right], joint))
    modality_fraction = mask.mean(axis=0).astype(np.float64)
    event_fraction = np.asarray([mask.any(axis=1).mean()], dtype=np.float64)
    modality_count = np.asarray([mask.sum(axis=1).mean() / len(VIEW_NAMES)], dtype=np.float64)
    return gate.finite(np.concatenate((*blocks, modality_fraction, event_fraction, modality_count)))


def full_mask(length: int) -> np.ndarray:
    return np.ones((int(length), len(VIEW_NAMES)), dtype=bool)


def complement(mask: np.ndarray) -> np.ndarray:
    value = ~np.asarray(mask, dtype=bool)
    if not value.any():
        value[:] = True
    return value


def random_sparse_mask(length: int, minimum: int, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    count = int(rng.integers(minimum, maximum + 1))
    events = rng.choice(length, size=min(length, count), replace=False)
    mask = np.zeros((length, len(VIEW_NAMES)), dtype=bool)
    for event in events:
        code = int(rng.integers(1, 2 ** len(VIEW_NAMES)))
        mask[event] = [bool(code & (1 << index)) for index in range(len(VIEW_NAMES))]
    return mask


def concordance_correlation(target: np.ndarray, predicted: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    covariance = float(np.mean((target - target.mean()) * (predicted - predicted.mean())))
    denominator = float(target.var() + predicted.var() + (target.mean() - predicted.mean()) ** 2)
    return float(2.0 * covariance / denominator) if denominator > 1e-12 else 0.0


@dataclass
class LinearProjection:
    scaler: StandardScaler
    reducer: PCA

    def transform(self, values: np.ndarray) -> np.ndarray:
        return gate.finite(self.reducer.transform(self.scaler.transform(gate.finite(values))))


def fit_projection(values: np.ndarray, seed: int, components: int) -> LinearProjection:
    values = gate.finite(values)
    scaler = StandardScaler().fit(values)
    count = max(1, min(int(components), values.shape[1], values.shape[0] - 1))
    reducer = PCA(
        n_components=count,
        whiten=True,
        svd_solver="randomized",
        random_state=int(seed),
    ).fit(scaler.transform(values))
    return LinearProjection(scaler, reducer)


@dataclass
class AlternatingPsychometricOperator:
    mode: str
    projection: LinearProjection
    severity_model: Ridge | None
    score_scaler: StandardScaler | None
    calibrator: LogisticRegression
    score_max: float
    selected_hyperparameter: float

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        transformed = self.projection.transform(np.atleast_2d(features))
        if self.mode == "pim":
            if self.severity_model is None or self.score_scaler is None:
                raise RuntimeError("incomplete PIM operator")
            severity = self.severity_model.predict(transformed).astype(np.float64)
            probability = self.calibrator.predict_proba(
                self.score_scaler.transform(severity[:, None])
            )[:, 1]
        elif self.mode == "direct":
            severity = np.full(len(transformed), np.nan, dtype=np.float64)
            probability = self.calibrator.predict_proba(transformed)[:, 1]
        else:
            raise ValueError(self.mode)
        return {
            "probability": np.clip(probability.astype(np.float64), 1e-6, 1.0 - 1e-6),
            "severity": np.clip(severity, 0.0, self.score_max),
        }


def _group_folds(
    groups: np.ndarray,
    group_targets: Mapping[Any, int],
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    targets = np.asarray([int(group_targets[value]) for value in unique], dtype=np.int64)
    counts = np.bincount(targets, minlength=2)
    nonzero = counts[counts > 0]
    if len(nonzero) < 2 or int(nonzero.min()) < 2:
        raise RuntimeError(f"insufficient classes for grouped OOF: {counts.tolist()}")
    folds = min(4, int(nonzero.min()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    result = []
    for fit_groups, held_groups in splitter.split(unique, targets):
        fit_values = set(unique[fit_groups].tolist())
        held_values = set(unique[held_groups].tolist())
        fit_index = np.asarray([index for index, value in enumerate(groups) if value in fit_values])
        held_index = np.asarray([index for index, value in enumerate(groups) if value in held_values])
        result.append((fit_index, held_index))
    return result


def _validate_oof(
    values: np.ndarray,
    expected: np.ndarray,
    labels: np.ndarray,
    context: str,
) -> np.ndarray:
    valid = np.asarray(expected, dtype=bool) & np.isfinite(values)
    if int(valid.sum()) != int(np.asarray(expected, dtype=bool).sum()):
        raise RuntimeError(
            f"incomplete {context} OOF coverage: {int(valid.sum())}/{int(np.asarray(expected, dtype=bool).sum())}"
        )
    if len(np.unique(np.asarray(labels)[valid])) < 2:
        raise RuntimeError(f"single-class {context} OOF calibration")
    return valid


def fit_pim_operator(
    augmented_features: np.ndarray,
    augmented_scores: np.ndarray,
    augmented_groups: np.ndarray,
    full_features: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    participant_ids: np.ndarray,
    *,
    score_max: float,
    seed: int,
    components: int,
) -> AlternatingPsychometricOperator:
    observed = np.isfinite(augmented_scores)
    augmented_features = augmented_features[observed]
    augmented_scores = augmented_scores[observed]
    augmented_groups = augmented_groups[observed]
    participant_observed = np.isfinite(scores)
    if participant_observed.sum() < 8:
        raise RuntimeError("insufficient observed continuous scores")
    group_targets = {pid: int(label) for pid, label in zip(participant_ids, labels)}
    folds = _group_folds(augmented_groups, group_targets, seed)
    choices = []
    for alpha in RIDGE_ALPHAS:
        oof = np.full(len(participant_ids), np.nan, dtype=np.float64)
        for fold, (fit_index, _) in enumerate(folds):
            fit_groups = set(augmented_groups[fit_index].tolist())
            held_participant = np.asarray(
                [pid not in fit_groups and participant_observed[index] for index, pid in enumerate(participant_ids)],
                dtype=bool,
            )
            if not held_participant.any():
                continue
            projection = fit_projection(augmented_features[fit_index], seed + fold * 101, components)
            model = Ridge(alpha=float(alpha)).fit(
                projection.transform(augmented_features[fit_index]), augmented_scores[fit_index]
            )
            oof[held_participant] = model.predict(projection.transform(full_features[held_participant]))
        valid = _validate_oof(oof, participant_observed, labels, "PIM")
        ccc = concordance_correlation(scores[valid], oof[valid])
        mae = float(np.mean(np.abs(scores[valid] - oof[valid])))
        choices.append((ccc, -mae / max(score_max, 1.0), -float(alpha), float(alpha), oof))
    best = max(choices, key=lambda item: item[:3])
    alpha = float(best[3])
    oof = np.asarray(best[4], dtype=np.float64)
    calibrator_mask = _validate_oof(oof, participant_observed, labels, "PIM")
    score_scaler = StandardScaler().fit(oof[calibrator_mask, None])
    calibrator = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=int(seed + 900),
    ).fit(score_scaler.transform(oof[calibrator_mask, None]), labels[calibrator_mask])
    projection = fit_projection(augmented_features, seed + 1000, components)
    severity_model = Ridge(alpha=alpha).fit(
        projection.transform(augmented_features), augmented_scores
    )
    return AlternatingPsychometricOperator(
        mode="pim",
        projection=projection,
        severity_model=severity_model,
        score_scaler=score_scaler,
        calibrator=calibrator,
        score_max=float(score_max),
        selected_hyperparameter=alpha,
    )


def fit_direct_operator(
    augmented_features: np.ndarray,
    augmented_labels: np.ndarray,
    augmented_groups: np.ndarray,
    full_features: np.ndarray,
    labels: np.ndarray,
    participant_ids: np.ndarray,
    *,
    score_max: float,
    seed: int,
    components: int,
) -> AlternatingPsychometricOperator:
    group_targets = {pid: int(label) for pid, label in zip(participant_ids, labels)}
    folds = _group_folds(augmented_groups, group_targets, seed)
    choices = []
    for c_value in LOGISTIC_C:
        oof = np.full(len(participant_ids), np.nan, dtype=np.float64)
        for fold, (fit_index, _) in enumerate(folds):
            fit_groups = set(augmented_groups[fit_index].tolist())
            held = np.asarray([pid not in fit_groups for pid in participant_ids], dtype=bool)
            if not held.any():
                continue
            projection = fit_projection(augmented_features[fit_index], seed + fold * 101, components)
            model = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=int(seed + fold * 101),
            ).fit(projection.transform(augmented_features[fit_index]), augmented_labels[fit_index])
            oof[held] = model.predict_proba(projection.transform(full_features[held]))[:, 1]
        valid = _validate_oof(oof, np.ones(len(oof), dtype=bool), labels, "direct")
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(labels[valid], oof[valid]))
        brier = float(np.mean(np.square(oof[valid] - labels[valid])))
        choices.append((auroc, -brier, -float(c_value), float(c_value)))
    c_value = float(max(choices)[3])
    projection = fit_projection(augmented_features, seed + 1000, components)
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=int(seed + 1000),
    ).fit(projection.transform(augmented_features), augmented_labels)
    return AlternatingPsychometricOperator(
        mode="direct",
        projection=projection,
        severity_model=None,
        score_scaler=None,
        calibrator=classifier,
        score_max=float(score_max),
        selected_hyperparameter=c_value,
    )


def operator_log_reward(
    operator: AlternatingPsychometricOperator,
    projected: Mapping[str, np.ndarray],
    trajectory: EvidenceTrajectory,
    *,
    label: int,
    score: float | None,
    random_seed: int,
    pim_weight: float,
    ensr_weight: float,
    sufficiency_weight: float,
    compactness_weight: float,
    necessity_margin: float,
    specificity_margin: float,
    temperature: float,
    path_process_weight: float = 0.0,
    path_monotonic_weight: float = 0.0,
    path_order_weight: float = 0.0,
    path_order_margin: float = 0.0,
    path_duplicate_weight: float = 0.0,
) -> dict[str, float]:
    return operator_log_rewards(
        operator,
        projected,
        [trajectory],
        label=label,
        score=score,
        random_seeds=[random_seed],
        pim_weight=pim_weight,
        ensr_weight=ensr_weight,
        sufficiency_weight=sufficiency_weight,
        compactness_weight=compactness_weight,
        necessity_margin=necessity_margin,
        specificity_margin=specificity_margin,
        temperature=temperature,
        path_process_weight=path_process_weight,
        path_monotonic_weight=path_monotonic_weight,
        path_order_weight=path_order_weight,
        path_order_margin=path_order_margin,
        path_duplicate_weight=path_duplicate_weight,
    )[0]


def operator_log_rewards(
    operator: AlternatingPsychometricOperator,
    projected: Mapping[str, np.ndarray],
    trajectories: list[EvidenceTrajectory],
    *,
    label: int,
    score: float | None,
    random_seeds: list[int],
    pim_weight: float,
    ensr_weight: float,
    sufficiency_weight: float,
    compactness_weight: float,
    necessity_margin: float,
    specificity_margin: float,
    temperature: float,
    path_process_weight: float = 0.0,
    path_monotonic_weight: float = 0.0,
    path_order_weight: float = 0.0,
    path_order_margin: float = 0.0,
    path_duplicate_weight: float = 0.0,
) -> list[dict[str, float]]:
    if len(trajectories) != len(random_seeds):
        raise ValueError("trajectory/random-seed count mismatch")
    if not trajectories:
        return []
    masks = [trajectory.mask.detach().cpu().numpy().astype(bool) for trajectory in trajectories]
    random_masks = [
        random_matched_mask(trajectory.mask, random_seed).cpu().numpy().astype(bool)
        for trajectory, random_seed in zip(trajectories, random_seeds)
    ]
    full = operator.predict(evidence_features(projected, full_mask(len(masks[0]))))
    selected = operator.predict(np.stack([evidence_features(projected, mask) for mask in masks]))
    selected_delete = operator.predict(
        np.stack([evidence_features(projected, complement(mask)) for mask in masks])
    )
    random_delete = operator.predict(
        np.stack([evidence_features(projected, complement(mask)) for mask in random_masks])
    )

    def ce(probability: float) -> float:
        probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
        return float(-(label * np.log(probability) + (1 - label) * np.log(1.0 - probability)))

    full_ce = ce(float(full["probability"][0]))
    selected_ce_values = np.asarray(
        [ce(float(value)) for value in selected["probability"]], dtype=np.float64
    )
    selected_delete_costs = np.asarray(
        [ce(float(value)) - full_ce for value in selected_delete["probability"]],
        dtype=np.float64,
    )
    random_delete_costs = np.asarray(
        [ce(float(value)) - full_ce for value in random_delete["probability"]],
        dtype=np.float64,
    )
    deletion_gaps = selected_delete_costs - random_delete_costs
    robust_values = np.abs(
        np.concatenate((selected_delete_costs, random_delete_costs, deletion_gaps))
    )
    positive_scale = robust_values[robust_values > 1e-8]
    ensr_scale = max(
        float(np.median(positive_scale)) if len(positive_scale) else 0.0,
        1e-4,
    )

    def softplus(value: float) -> float:
        return float(np.logaddexp(0.0, float(value)))

    def ordered_prefix_masks(trajectory: EvidenceTrajectory, reverse: bool = False) -> list[np.ndarray]:
        order = getattr(trajectory, "event_order", None)
        if order is None:
            raise ValueError("path-process reward requires trajectory.event_order")
        events = order.detach().cpu().tolist()
        if reverse:
            events = list(reversed(events))
        terminal = trajectory.mask.detach().cpu().numpy().astype(bool)
        prefix = np.zeros_like(terminal)
        result = []
        for event in events:
            prefix[int(event)] = terminal[int(event)]
            result.append(prefix.copy())
        if not result:
            result.append(terminal.copy())
        return result

    def path_losses(trajectory: EvidenceTrajectory, reverse: bool = False) -> np.ndarray:
        prefix_masks = ordered_prefix_masks(trajectory, reverse=reverse)
        empty = np.zeros_like(prefix_masks[0])
        prefix_masks = [empty, *prefix_masks]
        prediction = operator.predict(
            np.stack([evidence_features(projected, mask) for mask in prefix_masks])
        )
        classification = np.asarray(
            [ce(float(value)) for value in prediction["probability"]], dtype=np.float64
        )
        if operator.mode != "pim" or score is None or pim_weight <= 0:
            return classification
        severity = np.abs(
            np.asarray(prediction["severity"], dtype=np.float64) - float(score)
        ) / max(operator.score_max, 1.0)
        return classification + float(pim_weight) * severity

    duplicate_penalties = np.zeros(len(masks), dtype=np.float64)
    if path_duplicate_weight > 0 and len(masks) > 1:
        signatures = [mask.tobytes() for mask in masks]
        for index, signature in enumerate(signatures):
            duplicate_penalties[index] = (
                sum(other == signature for other in signatures) - 1
            ) / max(1, len(signatures) - 1)

    rewards = []
    for index, (mask, trajectory) in enumerate(zip(masks, trajectories)):
        selected_ce = float(selected_ce_values[index])
        selected_delete_cost = float(selected_delete_costs[index])
        random_delete_cost = float(random_delete_costs[index])
        normalized_necessity = selected_delete_cost / ensr_scale
        normalized_specificity = (selected_delete_cost - random_delete_cost) / ensr_scale
        necessity = softplus(float(necessity_margin) - normalized_necessity)
        specificity = softplus(float(specificity_margin) - normalized_specificity)
        severity = 0.0
        if operator.mode == "pim" and score is not None and pim_weight > 0:
            severity = abs(float(selected["severity"][index]) - float(score)) / max(operator.score_max, 1.0)
        compactness = float(mask.any(axis=1).mean())
        path_process = 0.0
        path_monotonic = 0.0
        path_order = 0.0
        forward_area = 0.0
        reverse_area = 0.0
        if path_process_weight > 0 or path_monotonic_weight > 0 or path_order_weight > 0:
            forward_losses = path_losses(trajectory, reverse=False)
            reverse_losses = path_losses(trajectory, reverse=True)
            forward_area = float(np.mean(forward_losses))
            reverse_area = float(np.mean(reverse_losses))
            # Reward incremental clinical improvement, while keeping the
            # terminal measurement term dominant. Positive gains reduce the
            # path cost. The empty-evidence state is included so that the
            # first acquired event is evaluated as well.
            gains = forward_losses[:-1] - forward_losses[1:]
            if len(gains):
                discount = np.power(0.85, np.arange(len(gains), dtype=np.float64))
                path_process = -float(np.sum(discount * gains) / np.sum(discount))
            else:
                path_process = 0.0
            if len(forward_losses) > 1:
                path_monotonic = float(np.maximum(np.diff(forward_losses), 0.0).mean())
            path_order = softplus(
                float(path_order_margin) - (reverse_area - forward_area)
            )
        total = (
            float(sufficiency_weight) * selected_ce
            + float(pim_weight) * severity
            + float(ensr_weight) * (necessity + specificity)
            + float(compactness_weight) * compactness
            + float(path_process_weight) * path_process
            + float(path_monotonic_weight) * path_monotonic
            + float(path_order_weight) * path_order
            + float(path_duplicate_weight) * float(duplicate_penalties[index])
        )
        rewards.append(
            {
                "log_reward": -total / max(float(temperature), 1e-6),
                "classification": selected_ce,
                "severity": severity,
                "necessity": necessity,
                "specificity": specificity,
                "compactness": compactness,
                "path_process": path_process,
                "path_monotonic": path_monotonic,
                "path_order": path_order,
                "path_duplicate": float(duplicate_penalties[index]),
                "forward_path_loss": forward_area,
                "reverse_path_loss": reverse_area,
                "selected_delete_cost": selected_delete_cost,
                "random_delete_cost": random_delete_cost,
                "ensr_scale": ensr_scale,
                "normalized_necessity": normalized_necessity,
                "normalized_specificity": normalized_specificity,
            }
        )
    return rewards
