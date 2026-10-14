#!/usr/bin/env python3
"""Differentiable online realization of the frozen METER-MH measurement path.

The module converts fitted scikit-learn preprocessing and estimators into a
pure PyTorch graph.  It intentionally contains no randomly initialized
replacement measurement head: at initialization, its probability and
continuous-severity outputs must reproduce the cached METER-PIM operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


EPSILON = 1e-12
EVIDENCE_FRACTION = 0.25
VIEW_NAMES = ("clinical", "answer", "context", "visual")


def _tensor(value: Any, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=dtype)


class FrozenStandardPCA(nn.Module):
    """Exact affine realization of ``StandardScaler -> PCA.transform``."""

    def __init__(self, scaler: Any, reducer: Any, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        scale = np.where(np.abs(scale) > EPSILON, scale, 1.0)
        self.register_buffer("scaler_mean", _tensor(scaler.mean_, dtype=dtype))
        self.register_buffer("scaler_scale", _tensor(scale, dtype=dtype))
        self.register_buffer("pca_mean", _tensor(reducer.mean_, dtype=dtype))
        self.register_buffer("components", _tensor(reducer.components_, dtype=dtype))
        if bool(getattr(reducer, "whiten", False)):
            whitening = np.sqrt(np.asarray(reducer.explained_variance_, dtype=np.float64))
            whitening = np.where(np.abs(whitening) > EPSILON, whitening, 1.0)
        else:
            whitening = np.ones(len(reducer.components_), dtype=np.float64)
        self.register_buffer("whitening", _tensor(whitening, dtype=dtype))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.nan_to_num(values.to(dtype=self.scaler_mean.dtype))
        expected = int(self.scaler_mean.numel())
        if values.shape[-1] > expected:
            tail = values[..., expected:]
            if torch.any(tail.abs() > 1e-8):
                raise ValueError(
                    f"non-zero features beyond fitted input width: "
                    f"received={values.shape[-1]} fitted={expected}"
                )
            values = values[..., :expected]
        elif values.shape[-1] < expected:
            raise ValueError(
                f"input narrower than fitted projection: "
                f"received={values.shape[-1]} fitted={expected}"
            )
        scaled = (values - self.scaler_mean) / self.scaler_scale
        projected = (scaled - self.pca_mean) @ self.components.transpose(0, 1)
        return projected / self.whitening


class FrozenLinearPipeline(nn.Module):
    """Frozen ``StandardScaler -> PCA -> linear estimator`` pipeline."""

    def __init__(
        self,
        scaler: Any,
        reducer: Any,
        estimator: Any,
        *,
        logistic: bool,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.projection = FrozenStandardPCA(scaler, reducer, dtype=dtype)
        coefficient = np.asarray(estimator.coef_, dtype=np.float64)
        if coefficient.ndim == 2:
            if coefficient.shape[0] != 1:
                raise ValueError(f"only binary/one-output estimators are supported: {coefficient.shape}")
            coefficient = coefficient[0]
        intercept = np.asarray(estimator.intercept_, dtype=np.float64).reshape(-1)
        if len(intercept) != 1:
            raise ValueError(f"only one intercept is supported: {intercept.shape}")
        self.register_buffer("coefficient", _tensor(coefficient, dtype=dtype))
        self.register_buffer("intercept", _tensor(intercept[0], dtype=dtype))
        self.logistic = bool(logistic)

    def logits(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values) @ self.coefficient + self.intercept

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.logits(values)
        return torch.sigmoid(output) if self.logistic else output


class FrozenScalarCalibrator(nn.Module):
    """Frozen scalar StandardScaler and binary logistic calibrator."""

    def __init__(self, scaler: Any, calibrator: Any, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        scale = float(np.asarray(scaler.scale_).reshape(-1)[0])
        if abs(scale) <= EPSILON:
            scale = 1.0
        coefficient = float(np.asarray(calibrator.coef_).reshape(-1)[0])
        intercept = float(np.asarray(calibrator.intercept_).reshape(-1)[0])
        self.register_buffer("mean", _tensor(float(np.asarray(scaler.mean_).reshape(-1)[0]), dtype=dtype))
        self.register_buffer("scale", _tensor(scale, dtype=dtype))
        self.register_buffer("coefficient", _tensor(coefficient, dtype=dtype))
        self.register_buffer("intercept", _tensor(intercept, dtype=dtype))

    def logits(self, severity: torch.Tensor) -> torch.Tensor:
        standardized = (severity - self.mean) / self.scale
        return standardized * self.coefficient + self.intercept

    def forward(self, severity: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(severity))


class FrozenEvidenceScorer(nn.Module):
    """Participant-normalized event evidence scorer used by ENSR."""

    def __init__(self, scaler: Any, classifier: Any, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        scale = np.where(np.abs(scale) > EPSILON, scale, 1.0)
        coefficient = np.asarray(classifier.coef_, dtype=np.float64)
        if coefficient.shape[0] != 1:
            raise ValueError(f"binary evidence scorer required: {coefficient.shape}")
        self.register_buffer("mean", _tensor(scaler.mean_, dtype=dtype))
        self.register_buffer("scale", _tensor(scale, dtype=dtype))
        self.register_buffer("coefficient", _tensor(coefficient[0], dtype=dtype))
        self.register_buffer("intercept", _tensor(float(classifier.intercept_[0]), dtype=dtype))

    def decision(self, events: torch.Tensor) -> torch.Tensor:
        values = (torch.nan_to_num(events.to(dtype=self.mean.dtype)) - self.mean) / self.scale
        return values @ self.coefficient + self.intercept

    def evidence(self, events: torch.Tensor) -> torch.Tensor:
        return self.decision(events).abs()

    def indices(self, events: torch.Tensor) -> torch.Tensor:
        scores = self.evidence(events)
        count = max(1, int(np.ceil(EVIDENCE_FRACTION * len(events))))
        # Stable argsort matches NumPy's stable ordering used by cached METER.
        return torch.sort(torch.argsort(scores, stable=True)[-count:]).values


def event_matrix(projected: Mapping[str, torch.Tensor]) -> torch.Tensor:
    length = min(len(projected["clinical"]), len(projected["answer"]), len(projected["context"]))
    clinical = projected["clinical"][:length]
    answer = projected["answer"][:length]
    context = projected["context"][:length]
    return torch.cat((clinical, answer, context, clinical - answer, clinical - context), dim=-1)


def summarize(sequence: torch.Tensor) -> torch.Tensor:
    """Differentiable equivalent of the cached eight-statistic sequence summary."""
    value = torch.nan_to_num(sequence)
    if value.ndim != 2 or len(value) == 0:
        raise ValueError(f"invalid event sequence {tuple(value.shape)}")
    position = torch.linspace(-1.0, 1.0, len(value), device=value.device, dtype=value.dtype)
    if len(value) > 1:
        delta = value[1:] - value[:-1]
    else:
        delta = torch.zeros((1, value.shape[1]), device=value.device, dtype=value.dtype)
    denominator = position.square().sum().clamp_min(1e-6)
    slope = (position[:, None] * value).sum(dim=0) / denominator
    quantiles = torch.quantile(value, torch.tensor((0.1, 0.9), device=value.device, dtype=value.dtype), dim=0)
    return torch.cat(
        (
            value.mean(dim=0),
            value.std(dim=0, correction=0),
            quantiles[0],
            quantiles[1],
            slope,
            value[-1] - value[0],
            delta.abs().mean(dim=0),
            delta.square().mean(dim=0).sqrt(),
        )
    )


def summarize_subset(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if len(indices) == 0:
        return torch.zeros_like(summarize(value[:1]))
    return summarize(value[torch.sort(indices).values])


@dataclass(frozen=True)
class CounterfactualViews:
    clean: torch.Tensor
    selected_delete: torch.Tensor
    selected_only: torch.Tensor
    random_delete: torch.Tensor
    selected_indices: torch.Tensor
    random_indices: torch.Tensor


class FrozenMeterPIM(nn.Module):
    """Online frozen METER-PIM graph from event views to risk and severity."""

    def __init__(
        self,
        *,
        shared_projection: FrozenStandardPCA,
        visual_projection: FrozenStandardPCA,
        evidence_scorer: FrozenEvidenceScorer,
        severity_regressor: FrozenLinearPipeline,
        calibrator: FrozenScalarCalibrator,
    ) -> None:
        super().__init__()
        self.shared_projection = shared_projection
        self.visual_projection = visual_projection
        self.evidence_scorer = evidence_scorer
        self.severity_regressor = severity_regressor
        self.calibrator = calibrator

    @classmethod
    def from_sklearn(
        cls,
        *,
        projections: Mapping[str, Any],
        evidence_scorer: tuple[Any, Any],
        severity_bundle: tuple[Any, Any, Any],
        score_scaler: Any,
        calibrator: Any,
        dtype: torch.dtype = torch.float64,
    ) -> "FrozenMeterPIM":
        shared_scaler, shared_pca = projections["clinical"]
        visual_scaler, visual_pca = projections["visual"]
        severity_scaler, severity_pca, severity_model = severity_bundle
        return cls(
            shared_projection=FrozenStandardPCA(shared_scaler, shared_pca, dtype=dtype),
            visual_projection=FrozenStandardPCA(visual_scaler, visual_pca, dtype=dtype),
            evidence_scorer=FrozenEvidenceScorer(*evidence_scorer, dtype=dtype),
            severity_regressor=FrozenLinearPipeline(
                severity_scaler,
                severity_pca,
                severity_model,
                logistic=False,
                dtype=dtype,
            ),
            calibrator=FrozenScalarCalibrator(score_scaler, calibrator, dtype=dtype),
        )

    def project_views(self, raw: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "clinical": self.shared_projection(raw["clinical"]),
            "answer": self.shared_projection(raw["answer"]),
            "context": self.shared_projection(raw["context"]),
            "visual": self.visual_projection(raw["visual"]),
        }

    @staticmethod
    def _apply_keep(projected: Mapping[str, torch.Tensor], keep: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: value * keep[:, None].to(value.dtype) for name, value in projected.items()}

    def meter_v7(self, projected: Mapping[str, torch.Tensor], selected: torch.Tensor | None = None) -> torch.Tensor:
        events = event_matrix(projected)
        if selected is None:
            selected = self.evidence_scorer.indices(events)
        all_indices = torch.arange(len(events), device=events.device)
        selected_mask = torch.zeros(len(events), dtype=torch.bool, device=events.device)
        selected_mask[selected] = True
        complement = all_indices[~selected_mask]

        clinical = projected["clinical"]
        answer = projected["answer"]
        context = projected["context"]
        visual = projected["visual"]
        process = torch.cat(
            (
                summarize(answer),
                summarize(clinical),
                summarize(context),
                summarize(clinical - answer),
                summarize(clinical - context),
            )
        )
        process_visual = torch.cat((process, summarize(visual)))
        selected_blocks = [summarize_subset(projected[name], selected) for name in VIEW_NAMES]
        complement_blocks = [summarize_subset(projected[name], complement) for name in VIEW_NAMES]
        selected_cat = torch.cat(selected_blocks)
        complement_cat = torch.cat(complement_blocks)
        return torch.cat((process_visual, selected_cat, complement_cat, selected_cat - complement_cat))

    def counterfactual_features(
        self,
        raw: Mapping[str, torch.Tensor],
        *,
        random_seed: int = 0,
    ) -> CounterfactualViews:
        projected = self.project_views(raw)
        selected = self.evidence_scorer.indices(event_matrix(projected))
        generator = torch.Generator(device=selected.device)
        generator.manual_seed(int(random_seed))
        random_indices = torch.sort(
            torch.randperm(len(projected["answer"]), generator=generator, device=selected.device)[: len(selected)]
        ).values

        selected_keep = torch.ones(len(projected["answer"]), dtype=torch.bool, device=selected.device)
        selected_keep[selected] = False
        selected_only_keep = ~selected_keep
        random_keep = torch.ones_like(selected_keep)
        random_keep[random_indices] = False
        return CounterfactualViews(
            clean=self.meter_v7(projected, selected),
            selected_delete=self.meter_v7(self._apply_keep(projected, selected_keep), selected),
            selected_only=self.meter_v7(self._apply_keep(projected, selected_only_keep), selected),
            random_delete=self.meter_v7(self._apply_keep(projected, random_keep), selected),
            selected_indices=selected,
            random_indices=random_indices,
        )

    def measure_features(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim == 1:
            features = features[None, :]
        severity = self.severity_regressor(features)
        logits = self.calibrator.logits(severity)
        return {"probability": torch.sigmoid(logits), "logit": logits, "severity": severity}

    def forward(self, raw: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        projected = self.project_views(raw)
        features = self.meter_v7(projected)
        measured = self.measure_features(features)
        return {**measured, "meter_v7": features, "projected": projected}


class BoundedSequentialIncrement(nn.Module):
    """Zero-initialized trainable increment on top of frozen METER-PIM.

    The recurrent branch models context-conditioned answer deviation.  Its
    scalar gates start at zero, so the complete model is exactly the frozen
    METER-PIM operator before optimization.
    """

    def __init__(self, event_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        input_dim = event_dim * 7
        self.expected_answer = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, event_dim),
        )
        self.sequence = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.severity_head = nn.Linear(hidden_dim, 1)
        self.risk_scale = nn.Parameter(torch.zeros(()))
        self.severity_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        projected: Mapping[str, torch.Tensor],
        *,
        answer_dynamic: bool = True,
        order: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        clinical = projected["clinical"]
        answer = projected["answer"]
        context = projected["context"]
        visual = projected["visual"]
        if order is not None:
            if order.ndim != 1 or len(order) != len(answer):
                raise ValueError(
                    f"invalid event order {tuple(order.shape)} for {len(answer)} events"
                )
            clinical = clinical[order]
            answer = answer[order]
            context = context[order]
            visual = visual[order]
        if answer_dynamic:
            expected = self.expected_answer(context)
            deviation = answer - expected
            delta = torch.zeros_like(deviation)
            if len(deviation) > 1:
                delta[1:] = deviation[1:] - deviation[:-1]
        else:
            # Preserve architecture and parameter count while removing all
            # event-to-event variation and order from the increment branch.
            clinical = clinical.mean(dim=0, keepdim=True).expand_as(clinical)
            answer = answer.mean(dim=0, keepdim=True).expand_as(answer)
            context = context.mean(dim=0, keepdim=True).expand_as(context)
            visual = visual.mean(dim=0, keepdim=True).expand_as(visual)
            expected = self.expected_answer(context)
            deviation = torch.zeros_like(answer)
            delta = torch.zeros_like(answer)
        sequence_input = torch.cat(
            (answer, context, clinical, visual, deviation, delta, delta.abs()), dim=-1
        )
        states, _ = self.sequence(sequence_input[None, :, :])
        states = states[0]
        weights = torch.softmax(self.attention(states).squeeze(-1), dim=0)
        pooled = (weights[:, None] * states).sum(dim=0)
        risk = torch.tanh(self.risk_scale) * self.risk_head(pooled).squeeze(-1)
        severity = torch.tanh(self.severity_scale) * self.severity_head(pooled).squeeze(-1)
        return {
            "risk_delta": risk,
            "severity_delta": severity,
            "attention": weights,
            "expected_answer": expected,
            "answer_target": answer,
            "answer_deviation": deviation,
        }


class OnlineSequentialPsychometricModel(nn.Module):
    """Frozen online METER-PIM plus a bounded trainable sequential increment."""

    def __init__(self, anchor: FrozenMeterPIM, increment: BoundedSequentialIncrement) -> None:
        super().__init__()
        self.anchor = anchor
        self.increment = increment
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _counterfactual_keep(
        projected: Mapping[str, torch.Tensor],
        selected: torch.Tensor,
        intervention: str,
        random_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        event_count = len(projected["answer"])
        if intervention == "clean":
            return torch.ones(event_count, dtype=torch.bool, device=selected.device), None
        if intervention == "selected_delete":
            keep = torch.ones(event_count, dtype=torch.bool, device=selected.device)
            keep[selected] = False
            return keep, None
        if intervention == "selected_only":
            keep = torch.zeros(event_count, dtype=torch.bool, device=selected.device)
            keep[selected] = True
            return keep, None
        if intervention == "random_delete":
            generator = torch.Generator(device=selected.device)
            generator.manual_seed(int(random_seed))
            random_indices = torch.sort(
                torch.randperm(
                    event_count,
                    generator=generator,
                    device=selected.device,
                )[: len(selected)]
            ).values
            keep = torch.ones(event_count, dtype=torch.bool, device=selected.device)
            keep[random_indices] = False
            return keep, random_indices
        raise ValueError(f"unknown evidence intervention: {intervention}")

    def forward(
        self,
        raw: Mapping[str, torch.Tensor],
        *,
        answer_dynamic: bool = True,
        order: torch.Tensor | None = None,
        intervention: str = "clean",
        intervention_seed: int = 0,
    ) -> dict[str, torch.Tensor]:
        anchor = self.anchor(raw)
        projected = anchor["projected"]
        selected_indices = None
        random_indices = None
        if intervention != "clean":
            selected_indices = self.anchor.evidence_scorer.indices(event_matrix(projected))
            keep, random_indices = self._counterfactual_keep(
                projected,
                selected_indices,
                intervention,
                intervention_seed,
            )
            projected = self.anchor._apply_keep(projected, keep)
            features = self.anchor.meter_v7(projected, selected_indices)
            measured = self.anchor.measure_features(features)
            anchor = {
                **anchor,
                **measured,
                "meter_v7": features,
                "projected": projected,
            }
        increment = self.increment(
            projected,
            answer_dynamic=answer_dynamic,
            order=order,
        )
        logit = anchor["logit"].squeeze(0) + increment["risk_delta"]
        severity = anchor["severity"].squeeze(0) + increment["severity_delta"]
        return {
            **anchor,
            **increment,
            "online_logit": logit,
            "online_probability": torch.sigmoid(logit),
            "online_severity": severity,
            "intervention": intervention,
            "selected_indices": selected_indices,
            "random_indices": random_indices,
        }
