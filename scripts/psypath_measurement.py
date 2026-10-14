#!/usr/bin/env python3
"""Differentiable psychometric measurement for PsyPath-GFN."""

from __future__ import annotations

from dataclasses import dataclass
import copy

import torch
import torch.nn.functional as F
from torch import nn

from scripts.meter_mh_online_operator import (
    FrozenLinearPipeline,
    FrozenScalarCalibrator,
)


@dataclass(frozen=True)
class PsychometricLossWeights:
    risk: float = 1.0
    severity: float = 1.0
    compactness: float = 0.03
    uncertainty: float = 0.02


class InstrumentConditionedPsychometricMeasurement(nn.Module):
    """Measure severity first and monotonically calibrate it into screening risk."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 96,
        channel_count: int = 4,
        temporal_dim: int = 4,
        score_maxima: tuple[float, ...] = (24.0, 85.0),
    ) -> None:
        super().__init__()
        if channel_count < 1 or len(score_maxima) < 1:
            raise ValueError("channel_count and score_maxima must be non-empty")
        if any(float(value) <= 0 for value in score_maxima):
            raise ValueError("score maxima must be positive")
        self.hidden_dim = int(hidden_dim)
        self.channel_count = int(channel_count)
        self.instrument_count = len(score_maxima)
        self.register_buffer(
            "score_maxima", torch.tensor(score_maxima, dtype=torch.float32)
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.channel_embedding = nn.Parameter(torch.empty(channel_count, hidden_dim))
        nn.init.normal_(self.channel_embedding, std=0.02)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.instrument_embedding = nn.Embedding(
            self.instrument_count, hidden_dim
        )
        self.sequence = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.measurement = nn.Sequential(
            nn.LayerNorm(hidden_dim * 9),
            nn.Linear(hidden_dim * 9, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.severity_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.risk_slope_raw = nn.Parameter(torch.zeros(self.instrument_count))
        self.risk_intercept = nn.Parameter(torch.zeros(self.instrument_count))

    def _instrument(
        self, instrument_id: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= int(instrument_id) < self.instrument_count:
            raise ValueError("instrument_id is outside the configured instruments")
        index = torch.tensor(int(instrument_id), dtype=torch.long, device=device)
        return index, self.instrument_embedding(index)

    def forward(
        self,
        nodes: torch.Tensor,
        temporal: torch.Tensor,
        evidence_mask: torch.Tensor,
        *,
        instrument_id: int,
    ) -> dict[str, torch.Tensor]:
        if nodes.ndim != 3 or nodes.shape[1] != self.channel_count:
            raise ValueError(
                f"nodes must be [events, {self.channel_count}, features], got {tuple(nodes.shape)}"
            )
        if temporal.ndim != 2 or temporal.shape[0] != nodes.shape[0]:
            raise ValueError("temporal features must align with events")
        if evidence_mask.shape != nodes.shape[:2]:
            raise ValueError("evidence_mask must align with event-channel nodes")
        active_events = evidence_mask.any(dim=1)
        if not bool(active_events.any()):
            raise ValueError("measurement requires at least one evidence node")

        index, instrument = self._instrument(instrument_id, nodes.device)
        encoded = self.node_encoder(nodes) + self.channel_embedding[None, :, :]
        weights = evidence_mask.to(encoded.dtype)
        event_count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        events = (encoded * weights[:, :, None]).sum(dim=1) / event_count
        events = events + self.temporal_encoder(temporal.to(events.dtype))
        events = events * active_events[:, None]
        sequence, _ = self.sequence(events[None, :, :])
        sequence = sequence[0]
        instrument_rows = instrument.expand(len(sequence), -1)
        logits = self.attention(torch.cat((sequence, instrument_rows), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~active_events, -torch.inf)
        attention = torch.softmax(logits, dim=0)
        weighted = (attention[:, None] * sequence).sum(dim=0)
        selected = sequence[active_events]
        mean = selected.mean(dim=0)
        std = selected.std(dim=0, correction=0)
        boundary = selected[-1] - selected[0]
        hidden = self.measurement(
            torch.cat((weighted, mean, std, boundary, instrument))
        )
        severity_unit = torch.sigmoid(self.severity_head(hidden).squeeze(-1))
        score_max = self.score_maxima[index].to(dtype=severity_unit.dtype)
        severity = severity_unit * score_max
        slope = F.softplus(self.risk_slope_raw[index]).to(severity_unit.dtype)
        risk_logit = slope * (severity_unit - 0.5) + self.risk_intercept[index]
        log_variance = self.uncertainty_head(hidden).squeeze(-1).clamp(-5.0, 5.0)
        return {
            "risk_logit": risk_logit,
            "probability": torch.sigmoid(risk_logit),
            "severity_unit": severity_unit,
            "severity": severity,
            "log_variance": log_variance,
            "attention": attention,
            "active_fraction": evidence_mask.to(nodes.dtype).mean(),
        }


def psychometric_measurement_loss(
    output: dict[str, torch.Tensor],
    *,
    label: float,
    score: float | None,
    score_max: float,
    weights: PsychometricLossWeights = PsychometricLossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = output["risk_logit"].new_tensor(float(label))
    risk = F.binary_cross_entropy_with_logits(output["risk_logit"], target)
    severity = output["risk_logit"].new_zeros(())
    uncertainty = output["risk_logit"].new_zeros(())
    if score is not None:
        normalized = output["risk_logit"].new_tensor(float(score) / float(score_max))
        error = output["severity_unit"] - normalized
        inverse_variance = torch.exp(-output["log_variance"])
        severity = F.smooth_l1_loss(output["severity_unit"], normalized)
        uncertainty = 0.5 * (
            inverse_variance * error.square() + output["log_variance"]
        )
    compactness = output["active_fraction"]
    total = (
        float(weights.risk) * risk
        + float(weights.severity) * severity
        + float(weights.compactness) * compactness
        + float(weights.uncertainty) * uncertainty
    )
    return total, {
        "risk": risk,
        "severity": severity,
        "compactness": compactness,
        "uncertainty": uncertainty,
    }


def psychometric_log_reward(
    output: dict[str, torch.Tensor],
    *,
    label: float,
    score: float | None,
    score_max: float,
    temperature: float,
    weights: PsychometricLossWeights = PsychometricLossWeights(),
) -> torch.Tensor:
    loss, _ = psychometric_measurement_loss(
        output,
        label=label,
        score=score,
        score_max=score_max,
        weights=weights,
    )
    return -loss.detach() / max(float(temperature), 1e-6)


class FrozenAlternatingPsychometricOperator(nn.Module):
    """Exact PyTorch realization of a fitted cached psychometric operator."""

    def __init__(self, operator: object, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        mode = str(getattr(operator, "mode", ""))
        projection = getattr(operator, "projection")
        calibrator = getattr(operator, "calibrator")
        self.mode = mode
        self.score_max = float(getattr(operator, "score_max"))
        if mode == "pim":
            severity_model = getattr(operator, "severity_model")
            score_scaler = getattr(operator, "score_scaler")
            if severity_model is None or score_scaler is None:
                raise ValueError("incomplete cached PIM operator")
            self.severity_pipeline = FrozenLinearPipeline(
                projection.scaler,
                projection.reducer,
                severity_model,
                logistic=False,
                dtype=dtype,
            )
            self.risk_pipeline = None
            self.calibrator = FrozenScalarCalibrator(
                score_scaler, calibrator, dtype=dtype
            )
        elif mode == "direct":
            self.severity_pipeline = None
            self.risk_pipeline = FrozenLinearPipeline(
                projection.scaler,
                projection.reducer,
                calibrator,
                logistic=True,
                dtype=dtype,
            )
            self.calibrator = None
        else:
            raise ValueError(f"unsupported psychometric operator mode: {mode}")

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim == 1:
            features = features[None, :]
        if self.mode == "pim":
            if self.severity_pipeline is None or self.calibrator is None:
                raise RuntimeError("PIM modules are unavailable")
            raw_severity = self.severity_pipeline(features)
            logit = self.calibrator.logits(raw_severity)
            probability = torch.sigmoid(logit).clamp(1e-6, 1.0 - 1e-6)
            severity = raw_severity.clamp(0.0, self.score_max)
        else:
            if self.risk_pipeline is None:
                raise RuntimeError("direct risk pipeline is unavailable")
            logit = self.risk_pipeline.logits(features)
            probability = torch.sigmoid(logit).clamp(1e-6, 1.0 - 1e-6)
            severity = torch.full_like(probability, torch.nan)
        return {
            "probability": probability,
            "logit": logit,
            "severity": severity,
        }


class CrossFittedPsychometricMeasurementBank(nn.Module):
    """Fold-indexed nuisance measurements for participant-held-out rewards."""

    def __init__(
        self,
        prototype: InstrumentConditionedPsychometricMeasurement,
        folds: int,
    ) -> None:
        super().__init__()
        if int(folds) < 2:
            raise ValueError("cross-fitted measurement requires at least two folds")
        self.folds = int(folds)
        self.models = nn.ModuleList(
            [copy.deepcopy(prototype) for _ in range(self.folds)]
        )

    def held_out(
        self,
        fold: int,
        nodes: torch.Tensor,
        temporal: torch.Tensor,
        evidence_mask: torch.Tensor,
        *,
        instrument_id: int,
    ) -> dict[str, torch.Tensor]:
        if not 0 <= int(fold) < self.folds:
            raise ValueError("fold is outside the measurement bank")
        return self.models[int(fold)](
            nodes,
            temporal,
            evidence_mask,
            instrument_id=instrument_id,
        )

    def fit_outputs(
        self,
        held_out_fold: int,
        nodes: torch.Tensor,
        temporal: torch.Tensor,
        evidence_mask: torch.Tensor,
        *,
        instrument_id: int,
    ) -> list[dict[str, torch.Tensor]]:
        return [
            model(
                nodes,
                temporal,
                evidence_mask,
                instrument_id=instrument_id,
            )
            for fold, model in enumerate(self.models)
            if fold != int(held_out_fold)
        ]

    def posterior(
        self,
        nodes: torch.Tensor,
        temporal: torch.Tensor,
        evidence_mask: torch.Tensor,
        *,
        instrument_id: int,
    ) -> dict[str, torch.Tensor]:
        outputs = [
            model(
                nodes,
                temporal,
                evidence_mask,
                instrument_id=instrument_id,
            )
            for model in self.models
        ]
        probability = torch.stack([item["probability"] for item in outputs])
        severity = torch.stack([item["severity"] for item in outputs])
        mean_probability = probability.mean()
        return {
            "probability": mean_probability,
            "risk_logit": torch.logit(
                mean_probability.clamp(1e-6, 1.0 - 1e-6)
            ),
            "severity": severity.mean(),
            "probability_std": probability.std(correction=0),
            "severity_std": severity.std(correction=0),
            "members": outputs,
        }
