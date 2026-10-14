"""Representation-matched Self-MM adapter for binary clinical screening.

The model and state manager are standalone by design.  The state manager
preserves Self-MM's defining train-indexed, epoch-dependent pseudo-label
refinement; it must never be initialized with or updated from held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


SELF_MM_PROVENANCE: Final[dict[str, object]] = {
    "paper": "Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis",
    "venue": "AAAI 2021",
    "doi": "10.1609/aaai.v35i12.17289",
    "official_repository": "https://github.com/thuiar/Self-MM",
    "official_commit": "1786283c81eeb507f317fa1c70a3faf77e67cee0",
    "license": "MIT",
    "official_tasks": (
        "CMU-MOSI sentiment regression",
        "CMU-MOSEI sentiment regression",
        "CH-SIMS sentiment regression",
    ),
    "adaptation_claim": "representation-matched architecture reproduction",
    "preserved": (
        "persistent Train-indexed fusion and stream-specific targets",
        "positive/negative representation centers for fusion and each stream",
        "center-distance pseudo-label refinement",
        "epoch-dependent pseudo-label moving average",
        "disagreement-weighted stream-specific supervision",
        "separate fusion and stream-specific prediction branches",
    ),
    "adapted": (
        "binary labels are signed continuous anchors in [-1, 1]",
        "fusion uses binary cross-entropy and stream heads use soft binary targets",
        "centers exclude not-yet-observed feature slots to avoid cold-start zero contamination",
    ),
    "not_preserved": (
        "original BERT, acoustic, and visual feature extractors",
        "MOSI/MOSEI/SIMS raw features, segmentation, sentiment labels, and regression metrics",
        "strict text/audio/visual semantics unless the supplied cache streams are independently modality-isolated",
    ),
    "upstream_audit_notes": (
        "the official trainer stores pseudo-label and feature state by stable Train index",
        "the official trainer recomputes centers after every batch, including zero-filled feature slots not yet visited",
        "the governed adapter excludes unobserved slots and rejects IDs outside its declared Train set",
    ),
}

STREAMS: Final[tuple[str, str, str]] = ("text", "audio", "visual")
ALL_STREAMS: Final[tuple[str, str, str, str]] = ("fusion", *STREAMS)


def _valid_mask(lengths: torch.Tensor, events: int) -> torch.Tensor:
    return torch.arange(events, device=lengths.device)[None, :] < lengths[:, None]


class _EventEncoder(nn.Module):
    def __init__(self, dimension: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Linear(dimension, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        projected = torch.relu(self.project(sequence))
        packed = pack_padded_sequence(projected, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=sequence.shape[1])
        valid = _valid_mask(lengths, sequence.shape[1]).to(output.dtype).unsqueeze(-1)
        pooled = (output * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        return self.dropout(pooled)


@dataclass
class SelfMMOutput:
    fusion_logits: torch.Tensor
    stream_logits: dict[str, torch.Tensor]
    features: dict[str, torch.Tensor]


class SelfMMCacheClassifier(nn.Module):
    """Self-MM topology adapted to aligned participant-event cache streams."""

    def __init__(
        self,
        dimension: int | tuple[int, int, int],
        hidden: int = 32,
        fusion_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dimensions = (dimension,) * 3 if isinstance(dimension, int) else dimension
        if len(dimensions) != 3:
            raise ValueError("dimension must be an int or a three-item tuple")
        self.encoders = nn.ModuleDict(
            {name: _EventEncoder(size, hidden, dropout) for name, size in zip(STREAMS, dimensions)}
        )
        self.stream_features = nn.ModuleDict(
            {name: nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU()) for name in STREAMS}
        )
        self.stream_heads = nn.ModuleDict({name: nn.Linear(hidden, 1) for name in STREAMS})
        self.fusion_feature = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, fusion_hidden),
            nn.ReLU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 1),
        )
        self.feature_dimensions: Final[dict[str, int]] = {
            "fusion": fusion_hidden,
            "text": hidden,
            "audio": hidden,
            "visual": hidden,
        }

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        lengths: torch.Tensor,
    ) -> SelfMMOutput:
        if min(text.shape[1], audio.shape[1], visual.shape[1]) < 1:
            raise ValueError("Self-MM requires at least one event per participant")
        max_events = min(text.shape[1], audio.shape[1], visual.shape[1])
        lengths = lengths.clamp(min=1, max=max_events)
        encoded = {
            name: self.encoders[name](value[:, :max_events], lengths)
            for name, value in zip(STREAMS, (text, audio, visual))
        }
        stream_features = {name: self.stream_features[name](encoded[name]) for name in STREAMS}
        fusion_feature = self.fusion_feature(torch.cat(tuple(encoded.values()), dim=-1))
        features = {"fusion": fusion_feature, **stream_features}
        return SelfMMOutput(
            fusion_logits=self.fusion_head(fusion_feature).squeeze(-1),
            stream_logits={name: self.stream_heads[name](stream_features[name]).squeeze(-1) for name in STREAMS},
            features=features,
        )

    @staticmethod
    def training_loss(
        output: SelfMMOutput,
        labels: torch.Tensor,
        sample_ids: Sequence[str],
        state: "SelfMMPseudoLabelState",
        positive_weight: torch.Tensor | None = None,
        stream_weight: float = 1.0,
    ) -> torch.Tensor:
        labels = labels.to(device=output.fusion_logits.device, dtype=output.fusion_logits.dtype)
        if positive_weight is not None:
            positive_weight = positive_weight.to(
                device=output.fusion_logits.device,
                dtype=output.fusion_logits.dtype,
            )
        fusion_loss = F.binary_cross_entropy_with_logits(
            output.fusion_logits, labels, pos_weight=positive_weight
        )
        pseudo = state.targets(sample_ids, device=output.fusion_logits.device)
        signed_anchor = pseudo["fusion"]
        auxiliary = output.fusion_logits.new_zeros(())
        for name in STREAMS:
            soft_target = (pseudo[name] + 1.0) / 2.0
            disagreement = torch.tanh(torch.abs(pseudo[name] - signed_anchor))
            per_sample = F.binary_cross_entropy_with_logits(
                output.stream_logits[name], soft_target, reduction="none"
            )
            auxiliary = auxiliary + torch.mean(disagreement * per_sample)
        return fusion_loss + float(stream_weight) * auxiliary


class SelfMMPseudoLabelState:
    """Persistent Self-MM state restricted to an explicitly declared Train set."""

    def __init__(self, feature_dimensions: Mapping[str, int], eps: float = 1e-8) -> None:
        if set(feature_dimensions) != set(ALL_STREAMS):
            raise ValueError(f"feature_dimensions must contain exactly {ALL_STREAMS}")
        self.feature_dimensions = {name: int(feature_dimensions[name]) for name in ALL_STREAMS}
        self.eps = float(eps)
        self._id_to_index: dict[str, int] = {}
        self._labels: dict[str, torch.Tensor] = {}
        self._features: dict[str, torch.Tensor] = {}
        self._seen: dict[str, torch.Tensor] = {}
        self._centers: dict[str, dict[str, torch.Tensor]] = {}

    @property
    def initialized(self) -> bool:
        return bool(self._id_to_index)

    def initialize(self, train_ids: Sequence[str], labels: torch.Tensor) -> None:
        ids = [str(value) for value in train_ids]
        if self.initialized:
            raise RuntimeError("Self-MM state is already initialized")
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("train_ids must be non-empty and unique")
        labels = torch.as_tensor(labels, dtype=torch.float32).view(-1).cpu()
        if labels.numel() != len(ids) or not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("labels must contain one binary label for every Train ID")
        signed = labels.mul(2.0).sub(1.0)
        self._id_to_index = {sample_id: index for index, sample_id in enumerate(ids)}
        self._labels = {name: signed.clone() for name in ALL_STREAMS}
        for name in ALL_STREAMS:
            self._features[name] = torch.zeros(len(ids), self.feature_dimensions[name])
            self._seen[name] = torch.zeros(len(ids), dtype=torch.bool)
        self._centers = {}

    def _indices(self, sample_ids: Sequence[str]) -> torch.Tensor:
        if not self.initialized:
            raise RuntimeError("initialize must be called with Train IDs first")
        ids = [str(value) for value in sample_ids]
        if len(ids) != len(set(ids)):
            raise ValueError("sample_ids must not contain duplicates")
        unknown = [value for value in ids if value not in self._id_to_index]
        if unknown:
            raise KeyError(f"IDs are outside the declared Train set: {unknown[:3]}")
        return torch.tensor([self._id_to_index[value] for value in ids], dtype=torch.long)

    def targets(self, sample_ids: Sequence[str], device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        indexes = self._indices(sample_ids)
        return {name: values[indexes].to(device=device) for name, values in self._labels.items()}

    def update_features(self, sample_ids: Sequence[str], features: Mapping[str, torch.Tensor]) -> None:
        indexes = self._indices(sample_ids)
        if set(features) != set(ALL_STREAMS):
            raise ValueError(f"features must contain exactly {ALL_STREAMS}")
        for name in ALL_STREAMS:
            values = features[name].detach().float().cpu()
            expected = (len(indexes), self.feature_dimensions[name])
            if tuple(values.shape) != expected:
                raise ValueError(f"{name} features have shape {tuple(values.shape)}; expected {expected}")
            self._features[name][indexes] = values
            self._seen[name][indexes] = True

    def update_centers(self) -> None:
        centers: dict[str, dict[str, torch.Tensor]] = {}
        for name in ALL_STREAMS:
            seen = self._seen[name]
            positive = seen & (self._labels[name] >= 0)
            negative = seen & (self._labels[name] < 0)
            if not positive.any() or not negative.any():
                raise RuntimeError(f"{name} needs observed Train features from both classes before center update")
            centers[name] = {
                "pos": self._features[name][positive].mean(0),
                "neg": self._features[name][negative].mean(0),
            }
        self._centers = centers

    def _distance_evidence(self, name: str, features: torch.Tensor) -> torch.Tensor:
        if name not in self._centers:
            raise RuntimeError("update_centers must be called before pseudo-label refinement")
        values = features.detach().float().cpu()
        positive_distance = torch.linalg.vector_norm(values - self._centers[name]["pos"], dim=-1)
        negative_distance = torch.linalg.vector_norm(values - self._centers[name]["neg"], dim=-1)
        return (negative_distance - positive_distance) / (positive_distance + self.eps)

    def update_pseudo_labels(
        self,
        epoch: int,
        sample_ids: Sequence[str],
        features: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if epoch <= 1:
            raise ValueError("official Self-MM refinement begins after epoch 1")
        indexes = self._indices(sample_ids)
        if set(features) != set(ALL_STREAMS):
            raise ValueError(f"features must contain exactly {ALL_STREAMS}")
        fusion_before = self._labels["fusion"].clone()
        delta_fusion = self._distance_evidence("fusion", features["fusion"])
        updated: dict[str, torch.Tensor] = {}
        for name in STREAMS:
            delta_stream = self._distance_evidence(name, features[name])
            alpha = delta_stream / (delta_fusion + self.eps)
            anchor = self._labels["fusion"][indexes]
            proposal = 0.5 * alpha * anchor + 0.5 * (anchor + delta_stream - delta_fusion)
            proposal = proposal.clamp(-1.0, 1.0)
            old = self._labels[name][indexes]
            new = ((epoch - 1.0) / (epoch + 1.0)) * old + (2.0 / (epoch + 1.0)) * proposal
            self._labels[name][indexes] = new
            updated[name] = new.clone()
        if not torch.equal(self._labels["fusion"], fusion_before):
            raise RuntimeError("fusion labels are immutable observed clinical anchors")
        return updated


__all__ = [
    "SELF_MM_PROVENANCE",
    "SelfMMCacheClassifier",
    "SelfMMOutput",
    "SelfMMPseudoLabelState",
]
