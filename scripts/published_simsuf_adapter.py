"""Representation-matched SIMSUF adapter for locked multimodal caches.

This module preserves SIMSUF's defining fusion topology, not its original
MOSI/MOSEI feature or sentiment-regression protocol.  It is intentionally
standalone and is not wired into the existing published-model runner.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


SIMSUF_PROVENANCE: Final[dict[str, object]] = {
    "paper": "Dominant SIngle-Modal SUpplementary Fusion (SIMSUF) for Multimodal Sentiment Analysis",
    "venue": "IEEE Transactions on Multimedia, volume 26, pages 8383-8394 (2024)",
    "doi": "10.1109/TMM.2023.3344358",
    "official_repository": "https://github.com/Yan-LiJI/SIMSUF",
    "official_commit": "ee823bf94335b280177030ea2715dcac2e6b3e23",
    "official_tasks": ("CMU-MOSI sentiment regression", "CMU-MOSEI sentiment regression"),
    "adaptation_claim": "representation-matched architecture reproduction",
    "preserved": (
        "pairwise modality-dependence estimation and dominant-modality selection",
        "dominant-modality supplementary attention",
        "common and modality-specific enhancement branches",
        "difference and central-moment-discrepancy regularization",
        "dominant-modality-guided multimodal fusion",
    ),
    "not_preserved": (
        "original BERT, acoustic, and visual feature extractors",
        "MOSI/MOSEI segmentation and labels",
        "SmoothL1 sentiment-regression objective and reported metrics",
    ),
    "upstream_audit_notes": (
        "the audited commit has no repository license",
        "model.py calls selector(hp) although selector.__init__ requires args and i",
        "the released dominant-modality branch converts batch tensors to NumPy and uses array equality in Python if statements",
        "the released training loop evaluates Test every epoch; governed reproductions must instead lock on Train/Dev before Test",
    ),
}


def _valid_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(x.shape[1], device=x.device)[None, :]
    return positions < lengths[:, None]


def _masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _valid_mask(x, lengths).to(x.dtype).unsqueeze(-1)
    return (x * mask).sum(1) / mask.sum(1).clamp_min(1.0)


def _difference_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = F.normalize(first - first.mean(0, keepdim=True), dim=1)
    second = F.normalize(second - second.mean(0, keepdim=True), dim=1)
    return (first.T @ second).pow(2).mean()


def _central_moment_discrepancy(first: torch.Tensor, second: torch.Tensor, moments: int = 5) -> torch.Tensor:
    first_mean, second_mean = first.mean(0), second.mean(0)
    result = torch.linalg.vector_norm(first_mean - second_mean)
    first_centered, second_centered = first - first_mean, second - second_mean
    for order in range(2, moments + 1):
        result = result + torch.linalg.vector_norm(
            first_centered.pow(order).mean(0) - second_centered.pow(order).mean(0)
        )
    return result


class _SequenceProjector(nn.Module):
    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(dimension, hidden, batch_first=True, bidirectional=True)
        self.project = nn.Linear(hidden * 2, hidden)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.shape[1])
        return torch.tanh(self.project(output))


class _DependenceEstimator(nn.Module):
    """Learned symmetric dependence score for one modality pair."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        features = torch.cat((first, second, first * second, (first - second).abs()), dim=-1)
        return self.score(features).squeeze(-1)


class SIMSUFCacheClassifier(nn.Module):
    """Topology-preserving SIMSUF adaptation with a binary clinical head.

    Inputs are aligned participant-event tensors ``[batch, events, dimension]``.
    The caller remains responsible for train-only preprocessing and governed
    train/dev/test splits.  ``last_diagnostics`` exposes the selected modality
    and auxiliary terms for auditing; it is not used to select on held-out data.
    """

    modality_names: Final[tuple[str, str, str]] = ("text", "audio", "visual")

    def __init__(
        self,
        dimension: int,
        hidden: int = 32,
        heads: int = 4,
        dropout: float = 0.1,
        auxiliary_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.auxiliary_weight = float(auxiliary_weight)
        self.encoders = nn.ModuleList(_SequenceProjector(dimension, hidden) for _ in range(3))
        self.dependence = nn.ModuleList(_DependenceEstimator(hidden) for _ in range(3))
        self.supplement = nn.ModuleList(
            nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True) for _ in range(3)
        )
        self.supplement_norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(3))
        self.common = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.specific = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh()) for _ in range(3)
        )
        self.fusion_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )
        self.auxiliary = torch.tensor(0.0)
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @staticmethod
    def dominant_indices(pair_scores: torch.Tensor) -> torch.Tensor:
        """Return dominant modality from scores ordered as TA, AV, VT."""
        if pair_scores.ndim != 2 or pair_scores.shape[1] != 3:
            raise ValueError("pair_scores must have shape [batch, 3]")
        ta, av, vt = pair_scores.unbind(dim=1)
        modality_scores = torch.stack((ta + vt, ta + av, av + vt), dim=1)
        return modality_scores.argmax(dim=1)

    def _select(self, streams: list[torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(streams, dim=1)
        batch = torch.arange(stacked.shape[0], device=stacked.device)
        return stacked[batch, indices]

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if min(text.shape[1], audio.shape[1], visual.shape[1]) < 1:
            raise ValueError("SIMSUF requires at least one event per participant")
        lengths = lengths.clamp(min=1, max=min(text.shape[1], audio.shape[1], visual.shape[1]))
        streams = [encoder(value, lengths) for encoder, value in zip(self.encoders, (text, audio, visual))]
        pooled = [_masked_mean(value, lengths) for value in streams]
        pair_scores = torch.stack(
            (
                self.dependence[0](pooled[0], pooled[1]),
                self.dependence[1](pooled[1], pooled[2]),
                self.dependence[2](pooled[2], pooled[0]),
            ),
            dim=1,
        )
        dominant_indices = self.dominant_indices(pair_scores)
        dominant_sequence = self._select(streams, dominant_indices)
        padding_mask = ~_valid_mask(dominant_sequence, lengths)

        enhanced: list[torch.Tensor] = []
        for index, (stream, attention, norm) in enumerate(zip(streams, self.supplement, self.supplement_norm)):
            supplement, _ = attention(
                query=stream,
                key=dominant_sequence,
                value=dominant_sequence,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            is_dominant = (dominant_indices == index)[:, None, None]
            enhanced.append(norm(stream + torch.where(is_dominant, torch.zeros_like(supplement), supplement)))

        enhanced_pooled = [_masked_mean(value, lengths) for value in enhanced]
        common = [self.common(value) for value in enhanced_pooled]
        specific = [project(value) for project, value in zip(self.specific, enhanced_pooled)]
        dominant_token = self._select([value[:, None, :] for value in enhanced_pooled], dominant_indices)
        modality_tokens = torch.stack(enhanced_pooled, dim=1)
        fused, _ = self.fusion_attention(dominant_token, modality_tokens, modality_tokens, need_weights=False)
        fused = self.fusion_norm(fused + dominant_token).squeeze(1)

        common_loss = sum(
            _central_moment_discrepancy(common[i], common[j]) for i, j in ((0, 1), (0, 2), (1, 2))
        ) / 3.0
        specific_loss = sum(
            _difference_loss(specific[i], specific[j]) for i, j in ((0, 1), (0, 2), (1, 2))
        ) / 3.0
        dependence_reward = pair_scores.mean()
        self.auxiliary = common_loss + specific_loss - dependence_reward
        self.last_diagnostics = {
            "pair_dependence": pair_scores.detach(),
            "dominant_modality": dominant_indices.detach(),
            "common_loss": common_loss.detach(),
            "specific_loss": specific_loss.detach(),
            "dependence_reward": dependence_reward.detach(),
        }
        return self.head(torch.cat((*specific, fused), dim=1))

    def training_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=class_weight) + self.auxiliary_weight * self.auxiliary


__all__ = ["SIMSUFCacheClassifier", "SIMSUF_PROVENANCE"]
