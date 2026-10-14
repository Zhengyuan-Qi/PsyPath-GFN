"""Representation-matched DMD adapter for governed aligned TAV caches.

The implementation preserves the defining components of Decoupled Multimodal
Distilling (DMD, CVPR 2023): modality-invariant/exclusive decomposition,
reconstruction and cycle constraints, homogeneous and heterogeneous dynamic
graph distillation, cross-modal exclusive fusion, and gated final prediction.
HumanOmniV2 text/audio event representations and official visual sequences
replace the original raw feature encoders.

This module intentionally contains no split, optimizer, checkpoint-selection,
or evaluation logic. All auxiliary losses are train-only; official test metrics
must use ``final_logits`` selected solely through the validation split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = 3
DIRECTED_EDGES = tuple((target, source) for target in range(3) for source in range(3) if source != target)


def _validate_inputs(
    text: torch.Tensor,
    audio: torch.Tensor,
    visual: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[int, int]:
    streams = (text, audio, visual)
    if any(value.ndim != 3 for value in streams):
        raise ValueError("text, audio, and visual must have shape [B, T, D].")
    batch, events = text.shape[:2]
    if any(value.shape[:2] != (batch, events) for value in streams[1:]):
        raise ValueError("DMD requires aligned modalities with identical B and T dimensions.")
    if lengths.ndim != 1 or lengths.shape[0] != batch:
        raise ValueError("lengths must have shape [B].")
    if torch.is_floating_point(lengths) or lengths.dtype == torch.bool:
        raise ValueError("lengths must use an integer dtype.")
    if bool(torch.any(lengths < 1)) or bool(torch.any(lengths > events)):
        raise ValueError("Every sequence length must be between 1 and T inclusive.")
    return batch, events


def _valid_mask(lengths: torch.Tensor, events: int, device: torch.device) -> torch.Tensor:
    return torch.arange(events, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.unsqueeze(-1).to(value.dtype)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _masked_mse(first: torch.Tensor, second: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.unsqueeze(-1).to(first.dtype)
    squared = (first - second).pow(2) * weight
    return squared.sum() / (weight.sum() * first.shape[-1]).clamp_min(1.0)


def _last_valid(value: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    index = (lengths.to(value.device) - 1).view(-1, 1, 1).expand(-1, 1, value.shape[-1])
    return value.gather(1, index).squeeze(1)


def _causal_mask(events: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(events, events, dtype=torch.bool, device=device), diagonal=1)


class _TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            hidden_dim * 4,
            dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        output = self.encoder(
            value,
            mask=_causal_mask(value.shape[1], value.device),
            src_key_padding_mask=~valid,
        )
        return output.masked_fill(~valid.unsqueeze(-1), 0.0)


class _CrossModalBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.source_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, source: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.source_norm(source),
            self.source_norm(source),
            attn_mask=_causal_mask(query.shape[1], query.device),
            key_padding_mask=~valid,
            need_weights=False,
        )
        value = query + self.dropout(attended)
        value = value + self.dropout(self.feed_forward(self.output_norm(value)))
        return value.masked_fill(~valid.unsqueeze(-1), 0.0)


class _ResidualHead(nn.Module):
    def __init__(self, input_dim: int, representation_dim: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.to_representation = nn.Linear(input_dim, representation_dim)
        self.back_projection = nn.Linear(representation_dim, input_dim)
        self.classifier = nn.Linear(input_dim, classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.to_representation(value)
        residual = value + self.back_projection(self.dropout(F.relu(representation)))
        return self.classifier(residual), representation


class DynamicGraphDistillation(nn.Module):
    """DMD GD-Unit with six learned directed edges among three modalities."""

    def __init__(
        self,
        representation_dim: int,
        classes: int,
        graph_dim: int,
        prior_scores: Sequence[float],
        *,
        prior_temperature: float = 0.25,
        edge_temperature: float = 1.0 / 8.0,
        prior_regularization: float = 10.0,
    ) -> None:
        super().__init__()
        if len(prior_scores) != len(DIRECTED_EDGES):
            raise ValueError("prior_scores must contain one value for each of the six directed edges.")
        if prior_temperature <= 0.0 or edge_temperature <= 0.0:
            raise ValueError("Graph temperatures must be positive.")
        self.logit_projection = nn.Linear(classes, graph_dim)
        self.representation_projection = nn.Linear(representation_dim, graph_dim)
        self.edge_projection = nn.Linear(graph_dim * 4, 1)
        self.edge_temperature = edge_temperature
        self.prior_regularization = prior_regularization
        prior = F.softmax(torch.tensor(prior_scores, dtype=torch.float32) / prior_temperature, dim=0)
        self.register_buffer("prior", prior)

    def forward(self, logits: torch.Tensor, representations: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or representations.ndim != 3 or logits.shape[:2] != representations.shape[:2]:
            raise ValueError("Graph inputs must have shapes [3, B, C] and [3, B, D].")
        if logits.shape[0] != MODALITIES:
            raise ValueError("DMD graph distillation requires exactly three modalities.")
        embedded = torch.cat(
            (self.logit_projection(logits), self.representation_projection(representations)),
            dim=-1,
        )
        raw_edges = [
            self.edge_projection(torch.cat((embedded[target], embedded[source]), dim=-1))
            for target, source in DIRECTED_EDGES
        ]
        # [edge, batch], normalized across the six directed edges per sample.
        return F.softmax(torch.cat(raw_edges, dim=1) * self.edge_temperature, dim=1).transpose(0, 1)

    def losses(
        self,
        logits: torch.Tensor,
        representations: torch.Tensor,
        edges: torch.Tensor,
        *,
        heterogeneous: bool,
        logit_weight: float = 1.0,
        representation_weight: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edges.shape != (len(DIRECTED_EDGES), logits.shape[1]):
            raise ValueError("edges must have shape [6, B].")
        regularization = (edges.mean(dim=1) - self.prior).pow(2).sum() * self.prior_regularization
        logit_loss = logits.new_zeros(())
        representation_loss = logits.new_zeros(())
        for edge_index, (target, source) in enumerate(DIRECTED_EDGES):
            weight = edges[edge_index] + self.prior[edge_index]
            logit_distance = (logits[target] - logits[source].detach()).abs().sum(dim=1)
            logit_loss = logit_loss + logit_weight * (logit_distance * weight).mean()
            if heterogeneous:
                # The official HeteroGD uses CosineEmbeddingLoss with target -1:
                # it preserves modality-exclusive diversity rather than forcing
                # exclusive representations to become identical.
                cosine = F.cosine_similarity(
                    representations[target],
                    representations[source].detach(),
                    dim=1,
                )
                distance = F.relu(cosine)
            else:
                distance = (
                    representations[target] - representations[source].detach()
                ).abs().sum(dim=1)
            representation_loss = representation_loss + representation_weight * (distance * weight).mean()
        return regularization, logit_loss, representation_loss


@dataclass(frozen=True)
class DMDOutput:
    final_logits: torch.Tensor
    common_logits: torch.Tensor
    homo_logits: torch.Tensor
    homo_representations: torch.Tensor
    hetero_logits: torch.Tensor
    hetero_representations: torch.Tensor
    origins: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    specifics: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    commons: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    reconstructions: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    cycled_specifics: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    aligned_commons: torch.Tensor
    homo_edges: torch.Tensor
    hetero_edges: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class DMDLosses:
    total: torch.Tensor
    task_final: torch.Tensor
    task_auxiliary: torch.Tensor
    reconstruction: torch.Tensor
    cycle_consistency: torch.Tensor
    orthogonality: torch.Tensor
    shared_margin: torch.Tensor
    homo_graph_regularization: torch.Tensor
    homo_logit_distillation: torch.Tensor
    homo_representation_distillation: torch.Tensor
    hetero_graph_regularization: torch.Tensor
    hetero_logit_distillation: torch.Tensor
    hetero_representation_distillation: torch.Tensor


class DMDCacheAdapter(nn.Module):
    """Binary clinical-screening adaptation of the official DMD topology."""

    def __init__(
        self,
        input_dims: Sequence[int],
        *,
        hidden_dim: int = 32,
        classes: int = 2,
        convolution_kernels: Sequence[int] = (3, 3, 3),
        attention_heads: int = 4,
        attention_layers: int = 2,
        dropout: float = 0.1,
        homo_graph_dim: int = 64,
        hetero_graph_dim: int = 32,
    ) -> None:
        super().__init__()
        if len(input_dims) != 3 or len(convolution_kernels) != 3:
            raise ValueError("input_dims and convolution_kernels must have length 3.")
        if any(value < 1 for value in (*input_dims, *convolution_kernels, hidden_dim, classes)):
            raise ValueError("Dimensions, classes, and kernels must be positive.")
        if any(kernel % 2 == 0 for kernel in convolution_kernels):
            raise ValueError("Convolution kernels must be odd to preserve aligned event length.")
        if hidden_dim % attention_heads != 0 or (hidden_dim * 2) % attention_heads != 0:
            raise ValueError("hidden_dim and 2 * hidden_dim must be divisible by attention_heads.")

        self.projections = nn.ModuleList(
            nn.Conv1d(source, hidden_dim, kernel, padding=kernel // 2, bias=False)
            for source, kernel in zip(input_dims, convolution_kernels)
        )
        self.specific_encoders = nn.ModuleList(
            nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False) for _ in range(3)
        )
        self.common_encoder = nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False)
        self.decoders = nn.ModuleList(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1, bias=False) for _ in range(3)
        )
        self.common_temporal = nn.ModuleList(
            _TemporalEncoder(hidden_dim, attention_heads, attention_layers, dropout) for _ in range(3)
        )
        self.cross_attention = nn.ModuleDict(
            {
                f"{target}_{source}": _CrossModalBlock(hidden_dim, attention_heads, dropout)
                for target, source in DIRECTED_EDGES
            }
        )
        self.exclusive_memory = nn.ModuleList(
            _TemporalEncoder(hidden_dim * 2, attention_heads, max(3, attention_layers), dropout)
            for _ in range(3)
        )

        self.common_align = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(3))
        self.homo_heads = nn.ModuleList(
            _ResidualHead(hidden_dim, hidden_dim, classes, dropout) for _ in range(3)
        )
        self.hetero_heads = nn.ModuleList(
            _ResidualHead(hidden_dim * 2, hidden_dim * 2, classes, dropout) for _ in range(3)
        )
        self.common_head = _ResidualHead(hidden_dim * 3, hidden_dim * 3, classes, dropout)

        self.exclusive_gates = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim * 2) for _ in range(3))
        self.common_gate = nn.Linear(hidden_dim * 3, hidden_dim * 3)
        final_dim = hidden_dim * 9
        self.final_head = _ResidualHead(final_dim, final_dim, classes, dropout)

        self.homo_graph = DynamicGraphDistillation(
            hidden_dim,
            classes,
            homo_graph_dim,
            prior_scores=(0, 0, 1, 0, 1, 0),
        )
        self.hetero_graph = DynamicGraphDistillation(
            hidden_dim * 2,
            classes,
            hetero_graph_dim,
            # Official prior is expressed in L,V,A order and favors L->V,
            # L->A, V->A. Remap it to this adapter's T,A,V contract.
            prior_scores=(0, 0, 1, 1, 1, 0),
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        lengths: torch.Tensor,
    ) -> DMDOutput:
        _, events = _validate_inputs(text, audio, visual, lengths)
        valid = _valid_mask(lengths, events, text.device)
        streams = []
        for value, projection in zip((text, audio, visual), self.projections):
            value = value.masked_fill(~valid.unsqueeze(-1), 0.0)
            projected = projection(value.transpose(1, 2)).transpose(1, 2)
            streams.append(projected.masked_fill(~valid.unsqueeze(-1), 0.0))

        specifics = tuple(
            encoder(value.transpose(1, 2)).transpose(1, 2).masked_fill(~valid.unsqueeze(-1), 0.0)
            for encoder, value in zip(self.specific_encoders, streams)
        )
        commons = tuple(
            self.common_encoder(value.transpose(1, 2)).transpose(1, 2).masked_fill(~valid.unsqueeze(-1), 0.0)
            for value in streams
        )
        reconstructions = tuple(
            decoder(torch.cat((specific, common), dim=-1).transpose(1, 2)).transpose(1, 2)
            for decoder, specific, common in zip(self.decoders, specifics, commons)
        )
        reconstructions = tuple(value.masked_fill(~valid.unsqueeze(-1), 0.0) for value in reconstructions)
        cycled_specifics = tuple(
            encoder(value.transpose(1, 2)).transpose(1, 2).masked_fill(~valid.unsqueeze(-1), 0.0)
            for encoder, value in zip(self.specific_encoders, reconstructions)
        )

        pooled_commons = tuple(_masked_mean(value, valid) for value in commons)
        homo_results = [head(value) for head, value in zip(self.homo_heads, pooled_commons)]
        homo_logits = torch.stack([value[0] for value in homo_results])
        homo_representations = torch.stack([value[1] for value in homo_results])
        aligned_commons = torch.stack(
            [align(value) for align, value in zip(self.common_align, pooled_commons)]
        )

        attended_commons = [
            temporal(common, valid) for temporal, common in zip(self.common_temporal, commons)
        ]
        common_fused = torch.cat(
            [_last_valid(value, lengths) for value in attended_commons],
            dim=-1,
        )
        common_logits, _ = self.common_head(common_fused)

        exclusive_vectors = []
        for target in range(3):
            sources = [source for source in range(3) if source != target]
            cross = [
                self.cross_attention[f"{target}_{source}"](specifics[target], specifics[source], valid)
                for source in sources
            ]
            memory = self.exclusive_memory[target](torch.cat(cross, dim=-1), valid)
            exclusive_vectors.append(_last_valid(memory, lengths))
        hetero_results = [head(value) for head, value in zip(self.hetero_heads, exclusive_vectors)]
        hetero_logits = torch.stack([value[0] for value in hetero_results])
        hetero_representations = torch.stack([value[1] for value in hetero_results])

        final_features = [
            torch.sigmoid(gate(value)) for gate, value in zip(self.exclusive_gates, exclusive_vectors)
        ]
        final_features.append(torch.sigmoid(self.common_gate(common_fused)))
        final_logits, _ = self.final_head(torch.cat(final_features, dim=-1))

        homo_edges = self.homo_graph(homo_logits, homo_representations)
        hetero_edges = self.hetero_graph(hetero_logits, hetero_representations)
        return DMDOutput(
            final_logits=final_logits,
            common_logits=common_logits,
            homo_logits=homo_logits,
            homo_representations=homo_representations,
            hetero_logits=hetero_logits,
            hetero_representations=hetero_representations,
            origins=tuple(streams),
            specifics=specifics,
            commons=commons,
            reconstructions=reconstructions,
            cycled_specifics=cycled_specifics,
            aligned_commons=aligned_commons,
            homo_edges=homo_edges,
            hetero_edges=hetero_edges,
            valid_mask=valid,
        )

    def training_losses(
        self,
        output: DMDOutput,
        labels: torch.Tensor,
        *,
        class_weight: torch.Tensor | None = None,
        graph_weight: float = 0.05,
        structure_weight: float = 0.1,
        separation_weight: float = 0.1,
        shared_margin: float = 0.15,
    ) -> DMDLosses:
        """Return the official joint DMD objective, decomposed for auditing."""

        if labels.ndim != 1 or labels.shape[0] != output.final_logits.shape[0]:
            raise ValueError("labels must have shape [B].")
        if any(value < 0.0 for value in (graph_weight, structure_weight, separation_weight, shared_margin)):
            raise ValueError("Loss weights and margin must be non-negative.")

        criterion = lambda logits: F.cross_entropy(logits, labels, weight=class_weight)
        task_final = criterion(output.final_logits)
        task_auxiliary = criterion(output.common_logits)
        for modality in range(3):
            task_auxiliary = task_auxiliary + criterion(output.homo_logits[modality])
            task_auxiliary = task_auxiliary + criterion(output.hetero_logits[modality])

        reconstruction = sum(
            _masked_mse(reconstructed, origin, output.valid_mask)
            for reconstructed, origin in zip(output.reconstructions, output.origins)
        )
        cycle = sum(
            _masked_mse(cycled, specific, output.valid_mask)
            for cycled, specific in zip(output.cycled_specifics, output.specifics)
        )
        pooled_specifics = [_masked_mean(value, output.valid_mask) for value in output.specifics]
        pooled_commons = [_masked_mean(value, output.valid_mask) for value in output.commons]
        orthogonality = sum(
            F.relu(F.cosine_similarity(specific, common, dim=1)).mean()
            for specific, common in zip(pooled_specifics, pooled_commons)
        )
        margin_loss = self._shared_margin_loss(output.aligned_commons, labels, shared_margin)

        homo_reg, homo_logit, homo_repr = self.homo_graph.losses(
            output.homo_logits,
            output.homo_representations,
            output.homo_edges,
            heterogeneous=False,
        )
        hetero_reg, hetero_logit, hetero_repr = self.hetero_graph.losses(
            output.hetero_logits,
            output.hetero_representations,
            output.hetero_edges,
            heterogeneous=True,
        )
        # The official implementation computes homo_repr but deliberately omits
        # it from the combined objective. HeteroGD representation loss remains.
        total = (
            task_final
            + task_auxiliary
            + graph_weight * (homo_reg + homo_logit)
            + graph_weight * (hetero_reg + hetero_logit + hetero_repr)
            + structure_weight * (cycle + reconstruction + separation_weight * (margin_loss + orthogonality))
        )
        return DMDLosses(
            total,
            task_final,
            task_auxiliary,
            reconstruction,
            cycle,
            orthogonality,
            margin_loss,
            homo_reg,
            homo_logit,
            homo_repr,
            hetero_reg,
            hetero_logit,
            hetero_repr,
        )

    @staticmethod
    def _shared_margin_loss(features: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
        # [modality, batch, hidden] -> modality-major participant examples.
        flat = features.reshape(-1, features.shape[-1])
        repeated_labels = labels.unsqueeze(0).expand(3, -1).reshape(-1)
        similarity = F.cosine_similarity(flat[:, None, :], flat[None, :, :], dim=-1)
        identity = torch.eye(flat.shape[0], dtype=torch.bool, device=flat.device)
        same = repeated_labels[:, None].eq(repeated_labels[None, :]) & ~identity
        different = ~repeated_labels[:, None].eq(repeated_labels[None, :])
        losses = []
        for row in range(flat.shape[0]):
            positives = similarity[row][same[row]]
            negatives = similarity[row][different[row]]
            if positives.numel() and negatives.numel():
                losses.append(F.relu(margin - positives[:, None] + negatives[None, :]).mean())
        return torch.stack(losses).mean() if losses else flat.sum() * 0.0


__all__ = [
    "DIRECTED_EDGES",
    "DMDCacheAdapter",
    "DMDLosses",
    "DMDOutput",
    "DynamicGraphDistillation",
]
