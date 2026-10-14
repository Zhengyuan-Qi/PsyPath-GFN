"""Topology-preserving cache adapters for published multimodal models.

These modules retain each paper's defining fusion operator while replacing its
raw feature extractor with the governed HumanOmniV2 event streams.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def _masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(x.shape[1], device=x.device)[None, :]
    mask = (positions < lengths[:, None]).to(x.dtype).unsqueeze(-1)
    return (x * mask).sum(1) / mask.sum(1).clamp_min(1.0)


class _SequenceEncoder(nn.Module):
    def __init__(self, dimension: int, hidden: int = 32) -> None:
        super().__init__()
        self.rnn = nn.GRU(dimension, hidden, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.rnn(packed)
        return self.projection(torch.cat((state[-2], state[-1]), dim=-1))


def _difference_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = F.normalize(first - first.mean(0, keepdim=True), dim=1)
    second = F.normalize(second - second.mean(0, keepdim=True), dim=1)
    return (first.T @ second).pow(2).mean()


def _cmd(first: torch.Tensor, second: torch.Tensor, moments: int = 5) -> torch.Tensor:
    first_mean, second_mean = first.mean(0), second.mean(0)
    loss = torch.linalg.vector_norm(first_mean - second_mean)
    first_centered, second_centered = first - first_mean, second - second_mean
    for order in range(2, moments + 1):
        loss = loss + torch.linalg.vector_norm(
            first_centered.pow(order).mean(0) - second_centered.pow(order).mean(0)
        )
    return loss


class MISAClassifier(nn.Module):
    """ACM MM 2020 MISA shared/private decomposition and fusion."""

    def __init__(self, dimension: int, hidden: int = 32) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(_SequenceEncoder(dimension, hidden) for _ in range(3))
        self.private = nn.ModuleList(nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid()) for _ in range(3))
        self.shared = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.reconstruct = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(3))
        layer = nn.TransformerEncoderLayer(hidden, 2, hidden * 2, 0.1, batch_first=False)
        self.fusion = nn.TransformerEncoder(layer, 1)
        self.head = nn.Sequential(nn.Linear(hidden * 6, hidden * 3), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden * 3, 2))
        self.auxiliary = torch.tensor(0.0)

    def forward(self, text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        original = [encoder(value, lengths) for encoder, value in zip(self.encoders, (text, visual, audio))]
        private = [encoder(value) for encoder, value in zip(self.private, original)]
        shared = [self.shared(value) for value in original]
        reconstructed = [decoder(p + s) for decoder, p, s in zip(self.reconstruct, private, shared)]
        diff = sum(_difference_loss(p, s) for p, s in zip(private, shared)) / 3.0
        diff = diff + sum(_difference_loss(private[i], private[j]) for i, j in ((0, 1), (0, 2), (1, 2))) / 3.0
        similarity = sum(_cmd(shared[i], shared[j]) for i, j in ((0, 1), (0, 2), (1, 2))) / 3.0
        reconstruction = sum(F.mse_loss(value, target) for value, target in zip(reconstructed, original)) / 3.0
        self.auxiliary = 0.3 * diff + similarity + reconstruction
        tokens = self.fusion(torch.stack((*private, *shared), dim=0)).transpose(0, 1).flatten(1)
        return self.head(tokens)

    def training_loss(self, logits: torch.Tensor, labels: torch.Tensor, class_weight: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=class_weight) + self.auxiliary


class _GaussianMILowerBound(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.logvar = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.mu(x), self.logvar(x).clamp(-8.0, 8.0)
        return (-0.5 * (mu - y).pow(2) / logvar.exp()).sum(-1).mean()


class _CPC(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.project = nn.Linear(hidden, hidden)

    def forward(self, target: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        prediction = F.normalize(self.project(fused), dim=1)
        target = F.normalize(target, dim=1)
        positive = (target * prediction).sum(-1)
        negative = torch.logsumexp(target @ prediction.T, dim=-1)
        return -(positive - negative).mean()


class MMIMClassifier(nn.Module):
    """EMNLP 2021 hierarchical mutual-information maximization."""

    def __init__(self, dimension: int, hidden: int = 32) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(_SequenceEncoder(dimension, hidden) for _ in range(3))
        self.mi = nn.ModuleList(_GaussianMILowerBound(hidden) for _ in range(3))
        self.fusion = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden * 3, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 2)
        self.cpc = nn.ModuleList(_CPC(hidden) for _ in range(3))
        self.auxiliary = torch.tensor(0.0)

    def forward(self, text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = [encoder(value, lengths) for encoder, value in zip(self.encoders, (text, audio, visual))]
        lld = self.mi[0](encoded[0], encoded[1]) + self.mi[1](encoded[0], encoded[2]) + self.mi[2](encoded[1], encoded[2])
        fused = self.fusion(torch.cat(encoded, dim=1))
        nce = sum(module(value, fused) for module, value in zip(self.cpc, encoded))
        self.auxiliary = 0.1 * nce - 0.1 * lld
        return self.head(fused)

    def training_loss(self, logits: torch.Tensor, labels: torch.Tensor, class_weight: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=class_weight) + self.auxiliary


class _AxisMLPBlock(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], hidden_shape: tuple[int, int, int], output_shape: tuple[int, int, int], dropout: float = 0.1) -> None:
        super().__init__()
        ti, mi, ci = input_shape
        th, mh, ch = hidden_shape
        to, mo, co = output_shape
        # CubeMLP's released depression/sentiment recipes use ln_first, so each
        # axis MLP sees a normalized input before the projected residual is added.
        self.norm_t, self.norm_m, self.norm_c = nn.LayerNorm(ti), nn.LayerNorm(mi), nn.LayerNorm(ci)
        self.time = nn.Sequential(nn.Linear(ti, th), nn.GELU(), nn.Linear(th, to))
        self.modality = nn.Sequential(nn.Linear(mi, mh), nn.GELU(), nn.Linear(mh, mo))
        self.channel = nn.Sequential(nn.Linear(ci, ch), nn.GELU(), nn.Linear(ch, co))
        self.residual_t = nn.Linear(ti, to, bias=False)
        self.residual_m = nn.Linear(mi, mo, bias=False)
        self.residual_c = nn.Linear(ci, co, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_t(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = residual + self.dropout(self.time(self.norm_t(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2))
        residual = self.residual_m(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = residual + self.dropout(self.modality(self.norm_m(x.permute(0, 1, 3, 2))).permute(0, 1, 3, 2))
        residual = self.residual_c(x)
        return residual + self.dropout(self.channel(self.norm_c(x)))


class CubeMLPClassifier(nn.Module):
    """ACM MM 2022 MLP mixing over time, modality, and channel axes."""

    def __init__(self, dimension: int, events: int, hidden: int = 32) -> None:
        super().__init__()
        self.project = nn.ModuleList(nn.Linear(dimension, hidden, bias=False) for _ in range(3))
        middle_events = min(50, events)
        output_events = min(10, middle_events)
        self.blocks = nn.Sequential(
            _AxisMLPBlock((events, 3, hidden), (middle_events, 3, 64), (middle_events, 3, 64)),
            _AxisMLPBlock((middle_events, 3, 64), (output_events, 3, hidden), (output_events, 3, hidden)),
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, 2))

    def forward(self, text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        streams = [projection(value) for projection, value in zip(self.project, (text, audio, visual))]
        x = torch.stack(streams, dim=2)
        positions = torch.arange(x.shape[1], device=x.device)[None, :]
        x = x.masked_fill((positions >= lengths[:, None])[:, :, None, None], 0.0)
        x = self.blocks(x)
        return self.head(x.mean(dim=1).flatten(1))


class ALMTClassifier(nn.Module):
    """EMNLP 2023 language-guided adaptive hyper-modality fusion."""

    def __init__(self, dimension: int, events: int, transformer: type[nn.Module], cross_transformer: type[nn.Module], hyper_encoder: type[nn.Module], hidden: int = 32, tokens: int = 8) -> None:
        super().__init__()
        self.hyper = nn.Parameter(torch.ones(1, tokens, hidden))
        def projector() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(dimension, hidden),
                transformer(num_frames=events, save_hidden=False, token_len=tokens, dim=hidden, depth=1, heads=4, mlp_dim=hidden * 4),
            )
        self.project_text, self.project_audio, self.project_visual = projector(), projector(), projector()
        self.language = transformer(num_frames=tokens, save_hidden=True, token_len=None, dim=hidden, depth=2, heads=4, mlp_dim=hidden * 4)
        self.hyper_encoder = hyper_encoder(dim=hidden, depth=3, heads=4, dim_head=8, dropout=0.1)
        self.fusion = cross_transformer(source_num_frames=tokens, tgt_num_frames=tokens, dim=hidden, depth=2, heads=4, mlp_dim=hidden * 4)
        self.head = nn.Linear(hidden, 2)

    def forward(self, text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        text = self.project_text(text)[:, : self.hyper.shape[1]]
        audio = self.project_audio(audio)[:, : self.hyper.shape[1]]
        visual = self.project_visual(visual)[:, : self.hyper.shape[1]]
        language = self.language(text)
        hyper = self.hyper.expand(text.shape[0], -1, -1)
        hyper = self.hyper_encoder(language, audio, visual, hyper)
        return self.head(self.fusion(hyper, language[-1])[:, 0])
