"""NIAT architecture adapter for governed multimodal event caches.

This is a representation-matched reproduction of NIAT, not a copy of its raw
MOSI/MOSEI input pipeline.  It preserves NIAT's temporal projections, shared
fusion encoder, missing-input reconstruction, adversarial domain objective,
and clean/corrupted task supervision.  HumanOmniV2 event representations and
official visual sequences replace the original BERT/acoustic/visual features.

The module deliberately contains no optimizer or data-split logic.  In
particular, callers must generate corruptions for training data only and must
report clean official-test performance separately from robustness results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F


CorruptionStrategy = Literal["structured", "random", "block"]


@dataclass(frozen=True)
class NIATCorruptionConfig:
    """Missing-event simulation used by the NIAT clean/corrupted objective."""

    missing_rate: float = 0.2
    strategy: CorruptionStrategy = "structured"

    def __post_init__(self) -> None:
        if not 0.0 <= self.missing_rate < 1.0:
            raise ValueError("missing_rate must be in [0, 1).")
        if self.strategy not in {"structured", "random", "block"}:
            raise ValueError(f"Unsupported corruption strategy: {self.strategy!r}")


@dataclass(frozen=True)
class NIATCorruptedBatch:
    text: torch.Tensor
    audio: torch.Tensor
    visual: torch.Tensor
    # True denotes an observed event. Shape: [batch, time, modality].
    observed_mask: torch.Tensor


@dataclass(frozen=True)
class NIATOutput:
    clean_logits: torch.Tensor
    corrupted_logits: torch.Tensor
    clean_fused: torch.Tensor
    corrupted_fused: torch.Tensor
    reconstructed_fused: torch.Tensor
    domain_clean_logits: torch.Tensor
    domain_corrupted_logits: torch.Tensor


@dataclass(frozen=True)
class NIATLosses:
    generator_total: torch.Tensor
    classification_clean: torch.Tensor
    classification_corrupted: torch.Tensor
    reconstruction: torch.Tensor
    generator_adversarial: torch.Tensor


def _validate_streams(
    text: torch.Tensor,
    audio: torch.Tensor,
    visual: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[int, int]:
    streams = (text, audio, visual)
    if any(value.ndim != 3 for value in streams):
        raise ValueError("text, audio, and visual must all have shape [B, T, D].")
    batch, events = text.shape[:2]
    if any(value.shape[:2] != (batch, events) for value in streams[1:]):
        raise ValueError("All modality streams must share batch and event dimensions.")
    if lengths.ndim != 1 or lengths.shape[0] != batch:
        raise ValueError("lengths must have shape [B].")
    if torch.is_floating_point(lengths) or lengths.dtype == torch.bool:
        raise ValueError("lengths must use an integer dtype.")
    if bool(torch.any(lengths < 1)) or bool(torch.any(lengths > events)):
        raise ValueError("Every sequence length must be between 1 and T inclusive.")
    return batch, events


def _valid_mask(lengths: torch.Tensor, events: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(events, device=device).unsqueeze(0)
    return positions < lengths.to(device=device).unsqueeze(1)


def _sample_random_observed(
    valid: torch.Tensor,
    missing_rate: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if missing_rate == 0.0:
        return valid.clone()
    random_device = valid.device if generator is None else generator.device
    random_values = torch.rand(valid.shape, device=random_device, generator=generator).to(valid.device)
    observed = valid & (random_values >= missing_rate)
    # NIAT needs a usable sequence representation. Keep one valid event if a
    # short sequence happened to be completely removed.
    empty = valid.any(dim=1) & ~observed.any(dim=1)
    if bool(empty.any()):
        first_valid = valid.to(torch.int64).argmax(dim=1)
        rows = torch.nonzero(empty, as_tuple=False).squeeze(1)
        observed[rows, first_valid[rows]] = True
    return observed


def _sample_block_observed(
    valid: torch.Tensor,
    lengths: torch.Tensor,
    missing_rate: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    observed = valid.clone()
    if missing_rate == 0.0:
        return observed
    for row, raw_length in enumerate(lengths.detach().cpu().tolist()):
        length = int(raw_length)
        removable = min(length - 1, max(1, round(length * missing_rate)))
        if removable <= 0:
            continue
        max_start = length - removable
        random_device = valid.device if generator is None else generator.device
        start = int(torch.randint(max_start + 1, (1,), device=random_device, generator=generator).item())
        observed[row, start : start + removable] = False
    return observed


def corrupt_aligned_modalities(
    text: torch.Tensor,
    audio: torch.Tensor,
    visual: torch.Tensor,
    lengths: torch.Tensor,
    config: NIATCorruptionConfig | None = None,
    *,
    generator: torch.Generator | None = None,
) -> NIATCorruptedBatch:
    """Return NIAT-style missing-event views without mutating the inputs.

    ``structured`` uses one shared event mask across all modalities, ``random``
    samples each modality independently, and ``block`` removes an independent
    contiguous block from each modality. Padding is always zero in the output.
    """

    config = config or NIATCorruptionConfig()
    _, events = _validate_streams(text, audio, visual, lengths)
    valid = _valid_mask(lengths, events, text.device)

    if config.strategy == "structured":
        mask = _sample_random_observed(valid, config.missing_rate, generator)
        masks = (mask, mask, mask)
    elif config.strategy == "random":
        masks = tuple(
            _sample_random_observed(valid, config.missing_rate, generator)
            for _ in range(3)
        )
    else:
        masks = tuple(
            _sample_block_observed(valid, lengths, config.missing_rate, generator)
            for _ in range(3)
        )

    corrupted = tuple(value * mask.unsqueeze(-1).to(value.dtype) for value, mask in zip((text, audio, visual), masks))
    return NIATCorruptedBatch(
        text=corrupted[0],
        audio=corrupted[1],
        visual=corrupted[2],
        observed_mask=torch.stack(masks, dim=-1),
    )


class NIATFusionEncoder(nn.Module):
    """NIAT temporal-convolution/Transformer fusion for cached event streams."""

    def __init__(
        self,
        input_dims: Sequence[int],
        projection_dims: Sequence[int] = (32, 16, 16),
        *,
        convolution_kernels: Sequence[int] = (3, 3, 3),
        transformer_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(input_dims) != 3 or len(projection_dims) != 3 or len(convolution_kernels) != 3:
            raise ValueError("input_dims, projection_dims, and convolution_kernels must have length 3.")
        if any(value < 1 for value in (*input_dims, *projection_dims, *convolution_kernels)):
            raise ValueError("All dimensions and convolution kernels must be positive.")
        if any(kernel % 2 == 0 for kernel in convolution_kernels):
            raise ValueError("Convolution kernels must be odd to preserve event alignment.")
        self.fusion_dim = int(sum(projection_dims))
        if self.fusion_dim % attention_heads != 0:
            raise ValueError("The summed projection dimension must be divisible by attention_heads.")

        self.projections = nn.ModuleList(
            nn.Conv1d(source, target, kernel, padding=kernel // 2, bias=False)
            for source, target, kernel in zip(input_dims, projection_dims, convolution_kernels)
        )
        # The original NIAT pools the BERT [CLS] position. Cached event streams
        # have no [CLS], so a learned summary token is the faithful equivalent.
        self.summary_token = nn.Parameter(torch.zeros(1, 1, self.fusion_dim))
        nn.init.normal_(self.summary_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=attention_heads,
            dim_feedforward=self.fusion_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(self.fusion_dim),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch, events = _validate_streams(text, audio, visual, lengths)
        valid = _valid_mask(lengths, events, text.device)
        streams = []
        for value, projection in zip((text, audio, visual), self.projections):
            value = value.masked_fill(~valid.unsqueeze(-1), 0.0)
            projected = projection(value.transpose(1, 2)).transpose(1, 2)
            streams.append(projected.masked_fill(~valid.unsqueeze(-1), 0.0))
        fused_events = torch.cat(streams, dim=-1)
        summary = self.summary_token.expand(batch, -1, -1)
        tokens = torch.cat((summary, fused_events), dim=1)
        padding_mask = torch.cat(
            (torch.zeros(batch, 1, dtype=torch.bool, device=text.device), ~valid),
            dim=1,
        )
        return self.transformer(tokens, src_key_padding_mask=padding_mask)[:, 0]


class NIATCacheAdapter(nn.Module):
    """Binary clinical-screening adaptation of the official NIAT topology."""

    def __init__(
        self,
        input_dims: Sequence[int],
        projection_dims: Sequence[int] = (32, 16, 16),
        *,
        hidden_dim: int = 64,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fusion = NIATFusionEncoder(
            input_dims,
            projection_dims,
            transformer_layers=transformer_layers,
            attention_heads=attention_heads,
            dropout=dropout,
        )
        fusion_dim = self.fusion.fusion_dim
        # LayerNorm replaces official BatchNorm so participant batch size one is
        # numerically valid on small clinical folds.
        self.reconstruction = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.Tanh(),
        )
        self.discriminator = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode_clean(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion(text, audio, visual, lengths)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        lengths: torch.Tensor,
        *,
        corrupted: NIATCorruptedBatch | None = None,
        corruption_config: NIATCorruptionConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> NIATOutput:
        if corrupted is None:
            corrupted = corrupt_aligned_modalities(
                text,
                audio,
                visual,
                lengths,
                corruption_config,
                generator=generator,
            )
        clean_fused = self.fusion(text, audio, visual, lengths)
        corrupted_fused = self.fusion(
            corrupted.text,
            corrupted.audio,
            corrupted.visual,
            lengths,
        )
        return NIATOutput(
            clean_logits=self.classifier(clean_fused),
            corrupted_logits=self.classifier(corrupted_fused),
            clean_fused=clean_fused,
            corrupted_fused=corrupted_fused,
            reconstructed_fused=self.reconstruction(corrupted_fused),
            domain_clean_logits=self.discriminator(clean_fused),
            domain_corrupted_logits=self.discriminator(corrupted_fused),
        )

    @staticmethod
    def training_losses(
        output: NIATOutput,
        labels: torch.Tensor,
        *,
        class_weight: torch.Tensor | None = None,
        corrupted_task_weight: float = 1.0,
        reconstruction_weight: float = 0.1,
        adversarial_weight: float = 0.1,
    ) -> NIATLosses:
        """Compute generator/task losses; optimizer policy remains external."""

        if any(weight < 0.0 for weight in (corrupted_task_weight, reconstruction_weight, adversarial_weight)):
            raise ValueError("Loss weights must be non-negative.")
        clean = F.cross_entropy(output.clean_logits, labels, weight=class_weight)
        corrupted = F.cross_entropy(output.corrupted_logits, labels, weight=class_weight)
        reconstruction = F.l1_loss(output.reconstructed_fused, output.clean_fused)
        adversarial = F.binary_cross_entropy_with_logits(
            output.domain_corrupted_logits,
            torch.ones_like(output.domain_corrupted_logits),
        )
        classification = (clean + corrupted_task_weight * corrupted) / (1.0 + corrupted_task_weight)
        total = (
            classification
            + reconstruction_weight * reconstruction
            + adversarial_weight * adversarial
        )
        return NIATLosses(total, clean, corrupted, reconstruction, adversarial)

    def discriminator_loss(
        self,
        clean_fused: torch.Tensor,
        corrupted_fused: torch.Tensor,
    ) -> torch.Tensor:
        """Domain loss with detached fusion features for a discriminator step."""

        clean_logits = self.discriminator(clean_fused.detach())
        corrupted_logits = self.discriminator(corrupted_fused.detach())
        real = F.binary_cross_entropy_with_logits(clean_logits, torch.ones_like(clean_logits))
        fake = F.binary_cross_entropy_with_logits(corrupted_logits, torch.zeros_like(corrupted_logits))
        return 0.5 * (real + fake)


__all__ = [
    "NIATCacheAdapter",
    "NIATCorruptedBatch",
    "NIATCorruptionConfig",
    "NIATFusionEncoder",
    "NIATLosses",
    "NIATOutput",
    "corrupt_aligned_modalities",
]
