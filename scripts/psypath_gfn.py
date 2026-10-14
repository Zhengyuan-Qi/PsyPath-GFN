#!/usr/bin/env python3
"""Core path policy for psychometric-supervised multimodal evidence GFlowNets.

The policy constructs a terminal evidence graph by visiting response events in
strict temporal order and selecting one non-empty modality bundle per event.
Because event indices are monotonic, every terminal graph has one valid reverse
trajectory: repeatedly remove the latest selected event, then return to source.
This gives an exact deterministic backward policy instead of approximating the
reverse probability from the number of selected events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class EvidencePath:
    """A sampled temporally monotonic event-modality evidence path."""

    phenotype: int
    mask: torch.Tensor
    event_order: torch.Tensor
    bundle_order: torch.Tensor
    log_forward: torch.Tensor
    log_backward: torch.Tensor
    action_count: int


def modality_bundles(view_count: int) -> torch.Tensor:
    """Enumerate every non-empty modality bundle for ``view_count`` views."""

    if view_count < 1:
        raise ValueError("view_count must be positive")
    return torch.tensor(
        [
            [bool(code & (1 << index)) for index in range(view_count)]
            for code in range(1, 2**view_count)
        ],
        dtype=torch.bool,
    )


class MonotonicPsychometricGFlowNet(nn.Module):
    """Instrument-conditioned GFlowNet over temporal event-modality paths.

    The public methods intentionally mirror the previous cached GFlowNet so the
    policy can be evaluated under the same locked Train/Dev/Test protocol.
    ``phenotype`` in the returned path is retained only for result-schema
    compatibility and stores the fixed instrument id; no latent phenotype is
    sampled or interpreted clinically.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 96,
        view_count: int = 4,
        instrument_count: int = 3,
        instrument_id: int = 0,
    ) -> None:
        super().__init__()
        if not 0 <= int(instrument_id) < int(instrument_count):
            raise ValueError("instrument_id is outside instrument_count")
        self.hidden_dim = int(hidden_dim)
        self.view_count = int(view_count)
        self.instrument_count = int(instrument_count)
        self.instrument_id = int(instrument_id)
        self.register_buffer("bundles", modality_bundles(view_count), persistent=False)

        self.event_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.instrument_embedding = nn.Embedding(instrument_count, hidden_dim)
        self.bundle_encoder = nn.Sequential(
            nn.Linear(view_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.event_query = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.modality_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.bundles)),
        )
        self.log_z = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _instrument(self, *, device: torch.device) -> torch.Tensor:
        index = torch.tensor(self.instrument_id, dtype=torch.long, device=device)
        return self.instrument_embedding(index)

    def encode(self, events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if events.ndim != 2 or len(events) == 0:
            raise ValueError(f"invalid event matrix: {tuple(events.shape)}")
        encoded = self.event_encoder(events)
        context = self.context_encoder(
            torch.cat((encoded.mean(dim=0), encoded.std(dim=0, correction=0)))
        )
        return encoded, context

    def partition(self, events: torch.Tensor) -> torch.Tensor:
        _, context = self.encode(events)
        return self.partition_from_context(context)

    def partition_from_context(self, context: torch.Tensor) -> torch.Tensor:
        instrument = self._instrument(device=context.device)
        return self.log_z(torch.cat((context, instrument))).squeeze(-1)

    def _selected_summary(
        self,
        encoded: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        active = selected_mask.any(dim=1)
        if not bool(active.any()):
            return torch.zeros(self.hidden_dim, dtype=encoded.dtype, device=encoded.device)
        bundle_state = self.bundle_encoder(selected_mask[active].to(encoded.dtype))
        return (encoded[active] + bundle_state).mean(dim=0)

    def _state(
        self,
        encoded: torch.Tensor,
        context: torch.Tensor,
        selected_mask: torch.Tensor,
        last_event: int,
        maximum_events: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        instrument = self._instrument(device=encoded.device)
        selected_summary = self._selected_summary(encoded, selected_mask)
        count = int(selected_mask.any(dim=1).sum().item())
        progress = torch.tensor(
            [float(count) / max(1, maximum_events)],
            dtype=encoded.dtype,
            device=encoded.device,
        )
        temporal_cursor = torch.tensor(
            [float(last_event + 1) / max(1, len(encoded))],
            dtype=encoded.dtype,
            device=encoded.device,
        )
        query = self.event_query(
            torch.cat(
                (
                    context,
                    instrument,
                    selected_summary,
                    context * instrument,
                    context * selected_summary,
                    progress,
                    temporal_cursor,
                )
            )
        )
        event_logits = (encoded * query).sum(dim=-1) / math.sqrt(self.hidden_dim)
        candidate = torch.arange(len(encoded), device=encoded.device) > int(last_event)
        event_logits = event_logits.masked_fill(~candidate, -torch.inf)
        stop_logit = self.stop_head(
            torch.cat(
                (
                    context,
                    instrument,
                    selected_summary,
                    context * instrument,
                    progress,
                    temporal_cursor,
                )
            )
        ).squeeze(-1)
        return event_logits, stop_logit, instrument, selected_summary

    def trajectory(
        self,
        events: torch.Tensor,
        *,
        minimum_events: int,
        maximum_events: int,
        generator: torch.Generator | None = None,
        greedy: bool = False,
    ) -> EvidencePath:
        encoded, context = self.encode(events)
        return self.trajectory_from_encoding(
            encoded,
            context,
            minimum_events=minimum_events,
            maximum_events=maximum_events,
            generator=generator,
            greedy=greedy,
        )

    def trajectory_from_encoding(
        self,
        encoded: torch.Tensor,
        context: torch.Tensor,
        *,
        minimum_events: int,
        maximum_events: int,
        generator: torch.Generator | None = None,
        greedy: bool = False,
    ) -> EvidencePath:
        minimum_events = max(1, min(int(minimum_events), len(encoded)))
        maximum_events = max(minimum_events, min(int(maximum_events), len(encoded)))
        selected_mask = torch.zeros(
            (len(encoded), self.view_count), dtype=torch.bool, device=encoded.device
        )
        event_order: list[int] = []
        bundle_order: list[int] = []
        log_forward = encoded.new_zeros(())
        last_event = -1

        while len(event_order) < maximum_events:
            event_logits, stop_logit, instrument, selected_summary = self._state(
                encoded,
                context,
                selected_mask,
                last_event,
                maximum_events,
            )
            remaining_required = max(0, minimum_events - len(event_order))
            if remaining_required:
                # A monotonic policy cannot move backwards. Before the minimum
                # path length is reached, retain only choices that leave enough
                # later events to complete a valid trajectory.
                latest_feasible = len(encoded) - remaining_required
                feasible = torch.arange(len(encoded), device=encoded.device) <= latest_feasible
                event_logits = event_logits.masked_fill(~feasible, -torch.inf)
            candidate_indices = torch.where(torch.isfinite(event_logits))[0]
            can_stop = len(event_order) >= minimum_events
            if len(candidate_indices) == 0:
                if not can_stop:
                    raise RuntimeError("monotonic path exhausted before minimum_events")
                break

            candidate_logits = event_logits[candidate_indices]
            if can_stop:
                choice_logits = torch.cat((candidate_logits, stop_logit[None]))
            else:
                choice_logits = candidate_logits
            choice_log_probability = F.log_softmax(choice_logits, dim=0)
            if greedy:
                choice = choice_logits.argmax()
            else:
                choice = torch.multinomial(
                    choice_log_probability.exp(), 1, generator=generator
                ).squeeze(0)
            log_forward = log_forward + choice_log_probability[choice]
            if can_stop and int(choice.item()) == len(candidate_indices):
                break

            event_index = int(candidate_indices[int(choice.item())].item())
            modality_logits = self.modality_head(
                torch.cat(
                    (
                        encoded[event_index],
                        context,
                        instrument,
                        selected_summary,
                        torch.tensor(
                            [float(event_index) / max(1, len(encoded) - 1)],
                            dtype=encoded.dtype,
                            device=encoded.device,
                        ),
                    )
                )
            )
            modality_log_probability = F.log_softmax(modality_logits, dim=0)
            if greedy:
                bundle_index = modality_logits.argmax()
            else:
                bundle_index = torch.multinomial(
                    modality_log_probability.exp(), 1, generator=generator
                ).squeeze(0)
            log_forward = log_forward + modality_log_probability[bundle_index]
            selected_mask[event_index] = self.bundles[bundle_index].to(encoded.device)
            event_order.append(event_index)
            bundle_order.append(int(bundle_index.item()))
            last_event = event_index

        # Strict temporal monotonicity gives every terminal state one parent at
        # each reverse step, so the exact backward-policy log probability is 0.
        log_backward = encoded.new_zeros(())
        return EvidencePath(
            phenotype=self.instrument_id,
            mask=selected_mask,
            event_order=torch.tensor(event_order, dtype=torch.long, device=encoded.device),
            bundle_order=torch.tensor(bundle_order, dtype=torch.long, device=encoded.device),
            log_forward=log_forward,
            log_backward=log_backward,
            action_count=len(event_order),
        )


class MultimodalMonotonicPsychometricGFlowNet(nn.Module):
    """Joint policy over temporally ordered event and evidence-channel actions.

    Unlike the cached compatibility policy above, this class scores every
    ``(event, non-empty channel bundle)`` action from the corresponding channel
    states. It is the policy used by the online PsyPath model, where the input
    has shape ``[events, channels, features]``.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 96,
        channel_count: int = 4,
        instrument_count: int = 3,
    ) -> None:
        super().__init__()
        if channel_count < 1:
            raise ValueError("channel_count must be positive")
        if instrument_count < 1:
            raise ValueError("instrument_count must be positive")
        self.hidden_dim = int(hidden_dim)
        self.channel_count = int(channel_count)
        self.instrument_count = int(instrument_count)
        self.register_buffer(
            "bundles", modality_bundles(channel_count), persistent=False
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.channel_embedding = nn.Parameter(torch.empty(channel_count, hidden_dim))
        nn.init.normal_(self.channel_embedding, std=0.02)
        self.bundle_encoder = nn.Sequential(
            nn.Linear(channel_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.instrument_embedding = nn.Embedding(instrument_count, hidden_dim)
        self.action_query = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_z = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if nodes.ndim != 3 or len(nodes) == 0:
            raise ValueError(f"invalid evidence-node tensor: {tuple(nodes.shape)}")
        if nodes.shape[1] != self.channel_count:
            raise ValueError(
                f"channel mismatch: received={nodes.shape[1]} expected={self.channel_count}"
            )
        encoded = self.node_encoder(nodes) + self.channel_embedding[None, :, :]
        pooled = encoded.mean(dim=1)
        context = self.context_encoder(
            torch.cat((pooled.mean(dim=0), pooled.std(dim=0, correction=0)))
        )
        return encoded, context

    def partition_from_context(
        self, context: torch.Tensor, instrument_id: int
    ) -> torch.Tensor:
        instrument = self._instrument(instrument_id, context.device)
        return self.log_z(torch.cat((context, instrument))).squeeze(-1)

    def partition(self, nodes: torch.Tensor, instrument_id: int) -> torch.Tensor:
        _, context = self.encode(nodes)
        return self.partition_from_context(context, instrument_id)

    def _instrument(self, instrument_id: int, device: torch.device) -> torch.Tensor:
        if not 0 <= int(instrument_id) < self.instrument_count:
            raise ValueError("instrument_id is outside instrument_count")
        index = torch.tensor(int(instrument_id), dtype=torch.long, device=device)
        return self.instrument_embedding(index)

    def _selected_summary(
        self, encoded: torch.Tensor, selected_mask: torch.Tensor
    ) -> torch.Tensor:
        if not bool(selected_mask.any()):
            return torch.zeros(
                self.hidden_dim, dtype=encoded.dtype, device=encoded.device
            )
        # The trajectory builder updates its mask after each action. Index from
        # a snapshot so autograd never observes an in-place version change.
        mask_snapshot = selected_mask.detach().clone()
        return encoded[mask_snapshot].mean(dim=0)

    def _bundle_states(self, encoded: torch.Tensor) -> torch.Tensor:
        weights = self.bundles.to(device=encoded.device, dtype=encoded.dtype)
        normalized = weights / weights.sum(dim=1, keepdim=True)
        states = torch.einsum("bc,tch->tbh", normalized, encoded)
        return states + self.bundle_encoder(weights)[None, :, :]

    def _action_logits(
        self,
        encoded: torch.Tensor,
        context: torch.Tensor,
        selected_mask: torch.Tensor,
        last_event: int,
        maximum_events: int,
        instrument_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        instrument = self._instrument(instrument_id, encoded.device)
        selected = self._selected_summary(encoded, selected_mask)
        count = int(selected_mask.any(dim=1).sum().item())
        progress = encoded.new_tensor([float(count) / max(1, maximum_events)])
        cursor = encoded.new_tensor(
            [float(last_event + 1) / max(1, len(encoded))]
        )
        query = self.action_query(
            torch.cat(
                (
                    context,
                    instrument,
                    selected,
                    context * instrument,
                    context * selected,
                    progress,
                    cursor,
                )
            )
        )
        bundle_states = self._bundle_states(encoded)
        action_logits = torch.einsum("tbh,h->tb", bundle_states, query)
        action_logits = action_logits / math.sqrt(self.hidden_dim)
        candidate = torch.arange(len(encoded), device=encoded.device) > int(last_event)
        action_logits = action_logits.masked_fill(~candidate[:, None], -torch.inf)
        stop_logit = self.stop_head(
            torch.cat(
                (
                    context,
                    instrument,
                    selected,
                    context * instrument,
                    progress,
                    cursor,
                )
            )
        ).squeeze(-1)
        return action_logits, stop_logit

    def trajectory(
        self,
        nodes: torch.Tensor,
        *,
        instrument_id: int,
        minimum_events: int,
        maximum_events: int,
        generator: torch.Generator | None = None,
        greedy: bool = False,
    ) -> EvidencePath:
        encoded, context = self.encode(nodes)
        return self.trajectory_from_encoding(
            encoded,
            context,
            instrument_id=instrument_id,
            minimum_events=minimum_events,
            maximum_events=maximum_events,
            generator=generator,
            greedy=greedy,
        )

    def trajectory_from_encoding(
        self,
        encoded: torch.Tensor,
        context: torch.Tensor,
        *,
        instrument_id: int,
        minimum_events: int,
        maximum_events: int,
        generator: torch.Generator | None = None,
        greedy: bool = False,
    ) -> EvidencePath:
        minimum_events = max(1, min(int(minimum_events), len(encoded)))
        maximum_events = max(
            minimum_events, min(int(maximum_events), len(encoded))
        )
        selected_mask = torch.zeros(
            (len(encoded), self.channel_count),
            dtype=torch.bool,
            device=encoded.device,
        )
        event_order: list[int] = []
        bundle_order: list[int] = []
        log_forward = encoded.new_zeros(())
        last_event = -1
        bundle_count = len(self.bundles)

        while len(event_order) < maximum_events:
            action_logits, stop_logit = self._action_logits(
                encoded,
                context,
                selected_mask,
                last_event,
                maximum_events,
                instrument_id,
            )
            remaining_required = max(0, minimum_events - len(event_order))
            if remaining_required:
                latest_feasible = len(encoded) - remaining_required
                feasible = (
                    torch.arange(len(encoded), device=encoded.device)
                    <= latest_feasible
                )
                action_logits = action_logits.masked_fill(
                    ~feasible[:, None], -torch.inf
                )
            flat = action_logits.reshape(-1)
            candidate_indices = torch.where(torch.isfinite(flat))[0]
            can_stop = len(event_order) >= minimum_events
            if len(candidate_indices) == 0:
                if not can_stop:
                    raise RuntimeError(
                        "multimodal monotonic path exhausted before minimum_events"
                    )
                break
            candidate_logits = flat[candidate_indices]
            choice_logits = (
                torch.cat((candidate_logits, stop_logit[None]))
                if can_stop
                else candidate_logits
            )
            log_probability = F.log_softmax(choice_logits, dim=0)
            if greedy:
                choice = choice_logits.argmax()
            else:
                choice = torch.multinomial(
                    log_probability.exp(), 1, generator=generator
                ).squeeze(0)
            log_forward = log_forward + log_probability[choice]
            if can_stop and int(choice.item()) == len(candidate_indices):
                break
            flat_index = int(candidate_indices[int(choice.item())].item())
            event_index = flat_index // bundle_count
            bundle_index = flat_index % bundle_count
            selected_mask[event_index] = self.bundles[bundle_index]
            event_order.append(event_index)
            bundle_order.append(bundle_index)
            last_event = event_index

        return EvidencePath(
            phenotype=int(instrument_id),
            mask=selected_mask,
            event_order=torch.tensor(
                event_order, dtype=torch.long, device=encoded.device
            ),
            bundle_order=torch.tensor(
                bundle_order, dtype=torch.long, device=encoded.device
            ),
            log_forward=log_forward,
            log_backward=encoded.new_zeros(()),
            action_count=len(event_order),
        )


def trajectory_balance_loss(
    model: MonotonicPsychometricGFlowNet
    | MultimodalMonotonicPsychometricGFlowNet,
    trajectories: list[EvidencePath],
    log_rewards: list[torch.Tensor],
    *,
    context: torch.Tensor,
    instrument_id: int | None = None,
) -> torch.Tensor:
    """Trajectory Balance with the exact deterministic backward policy."""

    if len(trajectories) != len(log_rewards) or not trajectories:
        raise ValueError("trajectory/reward mismatch")
    if isinstance(model, MultimodalMonotonicPsychometricGFlowNet):
        if instrument_id is None:
            raise ValueError("instrument_id is required for the multimodal policy")
        log_z = model.partition_from_context(context, instrument_id)
    else:
        log_z = model.partition_from_context(context)
    residual = torch.stack(
        [
            log_z + path.log_forward - path.log_backward - reward
            for path, reward in zip(trajectories, log_rewards)
        ]
    )
    return residual.square().mean()
