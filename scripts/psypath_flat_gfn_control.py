"""Flat conditional GFN control: one joint event/bundle action per step."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from scripts.meter_gfn import EvidenceTrajectory, MODALITY_BUNDLES


class FlatConditionalEvidenceGFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        h = int(hidden_dim)
        self.hidden_dim = h
        self.event_encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, h), nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(2 * h, h), nn.GELU(), nn.LayerNorm(h),
        )
        # Match the hierarchical policy's capacity without latent routes.
        self.action_head = nn.Sequential(
            nn.Linear(3 * h + 1, 5 * h // 2), nn.GELU(),
            nn.Linear(5 * h // 2, len(MODALITY_BUNDLES)),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(2 * h + 1, h), nn.GELU(), nn.Linear(h, 1),
        )
        self.log_z = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def encode(self, events):
        encoded = self.event_encoder(events)
        context = self.context_encoder(torch.cat((
            encoded.mean(0), encoded.std(0, correction=0),
        )))
        return encoded, context

    def partition_from_context(self, context):
        return self.log_z(context).squeeze(-1)

    def partition(self, events):
        return self.partition_from_context(self.encode(events)[1])

    def action_logits(self, encoded, context, selected_events, maximum_events):
        selected = selected_events.clone()
        summary = encoded[selected].mean(0) if bool(selected.any()) else torch.zeros_like(context)
        progress = context.new_tensor([float(selected.sum()) / maximum_events])
        state = torch.cat((context, summary, progress))
        per_event = torch.cat((encoded, state.expand(len(encoded), -1)), dim=1)
        joint = self.action_head(per_event).masked_fill(selected[:, None], -torch.inf)
        stop = self.stop_head(state).squeeze(-1)
        return joint.flatten(), stop

    def trajectory(self, events, **kwargs):
        return self.trajectory_from_encoding(*self.encode(events), **kwargs)

    def trajectory_from_encoding(
        self, encoded, context, *, minimum_events, maximum_events,
        generator=None, greedy=False, exploration_epsilon=0.0,
    ):
        if not 1 <= minimum_events <= maximum_events <= len(encoded):
            raise ValueError("invalid event limits")
        if exploration_epsilon != 0.0:
            raise ValueError("The locked control uses on-policy sampling only")
        selected = torch.zeros(len(encoded), dtype=torch.bool, device=encoded.device)
        mask = torch.zeros((len(encoded), 4), dtype=torch.bool, device=encoded.device)
        bundles = MODALITY_BUNDLES.to(encoded.device)
        log_forward = context.new_zeros(())
        events, choices = [], []
        while len(events) < maximum_events:
            logits, stop = self.action_logits(encoded, context, selected, maximum_events)
            if len(events) >= minimum_events:
                logits = torch.cat((logits, stop[None]))
            log_probs = F.log_softmax(logits, dim=0)
            action = logits.argmax() if greedy else torch.multinomial(
                log_probs.exp(), 1, generator=generator,
            ).squeeze(0)
            log_forward = log_forward + log_probs[action]
            value = int(action)
            if value == len(encoded) * len(bundles):
                break
            event, bundle = divmod(value, len(bundles))
            selected[event] = True
            mask[event] = bundles[bundle]
            events.append(event)
            choices.append(bundle)
        return EvidenceTrajectory(
            phenotype=0, mask=mask, log_forward=log_forward,
            log_backward=context.new_tensor(-math.lgamma(len(events) + 1)),
            action_count=len(events),
            event_order=torch.tensor(events, dtype=torch.long, device=encoded.device),
            bundle_order=torch.tensor(choices, dtype=torch.long, device=encoded.device),
        )
