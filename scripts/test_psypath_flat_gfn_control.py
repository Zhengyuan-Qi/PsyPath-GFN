"""Focused invariants for the same-reward structural control."""

import unittest

import numpy as np
import torch

from scripts.psypath_flat_gfn_control import FlatConditionalEvidenceGFN
from scripts.meter_gfn import HierarchicalClinicalGFlowNet


class FlatControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.model = FlatConditionalEvidenceGFN(20)
        self.events = torch.randn(5, 20)

    def test_capacity_and_parameter_gradients(self):
        baseline = HierarchicalClinicalGFlowNet(20, hidden_dim=64, phenotype_count=4)
        counts = [sum(p.numel() for p in model.parameters()) for model in (baseline, self.model)]
        self.assertLess(max(counts) / min(counts), 1.10)
        trajectory = self.model.trajectory(self.events, minimum_events=2, maximum_events=4)
        loss = (trajectory.log_forward + self.model.partition(self.events) - trajectory.log_backward).square()
        loss.backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_legal_actions_and_seed_replay(self):
        for seed in range(10):
            values = []
            for _ in range(2):
                generator = torch.Generator().manual_seed(seed)
                trajectory = self.model.trajectory(self.events, minimum_events=2,
                                                    maximum_events=4, generator=generator)
                self.assertGreaterEqual(trajectory.action_count, 2)
                self.assertLessEqual(trajectory.action_count, 4)
                self.assertEqual(len(set(trajectory.event_order.tolist())), trajectory.action_count)
                self.assertEqual(int(trajectory.mask.any(1).sum()), trajectory.action_count)
                self.assertTrue(bool(torch.isfinite(trajectory.log_forward)))
                values.append(trajectory)
            self.assertTrue(torch.equal(values[0].mask, values[1].mask))
            self.assertEqual(float(values[0].log_forward), float(values[1].log_forward))

    def test_selected_event_all_bundles_masked(self):
        encoded, context = self.model.encode(self.events)
        selected = torch.tensor([False, True, False, False, False])
        logits, stop = self.model.action_logits(encoded, context, selected, 4)
        self.assertTrue(bool(torch.isneginf(logits.reshape(5, 15)[1]).all()))
        probability = torch.cat((logits, stop[None])).softmax(0)
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=6)
        self.assertEqual(float(probability[:75].reshape(5, 15)[1].sum()), 0.0)

    def test_complete_tiny_trajectory_probability(self):
        encoded, context = self.model.encode(self.events[:2])
        first, _ = self.model.action_logits(encoded, context, torch.zeros(2, dtype=torch.bool), 2)
        total = 0.0
        for action, mass in enumerate(first.softmax(0).detach().numpy()):
            selected = torch.zeros(2, dtype=torch.bool)
            selected[action // 15] = True
            second, stop = self.model.action_logits(encoded, context, selected, 2)
            total += float(mass * torch.cat((second, stop[None])).softmax(0).sum().detach())
        np.testing.assert_allclose(total, 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
