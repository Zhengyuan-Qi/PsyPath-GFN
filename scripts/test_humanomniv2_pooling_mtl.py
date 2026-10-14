import unittest

import numpy as np
import torch
from sklearn.metrics import f1_score

from scripts.run_humanomniv2_pooling_mtl import objective, threshold, PoolHead


class PoolingTests(unittest.TestCase):
    def test_threshold_matches_reference(self):
        rng = np.random.default_rng(29)
        for _ in range(5):
            y = rng.integers(0, 2, 35)
            p = rng.random(35)
            grid = np.linspace(.1, .9, 161)
            expected = max(grid, key=lambda t: (
                f1_score(y, p >= t, average="weighted", zero_division=0),
                f1_score(y, p >= t, average="macro", zero_division=0), -abs(t-.5)))
            self.assertAlmostEqual(threshold(y, p), expected)

    def test_missing_scores_do_not_create_targets(self):
        logits = torch.tensor([0., 0.], requires_grad=True)
        scores = torch.tensor([.2, .6], requires_grad=True)
        y = torch.tensor([0., 1.]); q = torch.tensor([float("nan"), .8])
        loss = objective(logits, scores, y, q, torch.tensor(1.), .3)
        loss.backward()
        self.assertEqual(float(scores.grad[0]), 0.)
        self.assertAlmostEqual(float(scores.grad[1]), -.3)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_scores_missing(self):
        head = PoolHead(344, 64, .1)
        logits, q = head(torch.ones(3,344))
        loss = objective(logits, q, torch.ones(3), torch.full((3,),float("nan")),torch.tensor(1.),.3)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(head.score.weight.grad, torch.zeros_like(head.score.weight.grad)))


if __name__ == "__main__":
    unittest.main()
