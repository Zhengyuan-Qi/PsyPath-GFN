"""Technical tests for the locked review controls; no clinical data required."""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from scripts import run_psypath_review_controls as r


class ReviewControlTests(unittest.TestCase):
    def test_random_augmentation_is_independent_of_targets(self):
        args = SimpleNamespace(minimum_events=2, maximum_events=16,
                               maximum_fraction=.3, operator_trajectories=4)
        base = {"events": torch.zeros(30, 4), "pid": "synthetic", "label": 0, "score": 2.}
        changed = dict(base, label=1, score=23.)
        a = r.random_augmentations(None, base, args, cycle=1, seed=7)
        b = r.random_augmentations(None, changed, args, cycle=1, seed=7)
        self.assertEqual(len(a), 5)
        self.assertTrue(a[0].all())
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        for mask in a[1:]:
            self.assertLessEqual(mask.any(1).sum(), 9)
            self.assertGreaterEqual(mask.any(1).sum(), 2)

    def test_score_reader_requests_only_allowed_splits(self):
        calls = []
        def fake(root, dataset, profile, split, include_unlabeled):
            calls.append(split)
            return pd.DataFrame([{"pid": 1 if split == "train" else 2,
                                  "split": split, "score": 4., "mask": 1}]), SimpleNamespace(
                                      target_column="score", mask_column="mask")
        task = {"dataset": "daic", "score_profile": "depression_phq_score"}
        with mock.patch.object(r, "load_label_profile", side_effect=fake):
            values = r.isolated_score_map("unused", task, ("train", "dev"))
        self.assertEqual(calls, ["train", "dev"])
        self.assertEqual(set(values), {("1", "train"), ("2", "dev")})

    def test_missing_score_is_not_filled(self):
        frame = pd.DataFrame([{"pid": 1, "split": "train", "score": float("nan"), "mask": 0}])
        with mock.patch.object(r, "load_label_profile", return_value=(frame, SimpleNamespace(
                target_column="score", mask_column="mask"))):
            result = r.isolated_score_map("unused", {"dataset": "daic", "score_profile": "depression_phq_score"}, ("train",))
        self.assertIsNone(result[("1", "train")])

    def test_predict_preserves_supplied_threshold(self):
        args = SimpleNamespace(minimum_events=2, maximum_events=16, maximum_fraction=.3)
        records = [{"events": torch.zeros(10, 4), "pid": str(i), "label": i,
                    "score": float(i), "projected_np": {k: np.ones((10, 2)) for k in
                        ("clinical", "answer", "context", "visual")}} for i in (0, 1)]
        class Operator:
            def predict(self, features):
                return {"probability": np.repeat(.6, len(features)), "severity": np.repeat(.5, len(features))}
        with mock.patch.object(r.b, "choose_threshold", side_effect=AssertionError("test threshold tuning")):
            result = r.predict(records, Operator(), args, seed=7, samples=3, threshold=.7)
        self.assertEqual(result["threshold"], .7)
        self.assertEqual(result["posterior"]["sensitivity"], 0.)


if __name__ == "__main__":
    unittest.main()
