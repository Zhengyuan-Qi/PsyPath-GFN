import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from scripts import train_meter_mh_sequential_increment as loader


class ScoreIsolationTests(unittest.TestCase):
    def test_training_reader_never_requests_test_scores(self):
        calls = []

        def read(root, dataset, profile, *, split, include_unlabeled):
            calls.append(split)
            if split == "test":
                raise AssertionError("Test score access during development")
            return pd.DataFrame([dict(pid=1, split=split, value=7, known=1)]), SimpleNamespace(
                target_column="value", mask_column="known")

        with mock.patch.object(loader, "load_label_profile", side_effect=read):
            result = loader.score_map("unused", loader.TASKS["daic_depression"], ("train", "dev"))
        self.assertEqual(calls, ["train", "dev"])
        self.assertEqual(result, {("1", "train"): 7., ("1", "dev"): 7.})

    def test_unknown_score_remains_missing(self):
        frame = pd.DataFrame([dict(pid=1, split="train", value=7, known=0)])
        spec = SimpleNamespace(target_column="value", mask_column="known")
        with mock.patch.object(loader, "load_label_profile", return_value=(frame, spec)):
            result = loader.score_map("unused", loader.TASKS["daic_depression"], ("train",))
        self.assertIsNone(result[("1", "train")])


if __name__ == "__main__":
    unittest.main()
