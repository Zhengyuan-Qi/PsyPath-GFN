"""Guards for the bounded formal comparison, before any clinical-data run."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts import run_psypath_alternating_structure_formal as runner


class FormalStructureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_locked_budget_and_common_initialization(self):
        self.assertEqual(sum(runner.SCHEDULE), 36)
        self.assertEqual(runner.SEEDS, (7, 17, 27, 41, 53))
        pair = runner.models(20, "daic_depression", 7)
        for name in ("event_encoder", "context_encoder", "log_z"):
            left = getattr(pair["flat_gfn"], name).state_dict()
            right = getattr(pair["hierarchical_gfn"], name).state_dict()
            for key in left:
                self.assertTrue(torch.equal(left[key], right[key]))

    def test_reject_nonzero_ensr(self):
        with self.assertRaises(ValueError):
            runner.fast_reward(None, None, [], ensr_weight=0.05)

    def test_test_requires_global_lock_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            runner.formal.atomic_json(path / "RUN_CONTRACT.json", {"code_sha256": runner.code_hashes()})
            with self.assertRaises(FileNotFoundError):
                runner.test(SimpleNamespace(output=path))

    def test_nonfinite_gradient_stops_before_update(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter])
        optimizer.register_step_pre_hook(runner.finite_step)
        parameter.grad = torch.tensor(float("nan"))
        with self.assertRaises(FloatingPointError):
            optimizer.step()
        np.testing.assert_equal(float(parameter.detach()), 1.0)


if __name__ == "__main__":
    unittest.main()
