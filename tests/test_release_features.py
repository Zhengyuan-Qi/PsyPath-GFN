import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from deployment import (backbone_snapshot, contained_file, digest,
                        extraction_fingerprint, verify_extraction, verify_manifest)
from scripts.summarize_multipath import aggregate, losses, summary, TASKS, SEEDS, PROFILES

ROOT = Path(__file__).resolve().parents[1]


def record():
    return {"pid": "synthetic", "label": 1, "score": 1,
            "learned_probability": [0., 1.] * 24, "learned_severity": [0., 2.] * 24,
            "random_probability": [0.25] * 48, "random_severity": [0.5] * 48}


class AggregationTests(unittest.TestCase):
    def test_all_draws_not_first_prefix(self):
        self.assertEqual(losses([record()], "learned", 1), {"brier": .5, "mae": 1.})
        self.assertEqual(losses([record()], "learned", 2), {"brier": .25, "mae": 0.})
        self.assertEqual(losses([record()], "learned", 48), {"brier": .25, "mae": 0.})

    def test_unknown_score_is_excluded(self):
        missing = record()
        missing["score"] = None
        self.assertEqual(losses([record(), missing], "learned", 1)["mae"], 1.)

    def test_sample_sd(self):
        self.assertEqual(summary([1, 2, 3])["sd"], 1.)

    def test_invalid_draws(self):
        for key, value in (("learned_probability", [0.] * 47),
                           ("learned_probability", [float("nan")] * 48),
                           ("learned_probability", [1.5] * 48),
                           ("learned_severity", [float("inf")] * 48)):
            row = record()
            row[key] = value
            with self.assertRaises(ValueError):
                losses([row], "learned", 1)

    def make_inputs(self, directory):
        for task in TASKS:
            rows = [dict(record(), pid=str(i)) for i in range(47 if task == TASKS[0] else 56)]
            for seed in SEEDS:
                d = {"task": task, "seed": seed, "split": "test",
                     "label_profile": PROFILES[task], "rows": rows}
                (directory / f"{task}_seed_{seed}.json").write_text(json.dumps(d))

    def test_all_tasks_and_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_inputs(Path(directory))
            data = aggregate(directory)
            self.assertEqual(len(data["input_sha256"]), 15)
            for result in data["tasks"].values():
                self.assertEqual(result["learned"]["relative_reduction_percent"]["brier"]["48"]["mean"], 50.)
                self.assertEqual(result["learned"]["relative_reduction_percent"]["mae"]["48"]["mean"], 100.)
                self.assertAlmostEqual(result["paired_gap_48"]["brier"]["mean"], -.3125)
            self.assertNotIn('"pid"', json.dumps(data))

    def test_target_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inputs(root)
            path = root / f"{TASKS[0]}_seed_17.json"
            data = json.loads(path.read_text())
            data["rows"][0]["label"] = 0
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "targets differ"):
                aggregate(root)

    def test_profile_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inputs(root)
            path = root / f"{TASKS[0]}_seed_7.json"
            data = json.loads(path.read_text())
            data["label_profile"] = "depression_official_binary"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "label profile"):
                aggregate(root)

    def test_missing_seed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                aggregate(directory)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backbone = self.root / "backbone"
        self.backbone.mkdir()
        (self.backbone / "config.json").write_text("{}")
        (self.backbone / "model.safetensors").write_bytes(b"synthetic weight bytes")
        (self.backbone / "tokenizer.json").write_text("{}")
        (self.backbone / "chat_template.jinja").write_text("{{ messages }}")
        self.weights = self.root / "weights"
        self.weights.mkdir()
        (self.weights / "model.pt").write_bytes(b"synthetic checkpoint bytes")
        self.manifest = {"backbone_files": backbone_snapshot(self.backbone),
                         "tasks": {TASKS[0]: {"file": "model.pt",
                                             "sha256": digest(self.weights / "model.pt")}}}

    def save_manifest(self):
        (self.weights / "manifest.json").write_text(json.dumps(self.manifest))

    def test_manifest_roundtrip(self):
        self.save_manifest()
        path, _ = verify_manifest(self.weights, self.backbone, TASKS[0])
        self.assertEqual(path, (self.weights / "model.pt").resolve())
        self.assertIn("tokenizer.json", self.manifest["backbone_files"])
        self.assertIn("chat_template.jinja", self.manifest["backbone_files"])

    def test_changed_backbone_rejected(self):
        self.save_manifest()
        (self.backbone / "tokenizer.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "Backbone file differs"):
            verify_manifest(self.weights, self.backbone, TASKS[0])

    def test_changed_checkpoint_rejected(self):
        self.save_manifest()
        (self.weights / "model.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_manifest(self.weights, self.backbone, TASKS[0])

    def test_path_escape_rejected(self):
        for name in ("../model.pt", str(self.weights / "model.pt"), "..\\model.pt"):
            with self.assertRaises(ValueError):
                contained_file(self.weights, name)

    def test_missing_shard_rejected(self):
        (self.backbone / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "missing.safetensors"}}))
        with self.assertRaises(ValueError):
            backbone_snapshot(self.backbone)

    def test_extraction_contract(self):
        cache = self.root / "cache"
        (cache / "daic").mkdir(parents=True)
        contract = json.loads((ROOT / "configs/encoder_daic.json").read_text())
        contract["model_files_fingerprint_sha256"] = extraction_fingerprint(self.backbone)
        (cache / "daic/RUN_CONTRACT.json").write_text(json.dumps(contract))
        self.assertIn("daic", verify_extraction(cache, self.backbone, [TASKS[0]]))
        contract["view_contract"]["raw_labels_in_prompt"] = True
        (cache / "daic/RUN_CONTRACT.json").write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, "prompts differ"):
            verify_extraction(cache, self.backbone, [TASKS[0]])


class CommandTests(unittest.TestCase):
    def test_export_help_needs_no_model_dependencies(self):
        p = subprocess.run([sys.executable, str(ROOT / "prepare_weights.py"), "--help"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--runs-root", p.stdout)

    def test_dry_run_every_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            for experiment in ("main", "no-pim", "score-permutation", "reinforce",
                               "fixed-views", "pool-cls", "pool-mtl"):
                command = [sys.executable, str(ROOT / "reproduce.py"), experiment,
                           "--cache", directory, "--contract", directory,
                           "--formal-root", directory, "--output", str(Path(directory) / experiment),
                           "--tasks", TASKS[0], "--dry-run"]
                p = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn(TASKS[0], p.stdout)
                self.assertNotIn(TASKS[1], p.stdout)

    def test_inference_targets_absent(self):
        tree = ast.parse((ROOT / "online_pipeline.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict_views")
        strings = {n.value for n in ast.walk(function) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertFalse({"label", "score", "phq8"}.intersection(strings))


if __name__ == "__main__":
    unittest.main()
