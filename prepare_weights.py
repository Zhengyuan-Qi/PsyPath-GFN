"""Export development-selected training checkpoints for predict.py; no training."""
import argparse
import copy
import json
from pathlib import Path

from deployment import TASKS, SEEDS, backbone_snapshot, digest, verify_extraction

KEEP = ("schema", "task", "variant", "method", "seed", "selected_cycle",
        "selected_epoch", "threshold", "event_projections",
        "psychometric_operator", "gflownet_state")


def export(runs_root, cache, backbone, output, tasks=TASKS, seed=7):
    import numpy as np
    import torch
    from scripts.validate_alternating_psychometric_gfn_cached import load_trusted_checkpoint

    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory; no existing artifacts are overwritten")
    contracts = verify_extraction(cache, backbone, tasks)
    backbone_files = backbone_snapshot(backbone)
    sources = {}
    for task in tasks:
        path = Path(runs_root) / task / "development_checkpoints" / task / "gflownet" / f"seed_{seed}.pt"
        ck = load_trusted_checkpoint(path, torch.device("cpu"))
        if (ck["task"], ck["seed"], ck["variant"], ck["method"]) != (task, seed, "no_ensr", "gflownet"):
            raise ValueError(f"Wrong task, seed or method: {task}")
        if ck.get("stage") != "development_locked":
            raise ValueError("Use the development-selected checkpoint from reproduce.py main")
        sources[task] = (path, ck)
    output.mkdir(parents=True)
    manifest = {"schema": "psypath_deployment_v1", "backbone_files": backbone_files,
                "extraction_contract_sha256": contracts, "tasks": {}}
    for task, (source, ck) in sources.items():
        cleaned = {key: copy.deepcopy(ck[key]) for key in KEEP}
        # Runtime reconstruction uses scalar settings, not author filesystem paths.
        cleaned["config"] = {k: v for k, v in ck["config"].items()
                             if not isinstance(v, Path)
                             and not (isinstance(v, str) and ("/" in v or "\\" in v))}
        destination = output / f"{task}_seed{seed}.pt"
        torch.save(cleaned, destination)
        restored = load_trusted_checkpoint(destination, torch.device("cpu"))
        for name, value in ck["gflownet_state"].items():
            if not torch.equal(value, restored["gflownet_state"][name]):
                raise ValueError(f"Export changed policy parameter: {name}")
        dimension = ck["psychometric_operator"].projection.scaler.n_features_in_
        x = np.random.default_rng(1049).normal(size=(5, dimension))
        original = ck["psychometric_operator"].predict(x)
        reloaded = restored["psychometric_operator"].predict(x)
        for key in original:
            np.testing.assert_array_equal(original[key], reloaded[key])
        manifest["tasks"][task] = {
            "file": destination.name, "sha256": digest(destination),
            "source_sha256": digest(source), "seed": seed,
        }
    manifest["scope"] = f"One development-selected model per task, seed {seed}; not a five-seed average."
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=7)
    args = parser.parse_args()
    result = export(args.runs_root, args.cache, args.backbone, args.output, args.tasks, args.seed)
    print(json.dumps({"output": str(args.output), "tasks": list(result["tasks"]), "seed": args.seed}))


if __name__ == "__main__":
    main()
