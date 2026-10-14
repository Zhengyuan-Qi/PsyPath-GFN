#!/usr/bin/env python3
"""Run matched evidence controls; lock every run before test evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

TASKS = ("daic_depression", "edaic_depression", "edaic_ptsd")
SEEDS = (7, 17, 27, 41, 53)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--main-root", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        p.error("workers must be between one and eight")
    a.project, a.output = a.project.resolve(), a.output.resolve()
    cache = a.cache.resolve()
    contract = a.contract.resolve()
    main_root = a.main_root.resolve()
    script = Path(__file__).with_name("run_psypath_review_controls.py")
    pairs = [(task, seed) for task in TASKS for seed in SEEDS]

    def command_for(pair, stage):
        task, seed = pair
        ref = main_root / task / "development_checkpoints" / task / "gflownet" / f"seed_{seed}.pt"
        return [sys.executable, "-u", str(script), stage, "--cache", str(cache),
                "--contract", str(contract), "--reference", str(ref),
                "--output", str(a.output), "--task", task, "--seed", str(seed)]

    if a.dry_run:
        print(json.dumps({"development": [command_for(pair, "develop") for pair in pairs],
                          "test_after_all_development_locks": [command_for(pair, "test") for pair in pairs]}, indent=2))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "SUITE_STARTED.json").exists():
        p.error("Use a new output directory; do not overwrite a previous suite")
    protocol = {"tasks": TASKS, "seeds": SEEDS, "workers": a.workers,
                "source_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "test_gate": "all 15 development runs locked before any new test load"}
    (a.output / "SUITE_STARTED.json").write_text(json.dumps(protocol, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    running, mutex, aborted = {}, threading.Lock(), threading.Event()

    def work(pair, stage):
        task, seed = pair
        if aborted.is_set():
            return
        logdir = a.output / "logs"
        logdir.mkdir(exist_ok=True)
        command = command_for(pair, stage)
        with (logdir / f"{task}_{seed}_{stage}.log").open("w") as log:
            with mutex:
                if aborted.is_set():
                    return
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                running[(task, seed)] = process
            code = process.wait()
            with mutex:
                running.pop((task, seed), None)
                if code:
                    aborted.set()
                    for child in running.values():
                        child.terminate()
            if code:
                raise RuntimeError(f"Technical gate failed: {task}/{seed}/{stage}; see log")
        print(task, seed, stage, "complete", flush=True)

    try:
        for stage in ("develop", "test"):
            if stage == "test":
                locks = {}
                for task, seed in pairs:
                    directory = a.output / task / f"seed_{seed}"
                    complete = json.loads((directory / "DEVELOPMENT_COMPLETE.json").read_text())
                    if complete["smoke"] or complete["test_loaded"]:
                        raise RuntimeError("Invalid development completion marker")
                    for arm in complete["arms"]:
                        path = directory / f"{arm}_LOCK.json"
                        lock = json.loads(path.read_text())
                        assert lock["development_evaluations"] == 18
                        ckpt = directory / f"{arm}.pt"
                        assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == lock["checkpoint_sha256"]
                        locks[str(path.relative_to(a.output))] = hashlib.sha256(path.read_bytes()).hexdigest()
                (a.output / "ALL_DEVELOPMENT_LOCKED.json").write_text(json.dumps(locks, indent=2))
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                futures = [pool.submit(work, pair, stage) for pair in pairs]
                for future in as_completed(futures):
                    future.result()
        (a.output / "COMPLETE.json").write_text(json.dumps({"status": "complete", **protocol}, indent=2))
    except BaseException as error:
        (a.output / "FAILED.json").write_text(json.dumps({"error": repr(error)}, indent=2))
        raise


if __name__ == "__main__":
    main()
