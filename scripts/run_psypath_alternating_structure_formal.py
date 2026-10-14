#!/usr/bin/env python3
"""Bounded, paired structure ablation using the original alternating pipeline."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import validate_alternating_psychometric_gfn_cached as formal
from scripts.alternating_psychometric_gfn import evidence_features
from scripts.meter_gfn import HierarchicalClinicalGFlowNet
from scripts.psypath_flat_gfn_control import FlatConditionalEvidenceGFN
from scripts.run_psypath_same_reward_control import settings as base_settings

SCHEMA = "psypath_alternating_structure_formal_v1"
TASKS = ("daic_depression", "edaic_depression", "edaic_ptsd")
SEEDS = (7, 17, 27, 41, 53)
ARMS = ("flat_gfn", "hierarchical_gfn")
SCHEDULE = (6, 6, 24)
REFERENCE_REWARD = formal.operator_log_rewards
CODE_FILES = (
    "scripts/run_psypath_alternating_structure_formal.py",
    "scripts/psypath_flat_gfn_control.py",
    "scripts/run_psypath_same_reward_control.py",
    "scripts/validate_alternating_psychometric_gfn_cached.py",
    "scripts/alternating_psychometric_gfn.py", "scripts/meter_gfn.py",
    "scripts/clinical_label_contract.py",
    "protocols/psypath_alternating_structure_formal_20260908.md",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **values):
    print(json.dumps({"time": now(), "event": event, **values}, default=formal.json_default), flush=True)


def config():
    args = base_settings()
    args.ensr_reward_weight = 0.20
    args.sufficiency_weight = 1.0
    args.necessity_margin = 0.05
    args.specificity_margin = 0.02
    args.path_process = False
    args.path_process_weight = 0.15
    args.path_monotonic_weight = 0.10
    args.path_order_weight = 0.15
    args.path_order_margin = 0.02
    args.path_duplicate_weight = 0.0
    args.entropy_weight = 0.01
    args.policy = "legacy"
    args.development_samples = 16
    args.test_samples = 48
    return args


def code_hashes():
    return {path: formal.sha256_file(ROOT / path) for path in CODE_FILES}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pair_dir(args):
    return args.output / args.task / f"seed_{args.seed}"


def arm_dir(args):
    return pair_dir(args) / args.arm


def check_contract(args):
    contract = load_json(args.output / "RUN_CONTRACT.json")
    if contract["code_sha256"] != code_hashes():
        raise RuntimeError("Source/protocol changed after contract lock")
    return contract


def models(input_dim, task, seed):
    # The hierarchical initialization is identical to the original formal run.
    formal.seed_all(formal.stable_seed(task, seed, "paired_model"))
    hierarchical = HierarchicalClinicalGFlowNet(input_dim, hidden_dim=64, phenotype_count=4)
    flat = FlatConditionalEvidenceGFN(input_dim, hidden_dim=64)
    for name in ("event_encoder", "context_encoder", "log_z"):
        getattr(flat, name).load_state_dict(getattr(hierarchical, name).state_dict())
    return {"hierarchical_gfn": hierarchical, "flat_gfn": flat}


def fast_reward(operator, projected, trajectories, **kwargs):
    for name in ("ensr_weight", "path_process_weight", "path_monotonic_weight",
                 "path_order_weight", "path_duplicate_weight"):
        if kwargs.get(name, 0.0) != 0.0:
            raise ValueError(f"Fast no-ENSR reward is invalid for {name}")
    masks = [trajectory.mask.detach().cpu().numpy().astype(bool) for trajectory in trajectories]
    predicted = operator.predict(np.stack([evidence_features(projected, mask) for mask in masks]))
    probability = predicted["probability"]
    label = kwargs["label"]
    cost = -(label * np.log(probability) + (1 - label) * np.log(1 - probability))
    score = kwargs["score"]
    if score is not None:
        cost += kwargs["pim_weight"] * np.abs(predicted["severity"] - score) / operator.score_max
    cost += kwargs["compactness_weight"] * np.asarray([mask.any(1).mean() for mask in masks])
    return [{"log_reward": float(-value / kwargs["temperature"])} for value in cost]


def prepare(args):
    check_contract(args)
    started = time.time()
    directory = pair_dir(args)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "shared_train_dev.pt"
    if destination.exists():
        raise FileExistsError(destination)
    raw = formal.load_records(args.meter_root, args.contract_dir, args.task,
                              allowed_splits=("train", "dev"), split_policy="official")
    formal.attach_temporal(raw, args.meter_root, formal.TASKS[args.task]["dataset"])
    projections = formal.fit_event_projections(
        raw["train"], formal.stable_seed(args.task, args.seed, "event_projection"))
    train = formal.prepare_records(raw["train"], projections, torch.device("cpu"))
    dev = formal.prepare_records(raw["dev"], projections, torch.device("cpu"))
    train_pids, dev_pids = [x["pid"] for x in train], [x["pid"] for x in dev]
    if len(set(train_pids)) != len(train_pids) or len(set(dev_pids)) != len(dev_pids):
        raise RuntimeError("Duplicate participants")
    if set(train_pids) & set(dev_pids):
        raise RuntimeError("Train/Dev PID overlap")
    settings = config()
    operator = formal.fit_operator(None, train, settings, task_name=args.task,
                                  variant="no_ensr", cycle=0, seed=args.seed, permuted_scores={})
    full_operator = formal.fit_full_evidence_operator(
        train, settings, task_name=args.task, variant="no_ensr", seed=args.seed, permuted_scores={})
    pair = models(int(train[0]["events"].shape[1]), args.task, args.seed)
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in pair.items()}
    if max(counts.values()) / min(counts.values()) >= 1.10:
        raise RuntimeError(f"Parameter mismatch: {counts}")
    torch.save({"schema": SCHEMA, "task": args.task, "seed": args.seed, "train": train, "dev": dev,
                "projections": projections, "operator": operator, "full_operator": full_operator,
                "initial_states": {name: model.state_dict() for name, model in pair.items()}}, destination)
    formal.atomic_json(directory / "PREPARED.json", {
        "time": now(), "task": args.task, "seed": args.seed,
        "shared_sha256": formal.sha256_file(destination), "parameters": counts,
        "train_pids": train_pids, "dev_pids": dev_pids,
        "label_profile": formal.TASKS[args.task]["binary_profile"],
        "cache_signature": formal.cache_signature(args.meter_root, formal.TASKS[args.task]["dataset"]),
        "seconds": time.time() - started,
    })
    emit("prepared", task=args.task, seed=args.seed, train=len(train), dev=len(dev),
         parameters=counts, seconds=time.time() - started)


def shared(args):
    path = pair_dir(args) / "shared_train_dev.pt"
    if formal.sha256_file(path) != load_json(pair_dir(args) / "PREPARED.json")["shared_sha256"]:
        raise RuntimeError("Shared Train/Dev artifact changed")
    return formal.load_trusted_checkpoint(path, torch.device("cpu"))


def optimizer_for(model):
    settings = config()
    return torch.optim.AdamW(model.parameters(), lr=settings.learning_rate,
                            weight_decay=settings.weight_decay)


def epoch_call(model, operator, train, optimizer, args, cycle, epoch):
    return formal.train_selector_epoch(
        model, operator, train, optimizer, config(), task_name=args.task,
        variant="no_ensr", method="gflownet", seed=args.seed,
        cycle=cycle, epoch=epoch, permuted_scores={})


def finite_step(optimizer, unused_args, unused_kwargs):
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Nonfinite gradient; stop before optimizer update")


def smoke(args):
    check_contract(args)
    data = shared(args)
    settings = config()
    result = {}
    for arm, original in models(data["train"][0]["events"].shape[1], args.task, args.seed).items():
        fast = copy.deepcopy(original)
        formal.operator_log_rewards = REFERENCE_REWARD
        loss_original = epoch_call(original, data["operator"], data["train"][:3],
                                   optimizer_for(original), args, 0, 1)
        formal.operator_log_rewards = fast_reward
        loss_fast = epoch_call(fast, data["operator"], data["train"][:3],
                               optimizer_for(fast), args, 0, 1)
        np.testing.assert_allclose(loss_original, loss_fast, rtol=1e-7, atol=1e-6)
        max_error = 0.0
        for key, value in original.state_dict().items():
            difference = float((value - fast.state_dict()[key]).abs().max())
            max_error = max(max_error, difference)
            torch.testing.assert_close(value, fast.state_dict()[key], rtol=1e-6, atol=1e-7)
        for cycle in (1, 2):
            masks = formal.augmentation_masks(fast, data["train"][0], settings,
                                             cycle=cycle, seed=args.seed)
            if len(masks) != 5:
                raise RuntimeError("Unexpected alternating augmentation budget")
        evaluation = formal.evaluate(
            fast, data["operator"], data["full_operator"], data["dev"][:3], settings,
            samples=16, seed=9, threshold=0.5, full_threshold=0.5, include_controls=False)
        result[arm] = {"reference_loss": loss_original, "fast_loss": loss_fast,
                       "max_parameter_error": max_error, "dev_smoke_pids": evaluation["pids"]}
    formal.atomic_json(args.output / "SMOKE_PASS.json", {"time": now(), "test_loaded": False,
                                                        "result": result})
    emit("smoke_pass", result=result)


def develop(args):
    check_contract(args)
    if not (args.output / "SMOKE_PASS.json").exists():
        raise RuntimeError("Smoke verification is required")
    data = shared(args)
    output = arm_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DEVELOPMENT_LOCK.json").exists():
        raise RuntimeError("Refusing to overwrite a locked development run")
    model = models(data["train"][0]["events"].shape[1], args.task, args.seed)[args.arm]
    model.load_state_dict(data["initial_states"][args.arm])
    optimizer = optimizer_for(model)
    optimizer.register_step_pre_hook(finite_step)
    formal.operator_log_rewards = fast_reward
    history, best = [], None
    started = time.time()
    for cycle, epochs in enumerate(SCHEDULE):
        tick = time.time()
        operator = data["operator"] if cycle == 0 else formal.fit_operator(
            model, data["train"], config(), task_name=args.task, variant="no_ensr",
            cycle=cycle, seed=args.seed, permuted_scores={})
        operator_path = output / f"operator_cycle_{cycle + 1}.pkl"
        with operator_path.open("wb") as handle:
            pickle.dump(operator, handle, protocol=5)
        emit("operator_fit", task=args.task, seed=args.seed, arm=args.arm,
             cycle=cycle + 1, seconds=time.time() - tick)
        for epoch in range(1, epochs + 1):
            tick = time.time()
            loss = epoch_call(model, operator, data["train"], optimizer, args, cycle, epoch)
            if not math.isfinite(loss) or any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite loss/parameters")
            evaluation = formal.evaluate(
                model, operator, data["full_operator"], data["dev"], config(), samples=16,
                seed=formal.stable_seed(args.task, args.seed, cycle, epoch, "dev"),
                threshold=None, full_threshold=None, include_controls=False)
            key = formal.development_selection_key(evaluation)
            item = {"cycle": cycle + 1, "epoch": epoch, "global_epoch": len(history) + 1,
                    "tb_loss": loss, "dev": evaluation, "seconds": time.time() - tick}
            history.append(item)
            if best is None or key > best["selection_key"]:
                best = {"selection_key": key, "cycle": cycle + 1, "epoch": epoch,
                        "global_epoch": len(history), "dev": evaluation,
                        "state": copy.deepcopy(model.state_dict()),
                        "operator": copy.deepcopy(operator)}
            formal.atomic_json(output / "HISTORY.json", history)
            emit("epoch", task=args.task, seed=args.seed, arm=args.arm,
                 global_epoch=len(history), tb_loss=loss, dev_f1=key[0], seconds=item["seconds"])
    checkpoint = output / "selected.pt"
    torch.save({"schema": SCHEMA, "task": args.task, "seed": args.seed, "arm": args.arm,
                "selected": best}, checkpoint)
    tail = [row["tb_loss"] for row in history if row["cycle"] == 3]
    previous, final = float(np.mean(tail[-12:-6])), float(np.mean(tail[-6:]))
    formal.atomic_json(output / "DEVELOPMENT_LOCK.json", {
        "time": now(), "task": args.task, "seed": args.seed, "arm": args.arm,
        "checkpoint_sha256": formal.sha256_file(checkpoint),
        "shared_sha256": load_json(pair_dir(args) / "PREPARED.json")["shared_sha256"],
        "schedule": SCHEDULE, "selected_cycle": best["cycle"], "selected_epoch": best["epoch"],
        "selected_global_epoch": best["global_epoch"], "dev": best["dev"],
        "seconds": time.time() - started, "test_loaded": False,
        "convergence": {"preceding_six_tb": previous, "final_six_tb": final,
                        "relative_drop": (previous - final) / max(abs(previous), 1e-12),
                        "converged_claim": False},
    })
    emit("development_locked", task=args.task, seed=args.seed, arm=args.arm,
         selected_epoch=best["global_epoch"], seconds=time.time() - started)


def test(args):
    check_contract(args)
    manifest = load_json(args.output / "ALL_DEVELOPMENT_LOCKED.json")
    output = arm_dir(args)
    relative = str((output / "DEVELOPMENT_LOCK.json").relative_to(args.output))
    if formal.sha256_file(output / "DEVELOPMENT_LOCK.json") != manifest["locks"][relative]:
        raise RuntimeError("Development lock changed after global Test authorization")
    lock = load_json(output / "DEVELOPMENT_LOCK.json")
    checkpoint = output / "selected.pt"
    if formal.sha256_file(checkpoint) != lock["checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint changed")
    if (output / "OFFICIAL_TEST.json").exists():
        raise RuntimeError("Refusing to repeat locked Test")
    data = shared(args)
    selected = formal.load_trusted_checkpoint(checkpoint, torch.device("cpu"))["selected"]
    model = models(data["train"][0]["events"].shape[1], args.task, args.seed)[args.arm]
    model.load_state_dict(selected["state"])
    test_opened = now()
    raw = formal.load_records(args.meter_root, args.contract_dir, args.task,
                              allowed_splits=("test",), split_policy="official")
    formal.attach_temporal(raw, args.meter_root, formal.TASKS[args.task]["dataset"])
    records = formal.prepare_records(raw["test"], data["projections"], torch.device("cpu"))
    pids = {r["pid"] for r in records}
    if pids & {r["pid"] for r in data["train"] + data["dev"]}:
        raise RuntimeError("Test overlap with Train/Dev")
    expected = 47 if args.task == "daic_depression" else 56
    if len(records) != expected or len(pids) != expected:
        raise RuntimeError("Unexpected official Test participants")
    result = formal.evaluate(
        model, selected["operator"], data["full_operator"], records, config(), samples=48,
        seed=formal.stable_seed(args.task, args.seed, "test"),
        threshold=selected["dev"]["threshold"], full_threshold=selected["dev"]["full_threshold"],
        include_controls=False)
    formal.atomic_json(output / "OFFICIAL_TEST.json", {
        "time": now(), "test_opened": test_opened, "global_lock_time": manifest["time"],
        "task": args.task, "seed": args.seed, "arm": args.arm,
        "checkpoint_sha256": lock["checkpoint_sha256"], "result": result,
    })
    emit("test_complete", task=args.task, seed=args.seed, arm=args.arm,
         f1=result["posterior"]["weighted_f1"])


def worker_batch(args, phase, jobs, cores):
    running = []
    queue = list(jobs)
    launched = []
    try:
        while queue or running:
            while queue and len(running) < 30:
                task, seed, arm = queue.pop(0)
                name = f"{phase}_{task}_{seed}_{arm}"
                log = args.output / "logs" / f"{name}.jsonl"
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", phase,
                           "--output", str(args.output), "--meter-root", str(args.meter_root),
                           "--contract-dir", str(args.contract_dir), "--task", task,
                           "--seed", str(seed), "--arm", arm, "--cpu", str(cores[len(launched) % len(cores)])]
                environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                   OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
                with log.open("w", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                               stdout=stream, stderr=subprocess.STDOUT)
                entry = {"process": process, "name": name, "start": time.time()}
                running.append(entry)
                launched.append({"pid": process.pid, "name": name, "log": str(log)})
                formal.atomic_json(args.output / f"{phase.upper()}_WORKERS.json", launched)
            for entry in list(running):
                code = entry["process"].poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError(f"Worker failed: {entry['name']}, exit {code}")
                    running.remove(entry)
                elif time.time() - entry["start"] > 2700:
                    raise TimeoutError(entry["name"])
            if running:
                time.sleep(2)
    except BaseException:
        # Only terminate children launched by this orchestrator.
        for entry in running:
            if entry["process"].poll() is None:
                entry["process"].terminate()
        for entry in running:
            try:
                entry["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                entry["process"].kill()
                entry["process"].wait()
        raise


def aggregate(args):
    from scipy.stats import t
    import csv

    report = {"schema": SCHEMA, "time": now(), "schedule": SCHEDULE, "tasks": {}}
    rows = []
    for task in TASKS:
        results = {arm: [] for arm in ARMS}
        for seed in SEEDS:
            for arm in ARMS:
                path = args.output / task / f"seed_{seed}" / arm
                test_record = load_json(path / "OFFICIAL_TEST.json")
                lock = load_json(path / "DEVELOPMENT_LOCK.json")
                if test_record["checkpoint_sha256"] != lock["checkpoint_sha256"]:
                    raise RuntimeError("Test checkpoint differs from lock")
                if test_record["test_opened"] <= test_record["global_lock_time"]:
                    raise RuntimeError("Test opened before global lock")
                result = test_record["result"]
                result_row = {"task": task, "seed": seed, "arm": arm,
                              **result["posterior"],
                              "severity_ccc": result["severity"]["posterior"].get("ccc"),
                              "severity_mae": result["severity"]["posterior"].get("mae"),
                              "selected_epoch": lock["selected_global_epoch"],
                              "final_six_tb": lock["convergence"]["final_six_tb"],
                              "tail_loss_relative_drop": lock["convergence"]["relative_drop"],
                              "test_n": len(result["pids"])}
                rows.append(result_row)
                results[arm].append(result_row)
        stats = {}
        for arm in ARMS:
            stats[arm] = {}
            for metric in ("weighted_f1", "macro_f1", "auroc", "severity_ccc", "severity_mae"):
                values = np.asarray([r[metric] for r in results[arm]], dtype=float)
                stats[arm][metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
        difference = np.asarray([b["weighted_f1"] - a["weighted_f1"]
                                 for a, b in zip(results[ARMS[0]], results[ARMS[1]])])
        effect = float(difference.mean())
        half = float(t.ppf(0.975, 4) * difference.std(ddof=1) / np.sqrt(5))
        null = [abs(float((difference * signs).mean())) for signs in itertools.product((-1, 1), repeat=5)]
        stats["paired_hierarchical_minus_flat"] = {
            "mean": effect, "seed_t_ci95": [effect - half, effect + half],
            "positive_seeds": int((difference > 0).sum()), "deltas": difference.tolist(),
            "exact_sign_flip_p": float(np.mean(np.asarray(null) >= abs(effect) - 1e-12)),
        }
        report["tasks"][task] = stats
    report["rows"] = rows
    formal.atomic_json(args.output / "FINAL_METRICS.json", report)
    with (args.output / "ALL_SEEDS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    emit("formal_complete", tasks=report["tasks"])


def orchestrate(args):
    if (args.output / "RUN_CONTRACT.json").exists():
        raise RuntimeError("Output already initialized; no automatic outcome-based rerun")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir()
    available = sorted(os.sched_getaffinity(0))
    # Reserve a bounded affinity pool; BLAS and Torch each use one thread.
    cores = available[:30]
    if len(cores) < 30:
        raise RuntimeError("This fixed concurrency contract requires 30 permitted CPUs")
    os.sched_setaffinity(0, set(cores + available[30:32]))
    formal.atomic_json(args.output / "RUN_CONTRACT.json", {
        "schema": SCHEMA, "time": now(), "tasks": TASKS, "seeds": SEEDS, "arms": ARMS,
        "schedule": SCHEDULE, "config": vars(config()), "code_sha256": code_hashes(),
        "max_workers": 30, "threads_per_worker": 1, "affinity_pool": sorted(os.sched_getaffinity(0)),
        "scope": "joint routing/factorization ablation with matched alternating PIM procedure",
        "identical_post_refit_numeric_reward": False, "online_e2e_experiment": False,
        "prior_test_access_disclosed": True,
    })
    pairs = [(task, seed, ARMS[0]) for task in TASKS for seed in SEEDS]
    jobs = [(task, seed, arm) for task in TASKS for seed in SEEDS for arm in ARMS]
    try:
        worker_batch(args, "prepare", pairs, cores)
        worker_batch(args, "smoke", [(TASKS[0], SEEDS[0], ARMS[0])], cores)
        worker_batch(args, "develop", jobs, cores)
        locks = {}
        for task, seed, arm in jobs:
            path = args.output / task / f"seed_{seed}" / arm / "DEVELOPMENT_LOCK.json"
            lock = load_json(path)
            if list(lock["schedule"]) != list(SCHEDULE):
                raise RuntimeError("Training budget changed")
            if formal.sha256_file(path.parent / "selected.pt") != lock["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint changed before global lock")
            locks[str(path.relative_to(args.output))] = formal.sha256_file(path)
        check_contract(args)
        formal.atomic_json(args.output / "ALL_DEVELOPMENT_LOCKED.json", {"time": now(), "locks": locks})
        emit("all_development_locked", count=len(locks))
        worker_batch(args, "test", jobs, cores)
        aggregate(args)
    except BaseException as error:
        formal.atomic_json(args.output / "FAILED.json", {"time": now(), "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("orchestrate", "prepare", "smoke", "develop", "test", "aggregate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meter-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default=TASKS[0])
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--arm", choices=ARMS, default=ARMS[0])
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    formal.configure_runtime(1)
    formal.validate_contract(args.contract_dir)
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
