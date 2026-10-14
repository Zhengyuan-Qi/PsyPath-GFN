#!/usr/bin/env python3
"""Random-set PIM and paired fixed-PIM score-reward controls."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_alternating_psychometric_gfn_cached as b
from scripts import train_meter_mh_sequential_increment as loader
from scripts.clinical_label_contract import load_label_profile, validate_contract

ARMS = ("random_pim", "score_reward", "no_score_reward")


def isolated_score_map(contract_dir, task, allowed_splits=("train", "dev", "test")):
    values = {}
    for split in allowed_splits:
        frame, spec = load_label_profile(
            contract_dir, task["dataset"], task["score_profile"],
            split=split, include_unlabeled=True,
        )
        for row in frame.to_dict("records"):
            value, mask = row.get(spec.target_column), row.get(spec.mask_column)
            score = None
            if mask == 1 and value is not None and np.isfinite(float(value)):
                score = float(value)
            values[(loader.normalize_pid(row["pid"]), str(row["split"]))] = score
    return values


def load_partition(cache, contract, task, splits):
    # The legacy loader calls score_map without split arguments.
    original = loader.score_map
    loader.score_map = lambda root, spec, allowed_splits=None: isolated_score_map(root, spec, splits)
    try:
        result = loader.load_records(cache, contract, task, allowed_splits=splits,
                                     split_policy="official")
    finally:
        loader.score_map = original
    b.attach_temporal(result, cache, b.TASKS[task]["dataset"])
    return result


def random_augmentations(model, record, args, *, cycle, seed):
    del model
    length = len(record["events"])
    minimum, maximum = b.event_limits(length, args)
    return [b.full_mask(length)] + [
        b.random_sparse_mask(length, minimum, maximum,
            b.stable_seed("operator_random", seed, cycle, record["pid"], i))
        for i in range(args.operator_trajectories)
    ]


def fit_random_operator(records, args, task, seed, cycle):
    original = b.augmentation_masks
    b.augmentation_masks = random_augmentations
    try:
        return b.fit_operator(None, records, args, task_name=task, variant="no_ensr",
                              cycle=cycle, seed=seed, permuted_scores={})
    finally:
        b.augmentation_masks = original


def digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


@torch.no_grad()
def predict(records, operator, args, *, seed, samples, threshold=None, model=None):
    if model is not None:
        model.eval()
    probability, severity, counts = [], [], []
    for record in records:
        length = len(record["events"])
        minimum, maximum = b.event_limits(length, args)
        encoding = model.encode(record["events"]) if model is not None else None
        masks = []
        for i in range(samples):
            draw_seed = b.stable_seed(seed, record["pid"], i)
            if model is None:
                mask = b.random_sparse_mask(length, minimum, maximum, draw_seed)
            else:
                mask = b.sample_trajectory(model, record, args, draw_seed,
                    gradient=False, encoding=encoding).mask.cpu().numpy().astype(bool)
            if mask.shape != (length, 4) or not minimum <= mask.any(1).sum() <= maximum:
                raise RuntimeError("Invalid sampled evidence mask")
            masks.append(mask)
        features = np.stack([b.evidence_features(record["projected_np"], m) for m in masks])
        out = operator.predict(features)
        if not all(np.isfinite(out[k]).all() for k in ("probability", "severity")):
            raise RuntimeError("Nonfinite PIM prediction")
        probability.append(float(np.mean(out["probability"])))
        severity.append(float(np.mean(out["severity"])))
        counts.append(float(np.mean([m.any(1).sum() for m in masks])))
    labels = [r["label"] for r in records]
    if threshold is None:
        threshold = b.choose_threshold(labels, probability)
    metrics = b.metric_block(labels, probability, threshold)
    y, pred = np.asarray(labels), np.asarray(probability) >= threshold
    metrics["sensitivity"] = float(pred[y == 1].mean()) if (y == 1).any() else None
    metrics["specificity"] = float((~pred[y == 0]).mean()) if (y == 0).any() else None
    return {
        "threshold": threshold, "posterior": metrics,
        "severity": {"posterior": b.severity_block(records, severity)},
        "pids": [r["pid"] for r in records], "labels": labels,
        "scores": [r.get("score") for r in records],
        "probability": probability, "severity_predictions": severity,
        "mean_selected_events": float(np.mean(counts)),
    }


def parameters(checkpoint, cli):
    config = dict(checkpoint["config"])
    config.update(device="cpu", threads=cli.threads, policy="legacy", path_process=False,
                  path_process_weight=0., path_monotonic_weight=0., path_order_weight=0.,
                  path_order_margin=.02, path_duplicate_weight=0., relative_tb_weight=0.,
                  exploration_epsilon=0., cross_fitted_rewards=False)
    if cli.smoke:
        config.update(cycles=1, epochs_per_cycle=1, selector_trajectories=1,
                      development_samples=2, test_samples=2, measurement_components=8)
    return SimpleNamespace(**config)


def balanced_subset(records, count=12):
    selected = []
    for label in (0, 1):
        selected += [r for r in records if r["label"] == label][:count // 2]
    if len(selected) < count:
        raise ValueError("Smoke set has insufficient class support")
    return selected


def develop(cli):
    out = cli.output / cli.task / f"seed_{cli.seed}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh run directory: {out}")
    out.mkdir(parents=True)
    checkpoint = b.load_trusted_checkpoint(cli.reference, torch.device("cpu"))
    args = parameters(checkpoint, cli)
    raw = load_partition(cli.cache, cli.contract, cli.task, ("train", "dev"))
    if cli.smoke:
        raw = {s: balanced_subset(v) for s, v in raw.items()}
    if set(r["pid"] for r in raw["train"]) & set(r["pid"] for r in raw["dev"]):
        raise RuntimeError("Participant overlap between training and development")
    if checkpoint["task"] != cli.task or int(checkpoint["seed"]) != cli.seed:
        raise RuntimeError("Reference task/seed mismatch")
    if checkpoint.get("stage") != "development_locked":
        raise RuntimeError("Use a development-locked reference checkpoint")
    if cli.smoke:
        projections = b.fit_event_projections(raw["train"], b.stable_seed(cli.task, cli.seed, "event_projection"))
    else:
        projections = checkpoint["event_projections"]
        # Match the training sample count stored by each projection scaler.
        for name in ("shared", "visual"):
            projection = projections[name]
            if projection is None:
                raise RuntimeError("Unexpected projection schema")
            scaler = projection[0]
            n = sum(len(r["raw"]["visual"]) for r in raw["train"])
            expected = n * 3 if name == "shared" else n
            if not np.all(np.asarray(scaler.n_samples_seen_) == expected):
                raise RuntimeError("Train-fitted projection sample count mismatch")
    records = {s: b.prepare_records(v, projections, torch.device("cpu")) for s, v in raw.items()}
    b.atomic_json(out / "PROTOCOL.json", {
        "task": cli.task, "seed": cli.seed, "arms": cli.arms, "smoke": cli.smoke,
        "reference_sha256": b.sha256_file(cli.reference),
        "source_sha256": b.sha256_file(Path(__file__)), "config": vars(args),
        "label_profile": b.TASKS[cli.task], "split_policy": "official",
        "train_pids": [r["pid"] for r in records["train"]],
        "dev_pids": [r["pid"] for r in records["dev"]],
        "test_loaded": False, "selection_rule": "original development_selection_key",
    })
    start = time.monotonic()
    common = fit_random_operator(records["train"], args, cli.task, cli.seed, 0)
    common_digest = digest(common)
    common_path = out / "common_pim.pt"
    torch.save({"operator": common, "event_projections": projections}, common_path)
    common_hash = b.sha256_file(common_path)
    b.atomic_json(out / "COMMON.json", {"sha256": common_hash, "fit_seconds": time.monotonic()-start})

    for arm in cli.arms:
        history, best = [], None
        b.seed_all(b.stable_seed(cli.task, cli.seed, "paired_model"))
        model = None if arm == "random_pim" else b.new_model(records["train"][0], args, torch.device("cpu"), task_name=cli.task)
        initial_hash = digest({k: v.numpy() for k, v in model.state_dict().items()}) if model else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                     weight_decay=args.weight_decay) if model else None
        arm_args = copy.copy(args)
        arm_args.pim_reward_weight = 0.0 if arm == "no_score_reward" else args.pim_reward_weight
        for cycle in range(args.cycles):
            operator = (common if cycle == 0 else fit_random_operator(
                records["train"], args, cli.task, cli.seed, cycle)) if model is None else common
            for epoch in range(1, args.epochs_per_cycle + 1):
                loss = None
                if model is not None:
                    loss = b.train_selector_epoch(model, operator, records["train"], optimizer, arm_args,
                        task_name=cli.task, variant="no_ensr", method="gflownet", seed=cli.seed,
                        cycle=cycle, epoch=epoch, permuted_scores={})
                    if not np.isfinite(loss) or not all(torch.isfinite(p).all() for p in model.parameters()):
                        raise RuntimeError("Nonfinite policy loss or parameters")
                evaluation = predict(records["dev"], operator, args, model=model,
                    seed=b.stable_seed(cli.task, cli.seed, cycle, epoch, "dev"),
                    samples=args.development_samples)
                key = b.development_selection_key(evaluation)
                item = {"cycle": cycle+1, "epoch": epoch, "loss": loss, "development": evaluation}
                history.append(item)
                if best is None or key > best["key"]:
                    best = {"key": key, "cycle": cycle+1, "epoch": epoch,
                            "threshold": evaluation["threshold"], "development": evaluation}
                    torch.save({"operator": operator, "event_projections": projections,
                        "state_dict": copy.deepcopy(model.state_dict()) if model else None,
                        "config": vars(args), "task": cli.task, "seed": cli.seed,
                        "arm": arm, "threshold": evaluation["threshold"]}, out / f"{arm}.pt")
                b.atomic_json(out / f"{arm}_progress.json", {"history": history})
                print(json.dumps({"arm": arm, "task": cli.task, "seed": cli.seed,
                    "cycle": cycle+1, "epoch": epoch, "dev_f1": evaluation["posterior"]["weighted_f1"],
                    "loss": loss, "elapsed_s": time.monotonic()-start}), flush=True)
        if digest(common) != common_digest:
            raise RuntimeError("Common PIM was mutated during policy training")
        b.atomic_json(out / f"{arm}_LOCK.json", {"selection": best,
            "checkpoint_sha256": b.sha256_file(out / f"{arm}.pt"),
            "common_pim_sha256": common_hash, "initial_policy_sha256": initial_hash,
            "explicit_score_reward_weight": arm_args.pim_reward_weight if model else None,
            "development_evaluations": len(history), "test_loaded": False,
            "elapsed_s": time.monotonic()-start})
    if set(("score_reward", "no_score_reward")).issubset(cli.arms):
        locks = [json.loads((out / f"{a}_LOCK.json").read_text()) for a in ARMS[1:]]
        for key in ("initial_policy_sha256", "common_pim_sha256", "development_evaluations"):
            if locks[0][key] != locks[1][key]:
                raise RuntimeError(f"Unmatched paired control: {key}")
    b.atomic_json(out / "DEVELOPMENT_COMPLETE.json", {"arms": cli.arms, "smoke": cli.smoke,
        "test_loaded": False, "elapsed_s": time.monotonic()-start})


def test(cli):
    out = cli.output / cli.task / f"seed_{cli.seed}"
    complete = json.loads((out / "DEVELOPMENT_COMPLETE.json").read_text())
    if complete["smoke"]:
        raise RuntimeError("Smoke runs are never evaluated on the test set")
    if not set(cli.arms).issubset(complete["arms"]):
        raise RuntimeError("Requested arm has not completed development")
    locks = {a: json.loads((out / f"{a}_LOCK.json").read_text()) for a in cli.arms}
    for arm, lock in locks.items():
        if b.sha256_file(out / f"{arm}.pt") != lock["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed after development lock")
        if (out / f"{arm}_TEST.json").exists():
            raise FileExistsError("Refusing to overwrite test predictions")
    raw = load_partition(cli.cache, cli.contract, cli.task, ("test",))["test"]
    for arm in cli.arms:
        saved = b.load_trusted_checkpoint(out / f"{arm}.pt", torch.device("cpu"))
        args = SimpleNamespace(**saved["config"])
        records = b.prepare_records(raw, saved["event_projections"], torch.device("cpu"))
        model = None
        if saved["state_dict"] is not None:
            model = b.new_model(records[0], args, torch.device("cpu"), task_name=cli.task)
            model.load_state_dict(saved["state_dict"], strict=True)
        result = predict(records, saved["operator"], args, model=model,
            seed=b.stable_seed(cli.task, cli.seed, "review_test"),
            samples=args.test_samples, threshold=saved["threshold"])
        b.atomic_json(out / f"{arm}_TEST.json", {"arm": arm, "task": cli.task, "seed": cli.seed,
            "checkpoint_sha256": locks[arm]["checkpoint_sha256"], "test": result})
        print(json.dumps({"arm": arm, "task": cli.task, "seed": cli.seed,
                          "test": result["posterior"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("develop", "test"))
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", choices=("daic_depression", "edaic_depression", "edaic_ptsd"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    cli = p.parse_args()
    b.configure_runtime(cli.threads)
    validate_contract(cli.contract)
    (develop if cli.stage == "develop" else test)(cli)


if __name__ == "__main__":
    main()
