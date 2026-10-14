#!/usr/bin/env python3
"""Paired fixed-PIM structural ablation; Test is inaccessible until Dev lock."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import validate_alternating_psychometric_gfn_cached as formal
from scripts.alternating_psychometric_gfn import evidence_features, operator_log_rewards
from scripts.meter_gfn import HierarchicalClinicalGFlowNet, trajectory_balance_loss
from scripts.psypath_flat_gfn_control import FlatConditionalEvidenceGFN

SCHEMA = "psypath_same_reward_fixed_pim_control_v1"
ARMS = ("flat_gfn", "hierarchical_gfn")


def settings():
    return SimpleNamespace(
        hidden_dim=64, phenotype_count=4, minimum_events=2, maximum_events=16,
        maximum_fraction=0.30, operator_trajectories=4, selector_trajectories=4,
        exploration_epsilon=0.0, measurement_components=48,
        pim_reward_weight=0.30, compactness_weight=0.05, reward_temperature=0.50,
        learning_rate=3e-4, weight_decay=1e-4, gradient_accumulation=4,
        max_grad_norm=1.0, relative_tb_weight=0.0,
    )


def reward_values(operator, record, masks, config):
    features = np.stack([evidence_features(record["projected_np"], mask) for mask in masks])
    predictions = operator.predict(features)
    p = predictions["probability"]
    y = record["label"]
    cost = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if record["score"] is not None:
        cost += config.pim_reward_weight * np.abs(predictions["severity"] - record["score"]) / operator.score_max
    cost += config.compactness_weight * np.asarray([m.any(1).mean() for m in masks])
    return -cost / config.reward_temperature


def validate_fast_reward(operator, record, trajectories, config):
    reference = operator_log_rewards(
        operator, record["projected_np"], trajectories,
        label=record["label"], score=record["score"], random_seeds=list(range(len(trajectories))),
        pim_weight=0.30, ensr_weight=0.0, sufficiency_weight=1.0,
        compactness_weight=0.05, necessity_margin=0.05,
        specificity_margin=0.02, temperature=0.50,
    )
    fast = reward_values(operator, record, [t.mask.cpu().numpy() for t in trajectories], config)
    np.testing.assert_allclose(fast, [r["log_reward"] for r in reference], atol=1e-10, rtol=1e-10)
    return float(np.max(np.abs(fast - [r["log_reward"] for r in reference])))


def make_models(input_dim, seed, config):
    formal.seed_all(formal.stable_seed(SCHEMA, seed, "init"))
    hierarchical = HierarchicalClinicalGFlowNet(input_dim, hidden_dim=64, phenotype_count=4)
    flat = FlatConditionalEvidenceGFN(input_dim, hidden_dim=64)
    for name in ("event_encoder", "context_encoder", "log_z"):
        getattr(flat, name).load_state_dict(getattr(hierarchical, name).state_dict())
    return {"flat_gfn": flat, "hierarchical_gfn": hierarchical}


def epoch_train(model, operator, records, optimizer, config, seed, epoch):
    model.train()
    order = np.random.default_rng(formal.stable_seed(SCHEMA, seed, epoch, "order")).permutation(len(records))
    loss_values, clinical = [], []
    optimizer.zero_grad(set_to_none=True)
    for step, index in enumerate(order):
        record = records[int(index)]
        encoding = model.encode(record["events"])
        trajectories = [formal.sample_trajectory(
            model, record, config,
            formal.stable_seed(SCHEMA, seed, epoch, record["pid"], sample),
            gradient=True, encoding=encoding,
        ) for sample in range(config.selector_trajectories)]
        logs = reward_values(operator, record, [t.mask.cpu().numpy() for t in trajectories], config)
        rewards = [SimpleNamespace(log_reward=t.log_forward.new_tensor(float(log)))
                   for t, log in zip(trajectories, logs)]
        loss = trajectory_balance_loss(
            model, record["events"], trajectories, rewards,
            log_z=model.partition_from_context(encoding[1]),
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite TB loss")
        # Normalize the final partial accumulation group by its actual size.
        group_start = (step // config.gradient_accumulation) * config.gradient_accumulation
        group_size = min(config.gradient_accumulation, len(order) - group_start)
        (loss / group_size).backward()
        loss_values.append(float(loss.detach()))
        clinical.append(float(np.mean(logs)))
        if (step + 1) % config.gradient_accumulation == 0 or step + 1 == len(order):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return {"tb_loss": float(np.mean(loss_values)), "mean_log_reward": float(np.mean(clinical))}


@torch.no_grad()
def evaluate(model, operator, records, config, seed, samples, threshold=None):
    model.eval()
    predictions, severities, budgets, cells, rows = [], [], [], [], []
    for record in records:
        encoding = model.encode(record["events"])
        trajectories = [formal.sample_trajectory(
            model, record, config,
            formal.stable_seed(SCHEMA, seed, "evaluation", record["pid"], sample),
            gradient=False, encoding=encoding,
        ) for sample in range(samples)]
        masks = [t.mask.cpu().numpy() for t in trajectories]
        p = operator.predict(np.stack([evidence_features(record["projected_np"], m) for m in masks]))
        probability, severity = float(p["probability"].mean()), float(p["severity"].mean())
        predictions.append(probability)
        severities.append(severity)
        budgets.extend([float(m.any(1).sum()) for m in masks])
        cells.extend([float(m.sum()) for m in masks])
        rows.append({"pid": record["pid"], "label": record["label"], "score": record["score"],
                     "probability": probability, "severity": severity})
    labels = [r["label"] for r in records]
    if threshold is None:
        threshold = formal.choose_threshold(labels, predictions)
    return {
        "classification": formal.metric_block(labels, predictions, threshold),
        "severity": formal.severity_block(records, severities),
        "threshold": float(threshold), "participants": len(records),
        "mean_events": float(np.mean(budgets)), "mean_cells": float(np.mean(cells)),
        "path_samples": samples, "predictions": rows,
    }


def emit(event, **kwargs):
    print(json.dumps({"event": event, **kwargs}, default=formal.json_default), flush=True)


def run(args):
    formal.configure_runtime(args.threads)
    formal.validate_contract(args.contract_dir)
    config = settings()
    output = args.output / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DEVELOPMENT_LOCK.json").exists():
        raise RuntimeError("Refusing to overwrite a completed control")
    started = time.time()
    contract = {
        "schema": SCHEMA, "task": args.task, "seed": args.seed, "arms": ARMS,
        "protocol": "fixed_train_only_PIM_structural_ablation",
        "main_pipeline_reproduction": False, "settings": vars(config),
        "epochs": args.epochs, "dev_samples": args.dev_samples, "test_samples": args.test_samples,
        "label_profile": formal.TASKS[args.task]["binary_profile"],
        "test_read": False, "allow_test": args.evaluate_test,
        "source_projection_checkpoint": str(args.projection_checkpoint),
        "source_sha256": formal.sha256_file(args.projection_checkpoint),
        "selection": "Dev weighted-F1; tie AUROC then severity CCC; no outcome-based exclusions",
        "stop_rule": "Stop on invalid data/non-finite loss. No stopping because the control wins.",
        "scope": "Joint removal of latent routing and event/bundle factorization; not separate effects",
        "compute": {"device": "cpu", "threads": args.threads},
    }
    formal.atomic_json(output / "RUN_CONTRACT.json", contract)
    checkpoint = formal.load_trusted_checkpoint(args.projection_checkpoint, torch.device("cpu"))
    if checkpoint["task"] != args.task or checkpoint["variant"] != "no_ensr":
        raise RuntimeError("Projection source does not match the locked task/variant")
    projections = checkpoint["event_projections"]
    del checkpoint
    raw = formal.load_records(args.meter_root, args.contract_dir, args.task,
                              allowed_splits=("train", "dev"), split_policy="official")
    formal.attach_temporal(raw, args.meter_root, formal.TASKS[args.task]["dataset"])
    train = formal.prepare_records(raw["train"], projections, torch.device("cpu"))
    dev = formal.prepare_records(raw["dev"], projections, torch.device("cpu"))
    if set(r["pid"] for r in train) & set(r["pid"] for r in dev):
        raise RuntimeError("Train/Dev PID overlap")
    del raw
    emit("data_ready", train=len(train), dev=len(dev), seconds=time.time() - started)
    # Cycle zero uses only policy-independent random masks and full evidence.
    operator = formal.fit_operator(None, train, config, task_name=args.task, variant="no_ensr",
                                   cycle=0, seed=args.seed, permuted_scores={})
    measurement_path = output / "shared_measurement.pkl"
    with measurement_path.open("wb") as handle:
        pickle.dump({"operator": operator, "projections": projections}, handle, protocol=5)
    operator_hash = formal.sha256_file(measurement_path)
    models = make_models(train[0]["events"].shape[1], args.seed, config)
    parameters = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    if max(parameters.values()) / min(parameters.values()) > 1.10:
        raise RuntimeError(f"Parameter mismatch above 10%: {parameters}")
    reference_trajectories = [formal.sample_trajectory(
        model, train[0], config, args.seed, gradient=False,
    ) for model in models.values()]
    error = validate_fast_reward(operator, train[0], reference_trajectories, config)
    emit("preflight_pass", parameters=parameters, reward_error=error,
         shared_measurement_sha256=operator_hash, seconds=time.time() - started)
    histories, best, optimizers = {}, {}, {}
    for arm, model in models.items():
        histories[arm] = []
        optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay)
    for epoch in range(1, args.epochs + 1):
        for arm, model in models.items():
            tick = time.time()
            stats = epoch_train(model, operator, train, optimizers[arm], config, args.seed, epoch)
            result = evaluate(model, operator, dev, config, args.seed, args.dev_samples)
            key = (result["classification"]["weighted_f1"], result["classification"].get("auroc") or 0.0,
                   result["severity"].get("ccc") or 0.0)
            histories[arm].append({"epoch": epoch, **stats, "dev": result, "seconds": time.time()-tick})
            if arm not in best or key > best[arm]["key"]:
                best[arm] = {"key": key, "epoch": epoch, "dev": result,
                             "state": copy.deepcopy(model.state_dict())}
            formal.atomic_json(output / "HISTORY.json", histories)
            emit("epoch_complete", seed=args.seed, arm=arm, epoch=epoch,
                 dev_f1=key[0], tb_loss=stats["tb_loss"], seconds=time.time()-tick)
    locked = {**contract, "status": "development_locked", "measurement_sha256": operator_hash,
              "reward_equivalence_max_error": error, "parameter_counts": parameters,
              "elapsed_s": time.time()-started, "results": {}}
    for arm, chosen in best.items():
        path = output / f"{arm}.pt"
        torch.save({"schema": SCHEMA, "arm": arm, "seed": args.seed, "task": args.task,
                    "state": chosen["state"], "measurement_sha256": operator_hash,
                    "selected_epoch": chosen["epoch"], "threshold": chosen["dev"]["threshold"]}, path)
        locked["results"][arm] = {"selected_epoch": chosen["epoch"], "development": chosen["dev"],
                                   "checkpoint": str(path), "checkpoint_sha256": formal.sha256_file(path)}
    locked["hierarchical_minus_flat"] = (
        best["hierarchical_gfn"]["dev"]["classification"]["weighted_f1"]
        - best["flat_gfn"]["dev"]["classification"]["weighted_f1"])
    formal.atomic_json(output / "DEVELOPMENT_LOCK.json", locked)
    if args.evaluate_test:
        # Test is read only after all local choices have been serialized and hashed.
        locked_hash = formal.sha256_file(output / "DEVELOPMENT_LOCK.json")
        test_raw = formal.load_records(args.meter_root, args.contract_dir, args.task,
                                       allowed_splits=("test",), split_policy="official")
        formal.attach_temporal(test_raw, args.meter_root, formal.TASKS[args.task]["dataset"])
        test = formal.prepare_records(test_raw["test"], projections, torch.device("cpu"))
        results = {}
        for arm, model in models.items():
            model.load_state_dict(best[arm]["state"])
            results[arm] = evaluate(model, operator, test, config, args.seed,
                                    args.test_samples, best[arm]["dev"]["threshold"])
        final = {"schema": SCHEMA, "task": args.task, "seed": args.seed,
                 "protocol": contract["protocol"], "split": "official_test",
                 "development_lock_sha256": locked_hash, "results": results,
                 "hierarchical_minus_flat": results["hierarchical_gfn"]["classification"]["weighted_f1"]
                 - results["flat_gfn"]["classification"]["weighted_f1"],
                 "elapsed_s": time.time()-started}
        formal.atomic_json(output / "OFFICIAL_TEST.json", final)
        emit("test_complete", delta=final["hierarchical_minus_flat"], seconds=final["elapsed_s"])
    emit("complete", seed=args.seed, dev_delta=locked["hierarchical_minus_flat"], seconds=time.time()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meter-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--projection-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(formal.TASKS), default="daic_depression")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--dev-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=48)
    parser.add_argument("--evaluate-test", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
