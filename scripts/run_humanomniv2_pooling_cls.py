"""Classification-only pooled readout; all models locked before Test scoring."""
from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from run_humanomniv2_pooling_mtl import (
    ROOT, digest, formal, load_label_profile, metadata, pooled, save, threshold,
    validate_contract,
)


class PoolClassifier(torch.nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(344, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU())
        self.risk = torch.nn.Linear(hidden // 2, 1)

    def forward(self, x):
        return self.risk(self.encoder(x)).squeeze(-1)


def labels(args, task, split):
    if split == "test" and not (args.output / "PREDICTIONS_COMMITTED.json").exists():
        raise RuntimeError("Test targets locked until all predictions committed")
    spec = formal.TASKS[task]
    assert spec["binary_profile"] == args.protocol["label_profiles"][task]
    frame, profile = load_label_profile(
        args.contract, spec["dataset"], spec["binary_profile"], split=split)
    lookup = {str(int(r.pid)): int(getattr(r, profile.target_column)) for r in frame.itertuples()}
    ids = metadata(args, task, split)
    assert set(ids) == set(lookup)
    y = np.array([lookup[k] for k in ids], dtype=np.float32)
    assert np.isin(y, [0, 1]).all()
    return y


def predict(model, x):
    model.eval()
    with torch.no_grad():
        p = model(x).sigmoid().numpy().astype(float)
    assert np.isfinite(p).all()
    return p


def loss(model, x, y, pos):
    return torch.nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos)


def smoke(args):
    torch.manual_seed(19)
    model = PoolClassifier(64, 0)
    assert all(k.startswith(("encoder.", "risk.")) for k in model.state_dict())
    x = torch.randn(16, 344)
    y = torch.arange(16).remainder(2).float()
    opt = torch.optim.AdamW(model.parameters(), lr=.005)
    before = float(loss(model, x, y, torch.tensor(1.)).detach())
    for _ in range(40):
        opt.zero_grad()
        objective = loss(model, x, y, torch.tensor(1.))
        objective.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    after = float(loss(model, x, y, torch.tensor(1.)).detach())
    assert after < before * .5
    restored = PoolClassifier(64, 0)
    restored.load_state_dict(model.state_dict())
    assert np.array_equal(predict(model, x), predict(restored, x))
    for task in args.protocol["tasks"]:
        sets = [set(metadata(args, task, s)) for s in ("train", "dev", "test")]
        assert not any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    try:
        labels(args, args.protocol["tasks"][0], "test")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Test gate failed")
    save(args.output / "SMOKE.json", {"passed": True, "before": before, "after": after,
        "classification_only": True, "test_gate": True, "disjoint_within_task": True})


def train(args):
    conf = args.protocol
    locks = []
    for task in conf["tasks"]:
        y, yd = labels(args, task, "train"), labels(args, task, "dev")
        for seed in conf["seeds"]:
            start = time.monotonic()
            directory = args.output / task / f"seed_{seed}"
            directory.mkdir(parents=True)
            source = args.formal_root / task / "development_checkpoints" / task / "gflownet" / f"seed_{seed}.pt"
            projections = formal.load_trusted_checkpoint(source, torch.device("cpu"))["event_projections"]
            xa, hashes = pooled(args, task, "train", projections)
            xb, devhash = pooled(args, task, "dev", projections)
            scaler = StandardScaler().fit(xa)
            x = torch.tensor(scaler.transform(xa), dtype=torch.float32)
            xd = torch.tensor(scaler.transform(xb), dtype=torch.float32)
            yt, ydt = torch.tensor(y), torch.tensor(yd)
            pos = torch.tensor(float((1-y).sum() / max(y.sum(), 1)))
            best_key, best, traces = None, None, []
            for ci, cfg in enumerate(conf["grid"]):
                # Same shared-layer initialization and minibatch order as the MTL run.
                torch.manual_seed(formal.stable_seed(task, seed, ci, "pool_mtl"))
                model = PoolClassifier(cfg["hidden"], conf["dropout"])
                optimizer = torch.optim.AdamW(model.parameters(), lr=conf["learning_rate"], weight_decay=cfg["weight_decay"])
                rng = np.random.default_rng(formal.stable_seed(task, seed, ci, "order"))
                candidate_key, stale = None, 0
                for epoch in range(1, conf["max_epochs"] + 1):
                    model.train()
                    order, losses = rng.permutation(len(x)), []
                    for begin in range(0, len(order), conf["batch_size"]):
                        ix = order[begin:begin + conf["batch_size"]]
                        optimizer.zero_grad()
                        objective = loss(model, x[ix], yt[ix], pos)
                        assert torch.isfinite(objective)
                        objective.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        assert float(norm) > 0
                        optimizer.step()
                        losses.append(float(objective.detach()))
                    if epoch % conf["evaluate_every"]:
                        continue
                    p = predict(model, xd)
                    th = threshold(yd, p)
                    metrics = formal.metric_block(yd.tolist(), p.tolist(), th)
                    with torch.no_grad():
                        dev_loss = float(loss(model, xd, ydt, pos))
                    key = (metrics["weighted_f1"], metrics["auroc"], -dev_loss)
                    traces.append({"candidate": ci, "epoch": epoch, "train_loss": float(np.mean(losses)),
                                   "dev_loss": dev_loss, "dev": metrics})
                    if candidate_key is None or key > candidate_key:
                        candidate_key, stale = key, 0
                    else:
                        stale += 1
                    if best_key is None or key > best_key:
                        best_key = key
                        best = {"state": copy.deepcopy(model.state_dict()), "candidate": ci,
                                "epoch": epoch, "threshold": th, "dev": metrics, "config": cfg}
                    if epoch >= conf["minimum_epochs"] and stale >= conf["patience_evaluations"]:
                        break
                save(directory / "TRACE.json", traces)
            assert best is not None
            ck = directory / "selected.pt"
            torch.save({**best, "scaler": scaler, "projections": projections, "task": task, "seed": seed,
                        "protocol_sha256": digest(args.protocol_path), "projection_source_sha256": digest(source)}, ck)
            restored = PoolClassifier(best["config"]["hidden"], conf["dropout"])
            restored.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state"])
            actual = formal.metric_block(yd.tolist(), predict(restored, xd).tolist(), best["threshold"])
            assert abs(actual["weighted_f1"] - best["dev"]["weighted_f1"]) < 1e-12
            entry = {"task": task, "seed": seed, "path": str(ck), "sha256": digest(ck),
                     "threshold": best["threshold"], "candidate": best["candidate"], "epoch": best["epoch"],
                     "dev": best["dev"], "train_cache_sha256": hashes, "dev_cache_sha256": devhash,
                     "seconds": time.monotonic()-start}
            save(directory / "LOCK.json", entry)
            locks.append(entry)
            save(args.output / "STATUS.json", {"stage": "training", "completed": len(locks), "total": 15})
            print(json.dumps({"task": task, "seed": seed, "completed": len(locks)}), flush=True)
    save(args.output / "ALL_MODELS_LOCKED.json", {"protocol_sha256": digest(args.protocol_path), "models": locks})


def test(args):
    conf = args.protocol
    lock = json.loads((args.output / "ALL_MODELS_LOCKED.json").read_text())
    assert len(lock["models"]) == 15 and lock["protocol_sha256"] == digest(args.protocol_path)
    predictions = []
    for entry in lock["models"]:
        task, seed, path = entry["task"], entry["seed"], Path(entry["path"])
        assert digest(path) == entry["sha256"]
        ck = torch.load(path, map_location="cpu", weights_only=False)
        x, hashes = pooled(args, task, "test", ck["projections"])
        x = torch.tensor(ck["scaler"].transform(x), dtype=torch.float32)
        model = PoolClassifier(ck["config"]["hidden"], conf["dropout"])
        model.load_state_dict(ck["state"])
        predictions.append({"task": task, "seed": seed, "threshold": entry["threshold"],
            "ids": metadata(args, task, "test"), "probability": predict(model, x).tolist(), "cache_sha256": hashes})
    save(args.output / "TEST_PREDICTIONS.json", predictions)
    save(args.output / "PREDICTIONS_COMMITTED.json", {"sha256": digest(args.output / "TEST_PREDICTIONS.json")})
    metrics = [{"task": r["task"], "seed": r["seed"], "classification": formal.metric_block(
        labels(args, r["task"], "test").tolist(), r["probability"], r["threshold"])} for r in predictions]
    summary = {}
    for task in conf["tasks"]:
        summary[task] = {}
        for key in ("weighted_f1", "auroc", "brier"):
            values = np.array([r["classification"][key] for r in metrics if r["task"] == task])
            summary[task][key] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
    save(args.output / "RESULTS.json", {"runs": metrics, "summary": summary})
    save(args.output / "STATUS.json", {"stage": "complete", "completed": 15, "summary": summary})
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", dest="protocol_path", type=Path, required=True)
    p.add_argument("--stage", choices=["smoke", "all"], default="all")
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/meter_mh_official_e2e_pipeline_20260813/official_cache")
    p.add_argument("--formal-root", type=Path, default=ROOT / "outputs/psychometric_gfn_formal_suite_v1_20260818/pim_gflownet")
    p.add_argument("--contract", type=Path, default=ROOT / "data/clinical_label_contract_v1")
    p.add_argument("--tasks", nargs="+", choices=tuple(formal.TASKS))
    args = p.parse_args()
    args.protocol = json.loads(args.protocol_path.read_text())
    if args.tasks:
        if not set(args.tasks).issubset(args.protocol["tasks"]):
            p.error("Requested tasks are outside the selected protocol")
        args.protocol["tasks"] = args.tasks
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "RUN_CONTRACT.json").exists():
        raise FileExistsError("Fresh output directory required; no retuning after Test")
    formal.configure_runtime(args.protocol["threads"])
    validate_contract(args.contract)
    save(args.output / "RUN_CONTRACT.json", {"protocol": args.protocol, "created": time.time(),
        "script_sha256": digest(__file__), "label_contract_sha256": digest(args.contract / "manifest.json"),
        "pooling_helper_sha256": digest(ROOT / "scripts/run_humanomniv2_pooling_mtl.py")})
    try:
        smoke(args)
        if args.stage == "all":
            train(args)
            test(args)
    except Exception:
        save(args.output / "STATUS.json", {"stage": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
