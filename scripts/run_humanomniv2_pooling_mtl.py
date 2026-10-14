"""Independent pooled multitask readouts with a global locked-Test gate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_alternating_psychometric_gfn_cached as formal
from scripts.clinical_label_contract import load_label_profile, validate_contract

VIEWS = ("clinical", "answer", "context", "visual")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    formal.atomic_json(Path(path), obj)


def metadata(args, task, split):
    ds = formal.TASKS[task]["dataset"]
    frame = pd.read_csv(args.contract / f"{ds}_labels.csv", usecols=["pid", "split"])
    ids = [str(int(x)) for x in frame.loc[frame.split == split, "pid"]]
    assert len(ids) == len(set(ids)) == formal.EXPECTED_SPLITS[ds][split]
    return ids


def labels(args, task, split):
    if split == "test" and not (args.output / "PREDICTIONS_COMMITTED.json").exists():
        raise RuntimeError("Test targets are locked until all predictions are committed")
    spec = formal.TASKS[task]
    b, bp = load_label_profile(args.contract, spec["dataset"], spec["binary_profile"], split=split)
    q, qp = load_label_profile(args.contract, spec["dataset"], spec["score_profile"], split=split, include_unlabeled=True)
    bm = {str(int(r.pid)): int(getattr(r, bp.target_column)) for r in b.itertuples()}
    qm = {str(int(r.pid)): float(getattr(r, qp.target_column))
          for r in q.itertuples() if int(getattr(r, qp.mask_column)) == 1}
    ids = metadata(args, task, split)
    assert set(ids) == set(bm)
    y = np.array([bm[k] for k in ids], dtype=np.float32)
    s = np.array([qm.get(k, np.nan) for k in ids], dtype=np.float32)
    lo, hi = args.protocol["score_bounds"][task]
    assert np.isin(y, [0, 1]).all()
    assert ((s[np.isfinite(s)] >= lo) & (s[np.isfinite(s)] <= hi)).all()
    return y, s


def pooled(args, task, split, projections):
    ds = formal.TASKS[task]["dataset"]
    root = args.cache / ds
    signature = json.loads((root / "RUN_CONTRACT.json").read_text())["run_signature"]
    rows, hashes = [], {}
    for pid in metadata(args, task, split):
        path = root / "cache" / f"{pid}.npz"
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"].item()))
            assert meta["run_signature"] == signature
            raw = {k: np.array(z[k], dtype=np.float32) for k in VIEWS}
        n = len(raw["answer"])
        assert n > 0 and all(len(v) == n and np.isfinite(v).all() for v in raw.values())
        assert len(meta["turns"]) == n
        record = {"pid": pid, "label": 0, "score": None, "raw": raw,
                  "temporal": formal.temporal_features(meta["turns"])}
        event = formal.prepare_records([record], projections, torch.device("cpu"))[0]["events"].numpy()
        assert event.shape == (n, 172) and np.isfinite(event).all()
        rows.append(np.concatenate([event.mean(0), event.std(0)]))
        hashes[pid] = digest(path)
    return np.stack(rows), hashes


class PoolHead(torch.nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(width, hidden), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU())
        self.risk = torch.nn.Linear(hidden // 2, 1)
        self.score = torch.nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.risk(h).squeeze(-1), self.score(h).squeeze(-1)


def objective(logits, score, y, q, pos_weight, score_weight):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    mask = torch.isfinite(q)
    severity = torch.abs(score[mask] - q[mask]).mean() if mask.any() else score.sum() * 0
    return bce + score_weight * severity


def threshold(y, p):
    grid = np.linspace(.10, .90, 161)
    a = p[:, None] >= grid[None, :]
    positive = y[:, None] == 1
    tp = (a & positive).sum(0); tn = (~a & ~positive).sum(0)
    fp = (a & ~positive).sum(0); fn = (~a & positive).sum(0)
    fpos = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    fneg = 2 * tn / np.maximum(2 * tn + fp + fn, 1)
    w = (fpos * y.sum() + fneg * (len(y) - y.sum())) / len(y)
    idx = max(range(len(grid)), key=lambda i: (w[i], (fpos[i]+fneg[i])/2, -abs(grid[i]-.5)))
    return float(grid[idx])


def predict(model, x, bounds):
    model.eval()
    with torch.no_grad():
        logit, q = model(x)
    p = logit.sigmoid().numpy().astype(float)
    severity = np.clip(q.numpy() * bounds[1], *bounds).astype(float)
    if not np.isfinite(p).all() or not np.isfinite(severity).all():
        raise FloatingPointError("Invalid predictions")
    return p, severity


def smoke(args):
    torch.manual_seed(19)
    x = torch.randn(16, 344); y = torch.arange(16).remainder(2).float()
    q = torch.linspace(0, 1, 16); q[0] = float("nan")
    m = PoolHead(344, 64, 0)
    o = torch.optim.AdamW(m.parameters(), lr=.005)
    before = float(objective(*m(x), y, q, torch.tensor(1.), .3).detach())
    for _ in range(40):
        o.zero_grad(); loss = objective(*m(x), y, q, torch.tensor(1.), .3)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        o.step()
    after = float(objective(*m(x), y, q, torch.tensor(1.), .3).detach())
    assert after < before * .5
    state = copy.deepcopy(m.state_dict()); restored = PoolHead(344, 64, 0)
    restored.load_state_dict(state); m.eval(); restored.eval()
    assert torch.equal(m(x)[0], restored(x)[0])
    for task in args.protocol["tasks"]:
        sets = [set(metadata(args, task, s)) for s in ["train", "dev", "test"]]
        assert not any(sets[a] & sets[b] for a,b in [(0,1),(0,2),(1,2)])
        labels(args, task, "train"); labels(args, task, "dev")
    try:
        labels(args, args.protocol["tasks"][0], "test")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Test gate did not reject target access")
    save(args.output / "SMOKE.json", {"passed": True, "synthetic_loss_before": before,
        "synthetic_loss_after": after, "roundtrip_exact": True, "missing_score_mask": True,
        "test_targets_blocked": True, "split_overlap": False})


def train(args):
    conf = args.protocol
    if (args.output / "ALL_MODELS_LOCKED.json").exists():
        raise RuntimeError("Training forbidden after final lock")
    locks = []
    for task in conf["tasks"]:
        y, q = labels(args, task, "train"); yd, qd = labels(args, task, "dev")
        for seed in conf["seeds"]:
            start = time.monotonic(); directory = args.output / task / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            source = args.formal_root / task / "development_checkpoints" / task / "gflownet" / f"seed_{seed}.pt"
            projections = formal.load_trusted_checkpoint(source, torch.device("cpu"))["event_projections"]
            xa, hashes = pooled(args, task, "train", projections)
            xb, devhash = pooled(args, task, "dev", projections)
            scaler = StandardScaler().fit(xa)
            x = torch.tensor(scaler.transform(xa), dtype=torch.float32)
            xd = torch.tensor(scaler.transform(xb), dtype=torch.float32)
            yt, qt = torch.tensor(y), torch.tensor(q / conf["score_bounds"][task][1])
            ydt, qdt = torch.tensor(yd), torch.tensor(qd / conf["score_bounds"][task][1])
            pos = torch.tensor(float((1-y).sum() / max(y.sum(), 1)))
            best_key = None; best = None; traces = []
            for ci, cfg in enumerate(conf["grid"]):
                torch.manual_seed(formal.stable_seed(task, seed, ci, "pool_mtl"))
                model = PoolHead(x.shape[1], cfg["hidden"], conf["dropout"])
                with torch.no_grad():
                    model.score.bias.fill_(float(np.nanmean(q)/conf["score_bounds"][task][1]))
                optimizer = torch.optim.AdamW(model.parameters(), lr=conf["learning_rate"], weight_decay=cfg["weight_decay"])
                rng = np.random.default_rng(formal.stable_seed(task, seed, ci, "order"))
                candidate_key = None; stale = 0
                for epoch in range(1, conf["max_epochs"]+1):
                    model.train(); order = rng.permutation(len(x)); losses=[]
                    for begin in range(0, len(order), conf["batch_size"]):
                        ix = order[begin:begin+conf["batch_size"]]
                        optimizer.zero_grad()
                        loss = objective(*model(x[ix]), yt[ix], qt[ix], pos, conf["severity_weight"])
                        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        if float(norm) == 0: raise RuntimeError("Zero training gradient")
                        optimizer.step(); losses.append(float(loss.detach()))
                    if epoch % conf["evaluate_every"]: continue
                    p, s = predict(model, xd, conf["score_bounds"][task])
                    th = threshold(yd, p); metrics = formal.metric_block(yd.tolist(), p.tolist(), th)
                    with torch.no_grad(): joint = float(objective(*model(xd), ydt, qdt, pos, conf["severity_weight"]))
                    key = (metrics["weighted_f1"], metrics["auroc"], -joint)
                    traces.append({"candidate": ci, "epoch": epoch, "train_loss": np.mean(losses), "dev_loss": joint, "dev": metrics})
                    if candidate_key is None or key > candidate_key: candidate_key=key; stale=0
                    else: stale+=1
                    if best_key is None or key > best_key:
                        best_key=key
                        best={"state": copy.deepcopy(model.state_dict()), "candidate": ci, "epoch": epoch,
                              "threshold": th, "dev": metrics, "config": cfg}
                    if epoch >= conf["minimum_epochs"] and stale >= conf["patience_evaluations"]: break
                save(directory / "TRACE.json", traces)
                print(json.dumps({"stage":"candidate_complete","task":task,"seed":seed,"candidate":ci,"epoch":epoch}),flush=True)
            assert best is not None
            ck = directory / "selected.pt"
            torch.save({**best, "scaler": scaler, "projections": projections, "task": task, "seed": seed,
                        "protocol_sha256": digest(args.protocol_path), "projection_source_sha256": digest(source)}, ck)
            verify=PoolHead(344,best["config"]["hidden"],conf["dropout"])
            loaded=torch.load(ck,map_location="cpu",weights_only=False);verify.load_state_dict(loaded["state"])
            pp,_=predict(verify,xd,conf["score_bounds"][task])
            assert abs(formal.metric_block(yd.tolist(),pp.tolist(),best["threshold"])["weighted_f1"]-best["dev"]["weighted_f1"])<1e-12
            lock={"task": task,"seed":seed,"path":str(ck),"sha256":digest(ck),"threshold":best["threshold"],
                  "candidate":best["candidate"],"epoch":best["epoch"],"dev":best["dev"],
                  "train_cache_sha256":hashes,"dev_cache_sha256":devhash,"seconds":time.monotonic()-start}
            save(directory / "LOCK.json",lock); locks.append(lock)
            save(args.output / "STATUS.json",{"stage":"training","completed":len(locks),"total":15,"task":task,"seed":seed})
    save(args.output / "ALL_MODELS_LOCKED.json",{"protocol_sha256":digest(args.protocol_path),"models":locks})


def test(args):
    conf=args.protocol; lock=json.loads((args.output/"ALL_MODELS_LOCKED.json").read_text())
    assert lock["protocol_sha256"]==digest(args.protocol_path) and len(lock["models"])==15
    predictions=[]
    for entry in lock["models"]:
        task,seed=entry["task"],entry["seed"];path=Path(entry["path"])
        assert digest(path)==entry["sha256"]
        data=torch.load(path,map_location="cpu",weights_only=False)
        x,hashes=pooled(args,task,"test",data["projections"])
        x=torch.tensor(data["scaler"].transform(x),dtype=torch.float32)
        model=PoolHead(344,data["config"]["hidden"],conf["dropout"]);model.load_state_dict(data["state"])
        p,q=predict(model,x,conf["score_bounds"][task])
        predictions.append({"task":task,"seed":seed,"threshold":entry["threshold"],"ids":metadata(args,task,"test"),
            "probability":p.tolist(),"severity":q.tolist(),"cache_sha256":hashes})
    save(args.output/"TEST_PREDICTIONS.json",predictions)
    save(args.output/"PREDICTIONS_COMMITTED.json",{"sha256":digest(args.output/"TEST_PREDICTIONS.json")})
    metrics=[]
    for row in predictions:
        y,q=labels(args,row["task"],"test")
        m=formal.metric_block(y.tolist(),row["probability"],row["threshold"])
        s=formal.severity_block([{"score":float(v) if np.isfinite(v) else None} for v in q],row["severity"])
        metrics.append({"task":row["task"],"seed":row["seed"],"classification":m,"severity":s})
    summary={}
    for task in conf["tasks"]:
        entries=[r for r in metrics if r["task"]==task]
        summary[task]={}
        for group,names in [("classification",["weighted_f1","auroc","brier"]),("severity",["ccc","mae"])]:
            for key in names:
                values=np.array([r[group][key] for r in entries])
                summary[task][key]={"mean":values.mean(),"sample_sd":values.std(ddof=1)}
    save(args.output/"RESULTS.json",{"runs":metrics,"summary":summary})
    save(args.output/"STATUS.json",{"stage":"complete","completed":15,"summary":summary})
    print(json.dumps(summary,default=float),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--protocol",dest="protocol_path",type=Path,required=True)
    p.add_argument("--stage",choices=["smoke","all"],default="all")
    p.add_argument("--cache",type=Path,default=ROOT/"outputs/meter_mh_official_e2e_pipeline_20260813/official_cache")
    p.add_argument("--formal-root",type=Path,default=ROOT/"outputs/psychometric_gfn_formal_suite_v1_20260818/pim_gflownet")
    p.add_argument("--contract",type=Path,default=ROOT/"data/clinical_label_contract_v1")
    p.add_argument("--tasks",nargs="+",choices=tuple(formal.TASKS))
    args=p.parse_args();args.protocol=json.loads(args.protocol_path.read_text());args.output.mkdir(parents=True,exist_ok=True)
    if args.tasks:
        if not set(args.tasks).issubset(args.protocol["tasks"]):
            p.error("Requested tasks are outside the selected protocol")
        args.protocol["tasks"]=args.tasks
    formal.configure_runtime(args.protocol["threads"]);validate_contract(args.contract)
    if (args.output/"RUN_CONTRACT.json").exists():raise FileExistsError("Use a fresh output directory; no silent rerun")
    save(args.output/"RUN_CONTRACT.json",{"protocol":args.protocol,"script_sha256":digest(__file__),
        "label_contract_sha256":digest(args.contract/"manifest.json"),"created":time.time(),
        "dependencies":{str(Path(m.__file__).name):digest(m.__file__) for m in [formal,formal.gate]},
        "torch":torch.__version__,"numpy":np.__version__})
    try:
        smoke(args)
        if args.stage=="all":train(args);test(args)
    except Exception:
        save(args.output/"STATUS.json",{"stage":"failed","traceback":traceback.format_exc()});raise


if __name__=="__main__":main()
