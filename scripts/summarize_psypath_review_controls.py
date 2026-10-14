#!/usr/bin/env python3
"""Recompute matched-control metrics and paired participant intervals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TASKS = ("daic_depression", "edaic_depression", "edaic_ptsd")
SEEDS = (7, 17, 27, 41, 53)
METRICS = ("weighted_f1", "positive_f1", "auroc", "brier", "ccc", "mae", "sensitivity", "specificity")


def normalize(result, draw_rows=None):
    probability = result["probability"]
    if isinstance(probability, dict):
        probability = probability["posterior"]
    severity = result.get("severity_predictions")
    if severity is None:
        if draw_rows is None:
            raise ValueError("Legacy metrics need their locked per-path prediction records")
        lookup = {row["pid"]: row for row in draw_rows}
        severity = [float(np.mean(lookup[pid]["learned_severity"])) for pid in result["pids"]]
        reconstructed = [float(np.mean(lookup[pid]["learned_probability"])) for pid in result["pids"]]
        np.testing.assert_allclose(reconstructed, probability, atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal([lookup[pid]["label"] for pid in result["pids"]], result["labels"])
    elif isinstance(severity, dict):
        severity = severity["posterior"]
    return dict(pids=result["pids"], labels=result["labels"],
                probability=probability, severity=severity, threshold=result["threshold"])


def scores_for_indices(predictions, labels, scores, indices):
    y = labels[indices]
    q = scores[indices]
    all_seeds = []
    for item in predictions:
        p, s = item["probability"][indices], item["severity"][indices]
        yes = p >= item["threshold"]
        pos, neg = y == 1, y == 0
        tp, tn = (yes & pos).sum(1), (~yes & neg).sum(1)
        fp, fn = (yes & neg).sum(1), (~yes & pos).sum(1)
        f1 = 2*tp / np.maximum(2*tp+fp+fn, 1)
        f0 = 2*tn / np.maximum(2*tn+fp+fn, 1)
        # Pairwise ranking handles duplicates and tied scores exactly.
        pair = pos[:, :, None] & neg[:, None, :]
        ranking = (p[:, :, None] > p[:, None, :]) + .5*(p[:, :, None] == p[:, None, :])
        auc = (ranking*pair).sum((1, 2)) / np.maximum(pos.sum(1)*neg.sum(1), 1)
        valid = np.isfinite(q)
        count = np.maximum(valid.sum(1), 1)
        mq = np.where(valid, q, 0).sum(1)/count
        ms = np.where(valid, s, 0).sum(1)/count
        dq, ds = np.where(valid, q-mq[:, None], 0), np.where(valid, s-ms[:, None], 0)
        cov = (dq*ds).sum(1)/count
        denominator = (dq*dq).sum(1)/count + (ds*ds).sum(1)/count + (mq-ms)**2
        ccc = np.divide(2*cov, denominator, out=np.zeros_like(cov), where=denominator>1e-12)
        values = dict(weighted_f1=(f1*pos.sum(1)+f0*neg.sum(1))/y.shape[1],
                      positive_f1=f1, auroc=auc, brier=((p-y)**2).mean(1), ccc=ccc,
                      mae=np.where(valid, np.abs(s-q), 0).sum(1)/count,
                      sensitivity=tp/np.maximum(pos.sum(1), 1),
                      specificity=tn/np.maximum(neg.sum(1), 1))
        all_seeds.append(np.stack([values[k] for k in METRICS], axis=-1))
    return np.stack(all_seeds)


def align(methods, canonical):
    for name, predictions in methods.items():
        for item in predictions:
            if len(set(item["pids"])) != len(canonical):
                raise ValueError(f"Duplicate or missing participant: {name}")
            lookup = {pid: i for i, pid in enumerate(item["pids"])}
            if set(lookup) != set(canonical):
                raise ValueError(f"Participant sets differ: {name}")
            order = [lookup[pid] for pid in canonical]
            for key in ("labels", "probability", "severity"):
                item[key] = np.asarray(item[key])[order]
            item["pids"] = canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--score-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    report = {"schema": "paired_participant_control_audit_v1", "replicates": args.replicates,
              "sampling": "label-stratified participants, all five fitted seeds retained",
              "intervals": "comparison-wise percentile 95%", "tasks": {}, "input_sha256": {}}
    def read(path):
        raw = path.read_bytes()
        report["input_sha256"][str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    for task_index, task in enumerate(TASKS):
        original = read(args.main_root/task/"FINAL_METRICS.json")
        methods = {"paper_model": [], "random_pim": [], "score_reward": [], "no_score_reward": []}
        runs = {int(run["seed"]): run for run in original["runs"]}
        for seed in SEEDS:
            draw_rows = read(args.score_records/f"{task}_seed_{seed}.json")["rows"]
            methods["paper_model"].append(normalize(runs[seed]["methods"]["gflownet"]["test"], draw_rows))
            for arm in ("random_pim", "score_reward", "no_score_reward"):
                item = read(args.controls/task/f"seed_{seed}"/f"{arm}_TEST.json")
                if item["seed"] != seed or item["task"] != task:
                    raise ValueError("Task/seed mismatch")
                methods[arm].append(normalize(item["test"]))
        canonical = list(methods["paper_model"][0]["pids"])
        align(methods, canonical)
        y = methods["paper_model"][0]["labels"]
        for predictions in methods.values():
            for pred in predictions:
                np.testing.assert_array_equal(pred["labels"], y)
                if not np.isfinite(pred["probability"]).all() or not np.isfinite(pred["severity"]).all():
                    raise ValueError("Nonfinite prediction")
        score_rows = read(args.score_records/f"{task}_seed_7.json")["rows"]
        lookup = {str(row["pid"]): row["score"] for row in score_rows}
        scores = np.asarray([np.nan if lookup[pid] is None else lookup[pid] for pid in canonical])
        observed = {name: scores_for_indices(pred, y, scores, np.arange(len(y))[None])[:, 0]
                    for name, pred in methods.items()}
        for index, seed in enumerate(SEEDS):
            saved = runs[seed]["methods"]["gflownet"]["test"]
            for metric in ("weighted_f1", "positive_f1", "auroc", "brier", "ccc", "mae"):
                expected = saved["severity"]["posterior"][metric] if metric in ("ccc", "mae") else saved["posterior"][metric]
                np.testing.assert_allclose(observed["paper_model"][index, METRICS.index(metric)], expected,
                                           atol=2e-6, rtol=2e-6)
        rng = np.random.default_rng(20261014+task_index)
        bootstrap = {name: [] for name in methods}
        for first in range(0, args.replicates, 200):
            count = min(200, args.replicates-first)
            indices = np.concatenate([rng.choice(np.flatnonzero(y == label),
                size=(count, int((y == label).sum())), replace=True) for label in (0, 1)], axis=1)
            for name, pred in methods.items():
                bootstrap[name].append(scores_for_indices(pred, y, scores, indices).mean(0))
        bootstrap = {name: np.concatenate(chunks) for name, chunks in bootstrap.items()}
        pairs = (("paper_model", "random_pim"), ("score_reward", "no_score_reward"),
                 ("score_reward", "random_pim"))
        report["tasks"][task] = {
            "n": len(y), "n_score": int(np.isfinite(scores).sum()),
            "methods": {name: {metric: {"mean": float(values[:, i].mean()),
                "sd": float(values[:, i].std(ddof=1)), "seeds": values[:, i].tolist()}
                for i, metric in enumerate(METRICS)} for name, values in observed.items()},
            "differences": {left+"_minus_"+right: {metric: {
                "mean": float((observed[left][:, i]-observed[right][:, i]).mean()),
                "participant_95_ci": np.quantile(bootstrap[left][:, i]-bootstrap[right][:, i], [.025, .975]).tolist()}
                for i, metric in enumerate(METRICS)} for left, right in pairs},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
