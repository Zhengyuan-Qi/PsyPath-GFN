"""Reproduce path-budget curves from locked draws; outputs only aggregate statistics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev

TASKS = ("daic_depression", "edaic_depression", "edaic_ptsd")
SEEDS = (7, 17, 27, 41, 53)
KS = (1, 2, 4, 8, 16, 48)
PROFILES = {"daic_depression": "depression_phq8_screening",
            "edaic_depression": "depression_phq8_screening",
            "edaic_ptsd": "ptsd_dsm_iv_binary"}


def summary(values):
    return {"mean": fmean(values), "sd": stdev(values), "values": list(values)}


def losses(rows, method, k):
    brier, absolute = [], []
    for row in rows:
        probabilities, severities = row[f"{method}_probability"], row[f"{method}_severity"]
        if len(probabilities) != 48 or len(severities) != 48 or 48 % k:
            raise ValueError("Each participant must have exactly 48 fixed path draws")
        if row["label"] not in (0, 1):
            raise ValueError("Expected an observed binary screening label")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities):
            raise ValueError("Invalid probability")
        if any(not math.isfinite(v) for v in severities):
            raise ValueError("Invalid severity prediction")
        score = row["score"]
        if score is not None and not math.isfinite(score):
            raise ValueError("Unknown scores must be null, not nonfinite values")
        # Use every original draw at every budget, with disjoint consecutive groups.
        for start in range(0, 48, k):
            brier.append((fmean(probabilities[start:start+k]) - row["label"]) ** 2)
            if score is not None:
                absolute.append(abs(fmean(severities[start:start+k]) - score))
    if not brier or not absolute:
        raise ValueError("No evaluable screening labels or severity scores")
    return {"brier": fmean(brier), "mae": fmean(absolute)}


def aggregate(folder, tasks=TASKS):
    output = {"schema": "psypath_grouped_paths_v1", "seeds": list(SEEDS),
              "budgets": list(KS), "tasks": {}, "input_sha256": {}}
    for task in tasks:
        values = {m: {metric: {str(k): [] for k in KS} for metric in ("brier", "mae")}
                  for m in ("learned", "random")}
        identity = None
        for seed in SEEDS:
            path = Path(folder) / f"{task}_seed_{seed}.json"
            raw = path.read_bytes()
            record = json.loads(raw)
            if (record["task"], record["seed"], record["split"], record["label_profile"]) != (
                    task, seed, "test", PROFILES[task]):
                raise ValueError("Task, split, seed or label profile does not match the paper protocol")
            rows = sorted(record["rows"], key=lambda row: str(row["pid"]))
            expected = 47 if task == "daic_depression" else 56
            if len(rows) != expected or len({str(r["pid"]) for r in rows}) != expected:
                raise ValueError("Wrong participant count or duplicate participant IDs")
            current = [(str(r["pid"]), r["label"], r["score"]) for r in rows]
            if identity is not None and current != identity:
                raise ValueError("Participant identities or targets differ across seeds")
            identity = current
            output["input_sha256"][path.name] = hashlib.sha256(raw).hexdigest()
            for method in values:
                for k in KS:
                    result = losses(rows, method, k)
                    for metric, value in result.items():
                        values[method][metric][str(k)].append(value)
        result = {m: {metric: {k: summary(v) for k, v in by_k.items()}
                      for metric, by_k in measures.items()} for m, measures in values.items()}
        for method in values:
            result[method]["relative_reduction_percent"] = {}
            for metric in ("brier", "mae"):
                baseline = values[method][metric]["1"]
                if any(v <= 0 for v in baseline):
                    raise ValueError("Relative reduction requires positive single-path loss")
                result[method]["relative_reduction_percent"][metric] = {
                    str(k): summary([100 * (1 - end / start)
                                     for start, end in zip(baseline, values[method][metric][str(k)])])
                    for k in KS}
        result["paired_gap_48"] = {}
        for metric in ("brier", "mae"):
            diffs = [a - b for a, b in zip(values["learned"][metric]["48"],
                                          values["random"][metric]["48"])]
            entry = summary(diffs)
            half = 2.7764451051977987 * entry["sd"] / math.sqrt(5)
            entry["ci95"] = [entry["mean"] - half, entry["mean"] + half]
            result["paired_gap_48"][metric] = entry
        output["tasks"][task] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()
    result = aggregate(args.input_dir, args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"tasks": list(result["tasks"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
