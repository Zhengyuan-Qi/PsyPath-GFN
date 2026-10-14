#!/usr/bin/env python3
"""Locked EPCM-MH evaluation on the DAIC-WOZ official 107/35/47 split.

This is a retrospective official-test evaluation. Policies are selected from
official-train OOF predictions, the final estimators are fitted on train+dev,
and the 47-participant test split is opened only after the policy is written.
Raw datasets and existing caches remain read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_adyme_mh_daic_official_test as official
from scripts import run_adyme_mh_final_gate_v2 as gate
from scripts import run_epcm_mh_clinical_gate as epcm
from scripts.clinical_label_contract import load_label_profile, validate_contract


SCHEMA = "epcm_mh_daic_official_test_v1"


def normalize_pid(value: Any) -> str:
    return str(int(float(str(value).strip())))


def load_records_official(
    split: str,
    epcm_root: Path,
    benchmark_root: Path,
    contract_dir: Path,
) -> list[dict[str, Any]]:
    frame, binary_spec = load_label_profile(
        contract_dir,
        "daic",
        "depression_phq8_screening",
        split=split,
        include_unlabeled=False,
    )
    score_frame, score_spec = load_label_profile(
        contract_dir,
        "daic",
        "depression_phq_score",
        split=split,
        include_unlabeled=True,
    )
    scores: dict[str, float | None] = {}
    for row in score_frame.to_dict("records"):
        pid = normalize_pid(row["pid"])
        value = row.get(score_spec.target_column)
        mask = row.get(score_spec.mask_column)
        try:
            scores[pid] = (
                float(value)
                if int(mask) == 1 and value is not None and np.isfinite(float(value))
                else None
            )
        except (TypeError, ValueError):
            scores[pid] = None

    contract_path = epcm_root / "daic" / "RUN_CONTRACT.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"missing EPCM run contract: {contract_path}")
    expected_signature = json.loads(contract_path.read_text(encoding="utf-8")).get("run_signature")
    if not expected_signature:
        raise RuntimeError(f"missing run_signature in {contract_path}")

    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        if str(row["split"]) != split:
            continue
        pid = normalize_pid(row["pid"])
        epcm_path = epcm_root / "daic" / "cache" / f"{pid}.npz"
        benchmark_path = benchmark_root / "daic" / "depression" / "text_audio" / f"{pid}_all.npy"
        if not epcm_path.is_file() or not benchmark_path.is_file():
            raise FileNotFoundError(f"missing official DAIC cache for PID {pid}: {epcm_path} {benchmark_path}")
        with np.load(epcm_path, allow_pickle=False) as archive:
            raw = {name: gate.finite(archive[name]) for name in epcm.VIEW_NAMES}
            metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("run_signature") != expected_signature:
            raise RuntimeError(f"cache signature mismatch: {epcm_path}")
        if any(value.ndim != 2 or len(value) == 0 for value in raw.values()):
            raise ValueError(f"invalid EPCM views for PID {pid}: {[value.shape for value in raw.values()]}")
        if len({len(raw[name]) for name in epcm.VIEW_NAMES}) != 1:
            raise ValueError(f"unaligned EPCM views for PID {pid}: {[len(raw[name]) for name in epcm.VIEW_NAMES]}")
        benchmark = gate.finite(np.load(benchmark_path, mmap_mode="r"))
        records.append(
            {
                "pid": pid,
                "split": split,
                "label": int(row[binary_spec.target_column]),
                "score": scores.get(pid),
                "benchmark": benchmark,
                "raw": raw,
                "metadata": metadata,
                "alignment": {
                    "status": "epcm_turn_aligned_official_split",
                    "official_split_field": True,
                    "epcm_events": int(len(raw["clinical"])),
                    "benchmark_events": int(len(benchmark)),
                    "epcm_views_position_aligned": True,
                    "benchmark_pooled_independently": True,
                },
            }
        )
    expected = {"train": 107, "dev": 35, "test": 47}[split]
    if len(records) != expected:
        raise RuntimeError(f"official DAIC {split} count mismatch: {len(records)} != {expected}")
    return records


def configure_epcm_protocol() -> None:
    # The official-split runner provides the lock/evaluation protocol. EPCM
    # supplies the representations, variants, metrics and estimators.
    official.SCHEMA = SCHEMA
    official.VARIANT_NAMES = epcm.VARIANT_NAMES
    official.CONTROL_NAMES = ()
    official.classification_metrics = epcm.classification_metrics
    official.fit_all_variants = epcm.fit_all_variants
    official.load_records_official = load_records_official
    official.load_meld_train = lambda _root: np.empty((0, 0), dtype=np.float32)

    gate.SCHEMA = SCHEMA
    gate.VIEW_NAMES = epcm.VIEW_NAMES
    gate.VARIANT_NAMES = epcm.VARIANT_NAMES
    gate.CONTROL_NAMES = ()
    gate.classification_metrics = epcm.classification_metrics
    gate.fit_all_variants = epcm.fit_all_variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epcm-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    gate.configure_threads(args.threads)
    validate_contract(args.contract_dir)
    configure_epcm_protocol()
    args.output.mkdir(parents=True, exist_ok=True)

    official_args = argparse.Namespace(
        adyme_root=args.epcm_root,
        benchmark_root=args.benchmark_root,
        contract_dir=args.contract_dir,
        output=args.output,
        threads=args.threads,
    )
    payload = {
        "schema": SCHEMA,
        "status": "running",
        "evaluation_type": "retrospective_locked_official_test",
        "controls_computed": False,
        "raw_inputs_read_only": True,
    }
    gate.atomic_json(args.output / "PROGRESS.json", payload)
    payload["dataset"] = official.evaluate(official_args)
    payload["status"] = "complete"
    gate.atomic_json(args.output / "FINAL_METRICS.json", payload)
    print(json.dumps({"event": "epcm_mh_daic_official_test_complete", "output": str(args.output / "FINAL_METRICS.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
