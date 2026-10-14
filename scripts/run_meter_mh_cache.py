#!/usr/bin/env python3
"""Run the audited EPCM cache engine with the METER-MH prompt contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import meter_mh_prompts as meter_prompts
from scripts import run_epcm_mh_cache as engine
from scripts import visual_modality_contract


engine.SCHEMA = "meter_mh_cache_v1"
engine.prompts = meter_prompts


def meter_implementation_fingerprint() -> str:
    paths = (
        Path(__file__).resolve(),
        Path(meter_prompts.__file__).resolve(),
        Path(engine.__file__).resolve(),
        Path(engine.base.__file__).resolve(),
        Path(visual_modality_contract.__file__).resolve(),
    )
    payload = [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths
    ]
    return engine.sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


engine.encoder_implementation_fingerprint = meter_implementation_fingerprint


def load_all_official_rows(args: Any, dataset: str) -> list[dict[str, Any]]:
    """Cache every participant in the official release splits without labels."""
    frame, _ = engine.base.load_label_profile(
        args.contract_dir,
        dataset,
        "depression_phq8_screening",
        include_unlabeled=True,
    )
    rows: list[dict[str, Any]] = []
    for item in frame.to_dict("records"):
        if dataset == "edaic" and str(item.get("canonical_split", "")) != str(item["split"]):
            raise RuntimeError(
                f"E-DAIC official/canonical split mismatch pid={item['pid']}: "
                f"official={item['split']} canonical={item.get('canonical_split')}"
            )
        pid = str(int(item["pid"]))
        rows.append(
            {
                "item_id": pid,
                "pid": pid,
                "split": str(item["split"]),
                "canonical_split": str(item["canonical_split"]),
                "target": "instrument_neutral_clinical_measurement",
            }
        )
    rows.sort(key=lambda row: ({"train": 0, "dev": 1, "test": 2}[row["split"]], int(row["pid"])))
    return rows


engine.load_clinical_rows = load_all_official_rows


if __name__ == "__main__":
    engine.main()
