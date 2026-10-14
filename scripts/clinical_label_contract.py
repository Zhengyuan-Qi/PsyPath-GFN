#!/usr/bin/env python3
"""Validated loader for clinical_label_contract_v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ProfileSpec:
    target_column: str
    mask_column: str
    task: str
    datasets: frozenset[str]


PROFILE_SPECS = {
    "depression_phq8_screening": ProfileSpec(
        "phq8_screening_binary",
        "phq8_screening_binary_mask",
        "binary",
        frozenset({"daic", "edaic"}),
    ),
    "depression_official_binary": ProfileSpec(
        "official_binary",
        "official_binary_mask",
        "binary",
        frozenset({"daic", "edaic"}),
    ),
    "depression_phq_score": ProfileSpec(
        "phq_score",
        "phq_score_mask",
        "continuous",
        frozenset({"daic", "edaic"}),
    ),
    "ptsd_official_binary": ProfileSpec(
        "ptsd_official_binary",
        "ptsd_official_binary_mask",
        "binary",
        frozenset({"edaic"}),
    ),
    "ptsd_dsm_iv_binary": ProfileSpec(
        "ptsd_dsm_iv_binary",
        "ptsd_dsm_iv_binary_mask",
        "binary",
        frozenset({"edaic"}),
    ),
    "ptsd_score_repaired": ProfileSpec(
        "ptsd_score_repaired",
        "ptsd_score_repaired_mask",
        "continuous",
        frozenset({"edaic"}),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(contract_dir: Path) -> dict:
    contract_dir = contract_dir.resolve()
    complete_path = contract_dir / "COMPLETE.json"
    manifest_path = contract_dir / "manifest.json"
    if not complete_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"Contract is not atomically complete: {contract_dir}")

    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete":
        raise ValueError("Contract COMPLETE marker is not valid")
    if complete.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Contract manifest hash mismatch")
    if not complete.get("raw_source_hashes_verified_before_and_after"):
        raise ValueError("Contract lacks source immutability verification")
    if manifest.get("schema") != "clinical_label_contract_v1":
        raise ValueError(f"Unsupported contract schema: {manifest.get('schema')}")
    for relative, expected in manifest["output_files_sha256"].items():
        path = contract_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Contract file hash mismatch: {relative}")
    return manifest


def load_label_profile(
    contract_dir: Path | str,
    dataset: str,
    profile: str,
    split: str | None = None,
    include_unlabeled: bool = False,
) -> tuple[pd.DataFrame, ProfileSpec]:
    """Load a label table and return its explicit target specification."""

    root = Path(contract_dir).resolve()
    validate_contract(root)
    dataset = dataset.lower()
    if profile not in PROFILE_SPECS:
        raise ValueError(
            f"Unknown label profile {profile!r}; choose one of {sorted(PROFILE_SPECS)}"
        )
    spec = PROFILE_SPECS[profile]
    if dataset not in spec.datasets:
        raise ValueError(f"Profile {profile!r} is not available for {dataset!r}")
    if dataset not in {"daic", "edaic"}:
        raise ValueError("dataset must be 'daic' or 'edaic'")

    label_path = root / f"{dataset}_labels.csv"
    if split is None:
        frame = pd.read_csv(label_path)
    else:
        # Read the split column first, then parse only the requested data rows.
        # This prevents labels from other official splits from entering memory
        # before a locked test evaluation.
        split_values = pd.read_csv(label_path, usecols=["split"])["split"].astype(str)
        selected_lines = {
            index + 1
            for index, value in enumerate(split_values.tolist())
            if value == split
        }
        frame = pd.read_csv(
            label_path,
            skiprows=lambda line: line != 0 and line not in selected_lines,
        )
    forbidden = {"label", "target"} & set(frame.columns)
    if forbidden:
        raise ValueError(f"Ambiguous target columns are forbidden: {sorted(forbidden)}")
    if split is not None:
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        if not frame["split"].eq(split).all():
            raise ValueError(f"split-isolated load admitted a non-{split} row")
    if not include_unlabeled:
        frame = frame.loc[frame[spec.mask_column].eq(1)].copy()
    frame.attrs["label_profile"] = profile
    frame.attrs["target_column"] = spec.target_column
    frame.attrs["target_mask_column"] = spec.mask_column
    frame.attrs["task"] = spec.task
    return frame.reset_index(drop=True), spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("daic", "edaic"), required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_SPECS), required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--include-unlabeled", action="store_true")
    args = parser.parse_args()
    frame, spec = load_label_profile(
        args.contract_dir,
        args.dataset,
        args.profile,
        split=args.split,
        include_unlabeled=args.include_unlabeled,
    )
    target = pd.to_numeric(frame[spec.target_column], errors="coerce")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "profile": args.profile,
                "split": args.split,
                "task": spec.task,
                "target_column": spec.target_column,
                "target_mask_column": spec.mask_column,
                "rows": len(frame),
                "target_non_null": int(target.notna().sum()),
                "target_sum": float(target.sum()) if spec.task == "binary" else None,
                "target_min": float(target.min()) if len(target) else None,
                "target_max": float(target.max()) if len(target) else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
