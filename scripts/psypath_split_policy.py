#!/usr/bin/env python3
"""Leakage-safe identity governance for joint DAIC/E-DAIC PsyPath runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.clinical_label_contract import (
    PROFILE_SPECS,
    ProfileSpec,
    validate_contract,
)


SPLIT_POLICY = "joint_primary_clean"
PARTITIONS = ("train", "dev", "test")
EXPECTED_RELEASE_COUNTS = {
    "train": {"daic": 108, "edaic": 128},
    "dev": {"daic": 34, "edaic": 44},
    "test": {"daic": 47, "edaic": 56},
}
EXPECTED_IDENTITY_COUNTS = {"train": 128, "dev": 44, "test": 103}


def normalize_pid(value: Any) -> str:
    return str(int(float(str(value).strip())))


def _pid_set(frame: pd.DataFrame) -> set[str]:
    return {normalize_pid(value) for value in frame["pid"].tolist()}


def official_test_identities(contract_dir: Path | str) -> dict[str, set[str]]:
    root = Path(contract_dir).resolve()
    validate_contract(root)
    result: dict[str, set[str]] = {}
    for dataset in ("daic", "edaic"):
        frame = pd.read_csv(root / f"{dataset}_labels.csv", usecols=["pid", "split"])
        result[dataset] = _pid_set(frame.loc[frame["split"].astype(str).eq("test")])
    result["all"] = result["daic"] | result["edaic"]
    return result


def load_joint_partition(
    contract_dir: Path | str,
    dataset: str,
    profile: str,
    partition: str,
    *,
    include_unlabeled: bool = False,
) -> tuple[pd.DataFrame, ProfileSpec, dict[str, Any]]:
    """Load one governed partition without admitting labels from another partition."""

    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    root = Path(contract_dir).resolve()
    validate_contract(root)
    dataset = dataset.lower()
    if profile not in PROFILE_SPECS:
        raise ValueError(f"unknown label profile: {profile}")
    spec = PROFILE_SPECS[profile]
    if dataset not in spec.datasets:
        raise ValueError(f"profile {profile!r} is unavailable for {dataset!r}")
    label_path = root / f"{dataset}_labels.csv"
    metadata = pd.read_csv(
        label_path,
        usecols=["pid", "split", "canonical_split"],
    )
    metadata["pid"] = metadata["pid"].map(normalize_pid)
    metadata["split"] = metadata["split"].astype(str)
    metadata["canonical_split"] = metadata["canonical_split"].astype(str)
    reserved = official_test_identities(root)

    if partition in {"train", "dev"}:
        selected_mask = metadata["canonical_split"].eq(partition)
        selected_mask &= ~metadata["pid"].isin(reserved["all"])
        selection_field = "canonical_split"
    else:
        # Each released official Test is retained for its protocol-matched report.
        # Both Test identity sets were globally excluded from joint Train and Dev.
        selected_mask = metadata["split"].eq("test")
        selection_field = "official_split"

    selected_lines = {
        index + 1 for index in metadata.index[selected_mask].tolist()
    }
    selected = pd.read_csv(
        label_path,
        skiprows=lambda line: line != 0 and line not in selected_lines,
    )
    selected["pid"] = selected["pid"].map(normalize_pid)
    selected["split"] = selected["split"].astype(str)
    selected["canonical_split"] = selected["canonical_split"].astype(str)
    if not include_unlabeled:
        selected = selected.loc[selected[spec.mask_column].eq(1)].copy()

    if selected["pid"].duplicated().any():
        duplicates = selected.loc[selected["pid"].duplicated(False), "pid"].tolist()
        raise RuntimeError(f"duplicate {dataset} identities in {partition}: {duplicates[:10]}")
    expected = EXPECTED_RELEASE_COUNTS[partition][dataset]
    if len(selected) != expected:
        raise RuntimeError(
            f"{SPLIT_POLICY} {dataset} {partition} count mismatch: "
            f"{len(selected)} != {expected}"
        )
    selected["protocol_split"] = partition
    audit = {
        "split_policy": SPLIT_POLICY,
        "dataset": dataset,
        "partition": partition,
        "selection_field": selection_field,
        "rows": int(len(selected)),
        "identities": int(selected["pid"].nunique()),
        "excluded_official_test_identities": int(
            len(
                _pid_set(metadata.loc[metadata["canonical_split"].eq(partition)])
                & reserved["all"]
            )
        )
        if partition in {"train", "dev"}
        else 0,
        "official_test_identity_union": int(len(reserved["all"])),
    }
    return selected.reset_index(drop=True), spec, audit


def score_map_by_pid(
    contract_dir: Path | str,
    dataset: str,
    profile: str,
    partition: str,
) -> dict[str, float | None]:
    frame, spec, _ = load_joint_partition(
        contract_dir,
        dataset,
        profile,
        partition,
        include_unlabeled=True,
    )
    result: dict[str, float | None] = {}
    for row in frame.to_dict("records"):
        pid = normalize_pid(row["pid"])
        if pid in result:
            raise RuntimeError(f"duplicate score row for {dataset}/{pid}")
        value = row.get(spec.target_column)
        mask = row.get(spec.mask_column)
        try:
            result[pid] = float(value) if int(mask) == 1 and pd.notna(value) else None
        except (TypeError, ValueError):
            result[pid] = None
    return result


def split_audit(contract_dir: Path | str) -> dict[str, Any]:
    """Prove identity isolation for the shared model before any GPU launch."""

    root = Path(contract_dir).resolve()
    validate_contract(root)
    reserved = official_test_identities(root)
    partitions: dict[str, dict[str, set[str]]] = {}
    release_rows: dict[str, dict[str, int]] = {}
    metadata = {
        dataset: pd.read_csv(
            root / f"{dataset}_labels.csv",
            usecols=["pid", "split", "canonical_split"],
        )
        for dataset in ("daic", "edaic")
    }
    for frame in metadata.values():
        frame["pid"] = frame["pid"].map(normalize_pid)
        frame["split"] = frame["split"].astype(str)
        frame["canonical_split"] = frame["canonical_split"].astype(str)
    for partition in PARTITIONS:
        partitions[partition] = {}
        release_rows[partition] = {}
        for dataset in ("daic", "edaic"):
            frame = metadata[dataset]
            if partition in {"train", "dev"}:
                selected = frame.loc[
                    frame["canonical_split"].eq(partition)
                    & ~frame["pid"].isin(reserved["all"])
                ]
            else:
                selected = frame.loc[frame["split"].eq("test")]
            partitions[partition][dataset] = _pid_set(selected)
            release_rows[partition][dataset] = int(len(selected))

    identities = {
        partition: partitions[partition]["daic"] | partitions[partition]["edaic"]
        for partition in PARTITIONS
    }
    overlaps = {
        "train_dev": sorted(identities["train"] & identities["dev"]),
        "train_test": sorted(identities["train"] & identities["test"]),
        "dev_test": sorted(identities["dev"] & identities["test"]),
    }
    repeated_release_views = {
        partition: len(
            partitions[partition]["daic"] & partitions[partition]["edaic"]
        )
        for partition in PARTITIONS
    }
    failures = []
    for name, values in overlaps.items():
        if values:
            failures.append(f"identity overlap {name}: {values[:10]}")
    for partition, expected in EXPECTED_IDENTITY_COUNTS.items():
        if len(identities[partition]) != expected:
            failures.append(
                f"identity count {partition}: {len(identities[partition])} != {expected}"
            )

    cross_path = root / "cross_release_identity.csv"
    cross_digest = hashlib.sha256(cross_path.read_bytes()).hexdigest()
    return {
        "schema": "psypath_joint_identity_split_audit_v1",
        "status": "passed" if not failures else "failed",
        "split_policy": SPLIT_POLICY,
        "selection": {
            "train_dev": "canonical_split with all DAIC and E-DAIC official Test identities excluded",
            "test": "dataset-specific official Test after model and thresholds are locked",
        },
        "release_rows": release_rows,
        "unique_identities": {
            partition: len(values) for partition, values in identities.items()
        },
        "repeated_release_views": repeated_release_views,
        "overlaps": overlaps,
        "official_test_identity_union": len(reserved["all"]),
        "cross_release_identity_sha256": cross_digest,
        "failures": failures,
        "test_labels_loaded": False,
    }
