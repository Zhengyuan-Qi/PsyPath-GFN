#!/usr/bin/env python3
"""Build an immutable, audited DAIC-WOZ/E-DAIC label contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


CONTRACT_VERSION = "clinical_label_contract_v1"

DAIC_SOURCE_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "full_test_split.csv",
}

EDAIC_SOURCE_FILES = {
    "train": "labels/train_split.csv",
    "dev": "labels/dev_split.csv",
    "test": "labels/test_split.csv",
}

EDAIC_DETAIL_FILE = "labels/detailed_lables.csv"

# Contract v1 is bound to the exact source files audited on 2026-07-31.
EXPECTED_SOURCE_SHA256 = {
    "daic/train_split_Depression_AVEC2017.csv": "d6fd5dd6ed19f76b961ec67b0b005d2e466444fe3498119729d673b2edfad28a",
    "daic/dev_split_Depression_AVEC2017.csv": "81478dcb078b8cbed07aec79944b8cfdac8bc5d1f16851f6d0b75e81dae95e9d",
    "daic/full_test_split.csv": "66640def4bb104b62c4030453f7ef617441439247f1a99dbb83fa6c15271156b",
    "edaic/labels/train_split.csv": "047c0db53b09d314000943b01c9875487d28353107b57ff3af5352389fb81f84",
    "edaic/labels/dev_split.csv": "a5e25e22cbb09964f9bf605aff056060de8440240235d2b777ee7e729089e566",
    "edaic/labels/test_split.csv": "22b6c23a9e5b7b2fecf3853ef4dcbd9890831f0ce323dad47bf7d1da021aaad3",
    "edaic/labels/detailed_lables.csv": "691784f2fbe551a75afbf64c928df75ecb7b5891dfdf7263327a3663a2d9442f",
}

DAIC_PHQ_ITEMS = [
    "PHQ8_NoInterest",
    "PHQ8_Depressed",
    "PHQ8_Sleep",
    "PHQ8_Tired",
    "PHQ8_Appetite",
    "PHQ8_Failure",
    "PHQ8_Concentrating",
    "PHQ8_Moving",
]

EDAIC_PHQ_ITEMS = [
    "PHQ8_1_NoInterest",
    "PHQ8_2_Depressed",
    "PHQ8_3_Sleep",
    "PHQ8_4_Tired",
    "PHQ8_5_Appetite",
    "PHQ8_6_Failure",
    "PHQ8_7_Concentration",
    "PHQ8_8_Psychomotor",
]

EDAIC_PCL_ITEMS = [
    "PCL-C_1_Memories",
    "PCL-C_2_Dreams",
    "PCL-C_3_Reliving",
    "PCL-C_4_Upset",
    "PCL-C_5_Physical",
    "PCL-C_6_ThoughtAvoidance",
    "PCL-C_7_ActivityAvoidance",
    "PCL-C_8_TroubleRemembering",
    "PCL-C_9_NoInterest",
    "PCL-C_10_FeelDistant",
    "PCL-C_11_FeelNumb",
    "PCL-C_12_Future_Cut_Short",
    "PCL-C_13_Sleep",
    "PCL-C_14_Irritability",
    "PCL-C_15_Concentration",
    "PCL-C_16_HyperAlert",
    "PCL-C_17_Jumpy",
]

LABEL_COLUMNS = [
    "dataset",
    "pid",
    "split",
    "canonical_split",
    "canonical_split_differs",
    "phq_score",
    "phq_score_mask",
    "official_binary",
    "official_binary_mask",
    "phq8_screening_binary",
    "phq8_screening_binary_mask",
    "binary_conflict",
    "phq_items",
    "phq_item_mask",
    "ptsd_score",
    "ptsd_score_mask",
    "ptsd_score_repaired",
    "ptsd_score_repaired_mask",
    "ptsd_score_was_repaired",
    "ptsd_score_lower",
    "ptsd_score_upper",
    "ptsd_official_binary",
    "ptsd_official_binary_mask",
    "ptsd_dsm_iv_binary",
    "ptsd_dsm_iv_binary_mask",
    "ptsd_dsm_iv_lower",
    "ptsd_dsm_iv_upper",
    "ptsd_binary_conflict",
    "pcl_items",
    "pcl_item_mask",
    "label_status",
    "source_file",
    "source_sha256",
    "detail_source_file",
    "detail_source_sha256",
]

CROSS_COLUMNS = [
    "pid",
    "identity_relation",
    "daic_present",
    "edaic_present",
    "daic_split",
    "edaic_split",
    "canonical_split",
    "split_conflict",
    "daic_phq_score",
    "edaic_phq_score",
    "phq_score_conflict",
    "daic_official_binary",
    "edaic_official_binary",
    "official_binary_conflict",
    "phq8_screening_binary",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", skip_blank_lines=True)


def optional_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def json_array(values: Iterable[Any]) -> str:
    normalized = [None if pd.isna(value) else int(value) for value in values]
    return json.dumps(normalized, separators=(",", ":"))


def source_paths(daic_root: Path, edaic_root: Path) -> dict[str, Path]:
    paths = {
        f"daic/{relative}": daic_root / relative
        for relative in DAIC_SOURCE_FILES.values()
    }
    paths.update(
        {
            f"edaic/{relative}": edaic_root / relative
            for relative in EDAIC_SOURCE_FILES.values()
        }
    )
    paths[f"edaic/{EDAIC_DETAIL_FILE}"] = edaic_root / EDAIC_DETAIL_FILE
    return paths


def collect_source_hashes(paths: dict[str, Path]) -> dict[str, str]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source label files: {missing}")
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def assert_expected_source_hashes(hashes: dict[str, str]) -> None:
    if hashes != EXPECTED_SOURCE_SHA256:
        differences = {
            key: {"expected": EXPECTED_SOURCE_SHA256.get(key), "actual": hashes.get(key)}
            for key in sorted(set(hashes) | set(EXPECTED_SOURCE_SHA256))
            if hashes.get(key) != EXPECTED_SOURCE_SHA256.get(key)
        }
        raise ValueError(
            "Source labels do not match the audited contract-v1 release: "
            + json.dumps(differences, indent=2)
        )


def ensure_output_outside_sources(
    output_dir: Path, daic_root: Path, edaic_root: Path
) -> None:
    output = output_dir.resolve()
    for source in (daic_root.resolve(), edaic_root.resolve()):
        try:
            output.relative_to(source)
        except ValueError:
            continue
        raise ValueError(f"Output directory must not be inside raw data: {output}")


def normalize_items(
    values: Iterable[Any], minimum: int, maximum: int
) -> tuple[list[int | None], list[int]]:
    items: list[int | None] = []
    mask: list[int] = []
    for value in values:
        parsed = optional_int(value)
        valid = parsed is not None and minimum <= parsed <= maximum
        items.append(parsed if valid else None)
        mask.append(int(valid))
    return items, mask


def ptsd_cluster_bounds(
    items: list[int | None], mask: list[int]
) -> tuple[int, int]:
    requirements = ((0, 5, 1), (5, 12, 3), (12, 17, 2))
    lower_satisfied = []
    upper_satisfied = []
    for start, stop, required in requirements:
        known_positive = sum(
            int(mask[index] and items[index] is not None and items[index] >= 3)
            for index in range(start, stop)
        )
        unknown = sum(1 - mask[index] for index in range(start, stop))
        lower_satisfied.append(known_positive >= required)
        upper_satisfied.append(known_positive + unknown >= required)
    return int(all(lower_satisfied)), int(all(upper_satisfied))


def empty_ptsd_fields() -> dict[str, Any]:
    return {
        "ptsd_score": None,
        "ptsd_score_mask": 0,
        "ptsd_score_repaired": None,
        "ptsd_score_repaired_mask": 0,
        "ptsd_score_was_repaired": 0,
        "ptsd_score_lower": None,
        "ptsd_score_upper": None,
        "ptsd_official_binary": None,
        "ptsd_official_binary_mask": 0,
        "ptsd_dsm_iv_binary": None,
        "ptsd_dsm_iv_binary_mask": 0,
        "ptsd_dsm_iv_lower": None,
        "ptsd_dsm_iv_upper": None,
        "ptsd_binary_conflict": None,
        "pcl_items": json_array([None] * 17),
        "pcl_item_mask": json_array([0] * 17),
    }


def build_daic_rows(
    root: Path, source_hashes: dict[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, relative in DAIC_SOURCE_FILES.items():
        frame = read_csv(root / relative)
        score_column = "PHQ8_Score" if split != "test" else "PHQ_Score"
        binary_column = "PHQ8_Binary" if split != "test" else "PHQ_Binary"
        for raw in frame.to_dict("records"):
            pid = int(raw["Participant_ID"])
            score = int(raw[score_column])
            official = int(raw[binary_column])
            screening = int(score >= 10)
            statuses: list[str] = []
            if official != screening:
                statuses.append("depression_binary_profile_conflict")

            if split == "test":
                phq_items = [None] * 8
                phq_mask = [0] * 8
                statuses.append("phq_items_not_released")
            else:
                phq_items, phq_mask = normalize_items(
                    [raw.get(column) for column in DAIC_PHQ_ITEMS], 0, 3
                )
                if not all(phq_mask):
                    statuses.append("phq_item_missing_or_invalid_masked")
                if all(phq_mask) and sum(phq_items) != score:  # type: ignore[arg-type]
                    raise ValueError(f"DAIC PID {pid}: PHQ items do not sum to score")

            row = {
                "dataset": "daic",
                "pid": pid,
                "split": split,
                "canonical_split": None,
                "canonical_split_differs": None,
                "phq_score": score,
                "phq_score_mask": 1,
                "official_binary": official,
                "official_binary_mask": 1,
                "phq8_screening_binary": screening,
                "phq8_screening_binary_mask": 1,
                "binary_conflict": int(official != screening),
                "phq_items": json_array(phq_items),
                "phq_item_mask": json_array(phq_mask),
                "label_status": ";".join(statuses) if statuses else "ok",
                "source_file": relative,
                "source_sha256": source_hashes[f"daic/{relative}"],
                "detail_source_file": None,
                "detail_source_sha256": None,
                **empty_ptsd_fields(),
            }
            rows.append(row)
    return rows


def same_optional_number(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return float(left) == float(right)


def build_edaic_rows(
    root: Path, source_hashes: dict[str, str]
) -> list[dict[str, Any]]:
    detail = read_csv(root / EDAIC_DETAIL_FILE).copy()
    detail["pid"] = pd.to_numeric(detail["Participant"]).astype(int)
    if detail["pid"].duplicated().any():
        raise ValueError("E-DAIC detailed labels contain duplicate PIDs")
    detail = detail.set_index("pid", verify_integrity=True)

    rows: list[dict[str, Any]] = []
    for split, relative in EDAIC_SOURCE_FILES.items():
        frame = read_csv(root / relative)
        for raw in frame.to_dict("records"):
            pid = int(raw["Participant_ID"])
            if pid not in detail.index:
                raise ValueError(f"E-DAIC PID {pid}: missing detailed labels")
            item_row = detail.loc[pid]
            if str(item_row["split"]).lower() != split:
                raise ValueError(f"E-DAIC PID {pid}: split/detail mismatch")

            score = int(raw["PHQ_Score"])
            official = int(raw["PHQ_Binary"])
            if score != int(item_row["Depression_severity"]):
                raise ValueError(f"E-DAIC PID {pid}: PHQ score/detail mismatch")
            if official != int(item_row["Depression_label"]):
                raise ValueError(f"E-DAIC PID {pid}: PHQ label/detail mismatch")

            phq_items, phq_mask = normalize_items(
                [item_row[column] for column in EDAIC_PHQ_ITEMS], 0, 3
            )
            if not all(phq_mask) or sum(phq_items) != score:  # type: ignore[arg-type]
                raise ValueError(f"E-DAIC PID {pid}: invalid PHQ detail")

            pcl_items, pcl_mask = normalize_items(
                [item_row[column] for column in EDAIC_PCL_ITEMS], 1, 5
            )
            ptsd_score = optional_int(raw["PTSD Severity"])
            detail_ptsd_score = optional_int(item_row["PTSD_severity"])
            if not same_optional_number(ptsd_score, detail_ptsd_score):
                raise ValueError(f"E-DAIC PID {pid}: PTSD score/detail mismatch")
            ptsd_official = int(raw["PCL-C (PTSD)"])
            if ptsd_official != int(item_row["PTSD_label"]):
                raise ValueError(f"E-DAIC PID {pid}: PTSD label/detail mismatch")

            valid_sum = sum(item for item in pcl_items if item is not None)
            missing_count = pcl_mask.count(0)
            lower_score = valid_sum + missing_count
            upper_score = valid_sum + 5 * missing_count
            was_repaired = int(ptsd_score is None and missing_count == 0)
            repaired_score = (
                ptsd_score
                if ptsd_score is not None
                else valid_sum if missing_count == 0 else None
            )
            if ptsd_score is not None:
                if missing_count or ptsd_score != valid_sum:
                    raise ValueError(f"E-DAIC PID {pid}: invalid PTSD total")
                lower_score = ptsd_score
                upper_score = ptsd_score

            dsm_lower, dsm_upper = ptsd_cluster_bounds(pcl_items, pcl_mask)
            dsm_value = dsm_lower if dsm_lower == dsm_upper else None
            dsm_mask = int(dsm_value is not None)
            conflict = (
                int(ptsd_official != dsm_value) if dsm_value is not None else None
            )
            screening = int(score >= 10)
            statuses: list[str] = []
            if official != screening:
                statuses.append("depression_binary_profile_conflict")
            if pid in {485, 486}:
                statuses.append("current_detail_selected_over_swapped_legacy_rows")
            if was_repaired:
                statuses.append("ptsd_score_repaired_from_complete_items")
            if missing_count:
                statuses.append("pcl_item_out_of_range_or_missing_masked")
                statuses.append("ptsd_score_interval_only")
            if dsm_value is not None and missing_count:
                statuses.append("ptsd_dsm_iv_label_robust_to_missing_item")
            if conflict:
                raise ValueError(f"E-DAIC PID {pid}: official/DSM-IV conflict")

            rows.append(
                {
                    "dataset": "edaic",
                    "pid": pid,
                    "split": split,
                    "canonical_split": split,
                    "canonical_split_differs": 0,
                    "phq_score": score,
                    "phq_score_mask": 1,
                    "official_binary": official,
                    "official_binary_mask": 1,
                    "phq8_screening_binary": screening,
                    "phq8_screening_binary_mask": 1,
                    "binary_conflict": int(official != screening),
                    "phq_items": json_array(phq_items),
                    "phq_item_mask": json_array(phq_mask),
                    "ptsd_score": ptsd_score,
                    "ptsd_score_mask": int(ptsd_score is not None),
                    "ptsd_score_repaired": repaired_score,
                    "ptsd_score_repaired_mask": int(repaired_score is not None),
                    "ptsd_score_was_repaired": was_repaired,
                    "ptsd_score_lower": lower_score,
                    "ptsd_score_upper": upper_score,
                    "ptsd_official_binary": ptsd_official,
                    "ptsd_official_binary_mask": 1,
                    "ptsd_dsm_iv_binary": dsm_value,
                    "ptsd_dsm_iv_binary_mask": dsm_mask,
                    "ptsd_dsm_iv_lower": dsm_lower,
                    "ptsd_dsm_iv_upper": dsm_upper,
                    "ptsd_binary_conflict": conflict,
                    "pcl_items": json_array(pcl_items),
                    "pcl_item_mask": json_array(pcl_mask),
                    "label_status": ";".join(statuses) if statuses else "ok",
                    "source_file": relative,
                    "source_sha256": source_hashes[f"edaic/{relative}"],
                    "detail_source_file": EDAIC_DETAIL_FILE,
                    "detail_source_sha256": source_hashes[
                        f"edaic/{EDAIC_DETAIL_FILE}"
                    ],
                }
            )

    if set(detail.index) != {row["pid"] for row in rows}:
        raise ValueError("E-DAIC split files and detailed labels do not cover the same PIDs")
    return rows


def attach_canonical_splits(
    daic_rows: list[dict[str, Any]], edaic_rows: list[dict[str, Any]]
) -> None:
    canonical = {row["pid"]: row["split"] for row in edaic_rows}
    for row in daic_rows:
        if row["pid"] not in canonical:
            raise ValueError(f"DAIC PID {row['pid']} is absent from E-DAIC")
        row["canonical_split"] = canonical[row["pid"]]
        row["canonical_split_differs"] = int(
            row["split"] != row["canonical_split"]
        )


def build_cross_release_rows(
    daic_rows: list[dict[str, Any]], edaic_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    daic = {row["pid"]: row for row in daic_rows}
    edaic = {row["pid"]: row for row in edaic_rows}
    rows = []
    for pid in sorted(set(daic) | set(edaic)):
        left = daic.get(pid)
        right = edaic.get(pid)
        score_conflict = (
            int(left["phq_score"] != right["phq_score"])
            if left is not None and right is not None
            else None
        )
        binary_conflict = (
            int(left["official_binary"] != right["official_binary"])
            if left is not None and right is not None
            else None
        )
        rows.append(
            {
                "pid": pid,
                "identity_relation": (
                    "paired_release_view" if left is not None else "edaic_only"
                ),
                "daic_present": int(left is not None),
                "edaic_present": int(right is not None),
                "daic_split": left["split"] if left else None,
                "edaic_split": right["split"] if right else None,
                "canonical_split": right["split"] if right else None,
                "split_conflict": (
                    int(left["split"] != right["split"])
                    if left is not None and right is not None
                    else None
                ),
                "daic_phq_score": left["phq_score"] if left else None,
                "edaic_phq_score": right["phq_score"] if right else None,
                "phq_score_conflict": score_conflict,
                "daic_official_binary": left["official_binary"] if left else None,
                "edaic_official_binary": right["official_binary"] if right else None,
                "official_binary_conflict": binary_conflict,
                "phq8_screening_binary": (
                    right["phq8_screening_binary"] if right else None
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["pid"])):
            writer.writerow({column: row.get(column) for column in columns})


def positive_counts(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    return {
        split: sum(int(row[column]) for row in rows if row["split"] == split)
        for split in ("train", "dev", "test")
    }


def validate_records(
    daic_rows: list[dict[str, Any]],
    edaic_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    assert len(daic_rows) == 189
    assert len({row["pid"] for row in daic_rows}) == 189
    assert len(edaic_rows) == 275
    assert len({row["pid"] for row in edaic_rows}) == 275
    assert positive_counts(daic_rows, "phq8_screening_binary") == {
        "train": 31,
        "dev": 12,
        "test": 14,
    }
    assert positive_counts(edaic_rows, "phq8_screening_binary") == {
        "train": 50,
        "dev": 15,
        "test": 21,
    }
    assert sum(row["binary_conflict"] for row in daic_rows) == 1
    assert sum(row["binary_conflict"] for row in edaic_rows) == 20
    assert sum(row["ptsd_binary_conflict"] or 0 for row in edaic_rows) == 0
    assert sum(row["ptsd_score_was_repaired"] for row in edaic_rows) == 1
    assert len(cross_rows) == 275
    assert sum(row["identity_relation"] == "paired_release_view" for row in cross_rows) == 189
    assert sum(row["split_conflict"] or 0 for row in cross_rows) == 104
    assert sum(row["official_binary_conflict"] or 0 for row in cross_rows) == 14
    assert sum(row["phq_score_conflict"] or 0 for row in cross_rows) == 0

    daic = {row["pid"]: row for row in daic_rows}
    edaic = {row["pid"]: row for row in edaic_rows}
    assert daic[409]["official_binary"] == 0
    assert daic[409]["phq8_screening_binary"] == 1
    assert json.loads(daic[319]["phq_item_mask"])[2] == 0
    assert json.loads(edaic[485]["phq_items"]) == [0, 1, 0, 0, 0, 0, 0, 1]
    assert json.loads(edaic[486]["phq_items"]) == [1, 1, 0, 1, 0, 1, 0, 0]
    assert edaic[669]["ptsd_score"] is None
    assert edaic[669]["ptsd_score_repaired"] == 48
    assert edaic[683]["ptsd_score_repaired"] is None
    assert (edaic[683]["ptsd_score_lower"], edaic[683]["ptsd_score_upper"]) == (
        22,
        26,
    )
    assert json.loads(edaic[683]["pcl_item_mask"])[15] == 0
    assert edaic[683]["ptsd_dsm_iv_binary"] == 0

    return {
        "daic_rows": len(daic_rows),
        "edaic_rows": len(edaic_rows),
        "daic_phq8_screening_positive": positive_counts(
            daic_rows, "phq8_screening_binary"
        ),
        "edaic_phq8_screening_positive": positive_counts(
            edaic_rows, "phq8_screening_binary"
        ),
        "daic_binary_profile_conflicts": sum(
            row["binary_conflict"] for row in daic_rows
        ),
        "edaic_binary_profile_conflicts": sum(
            row["binary_conflict"] for row in edaic_rows
        ),
        "ptsd_dsm_iv_conflicts": 0,
        "cross_release_overlap": 189,
        "cross_release_split_conflicts": 104,
        "cross_release_official_binary_conflicts": 14,
    }


def contract_readme() -> str:
    return """# Clinical label contract v1

This directory is a derived, auditable label layer. It never replaces or edits
the raw DAIC-WOZ or E-DAIC CSV files.

## Required profile selection

Every training script must select one explicit profile through
`scripts/clinical_label_contract.py`:

- `depression_phq8_screening`: PHQ-8 score >= 10. This is the default task for
  new depression-screening experiments.
- `depression_official_binary`: the binary field released with each dataset.
  Use it for protocol-matched comparison with papers that used that field.
- `depression_phq_score`: continuous PHQ-8 severity.
- `ptsd_official_binary`: E-DAIC's released PTSD diagnosis field.
- `ptsd_dsm_iv_binary`: the equivalent PCL-C DSM-IV symptom-cluster rule.
- `ptsd_score_repaired`: PCL-C severity where uniquely recoverable.

There is deliberately no generic `label` column. Results from different
profiles are different tasks and must never be placed in the same result column.

## Audited repairs and masks

- DAIC PID 409 remains 0 in `official_binary` and is 1 in
  `phq8_screening_binary`; the source conflict is explicit.
- DAIC PID 319 has its missing sleep item masked, never filled with zero.
- E-DAIC PHQ items use `detailed_lables.csv`; the swapped PID 485/486 rows in
  the legacy `Detailed_PHQ8_Labels.csv` are excluded.
- E-DAIC PID 669 has a uniquely reconstructed PCL-C total of 48.
- E-DAIC PID 683 has PCL item 16 masked and only a total interval of 22--26.
  No guessed total is supplied. Its negative DSM-IV label is interval-robust.

`manifest.json` binds this contract to exact source SHA-256 values.
`COMPLETE.json` is written last and proves atomic publication completed.
"""


def build_manifest(
    daic_root: Path,
    edaic_root: Path,
    source_hashes: dict[str, str],
    output_hashes: dict[str, str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CONTRACT_VERSION,
        "contract_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Derived labels only; raw source labels remain byte-identical.",
        "source_roots": {
            "daic": str(daic_root.resolve()),
            "edaic": str(edaic_root.resolve()),
        },
        "source_files_sha256": source_hashes,
        "source_hashes_match_audit_20260731": True,
        "excluded_source": {
            "edaic/labels/Detailed_PHQ8_Labels.csv": (
                "Legacy PHQ item table with swapped PID 485/486 rows; never loaded."
            )
        },
        "profiles": {
            "depression_phq8_screening": {
                "column": "phq8_screening_binary",
                "task": "binary",
                "rule": "PHQ-8 score >= 10",
                "recommended_for": "new depression-screening experiments",
            },
            "depression_official_binary": {
                "column": "official_binary",
                "task": "binary",
                "recommended_for": "protocol-matched historical comparison",
            },
            "depression_phq_score": {
                "column": "phq_score",
                "task": "continuous",
                "recommended_metrics": ["CCC", "MAE", "RMSE"],
            },
            "ptsd_official_binary": {
                "column": "ptsd_official_binary",
                "task": "binary",
                "datasets": ["edaic"],
            },
            "ptsd_dsm_iv_binary": {
                "column": "ptsd_dsm_iv_binary",
                "task": "binary",
                "rule": "PCL-C DSM-IV clusters: 1/5, 3/7, 2/5 items >= 3",
                "datasets": ["edaic"],
            },
            "ptsd_score_repaired": {
                "column": "ptsd_score_repaired",
                "task": "continuous",
                "datasets": ["edaic"],
            },
        },
        "repairs": {
            "daic_409": "Threshold profile is 1; official profile remains 0.",
            "daic_319": "Missing PHQ sleep item is masked.",
            "edaic_485_486": "Current corrected detail table selected over legacy table.",
            "edaic_669": "Missing PTSD total uniquely reconstructed as 48.",
            "edaic_683": "Invalid PCL item 16 masked; total retained only as [22, 26].",
        },
        "summary": summary,
        "invariants": {
            "raw_source_hashes_equal_before_after": True,
            "generic_label_column_forbidden": True,
            "canonical_joint_split_source": "E-DAIC official split",
        },
        "output_files_sha256": output_hashes,
    }


def verify_existing(output_dir: Path) -> dict[str, Any]:
    complete_path = output_dir / "COMPLETE.json"
    manifest_path = output_dir / "manifest.json"
    if not complete_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"Incomplete contract directory: {output_dir}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if complete["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("Manifest hash does not match COMPLETE.json")
    for relative, expected in manifest["output_files_sha256"].items():
        actual = sha256_file(output_dir / relative)
        if actual != expected:
            raise ValueError(f"Output hash mismatch for {relative}")
    return manifest


def build_contract(daic_root: Path, edaic_root: Path, output_dir: Path) -> None:
    daic_root = daic_root.resolve()
    edaic_root = edaic_root.resolve()
    output_dir = output_dir.resolve()
    ensure_output_outside_sources(output_dir, daic_root, edaic_root)
    if output_dir.exists():
        raise FileExistsError(
            f"Immutable contract already exists: {output_dir}. Use --verify-existing."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    paths = source_paths(daic_root, edaic_root)
    hashes_before = collect_source_hashes(paths)
    assert_expected_source_hashes(hashes_before)

    daic_rows = build_daic_rows(daic_root, hashes_before)
    edaic_rows = build_edaic_rows(edaic_root, hashes_before)
    attach_canonical_splits(daic_rows, edaic_rows)
    cross_rows = build_cross_release_rows(daic_rows, edaic_rows)
    summary = validate_records(daic_rows, edaic_rows, cross_rows)

    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    )
    try:
        write_csv(temp_dir / "daic_labels.csv", daic_rows, LABEL_COLUMNS)
        write_csv(temp_dir / "edaic_labels.csv", edaic_rows, LABEL_COLUMNS)
        write_csv(
            temp_dir / "cross_release_identity.csv", cross_rows, CROSS_COLUMNS
        )
        (temp_dir / "README.md").write_text(contract_readme(), encoding="utf-8")

        hashes_after = collect_source_hashes(paths)
        if hashes_before != hashes_after:
            raise RuntimeError("Raw source labels changed while building the contract")

        output_files = [
            "daic_labels.csv",
            "edaic_labels.csv",
            "cross_release_identity.csv",
            "README.md",
        ]
        output_hashes = {
            name: sha256_file(temp_dir / name) for name in output_files
        }
        manifest = build_manifest(
            daic_root,
            edaic_root,
            hashes_before,
            output_hashes,
            summary,
        )
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        complete = {
            "status": "complete",
            "schema": CONTRACT_VERSION,
            "manifest_sha256": sha256_file(manifest_path),
            "raw_source_hashes_verified_before_and_after": True,
        }
        (temp_dir / "COMPLETE.json").write_text(
            json.dumps(complete, indent=2) + "\n", encoding="utf-8"
        )

        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    verify_existing(output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "summary": summary,
                "raw_source_hashes_unchanged": True,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daic-root", type=Path)
    parser.add_argument("--edaic-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    if args.verify_existing:
        manifest = verify_existing(args.output_dir.resolve())
        print(json.dumps({"status": "verified", "summary": manifest["summary"]}, indent=2))
        return
    if args.daic_root is None or args.edaic_root is None:
        parser.error("--daic-root and --edaic-root are required when building")
    build_contract(args.daic_root, args.edaic_root, args.output_dir)


if __name__ == "__main__":
    main()
