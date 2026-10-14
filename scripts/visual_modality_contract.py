#!/usr/bin/env python3
"""Strict provenance contract for clinical visual feature inputs.

The released DAIC-WOZ/E-DAIC packages contain several feature files whose
names look visual but are actually audio embeddings.  This module is the
single admission point for visual sources used by cache builders.  A source
must be whitelisted by dataset and must not match a forbidden audio pattern.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "clinical_visual_modality_contract_v1"

_ALLOWED: dict[str, tuple[tuple[str, str], ...]] = {
    "daic": (
        ("openface_au", "{pid}_CLNF_AUs.txt"),
        ("openface_gaze", "{pid}_CLNF_gaze.txt"),
        ("openface_pose", "{pid}_CLNF_pose.txt"),
        ("openface_geometry_2d", "{pid}_CLNF_features.txt"),
        ("openface_geometry_3d", "{pid}_CLNF_features3D.txt"),
    ),
    "edaic": (
        ("openface", "{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"),
        ("cnn_resnet", "{pid}_CNN_ResNet.mat"),
        ("cnn_vgg", "{pid}_CNN_VGG.mat"),
    ),
}

_FORBIDDEN_AUDIO_PATTERNS = (
    re.compile(r"(?:^|[_-])densenet201\.csv$", re.I),
    re.compile(r"(?:^|[_-])vgg16\.csv$", re.I),
    re.compile(r"opensmile", re.I),
    re.compile(r"openface.*(?:egemaps|mfcc|boaw)", re.I),
    re.compile(r"(?:^|[_-])(?:egemaps|mfcc|boaw).*\.csv$", re.I),
)

_DEFAULT_PROTOCOL_SOURCES: dict[str, tuple[str, ...]] = {
    "daic": ("openface_au", "openface_gaze", "openface_pose", "openface_geometry_2d"),
    "edaic": ("openface", "cnn_resnet", "cnn_vgg"),
}


def _pid_text(pid: str | int) -> str:
    return str(pid).strip()


def allowed_source_specs(dataset: str, pid: str | int) -> dict[str, dict[str, Any]]:
    dataset = str(dataset).lower()
    if dataset not in _ALLOWED:
        raise ValueError(f"unsupported clinical dataset: {dataset}")
    value = _pid_text(pid)
    return {
        name: {
            "name": name,
            "modality": "visual",
            "dataset": dataset,
            "pattern": pattern,
            "filename": pattern.format(pid=value),
        }
        for name, pattern in _ALLOWED[dataset]
    }


def visual_paths(dataset: str, root: Path, pid: str | int) -> dict[str, Path]:
    """Return only existing, explicitly whitelisted visual files."""
    specs = allowed_source_specs(dataset, pid)
    if dataset == "daic":
        base = root / "data"
    else:
        base = root / "data" / f"{_pid_text(pid)}_P" / "features"
    result = {name: base / spec["filename"] for name, spec in specs.items()}
    return {name: path for name, path in result.items() if path.is_file()}


def default_protocol_visual_paths(dataset: str, root: Path, pid: str | int) -> dict[str, Path]:
    """Return the complete fixed source set for the primary visual protocol."""
    dataset = str(dataset).lower()
    available = visual_paths(dataset, root, pid)
    required = _DEFAULT_PROTOCOL_SOURCES[dataset]
    missing = [name for name in required if name not in available]
    if missing:
        raise FileNotFoundError(
            f"incomplete default visual protocol for {dataset}/{pid}; missing={missing}"
        )
    return {name: available[name] for name in required}


def _forbidden_matches(path: Path) -> list[str]:
    name = path.name
    return [pattern.pattern for pattern in _FORBIDDEN_AUDIO_PATTERNS if pattern.search(name)]


def validate_visual_paths(
    dataset: str,
    root: Path,
    pid: str | int,
    paths: dict[str, Path] | None = None,
    *,
    require_any: bool = True,
) -> dict[str, Any]:
    """Validate a visual plan and return serializable provenance metadata.

    This deliberately rejects unknown keys and unknown filenames, rather than
    trying to infer a modality from numeric dimensionality.
    """
    expected = allowed_source_specs(dataset, pid)
    selected = visual_paths(dataset, root, pid) if paths is None else dict(paths)
    unknown_keys = sorted(set(selected) - set(expected))
    if unknown_keys:
        raise ValueError(f"visual source key is not whitelisted: {unknown_keys}")
    records = []
    for key, path_value in selected.items():
        path = Path(path_value)
        forbidden = _forbidden_matches(path)
        if forbidden:
            raise ValueError(
                f"audio feature admitted to visual branch: {path.name}; patterns={forbidden}"
            )
        spec = expected[key]
        if path.name != spec["filename"]:
            raise ValueError(
                f"visual filename mismatch for {dataset}/{pid}/{key}: "
                f"got={path.name} expected={spec['filename']}"
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "name": key,
                "modality": "visual",
                "dataset": str(dataset).lower(),
                "path": str(path),
                "filename": path.name,
                "format": path.suffix.lower().lstrip("."),
                "size": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
            }
        )
    if require_any and not records:
        raise FileNotFoundError(f"no whitelisted visual source for {dataset}/{pid}")
    return {
        "schema": SCHEMA,
        "dataset": str(dataset).lower(),
        "pid": _pid_text(pid),
        "modality": "visual",
        "source_count": len(records),
        "sources": records,
        "forbidden_audio_patterns": [pattern.pattern for pattern in _FORBIDDEN_AUDIO_PATTERNS],
    }


def contract_hash() -> str:
    payload = {
        "schema": SCHEMA,
        "allowed": _ALLOWED,
        "forbidden_audio_patterns": [pattern.pattern for pattern in _FORBIDDEN_AUDIO_PATTERNS],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def assert_visual_metadata(metadata: dict[str, Any]) -> None:
    """Reject cache metadata that is absent, stale, or not visual-only."""
    if metadata.get("visual_modality_contract") != SCHEMA:
        raise ValueError("cache is missing the strict visual modality contract")
    if metadata.get("visual_modality_contract_sha256") != contract_hash():
        raise ValueError("cache visual modality contract hash mismatch")
    audit = metadata.get("visual_audit")
    if not isinstance(audit, dict) or audit.get("modality") != "visual":
        raise ValueError("cache visual audit is missing or has a non-visual modality")
    dataset = str(audit.get("dataset", metadata.get("dataset", ""))).lower()
    pid = str(audit.get("pid", metadata.get("pid", "")))
    expected = allowed_source_specs(dataset, pid)
    for source in audit.get("sources", []):
        if source.get("modality") != "visual":
            raise ValueError(f"non-visual source in visual audit: {source}")
        if _forbidden_matches(Path(str(source.get("path", "")))):
            raise ValueError(f"forbidden audio source in visual audit: {source}")
        name = str(source.get("name", ""))
        if name not in expected:
            raise ValueError(f"unknown visual source in audit: {source}")
        if source.get("filename") != expected[name]["filename"]:
            raise ValueError(f"visual audit filename mismatch: {source}")
