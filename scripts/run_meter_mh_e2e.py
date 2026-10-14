#!/usr/bin/env python3
"""Strict end-to-end METER-MH validation.

HumanOmniV2 is executed online for every response.  Only the previous strict
METER OOF probabilities may be used as training-only teachers; cached hidden
states are used solely by Gate-0 to audit frozen online equivalence.  The
official test labels are not loaded until Gate-0 and all three Gate-1
non-inferiority checks have passed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import meter_mh_prompts as prompts
from scripts import run_epcm_mh_cache as cache_engine
from scripts import visual_modality_contract
from scripts.clinical_label_contract import load_label_profile, validate_contract
from scripts.meter_mh_e2e_model import (
    M1_ANSWER_DYNAMIC_ABLATIONS,
    M1_STATIC_CONTROLS,
    TASK_DEPRESSION,
    TASK_PTSD,
    MeterE2EHead,
    measurement_loss,
    trainable_parameter_summary,
)


SCHEMA = "meter_mh_e2e_v1"
HIDDEN_SIZE = 3584
VISUAL_SIZE = 1280
TASKS = {
    "daic_depression": {
        "dataset": "daic",
        "target": "depression",
        "task_id": TASK_DEPRESSION,
        "binary_profile": "depression_phq8_screening",
        "score_profile": "depression_phq_score",
    },
    "edaic_depression": {
        "dataset": "edaic",
        "target": "depression",
        "task_id": TASK_DEPRESSION,
        "binary_profile": "depression_phq8_screening",
        "score_profile": "depression_phq_score",
    },
    "edaic_ptsd": {
        "dataset": "edaic",
        "target": "ptsd",
        "task_id": TASK_PTSD,
        "binary_profile": "ptsd_dsm_iv_binary",
        "score_profile": "ptsd_score_repaired",
    },
}
ABLATIONS = (
    "no_answer_dynamics",
    "no_pim",
    "no_ensr",
    "no_visual",
    "no_teacher",
    "frozen_backbone",
)
MATCHED_CONTROLS = ("matched_ta", "matched_tav")
M1_CONTROLS = ("m1_static_continue", "m1_answer_dynamic")
SINGLE_ABLATIONS = ("none", "frozen_backbone") + ABLATIONS + MATCHED_CONTROLS + M1_CONTROLS
EXTERNAL_TAV = {
    "daic_depression": {"weighted_f1": 0.6963, "std": 0.0224, "n": 47},
    "edaic_depression": {"weighted_f1": 0.5960, "std": 0.0120, "n": 56},
    "edaic_ptsd": {"weighted_f1": 0.5000, "std": 0.0100, "n": 56},
}


def configure_runtime(threads: int) -> None:
    threads = max(1, min(32, int(threads)))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(min(4, threads))
    except RuntimeError:
        pass


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).split())


def normalize_pid(value: Any) -> str:
    return str(int(float(str(value).strip())))


def metric_block(labels: Iterable[int], probabilities: Iterable[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=np.int64)
    probability = np.clip(np.asarray(list(probabilities), dtype=np.float64), 1e-6, 1.0 - 1e-6)
    prediction = (probability >= float(threshold)).astype(np.int64)
    result = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
        "average_precision": float(average_precision_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).astype(int).tolist(),
    }
    result["auroc"] = float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None
    return result


def choose_threshold(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y = list(labels)
    probability = list(probabilities)
    candidates = np.linspace(0.10, 0.90, 161)
    return max(
        (
            metric_block(y, probability, float(value))["weighted_f1"],
            metric_block(y, probability, float(value))["macro_f1"],
            -abs(float(value) - 0.5),
            float(value),
        )
        for value in candidates
    )[-1]


def severity_metrics(rows: list[dict[str, Any]], predicted: list[float]) -> dict[str, Any]:
    pairs = [(row.get("score"), value) for row, value in zip(rows, predicted) if row.get("score") is not None]
    if len(pairs) < 2:
        return {"n": len(pairs), "mae": None, "rmse": None}
    gold = np.asarray([item[0] for item in pairs], dtype=np.float64)
    value = np.asarray([item[1] for item in pairs], dtype=np.float64)
    return {
        "n": int(len(gold)),
        "mae": float(mean_absolute_error(gold, value)),
        "rmse": float(math.sqrt(mean_squared_error(gold, value))),
    }


def protocol_audit(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "meter_mh_e2e_protocol_v1":
        raise ValueError(f"unexpected protocol schema: {protocol.get('schema')}")
    protocol_event_batch_size = int(protocol.get("optimization", {}).get("event_micro_batch", 1))
    if protocol_event_batch_size != args.event_batch_size:
        raise ValueError(
            "protocol event_micro_batch mismatch: "
            f"protocol={protocol_event_batch_size} requested={args.event_batch_size}"
        )
    splits = protocol.get("splits", {})
    if splits.get("daic") != "official train/dev/test = 107/35/47 participants":
        raise ValueError("protocol must declare official DAIC train/dev/test = 107/35/47")
    expected_edaic = "official train/dev/test = 163/56/56 participants; canonical split must match as an audit invariant"
    if splits.get("edaic") != expected_edaic:
        raise ValueError("protocol must declare official E-DAIC train/dev/test = 163/56/56")
    if splits.get("edaic_split_field") != "official split":
        raise ValueError("protocol must use the official E-DAIC split field")
    if splits.get("all_answer_events") != "every answer event inherits its participant official split; no event sampling or cross-split reuse":
        raise ValueError("protocol must enforce participant-level split inheritance for all answer events")
    if protocol.get("development_stage", {}).get("name") == "M1":
        optimization = protocol.get("optimization", {})
        locked_runtime = {
            "fixed_epochs": (int(args.epochs), int),
            "gradient_accumulation_participants": (
                int(args.gradient_accumulation_participants),
                int,
            ),
            "max_events_per_participant": (int(args.max_events_per_participant), int),
            "max_seconds": (float(args.max_seconds), float),
            "pad_seconds": (float(args.pad), float),
            "learning_rate": (float(args.learning_rate), float),
            "weight_decay": (float(args.weight_decay), float),
        }
        for field, (requested, cast) in locked_runtime.items():
            locked = cast(optimization.get(field))
            if locked != requested:
                raise ValueError(
                    f"M1 protocol runtime mismatch {field}: "
                    f"protocol={locked} requested={requested}"
                )
        locked_checkpointing = bool(optimization.get("gradient_checkpointing"))
        if locked_checkpointing != bool(args.gradient_checkpointing):
            raise ValueError(
                "M1 protocol gradient_checkpointing mismatch: "
                f"protocol={locked_checkpointing} requested={args.gradient_checkpointing}"
            )
    validate_contract(args.contract_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("METER-MH-E2E requires CUDA")
    return {
        "schema": SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "prompt_contract": prompts.CONTRACT,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(0),
        "raw_inputs_read_only": True,
        "cached_hidden_inference_forbidden": True,
        "vectorize_event_views": bool(args.vectorize_event_views),
        "vectorized_views": ["clinical", "answer"] if args.vectorize_event_views else [],
        "event_batch_size": int(args.event_batch_size),
        "event_batching": "length_bucketed_same_view" if args.event_batch_size > 1 else "disabled",
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "max_gpu_memory": str(args.max_gpu_memory),
    }


def load_score_map(contract_dir: Path, spec: dict[str, Any]) -> dict[tuple[str, str], float | None]:
    frame, profile = load_label_profile(
        contract_dir, spec["dataset"], spec["score_profile"], include_unlabeled=True
    )
    result: dict[tuple[str, str], float | None] = {}
    for row in frame.to_dict("records"):
        value = row.get(profile.target_column)
        mask = row.get(profile.mask_column)
        key = (normalize_pid(row["pid"]), str(row["split"]))
        try:
            result[key] = float(value) if int(mask) == 1 and pd.notna(value) else None
        except (TypeError, ValueError):
            result[key] = None
    return result


def load_task_rows(
    contract_dir: Path,
    task_name: str,
    allowed_splits: set[str],
) -> list[dict[str, Any]]:
    spec = TASKS[task_name]
    frame, profile = load_label_profile(
        contract_dir, spec["dataset"], spec["binary_profile"], include_unlabeled=False
    )
    scores = load_score_map(contract_dir, spec)
    rows = []
    for row in frame.to_dict("records"):
        # Both releases use the official participant split as the protocol
        # boundary.  E-DAIC's canonical view is an audit field, never a
        # replacement split; mismatches indicate a contract violation.
        source_split = str(row["split"])
        if spec["dataset"] == "edaic":
            canonical_split = str(row.get("canonical_split", ""))
            if canonical_split and canonical_split != source_split:
                raise RuntimeError(
                    f"E-DAIC canonical split mismatch pid={row['pid']}: "
                    f"official={source_split} canonical={canonical_split}"
                )
        protocol_split = source_split
        if protocol_split not in allowed_splits:
            continue
        pid = normalize_pid(row["pid"])
        rows.append(
            {
                "key": f"{task_name}:{pid}",
                "group": pid,
                "pid": pid,
                "task": task_name,
                "task_id": int(spec["task_id"]),
                "dataset": spec["dataset"],
                "target": spec["target"],
                "split": protocol_split,
                "source_split": str(row["split"]),
                "label": int(row[profile.target_column]),
                "score": scores.get((pid, str(row["split"]))),
            }
        )
    return rows


def load_rows(contract_dir: Path, allowed_splits: set[str]) -> list[dict[str, Any]]:
    rows = [row for task in TASKS for row in load_task_rows(contract_dir, task, allowed_splits)]
    rows.sort(key=lambda row: (row["task"], int(row["pid"])))
    if not rows:
        raise RuntimeError(f"no rows for splits {sorted(allowed_splits)}")
    return rows


def load_teacher_map(root: Path, contract_dir: Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Rebuild strict METER OOF teachers from contract-ordered arrays.

    The frozen runner stored OOF arrays without PIDs.  Their row order is the
    validated label-contract train order, so the mapping is reconstructed only
    after exact task and length checks.  Five seed arrays are averaged.
    """

    result: dict[str, float] = {}
    audit: dict[str, Any] = {}
    for task_name, spec in TASKS.items():
        ordered_rows = load_task_rows(contract_dir, task_name, {"train"})
        directory = root / "work" / task_name / spec["dataset"] / spec["target"]
        seed_arrays = []
        seed_files = []
        for seed in (7, 17, 27, 41, 53):
            path = directory / f"seed_{seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = np.asarray(payload["oof_predictions"]["meter_pim"], dtype=np.float64)
            if len(values) != len(ordered_rows):
                raise RuntimeError(
                    f"teacher length mismatch {task_name} seed={seed}: "
                    f"{len(values)} != {len(ordered_rows)}"
                )
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite teacher probabilities: {path}")
            seed_arrays.append(values)
            seed_files.append(str(path))
        mean_probability = np.stack(seed_arrays).mean(axis=0)
        for row, probability in zip(ordered_rows, mean_probability):
            result[row["key"]] = float(np.clip(probability, 1e-5, 1.0 - 1e-5))
        audit[task_name] = {
            "rows": len(ordered_rows),
            "seeds": [7, 17, 27, 41, 53],
            "variant": "meter_pim",
            "files": seed_files,
            "order": "clinical_label_contract_v1 official/canonical train order",
        }
    return result, audit


def assign_teacher(rows: list[dict[str, Any]], teacher: dict[str, float], minimum_coverage: float) -> dict[str, Any]:
    covered = 0
    by_task: dict[str, dict[str, int]] = {task: {"rows": 0, "covered": 0} for task in TASKS}
    for row in rows:
        row["teacher_probability"] = teacher.get(row["key"])
        by_task[row["task"]]["rows"] += 1
        if row["teacher_probability"] is not None:
            covered += 1
            by_task[row["task"]]["covered"] += 1
    coverage = covered / max(len(rows), 1)
    audit = {"rows": len(rows), "covered": covered, "coverage": coverage, "by_task": by_task}
    if coverage < float(minimum_coverage):
        raise RuntimeError(
            f"training-only OOF teacher coverage {coverage:.3f} is below {minimum_coverage:.3f}: {audit}"
        )
    return audit


def dataset_root(args: argparse.Namespace, dataset: str) -> Path:
    return args.daic_root if dataset == "daic" else args.edaic_root


def meter_visual_event_features(
    dataset: str,
    root: Path,
    pid: str,
    turns: list[dict[str, Any]],
    max_dims: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the fixed visual layout, preserving release-native missingness.

    E-DAIC participant 657 is missing the released CNN_VGG file while retaining
    the OpenFace and CNN_ResNet sources.  The participant must not be dropped,
    and an audio feature must never be substituted.  The absent source keeps
    its fixed 320-dimensional slot as zeros and is recorded in provenance.
    """
    try:
        return cache_engine.base.visual_event_features(dataset, root, pid, turns, max_dims=max_dims)
    except FileNotFoundError:
        if dataset != "edaic":
            raise
        required = ("openface", "cnn_resnet", "cnn_vgg")
        available = visual_modality_contract.visual_paths(dataset, root, pid)
        missing = [name for name in required if name not in available]
        if missing != ["cnn_vgg"]:
            raise
        selected = {name: available[name] for name in required if name in available}
        visual_audit = visual_modality_contract.validate_visual_paths(
            dataset, root, pid, selected
        )
        reference_times = None
        tables: list[tuple[np.ndarray, np.ndarray] | None] = []
        for name in required:
            path = available.get(name)
            if path is None:
                tables.append(None)
                continue
            if path.suffix.lower() == ".mat":
                table = cache_engine.base.read_mat_visual_table(path, max_dims, reference_times)
            else:
                table = cache_engine.base.read_visual_table(path, max_dims, reference_times)
            if reference_times is None and len(table[0]):
                reference_times = table[0]
            tables.append(table)

        per_turn = []
        for turn in turns:
            chunks = []
            for table in tables:
                if table is None:
                    chunks.append(np.zeros(max_dims * 5, dtype=np.float32))
                    continue
                times, matrix = table
                mask = (times >= float(turn["start"])) & (times <= float(turn["stop"]))
                block = matrix[mask]
                if len(block) == 0:
                    chunks.append(np.zeros(max_dims * 5, dtype=np.float32))
                    continue
                position = np.linspace(-1.0, 1.0, len(block), dtype=np.float32)
                denominator = max(float(np.square(position).sum()), 1e-6)
                slope = (position[:, None] * block).sum(axis=0) / denominator
                chunks.append(
                    np.concatenate(
                        (
                            block.mean(0),
                            block.std(0),
                            np.quantile(block, 0.25, axis=0),
                            np.quantile(block, 0.75, axis=0),
                            slope,
                        )
                    ).astype(np.float32)
                )
            per_turn.append(np.concatenate(chunks))
        value = cache_engine.base.finite(
            np.stack(per_turn)
            if per_turn
            else np.zeros((0, max_dims * 5 * len(required)), dtype=np.float32)
        )
        return value, {
            **visual_audit,
            "required_source_order": list(required),
            "missing_sources": missing,
            "missing_source_policy": "fixed-slot zero fill; no cross-modal substitution",
            "source_frame_rows": int(
                sum(table is not None and len(table[0]) > 0 for table in tables)
            ),
            "feature_dim": int(value.shape[1]) if value.ndim == 2 else 0,
            "event_rows_with_any_visual": int(
                sum(np.linalg.norm(row) > 0 for row in value)
            ),
            "mat_alignment": "openface_timestamp_support",
        }


def materialize(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    root = dataset_root(args, row["dataset"])
    audio = cache_engine.base.source_file(row["dataset"], root, row["pid"], "AUDIO.wav")
    turns = cache_engine.base.read_turns(row["dataset"], root, row["pid"])
    if not audio.is_file() or not turns:
        raise RuntimeError(f"missing raw participant inputs: {row['key']} audio={audio} turns={len(turns)}")
    if args.max_events_per_participant > 0:
        raise RuntimeError("max-events-per-participant must remain zero in the strict full protocol")
    clips = []
    clip_root = getattr(args, "clip_root", args.output / "clips")
    for turn in turns:
        digest = hashlib.sha256(
            f"{row['dataset']}|{row['pid']}|{turn['index']}|{turn['start']}|{turn['stop']}".encode()
        ).hexdigest()[:20]
        clips.append(
            cache_engine.base.write_clip(
                audio,
                turn["start"],
                turn["stop"],
                clip_root / row["dataset"] / f"{row['pid']}_{digest}.wav",
                args.pad,
                args.max_seconds,
            )
        )
    visual, visual_audit = meter_visual_event_features(
        row["dataset"], root, row["pid"], turns
    )
    visual = np.asarray(visual, dtype=np.float32)
    if visual.shape[1] > VISUAL_SIZE:
        raise RuntimeError(f"visual dimension exceeds frozen contract: {row['key']} {visual.shape}")
    if visual.shape[1] < VISUAL_SIZE:
        visual = np.pad(visual, ((0, 0), (0, VISUAL_SIZE - visual.shape[1])))
    timing = []
    for index, turn in enumerate(turns):
        duration = max(0.0, float(turn["stop"]) - float(turn["start"]))
        timing.append(
            [
                min(max(float(turn.get("latency", 0.0)), 0.0), 30.0) / 30.0,
                min(duration, 120.0) / 120.0,
                float(bool(turn.get("question_available", False))),
                float(index > 0),
                index / max(len(turns) - 1, 1),
            ]
        )
    return {
        **row,
        "audio": audio,
        "turns": turns,
        "clips": clips,
        "visual": torch.from_numpy(visual),
        "timing": torch.tensor(timing, dtype=torch.float32),
        "visual_audit": visual_audit,
    }


def event_messages(participant: dict[str, Any], index: int, view: str) -> list[dict[str, Any]]:
    turn = participant["turns"][index]
    history = cache_engine.history_text(participant["turns"], index)
    return prompts.build_messages(
        view=view,
        question=clean(turn.get("question")),
        history=history,
        answer=clean(turn.get("answer")),
        question_available=bool(turn.get("question_available", False)),
        history_available=bool(history),
        audio_path=str(participant["clips"][index]) if view != "context" else None,
        video_payload=None,
        start_seconds=float(turn["start"]),
        stop_seconds=float(turn["stop"]),
    )


def online_encode(
    processor: Any,
    backbone: torch.nn.Module,
    messages: list[dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    audios, images, videos = cache_engine.base.process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=[rendered],
        audio=audios or None,
        images=images or None,
        videos=videos or None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
        audio_kwargs={"max_length": cache_engine.base.AUDIO_MAX_SAMPLES},
    )
    inputs = cache_engine.base.move_inputs(inputs, device)
    output = backbone(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = output.hidden_states[-4].float()
    for layer in output.hidden_states[-3:]:
        hidden = hidden + layer.float()
    hidden = hidden * 0.25
    mask = inputs.get("attention_mask")
    if mask is None:
        pooled = hidden.mean(dim=1)
    else:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return pooled[0]


def online_encode_batch(
    processor: Any,
    backbone: torch.nn.Module,
    messages_batch: list[list[dict[str, Any]]],
    device: torch.device,
) -> torch.Tensor:
    """Vectorize independent event encodes without changing their semantics."""
    if not messages_batch:
        raise ValueError("messages_batch must not be empty")
    if len(messages_batch) == 1:
        return online_encode(processor, backbone, messages_batch[0], device).unsqueeze(0)

    rendered = []
    audios: list[Any] = []
    images: list[Any] = []
    videos: list[Any] = []
    for messages in messages_batch:
        rendered.append(
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        )
        sample_audio, sample_images, sample_videos = cache_engine.base.process_mm_info(
            messages, use_audio_in_video=False
        )
        audios.extend(sample_audio or [])
        images.extend(sample_images or [])
        videos.extend(sample_videos or [])

    inputs = processor(
        text=rendered,
        audio=audios or None,
        images=images or None,
        videos=videos or None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
        audio_kwargs={"max_length": cache_engine.base.AUDIO_MAX_SAMPLES},
    )
    inputs = cache_engine.base.move_inputs(inputs, device)
    output = backbone(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = output.hidden_states[-4].float()
    for layer in output.hidden_states[-3:]:
        hidden = hidden + layer.float()
    hidden = hidden * 0.25
    mask = inputs.get("attention_mask")
    if mask is None:
        return hidden.mean(dim=1)
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def event_batch_cost(participant: dict[str, Any], index: int, view: str) -> float:
    """Estimate padding cost without tokenizing or reading an audio clip twice."""
    turn = participant["turns"][index]
    history = cache_engine.history_text(participant["turns"], index)
    text_chars = len(clean(turn.get("question"))) + len(clean(turn.get("answer"))) + len(history)
    if view == "answer":
        text_chars = len(clean(turn.get("answer")))
    elif view == "context":
        text_chars = len(clean(turn.get("question"))) + len(history)
    duration = 0.0
    if view != "context":
        duration = min(
            max(0.0, float(turn["stop"]) - float(turn["start"])) + 2.0 * 0.4,
            12.0,
        )
    # Audio tokens dominate short answers; character count separates long
    # contextual prompts within the same deterministic bucket.
    return duration * 512.0 + float(text_chars)


def event_batch_plan(
    participant: dict[str, Any],
    view: str,
    batch_size: int,
) -> list[list[int]]:
    """Build deterministic, length-bucketed event batches and preserve coverage."""
    count = len(participant["turns"])
    if batch_size < 1:
        raise ValueError(f"event batch size must be positive, got {batch_size}")
    if batch_size == 1:
        return [[index] for index in range(count)]
    ordered = sorted(
        range(count),
        key=lambda index: (event_batch_cost(participant, index, view), index),
    )
    return [ordered[start : start + batch_size] for start in range(0, count, batch_size)]


def batched_participant_views(
    processor: Any,
    backbone: torch.nn.Module,
    participant: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode same-view events in length buckets, then restore interview order."""
    count = len(participant["turns"])
    views: list[torch.Tensor] = []
    for view in ("clinical", "answer", "context"):
        ordered: list[torch.Tensor | None] = [None] * count
        for indices in event_batch_plan(participant, view, batch_size):
            encoded = online_encode_batch(
                processor,
                backbone,
                [event_messages(participant, index, view) for index in indices],
                device,
            )
            for offset, index in enumerate(indices):
                ordered[index] = encoded[offset]
        if any(value is None for value in ordered):
            raise RuntimeError(f"incomplete event batch coverage for view={view}")
        views.append(torch.stack([value for value in ordered if value is not None]))
    return tuple(views)  # type: ignore[return-value]


def batched_event_backward(
    processor: Any,
    backbone: torch.nn.Module,
    participant: dict[str, Any],
    representation_gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> None:
    """Inject participant-level representation gradients in same-view batches."""
    for view_index, view in enumerate(("clinical", "answer", "context")):
        for indices in event_batch_plan(participant, view, batch_size):
            live = online_encode_batch(
                processor,
                backbone,
                [event_messages(participant, index, view) for index in indices],
                device,
            )
            upstream = torch.stack([representation_gradients[view_index][index] for index in indices])
            torch.autograd.backward(live, upstream)
            del live, upstream


def encode_event_views(
    processor: Any,
    backbone: torch.nn.Module,
    participant: dict[str, Any],
    index: int,
    device: torch.device,
    vectorize_views: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if vectorize_views:
        audio_views = online_encode_batch(
            processor,
            backbone,
            [event_messages(participant, index, view) for view in ("clinical", "answer")],
            device,
        )
        context = online_encode(
            processor,
            backbone,
            event_messages(participant, index, "context"),
            device,
        )
        return audio_views[0], audio_views[1], context
    return tuple(
        online_encode(processor, backbone, event_messages(participant, index, view), device)
        for view in ("clinical", "answer", "context")
    )  # type: ignore[return-value]


def frozen_participant_views(
    processor: Any,
    backbone: torch.nn.Module,
    participant: dict[str, Any],
    device: torch.device,
    vectorize_views: bool = False,
    event_batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if event_batch_size > 1:
        with torch.no_grad():
            return batched_participant_views(
                processor,
                backbone,
                participant,
                device,
                event_batch_size,
            )
    values = {name: [] for name in ("clinical", "answer", "context")}
    with torch.no_grad():
        for index in range(len(participant["turns"])):
            event = encode_event_views(
                processor,
                backbone,
                participant,
                index,
                device,
                vectorize_views=vectorize_views,
            )
            for name, value in zip(values, event):
                values[name].append(value.detach())
    return tuple(torch.stack(values[name]) for name in values)  # type: ignore[return-value]


def lora_target_names(backbone: torch.nn.Module) -> list[str]:
    # HumanOmniV2 also has audio_tower.layers.24-27.  The frozen protocol
    # permits LoRA only in the text/LLM trunk, whose path ends in
    # model.layers.<n>.self_attn.<projection>.
    pattern = re.compile(
        r"(?:^|\.)model\.layers\.(24|25|26|27)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
    )
    result = [name for name, module in backbone.named_modules() if pattern.search(name) and isinstance(module, torch.nn.Linear)]
    if len(result) != 16:
        raise RuntimeError(f"expected exactly 16 text-trunk attention projections, found {len(result)}: {result}")
    return sorted(result)


def load_backbone(args: argparse.Namespace, train_lora: bool) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    memory_match = re.fullmatch(r"(\d+(?:\.\d+)?)GiB", str(args.max_gpu_memory))
    if memory_match:
        requested_bytes = float(memory_match.group(1)) * (2**30)
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(0.98, requested_bytes / total_bytes),
            device=0,
        )
    processor = Qwen2_5OmniProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    backbone = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        max_memory={0: args.max_gpu_memory, "cpu": "400GiB"},
    )
    backbone.config.use_cache = False
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    audit: dict[str, Any] = {"lora": False, "targets": []}
    if train_lora:
        from peft import LoraConfig, get_peft_model

        targets = lora_target_names(backbone)
        if args.gradient_checkpointing:
            backbone.gradient_checkpointing_enable()
            backbone.enable_input_require_grads()
        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            ),
        )
        backbone.eval()
        audit = {
            "lora": True,
            "targets": targets,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "parameters": trainable_parameter_summary(backbone),
        }
    return processor, backbone, audit


def adapter_state(backbone: torch.nn.Module, train_lora: bool) -> dict[str, torch.Tensor] | None:
    if not train_lora:
        return None
    from peft import get_peft_model_state_dict

    return {name: value.detach().cpu() for name, value in get_peft_model_state_dict(backbone).items()}


def restore_adapter(backbone: torch.nn.Module, state: dict[str, torch.Tensor] | None) -> None:
    if state is None:
        return
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(backbone, state)


def set_backbone_training_mode(backbone: torch.nn.Module, train: bool) -> None:
    """Enable checkpointed backprop while keeping stochastic dropout off."""
    backbone.train(bool(train))
    if train:
        for module in backbone.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()


def _gate0_compute(args: argparse.Namespace) -> dict[str, Any]:
    output = gate0_result_path(args)
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    rows = load_rows(args.contract_dir, {"train"})
    sample = []
    seen = set()
    for row in rows:
        if row["dataset"] in seen:
            continue
        seen.add(row["dataset"])
        sample.append(row)
    processor, backbone, model_audit = load_backbone(args, train_lora=False)
    device = torch.device("cuda:0")
    checks = []
    for row in sample:
        participant = materialize(row, args)
        cache_path = args.meter_cache / row["dataset"] / "cache" / f"{row['pid']}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        with np.load(cache_path, allow_pickle=False) as archive:
            cached = {view: np.asarray(archive[view], dtype=np.float32) for view in ("clinical", "answer", "context")}
        batched = None
        if args.event_batch_size > 1:
            with torch.no_grad():
                batched = frozen_participant_views(
                    processor,
                    backbone,
                    participant,
                    device,
                    event_batch_size=args.event_batch_size,
                )
        for index in range(min(2, len(participant["turns"]))):
            if batched is None:
                with torch.no_grad():
                    online = encode_event_views(
                        processor,
                        backbone,
                        participant,
                        index,
                        device,
                        vectorize_views=args.vectorize_event_views,
                    )
            else:
                online = tuple(view[index] for view in batched)
            for view, value in zip(("clinical", "answer", "context"), online):
                expected = cached[view][index]
                actual = value.detach().cpu().numpy()
                cosine = float(np.dot(actual, expected) / max(np.linalg.norm(actual) * np.linalg.norm(expected), 1e-12))
                relative_l2 = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12))
                checks.append(
                    {
                        "key": row["key"],
                        "event": index,
                        "view": view,
                        "cosine": cosine,
                        "relative_l2": relative_l2,
                        "event_batch_size": args.event_batch_size,
                    }
                )
    passed = all(item["cosine"] >= 0.999 and item["relative_l2"] <= 0.02 for item in checks)
    result = {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "event_batch_size": args.event_batch_size,
        "checks": checks,
        "model": model_audit,
    }
    atomic_json(output, result)
    del backbone, processor
    gc.collect()
    torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError(f"Gate-0 frozen online equivalence failed: {checks}")
    return result


def gate0_result_path(args: argparse.Namespace) -> Path:
    if args.event_batch_size > 1:
        name = f"EVENT_BATCH_{args.event_batch_size}_RESULT.json"
    else:
        name = "VECTORIZED_RESULT.json" if args.vectorize_event_views else "RESULT.json"
    return args.output / "gate0" / name


def load_gate0_result(args: argparse.Namespace) -> dict[str, Any]:
    """Load either the local audit or an explicitly imported full audit.

    The imported form is used only by the official-split orchestrator after
    the independent DAIC and E-DAIC staged audits have both passed.  It keeps
    the next online experiment from silently falling back to the old two-event
    smoke Gate-0.
    """
    path = args.external_gate0_result or gate0_result_path(args)
    if not path.is_file():
        if args.external_gate0_result:
            raise FileNotFoundError(path)
        return gate0(args)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "passed":
        raise RuntimeError(f"Gate-0 has not passed: {path}")
    return result


def gate0(args: argparse.Namespace) -> dict[str, Any]:
    output = gate0_result_path(args)
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(output.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        import fcntl

        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        return _gate0_compute(args)


def fold_assignments(rows: list[dict[str, Any]], seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    strata = np.asarray([f"{row['task']}:{row['label']}" for row in rows])
    groups = np.asarray([row["group"] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(rows)), strata, groups))


def positive_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = {}
    for task in TASKS:
        labels = [row["label"] for row in rows if row["task"] == task]
        positives = sum(labels)
        result[task] = (len(labels) - positives) / max(positives, 1)
    return result


def training_seed_name(ablation: str) -> str:
    """Pair M1 and its shuffle control with the exact M0-TAV initialization."""
    if ablation in M1_ANSWER_DYNAMIC_ABLATIONS or ablation in M1_STATIC_CONTROLS:
        return "matched_tav"
    return ablation


def m1_base_state_path(args: argparse.Namespace, fold: int) -> Path:
    if args.m1_base_root is None:
        raise RuntimeError("M1 continuation requires --m1-base-root")
    return args.m1_base_root / "oof" / "matched_tav" / f"fold_{fold}" / "FINAL_STATE.pt"


def m1_base_audit_path(args: argparse.Namespace, fold: int) -> Path:
    if args.m1_base_root is None:
        raise RuntimeError("M1 continuation requires --m1-base-root")
    return args.m1_base_root / "oof" / "matched_tav" / f"fold_{fold}" / "REPLAY_AUDIT.json"


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise RuntimeError(f"checkpoint RNG state is incomplete: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def temporal_permutation(
    row: dict[str, Any],
    count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if count <= 1:
        return torch.arange(count, device=device)
    digest = hashlib.sha256(f"{seed}:{row['key']}:m1_time_shuffle".encode("utf-8")).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
    permutation = torch.randperm(count, generator=generator)
    if torch.equal(permutation, torch.arange(count)):
        permutation = torch.roll(permutation, shifts=1)
    return permutation.to(device)


@torch.no_grad()
def audit_m1_zero_impact_initialization(
    processor: Any,
    backbone: torch.nn.Module,
    head: MeterE2EHead,
    row: dict[str, Any],
    args: argparse.Namespace,
    ablation: str,
) -> dict[str, Any]:
    if ablation not in M1_ANSWER_DYNAMIC_ABLATIONS:
        return {"status": "not_applicable"}
    device = torch.device("cuda:0")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    original_training = head.training
    try:
        participant = materialize(row, args)
        views = frozen_participant_views(
            processor,
            backbone,
            participant,
            device,
            vectorize_views=args.vectorize_event_views,
            event_batch_size=args.event_batch_size,
        )
        visual = participant["visual"].to(device)
        timing = participant["timing"].to(device)
        permutation = (
            temporal_permutation(row, len(participant["turns"]), args.seed, device)
            if ablation == "m1_time_shuffle"
            else None
        )
        fields = ("event_states", "logit", "severity_unit", "log_variance")
        mode_results = {}
        for training in (False, True):
            head.train(training)
            mode_cpu_state = torch.get_rng_state()
            mode_cuda_state = torch.cuda.get_rng_state(device)
            anchor = head(*views, visual, timing, row["task_id"], ablation="matched_tav")
            torch.set_rng_state(mode_cpu_state)
            torch.cuda.set_rng_state(mode_cuda_state, device)
            candidate = head(
                *views,
                visual,
                timing,
                row["task_id"],
                ablation=ablation,
                temporal_permutation=permutation,
            )
            maximum_absolute_difference = {
                field: float((candidate[field] - anchor[field]).abs().max().cpu())
                for field in fields
            }
            exact = {field: bool(torch.equal(candidate[field], anchor[field])) for field in fields}
            mode_results["train" if training else "eval"] = {
                "exact": exact,
                "maximum_absolute_difference": maximum_absolute_difference,
            }
            del anchor, candidate
        passed = (
            all(all(values["exact"].values()) for values in mode_results.values())
            and float(head.dynamic_adapter[-1].weight.detach().abs().max().cpu()) == 0.0
            and float(head.dynamic_adapter[-1].bias.detach().abs().max().cpu()) == 0.0
        )
        result = {
            "status": "passed" if passed else "failed",
            "row_key": row["key"],
            "events": len(participant["turns"]),
            "answer_dynamic_scale": float(head.answer_dynamic_scale.detach().cpu()),
            "dynamic_output_projection_zero": True,
            "modes": mode_results,
            "rng_state_restored": True,
        }
        del participant, views, visual, timing
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)
        head.train(original_training)
    if not passed:
        raise RuntimeError(f"M1 zero-impact initialization audit failed: {result}")
    return result


def participant_backward(
    processor: Any,
    backbone: torch.nn.Module,
    head: MeterE2EHead,
    row: dict[str, Any],
    args: argparse.Namespace,
    positive_weight: float,
    ablation: str,
    loss_scale: float,
    train_backbone: bool,
) -> dict[str, float]:
    device = torch.device("cuda:0")
    participant = materialize(row, args)
    frozen = frozen_participant_views(
        processor,
        backbone,
        participant,
        device,
        vectorize_views=args.vectorize_event_views,
        event_batch_size=args.event_batch_size,
    )
    representation_leaves = tuple(value.detach().requires_grad_(True) for value in frozen)
    visual = participant["visual"].to(device)
    timing = participant["timing"].to(device)
    count = len(participant["turns"])
    effective = "none" if ablation == "frozen_backbone" else ablation
    permutation = (
        temporal_permutation(row, count, args.seed, device)
        if effective == "m1_time_shuffle"
        else None
    )
    output = head(
        representation_leaves[0],
        representation_leaves[1],
        representation_leaves[2],
        visual,
        timing,
        row["task_id"],
        ablation=effective,
        temporal_permutation=permutation,
    )
    loss, values = measurement_loss(
        head,
        output,
        label=row["label"],
        task_id=row["task_id"],
        score=row.get("score"),
        teacher_probability=row.get("teacher_probability"),
        timing_target=timing,
        timing_mask=torch.ones(count, dtype=torch.bool, device=device),
        positive_weight=positive_weight,
        random_seed=args.seed,
        ablation=effective,
    )
    # One participant-level backward computes the head gradient exactly and
    # materializes dL/d(event representation) for every view and event.
    (loss * float(loss_scale)).backward()
    representation_gradients = tuple(
        value.grad.detach().clone() if value.grad is not None else torch.zeros_like(value)
        for value in representation_leaves
    )

    # Rematerialize each online HumanOmniV2 event and inject its exact upstream
    # gradient.  No cached hidden state is an inference input or a trainable
    # substitute for the online backbone.
    if train_backbone:
        if args.event_batch_size > 1:
            batched_event_backward(
                processor,
                backbone,
                participant,
                representation_gradients,
                device,
                args.event_batch_size,
            )
        else:
            for index in range(count):
                live = encode_event_views(
                    processor,
                    backbone,
                    participant,
                    index,
                    device,
                    vectorize_views=args.vectorize_event_views,
                )
                torch.autograd.backward(
                    live,
                    tuple(gradient[index] for gradient in representation_gradients),
                )
                del live
    del frozen, representation_leaves, representation_gradients, participant, visual, timing, output, loss
    return values


@torch.no_grad()
def evaluate_rows(
    processor: Any,
    backbone: torch.nn.Module,
    head: MeterE2EHead,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    ablation: str,
) -> list[dict[str, Any]]:
    device = torch.device("cuda:0")
    head.eval()
    predictions = []
    for number, row in enumerate(rows, 1):
        participant = materialize(row, args)
        views = frozen_participant_views(
            processor,
            backbone,
            participant,
            device,
            vectorize_views=args.vectorize_event_views,
            event_batch_size=args.event_batch_size,
        )
        visual = participant["visual"].to(device)
        timing = participant["timing"].to(device)
        effective = "none" if ablation == "frozen_backbone" else ablation
        permutation = (
            temporal_permutation(row, len(participant["turns"]), args.seed, device)
            if effective == "m1_time_shuffle"
            else None
        )
        output = head(
            *views,
            visual,
            timing,
            row["task_id"],
            ablation=effective,
            temporal_permutation=permutation,
        )
        predictions.append(
            {
                "key": row["key"],
                "pid": row["pid"],
                "task": row["task"],
                "dataset": row["dataset"],
                "label": row["label"],
                "score": row.get("score"),
                "probability": float(torch.sigmoid(output["logit"]).cpu()),
                "predicted_score": float(output["severity_unit"].cpu()) * (24.0 if row["task_id"] == TASK_DEPRESSION else 85.0),
                "uncertainty": float(output["log_variance"].exp().cpu()),
                "events": len(participant["turns"]),
            }
        )
        if number == 1 or number % 10 == 0 or number == len(rows):
            print(f"METER_E2E_EVAL ablation={ablation} {number}/{len(rows)}", flush=True)
        del participant, views, visual, timing, output
    return predictions


def train_fold(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    held_rows: list[dict[str, Any]],
    ablation: str,
    fold: int,
) -> dict[str, Any]:
    fold_root = args.output / "oof" / ablation / f"fold_{fold}"
    result_path = fold_root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    seed_token = training_seed_name(ablation)
    seed_all(args.seed + fold * 1009 + sum(map(ord, seed_token)))
    train_lora = ablation not in {
        "frozen_backbone",
        "matched_ta",
        "matched_tav",
        *M1_ANSWER_DYNAMIC_ABLATIONS,
        *M1_STATIC_CONTROLS,
    }
    processor, backbone, backbone_audit = load_backbone(args, train_lora=train_lora)
    device = torch.device("cuda:0")
    head = MeterE2EHead(hidden_size=HIDDEN_SIZE, visual_size=VISUAL_SIZE).to(device=device, dtype=torch.float32)
    checkpoint_path = fold_root / "checkpoint.pt"
    resuming = checkpoint_path.is_file()
    base_state_audit: dict[str, Any] = {"status": "not_applicable"}
    if (ablation in M1_ANSWER_DYNAMIC_ABLATIONS or ablation in M1_STATIC_CONTROLS) and not resuming:
        base_path = m1_base_state_path(args, fold)
        replay_audit_path = m1_base_audit_path(args, fold)
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        if not replay_audit_path.is_file():
            raise FileNotFoundError(replay_audit_path)
        replay_audit = json.loads(replay_audit_path.read_text(encoding="utf-8"))
        if replay_audit.get("status") != "passed":
            raise RuntimeError(f"M1 replay audit did not pass: {replay_audit_path}")
        if replay_audit.get("replay_state_sha256") != sha256_file(base_path):
            raise RuntimeError(f"M1 replay state hash mismatch: {base_path}")
        base_state = torch.load(base_path, map_location="cpu", weights_only=False)
        if base_state.get("schema") != SCHEMA or base_state.get("ablation") != "matched_tav":
            raise RuntimeError(f"invalid M1 base state: {base_path}")
        if int(base_state.get("fold", -1)) != fold:
            raise RuntimeError(f"M1 base fold mismatch: {base_state.get('fold')} != {fold}")
        incompatible = head.load_state_dict(base_state["head"], strict=False)
        allowed_missing = {"answer_dynamic_scale"}
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "M1 base head mismatch: "
                f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )
        base_state_audit = {
            "status": "loaded",
            "path": str(base_path.resolve()),
            "sha256": sha256_file(base_path),
            "source_ablation": base_state.get("ablation"),
            "source_fold": int(base_state.get("fold")),
            "source_protocol_sha256": base_state.get("protocol_sha256"),
            "replay_audit": str(replay_audit_path.resolve()),
            "replay_audit_sha256": sha256_file(replay_audit_path),
        }
    head.answer_dynamic_scale.requires_grad_(ablation in M1_ANSWER_DYNAMIC_ABLATIONS)
    if ablation in M1_ANSWER_DYNAMIC_ABLATIONS and not resuming:
        head.initialize_answer_dynamic_residual()
    if ablation in M1_ANSWER_DYNAMIC_ABLATIONS:
        for module in (head.severity_head, head.evidence_head, head.visual_reliability):
            module.requires_grad_(False)
    initialization_audit = (
        {"status": "pending_checkpoint_restore"}
        if resuming
        else audit_m1_zero_impact_initialization(
            processor,
            backbone,
            head,
            train_rows[0],
            args,
            ablation,
        )
    )
    parameters = [parameter for parameter in list(backbone.parameters()) + list(head.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = 0
    start_position = 0
    order: list[int] | None = None
    history: list[dict[str, Any]] = []
    resumed_epoch_losses: list[float] = []
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") == SCHEMA and checkpoint.get("ablation") == ablation:
            checkpoint_protocol = checkpoint.get("protocol_sha256")
            current_protocol = sha256_file(args.protocol)
            if checkpoint_protocol is not None and checkpoint_protocol != current_protocol:
                raise RuntimeError(
                    "checkpoint protocol mismatch: "
                    f"checkpoint={checkpoint_protocol} requested={current_protocol}"
                )
            if ablation in M1_ANSWER_DYNAMIC_ABLATIONS and checkpoint_protocol is None:
                raise RuntimeError("M1 checkpoint is missing its locked protocol signature")
            checkpoint_vectorization = bool(checkpoint.get("vectorize_event_views", False))
            if checkpoint_vectorization != args.vectorize_event_views:
                raise RuntimeError(
                    "checkpoint event-view vectorization mismatch: "
                    f"checkpoint={checkpoint_vectorization} requested={args.vectorize_event_views}"
                )
            checkpoint_event_batch_size = int(checkpoint.get("event_batch_size", 1))
            if checkpoint_event_batch_size != args.event_batch_size:
                raise RuntimeError(
                    "checkpoint event batch size mismatch: "
                    f"checkpoint={checkpoint_event_batch_size} requested={args.event_batch_size}"
                )
            restore_adapter(backbone, checkpoint.get("adapter"))
            incompatible = head.load_state_dict(checkpoint["head"], strict=False)
            allowed_missing = {"answer_dynamic_scale"} if ablation not in M1_ANSWER_DYNAMIC_ABLATIONS else set()
            if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "checkpoint head mismatch: "
                    f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
                )
            optimizer.load_state_dict(checkpoint["optimizer"])
            initialization_audit = checkpoint.get(
                "initialization_audit", {"status": "missing_from_checkpoint"}
            )
            base_state_audit = checkpoint.get(
                "base_state_audit", {"status": "missing_from_checkpoint"}
            )
            start_epoch = int(checkpoint["epoch"])
            start_position = int(checkpoint["position"])
            order = [int(value) for value in checkpoint["order"]]
            history = list(checkpoint.get("history", []))
            resumed_epoch_losses = [
                float(value) for value in checkpoint.get("active_epoch_losses", [])
            ]
            if ablation in M1_CONTROLS:
                restore_rng_state(checkpoint.get("rng_state", {}))
            if start_position == 0:
                if resumed_epoch_losses:
                    history.append(
                        {
                            "epoch": start_epoch,
                            "mean_loss": float(np.mean(resumed_epoch_losses)),
                        }
                    )
                    resumed_epoch_losses = []
                order = None
    weights = positive_weights(train_rows)
    optimizer.zero_grad(set_to_none=True)
    set_backbone_training_mode(backbone, train_lora)
    for epoch in range(start_epoch, args.epochs):
        if order is None or epoch != start_epoch:
            generator = random.Random(args.seed + fold * 1009 + epoch * 7919 + sum(map(ord, seed_token)))
            order = list(range(len(train_rows)))
            generator.shuffle(order)
            start_position = 0
        head.train()
        epoch_losses = list(resumed_epoch_losses) if epoch == start_epoch else []
        resumed_epoch_losses = []
        for position in range(start_position, len(order)):
            row = train_rows[order[position]]
            values = participant_backward(
                processor,
                backbone,
                head,
                row,
                args,
                weights[row["task"]],
                ablation,
                loss_scale=1.0 / args.gradient_accumulation_participants,
                train_backbone=train_lora,
            )
            epoch_losses.append(values["total"])
            boundary = (position + 1) % args.gradient_accumulation_participants == 0 or position + 1 == len(order)
            if boundary:
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if boundary and ((position + 1) % args.checkpoint_every == 0 or position + 1 == len(order)):
                atomic_torch(
                    checkpoint_path,
                    {
                        "schema": SCHEMA,
                        "ablation": ablation,
                        "fold": fold,
                        "vectorize_event_views": args.vectorize_event_views,
                        "event_batch_size": args.event_batch_size,
                        "protocol_sha256": sha256_file(args.protocol),
                        "epoch": epoch + (1 if position + 1 == len(order) else 0),
                        "position": 0 if position + 1 == len(order) else position + 1,
                        "order": order,
                        "adapter": adapter_state(backbone, train_lora),
                        "head": {name: value.detach().cpu() for name, value in head.state_dict().items()},
                        "optimizer": optimizer.state_dict(),
                        "initialization_audit": initialization_audit,
                        "base_state_audit": base_state_audit,
                        "history": history,
                        "active_epoch_losses": epoch_losses,
                        "rng_state": capture_rng_state(),
                    },
                )
            if position == start_position or (position + 1) % 5 == 0 or position + 1 == len(order):
                print(
                    f"METER_E2E_TRAIN ablation={ablation} fold={fold} epoch={epoch + 1}/{args.epochs} "
                    f"participant={position + 1}/{len(order)} loss={values['total']:.5f} "
                    f"gpu_gib={torch.cuda.memory_allocated() / 2**30:.2f}",
                    flush=True,
                )
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(epoch_losses))})
        start_position = 0
        order = None
    predictions = evaluate_rows(processor, backbone, head, held_rows, args, ablation)
    mechanism_predictions = {}
    if ablation == "m1_answer_dynamic":
        mechanism_predictions["time_shuffle"] = evaluate_rows(
            processor,
            backbone,
            head,
            held_rows,
            args,
            "m1_time_shuffle",
        )
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "ablation": ablation,
        "fold": fold,
        "vectorize_event_views": args.vectorize_event_views,
        "event_batch_size": args.event_batch_size,
        "protocol_sha256": sha256_file(args.protocol),
        "train_rows": len(train_rows),
        "held_rows": len(held_rows),
        "history": history,
        "backbone": backbone_audit,
        "head": trainable_parameter_summary(head),
        "initialization_audit": initialization_audit,
        "base_state_audit": base_state_audit,
        "predictions": predictions,
        "mechanism_predictions": mechanism_predictions,
    }
    atomic_torch(
        fold_root / "FINAL_STATE.pt",
        {
            "schema": SCHEMA,
            "ablation": ablation,
            "fold": fold,
            "protocol_sha256": sha256_file(args.protocol),
            "head": {name: value.detach().cpu() for name, value in head.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "initialization_audit": initialization_audit,
            "base_state_audit": base_state_audit,
        },
    )
    atomic_json(result_path, result)
    checkpoint_path.unlink(missing_ok=True)
    del optimizer, head, backbone, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result


def summarize_oof(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for task in TASKS:
        rows = [row for row in predictions if row["task"] == task]
        threshold = choose_threshold([row["label"] for row in rows], [row["probability"] for row in rows])
        summary[task] = {
            "threshold_locked_from_oof": threshold,
            "classification": metric_block(
                [row["label"] for row in rows], [row["probability"] for row in rows], threshold
            ),
            "severity": severity_metrics(rows, [row["predicted_score"] for row in rows]),
        }
    return summary


def run_oof(args: argparse.Namespace, rows: list[dict[str, Any]], ablation: str) -> dict[str, Any]:
    output = args.output / "oof" / ablation / "SUMMARY.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    folds = fold_assignments(rows, args.seed)
    predictions = []
    mechanism_predictions: dict[str, list[dict[str, Any]]] = {}
    for fold, (fit_indices, held_indices) in enumerate(folds):
        train_rows = [rows[int(index)] for index in fit_indices]
        held_rows = [rows[int(index)] for index in held_indices]
        result = train_fold(args, train_rows, held_rows, ablation, fold)
        predictions.extend(result["predictions"])
        for control, control_rows in result.get("mechanism_predictions", {}).items():
            mechanism_predictions.setdefault(control, []).extend(control_rows)
    if len({row["key"] for row in predictions}) != len(rows):
        raise RuntimeError(f"OOF coverage mismatch for {ablation}: {len(predictions)} predictions, {len(rows)} rows")
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "ablation": ablation,
        "predictions": predictions,
        "tasks": summarize_oof(predictions),
    }
    if mechanism_predictions:
        result["mechanism_controls"] = {}
        for control, control_rows in mechanism_predictions.items():
            if len({row["key"] for row in control_rows}) != len(rows):
                raise RuntimeError(
                    f"OOF mechanism coverage mismatch for {ablation}/{control}: "
                    f"{len(control_rows)} predictions, {len(rows)} rows"
                )
            result["mechanism_controls"][control] = {
                "predictions": control_rows,
                "tasks": summarize_oof(control_rows),
            }
            atomic_json(
                args.output / "oof" / ablation / f"{control.upper()}_SUMMARY.json",
                result["mechanism_controls"][control],
            )
    atomic_json(output, result)
    return result


def gate1_decision(
    full: dict[str, Any],
    frozen: dict[str, Any],
    matched: dict[str, Any] | None,
    training_rows: list[dict[str, Any]],
    margin: float = 0.01,
) -> dict[str, Any]:
    tasks = {}
    for task in TASKS:
        full_f1 = float(full["tasks"][task]["classification"]["weighted_f1"])
        frozen_f1 = float(frozen["tasks"][task]["classification"]["weighted_f1"])
        if matched is not None:
            baseline_metrics = matched["matched_tav"][task]["classification"]
            baseline_f1 = float(baseline_metrics["weighted_f1"])
            tasks[task] = {
                "full_weighted_f1": full_f1,
                "matched_tav_weighted_f1": baseline_f1,
                "matched_tav_metrics": baseline_metrics,
                "frozen_backbone_weighted_f1": frozen_f1,
                "delta_vs_matched_tav": full_f1 - baseline_f1,
                "margin": margin,
                "passed": full_f1 >= baseline_f1 - margin,
            }
        else:
            teacher_rows = [row for row in training_rows if row["task"] == task]
            teacher_threshold = choose_threshold(
                [row["label"] for row in teacher_rows],
                [row["teacher_probability"] for row in teacher_rows],
            )
            teacher_metrics = metric_block(
                [row["label"] for row in teacher_rows],
                [row["teacher_probability"] for row in teacher_rows],
                teacher_threshold,
            )
            baseline_f1 = float(teacher_metrics["weighted_f1"])
            tasks[task] = {
                "full_weighted_f1": full_f1,
                "strict_meter_teacher_oof": teacher_metrics,
                "frozen_backbone_weighted_f1": frozen_f1,
                "delta": full_f1 - baseline_f1,
                "margin": margin,
                "passed": full_f1 >= baseline_f1 - margin,
            }
    passed = all(value["passed"] for value in tasks.values())
    return {"schema": SCHEMA, "status": "passed" if passed else "failed", "tasks": tasks}


def m1_gate_decision(
    matched: dict[str, Any],
    static_continue: dict[str, Any],
    candidate: dict[str, Any],
    time_shuffle: dict[str, Any],
    margin: float = 0.01,
    mechanism_margin: float = 0.01,
) -> dict[str, Any]:
    tasks = {}
    candidate_minus_shuffle = []
    candidate_minus_static = []
    for task in TASKS:
        baseline_metrics = matched["controls"]["matched_tav"][task]["classification"]
        static_metrics = static_continue["tasks"][task]["classification"]
        candidate_task = candidate["tasks"][task]
        candidate_metrics = candidate_task["classification"]
        candidate_threshold = float(candidate_task["threshold_locked_from_oof"])
        shuffle_rows = [row for row in time_shuffle["predictions"] if row["task"] == task]
        shuffle_metrics = metric_block(
            [row["label"] for row in shuffle_rows],
            [row["probability"] for row in shuffle_rows],
            candidate_threshold,
        )
        shuffle_oracle_metrics = time_shuffle["tasks"][task]["classification"]
        baseline_f1 = float(baseline_metrics["weighted_f1"])
        candidate_f1 = float(candidate_metrics["weighted_f1"])
        shuffle_f1 = float(shuffle_metrics["weighted_f1"])
        dynamics_delta = candidate_f1 - shuffle_f1
        static_delta = candidate_f1 - float(static_metrics["weighted_f1"])
        candidate_minus_shuffle.append(dynamics_delta)
        candidate_minus_static.append(static_delta)
        tasks[task] = {
            "matched_tav": baseline_metrics,
            "m1_static_continue": static_metrics,
            "m1_answer_dynamic": candidate_metrics,
            "m1_time_shuffle": shuffle_metrics,
            "m1_time_shuffle_independently_recalibrated_diagnostic": shuffle_oracle_metrics,
            "mechanism_threshold_source": "m1_answer_dynamic OOF",
            "delta_vs_matched_tav": candidate_f1 - baseline_f1,
            "delta_vs_equal_budget_static": static_delta,
            "delta_vs_time_shuffle": dynamics_delta,
            "noninferiority_margin": margin,
            "noninferiority_passed": candidate_f1 >= baseline_f1 - margin,
        }

    performance_passed = all(value["noninferiority_passed"] for value in tasks.values())
    mean_static_delta = float(np.mean(candidate_minus_static))
    attribution_passed = mean_static_delta >= 0.0
    mean_dynamics_delta = float(np.mean(candidate_minus_shuffle))
    mechanism_passed = (
        mean_dynamics_delta >= 0.0
        and any(value >= mechanism_margin for value in candidate_minus_shuffle)
    )
    passed = performance_passed and attribution_passed and mechanism_passed
    return {
        "schema": SCHEMA,
        "stage": "M1",
        "status": "passed" if passed else "failed",
        "test_labels_loaded": False,
        "primary_model": "m1_answer_dynamic",
        "baseline": "matched_tav",
        "mechanism_control": "m1_time_shuffle",
        "equal_budget_control": "m1_static_continue",
        "performance_passed": performance_passed,
        "attribution_passed": attribution_passed,
        "mechanism_passed": mechanism_passed,
        "mechanism_rule": {
            "mean_candidate_minus_shuffle_at_least": 0.0,
            "at_least_one_task_candidate_minus_shuffle_at_least": mechanism_margin,
        },
        "mean_candidate_minus_shuffle_weighted_f1": mean_dynamics_delta,
        "mean_candidate_minus_equal_budget_static_weighted_f1": mean_static_delta,
        "tasks": tasks,
    }


def final_train_and_test(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    output = args.output / "gate3" / "FINAL_METRICS.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    result = train_fold(args, train_rows, test_rows, "final_full", 0)
    predictions = result["predictions"]
    tasks = {}
    for task in TASKS:
        rows = [row for row in predictions if row["task"] == task]
        metrics = metric_block(
            [row["label"] for row in rows],
            [row["probability"] for row in rows],
            thresholds[task],
        )
        tasks[task] = {
            "classification": metrics,
            "severity": severity_metrics(rows, [row["predicted_score"] for row in rows]),
            "external_tav": EXTERNAL_TAV[task],
            "delta_weighted_f1": metrics["weighted_f1"] - EXTERNAL_TAV[task]["weighted_f1"],
            "predictions": rows,
        }
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "single_shared_checkpoint": True,
        "online_humanomniv2_inference": True,
        "cached_hidden_inference": False,
        "thresholds_locked_from_training_oof": thresholds,
        "tasks": tasks,
        "retrospective_test_disclosure": True,
    }
    atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--daic-root", type=Path, required=True)
    parser.add_argument("--edaic-root", type=Path, required=True)
    parser.add_argument("--meter-cache", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-participants", type=int, default=2)
    parser.add_argument("--vectorize-event-views", action="store_true")
    parser.add_argument("--event-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--pad", type=float, default=0.4)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--max-events-per-participant", type=int, default=0)
    parser.add_argument("--max-gpu-memory", default="74GiB")
    parser.add_argument("--minimum-teacher-coverage", type=float, default=0.90)
    parser.add_argument(
        "--stage",
        choices=("all", "gate0", "matched", "m1", "gate1", "gate2", "gate3", "single_fold"),
        default="all",
    )
    parser.add_argument("--single-fold", type=int, default=None)
    parser.add_argument("--single-ablation", choices=SINGLE_ABLATIONS, default=None)
    parser.add_argument(
        "--matched-baselines",
        type=Path,
        default=None,
        help="Locked M0 MATCHED_BASELINES.json used by the staged M1 decision.",
    )
    parser.add_argument(
        "--m1-base-root",
        type=Path,
        default=None,
        help="Root containing deterministic matched_tav replay FINAL_STATE.pt files.",
    )
    parser.add_argument(
        "--external-gate0-result",
        type=Path,
        default=None,
        help="Combined official DAIC/E-DAIC Gate-0 result created by the strict orchestrator.",
    )
    args = parser.parse_args()

    if args.event_batch_size < 1:
        parser.error("--event-batch-size must be at least 1")

    configure_runtime(args.threads)
    seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "PROTOCOL_AUDIT.json", protocol_audit(args))

    if args.stage == "single_fold":
        if args.single_fold is None or args.single_ablation is None:
            raise ValueError("single_fold requires --single-fold and --single-ablation")
        load_gate0_result(args)
        train_rows = load_rows(args.contract_dir, {"train"})
        if args.single_ablation in MATCHED_CONTROLS + M1_CONTROLS:
            teacher_audit = {
                "status": "not_loaded",
                "reason": f"{args.single_ablation} forbids teacher inputs",
                "rows": len(train_rows),
            }
            for row in train_rows:
                row["teacher_probability"] = None
                if args.single_ablation in M1_CONTROLS:
                    row["score"] = None
            if args.single_ablation in M1_CONTROLS:
                teacher_audit["continuous_score_supervision"] = "disabled"
                teacher_audit["pim"] = "excluded"
                teacher_audit["ensr"] = "excluded"
                teacher_audit["lora"] = "excluded"
        else:
            teacher, teacher_source_audit = load_teacher_map(args.teacher_root, args.contract_dir)
            teacher_audit = assign_teacher(train_rows, teacher, args.minimum_teacher_coverage)
            teacher_audit["source"] = teacher_source_audit
        atomic_json(args.output / "TEACHER_AUDIT.json", teacher_audit)
        folds = fold_assignments(train_rows, args.seed)
        if not 0 <= args.single_fold < len(folds):
            raise ValueError(f"single fold must be in [0, {len(folds) - 1}], got {args.single_fold}")
        fold_audit = protocol_audit(args)
        fold_audit.update(
            {
                "stage": "single_fold",
                "fold": int(args.single_fold),
                "ablation": str(args.single_ablation),
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "gradient_accumulation_participants": int(args.gradient_accumulation_participants),
                "checkpoint_every": int(args.checkpoint_every),
                "max_events_per_participant": int(args.max_events_per_participant),
                "pad": float(args.pad),
                "max_seconds": float(args.max_seconds),
            }
        )
        atomic_json(
            args.output / "oof" / args.single_ablation / f"fold_{args.single_fold}" / "PROTOCOL_AUDIT.json",
            fold_audit,
        )
        fit_indices, held_indices = folds[args.single_fold]
        train_fold_rows = [train_rows[int(index)] for index in fit_indices]
        held_rows = [train_rows[int(index)] for index in held_indices]
        result = train_fold(
            args,
            train_fold_rows,
            held_rows,
            args.single_ablation,
            args.single_fold,
        )
        print(
            json.dumps(
                {
                    "event": "METER_MH_SINGLE_FOLD_COMPLETE",
                    "ablation": args.single_ablation,
                    "fold": args.single_fold,
                    "held_rows": len(held_rows),
                    "status": result.get("status"),
                }
            ),
            flush=True,
        )
        return 0

    if args.stage in {"all", "gate0", "matched"}:
        load_gate0_result(args)
        if args.stage == "gate0":
            return 0

    load_gate0_result(args)

    train_rows = load_rows(args.contract_dir, {"train"})

    if args.stage == "matched":
        matched = {
            ablation: run_oof(args, train_rows, ablation)["tasks"]
            for ablation in MATCHED_CONTROLS
        }
        atomic_json(
            args.output / "MATCHED_BASELINES.json",
            {
                "schema": SCHEMA,
                "status": "complete",
                "test_labels_loaded": False,
                "controls": matched,
                "definitions": {
                    "matched_ta": "online HumanOmniV2 TA clinical view + frozen backbone + ordinary pooling head",
                    "matched_tav": "matched_ta plus audited official visual adapter + ordinary pooling head",
                    "excluded": ["Answer-Dynamic", "PIM", "ENSR", "teacher", "LoRA"],
                },
            },
        )
        return 0

    if args.stage == "m1":
        matched_path = args.matched_baselines
        if matched_path is None or not matched_path.is_file():
            raise RuntimeError("M1 requires --matched-baselines from the completed strict M0 stage")
        matched = json.loads(matched_path.read_text(encoding="utf-8"))
        if matched.get("status") != "complete" or matched.get("test_labels_loaded") is not False:
            raise RuntimeError("M1 requires a complete training-only M0 baseline artifact")
        if args.m1_base_root is None:
            raise RuntimeError("M1 requires --m1-base-root with audited matched_tav replay states")
        static_continue = run_oof(args, train_rows, "m1_static_continue")
        candidate = run_oof(args, train_rows, "m1_answer_dynamic")
        time_shuffle = candidate.get("mechanism_controls", {}).get("time_shuffle")
        if time_shuffle is None:
            raise RuntimeError("M1 candidate is missing the same-checkpoint time-shuffle control")
        decision = m1_gate_decision(matched, static_continue, candidate, time_shuffle)
        atomic_json(args.output / "gate_m1" / "DECISION.json", decision)
        atomic_json(
            args.output / "M1_RESULTS.json",
            {
                "schema": SCHEMA,
                "stage": "M1",
                "status": "complete",
                "test_labels_loaded": False,
                "matched_baselines": str(matched_path.resolve()),
                "static_continue": static_continue["tasks"],
                "candidate": candidate["tasks"],
                "time_shuffle": time_shuffle["tasks"],
                "decision": decision,
            },
        )
        return 0 if decision["status"] == "passed" else 2

    teacher, teacher_source_audit = load_teacher_map(args.teacher_root, args.contract_dir)
    teacher_audit = assign_teacher(train_rows, teacher, args.minimum_teacher_coverage)
    teacher_audit["source"] = teacher_source_audit
    atomic_json(args.output / "TEACHER_AUDIT.json", teacher_audit)

    matched_path = args.output / "MATCHED_BASELINES.json"
    official_pipeline = args.external_gate0_result is not None
    if official_pipeline and args.stage in {"gate1", "gate2", "gate3", "all"} and not matched_path.is_file():
        raise RuntimeError("strict matched TA/TAV baselines are required before Gate-1")
    matched_payload = json.loads(matched_path.read_text(encoding="utf-8")) if matched_path.is_file() else None

    if args.stage in {"all", "gate1"}:
        frozen = run_oof(args, train_rows, "frozen_backbone")
        full = run_oof(args, train_rows, "none")
        decision = gate1_decision(
            full,
            frozen,
            matched_payload["controls"] if matched_payload is not None else None,
            train_rows,
        )
        atomic_json(args.output / "gate1" / "DECISION.json", decision)
        if decision["status"] != "passed":
            atomic_json(
                args.output / "STOPPED.json",
                {"status": "stopped_at_gate1", "test_labels_loaded": False, "decision": decision},
            )
            return 2
        if args.stage == "gate1":
            return 0
    else:
        decision = json.loads((args.output / "gate1" / "DECISION.json").read_text(encoding="utf-8"))
        if decision.get("status") != "passed":
            raise RuntimeError("Gate-1 has not passed")

    if args.stage in {"all", "gate2"}:
        ablations = {}
        for ablation in ABLATIONS:
            ablations[ablation] = run_oof(args, train_rows, ablation)["tasks"]
        atomic_json(args.output / "gate2" / "ABLATIONS.json", {"schema": SCHEMA, "status": "complete", "ablations": ablations})
        if args.stage == "gate2":
            return 0

    full = json.loads((args.output / "oof" / "none" / "SUMMARY.json").read_text(encoding="utf-8"))
    thresholds = {task: float(full["tasks"][task]["threshold_locked_from_oof"]) for task in TASKS}

    final_train_rows = load_rows(args.contract_dir, {"train", "dev"})
    assign_teacher(final_train_rows, teacher, 0.0)
    # This is intentionally the first point at which official test labels are loaded.
    test_rows = load_rows(args.contract_dir, {"test"})
    final = final_train_and_test(args, final_train_rows, test_rows, thresholds)
    atomic_json(
        args.output / "COMPLETE.json",
        {"schema": SCHEMA, "status": "complete", "tasks": list(final["tasks"]), "completed_at": time.time()},
    )
    print(json.dumps({"event": "METER_MH_E2E_COMPLETE", "tasks": final["tasks"]}, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
