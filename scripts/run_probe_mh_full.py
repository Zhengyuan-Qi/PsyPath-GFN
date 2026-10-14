#!/usr/bin/env python3
"""Full PROBE-MH validation with resumable HumanOmniV2 re-encoding.

PROBE-MH treats the interview prompt as an observable condition and models the
participant response state that follows it.  The DAIC-WOZ transcript contains
explicit interviewer prompts.  E-DAIC is retained as an external response
dynamics evaluation, but its released participant transcript does not expose
the interviewer role; the cache records that limitation instead of inventing
question labels.

The script has two stages in one resumable job:
1. Re-encode every usable response with a PROBE-specific HumanOmniV2 prompt.
2. Run participant-level nested OOF and locked official dev/test evaluation.

Raw data and labels are read-only.  All published artifacts are atomic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import av
from decord import VideoReader, cpu
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from qwen_omni_utils import process_mm_info

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clinical_label_contract import load_label_profile, validate_contract
from scripts import visual_modality_contract


SCHEMA = "probe_mh_full_v3"
SEEDS = (7, 17, 27, 41, 53)
AUDIO_MAX_SAMPLES = 208000
HIDDEN_DIM = 3584
PROMPT_VERSION = (
    "PROBE-MH v1: encode one participant response under an explicit interview "
    "condition. Do not diagnose, infer identity, demographics, or a clinical "
    "label. Represent semantic content, vocal delivery, timing, uncertainty, "
    "and the response change elicited by the interviewer context."
)


def configure_threads(value: int) -> None:
    value = max(1, min(int(value), 32))
    for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS",
    ):
        os.environ[key] = str(value)
    torch.set_num_threads(value)
    try:
        torch.set_num_interop_threads(min(4, value))
    except RuntimeError:
        pass


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
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


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def finite(value: Any, dtype=np.float32) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)


def norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).split())


def source_file(dataset: str, root: Path, pid: str, suffix: str) -> Path:
    if dataset == "daic":
        return root / "data" / f"{pid}_{suffix}"
    base = root / "data" / f"{pid}_P"
    candidates = [base / f"{pid}_{suffix}", base / f"{pid}{suffix}"]
    if suffix == "AUDIO.wav":
        candidates += [base / f"{pid}_AUDIO.wav", base / f"{pid}AUDIO.wav"]
    return next((path for path in candidates if path.is_file()), candidates[0])


def load_rows(dataset: str, root: Path, contract_dir: Path, target: str) -> list[dict[str, Any]]:
    profile = "depression_phq8_screening" if target == "depression" else "ptsd_dsm_iv_binary"
    frame, spec = load_label_profile(contract_dir, dataset, profile, include_unlabeled=False)
    rows = []
    for item in frame.to_dict("records"):
        pid = str(int(item["pid"]))
        rows.append({
            "pid": pid,
            "split": str(item["split"]),
            "canonical_split": str(item["canonical_split"]),
            "label": int(item[spec.target_column]),
            "target": target,
        })
    return rows


def load_scores(dataset: str, contract_dir: Path, target: str) -> dict[tuple[str, str], float | None]:
    profile = "depression_phq_score" if target == "depression" else "ptsd_score_repaired"
    frame, spec = load_label_profile(contract_dir, dataset, profile, include_unlabeled=True)
    result = {}
    for item in frame.to_dict("records"):
        key = (str(int(item["pid"])), str(item["split"]))
        value = item.get(spec.target_column)
        mask = item.get(spec.mask_column)
        result[key] = None if int(mask) != 1 or pd.isna(value) else float(value)
    return result


def read_daic_turns(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t", encoding="utf-8")
    frame.columns = [str(col).strip().lstrip("#").strip() for col in frame.columns]
    pending: list[tuple[float, float, str]] = []
    active: dict[str, Any] | None = None
    output: list[dict[str, Any]] = []
    for item in frame.to_dict("records"):
        try:
            start, stop = float(item["start_time"]), float(item["stop_time"])
        except (TypeError, ValueError, KeyError):
            continue
        if stop <= start:
            continue
        speaker = str(item.get("speaker", "")).strip().lower()
        text = norm_text(item.get("value", ""))
        if not text or text.lower() in {"<sync>", "<synch>"}:
            continue
        if speaker == "ellie":
            if active is not None:
                output.append(active)
                active = None
            pending.append((start, stop, text))
            continue
        if speaker != "participant":
            continue
        if active is None:
            prompt_start = min((row[0] for row in pending), default=0.0)
            prompt_stop = max((row[1] for row in pending), default=0.0)
            prompt = " ".join(row[2] for row in pending)
            active = {
                "question": prompt,
                "question_start": prompt_start,
                "question_stop": prompt_stop,
                "start": start,
                "stop": stop,
                "answer": text,
                "question_available": bool(prompt),
            }
            pending = []
        else:
            active["stop"] = stop
            active["answer"] = f"{active['answer']} {text}".strip()
    if active is not None:
        output.append(active)
    for index, item in enumerate(output):
        item["index"] = index
        item["latency"] = max(0.0, float(item["start"]) - float(item["question_stop"])) if item["question_available"] else 0.0
    return [item for item in output if float(item["stop"]) > float(item["start"])]


def read_edaic_turns(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, encoding="utf-8")
    columns = {str(col).strip().lower(): col for col in frame.columns}
    start_col = columns.get("start_time", columns.get("start"))
    stop_col = columns.get("end_time", columns.get("stop_time", columns.get("end")))
    text_col = columns.get("text", columns.get("value"))
    if not start_col or not stop_col or not text_col:
        raise ValueError(f"unexpected E-DAIC transcript columns: {list(frame.columns)}")
    segments = []
    for _, item in frame.iterrows():
        try:
            start, stop = float(item[start_col]), float(item[stop_col])
        except (TypeError, ValueError):
            continue
        text = norm_text(item[text_col])
        if stop <= start or not text or text.lower() in {"nan", "<sync>", "<synch>"}:
            continue
        segments.append({
            "index": len(segments), "question": "", "question_start": 0.0,
            "question_stop": 0.0, "start": start, "stop": stop,
            "answer": text, "latency": 0.0, "question_available": False,
        })
    if not segments:
        return []
    # E-DAIC releases participant ASR without role labels. Merge adjacent ASR
    # fragments into response events; a large gap is treated as a new answer.
    output: list[dict[str, Any]] = []
    max_gap = 2.0
    for segment in segments:
        if output and float(segment["start"]) - float(output[-1]["stop"]) <= max_gap:
            output[-1]["stop"] = segment["stop"]
            output[-1]["answer"] = f"{output[-1]['answer']} {segment['answer']}".strip()
        else:
            segment = dict(segment)
            segment["index"] = len(output)
            output.append(segment)
    return output


def read_turns(dataset: str, root: Path, pid: str) -> list[dict[str, Any]]:
    path = source_file(dataset, root, pid, "TRANSCRIPT.csv" if dataset == "daic" else "Transcript.csv")
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_daic_turns(path) if dataset == "daic" else read_edaic_turns(path)


def write_clip(audio: Path, start: float, stop: float, destination: Path, pad: float, max_seconds: float) -> Path:
    if destination.is_file() and destination.stat().st_size > 44:
        return destination
    info = sf.info(audio)
    left = max(0.0, float(start) - pad)
    right = min(float(info.duration), float(stop) + pad, left + max_seconds)
    signal, rate = sf.read(audio, start=int(left * info.samplerate), stop=int(right * info.samplerate), dtype="float32")
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if rate != 16000:
        import librosa
        signal = librosa.resample(signal, orig_sr=rate, target_sr=16000)
        rate = 16000
    if len(signal) < int(rate):
        signal = np.pad(signal, (0, int(rate) - len(signal)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".wav", dir=destination.parent)
    temporary = Path(name)
    try:
        os.close(fd)
        sf.write(temporary, signal, rate)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_visual_table(
    path: Path,
    max_dims: int = 64,
    reference_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        return np.zeros(0, dtype=np.float32), np.zeros((0, max_dims), dtype=np.float32)
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return np.zeros(0, dtype=np.float32), np.zeros((0, max_dims), dtype=np.float32)
    lower = {str(col).strip().lower(): col for col in frame.columns}
    time_col = next((lower[key] for key in ("timestamp", "time", "timestamp_sec") if key in lower), None)
    if time_col is None:
        time_col = frame.columns[1] if len(frame.columns) > 1 else frame.columns[0]
    times = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=np.float32)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    drop = {str(col).strip().lower() for col in ("frame", "timestamp", "time", "timestamp_sec", "confidence", "success", "name")}
    keep = [col for col in numeric.columns if str(col).strip().lower() not in drop]
    matrix = numeric[keep].to_numpy(dtype=np.float32) if keep else np.zeros((len(frame), 0), dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.shape[1] > max_dims:
        matrix = matrix[:, :max_dims]
    if matrix.shape[1] < max_dims:
        matrix = np.pad(matrix, ((0, 0), (0, max_dims - matrix.shape[1])))
    valid = np.isfinite(times)
    return times[valid], matrix[valid]


def read_mat_visual_table(
    path: Path,
    max_dims: int = 64,
    reference_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read an approved frame-level CNN visual MAT file.

    E-DAIC CNN_ResNet/CNN_VGG files do not carry a timestamp column.  Their
    frame index is aligned to the OpenFace timestamp support, which is stored
    in the cache audit.  They are never parsed through the CSV/audio reader.
    """
    from scipy.io import loadmat

    if not path.is_file():
        return np.zeros(0, dtype=np.float32), np.zeros((0, max_dims), dtype=np.float32)
    payload = loadmat(path)
    arrays = [
        value for key, value in payload.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
    ]
    if not arrays:
        return np.zeros(0, dtype=np.float32), np.zeros((0, max_dims), dtype=np.float32)
    matrix = np.asarray(max(arrays, key=lambda value: value.size), dtype=np.float32)
    matrix = np.squeeze(matrix)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"approved visual MAT must be 2-D: {path}: {matrix.shape}")
    if matrix.shape[0] < matrix.shape[1] and matrix.shape[0] <= 64:
        matrix = matrix.T
    matrix = matrix[:, :max_dims]
    if matrix.shape[1] < max_dims:
        matrix = np.pad(matrix, ((0, 0), (0, max_dims - matrix.shape[1])))
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if reference_times is None or len(reference_times) == 0:
        times = np.arange(len(matrix), dtype=np.float32)
    elif len(matrix) == len(reference_times):
        times = np.asarray(reference_times, dtype=np.float32)
    else:
        times = np.linspace(
            float(reference_times[0]), float(reference_times[-1]), len(matrix), dtype=np.float32
        )
    return times, matrix


def visual_sources(dataset: str, root: Path, pid: str) -> list[Path]:
    paths = visual_modality_contract.default_protocol_visual_paths(dataset, root, pid)
    visual_modality_contract.validate_visual_paths(dataset, root, pid, paths)
    return list(paths.values())


def visual_event_features(dataset: str, root: Path, pid: str, turns: list[dict[str, Any]], max_dims: int = 64) -> tuple[np.ndarray, dict[str, Any]]:
    paths = visual_modality_contract.default_protocol_visual_paths(dataset, root, pid)
    visual_audit = visual_modality_contract.validate_visual_paths(dataset, root, pid, paths)
    if dataset == "edaic" and "openface" not in paths:
        raise RuntimeError(
            f"E-DAIC visual event alignment requires the approved OpenFace timestamp anchor: {pid}"
        )
    tables = []
    reference_times = None
    for name, path in paths.items():
        if path.suffix.lower() == ".mat":
            table = read_mat_visual_table(path, max_dims, reference_times)
        else:
            table = read_visual_table(path, max_dims, reference_times)
        if reference_times is None and len(table[0]):
            reference_times = table[0]
        tables.append(table)
    per_turn = []
    for turn in turns:
        chunks = []
        for times, matrix in tables:
            mask = (times >= float(turn["start"])) & (times <= float(turn["stop"]))
            block = matrix[mask]
            if len(block) == 0:
                chunks.append(np.zeros(max_dims * 5, dtype=np.float32))
                continue
            position = np.linspace(-1.0, 1.0, len(block), dtype=np.float32)
            slope = (position[:, None] * block).sum(axis=0) / max(float(np.square(position).sum()), 1e-6)
            chunks.append(np.concatenate((block.mean(0), block.std(0), np.quantile(block, .25, axis=0), np.quantile(block, .75, axis=0), slope)).astype(np.float32))
        per_turn.append(np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32))
    value = finite(np.stack(per_turn) if per_turn else np.zeros((0, max_dims * 5 * max(1, len(tables)))))
    return value, {
        **visual_audit,
        "sources": visual_audit["sources"],
        "source_frame_rows": int(sum(len(times) > 0 for times, _ in tables)),
        "feature_dim": int(value.shape[1]) if value.ndim == 2 else 0,
        "event_rows_with_any_visual": int(sum(np.linalg.norm(row) > 0 for row in value)),
        "mat_alignment": "openface_timestamp_support_or_frame_index_if_no_openface",
    }


def move_inputs(inputs: Any, device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def messages(prompt: str, answer: str, clip: Path | None, include_audio: bool) -> list[dict[str, Any]]:
    content = []
    if include_audio:
        content.append({"type": "audio", "audio": str(clip)})
    content.append({"type": "text", "text": f"{PROMPT_VERSION}\n{prompt}" + (f"\nParticipant response transcript: {answer}" if answer else "")})
    return [{"role": "user", "content": content}]


def sample_meld_video(path: Path, count: int = 8) -> Any:
    reader = VideoReader(str(path), ctx=cpu(0))
    total = int(len(reader))
    if total <= 0:
        raise RuntimeError(f"MELD video contains no frames: {path}")
    target = max(1, min(int(count), total))
    indices = np.linspace(0, total - 1, target).round().astype(np.int64).tolist()
    frames = reader.get_batch(indices).asnumpy()
    if total >= int(count):
        return str(path)
    return [Image.fromarray(np.asarray(frame)).convert("RGB") for frame in frames]


def meld_messages(row: dict[str, Any], context: str, audio_path: Path, video_payload: Any) -> list[dict[str, Any]]:
    content = [
        {"type": "video", "video": video_payload, **({"nframes": 8} if isinstance(video_payload, str) else {})},
        {"type": "audio", "audio": str(audio_path)},
        {"type": "text", "text": (
            f"{PROMPT_VERSION}\nNormal affective-dialogue context (unlabeled): {context or '[none]'}\n"
            f"Current response transcript: {norm_text(row.get('text', ''))}"
        )},
    ]
    return [{"role": "user", "content": content}]


def materialize_meld_audio(source: Path, destination: Path) -> Path:
    if destination.is_file() and destination.stat().st_size > 44:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.{os.getpid()}.wav")
    temporary.unlink(missing_ok=True)
    container = av.open(str(source))
    stream = next((item for item in container.streams if item.type == "audio"), None)
    if stream is None:
        container.close()
        raise RuntimeError(f"MELD media has no audio stream: {source}")
    chunks = []
    for frame in container.decode(stream):
        array = frame.to_ndarray()
        if array.ndim == 2:
            array = array.mean(axis=0)
        chunks.append(np.asarray(array, dtype=np.float32).reshape(-1))
    container.close()
    if not chunks:
        raise RuntimeError(f"MELD audio decoded no frames: {source}")
    signal = np.concatenate(chunks)
    rate = int(stream.rate or 16000)
    if rate != 16000:
        import librosa
        signal = librosa.resample(signal, orig_sr=rate, target_sr=16000)
        rate = 16000
    try:
        sf.write(temporary, signal, rate, subtype="PCM_16")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract_meld_prior(args: argparse.Namespace, processor: Any, model: Any, device: torch.device, signature: str) -> Path | None:
    """Re-encode unlabeled MELD with the exact PROBE Prompt for the normal prior."""
    if args.meld_source_rows is None or not args.meld_source_rows.is_file():
        return None
    raw_rows = json.loads(args.meld_source_rows.read_text(encoding="utf-8"))
    rows = [row for row in raw_rows if str(row.get("dataset", "")).upper() == "MELD" and Path(str(row.get("video_path", ""))).is_file()]
    rows.sort(key=lambda row: (str(row.get("group", "")), int(str(row.get("sample_id", "utt0")).split("utt")[-1])))
    if not rows:
        raise RuntimeError("no usable MELD rows for PROBE normal prior")
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_key = f"{row.get('split', '')}:{row.get('group', '')}"
        by_group.setdefault(group_key, []).append(row)
    prior_root = args.output / "meld_probe_prior"
    row_cache = prior_root / "rows"
    audio_root = prior_root / "audio"
    row_cache.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    start_time = time.time()
    for batch_start in range(0, len(rows), max(1, int(args.batch_size))):
        batch_rows = rows[batch_start:batch_start + max(1, int(args.batch_size))]
        pending = []
        for row in batch_rows:
            key = str(row.get("cache_key") or sha256_text(f"MELD|{row.get('split')}|{row.get('sample_id')}"))[:20]
            destination = row_cache / f"{key}.npz"
            if cache_valid(destination, signature, required=("tav",)):
                continue
            try:
                audio = materialize_meld_audio(Path(row["video_path"]), audio_root / f"{key}.wav")
                video_payload = sample_meld_video(Path(row["video_path"]), 8)
                current_utt = int(str(row.get("sample_id", "utt0")).split("utt")[-1])
                group_key = f"{row.get('split', '')}:{row.get('group', '')}"
                previous = [
                    item for item in by_group[group_key]
                    if int(str(item.get("sample_id", "utt0")).split("utt")[-1]) < current_utt
                ][-4:]
                context = " ".join(f"{item.get('speaker', 'speaker')}: {norm_text(item.get('text', ''))}" for item in previous)
                pending.append((row, key, destination, [meld_messages(row, context, audio, video_payload)]))
            except Exception as exc:
                failures.append({"sample_id": str(row.get("sample_id")), "error": f"{type(exc).__name__}: {exc}"})
                atomic_json(prior_root / "failures.json", failures)
                print(f"PROBE_MELD_FAIL {batch_start + len(pending) + 1}/{len(rows)} {failures[-1]}", flush=True)
        if pending:
            values = encode_batch(processor, model, [item[3][0] for item in pending], device)
            for item, value in zip(pending, values):
                row, key, destination, _ = item
                metadata = {"schema": "probe_mh_meld_prior_v1", "run_signature": signature, "row": row, "prompt_version": PROMPT_VERSION}
                atomic_npz(destination, {"tav": finite(value[None, :]), "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))})
        done = min(batch_start + len(batch_rows), len(rows))
        if batch_start == 0 or done % 100 < max(1, int(args.batch_size)) or done == len(rows):
            print(f"PROBE_MELD_CACHE {done}/{len(rows)} elapsed_s={time.time()-start_time:.1f}", flush=True)
    if failures:
        raise RuntimeError(f"MELD PROBE prior failed rows={len(failures)}")
    ordered = []
    groups = []
    for row in rows:
        key = str(row.get("cache_key") or sha256_text(f"MELD|{row.get('split')}|{row.get('sample_id')}"))[:20]
        path = row_cache / f"{key}.npz"
        if not path.is_file():
            raise RuntimeError(f"missing MELD prior cache: {path}")
        with np.load(path, allow_pickle=False) as archive:
            ordered.append(finite(archive["tav"])[0])
        groups.append(str(row.get("group", "")))
    aggregate = prior_root / "probe_tav_embeddings.npz"
    atomic_npz(aggregate, {
        "tav": finite(np.stack(ordered)),
        "group_id": np.asarray(groups),
        "metadata_json": np.asarray(json.dumps({"schema": "probe_mh_meld_prior_v1", "run_signature": signature, "rows": len(rows)}, ensure_ascii=False, sort_keys=True)),
    })
    return aggregate


@torch.inference_mode()
def encode_batch(processor: Any, model: Any, batch: list[list[dict[str, Any]]], device: torch.device) -> np.ndarray:
    rendered, audios, images, videos = [], [], [], []
    for item in batch:
        rendered.append(processor.apply_chat_template(item, tokenize=False, add_generation_prompt=False))
        aa, ii, vv = process_mm_info(item, use_audio_in_video=False)
        audios.extend(aa or [])
        images.extend(ii or [])
        videos.extend(vv or [])
    inputs = processor(
        text=rendered,
        audio=audios or None,
        images=images or None,
        videos=videos or None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
        audio_kwargs={"max_length": AUDIO_MAX_SAMPLES},
    )
    inputs = move_inputs(inputs, device)
    output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = torch.stack([layer.float() for layer in output.hidden_states[-4:]], dim=0).mean(dim=0)
    mask = inputs.get("attention_mask")
    if mask is None:
        pooled = hidden.mean(dim=1)
    else:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    value = pooled.detach().cpu().numpy().astype(np.float32)
    del output, hidden, pooled, inputs
    torch.cuda.empty_cache()
    return value


def load_model(model_path: Path, device: torch.device):
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True, local_files_only=True,
        device_map={"": 0} if device.type == "cuda" else None,
        max_memory={0: "72GiB", "cpu": "400GiB"} if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    ).eval()
    return processor, model


def cache_file(output: Path, dataset: str, pid: str) -> Path:
    return output / "cache" / dataset / f"{pid}.npz"


def cache_valid(path: Path, signature: str, required: tuple[str, ...] = ("probe", "answer", "question", "visual")) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"].item()))
            if meta.get("dataset") != "meld":
                visual_modality_contract.assert_visual_metadata(meta)
            return meta.get("run_signature") == signature and all(
                name in archive.files and np.isfinite(np.asarray(archive[name])).all()
                for name in required
            )
    except Exception:
        return False


def extract_dataset(args: argparse.Namespace, dataset: str, root: Path, rows: list[dict[str, Any]], processor: Any, model: Any, device: torch.device, signature: str) -> None:
    clip_root = args.output / "clips" / dataset
    failures: list[dict[str, Any]] = []
    failure_path = args.output / "cache" / dataset / "failures.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(rows, 1):
        destination = cache_file(args.output, dataset, row["pid"])
        if cache_valid(destination, signature):
            continue
        try:
            audio = source_file(dataset, root, row["pid"], "AUDIO.wav")
            turns = read_turns(dataset, root, row["pid"])
            if not audio.is_file() or not turns:
                raise RuntimeError(f"missing audio or turns audio={audio} turns={len(turns)}")
            clips = []
            for turn in turns:
                key = sha256_text(f"{dataset}|{row['pid']}|{turn['index']}|{turn['start']}|{turn['stop']}")[:20]
                clips.append(write_clip(audio, turn["start"], turn["stop"], clip_root / f"{row['pid']}_{key}.wav", args.pad, args.max_seconds))
            visual, visual_audit = visual_event_features(dataset, root, row["pid"], turns)
            probe_values, answer_values, question_values = [], [], []
            for start in range(0, len(turns), args.batch_size):
                block = turns[start:start + args.batch_size]
                block_clips = clips[start:start + args.batch_size]
                probe_batch = [messages(
                    f"Interviewer question/context: {turn['question'] or '[not released]'}",
                    str(turn["answer"]), clip, True
                ) for turn, clip in zip(block, block_clips)]
                answer_batch = [messages("No interviewer question is supplied; encode the participant response only.", str(turn["answer"]), clip, True) for turn, clip in zip(block, block_clips)]
                probe_values.extend(encode_batch(processor, model, probe_batch, device))
                answer_values.extend(encode_batch(processor, model, answer_batch, device))
                q_block = [turn for turn in block if turn["question_available"]]
                if q_block:
                    question_batch = [messages(f"Encode only this interviewer question/context: {turn['question']}", "", None, False) for turn in q_block]
                    question_values.extend(encode_batch(processor, model, question_batch, device))
            expected_questions = sum(bool(turn["question_available"]) for turn in turns)
            if len(question_values) != expected_questions:
                raise RuntimeError(
                    f"question cache count mismatch: got={len(question_values)} expected={expected_questions}"
                )
            question_iter = iter(question_values)
            question_values = [
                next(question_iter) if turn["question_available"] else np.zeros(HIDDEN_DIM, dtype=np.float32)
                for turn in turns
            ]
            metadata = {
                "schema": SCHEMA, "run_signature": signature, "dataset": dataset,
                "pid": row["pid"], "split": row["split"], "target_inventory": row["target"],
                "question_context": "explicit_interviewer_role" if dataset == "daic" else "unavailable_in_released_participant_transcript",
                "prompt_version": PROMPT_VERSION, "prompt_sha256": sha256_text(PROMPT_VERSION),
                "turns": turns, "visual_audit": visual_audit,
                "visual_modality_contract": visual_modality_contract.SCHEMA,
                "visual_modality_contract_sha256": visual_modality_contract.contract_hash(),
            }
            visual_modality_contract.assert_visual_metadata(metadata)
            atomic_npz(destination, {
                "probe": finite(np.stack(probe_values)),
                "answer": finite(np.stack(answer_values)),
                "question": finite(np.stack(question_values)),
                "visual": finite(visual),
                "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
            })
            print(f"PROBE_CACHE {dataset} {number}/{len(rows)} pid={row['pid']} turns={len(turns)}", flush=True)
        except Exception as exc:
            item = {"pid": row["pid"], "error": f"{type(exc).__name__}: {exc}"}
            failures.append(item)
            atomic_json(failure_path, failures)
            print(f"PROBE_FAIL {dataset} {number}/{len(rows)} pid={row['pid']} {item['error']}", flush=True)
            if "out of memory" in str(exc).lower():
                raise
    if failures:
        raise RuntimeError(f"{dataset}: {len(failures)} cache rows failed; see {failure_path}")


class Projection:
    def __init__(self, components: int):
        self.components = components

    def fit(self, records: list[dict[str, Any]], key: str, extra_matrix: np.ndarray | None = None) -> "Projection":
        parts = [record[key] for record in records]
        if extra_matrix is not None and len(extra_matrix):
            parts.append(finite(extra_matrix))
        matrix = np.concatenate(parts, axis=0)
        self.scaler = StandardScaler().fit(matrix)
        scaled = self.scaler.transform(matrix)
        n = max(1, min(self.components, scaled.shape[0] - 1, scaled.shape[1]))
        self.pca = PCA(n_components=n, whiten=True, random_state=20260807).fit(scaled)
        return self

    def transform(self, value: np.ndarray) -> np.ndarray:
        return finite(self.pca.transform(self.scaler.transform(finite(value))))


def summarize(matrix: np.ndarray) -> np.ndarray:
    matrix = finite(matrix)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if len(matrix) == 0:
        return np.zeros(matrix.shape[1] * 8 if matrix.ndim == 2 else 1, dtype=np.float32)
    position = np.linspace(-1.0, 1.0, len(matrix), dtype=np.float32)
    slope = (position[:, None] * matrix).sum(0) / max(float(np.square(position).sum()), 1e-6)
    delta = np.diff(matrix, axis=0) if len(matrix) > 1 else np.zeros((1, matrix.shape[1]), dtype=np.float32)
    return finite(np.concatenate((matrix.mean(0), matrix.std(0), np.quantile(matrix, .1, axis=0), np.quantile(matrix, .9, axis=0), slope, matrix[-1] - matrix[0], np.mean(np.abs(delta), axis=0), np.sqrt(np.mean(np.square(delta), axis=0)))))


def scalar_summary(value: np.ndarray) -> np.ndarray:
    value = finite(value).reshape(-1)
    if len(value) == 0:
        return np.zeros(8, dtype=np.float32)
    delta = np.diff(value) if len(value) > 1 else np.zeros(1, dtype=np.float32)
    position = np.linspace(-1.0, 1.0, len(value), dtype=np.float32)
    slope = (position * value).sum() / max(float(np.square(position).sum()), 1e-6)
    return finite(np.asarray((value.mean(), value.std(), np.quantile(value, .1), np.quantile(value, .9), slope, value[-1] - value[0], np.mean(np.abs(delta)), np.sqrt(np.mean(np.square(delta))))))


def fit_updater(records: list[dict[str, Any]], projections: dict[str, Projection]) -> tuple[StandardScaler, Ridge]:
    xs, ys = [], []
    for record in records:
        probe = projections["probe"].transform(record["probe"])
        question = projections["question"].transform(record["question"])
        for i in range(1, len(probe)):
            timing = np.asarray((i / max(len(probe) - 1, 1), math.log1p(record["latency"][i]), math.log1p(record["duration"][i]), math.log1p(record["words"][i])), dtype=np.float32)
            xs.append(np.concatenate((probe[i - 1], question[i], question[i] - question[i - 1], timing)))
            ys.append(probe[i])
    if not xs:
        raise RuntimeError("no transitions available for PROBE updater")
    scaler = StandardScaler().fit(np.stack(xs))
    model = Ridge(alpha=100.0).fit(scaler.transform(np.stack(xs)), np.stack(ys))
    return scaler, model


def fit_normal_prior(path: Path | None, projection: Projection) -> tuple[StandardScaler, Ridge] | None:
    """Fit an unlabeled affect-dialogue next-state prior from MELD TAV cache."""
    if path is None or not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if "tav" not in archive.files:
            raise RuntimeError(f"MELD prior cache has no tav view: {path}")
        matrix = finite(archive["tav"])
        groups = np.asarray(archive["group_id"]) if "group_id" in archive.files else np.zeros(len(matrix), dtype=np.int64)
    if len(matrix) < 4:
        return None
    state = projection.transform(matrix)
    keep = groups[1:] == groups[:-1]
    x = state[:-1][keep]
    y = state[1:][keep]
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=100.0).fit(scaler.transform(x), y)
    return scaler, model


def participant_features(
    record: dict[str, Any], projections: dict[str, Projection], updater: tuple[StandardScaler, Ridge],
    normal_prior: tuple[StandardScaler, Ridge] | None = None, shuffle_question: bool = False,
) -> dict[str, np.ndarray]:
    probe = projections["probe"].transform(record["probe"])
    answer = projections["answer"].transform(record["answer"])
    question = projections["question"].transform(record["question"])
    visual = projections["visual"].transform(record["visual"])
    if shuffle_question and len(question) > 1:
        question = question[np.random.default_rng(20260807 + int(record["pid"])).permutation(len(question))]
    base = summarize(answer)
    if len(probe) < 2:
        prior_feature = np.zeros(8 * 24, dtype=np.float32)
        return {"ta": base, "probe": np.concatenate((base, summarize(probe), summarize(visual), prior_feature)), "probe_no_visual": np.concatenate((base, summarize(probe), prior_feature)), "question_only": summarize(question)}
    xs, pred = [], []
    for i in range(1, len(probe)):
        timing = np.asarray((i / max(len(probe) - 1, 1), math.log1p(record["latency"][i]), math.log1p(record["duration"][i]), math.log1p(record["words"][i])), dtype=np.float32)
        xs.append(np.concatenate((probe[i - 1], question[i], question[i] - question[i - 1], timing)))
    predicted = updater[1].predict(updater[0].transform(np.stack(xs)))
    prior_residual = np.zeros((len(probe) - 1, probe.shape[1]), dtype=np.float32)
    if normal_prior is not None:
        transition_scaler, transition_model = normal_prior
        prior_pred = transition_model.predict(transition_scaler.transform(probe[:-1]))
        prior_residual = probe[1:] - prior_pred
    observed_delta = probe[1:] - probe[:-1]
    predicted_delta = predicted - probe[:-1]
    residual = probe[1:] - predicted
    q_delta = question[1:] - question[:-1]
    delta_norm = np.linalg.norm(observed_delta, axis=1)
    pred_norm = np.linalg.norm(predicted_delta, axis=1)
    q_norm = np.linalg.norm(q_delta, axis=1)
    alignment = np.sum(observed_delta * predicted_delta, axis=1) / np.maximum(delta_norm * pred_norm, 1e-6)
    q_alignment = np.sum(observed_delta * q_delta, axis=1) / np.maximum(delta_norm * q_norm, 1e-6)
    timing = np.stack((np.asarray(record["latency"][1:], dtype=np.float32), np.asarray(record["duration"][1:], dtype=np.float32), np.asarray(record["words"][1:], dtype=np.float32)), axis=1)
    dynamic = np.concatenate((summarize(probe), summarize(observed_delta), summarize(q_delta), summarize(predicted_delta), summarize(residual), summarize(prior_residual), summarize(timing), scalar_summary(delta_norm), scalar_summary(pred_norm), scalar_summary(q_norm), scalar_summary(alignment), scalar_summary(q_alignment)))
    no_visual = np.concatenate((base, dynamic))
    full = np.concatenate((no_visual, summarize(visual)))
    return {"ta": base, "probe_no_visual": no_visual, "probe": full, "question_only": summarize(question)}


def build_features(records: list[dict[str, Any]], fit_records: list[dict[str, Any]], meld_cache: Path | None = None, shuffle_question: bool = False) -> dict[str, np.ndarray]:
    meld_matrix = None
    if meld_cache is not None and meld_cache.is_file():
        with np.load(meld_cache, allow_pickle=False) as archive:
            meld_matrix = finite(archive["tav"]) if "tav" in archive.files else None
    projections = {}
    for key, components in (("probe", 24), ("answer", 24), ("question", 16), ("visual", 16)):
        projections[key] = Projection(components).fit(fit_records, key, meld_matrix if key == "probe" else None)
    updater = fit_updater(fit_records, projections)
    normal_prior = fit_normal_prior(meld_cache, projections["probe"])
    values = {name: [] for name in ("ta", "probe_no_visual", "probe", "question_only")}
    for record in records:
        item = participant_features(record, projections, updater, normal_prior, shuffle_question)
        for name in values:
            values[name].append(item[name])
    return {name: finite(np.stack(items)) for name, items in values.items()}


def transition_metrics(
    records: list[dict[str, Any]], fit_records: list[dict[str, Any]], meld_cache: Path | None = None,
    shuffle_question: bool = False, use_operator: bool = True,
) -> dict[str, float]:
    """Evaluate the learned conditional state transition without clinical labels."""
    projections = {}
    for key, components in (("probe", 24), ("answer", 24), ("question", 16), ("visual", 16)):
        projections[key] = Projection(components).fit(fit_records, key)
    updater = fit_updater(fit_records, projections)
    _ = fit_normal_prior(meld_cache, projections["probe"])
    squared_errors: list[np.ndarray] = []
    cosines: list[np.ndarray] = []
    for record in records:
        probe = projections["probe"].transform(record["probe"])
        question = projections["question"].transform(record["question"])
        if shuffle_question and len(question) > 1:
            question = question[np.random.default_rng(20260807 + int(record["pid"])).permutation(len(question))]
        if len(probe) < 2:
            continue
        observed = probe[1:]
        if use_operator:
            xs = []
            for i in range(1, len(probe)):
                timing = np.asarray((
                    i / max(len(probe) - 1, 1), math.log1p(record["latency"][i]),
                    math.log1p(record["duration"][i]), math.log1p(record["words"][i])
                ), dtype=np.float32)
                xs.append(np.concatenate((probe[i - 1], question[i], question[i] - question[i - 1], timing)))
            predicted = updater[1].predict(updater[0].transform(np.stack(xs)))
        else:
            predicted = probe[:-1]
        observed_delta = observed - probe[:-1]
        predicted_delta = predicted - probe[:-1]
        denom = np.linalg.norm(observed_delta, axis=1) * np.linalg.norm(predicted_delta, axis=1)
        cosine = np.sum(observed_delta * predicted_delta, axis=1) / np.maximum(denom, 1e-6)
        squared_errors.append(np.mean(np.square(observed - predicted), axis=1))
        cosines.append(cosine)
    if not squared_errors:
        return {"transitions": 0.0, "next_state_mse": 0.0, "update_cosine": 0.0}
    return {
        "transitions": float(sum(len(value) for value in squared_errors)),
        "next_state_mse": float(np.mean(np.concatenate(squared_errors))),
        "update_cosine": float(np.mean(np.concatenate(cosines))),
    }


def metric(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = np.asarray(prob) >= threshold
    result = {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }
    result["roc_auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else 0.0
    return result


def metric_per_sample_threshold(y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    """OOF metrics with the threshold selected inside each outer fold."""
    thresholds = np.asarray(thresholds, dtype=np.float64)
    pred = np.asarray(prob, dtype=np.float64) >= thresholds
    result = {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }
    result["roc_auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else 0.0
    return result


def fit_bundle(x: np.ndarray, y: np.ndarray, c_value: float, seed: int):
    scaler = StandardScaler().fit(finite(x))
    scaled = scaler.transform(finite(x))
    n = max(1, min(24, len(scaled) - 1, scaled.shape[1]))
    pca = PCA(n_components=n, whiten=True, random_state=seed).fit(scaled)
    model = LogisticRegression(C=c_value, max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed)
    model.fit(pca.transform(scaled), y)
    return scaler, pca, model


def predict_bundle(bundle, x: np.ndarray) -> np.ndarray:
    scaler, pca, model = bundle
    return bundle[2].predict_proba(bundle[1].transform(bundle[0].transform(finite(x))))[:, 1]


def choose_config(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    folds = max(2, min(4, int(np.bincount(y).min())))
    split = StratifiedKFold(folds, shuffle=True, random_state=seed + 31)
    candidates = []
    for c_value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        probs = np.zeros(len(y), dtype=np.float64)
        for fit, held in split.split(np.zeros(len(y)), y):
            probs[held] = predict_bundle(fit_bundle(x[fit], y[fit], c_value, seed + int(held[0])), x[held])
        for threshold in np.arange(.30, .701, .025):
            m = metric(y, probs, float(threshold))
            candidates.append((m["weighted_f1"] - .05 * m["brier"], m["pr_auc"], c_value, float(threshold)))
    best = max(candidates, key=lambda item: item[:2])
    return float(best[2]), float(best[3])


def ccc(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = finite(y, np.float64), finite(pred, np.float64)
    if len(y) < 2:
        return 0.0
    vy, vp = np.var(y), np.var(pred)
    cov = np.mean((y - y.mean()) * (pred - pred.mean()))
    den = vy + vp + (y.mean() - pred.mean()) ** 2
    return float(2 * cov / den) if den > 1e-12 else 0.0


def fit_regressor(x: np.ndarray, y: np.ndarray, alpha: float, seed: int):
    scaler = StandardScaler().fit(finite(x))
    scaled = scaler.transform(finite(x))
    n = max(1, min(24, len(scaled) - 1, scaled.shape[1]))
    pca = PCA(n_components=n, whiten=True, random_state=seed).fit(scaled)
    model = Ridge(alpha=alpha).fit(pca.transform(scaled), y)
    return scaler, pca, model


def predict_regressor(bundle, x: np.ndarray) -> np.ndarray:
    return bundle[2].predict(bundle[1].transform(bundle[0].transform(finite(x))))


def task_records(cache_dir: Path, rows: list[dict[str, Any]], scores: dict[tuple[str, str], float | None]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        path = cache_dir / f"{row['pid']}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"].item()))
            turns = meta["turns"]
            result.append({
                "pid": row["pid"], "split": row["split"], "label": row["label"],
                "score": scores.get((row["pid"], row["split"])),
                "probe": finite(archive["probe"]), "answer": finite(archive["answer"]),
                "question": finite(archive["question"]), "visual": finite(archive["visual"]),
                "latency": np.asarray([float(turn.get("latency", 0.0)) for turn in turns], dtype=np.float32),
                "duration": np.asarray([float(turn["stop"]) - float(turn["start"]) for turn in turns], dtype=np.float32),
                "words": np.asarray([len(str(turn["answer"]).split()) for turn in turns], dtype=np.float32),
            })
    return result


def evaluate_dataset(args: argparse.Namespace, dataset: str, rows_by_target: dict[str, list[dict[str, Any]]], cache_dir: Path, contract_dir: Path) -> dict[str, Any]:
    output = {"schema": SCHEMA, "dataset": dataset, "tasks": {}}
    for target, rows in rows_by_target.items():
        scores = load_scores(dataset, contract_dir, target)
        records = task_records(cache_dir, rows, scores)
        train = [r for r in records if r["split"] == "train"]
        dev = [r for r in records if r["split"] == "dev"]
        test = [r for r in records if r["split"] == "test"]
        y_train = np.asarray([r["label"] for r in train], dtype=np.int64)
        runs = []
        for seed in SEEDS:
            path = args.output / "evaluation" / dataset / target / f"seed_{seed}.json"
            if path.is_file():
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    if int(cached["seed"]) == seed:
                        runs.append(cached)
                        continue
                except Exception:
                    pass
            outer = StratifiedKFold(5, shuffle=True, random_state=seed)
            names = ("ta", "probe_no_visual", "probe", "probe_shuffle", "question_only")
            oof = {name: np.zeros(len(train), dtype=np.float64) for name in names}
            oof_thresholds = {name: np.full(len(train), 0.5, dtype=np.float64) for name in names}
            for fold, (fit_idx, held_idx) in enumerate(outer.split(np.zeros(len(train)), y_train)):
                fit_records = [train[int(i)] for i in fit_idx]
                held_records = [train[int(i)] for i in held_idx]
                fit_values = build_features(fit_records, fit_records, args.meld_cache, False)
                held_values = build_features(held_records, fit_records, args.meld_cache, False)
                held_shuffle = build_features(held_records, fit_records, args.meld_cache, True)
                y_fit = y_train[fit_idx]
                for name in ("ta", "probe_no_visual", "probe", "question_only"):
                    c_value, threshold = choose_config(fit_values[name], y_fit, seed + fold * 101 + len(name))
                    oof[name][held_idx] = predict_bundle(fit_bundle(fit_values[name], y_fit, c_value, seed + fold), held_values[name])
                    oof_thresholds[name][held_idx] = threshold
                c_value, threshold = choose_config(fit_values["probe_no_visual"], y_fit, seed + fold * 101 + 77)
                oof["probe_shuffle"][held_idx] = predict_bundle(fit_bundle(fit_values["probe_no_visual"], y_fit, c_value, seed + fold + 77), held_shuffle["probe_no_visual"])
                oof_thresholds["probe_shuffle"][held_idx] = threshold
            fold_metrics = {name: metric_per_sample_threshold(y_train, oof[name], oof_thresholds[name]) for name in names}
            runs.append({"seed": seed, "metrics": fold_metrics})
            atomic_json(path, runs[-1])
        aggregate = {}
        for name in ("ta", "probe_no_visual", "probe", "probe_shuffle", "question_only"):
            aggregate[name] = {}
            for key in ("weighted_f1", "macro_f1", "positive_f1", "accuracy", "pr_auc", "roc_auc", "brier"):
                vals = [run["metrics"][name][key] for run in runs]
                aggregate[name][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}
        fit_values = build_features(train, train, args.meld_cache, False)
        dev_values = build_features(dev, train, args.meld_cache, False)
        test_values = build_features(test, train, args.meld_cache, False)
        dev_shuffle = build_features(dev, train, args.meld_cache, True)
        test_shuffle = build_features(test, train, args.meld_cache, True)
        locked = {}
        for name, x_train, x_dev, x_test in (
            ("ta", fit_values["ta"], dev_values["ta"], test_values["ta"]),
            ("probe_no_visual", fit_values["probe_no_visual"], dev_values["probe_no_visual"], test_values["probe_no_visual"]),
            ("probe", fit_values["probe"], dev_values["probe"], test_values["probe"]),
            ("probe_shuffle", fit_values["probe_no_visual"], dev_shuffle["probe_no_visual"], test_shuffle["probe_no_visual"]),
            ("question_only", fit_values["question_only"], dev_values["question_only"], test_values["question_only"]),
        ):
            c_value, threshold = choose_config(x_train, y_train, 20260807 + len(name))
            bundle = fit_bundle(x_train, y_train, c_value, 20260807)
            locked[name] = {"c": c_value, "threshold": threshold, "dev": metric(np.asarray([r["label"] for r in dev]), predict_bundle(bundle, x_dev), threshold), "test": metric(np.asarray([r["label"] for r in test]), predict_bundle(bundle, x_test), threshold)}
        continuous = {}
        scored_train = [r for r in train if r["score"] is not None]
        scored_dev = [r for r in dev if r["score"] is not None]
        scored_test = [r for r in test if r["score"] is not None]
        if scored_train and scored_test:
            train_idx = [train.index(r) for r in scored_train]
            dev_idx = [dev.index(r) for r in scored_dev]
            test_idx = [test.index(r) for r in scored_test]
            y_score_train = np.asarray([r["score"] for r in scored_train], dtype=np.float64)
            y_score_dev = np.asarray([r["score"] for r in scored_dev], dtype=np.float64)
            y_score_test = np.asarray([r["score"] for r in scored_test], dtype=np.float64)
            for name in ("ta", "probe_no_visual", "probe"):
                bundle = fit_regressor(fit_values[name][train_idx], y_score_train, 10.0, 20260807)
                pred_dev = predict_regressor(bundle, dev_values[name][dev_idx])
                pred_test = predict_regressor(bundle, test_values[name][test_idx])
                continuous[name] = {
                    "dev": {"ccc": ccc(y_score_dev, pred_dev), "mae": float(mean_absolute_error(y_score_dev, pred_dev)), "rmse": float(np.sqrt(mean_squared_error(y_score_dev, pred_dev)))},
                    "test": {"ccc": ccc(y_score_test, pred_test), "mae": float(mean_absolute_error(y_score_test, pred_test)), "rmse": float(np.sqrt(mean_squared_error(y_score_test, pred_test)))},
                }
        output["tasks"][target] = {
            "inventory": {"train": len(train), "dev": len(dev), "test": len(test), "cache_records": len(records)},
            "oof_train": aggregate, "locked_official": locked, "continuous": continuous,
            "mechanism": {
                "dev": {
                    "history": transition_metrics(dev, train, args.meld_cache, False, False),
                    "probe_operator": transition_metrics(dev, train, args.meld_cache, False, True),
                    "history_baseline": transition_metrics(dev, train, args.meld_cache, False, False),
                    "question_shuffle": transition_metrics(dev, train, args.meld_cache, True),
                },
                "test": {
                    "history": transition_metrics(test, train, args.meld_cache, False, False),
                    "probe_operator": transition_metrics(test, train, args.meld_cache, False, True),
                    "history_baseline": transition_metrics(test, train, args.meld_cache, False, False),
                    "question_shuffle": transition_metrics(test, train, args.meld_cache, True),
                },
            },
            "protocol": {"label_profile": "depression_phq8_screening" if target == "depression" else "ptsd_dsm_iv_binary", "seeds": list(SEEDS), "outer_folds": 5, "question_control": "within-participant permutation", "question_only_control": True, "normal_reaction_prior": str(args.meld_cache) if args.meld_cache else None, "raw_inputs_read_only": True},
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("daic", "edaic", "all"), default="all")
    parser.add_argument("--daic-root", type=Path, default=Path("data/DAIC-WOZ"))
    parser.add_argument("--edaic-root", type=Path, default=Path("data/E-DAIC"))
    parser.add_argument("--model", type=Path, default=Path("HumanOmniV2"))
    parser.add_argument("--contract-dir", type=Path, default=Path("data/clinical_label_contract_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/probe_mh_full_20260807"))
    parser.add_argument("--meld-cache", type=Path, default=None, help="Unlabeled MELD HumanOmniV2 TAV matrix used only as a normal reaction prior.")
    parser.add_argument("--meld-source-rows", type=Path, default=None, help="MELD source_rows.json; enables exact-Prompt normal-prior re-encoding.")
    parser.add_argument("--limit-rows", type=int, default=0, help="Code smoke only; limit clinical rows without changing the formal protocol.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pad", type=float, default=.4)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    args = parser.parse_args()
    configure_threads(args.threads)
    validate_contract(args.contract_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = ["daic", "edaic"] if args.dataset == "all" else [args.dataset]
    config = {"schema": SCHEMA, "model": str(args.model), "prompt": PROMPT_VERSION, "prompt_sha256": sha256_text(PROMPT_VERSION), "datasets": datasets, "batch_size": args.batch_size, "threads": min(args.threads, 32), "meld_cache": str(args.meld_cache) if args.meld_cache else None, "meld_source_rows": str(args.meld_source_rows) if args.meld_source_rows else None, "visual_modality_contract": visual_modality_contract.SCHEMA, "visual_modality_contract_sha256": visual_modality_contract.contract_hash(), "raw_inputs_read_only": True}
    signature = sha256_text(json.dumps(config, sort_keys=True, separators=(",", ":")))
    config["run_signature"] = signature
    atomic_json(args.output / "RUN_CONTRACT.json", config)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(0)
        if free < 40 * 2**30:
            raise RuntimeError(f"refusing shared GPU; free={free / 2**30:.2f} GiB")
        print(f"GPU_READY free_gb={free / 2**30:.2f} total_gb={total / 2**30:.2f}", flush=True)
    processor, model = load_model(args.model, device)
    if args.meld_source_rows is not None and args.meld_cache is None:
        args.meld_cache = extract_meld_prior(args, processor, model, device, signature)
        config["meld_cache"] = str(args.meld_cache) if args.meld_cache else None
        atomic_json(args.output / "RUN_CONTRACT.json", config)
    all_reports = {"schema": SCHEMA, "status": "running", "run_signature": signature, "datasets": {}}
    for dataset in datasets:
        root = args.daic_root if dataset == "daic" else args.edaic_root
        targets = {"depression": load_rows(dataset, root, args.contract_dir, "depression")}
        if dataset == "edaic":
            targets["ptsd"] = load_rows(dataset, root, args.contract_dir, "ptsd")
        if args.limit_rows > 0:
            targets = {name: rows[:args.limit_rows] for name, rows in targets.items()}
        inventory = targets["depression"]
        extract_dataset(args, dataset, root, inventory, processor, model, device, signature)
        cache_dir = args.output / "cache" / dataset
        all_reports["datasets"][dataset] = evaluate_dataset(args, dataset, targets, cache_dir, args.contract_dir)
        atomic_json(args.output / "SUMMARY_RUNNING.json", all_reports)
    all_reports["status"] = "complete"
    atomic_json(args.output / "FINAL_METRICS.json", all_reports)
    print(json.dumps({"event": "probe_mh_complete", "output": str(args.output / "FINAL_METRICS.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
