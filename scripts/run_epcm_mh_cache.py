#!/usr/bin/env python3
"""Lossless, resumable EPCM-MH HumanOmniV2 cache extraction.

Each dataset is run as an independent job, but every cache row is atomic and
can be resumed.  The clinical source data and labels are read-only.  This
script only creates a new derived cache and records the exact prompt/model
contract used for it.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import epcm_mh_prompts as prompts
from scripts import run_probe_mh_full as base
from scripts import visual_modality_contract


SCHEMA = "epcm_mh_cache_v1"
HIDDEN_DIM = int(base.HIDDEN_DIM)
SEED = 20260808
HISTORY_TURNS = 8
MAX_HISTORY_CHARS = 10000
# The E2E measurement head uses one fixed visual input width.  Dataset-native
# visual summaries may be narrower; right-padding preserves every source
# value while making the cache and online path exactly shape-compatible.
MODEL_VISUAL_SIZE = 1280


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=base.json_default)
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


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def finite(value: Any, dtype=np.float32) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return " ".join(str(value).split())


def history_text(turns: list[dict[str, Any]], index: int) -> str:
    start = max(0, index - HISTORY_TURNS)
    blocks = []
    for turn in turns[start:index]:
        question = clean(turn.get("question"))
        answer = clean(turn.get("answer"))
        if question:
            blocks.append(f"INTERVIEWER: {question}")
        if answer:
            blocks.append(f"PARTICIPANT: {answer}")
    value = "\n".join(blocks)
    if len(value) > MAX_HISTORY_CHARS:
        value = value[-MAX_HISTORY_CHARS:]
    return value


def model_files_fingerprint(model_path: Path) -> str:
    parts = []
    for path in sorted(model_path.glob("*.safetensors")):
        parts.append(file_fingerprint(path))
    for name in ("config.json", "model.safetensors.index.json"):
        path = model_path / name
        if path.is_file():
            parts.append(file_fingerprint(path))
    return sha256_text(json.dumps(parts, sort_keys=True, separators=(",", ":")))


def encoder_implementation_fingerprint() -> str:
    """Hash code that changes the derived representation, not just prompts."""
    paths = (
        Path(__file__).resolve(),
        Path(prompts.__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(visual_modality_contract.__file__).resolve(),
    )
    payload = []
    for path in paths:
        payload.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def wait_for_free_memory(min_free_gib: float = 3.0, label: str = "inference") -> None:
    """Wait for transient external GPU pressure instead of self-terminating."""
    while True:
        free, _ = torch.cuda.mem_get_info(0)
        free_gib = free / 2**30
        if free_gib >= float(min_free_gib):
            return
        print(
            f"EPCM_WAIT reason=low_gpu_memory stage={label} "
            f"free_gib={free_gib:.2f} required_gib={float(min_free_gib):.2f}",
            flush=True,
        )
        time.sleep(30.0)


def load_model_safe(model_path: Path, device: torch.device):
    """Load one worker model with a per-worker memory ceiling.

    Three independent workers share one A100.  The ceiling prevents one
    worker from claiming all remaining memory.  The model itself is unchanged.
    """
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    if device.type == "cuda":
        fraction = float(os.environ.get("EPCM_CUDA_MEMORY_FRACTION", "0.90"))
        if not 0.1 <= fraction <= 0.99:
            raise ValueError(f"invalid EPCM_CUDA_MEMORY_FRACTION={fraction}")
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    while True:
        wait_for_free_memory(22.0, "model_load")
        processor = None
        model = None
        try:
            processor = Qwen2_5OmniProcessor.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                trust_remote_code=True,
                local_files_only=True,
                device_map={"": 0} if device.type == "cuda" else None,
                max_memory={0: os.environ.get("EPCM_MAX_MEMORY", "72GiB"), "cpu": "400GiB"} if device.type == "cuda" else None,
                low_cpu_mem_usage=True,
            ).eval()
            return processor, model
        except torch.cuda.OutOfMemoryError as exc:
            print(f"EPCM_WAIT reason=oom_during_model_load error={exc}", flush=True)
            del processor, model
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(30.0)


def cache_path(output: Path, item_id: str) -> Path:
    return output / "cache" / f"{item_id}.npz"


def cache_valid(
    path: Path,
    signatures: str | set[str],
    required: tuple[str, ...] = ("clinical", "answer", "context", "visual"),
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if any(name not in archive.files for name in required):
                return False
            if any(not np.isfinite(np.asarray(archive[name])).all() for name in required):
                return False
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("schema") != SCHEMA:
                return False
            # A matching encoder signature is necessary but not sufficient:
            # old caches created before the visual-source contract must never
            # re-enter a new experiment through resume logic.
            if metadata.get("dataset") != "meld" and required and "visual" in required:
                visual_modality_contract.assert_visual_metadata(metadata)
                visual = np.asarray(archive["visual"])
                storage = metadata.get("visual_storage_contract", {})
                if int(storage.get("stored_feature_dim", visual.shape[-1])) != MODEL_VISUAL_SIZE:
                    return False
                if visual.ndim != 2 or visual.shape[-1] != MODEL_VISUAL_SIZE:
                    return False
            accepted = {signatures} if isinstance(signatures, str) else set(signatures)
            # A matching prompt is insufficient: pooling, media boundaries,
            # batching, or hidden-state selection may still differ.  Strict
            # signature matching prevents mixed implementations in one cache.
            return metadata.get("run_signature") in accepted
    except Exception:
        return False


def _turn_meta(turn: dict[str, Any], index: int, history_available: bool) -> dict[str, Any]:
    return {
        "index": int(turn.get("index", index)),
        "start": float(turn["start"]),
        "stop": float(turn["stop"]),
        "question_start": float(turn.get("question_start", 0.0)),
        "question_stop": float(turn.get("question_stop", 0.0)),
        "latency": float(turn.get("latency", 0.0)),
        "question_available": bool(turn.get("question_available", False)),
        "history_available": bool(history_available),
        "source_boundary_hash": sha256_text(json.dumps({
            "start": float(turn["start"]),
            "stop": float(turn["stop"]),
            "question_start": float(turn.get("question_start", 0.0)),
            "question_stop": float(turn.get("question_stop", 0.0)),
        }, sort_keys=True, separators=(",", ":"))),
    }


def _clinical_messages(dataset: str, turn: dict[str, Any], history: str, clip: Path, view: str):
    question_available = bool(turn.get("question_available", False))
    return prompts.build_messages(
        view=view,
        question=clean(turn.get("question")),
        history=history,
        answer=clean(turn.get("answer")),
        question_available=question_available,
        history_available=bool(history),
        # Context-only must not see the current response audio or transcript.
        audio_path=str(clip) if view != "context" else None,
        video_payload=None,
        start_seconds=float(turn["start"]),
        stop_seconds=float(turn["stop"]),
    )


def _meld_messages(row: dict[str, Any], context: str, audio: Path, video: Any, view: str):
    speaker = clean(row.get("speaker", "speaker")) or "speaker"
    text = clean(row.get("text", ""))
    if view == "context":
        return prompts.build_messages(
            view="context",
            question="",
            history=context,
            question_available=False,
            history_available=bool(context),
            audio_path=None,
            video_payload=None,
        )
    prompt_view = "source_dialogue" if view == "clinical" else view
    return prompts.build_messages(
        view=prompt_view,
        question="",
        history=context,
        answer=text,
        question_available=False,
        history_available=bool(context),
        audio_path=str(audio),
        video_payload=video,
    )


@torch.inference_mode()
def encode_batch_low_memory(
    processor: Any,
    model: Any,
    batch: list[list[dict[str, Any]]],
    device: torch.device,
) -> np.ndarray:
    """Match the anchor encoder while avoiding a four-layer float32 stack."""
    rendered, audios, images, videos = [], [], [], []
    for item in batch:
        rendered.append(
            processor.apply_chat_template(
                item, tokenize=False, add_generation_prompt=False
            )
        )
        aa, ii, vv = base.process_mm_info(item, use_audio_in_video=False)
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
        audio_kwargs={"max_length": base.AUDIO_MAX_SAMPLES},
    )
    inputs = base.move_inputs(inputs, device)
    output = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = output.hidden_states[-4].float()
    for layer in output.hidden_states[-3:]:
        current = layer.float()
        hidden.add_(current)
        del current
    hidden.mul_(0.25)
    mask = inputs.get("attention_mask")
    if mask is None:
        pooled = hidden.mean(dim=1)
    else:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    value = pooled.detach().cpu().numpy().astype(np.float32)
    del output, hidden, pooled, inputs
    return value


def _encode_once(
    processor: Any,
    model: Any,
    device: torch.device,
    batch: list[list[dict[str, Any]]],
) -> tuple[np.ndarray | None, str | None]:
    """Return after the OOM exception scope has been destroyed."""
    try:
        return encode_batch_low_memory(processor, model, batch, device), None
    except torch.cuda.OutOfMemoryError as exc:
        return None, str(exc)


def encode_resilient(
    processor: Any,
    model: Any,
    device: torch.device,
    batch: list[list[dict[str, Any]]],
    label: str,
) -> np.ndarray:
    """Encode with adaptive splitting and indefinite transient retries."""
    while True:
        wait_for_free_memory(3.0, label)
        value, error_text = _encode_once(processor, model, device, batch)
        if value is not None:
            return value
        # Cleanup happens only after _encode_once has returned, so the failed
        # forward's traceback can no longer retain processor/model inputs.
        gc.collect()
        torch.cuda.empty_cache()
        if len(batch) > 1:
            midpoint = max(1, len(batch) // 2)
            print(
                f"EPCM_REDUCE stage={label} batch={len(batch)} "
                f"left={midpoint} right={len(batch) - midpoint}",
                flush=True,
            )
            left = encode_resilient(
                processor, model, device, batch[:midpoint], f"{label}:left"
            )
            right = encode_resilient(
                processor, model, device, batch[midpoint:], f"{label}:right"
            )
            return np.concatenate((left, right), axis=0)
        print(f"EPCM_WAIT reason=oom_during_forward stage={label} error={error_text}", flush=True)
        time.sleep(30.0)


def extract_views(processor: Any, model: Any, device: torch.device, messages_by_view: dict[str, list[dict[str, Any]]]) -> dict[str, np.ndarray]:
    result = {}
    for view in ("clinical", "answer", "context"):
        result[view] = base.finite(encode_resilient(processor, model, device, [messages_by_view[view]], view)[0])
    return result


def load_clinical_rows(args: argparse.Namespace, dataset: str) -> list[dict[str, Any]]:
    target = "depression"
    frame, spec = base.load_label_profile(
        args.contract_dir, dataset, "depression_phq8_screening", include_unlabeled=False
    )
    rows = []
    for item in frame.to_dict("records"):
        pid = str(int(item["pid"]))
        rows.append({
            "item_id": pid,
            "pid": pid,
            "split": str(item["split"]),
            "canonical_split": str(item["canonical_split"]),
            "label": int(item[spec.target_column]),
            "target": target,
        })
    return rows


def load_meld_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = [row for row in rows if str(row.get("dataset", "")).upper() == "MELD" and Path(str(row.get("video_path", ""))).is_file()]
    result.sort(key=lambda row: (str(row.get("split", "")), str(row.get("group", "")), int(str(row.get("sample_id", "utt0")).split("utt")[-1])))
    for row in result:
        row["item_id"] = str(row.get("cache_key") or sha256_text(f"MELD|{row.get('split')}|{row.get('sample_id')}"))[:24]
    return result


def extract_clinical(args: argparse.Namespace, dataset: str, rows: list[dict[str, Any]], processor: Any, model: Any, device: torch.device, signature: str) -> dict[str, int]:
    root = args.daic_root if dataset == "daic" else args.edaic_root
    clips_root = args.output / "clips"
    counts = {"written": 0, "cached": 0, "failed": 0}
    failures = []
    for number, row in enumerate(rows, 1):
        destination = cache_path(args.output, row["item_id"])
        if cache_valid(destination, signature, ("clinical", "answer", "context", "visual")):
            counts["cached"] += 1
            continue
        try:
            audio = base.source_file(dataset, root, row["pid"], "AUDIO.wav")
            turns = base.read_turns(dataset, root, row["pid"])
            if not audio.is_file() or not turns:
                raise RuntimeError(f"missing audio or turns audio={audio} turns={len(turns)}")
            clips = []
            for turn in turns:
                key = sha256_text(f"{dataset}|{row['pid']}|{turn['index']}|{turn['start']}|{turn['stop']}")[:20]
                clips.append(base.write_clip(audio, turn["start"], turn["stop"], clips_root / f"{row['pid']}_{key}.wav", args.pad, args.max_seconds))
            visual, visual_audit = base.visual_event_features(dataset, root, row["pid"], turns)
            visual = base.finite(visual)
            if visual.ndim != 2 or visual.shape[1] > MODEL_VISUAL_SIZE:
                raise RuntimeError(
                    f"visual feature width exceeds model contract: {dataset}/{row['pid']} {visual.shape}"
                )
            native_visual_dim = int(visual.shape[1])
            if native_visual_dim < MODEL_VISUAL_SIZE:
                visual = np.pad(
                    visual,
                    ((0, 0), (0, MODEL_VISUAL_SIZE - native_visual_dim)),
                    mode="constant",
                )
            visual_audit = {
                **visual_audit,
                "native_feature_dim": native_visual_dim,
                "stored_feature_dim": int(visual.shape[1]),
                "right_zero_padding": int(MODEL_VISUAL_SIZE - native_visual_dim),
            }
            vectors = {"clinical": [], "answer": [], "context": []}
            turn_metadata = []
            for index, (turn, clip) in enumerate(zip(turns, clips)):
                history = history_text(turns, index)
                for view in vectors:
                    messages = _clinical_messages(dataset, turn, history, clip, view)
                    vectors[view].append(encode_resilient(processor, model, device, [messages], f"{dataset}:{view}")[0])
                turn_metadata.append(_turn_meta(turn, index, bool(history)))
                if number == 1 or index == len(turns) - 1:
                    print(f"EPCM_TURN {dataset} pid={row['pid']} turn={index + 1}/{len(turns)}", flush=True)
            metadata = {
                "schema": SCHEMA,
                "run_signature": signature,
                "dataset": dataset,
                "item_id": row["item_id"],
                "pid": row["pid"],
                "split": row["split"],
                "canonical_split": row["canonical_split"],
                "target_inventory": row["target"],
                "view_contract": prompts.contract(),
                "history_turns": HISTORY_TURNS,
                "max_history_chars": MAX_HISTORY_CHARS,
                "turns": turn_metadata,
                "visual_audit": visual_audit,
                "visual_modality_contract": visual_modality_contract.SCHEMA,
                "visual_modality_contract_sha256": visual_modality_contract.contract_hash(),
                "visual_storage_contract": {
                    "model_input_dim": MODEL_VISUAL_SIZE,
                    "native_feature_dim": native_visual_dim,
                    "stored_feature_dim": MODEL_VISUAL_SIZE,
                    "right_zero_padding": MODEL_VISUAL_SIZE - native_visual_dim,
                    "padding_policy": "right_zero_pad_only; no source feature dropped",
                },
                "raw_input_fingerprints": [file_fingerprint(audio)],
                "raw_inputs_read_only": True,
            }
            visual_modality_contract.assert_visual_metadata(metadata)
            atomic_npz(destination, {
                "clinical": base.finite(np.stack(vectors["clinical"])),
                "answer": base.finite(np.stack(vectors["answer"])),
                "context": base.finite(np.stack(vectors["context"])),
                "visual": visual,
                "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
            })
            counts["written"] += 1
            print(f"EPCM_CACHE {dataset} {number}/{len(rows)} pid={row['pid']} turns={len(turns)}", flush=True)
        except Exception as exc:
            counts["failed"] += 1
            item = {"dataset": dataset, "pid": row["pid"], "error": f"{type(exc).__name__}: {exc}"}
            failures.append(item)
            atomic_json(args.output / "failures.json", failures)
            print(f"EPCM_FAIL {json.dumps(item, ensure_ascii=False)}", flush=True)
            if "out of memory" in str(exc).lower():
                raise
    if failures:
        raise RuntimeError(f"{dataset}: failed rows={len(failures)}")
    return counts


def extract_meld(args: argparse.Namespace, rows: list[dict[str, Any]], processor: Any, model: Any, device: torch.device, signature: str) -> dict[str, int]:
    counts = {"written": 0, "cached": 0, "failed": 0}
    failures = []
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(f"{row.get('split', '')}:{row.get('group', '')}", []).append(row)
    pending: list[dict[str, Any]] = []

    def flush_pending() -> None:
        if not pending:
            return
        # Batch only within the same view. This preserves the exact three-view
        # contract while avoiding one model forward per sample and view.
        encoded: dict[str, np.ndarray] = {}
        for view in ("clinical", "answer", "context"):
            batch = [item["messages"][view] for item in pending]
            encoded[view] = encode_resilient(
                processor, model, device, batch, f"meld:{view}:batch{len(pending)}"
            )
        for index, item in enumerate(pending):
            vectors = {view: encoded[view][index] for view in ("clinical", "answer", "context")}
            atomic_npz(item["destination"], {
                "source_dialogue": vectors["clinical"][None, :],
                "answer": vectors["answer"][None, :],
                "context": vectors["context"][None, :],
                "visual": np.zeros((1, 0), dtype=np.float32),
                "metadata_json": np.asarray(json.dumps(item["metadata"], ensure_ascii=False, sort_keys=True)),
            })
            counts["written"] += 1
            if item["ordinal"] == 1 or item["ordinal"] % 100 == 0 or item["ordinal"] == len(rows):
                print(f"EPCM_CACHE meld {item['ordinal']}/{len(rows)} item={item['row']['item_id']} batch={len(pending)}", flush=True)
        pending.clear()

    for ordinal, row in enumerate(rows, 1):
        destination = cache_path(args.output, row["item_id"])
        if cache_valid(destination, signature, ("source_dialogue", "answer", "context", "visual")):
            counts["cached"] += 1
            continue
        try:
            source = Path(str(row["video_path"]))
            audio = base.materialize_meld_audio(source, args.output / "audio" / f"{row['item_id']}.wav")
            video = base.sample_meld_video(source, 8)
            current = int(str(row.get("sample_id", "utt0")).split("utt")[-1])
            group = f"{row.get('split', '')}:{row.get('group', '')}"
            previous = [item for item in by_group[group] if int(str(item.get("sample_id", "utt0")).split("utt")[-1]) < current][-HISTORY_TURNS:]
            history = "\n".join(f"{clean(item.get('speaker', 'speaker'))}: {clean(item.get('text', ''))}" for item in previous)
            metadata = {
                "schema": SCHEMA,
                "run_signature": signature,
                "dataset": "meld",
                "item_id": row["item_id"],
                "split": str(row.get("split", "")),
                "dialogue_group": str(row.get("group", "")),
                "source": {
                    "sample_id": str(row.get("sample_id", "")),
                    "speaker": str(row.get("speaker", "")),
                    "video_path": str(row.get("video_path", "")),
                },
                "view_contract": prompts.contract(),
                "history_turns": min(HISTORY_TURNS, len(previous)),
                "raw_input_fingerprints": [file_fingerprint(source), file_fingerprint(audio)],
                "raw_inputs_read_only": True,
            }
            pending.append({
                "row": row,
                "ordinal": ordinal,
                "destination": destination,
                "metadata": metadata,
                "messages": {
                    view: _meld_messages(row, history, audio, video, view)
                    for view in ("clinical", "answer", "context")
                },
            })
            if len(pending) >= max(1, int(args.batch_size)):
                flush_pending()
        except Exception as exc:
            counts["failed"] += 1
            item = {"dataset": "meld", "item_id": row["item_id"], "error": f"{type(exc).__name__}: {exc}"}
            failures.append(item)
            atomic_json(args.output / "failures.json", failures)
            print(f"EPCM_FAIL {json.dumps(item, ensure_ascii=False)}", flush=True)
            if "out of memory" in str(exc).lower():
                raise
    flush_pending()
    if failures:
        raise RuntimeError(f"MELD: failed rows={len(failures)}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("daic", "edaic", "meld"))
    parser.add_argument("--daic-root", type=Path, required=True)
    parser.add_argument("--edaic-root", type=Path, required=True)
    parser.add_argument("--meld-source-rows", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pad", type=float, default=0.4)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    args = parser.parse_args()
    base.configure_threads(min(int(args.threads), 32))
    base.validate_contract(args.contract_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("EPCM-MH requires CUDA")
    wait_for_free_memory(22.0, "startup")
    config = {
        "schema": SCHEMA,
        "dataset": args.dataset,
        "model": str(args.model),
        "model_files_fingerprint_sha256": model_files_fingerprint(args.model),
        "view_contract": prompts.contract(),
        "history_turns": HISTORY_TURNS,
        "max_history_chars": MAX_HISTORY_CHARS,
        "batch_size": int(args.batch_size),
        "threads": min(int(args.threads), 32),
        "raw_inputs_read_only": True,
    }
    config["encoder_implementation_sha256"] = encoder_implementation_fingerprint()
    # The implementation fingerprint must be part of the signature itself.
    # Recording it after hashing would allow incompatible encoders to share a
    # run signature and would make interruption/resume provenance ambiguous.
    signature = sha256_text(json.dumps(config, sort_keys=True, separators=(",", ":")))
    config["run_signature"] = signature
    atomic_json(args.output / "RUN_CONTRACT.json", config)
    processor, model = load_model_safe(args.model, device)
    started = time.time()
    atomic_json(args.output / "PROGRESS.json", {"status": "running", "dataset": args.dataset, "run_signature": signature})
    if args.dataset == "meld":
        rows = load_meld_rows(args.meld_source_rows)
        counters = extract_meld(args, rows, processor, model, device, signature)
        total_items = len(rows)
    else:
        rows = load_clinical_rows(args, args.dataset)
        counters = extract_clinical(args, args.dataset, rows, processor, model, device, signature)
        total_items = len(rows)
    if counters["failed"]:
        raise RuntimeError(f"A-DyME failed: {counters}")
    final = {"status": "complete", "schema": SCHEMA, "dataset": args.dataset, "total_items": total_items, "counters": counters, "elapsed_s": time.time() - started, "run_signature": signature, "raw_inputs_read_only": True}
    atomic_json(args.output / "FINAL_CACHE.json", final)
    atomic_json(args.output / "PROGRESS.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
