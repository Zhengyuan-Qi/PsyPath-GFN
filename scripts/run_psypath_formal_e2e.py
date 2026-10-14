#!/usr/bin/env python3
"""Strict task-specific online training above a formal PsyPath-GFN anchor."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.psypath_formal_e2e_model import FormalPsyPathEndToEndModel
from scripts.psypath_online_model import FrozenPsyPathAnchor
from scripts.run_meter_mh_e2e import (
    adapter_state,
    encode_event_views,
    frozen_participant_views,
    load_backbone,
    load_task_rows,
    materialize,
    restore_adapter,
    set_backbone_training_mode,
)
from scripts.validate_alternating_psychometric_gfn_cached import (
    stable_seed,
    temporal_features as formal_temporal_features,
)


SCHEMA = "psypath_formal_e2e_runner_v1"
INSTRUMENT_ID = {
    "daic_depression": 0,
    "edaic_depression": 0,
    "edaic_ptsd": 1,
}
TASKS = {
    "daic_depression": {
        "dataset": "daic",
        "binary_profile": "depression_phq8_screening",
        "score_profile": "depression_phq_score",
    },
    "edaic_depression": {
        "dataset": "edaic",
        "binary_profile": "depression_phq8_screening",
        "score_profile": "depression_phq_score",
    },
    "edaic_ptsd": {
        "dataset": "edaic",
        "binary_profile": "ptsd_dsm_iv_binary",
        "score_profile": "ptsd_score_repaired",
    },
}
SCORE_MAX = {
    "daic_depression": 24.0,
    "edaic_depression": 24.0,
    "edaic_ptsd": 85.0,
}
EXPECTED_SPLITS = {
    "daic": {"train": 107, "dev": 35, "test": 47},
    "edaic": {"train": 163, "dev": 56, "test": 56},
}


def configure_runtime(threads: int) -> None:
    threads = max(1, min(32, int(threads)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_hashes() -> dict[str, str]:
    files = (
        "scripts/run_psypath_formal_e2e.py",
        "scripts/psypath_formal_e2e_model.py",
        "scripts/psypath_online_model.py",
        "scripts/meter_gfn.py",
        "scripts/run_meter_mh_e2e.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in files}


def protocol_sha256(args: argparse.Namespace) -> str | None:
    return sha256_file(args.protocol) if args.protocol is not None else None


def validate_runtime_protocol(
    protocol: Mapping[str, Any], args: argparse.Namespace
) -> None:
    optimization = protocol.get("optimization")
    if not isinstance(optimization, Mapping):
        raise ValueError("formal PsyPath protocol lacks optimization settings")
    expected_seeds = {int(seed) for seed in optimization.get("development_seeds", [])}
    if int(args.seed) not in expected_seeds:
        raise ValueError(
            f"seed {args.seed} is not preregistered; expected {sorted(expected_seeds)}"
        )
    integer_contract = {
        "reward_folds": args.reward_folds,
        "train_path_samples": args.train_samples,
        "development_path_samples": args.development_samples,
        "test_path_samples": args.test_samples,
        "event_batch_size": args.event_batch_size,
        "gradient_accumulation_participants": args.gradient_accumulation_participants,
        "checkpoint_every_participants": args.checkpoint_every,
        "max_events_per_participant": args.max_events_per_participant,
    }
    if args.stage == "frozen":
        integer_contract["epochs_frozen"] = args.epochs
    elif args.stage == "lora":
        integer_contract["epochs_lora"] = args.epochs
    failures = []
    for key, actual in integer_contract.items():
        if key not in optimization:
            failures.append(f"missing protocol field optimization.{key}")
        elif int(actual) != int(optimization[key]):
            failures.append(
                f"{key}: runtime={actual}, protocol={optimization[key]}"
            )
    float_contract = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "reward_temperature": args.reward_temperature,
    }
    for key, actual in float_contract.items():
        if key not in optimization:
            failures.append(f"missing protocol field optimization.{key}")
        elif not math.isclose(
            float(actual), float(optimization[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            failures.append(
                f"{key}: runtime={actual}, protocol={optimization[key]}"
            )
    if args.event_batch_size != 1 or args.vectorize_event_views:
        failures.append(
            "strict formal encoding requires event_batch_size=1 and "
            "vectorize_event_views=false"
        )
    if failures:
        raise ValueError("formal PsyPath runtime/protocol mismatch: " + "; ".join(failures))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def metric_block(labels: Iterable[int], probability: Iterable[float], threshold: float) -> dict[str, Any]:
    labels = np.asarray(list(labels), dtype=np.int64)
    probability = np.clip(np.asarray(list(probability), dtype=np.float64), 1e-6, 1.0 - 1e-6)
    prediction = (probability >= float(threshold)).astype(np.int64)
    result = {
        "n": int(len(labels)),
        "threshold": float(threshold),
        "weighted_f1": float(f1_score(labels, prediction, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(labels, prediction, zero_division=0)),
        "brier": float(np.mean(np.square(probability - labels))),
    }
    result["auroc"] = float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else None
    return result


def choose_threshold(labels: list[int], probability: list[float]) -> float:
    choices = []
    for threshold in np.arange(0.15, 0.851, 0.025):
        metrics = metric_block(labels, probability, float(threshold))
        choices.append((metrics["weighted_f1"], metrics["macro_f1"], metrics.get("auroc") or 0.0, -metrics["brier"], -abs(float(threshold) - 0.5), float(threshold)))
    return float(max(choices)[-1])


def severity_block(rows: list[dict[str, Any]], prediction: list[float]) -> dict[str, Any]:
    pairs = [(float(row["score"]), float(value)) for row, value in zip(rows, prediction) if row.get("score") is not None]
    if len(pairs) < 2:
        return {"n": len(pairs), "ccc": None, "mae": None, "rmse": None}
    target = np.asarray([item[0] for item in pairs], dtype=np.float64)
    predicted = np.asarray([item[1] for item in pairs], dtype=np.float64)
    covariance = float(np.mean((target - target.mean()) * (predicted - predicted.mean())))
    denominator = float(target.var() + predicted.var() + (target.mean() - predicted.mean()) ** 2)
    return {
        "n": len(pairs),
        "ccc": float(2.0 * covariance / denominator) if denominator > 1e-12 else None,
        "mae": float(mean_absolute_error(target, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(target, predicted))),
    }


@contextlib.contextmanager
def adapter_disabled(backbone: torch.nn.Module):
    method = getattr(backbone, "disable_adapter", None)
    with (method() if callable(method) else contextlib.nullcontext()):
        yield


def checkpoint_path(formal_root: Path, task: str, seed: int, test: bool = False) -> Path:
    group = "checkpoints" if test else "development_checkpoints"
    return formal_root / task / group / task / "gflownet" / f"seed_{seed}.pt"


def load_anchor(args: argparse.Namespace) -> tuple[FrozenPsyPathAnchor, dict[str, Any]]:
    path = checkpoint_path(args.formal_root, args.task, args.seed, test=False)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # The locked formal suite sampled its discrete evidence paths on CPU.
    # Torch CPU and CUDA generators do not produce the same categorical draws
    # from one seed, so moving the frozen anchor to CUDA changes the model.
    anchor = FrozenPsyPathAnchor.from_checkpoint(
        payload, instrument_id=INSTRUMENT_ID[args.task]
    ).cpu()
    return anchor, payload


def restore_trainable_state(
    model: FormalPsyPathEndToEndModel,
    state: Mapping[str, torch.Tensor],
) -> None:
    current = model.state_dict()
    filtered = {name: value for name, value in state.items() if not name.startswith("anchor.")}
    unexpected = sorted(set(filtered) - set(current))
    missing = sorted(
        name for name in current if not name.startswith("anchor.") and name not in filtered
    )
    if unexpected or missing:
        raise RuntimeError(
            f"formal trainable state mismatch missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    current.update(filtered)
    model.load_state_dict(current, strict=True)


def participant_inputs(
    processor: Any,
    backbone: torch.nn.Module,
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    train_lora: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor] | None, torch.Tensor]:
    device = torch.device("cuda:0")
    online_row = {**row, "key": f"{args.task}:{row['pid']}", "dataset": TASKS[args.task]["dataset"]}
    participant = materialize(online_row, args)
    with adapter_disabled(backbone):
        base_views = frozen_participant_views(
            processor,
            backbone,
            participant,
            device,
            vectorize_views=args.vectorize_event_views,
            # The locked formal cache was encoded one event at a time. Batch
            # padding changes HumanOmniV2 hidden states, so the non-degradation
            # anchor must retain event batch 1 exactly.
            event_batch_size=1,
        )
    visual = participant["visual"].to(device=device, dtype=torch.float64)
    base_raw = {
        "clinical": base_views[0].to(torch.float64),
        "answer": base_views[1].to(torch.float64),
        "context": base_views[2].to(torch.float64),
        "visual": visual,
    }
    adapted_raw = None
    if train_lora:
        adapted_views = frozen_participant_views(
            processor,
            backbone,
            participant,
            device,
            vectorize_views=args.vectorize_event_views,
            # Batched BF16 HumanOmniV2 forward passes are not representation
            # equivalent to the locked one-event encoder because sequence
            # padding changes hidden states. Keep the formal path exact.
            event_batch_size=1,
        )
        adapted_raw = {
            "clinical": adapted_views[0].detach().to(torch.float64).requires_grad_(True),
            "answer": adapted_views[1].detach().to(torch.float64).requires_grad_(True),
            "context": adapted_views[2].detach().to(torch.float64).requires_grad_(True),
            "visual": visual,
        }
    temporal = torch.as_tensor(
        formal_temporal_features(participant["turns"]),
        dtype=torch.float32,
        device=device,
    )
    return participant, base_raw, adapted_raw, temporal


def load_official_rows(
    contract_dir: Path,
    task: str,
    split: str,
) -> list[dict[str, Any]]:
    rows = load_task_rows(contract_dir, task, {split})
    expected = EXPECTED_SPLITS[TASKS[task]["dataset"]][split]
    if len(rows) != expected:
        raise RuntimeError(
            f"official {task}/{split} count mismatch: {len(rows)} != {expected}"
        )
    if any(row["split"] != split for row in rows):
        raise RuntimeError(f"cross-split participant in {task}/{split}")
    return rows


def formal_posterior_seed(payload: Mapping[str, Any], task: str, split: str) -> int:
    seed = int(payload["seed"])
    if split == "dev":
        evaluation_seed = stable_seed(task, seed, int(payload["selected_cycle"]) - 1, int(payload["selected_epoch"]), "dev")
    elif split == "test":
        evaluation_seed = stable_seed(task, seed, "test")
    else:
        raise ValueError(split)
    return stable_seed(evaluation_seed, "posterior")


def path_seeds(base: int, pid: str, samples: int) -> list[int]:
    return [stable_seed(base, pid, sample) for sample in range(int(samples))]


def evaluate(
    processor: Any,
    backbone: torch.nn.Module,
    model: FormalPsyPathEndToEndModel,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    split: str,
    samples: int,
    threshold: float | None,
    anchor_threshold: float | None,
    formal_payload: Mapping[str, Any],
) -> dict[str, Any]:
    model.eval()
    predictions = []
    anchor_base = formal_posterior_seed(formal_payload, args.task, split)
    with torch.no_grad():
        for number, row in enumerate(rows, 1):
            participant, base_raw, adapted_raw, temporal = participant_inputs(
                processor, backbone, row, args, train_lora=args.train_lora
            )
            seeds = path_seeds(anchor_base, row["pid"], samples)
            policy = path_seeds(stable_seed(args.task, args.seed, split, "e2e"), row["pid"], samples)
            output = model(
                base_raw,
                temporal,
                samples=samples,
                policy_seeds=policy,
                anchor_seeds=seeds,
                adapted_raw=adapted_raw or base_raw,
            )
            predictions.append(
                {
                    "pid": row["pid"],
                    "label": row["label"],
                    "score": row.get("score"),
                    "probability": float(output["online_probability"]),
                    "severity": float(output["online_severity"]),
                    "anchor_probability": float(output["anchor_probability"]),
                    "anchor_severity": float(output["anchor_severity"]),
                    "path_probability_std": float(output["path_probability_std"]),
                    "path_severity_std": float(output["path_severity_std"]),
                }
            )
            del participant, base_raw, adapted_raw, temporal, output
            if number == 1 or number % 10 == 0 or number == len(rows):
                print(json.dumps({"event": "psypath_formal_e2e_eval", "task": args.task, "split": split, "completed": number, "total": len(rows)}), flush=True)
    labels = [row["label"] for row in predictions]
    probability = [row["probability"] for row in predictions]
    anchor_probability = [row["anchor_probability"] for row in predictions]
    threshold = choose_threshold(labels, probability) if threshold is None else float(threshold)
    anchor_threshold = choose_threshold(labels, anchor_probability) if anchor_threshold is None else float(anchor_threshold)
    return {
        "classification": metric_block(labels, probability, threshold),
        "anchor_classification": metric_block(labels, anchor_probability, anchor_threshold),
        "threshold": threshold,
        "anchor_threshold": anchor_threshold,
        "severity": severity_block(predictions, [row["severity"] for row in predictions]),
        "anchor_severity": severity_block(predictions, [row["anchor_severity"] for row in predictions]),
        "mean_path_probability_std": float(np.mean([row["path_probability_std"] for row in predictions])),
        "mean_path_severity_std": float(np.mean([row["path_severity_std"] for row in predictions])),
        "predictions": predictions,
    }


def fold_map(rows: list[dict[str, Any]], folds: int, seed: int) -> dict[str, int]:
    groups = np.asarray([row["pid"] for row in rows])
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = {}
    for fold, (_, held) in enumerate(splitter.split(groups, labels)):
        for index in held:
            result[str(groups[index])] = fold
    return result


def formal_development_reference(
    formal_root: Path,
    task: str,
    seed: int,
) -> dict[str, Any]:
    path = formal_root / task / "tasks" / f"{task}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    run = next(item for item in payload["runs"] if int(item["seed"]) == int(seed))
    return run["methods"]["gflownet"]["development"]


def save_state(
    path: Path,
    model: FormalPsyPathEndToEndModel,
    backbone: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    epoch: int,
    position: int,
    order: list[int],
    history: list[dict[str, Any]],
) -> None:
    atomic_torch(path, {
        "schema": SCHEMA,
        "stage": args.stage,
        "task": args.task,
        "seed": args.seed,
        "epoch": epoch,
        "position": position,
        "order": order,
        "history": history,
        "model": model.state_dict(),
        "adapter": adapter_state(backbone, args.train_lora),
        "optimizer": optimizer.state_dict(),
        "implementation_hashes": implementation_hashes(),
        "protocol_sha256": protocol_sha256(args),
    })


def train(args: argparse.Namespace) -> int:
    device = torch.device("cuda:0")
    train_rows = load_official_rows(args.contract_dir, args.task, "train")
    dev_rows = load_official_rows(args.contract_dir, args.task, "dev")
    anchor, formal_payload = load_anchor(args)
    model = FormalPsyPathEndToEndModel(anchor).move_trainable(device)
    processor, backbone, backbone_audit = load_backbone(args, train_lora=args.train_lora)
    initial = None
    if args.initial_state is not None:
        initial = torch.load(args.initial_state, map_location="cpu", weights_only=False)
        restore_trainable_state(model, initial["model"])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters += [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint = args.output / args.task / args.stage / f"seed_{args.seed}" / "checkpoint.pt"
    best_path = args.output / args.task / args.stage / f"seed_{args.seed}" / "best.pt"
    start_epoch = 0
    start_position = 0
    order = None
    history = []
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("implementation_hashes") != implementation_hashes():
            raise RuntimeError("formal E2E checkpoint implementation mismatch")
        if payload.get("protocol_sha256") != protocol_sha256(args):
            raise RuntimeError("formal E2E checkpoint protocol mismatch")
        restore_trainable_state(model, payload["model"])
        restore_adapter(backbone, payload.get("adapter"))
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"])
        start_position = int(payload["position"])
        order = list(payload["order"]) if payload.get("order") else None
        history = list(payload.get("history", []))

    anchor_base = formal_posterior_seed(formal_payload, args.task, "dev")
    if checkpoint.is_file():
        # Resumed states are no longer expected to equal the epoch-0 anchor.
        if start_epoch > args.epochs or (
            start_epoch == args.epochs and start_position != 0
        ):
            raise RuntimeError(
                f"invalid formal E2E resume point epoch={start_epoch} "
                f"position={start_position} epochs={args.epochs}"
            )
    elif args.initial_state is None:
        with torch.no_grad():
            row = dev_rows[0]
            participant, base_raw, adapted_raw, temporal = participant_inputs(
                processor, backbone, row, args, train_lora=args.train_lora
            )
            seeds = path_seeds(anchor_base, row["pid"], 2)
            output = model(
                base_raw,
                temporal,
                samples=2,
                policy_seeds=seeds,
                anchor_seeds=seeds,
                adapted_raw=adapted_raw or base_raw,
            )
            if abs(
                float(output["online_probability"] - output["anchor_probability"])
            ) > 2e-5:
                raise RuntimeError("formal E2E zero-impact initialization failed")
            del participant, base_raw, adapted_raw, temporal, output
    else:
        with torch.no_grad():
            row = dev_rows[0]
            participant, base_raw, adapted_raw, temporal = participant_inputs(
                processor, backbone, row, args, train_lora=True
            )
            if adapted_raw is None:
                raise RuntimeError("LoRA initialization lacks adapted representations")
            representation_error = max(
                float(torch.max(torch.abs(adapted_raw[name] - base_raw[name])))
                for name in ("clinical", "answer", "context")
            )
            if representation_error > 2e-5:
                raise RuntimeError(
                    "zero-initialized LoRA changed the frozen representation: "
                    f"max_abs_error={representation_error}"
                )
            del participant, base_raw, adapted_raw, temporal

    best_key = None
    if best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        best_key = tuple(best["selection_key"])
    if best_key is None:
        baseline = evaluate(processor, backbone, model, dev_rows, args, split="dev", samples=args.development_samples, threshold=None, anchor_threshold=None, formal_payload=formal_payload)
        formal_reference = formal_development_reference(
            args.formal_root, args.task, args.seed
        )
        observed_pids = [row["pid"] for row in baseline["predictions"]]
        if observed_pids != [str(pid) for pid in formal_reference["pids"]]:
            raise RuntimeError("formal development PID order changed before training")
        formal_probability = list(
            map(float, formal_reference["probability"]["posterior"])
        )
        probability_error = max(
            abs(row["anchor_probability"] - expected)
            for row, expected in zip(baseline["predictions"], formal_probability)
        )
        if probability_error > 2e-5:
            raise RuntimeError(
                f"epoch-0 formal anchor mismatch: max_probability_error={probability_error}"
            )
        formal_threshold = float(formal_reference["threshold"])
        baseline["anchor_threshold"] = formal_threshold
        baseline["anchor_classification"] = metric_block(
            [row["label"] for row in baseline["predictions"]],
            [row["anchor_probability"] for row in baseline["predictions"]],
            formal_threshold,
        )
        baseline["formal_anchor_max_probability_abs_error"] = probability_error
        if abs(
            baseline["anchor_classification"]["weighted_f1"]
            - float(formal_reference["posterior"]["weighted_f1"])
        ) > 1e-12:
            raise RuntimeError("epoch-0 formal anchor F1 does not reproduce the locked run")
        best_key = (float(baseline["classification"]["weighted_f1"]), float(baseline["classification"].get("auroc") or 0.0))
        atomic_torch(best_path, {
            "schema": SCHEMA,
            "stage": args.stage,
            "task": args.task,
            "seed": args.seed,
            "epoch": 0,
            "threshold": baseline["threshold"],
            "anchor_threshold": baseline["anchor_threshold"],
            "development": baseline,
            "model": copy.deepcopy(model.state_dict()),
            "adapter": adapter_state(backbone, args.train_lora),
            "selection_key": best_key,
            "selection": "exact_formal_anchor",
            "implementation_hashes": implementation_hashes(),
            "protocol_sha256": protocol_sha256(args),
        })

    folds = fold_map(train_rows, args.reward_folds, args.seed)
    set_backbone_training_mode(backbone, args.train_lora)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        model.anchor.eval()
        if order is None or epoch != start_epoch:
            order = list(range(len(train_rows)))
            random.Random(stable_seed(args.task, args.stage, args.seed, epoch)).shuffle(order)
            start_position = 0
        losses = []
        for position in range(start_position, len(order)):
            row = train_rows[order[position]]
            participant, base_raw, adapted_raw, temporal = participant_inputs(processor, backbone, row, args, train_lora=args.train_lora)
            train_base = stable_seed(args.task, args.stage, args.seed, epoch, row["pid"], folds[row["pid"]])
            seeds = path_seeds(train_base, row["pid"], args.train_samples)
            output = model(base_raw, temporal, samples=args.train_samples, policy_seeds=seeds, anchor_seeds=seeds, adapted_raw=adapted_raw or base_raw)
            loss, _ = model.training_loss(output, label=float(row["label"]), score=row.get("score"), score_max=SCORE_MAX[args.task], reward_temperature=args.reward_temperature)
            (loss / float(args.gradient_accumulation_participants)).backward()
            if args.train_lora and adapted_raw is not None:
                gradients = tuple(
                    adapted_raw[name].grad.detach().to(torch.float32) if adapted_raw[name].grad is not None else torch.zeros_like(adapted_raw[name], dtype=torch.float32)
                    for name in ("clinical", "answer", "context")
                )
                for event in range(len(participant["turns"])):
                    live = encode_event_views(
                        processor,
                        backbone,
                        participant,
                        event,
                        device,
                        vectorize_views=False,
                    )
                    torch.autograd.backward(
                        live, tuple(gradient[event] for gradient in gradients)
                    )
            losses.append(float(loss.detach().cpu()))
            step = position + 1
            if step % args.gradient_accumulation_participants == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % args.checkpoint_every == 0 or step == len(order):
                save_state(checkpoint, model, backbone, optimizer, args, epoch=epoch, position=step, order=order, history=history)
            print(json.dumps({"event": "psypath_formal_e2e_train", "task": args.task, "stage": args.stage, "seed": args.seed, "epoch": epoch + 1, "position": step, "total": len(order), "loss": losses[-1]}), flush=True)
            del participant, base_raw, adapted_raw, temporal, output, loss
        order = None
        start_position = 0
        development = evaluate(processor, backbone, model, dev_rows, args, split="dev", samples=args.development_samples, threshold=None, anchor_threshold=None, formal_payload=formal_payload)
        key = (float(development["classification"]["weighted_f1"]), float(development["classification"].get("auroc") or 0.0))
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses)), "development": development})
        if key > best_key:
            best_key = key
            atomic_torch(best_path, {
                "schema": SCHEMA,
                "stage": args.stage,
                "task": args.task,
                "seed": args.seed,
                "epoch": epoch + 1,
                "threshold": development["threshold"],
                "anchor_threshold": development["anchor_threshold"],
                "development": development,
                "model": copy.deepcopy(model.state_dict()),
                "adapter": adapter_state(backbone, args.train_lora),
                "selection_key": best_key,
                "selection": "trained_epoch",
                "implementation_hashes": implementation_hashes(),
                "protocol_sha256": protocol_sha256(args),
            })
        save_state(checkpoint, model, backbone, optimizer, args, epoch=epoch + 1, position=0, order=[], history=history)
        atomic_json(args.output / args.task / args.stage / f"seed_{args.seed}" / "PROGRESS.json", {"schema": SCHEMA, "status": "training", "task": args.task, "stage": args.stage, "seed": args.seed, "history": history, "backbone": backbone_audit, "test_labels_loaded": False})

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    atomic_json(args.output / args.task / args.stage / f"seed_{args.seed}" / "LOCK.json", {
        "schema": SCHEMA,
        "status": "locked_before_test_read",
        "task": args.task,
        "stage": args.stage,
        "seed": args.seed,
        "selected_epoch": best["epoch"],
        "selection": best["selection"],
        "threshold": best["threshold"],
        "anchor_threshold": best["anchor_threshold"],
        "development": best["development"],
        "checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": sha256_file(best_path),
        "test_labels_loaded": False,
    })
    return 0


def locked_test(args: argparse.Namespace) -> int:
    payload = torch.load(args.locked_state, map_location="cpu", weights_only=False)
    if payload.get("task") != args.task or int(payload.get("seed", -1)) != args.seed:
        raise RuntimeError("locked state identity mismatch")
    device = torch.device("cuda:0")
    anchor, formal_payload = load_anchor(args)
    model = FormalPsyPathEndToEndModel(anchor).move_trainable(device)
    processor, backbone, backbone_audit = load_backbone(args, train_lora=payload.get("adapter") is not None)
    restore_trainable_state(model, payload["model"])
    restore_adapter(backbone, payload.get("adapter"))
    args.train_lora = payload.get("adapter") is not None
    test_rows = load_official_rows(args.contract_dir, args.task, "test")
    result = evaluate(processor, backbone, model, test_rows, args, split="test", samples=args.test_samples, threshold=float(payload["threshold"]), anchor_threshold=float(payload["anchor_threshold"]), formal_payload=formal_payload)
    output = {
        "schema": SCHEMA,
        "status": "complete",
        "task": args.task,
        "stage": "test",
        "seed": args.seed,
        "selected_development_state": str(args.locked_state.resolve()),
        "selected_development_state_sha256": sha256_file(args.locked_state),
        "selected_epoch": payload.get("epoch"),
        "selection": payload.get("selection"),
        "threshold_from_dev": payload["threshold"],
        "anchor_threshold_from_dev": payload["anchor_threshold"],
        "result": result,
        "backbone": backbone_audit,
        "test_labels_loaded": True,
    }
    atomic_json(args.output / args.task / "test" / f"seed_{args.seed}" / "FINAL_METRICS.json", output)
    print(json.dumps(output, default=json_default), flush=True)
    return 0


def anchor_audit(args: argparse.Namespace) -> int:
    device = torch.device("cuda:0")
    dev_rows = load_official_rows(args.contract_dir, args.task, "dev")
    anchor, formal_payload = load_anchor(args)
    model = FormalPsyPathEndToEndModel(anchor).move_trainable(device)
    processor, backbone, backbone_audit = load_backbone(args, train_lora=False)
    result = evaluate(
        processor,
        backbone,
        model,
        dev_rows,
        args,
        split="dev",
        samples=args.development_samples,
        threshold=None,
        anchor_threshold=None,
        formal_payload=formal_payload,
    )
    formal_reference = formal_development_reference(
        args.formal_root, args.task, args.seed
    )
    observed_pids = [row["pid"] for row in result["predictions"]]
    expected_pids = [str(pid) for pid in formal_reference["pids"]]
    if observed_pids != expected_pids:
        raise RuntimeError("formal development PID order changed during online audit")
    probability_error = max(
        abs(row["anchor_probability"] - float(expected))
        for row, expected in zip(
            result["predictions"], formal_reference["probability"]["posterior"]
        )
    )
    formal_threshold = float(formal_reference["threshold"])
    classification = metric_block(
        [row["label"] for row in result["predictions"]],
        [row["anchor_probability"] for row in result["predictions"]],
        formal_threshold,
    )
    f1_error = abs(
        classification["weighted_f1"]
        - float(formal_reference["posterior"]["weighted_f1"])
    )
    status = "passed" if probability_error <= 2e-5 and f1_error <= 1e-12 else "failed"
    output = {
        "schema": SCHEMA,
        "status": status,
        "task": args.task,
        "stage": "anchor_audit",
        "seed": args.seed,
        "split_policy": "official",
        "participants": len(dev_rows),
        "maximum_probability_abs_error": probability_error,
        "weighted_f1_abs_error": f1_error,
        "classification": classification,
        "formal_classification": formal_reference["posterior"],
        "backbone": backbone_audit,
        "test_labels_loaded": False,
    }
    atomic_json(
        args.output
        / args.task
        / "anchor_audit"
        / f"seed_{args.seed}"
        / "SUMMARY.json",
        output,
    )
    print(json.dumps(output, default=json_default), flush=True)
    return 0 if status == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(INSTRUMENT_ID), required=True)
    parser.add_argument(
        "--stage", choices=("anchor_audit", "frozen", "lora", "test"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--daic-root", type=Path, required=True)
    parser.add_argument("--edaic-root", type=Path, required=True)
    parser.add_argument("--meter-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--formal-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, default=None)
    parser.add_argument("--locked-state", type=Path, default=None)
    parser.add_argument("--train-lora", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--reward-folds", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--development-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=48)
    parser.add_argument("--reward-temperature", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-participants", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--event-batch-size", type=int, default=8)
    parser.add_argument("--vectorize-event-views", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-gpu-memory", default="38GiB")
    parser.add_argument("--max-events-per-participant", type=int, default=0)
    parser.add_argument("--pad", type=float, default=0.4)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    args = parser.parse_args()

    configure_runtime(args.threads)
    seed_all(args.seed)
    if args.protocol is not None:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        if protocol.get("schema") != "psypath_gfn_e2e_formal_anchor_protocol_v1":
            raise ValueError("unexpected formal PsyPath E2E protocol")
        validate_runtime_protocol(protocol, args)
    audit = json.loads(args.formal_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("split_policy") != "official" or audit.get("test_labels_loaded") is not False:
        raise RuntimeError("formal Dev online anchor audit has not passed")
    if args.stage in {"anchor_audit", "frozen"} and (
        args.train_lora or args.initial_state is not None
    ):
        raise ValueError("frozen stage must start at the exact formal anchor")
    if args.stage == "lora" and (not args.train_lora or args.initial_state is None):
        raise ValueError("LoRA stage requires --train-lora and --initial-state")
    if args.stage == "test" and args.locked_state is None:
        raise ValueError("Test requires a Dev-locked state")
    if args.max_events_per_participant != 0:
        raise ValueError("strict formal E2E requires every answer event")
    args.output.mkdir(parents=True, exist_ok=True)
    args.clip_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / args.task / args.stage / f"seed_{args.seed}" / "RUN_CONTRACT.json", {
        "schema": SCHEMA,
        "task": args.task,
        "stage": args.stage,
        "seed": args.seed,
        "split_policy": "official",
        "label_profile": TASKS[args.task]["binary_profile"],
        "all_answer_events": True,
        "cached_hidden_inference": False,
        "online_inputs": [
            "raw participant audio",
            "released transcript and interviewer context",
            "official visual sequence",
        ],
        "formal_anchor": str(checkpoint_path(args.formal_root, args.task, args.seed, test=False).resolve()),
        "formal_audit": str(args.formal_audit.resolve()),
        "implementation_hashes": implementation_hashes(),
        "protocol": str(args.protocol.resolve()) if args.protocol is not None else None,
        "protocol_sha256": protocol_sha256(args),
        "epochs": args.epochs,
        "event_batch_size": args.event_batch_size,
        "anchor_event_batch_size": 1,
        "trainable_event_batch_size": 1,
        "formal_anchor_device": "cpu",
        "test_labels_loaded": False,
    })
    if args.stage == "test":
        return locked_test(args)
    if args.stage == "anchor_audit":
        return anchor_audit(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
