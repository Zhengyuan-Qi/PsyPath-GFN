#!/usr/bin/env python3
"""Protocol-matched reproductions of published fusion models on METER caches.

The cached clinical/context, answer, and official-visual streams are mapped to
three model inputs by a train-only StandardScaler/PCA adapter. The published
fusion topology is then trained without using Dev/Test data in preprocessing.
This is a representation-matched architecture reproduction, not a claim that
the original paper's raw feature protocol has been reproduced.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(os.environ.get("ECRL_MH_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_meter_mh_sequential_increment import load_records
try:
    from scripts.published_dmd_adapter import DMDCacheAdapter
    from scripts.published_fusion_adapters import ALMTClassifier, CubeMLPClassifier, MISAClassifier, MMIMClassifier
    from scripts.published_niat_adapter import NIATCacheAdapter, NIATCorruptionConfig
    from scripts.published_self_mm_adapter import SelfMMCacheClassifier, SelfMMPseudoLabelState
    from scripts.published_simsuf_adapter import SIMSUFCacheClassifier
except ModuleNotFoundError:
    from published_dmd_adapter import DMDCacheAdapter
    from published_fusion_adapters import ALMTClassifier, CubeMLPClassifier, MISAClassifier, MMIMClassifier
    from published_niat_adapter import NIATCacheAdapter, NIATCorruptionConfig
    from published_self_mm_adapter import SelfMMCacheClassifier, SelfMMPseudoLabelState
    from published_simsuf_adapter import SIMSUFCacheClassifier


SCHEMA = "published_model_meter_cache_reproduction_v1"
TASKS = ("daic_depression", "edaic_depression", "edaic_ptsd")
MODELS = ("tfn", "lmf", "mult", "cross_modal_attention", "misa", "mmim", "cubemlp", "almt", "niat", "simsuf", "dmd", "self_mm")
PROVENANCE = {
    "tfn": {
        "paper": "Tensor Fusion Network for Multimodal Sentiment Analysis",
        "venue": "EMNLP 2017",
        "doi": "10.18653/v1/D17-1115",
        "implementation": "MultiBench fusions.common_fusions.TensorFusion",
        "adaptation": "train-only PCA inputs and binary task head",
    },
    "lmf": {
        "paper": "Efficient Low-rank Multimodal Fusion with Modality-Specific Factors",
        "venue": "ACL 2018",
        "doi": "10.18653/v1/P18-1209",
        "implementation": "MultiBench fusions.common_fusions.LowRankTensorFusion",
        "adaptation": "train-only PCA inputs; output dimension changed to two classes",
    },
    "mult": {
        "paper": "Multimodal Transformer for Unaligned Multimodal Language Sequences",
        "venue": "ACL 2019",
        "doi": "10.18653/v1/P19-1656",
        "implementation": "MultiBench fusions.mult.MULTModel",
        "adaptation": "train-only PCA inputs; official model output dimension set to two",
    },
    "cross_modal_attention": {
        "paper": "Cross-Modal Attention for Multimodal Depression Detection Using Limited DAIC-WOZ Data",
        "venue": "ICITACEE 2025",
        "doi": "10.1109/ICITACEE66165.2025.11232572",
        "implementation": "faithful PyTorch port of the authors' public Keras notebook",
        "adaptation": "train-only PCA inputs, governed splits, no SMOTE on held-out data",
    },
    "misa": {
        "paper": "MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis",
        "venue": "ACM MM 2020",
        "doi": "10.1145/3394171.3413678",
        "implementation": "topology-preserving cache port audited against declare-lab/MISA",
        "upstream_commit": "ec42faddde0d210cf7368aebf2118fe9570e7102",
        "adaptation": "cached sequence encoders and binary head; CMD, difference, and reconstruction losses retained",
    },
    "mmim": {
        "paper": "Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis",
        "venue": "EMNLP 2021",
        "doi": "10.18653/v1/2021.emnlp-main.723",
        "implementation": "topology-preserving cache port audited against declare-lab/Multimodal-Infomax",
        "upstream_commit": "cd0774c5a712ca5f1a5497dbf27dde11cade7434",
        "adaptation": "cached sequence encoders and binary head; MMILB and CPC objectives retained",
    },
    "cubemlp": {
        "paper": "CubeMLP: An MLP-based Model for Multimodal Sentiment Analysis and Depression Estimation",
        "venue": "ACM MM 2022",
        "doi": "10.1145/3503161.3548025",
        "implementation": "topology-preserving cache port audited against kiva12138/CubeMLP",
        "upstream_commit": "49f75bdc4caa5add8f2710b8053b000916ea970e",
        "adaptation": "three-axis MLP mixing retained; input projection and binary head protocol-matched",
    },
    "almt": {
        "paper": "Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis",
        "venue": "EMNLP 2023",
        "doi": "10.18653/v1/2023.emnlp-main.49",
        "implementation": "topology-preserving cache port audited against Haoyu-ha/ALMT",
        "upstream_commit": "da9461071f21e75eb8bf2bedff14ac23f5954605",
        "adaptation": "HumanOmniV2 text replaces BERT; language-guided hyper-modality and cross-fusion retained",
    },
    "niat": {
        "paper": "Noise Imitation Based Adversarial Training for Robust Multimodal Sentiment Analysis",
        "venue": "IEEE TMM 2023",
        "doi": "10.1109/TMM.2023.3267882",
        "implementation": "topology-preserving cache port audited against Columbine21/NIAT",
        "upstream_commit": "ad1cb40f23102f1f5748e2c3ab608aa5d82a891e",
        "adaptation": "HumanOmniV2 event streams replace original features; clean/corrupted supervision, reconstruction, and alternating adversarial updates retained",
    },
    "simsuf": {
        "paper": "Dominant SIngle-Modal SUpplementary Fusion (SIMSUF) for Multimodal Sentiment Analysis",
        "venue": "IEEE TMM 2024",
        "doi": "10.1109/TMM.2023.3344358",
        "implementation": "topology-preserving cache port audited against Yan-LiJI/SIMSUF",
        "upstream_commit": "ee823bf94335b280177030ea2715dcac2e6b3e23",
        "adaptation": "dominant-stream selection, supplementary attention, common/specific representations, CMD, and difference regularization retained",
    },
    "dmd": {
        "paper": "Decoupled Multimodal Distilling for Emotion Recognition",
        "venue": "CVPR 2023 Highlight",
        "implementation": "topology-preserving cache port audited against mdswyz/DMD",
        "upstream_commit": "516d4e116c22220e08873fa38df828d1a0d6f7e8",
        "adaptation": "shared/specific decomposition, reconstruction and cycle consistency, HomoGD/HeteroGD dynamic directed distillation, auxiliary heads, and joint objective retained",
    },
    "self_mm": {
        "paper": "Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis",
        "venue": "AAAI 2021",
        "doi": "10.1609/aaai.v35i12.17289",
        "implementation": "topology-preserving cache port audited against thuiar/Self-MM",
        "upstream_commit": "1786283c81eeb507f317fa1c70a3faf77e67cee0",
        "adaptation": "persistent Train-only stream pseudo-label state, center-distance refinement, epoch EMA, and disagreement-weighted stream supervision retained",
    },
}


def configure_runtime(threads: int) -> None:
    threads = max(1, min(32, int(threads)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    torch.set_num_threads(threads)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


@dataclasses.dataclass
class StreamAdapter:
    scaler: StandardScaler
    pca: PCA

    def transform(self, value: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(value)).astype(np.float32)


def _training_events(records: list[dict[str, Any]], stream: str) -> np.ndarray:
    if stream == "clinical_context":
        values = [np.concatenate((row["raw"]["clinical"], row["raw"]["context"]), axis=1) for row in records]
    else:
        values = [row["raw"][stream] for row in records]
    return np.concatenate(values, axis=0).astype(np.float32)


def fit_adapters(train: list[dict[str, Any]], dimension: int) -> dict[str, StreamAdapter]:
    result: dict[str, StreamAdapter] = {}
    for stream in ("clinical_context", "answer", "visual"):
        values = _training_events(train, stream)
        scaler = StandardScaler().fit(values)
        standardized = scaler.transform(values)
        components = min(int(dimension), standardized.shape[0] - 1, standardized.shape[1])
        pca = PCA(n_components=components, svd_solver="randomized", random_state=0).fit(standardized)
        result[stream] = StreamAdapter(scaler=scaler, pca=pca)
    return result


def adapt_record(row: dict[str, Any], adapters: dict[str, StreamAdapter], max_events: int) -> dict[str, Any]:
    clinical_context = np.concatenate((row["raw"]["clinical"], row["raw"]["context"]), axis=1)
    streams = {
        "text": adapters["clinical_context"].transform(clinical_context),
        "audio": adapters["answer"].transform(row["raw"]["answer"]),
        "visual": adapters["visual"].transform(row["raw"]["visual"]),
    }
    length = min(max_events, min(len(value) for value in streams.values()))
    padded = {}
    for name, value in streams.items():
        output = np.zeros((max_events, value.shape[1]), dtype=np.float32)
        output[:length] = value[:length]
        padded[name] = output
    return {"pid": row["pid"], "label": int(row["label"]), "length": length, **padded}


class CachedParticipantDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        return {
            "pid": row["pid"],
            "label": torch.tensor(row["label"], dtype=torch.long),
            "length": torch.tensor(row["length"], dtype=torch.long),
            "text": torch.from_numpy(row["text"]),
            "audio": torch.from_numpy(row["audio"]),
            "visual": torch.from_numpy(row["visual"]),
        }


def masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(x.shape[1], device=x.device)[None, :]
    mask = (positions < lengths[:, None]).to(x.dtype).unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class TFNClassifier(nn.Module):
    def __init__(self, fusion_class: type[nn.Module], dimension: int) -> None:
        super().__init__()
        self.fusion = fusion_class()
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear((dimension + 1) ** 3, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        values = [masked_mean(x, lengths) for x in (audio, visual, text)]
        return self.head(self.fusion(values))


class LMFClassifier(nn.Module):
    def __init__(self, fusion_class: type[nn.Module], dimension: int) -> None:
        super().__init__()
        self.fusion = fusion_class([dimension] * 3, 2, rank=4)
        # MultiBench commit 49f9be calls .to() on Parameters and stores factors
        # in a plain list, which unregisters them on current PyTorch. Re-register
        # the same tensors without changing the published LMF computation.
        self.fusion.factors = nn.ParameterList(
            nn.Parameter(value.detach().clone()) for value in self.fusion.factors
        )
        self.fusion.fusion_weights = nn.Parameter(self.fusion.fusion_weights.detach().clone())
        self.fusion.fusion_bias = nn.Parameter(self.fusion.fusion_bias.detach().clone())

    def forward(self, text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.fusion([masked_mean(x, lengths) for x in (audio, visual, text)])


class MulTClassifier(nn.Module):
    def __init__(self, model_class: type[nn.Module], dimension: int) -> None:
        super().__init__()
        hyper = SimpleNamespace(
            embed_dim=30,
            num_heads=5,
            layers=2,
            attn_dropout=0.1,
            attn_dropout_modalities=[0.1, 0.1, 0.1],
            relu_dropout=0.1,
            res_dropout=0.1,
            out_dropout=0.1,
            embed_dropout=0.1,
            attn_mask=False,
            output_dim=2,
            all_steps=False,
        )
        self.model = model_class(3, [dimension] * 3, hyper)

    def forward(self, text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        return self.model([text, audio, visual])


class CrossModalAttentionClassifier(nn.Module):
    """Topology-preserving port of the public ICITACEE 2025 Keras model."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.audio_rnn = nn.RNN(dimension, 64, batch_first=True)
        self.audio_dense = nn.Linear(64, 32)
        self.text_dense1 = nn.Linear(dimension, 128)
        self.text_bn = nn.BatchNorm1d(128)
        self.text_dense2 = nn.Linear(128, 64)
        self.visual_conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.visual_pool = nn.MaxPool1d(2)
        self.visual_dense = nn.Linear(32 * (dimension // 2), 16)
        self.projections = nn.ModuleList((nn.Linear(32, 32), nn.Linear(64, 32), nn.Linear(16, 32)))
        self.attention = nn.ModuleList(nn.MultiheadAttention(32, 2, dropout=0.0, batch_first=True) for _ in range(6))
        self.head = nn.Sequential(nn.Linear(32 * 6, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        text_vector = masked_mean(text, lengths)
        visual_vector = masked_mean(visual, lengths)
        audio_seq, _ = self.audio_rnn(audio)
        audio_seq = F.relu(self.projections[0](F.relu(self.audio_dense(audio_seq))))
        text_vector = F.dropout(F.relu(self.text_bn(self.text_dense1(text_vector))), 0.3, self.training)
        text_seq = F.relu(self.projections[1](F.relu(self.text_dense2(text_vector))))[:, None, :].expand(-1, audio.shape[1], -1)
        visual_vector = self.visual_pool(F.relu(self.visual_conv(visual_vector[:, None, :]))).flatten(1)
        visual_seq = F.relu(self.projections[2](F.relu(self.visual_dense(visual_vector))))[:, None, :].expand(-1, audio.shape[1], -1)
        pairs = ((audio_seq, text_seq), (audio_seq, visual_seq), (text_seq, audio_seq), (text_seq, visual_seq), (visual_seq, audio_seq), (visual_seq, text_seq))
        fused = [layer(query, value, value, need_weights=False)[0] for layer, (query, value) in zip(self.attention, pairs)]
        return self.head(torch.cat(fused, dim=-1).mean(dim=1))


def external_classes(root: Path) -> tuple[type[nn.Module], type[nn.Module], type[nn.Module]]:
    root = root.resolve()
    if not (root / "fusions" / "common_fusions.py").exists():
        raise FileNotFoundError(f"MultiBench checkout not found: {root}")
    sys.path.insert(0, str(root))
    from fusions.common_fusions import LowRankTensorFusion, TensorFusion
    from fusions.mult import MULTModel
    return TensorFusion, LowRankTensorFusion, MULTModel


def almt_classes(external_models_root: Path) -> tuple[type[nn.Module], type[nn.Module], type[nn.Module]]:
    source = external_models_root.resolve() / "ALMT" / "models" / "almt_layer.py"
    if not source.exists():
        raise FileNotFoundError(f"official ALMT checkout not found: {source}")
    spec = importlib.util.spec_from_file_location("official_almt_layer", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Transformer, module.CrossTransformer, module.HhyperLearningEncoder


def build_model(name: str, dimension: int, external_root: Path, external_models_root: Path, max_events: int) -> nn.Module:
    tensor_fusion, low_rank_fusion, mult_model = external_classes(external_root)
    if name == "tfn":
        return TFNClassifier(tensor_fusion, dimension)
    if name == "lmf":
        return LMFClassifier(low_rank_fusion, dimension)
    if name == "mult":
        return MulTClassifier(mult_model, dimension)
    if name == "cross_modal_attention":
        return CrossModalAttentionClassifier(dimension)
    if name == "misa":
        return MISAClassifier(dimension)
    if name == "mmim":
        return MMIMClassifier(dimension)
    if name == "cubemlp":
        return CubeMLPClassifier(dimension, max_events)
    if name == "almt":
        transformer, cross_transformer, hyper_encoder = almt_classes(external_models_root)
        return ALMTClassifier(dimension, max_events, transformer, cross_transformer, hyper_encoder)
    if name == "niat":
        return NIATCacheAdapter((dimension, dimension, dimension))
    if name == "simsuf":
        return SIMSUFCacheClassifier(dimension)
    if name == "dmd":
        return DMDCacheAdapter((dimension, dimension, dimension))
    if name == "self_mm":
        return SelfMMCacheClassifier(dimension)
    raise ValueError(name)


def metric_block(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(np.int64)
    return {
        "n": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, prediction)),
        "weighted_f1": float(f1_score(labels, prediction, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(labels, prediction, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else None,
    }


def choose_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = []
    for threshold in np.arange(0.20, 0.801, 0.025):
        metrics = metric_block(labels, probability, float(threshold))
        candidates.append((metrics["weighted_f1"], metrics["macro_f1"], -abs(threshold - 0.5), float(threshold)))
    return max(candidates)[-1]


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    pids, labels, probability = [], [], []
    for batch in loader:
        inputs = tuple(batch[key].to(device) for key in ("text", "audio", "visual", "length"))
        if isinstance(model, NIATCacheAdapter):
            logits = model.classifier(model.encode_clean(*inputs))
        elif isinstance(model, DMDCacheAdapter):
            logits = model(*inputs).final_logits
        elif isinstance(model, SelfMMCacheClassifier):
            logits = model(*inputs).fusion_logits
            pids.extend(batch["pid"])
            labels.extend(batch["label"].numpy().tolist())
            probability.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            continue
        else:
            logits = model(*inputs)
        pids.extend(batch["pid"])
        labels.extend(batch["label"].numpy().tolist())
        probability.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    return pids, np.asarray(labels, dtype=np.int64), np.asarray(probability, dtype=np.float64)


def make_loader(
    records: list[dict[str, Any]],
    batch_size: int,
    *,
    train: bool,
    seed: int,
    weighted_sampling: bool = True,
) -> DataLoader:
    dataset = CachedParticipantDataset(records)
    sampler = None
    shuffle = False
    generator = None
    if train and weighted_sampling:
        labels = np.asarray([row["label"] for row in records], dtype=np.int64)
        counts = np.bincount(labels, minlength=2).clip(min=1)
        weights = torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    elif train:
        shuffle = True
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, generator=generator, num_workers=0)


def run_seed(
    args: argparse.Namespace,
    task: str,
    model_name: str,
    records: dict[str, list[dict[str, Any]]],
    adapters: dict[str, StreamAdapter],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    seed_all(seed)
    loaders = {
        split: make_loader(
            rows,
            args.batch_size,
            train=split == "train",
            seed=seed,
            weighted_sampling=model_name != "self_mm",
        )
        for split, rows in records.items()
    }
    model = build_model(model_name, args.pca_dim, args.multibench_root, args.external_models_root, args.max_events).to(device)
    if isinstance(model, NIATCacheAdapter):
        task_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("discriminator.")]
        optimizer = torch.optim.AdamW(task_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        discriminator_optimizer = torch.optim.AdamW(model.discriminator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        discriminator_optimizer = None
    train_labels = np.asarray([row["label"] for row in records["train"]], dtype=np.int64)
    counts = np.bincount(train_labels, minlength=2).clip(min=1)
    class_weight = torch.as_tensor(len(train_labels) / (2.0 * counts), dtype=torch.float32, device=device)
    self_mm_state = None
    if isinstance(model, SelfMMCacheClassifier):
        self_mm_state = SelfMMPseudoLabelState(model.feature_dimensions)
        self_mm_state.initialize([row["pid"] for row in records["train"]], torch.from_numpy(train_labels))
        self_mm_positive_weight = torch.tensor(counts[0] / counts[1], dtype=torch.float32, device=device)
    best_state, best_loss, best_epoch, stale = None, math.inf, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_batches = len(loaders["train"])
        for batch_index, batch in enumerate(loaders["train"], start=1):
            inputs = tuple(batch[key].to(device) for key in ("text", "audio", "visual", "length"))
            labels = batch["label"].to(device)
            if isinstance(model, NIATCacheAdapter):
                for parameter in model.discriminator.parameters():
                    parameter.requires_grad_(False)
                output = model(
                    *inputs,
                    corruption_config=NIATCorruptionConfig(args.niat_missing_rate, args.niat_corruption),
                )
                loss = model.training_losses(output, labels, class_weight=class_weight).generator_total
            elif isinstance(model, DMDCacheAdapter):
                output = model(*inputs)
                loss = model.training_losses(output, labels, class_weight=class_weight).total
            elif isinstance(model, SelfMMCacheClassifier):
                assert self_mm_state is not None
                output = model(*inputs)
                loss = model.training_loss(
                    output,
                    labels,
                    batch["pid"],
                    self_mm_state,
                    positive_weight=self_mm_positive_weight,
                )
            else:
                logits = model(*inputs)
                loss = model.training_loss(logits, labels, class_weight) if hasattr(model, "training_loss") else F.cross_entropy(logits, labels, weight=class_weight)
            accumulation_steps = args.dmd_accumulation_steps if isinstance(model, DMDCacheAdapter) else 1
            group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, train_batches - group_start)
            (loss / group_size).backward()
            if isinstance(model, SelfMMCacheClassifier):
                assert self_mm_state is not None
                if epoch > 1:
                    self_mm_state.update_pseudo_labels(epoch, batch["pid"], output.features)
                self_mm_state.update_features(batch["pid"], output.features)
                if epoch > 1:
                    self_mm_state.update_centers()
            should_step = batch_index % accumulation_steps == 0 or batch_index == train_batches
            if should_step:
                nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if isinstance(model, NIATCacheAdapter):
                assert discriminator_optimizer is not None
                for parameter in model.discriminator.parameters():
                    parameter.requires_grad_(True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_loss = model.discriminator_loss(output.clean_fused, output.corrupted_fused)
                discriminator_loss.backward()
                nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
                discriminator_optimizer.step()
        if isinstance(model, SelfMMCacheClassifier) and epoch == 1:
            assert self_mm_state is not None
            self_mm_state.update_centers()
        _, dev_labels, dev_probability = predict(model, loaders["dev"], device)
        dev_loss = float(F.binary_cross_entropy(torch.from_numpy(dev_probability), torch.from_numpy(dev_labels.astype(np.float64))).item())
        if dev_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = dev_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    _, dev_labels, dev_probability = predict(model, loaders["dev"], device)
    threshold = choose_threshold(dev_labels, dev_probability)
    result: dict[str, Any] = {"seed": seed, "selected_epoch": best_epoch, "development": metric_block(dev_labels, dev_probability, threshold)}
    # Official Test is not opened until the checkpoint and threshold above are
    # fixed from Train/Dev. This deliberately reloads Test for each locked run.
    test_raw = load_records(
        args.meter_root,
        args.contract_dir,
        task,
        allowed_splits=("test",),
        split_policy="official",
    )["test"]
    test_records = [adapt_record(row, adapters, args.max_events) for row in test_raw]
    test_loader = make_loader(test_records, args.batch_size, train=False, seed=seed)
    pids, labels, probability = predict(model, test_loader, device)
    result["test"] = metric_block(labels, probability, threshold)
    result["test"]["pid"] = pids
    result["test"]["label"] = labels.tolist()
    result["test"]["probability"] = probability.tolist()
    return result


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for metric in ("accuracy", "weighted_f1", "macro_f1", "positive_f1", "auroc"):
        values = np.asarray([run["test"][metric] for run in runs], dtype=np.float64)
        result[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "values": values.tolist()}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meter-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--multibench-root", type=Path, required=True)
    parser.add_argument("--external-models-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27, 41, 53])
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--max-events", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--niat-missing-rate", type=float, default=0.2)
    parser.add_argument("--niat-corruption", choices=("structured", "random", "block"), default="structured")
    parser.add_argument("--dmd-accumulation-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dmd_accumulation_steps < 1:
        raise ValueError("--dmd-accumulation-steps must be positive")
    if args.external_models_root is None:
        args.external_models_root = args.multibench_root.resolve().parent
    configure_runtime(args.threads)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": SCHEMA,
        "meter_root": str(args.meter_root.resolve()),
        "contract_dir": str(args.contract_dir.resolve()),
        "multibench_root": str(args.multibench_root.resolve()),
        "multibench_commit": "49f9be224f342de005b7f001281f33df4301eb22",
        "split_policy": "official",
        "preprocessing_fit_partition": "train_only",
        "checkpoint_selection_partition": "dev_only",
        "test_policy": "opened_after preprocessing/checkpoint/threshold lock",
        "stream_mapping": {"text": "concat(clinical, context)", "audio": "answer", "visual": "official visual"},
        "representation_claim": "protocol-matched architecture reproduction on HumanOmniV2 cache",
        "models": {name: PROVENANCE[name] for name in args.models},
        "parameters": vars(args),
    }
    atomic_json(args.output / "RUN_CONTRACT.json", contract)
    started = time.time()
    all_results = {}
    for task in args.tasks:
        raw = load_records(
            args.meter_root,
            args.contract_dir,
            task,
            allowed_splits=("train", "dev"),
            split_policy="official",
        )
        adapters = fit_adapters(raw["train"], args.pca_dim)
        records = {split: [adapt_record(row, adapters, args.max_events) for row in rows] for split, rows in raw.items()}
        all_results[task] = {}
        for model_name in args.models:
            runs = []
            for seed in args.seeds:
                run = run_seed(args, task, model_name, records, adapters, seed, device)
                runs.append(run)
                atomic_json(args.output / task / model_name / f"seed_{seed}.json", run)
            block = {"provenance": PROVENANCE[model_name], "runs": runs, "summary": summarize(runs)}
            all_results[task][model_name] = block
            atomic_json(args.output / task / model_name / "FINAL_METRICS.json", block)
    final = {"schema": SCHEMA, "status": "complete", "elapsed_s": time.time() - started, "tasks": all_results}
    atomic_json(args.output / "FINAL_METRICS.json", final)
    print(json.dumps({"status": "complete", "elapsed_s": final["elapsed_s"]}, indent=2))


if __name__ == "__main__":
    main()
