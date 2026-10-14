"""File identities and path checks shared by checkpoint export and inference."""
import hashlib
import json
from pathlib import Path

TASKS = ("daic_depression", "edaic_depression", "edaic_ptsd")
SEEDS = (7, 17, 27, 41, 53)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def contained_file(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(name):
        raise ValueError("Manifest paths must be relative and remain inside their directory")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or unsafe manifest file: {name}")
    return path


def backbone_snapshot(root):
    root = Path(root).resolve()
    shards = sorted(root.glob("*.safetensors"))
    if not (root / "config.json").is_file() or not shards:
        raise ValueError("Expected the complete HumanOmniV2 config and safetensors weights")
    files = sorted(p for p in root.iterdir() if p.is_file() and
                   p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken",
                                ".jinja", ".py"})
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        for name in set(mapping.values()):
            contained_file(root, name)
        if not set(mapping.values()).issubset({p.name for p in shards}):
            raise ValueError("Backbone shard index does not match the available weight files")
    return {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}


def extraction_fingerprint(root):
    """Match the recorded extraction contract's path/size/mtime fingerprint."""
    root = Path(root)
    paths = sorted(root.glob("*.safetensors"))
    paths += [root / name for name in ("config.json", "model.safetensors.index.json")
              if (root / name).is_file()]
    records = [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
               for p in paths]
    return hashlib.sha256(json.dumps(records, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def verify_extraction(cache_root, backbone, tasks):
    fingerprint = extraction_fingerprint(backbone)
    records = {}
    root = Path(__file__).resolve().parent
    for dataset in sorted({"daic" if t == "daic_depression" else "edaic" for t in tasks}):
        path = Path(cache_root) / dataset / "RUN_CONTRACT.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        expected = json.loads((root / "configs" / f"encoder_{dataset}.json").read_text())
        if contract["schema"] != expected["schema"] or contract["dataset"] != dataset:
            raise ValueError("Wrong extraction schema or dataset")
        if contract["view_contract"] != expected["view_contract"]:
            raise ValueError("Encoding prompts differ from the paper configuration")
        if contract["model_files_fingerprint_sha256"] != fingerprint:
            raise ValueError("Use the same backbone directory used for extraction; "
                             "the extraction contract records paths and file timestamps")
        records[dataset] = digest(path)
    return records


def verify_manifest(model_dir, backbone, task):
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["tasks"][task]
    weight = contained_file(model_dir, entry["file"])
    if digest(weight) != entry["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    files = manifest["backbone_files"]
    if "config.json" not in files or not any(n.endswith(".safetensors") for n in files):
        raise ValueError("Manifest must identify the backbone config and weights")
    for name, expected in files.items():
        path = contained_file(backbone, name)
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Backbone file differs from the exported model: {name}")
    return weight, entry
