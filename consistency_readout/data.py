"""Load bundled datasets or user-provided NumPy representations."""

from dataclasses import dataclass
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np


@dataclass
class TrainingData:
    ids: np.ndarray
    targets: np.ndarray
    folds: np.ndarray | None
    x: np.ndarray
    available: np.ndarray
    variants: tuple[str, ...]
    inputs: dict


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(values, n):
    ids = np.asarray(values, dtype=str)
    if ids.shape != (n,) or any(not sid.strip() for sid in ids) or len(set(ids)) != n:
        raise ValueError("sample_id must contain one unique, nonempty ID per sample")
    return ids


def _features(values, ndim):
    x = np.asarray(values)
    if x.ndim != ndim or any(size == 0 for size in x.shape) or x.dtype.kind not in "fiu":
        raise ValueError(f"Representations must be a nonempty {ndim}D real numeric array")
    return x if x.dtype.kind == "f" else x.astype(np.float64)


def _feature_paths(folder):
    """Find one complete local feature file or a contiguous set of chunks."""
    folder = Path(folder)
    single, chunks = folder / "features.npz", list(folder.glob("features_*.npz"))
    if single.exists() and chunks:
        raise ValueError("Use features.npz OR feature chunks, not both")
    if single.exists():
        paths = [single]
        _validate_feature_manifest(folder, paths)
        return paths
    if chunks:
        indexed = []
        for chunk in chunks:
            match = re.fullmatch(r"features_(\d+)\.npz", chunk.name)
            if match is None:
                raise ValueError(f"Invalid feature chunk name: {chunk.name}")
            indexed.append((int(match.group(1)), chunk))
        indexed.sort(key=lambda item: item[0])
        if [index for index, _ in indexed] != list(range(len(indexed))):
            raise ValueError("Feature chunk indices must be unique and consecutive from zero")
        paths = [chunk for _, chunk in indexed]
        _validate_feature_manifest(folder, paths)
        return paths
    raise FileNotFoundError(
        f"No local features found in {folder}. Install the extractor with "
        "python -m pip install '.[extract]', then run "
        f'python run.py extract --data "{folder}" '
        '--model models/pretrained/Qwen2.5-7B-Instruct. '
        "The pretrained model must already be available locally."
    )


def _validate_feature_manifest(folder, paths):
    """Keep interrupted or modified extraction caches out of training/prediction."""
    manifest_path = folder / "features_manifest.json"
    if not manifest_path.exists():
        return  # Existing user-supplied feature files remain supported.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported local feature extraction manifest")
    configuration = manifest.get("configuration", {})
    total = configuration.get("prompt_count") if isinstance(configuration, dict) else None
    if (manifest.get("status") != "complete" or type(total) is not int or total < 1
            or manifest.get("completed_rows") != total):
        raise ValueError("Feature extraction is incomplete; finish it with run.py extract --resume before training or prediction")
    prompt_path = folder / "prompts.jsonl.gz"
    if not prompt_path.exists():
        prompt_path = folder / "prompts.jsonl"
    if not prompt_path.exists() and configuration.get("prompts_file"):
        filename = configuration["prompts_file"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Invalid source prompt filename in extraction manifest")
        prompt_path = folder / filename
    if prompt_path.exists() and _hash(prompt_path) != configuration.get("prompts_sha256"):
        raise ValueError("Input prompt checksum differs from the extraction manifest; regenerate the feature cache")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not all(isinstance(chunk, dict) for chunk in chunks):
        raise ValueError("Invalid feature extraction chunk manifest")
    expected = [chunk.get("file") for chunk in chunks]
    if (not all(isinstance(name, str) for name in expected)
            or len(expected) != len(set(expected)) or set(expected) != {path.name for path in paths}):
        raise ValueError("Feature files do not match the completed extraction manifest")
    width = manifest.get("feature_dimension")
    if type(width) is not int or width < 1:
        raise ValueError("Invalid extracted feature dimension")
    position = 0
    files = {path.name: path for path in paths}
    for chunk in chunks:
        rows = chunk.get("rows")
        if type(rows) is not int or rows < 1 or chunk.get("start") != position:
            raise ValueError("Feature extraction chunk rows are incomplete or out of order")
        path = files[chunk["file"]]
        if _hash(path) != chunk.get("sha256"):
            raise ValueError(f"Extracted feature checksum differs: {path.name}")
        with np.load(path, allow_pickle=False) as bundle:
            if ("features" not in bundle.files or "sample_id" not in bundle.files
                    or bundle["features"].shape != (rows, width)
                    or bundle["sample_id"].shape != (rows,)):
                raise ValueError(f"Extracted feature shapes differ from the manifest: {path.name}")
        position += rows
    if position != total:
        raise ValueError("Extracted features do not cover every input prompt")


def _training(ids, y, folds, x, available, variants, inputs):
    x = _features(x, 3)
    ids = _ids(ids, len(x))
    y = np.asarray(y, dtype=float)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("y must contain one finite target per sample")
    if available is None:
        available = ~np.isnan(x).all(axis=-1)
    available = np.asarray(available)
    if available.dtype != np.bool_ or available.shape != x.shape[:2]:
        raise ValueError("available must be a boolean mask with shape (samples, variants)")
    if not available[:, 0].all() or not np.isfinite(x[available]).all():
        raise ValueError("Original representations and all available rewrites must be finite")
    x = np.where(available[:, :, None], x, 0)
    if folds is not None:
        raw_folds = np.asarray(folds)
        if raw_folds.shape != (len(x),) or raw_folds.dtype.kind not in "iu" or (raw_folds < 0).any():
            raise ValueError("outer_fold must contain one nonnegative integer per sample")
        folds = raw_folds.astype(int)
    return TrainingData(ids, y, folds, x, available, variants, inputs)


def load_data(path: str | Path) -> TrainingData:
    """Load a dataset folder or an NPZ with X, y and optional available/sample_id/outer_fold."""
    path = Path(path)
    if path.is_file():
        with np.load(path, allow_pickle=False) as bundle:
            if not {"X", "y"}.issubset(bundle.files):
                raise ValueError("Training NPZ must contain X and y arrays")
            x, y = bundle["X"], bundle["y"]
            if x.ndim != 3:
                raise ValueError("X must have shape (samples, variants, hidden dimension)")
            ids = bundle["sample_id"] if "sample_id" in bundle else [f"sample_{i:06d}" for i in range(len(x))]
            available = bundle["available"] if "available" in bundle else None
            folds = bundle["outer_fold"] if "outer_fold" in bundle else None
        variants = ("original", *(f"rewrite_{i}" for i in range(1, x.shape[1])))
        return _training(ids, y, folds, x, available, variants, {path.name: _hash(path)})
    if not path.is_dir():
        raise FileNotFoundError(f"Training data not found: {path}")

    labels_path, prompts_path = path / "labels.csv", path / "prompts.jsonl.gz"
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"sample_id", "target"}.issubset(reader.fieldnames or []):
            raise ValueError("labels.csv requires sample_id and target columns")
        has_folds = "outer_fold" in reader.fieldnames
        labels = list(reader)
    if not labels:
        raise ValueError("labels.csv is empty")
    ids = _ids([row["sample_id"] for row in labels], len(labels))
    y = np.asarray([float(row["target"]) for row in labels])
    folds = np.asarray([int(row["outer_fold"]) for row in labels]) if has_folds else None
    sample_rows = {sid: i for i, sid in enumerate(ids)}
    with gzip.open(prompts_path, "rt", encoding="utf-8") as stream:
        prompts = [json.loads(line) for line in stream]
    names = {p["variant"] for p in prompts}
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Prompt variants must be nonempty strings")
    variants = ("original", *sorted(names - {"original"}))
    slots, prompt_ids = {}, set()
    available = np.zeros((len(ids), len(variants)), dtype=bool)
    for prompt in prompts:
        sid, parent, variant = prompt["sample_id"], prompt["parent_id"], prompt["variant"]
        if not isinstance(sid, str) or not sid.strip() or sid in prompt_ids or parent not in sample_rows:
            raise ValueError("Duplicate/invalid prompt ID or unknown parent_id")
        slot = (sample_rows[parent], variants.index(variant))
        if available[slot]:
            raise ValueError("Multiple prompts occupy the same sample and variant")
        prompt_ids.add(sid)
        slots[sid] = slot
        available[slot] = True
    if not available[:, 0].all():
        raise ValueError("Every sample needs an original prompt")

    feature_paths = _feature_paths(path)
    x, seen = None, set()
    for feature_path in feature_paths:
        with np.load(feature_path, allow_pickle=False) as bundle:
            if not {"sample_id", "features"}.issubset(bundle.files):
                raise ValueError("Feature bundles require sample_id and features")
            features = _features(bundle["features"], 2)
            feature_ids = _ids(bundle["sample_id"], len(features))
        if not np.isfinite(features).all():
            raise ValueError("Precomputed features must be finite")
        if x is None:
            x = np.zeros((len(ids), len(variants), features.shape[1]), dtype=features.dtype)
        if x.shape[-1] != features.shape[1] or x.dtype != features.dtype:
            raise ValueError("Feature chunks must use the same dimension and dtype")
        for sid, vector in zip(feature_ids, features):
            if sid in seen or sid not in slots:
                raise ValueError(f"Duplicate or unmatched feature sample_id: {sid}")
            seen.add(sid)
            x[slots[sid]] = vector
    if seen != prompt_ids:
        raise ValueError(f"Missing features for {len(prompt_ids - seen)} prompts")
    files = [labels_path, prompts_path, *feature_paths]
    return _training(ids, y, folds, x, available, variants, {p.name: _hash(p) for p in files})


def load_features(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load prediction representations from NPY, NPZ, or an extracted-feature directory."""
    path = Path(path)
    if path.is_dir():
        arrays, identifiers, width, dtype = [], [], None, None
        for chunk in _feature_paths(path):
            with np.load(chunk, allow_pickle=False) as bundle:
                if not {"features", "sample_id"}.issubset(bundle.files):
                    raise ValueError("Feature directory chunks require features and sample_id arrays")
                features = _features(bundle["features"], 2)
                ids = _ids(bundle["sample_id"], len(features))
            if not np.isfinite(features).all():
                raise ValueError("Prediction representations must be finite")
            if width is None:
                width, dtype = features.shape[1], features.dtype
            if features.shape[1] != width or features.dtype != dtype:
                raise ValueError("Feature chunks must use the same dimension and dtype")
            arrays.append(features)
            identifiers.extend(ids)
        ids = _ids(identifiers, sum(len(array) for array in arrays))
        prompt_path = path / "prompts.jsonl.gz"
        if not prompt_path.exists():
            prompt_path = path / "prompts.jsonl"
        if prompt_path.exists():
            opener = gzip.open if prompt_path.suffix == ".gz" else open
            with opener(prompt_path, "rt", encoding="utf-8") as stream:
                expected = [json.loads(line)["sample_id"] for line in stream]
            _ids(expected, len(expected))
            if set(ids) != set(expected):
                raise ValueError("Prediction feature IDs do not exactly match the input prompts")
        return ids, np.concatenate(arrays)
    if path.suffix.lower() == ".npy":
        features = _features(np.load(path, allow_pickle=False), 2)
        ids = [f"sample_{i:06d}" for i in range(len(features))]
    else:
        with np.load(path, allow_pickle=False) as bundle:
            if "features" not in bundle.files:
                raise ValueError("Prediction NPZ must contain a features array")
            features = _features(bundle["features"], 2)
            ids = bundle["sample_id"] if "sample_id" in bundle else [f"sample_{i:06d}" for i in range(len(features))]
    if not np.isfinite(features).all():
        raise ValueError("Prediction representations must be finite")
    return _ids(ids, len(features)), features
