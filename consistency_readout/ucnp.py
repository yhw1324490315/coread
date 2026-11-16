"""UCNP input adapters, preserving sparse peaks and masked band labels."""

import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path

import numpy as np

from .data import _feature_paths, _features, _hash, _ids

BANDS = tuple(f"B{i}" for i in range(1, 8))
VARIANTS = ("original", "equivalent_order", "equivalent_units")


@dataclass
class PeakData:
    ids: np.ndarray
    targets: np.ndarray
    folds: np.ndarray
    components: np.ndarray
    scientific_groups: list
    x: np.ndarray
    available: np.ndarray
    inputs: dict
    variants: tuple = VARIANTS


@dataclass
class DetectionData:
    train_ids: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    projection_mask: np.ndarray
    reorder: np.ndarray
    reorder_parents: np.ndarray
    paraphrase: np.ndarray
    paraphrase_parents: np.ndarray
    valid_ids: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    valid_papers: np.ndarray
    inputs: dict


def task_for_path(path):
    """Recognize label schemas, so custom folder names also work."""
    path = Path(path)
    if not path.is_dir():
        return "scalar"
    with (path / "labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        columns = set(next(csv.reader(stream)))
    if {f"target_{band}" for band in BANDS}.issubset(columns):
        return "peaks"
    if {"partition", "subpartition", *(f"mlabel_{band}" for band in BANDS)}.issubset(columns):
        return "detection"
    return "scalar"


def _flat_inputs(path):
    path = Path(path)
    label_path, prompt_path = path / "labels.csv", path / "prompts.jsonl.gz"
    with label_path.open(encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))
    if not labels:
        raise ValueError("UCNP labels are empty")
    ids = _ids([row["sample_id"] for row in labels], len(labels))
    parents = set(ids)
    with gzip.open(prompt_path, "rt", encoding="utf-8") as stream:
        prompts = [json.loads(line) for line in stream]
    prompt_ids = _ids([row["sample_id"] for row in prompts], len(prompts))
    for row in prompts:
        if row["parent_id"] not in parents or row["variant"] not in {*VARIANTS, "paraphrase"}:
            raise ValueError("Unknown UCNP prompt parent or variant")
    feature_paths = _feature_paths(path)
    vectors, width, dtype = {}, None, None
    for feature_path in feature_paths:
        with np.load(feature_path, allow_pickle=False) as bundle:
            features = _features(bundle["features"], 2)
            feature_ids = _ids(bundle["sample_id"], len(features))
        if not np.isfinite(features).all():
            raise ValueError("UCNP features must be finite")
        if width is None:
            width, dtype = features.shape[1], features.dtype
        if features.shape[1] != width or features.dtype != dtype:
            raise ValueError("UCNP feature chunks differ in dimension or dtype")
        for sid, vector in zip(feature_ids, features):
            if sid in vectors:
                raise ValueError(f"Duplicate UCNP feature ID: {sid}")
            vectors[sid] = vector
    if set(vectors) != set(prompt_ids):
        raise ValueError("UCNP feature IDs do not exactly match the prompts")
    files = [label_path, prompt_path, *feature_paths]
    return labels, ids, prompts, vectors, {p.name: _hash(p) for p in files}


def load_peaks(path):
    labels, ids, prompts, vectors, inputs = _flat_inputs(path)
    index = {sid: i for i, sid in enumerate(ids)}
    first = next(iter(vectors.values()))
    x = np.zeros((len(ids), len(VARIANTS), len(first)), dtype=first.dtype)
    available = np.zeros(x.shape[:2], dtype=bool)
    for row in prompts:
        if row["variant"] not in VARIANTS:
            raise ValueError("Peak regression accepts original, order and unit variants")
        slot = (index[row["parent_id"]], VARIANTS.index(row["variant"]))
        if available[slot]:
            raise ValueError("Duplicate UCNP peak prompt slot")
        x[slot] = vectors[row["sample_id"]]
        available[slot] = True
    if not available[:, 0].all():
        raise ValueError("Every peak-regression sample needs an original representation")
    y = np.array([[float(row[f"target_{band}"]) if row[f"target_{band}"].strip() else np.nan
                   for band in BANDS] for row in labels])
    if np.isinf(y).any():
        raise ValueError("Peak targets must be finite values or missing")
    folds = np.array([int(row["outer_fold"]) for row in labels])
    components = np.array([row["component_id"] for row in labels], dtype=str)
    groups = [tuple(sorted(set(json.loads(row["scientific_group_ids"])))) for row in labels]
    if any(not keys or any(not isinstance(key, str) or not key for key in keys) for keys in groups):
        raise ValueError("Peak samples require scientific-group memberships")
    membership = {}
    component_folds = {}
    for component, fold, keys in zip(components, folds, groups):
        if not component:
            raise ValueError("Missing UCNP component ID")
        if component in component_folds and component_folds[component] != fold:
            raise ValueError("UCNP component crosses fixed outer folds")
        component_folds[component] = fold
        for key in keys:
            if key in membership and membership[key] != component:
                raise ValueError("UCNP scientific group crosses components")
            membership[key] = component
    if set(folds) != set(range(5)):
        raise ValueError("UCNP peaks require all five fixed outer folds")
    return PeakData(ids, y, folds, components, groups, x, available, inputs)


def load_detection(path):
    labels, ids, prompts, vectors, inputs = _flat_inputs(path)
    partitions = np.array([row["partition"] for row in labels])
    if not set(partitions).issubset({"train", "valid"}) or not (partitions == "train").any():
        raise ValueError("Detection partitions must be train/valid, with nonempty training data")
    train, valid = partitions == "train", partitions == "valid"
    papers = np.array([row.get("paper_id", "") for row in labels])
    training_papers = set(papers[train]) - {""}
    validation_papers = set(papers[valid]) - {""}
    if training_papers & validation_papers:
        raise ValueError("Detection training and validation papers overlap")
    train_ids, valid_ids = ids[train], ids[valid]
    training_index = {sid: i for i, sid in enumerate(train_ids)}
    subpartitions = np.array([row["subpartition"] for row in labels])
    if not set(subpartitions[train]).issubset({"cohort", "extra", "supp"}):
        raise ValueError("Unknown detection training subpartition")
    projection_mask = np.isin(subpartitions[train], ["cohort", "extra"])
    originals, reordered, paraphrases, paraphrase_parents = {}, {}, [], []
    for row in prompts:
        parent, variant = row["parent_id"], row["variant"]
        vector = vectors[row["sample_id"]]
        if variant == "original":
            if parent in originals:
                raise ValueError("Duplicate detection original")
            originals[parent] = vector
        elif variant in {"equivalent_order", "paraphrase"}:
            if parent not in training_index or not projection_mask[training_index[parent]]:
                raise ValueError("Only original training-pool samples may contribute detection rewrites")
            if variant == "equivalent_order":
                if parent in reordered:
                    raise ValueError("Duplicate detection order rewrite")
                reordered[parent] = vector
            else:
                paraphrases.append(vector)
                paraphrase_parents.append(training_index[parent])
        else:
            raise ValueError("Unsupported detection rewrite variant")
    if set(originals) != set(ids):
        raise ValueError("Missing original detection representations")
    projection_ids = train_ids[projection_mask]
    if set(reordered) != set(projection_ids) or not paraphrases:
        raise ValueError("Detection consistency pool needs complete order rewrites and accepted paraphrases")
    y = np.array([[int(row[f"mlabel_{band}"]) for band in BANDS] for row in labels])
    if not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("Detection masked labels must be -1, 0 or 1")
    width = len(next(iter(originals.values())))
    x_valid = np.stack([originals[sid] for sid in valid_ids]) if len(valid_ids) else np.empty((0, width))
    return DetectionData(train_ids, np.stack([originals[sid] for sid in train_ids]), y[train],
                         projection_mask, np.stack([reordered[sid] for sid in projection_ids]),
                         np.flatnonzero(projection_mask), np.stack(paraphrases),
                         np.array(paraphrase_parents), valid_ids, x_valid, y[valid],
                         papers[valid], inputs)
