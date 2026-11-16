"""Seven-band consistency readout for sparsely labelled UCNP peak wavelengths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, qr

from .ridge import ALPHAS, LAMBDAS

BANDS = tuple(f"B{i}" for i in range(1, 8))
SCHEMA_VERSION = 1


def component_inner(components, outer_fold="full"):
    """Keep components together using the frozen target-blind three-fold rule."""
    components = np.asarray(components, dtype=str)
    unique, counts = np.unique(components, return_counts=True)
    if len(unique) < 3:
        raise ValueError("Peak readout requires at least three training components")
    ordered = sorted(zip(unique, counts), key=lambda item: (
        -item[1], hashlib.sha256(
            f"20261019|outer={outer_fold}|inner|{item[0]}".encode()).hexdigest()))
    sizes, mapping = [0, 0, 0], {}
    for key, count in ordered:
        fold = min(range(3), key=lambda f: (sizes[f], f))
        mapping[key] = fold
        sizes[fold] += int(count)
    return np.asarray([mapping[key] for key in components], dtype=int)


def scientific_weights(scientific_groups, labeled):
    """Average parent errors within each original scientific key, then across keys."""
    counts = {}
    for keep, keys in zip(labeled, scientific_groups):
        if keep:
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
    weights = np.zeros(len(labeled), dtype=float)
    if counts:
        for i, (keep, keys) in enumerate(zip(labeled, scientific_groups)):
            if keep:
                weights[i] = sum(1.0 / (len(counts) * counts[key]) for key in keys)
    return weights


def prepare_peak_stats(x, y, available, reduce=True):
    """Include every training parent in normalization and every available rewrite pair."""
    mu = x[:, 0].mean(axis=0)
    scale = max(float(np.sqrt(np.mean(np.sum((x[:, 0] - mu) ** 2, axis=1)))), 1e-12)
    labeled = np.isfinite(y)
    if not labeled.any():
        raise ValueError("No measured peak targets in this training partition")
    supervised = (x[labeled, 0] - mu) / scale
    response = y[labeled]
    center, target_center = supervised.mean(axis=0), float(response.mean())
    differences = [(x[available[:, j], 0] - x[available[:, j], j]) / scale
                   for j in range(1, x.shape[1])]
    difference = np.concatenate(differences) if differences else np.empty((0, x.shape[2]))
    spanning = np.vstack((supervised, difference, center[None], (mu / scale)[None]))
    basis = qr(spanning.T, mode="economic", check_finite=True)[0] if reduce and len(spanning) < x.shape[2] else None
    a, d = supervised - center, difference
    if basis is not None:
        a, d = a @ basis, d @ basis
    return {
        "gram": a.T @ a, "rhs": a.T @ (response - target_center),
        "reg": d.T @ d, "basis": basis, "center": center, "mu": mu,
        "scale": scale, "target_center": target_center,
        "labeled_parents": int(labeled.sum()), "all_parents": len(y),
        "pairs": len(difference),
        "unlabeled_pairs": sum(int((available[:, j] & ~labeled).sum())
                               for j in range(1, x.shape[1])),
        "solve_dimension": a.shape[1],
    }


def solution_path(stats, lam, alphas):
    """Return full-width coefficients without dropping row-space directions."""
    values, vectors = eigh(stats["gram"] + lam * stats["reg"], check_finite=True)
    if values.min() < -1e-7 * max(1.0, float(values.max())):
        raise ValueError("Non-PSD peak-readout training matrix")
    values = np.maximum(values, 0)
    beta = vectors @ ((vectors.T @ stats["rhs"])[:, None] / (values[:, None] + alphas[None]))
    return beta if stats["basis"] is None else stats["basis"] @ beta


def predict_path(stats, beta, x):
    return ((x - stats["mu"]) / stats["scale"] - stats["center"]) @ beta + stats["target_center"]


class ConsistencyPeaks:
    """Fit seven independent peak readouts with missing-label-aware consistency.

    Every target-missing training parent still contributes to the normalization
    and rewrite penalty. Component folds and scientific-key-equal inner scores
    preserve the continuous UCNP task's original evaluation convention.
    """

    def __init__(self, alphas=ALPHAS, lambdas=LAMBDAS, reduce=True):
        self.alphas = np.asarray(alphas, dtype=float).copy()
        self.lambdas = np.asarray(lambdas, dtype=float).copy()
        for name, grid, minimum in (("alphas", self.alphas, 0), ("lambdas", self.lambdas, -1)):
            if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all():
                raise ValueError(f"{name} must be a nonempty finite vector")
            invalid_sign = (grid <= 0).any() if minimum == 0 else (grid < 0).any()
            if invalid_sign:
                raise ValueError(f"Invalid {name} values")
        self.reduce = bool(reduce)
        self.model_ = {}
        self.diagnostic_ = {}

    def fit(self, x, y, available, components, scientific_groups, progress=None,
            outer_fold="full"):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        available = np.asarray(available)
        if x.ndim != 3 or min(x.shape) < 1 or y.shape != (len(x), len(BANDS)):
            raise ValueError("X must be (samples, variants, features) and Y must be (samples, 7)")
        if available.dtype.kind != "b" or available.shape != x.shape[:2] or not available[:, 0].all():
            raise ValueError("available must be a boolean mask with every original present")
        if not np.isfinite(x[available]).all() or np.isinf(y).any():
            raise ValueError("Representations must be finite and targets may contain NaN, not infinity")
        x = np.where(available[:, :, None], x, 0)
        components = np.asarray(components, dtype=str)
        if components.shape != (len(x),) or any(not value.strip() for value in components):
            raise ValueError("Each sample requires a nonempty component ID")
        if len(scientific_groups) != len(x):
            raise ValueError("Each sample requires scientific-group memberships")
        groups, owners = [], {}
        for component, keys in zip(components, scientific_groups):
            if isinstance(keys, str) or not keys or any(not isinstance(key, str) or not key for key in keys):
                raise ValueError("Scientific memberships must be nonempty sequences of string keys")
            keys = tuple(sorted(set(keys)))
            groups.append(keys)
            for key in keys:
                if key in owners and owners[key] != component:
                    raise ValueError("A scientific group crosses component boundaries")
                owners[key] = component
        inner = component_inner(components, outer_fold)
        coefficients, centers, targets, diagnostics = [], [], [], []
        for band, name in enumerate(BANDS):
            target = y[:, band]
            weights = scientific_weights(groups, np.isfinite(target))
            if not weights.any():
                raise ValueError(f"{name}: no measured training targets; no model was fabricated")
            scores = np.zeros((len(self.lambdas), len(self.alphas)))
            for fold in range(3):
                train, valid = inner != fold, (inner == fold) & np.isfinite(target)
                if not valid.any():
                    continue
                if not np.isfinite(target[train]).any():
                    raise ValueError(f"{name}: inner training fold {fold} has no measured target")
                stats = prepare_peak_stats(x[train], target[train], available[train], self.reduce)
                for li, lam in enumerate(self.lambdas):
                    if progress:
                        progress(band, fold, float(lam), stats["solve_dimension"])
                    beta = solution_path(stats, lam, self.alphas)
                    prediction = predict_path(stats, beta, x[valid])
                    errors = np.where(available[valid, :, None],
                                      abs(prediction - target[valid, None, None]), 0)
                    parent_error = errors.sum(axis=1) / available[valid].sum(axis=1)[:, None]
                    scores[li] += weights[valid] @ parent_error
            if not np.isfinite(scores).all():
                raise ValueError(f"{name}: non-finite parameter selection score")
            _, _, _, li, ai = min((float(scores[i, j]), -float(alpha), float(lam), i, j)
                                 for i, lam in enumerate(self.lambdas)
                                 for j, alpha in enumerate(self.alphas))
            stats = prepare_peak_stats(x, target, available, self.reduce)
            if progress:
                progress(band, None, float(self.lambdas[li]), stats["solve_dimension"])
            beta = solution_path(stats, self.lambdas[li], self.alphas[[ai]])[:, 0]
            coefficients.append(beta)
            centers.append(stats["center"])
            targets.append(stats["target_center"])
            diagnostics.append({
                "band": name, "alpha": float(self.alphas[ai]),
                "consistency_lambda": float(self.lambdas[li]),
                "inner_scientific_group_equal_expression_mae": float(scores[li, ai]),
                "training_labeled_parents": stats["labeled_parents"],
                "training_all_parents": stats["all_parents"],
                "consistency_pairs_including_unlabeled": stats["pairs"],
                "consistency_pairs_with_missing_target": stats["unlabeled_pairs"],
                "solver_dimension": stats["solve_dimension"],
            })
        self.model_ = {
            "beta": np.stack(coefficients, axis=1), "feature_centers": np.stack(centers),
            "target_centers": np.asarray(targets), "mu": stats["mu"], "scale": stats["scale"],
        }
        self.diagnostic_ = {"task": "ucnp_peaks", "bands": list(BANDS),
                            "full_dimension": x.shape[-1], "training_samples": len(x),
                            "outer_fold": str(outer_fold), "band_parameters": diagnostics}
        return self

    def predict(self, x):
        if not self.model_:
            raise ValueError("Peak model is not fitted; train or load it first")
        x = np.asarray(x, dtype=float)
        state = self.model_
        if x.ndim not in (2, 3) or x.shape[-1] != state["beta"].shape[0] or not np.isfinite(x).all():
            raise ValueError("Prediction X must be finite 2D/3D representations with matching dimension")
        z = (x - state["mu"]) / state["scale"]
        offset = state["target_centers"] - np.sum(state["feature_centers"].T * state["beta"], axis=0)
        return z @ state["beta"] + offset

    def save(self, path):
        if not self.model_:
            raise ValueError("Peak model is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, schema_version=np.asarray(SCHEMA_VERSION),
                                model_type=np.asarray("consistency_peaks"),
                                diagnostic_json=np.asarray(json.dumps(self.diagnostic_, allow_nan=False)),
                                **self.model_)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as bundle:
            required = {"schema_version", "model_type", "diagnostic_json", "beta", "feature_centers",
                        "target_centers", "mu", "scale"}
            if not required.issubset(bundle.files):
                raise ValueError("Incomplete UCNP peak checkpoint")
            version = bundle["schema_version"]
            if version.shape != () or version.dtype.kind not in "iu" or int(version) != SCHEMA_VERSION:
                raise ValueError("Unsupported UCNP peak checkpoint version")
            model_type = bundle["model_type"]
            if model_type.shape != () or model_type.dtype.kind != "U" or str(model_type) != "consistency_peaks":
                raise ValueError("Not a consistency peak checkpoint")
            model = {key: bundle[key] for key in required - {"schema_version", "model_type", "diagnostic_json"}}
            beta = model["beta"]
            if beta.ndim != 2 or beta.shape[1] != 7 or beta.shape[0] < 1:
                raise ValueError("Invalid peak coefficient dimensions")
            dimension = beta.shape[0]
            shapes = {"feature_centers": (7, dimension), "target_centers": (7,),
                      "mu": (dimension,), "scale": ()}
            if any(model[key].shape != shape for key, shape in shapes.items()):
                raise ValueError("Invalid peak preprocessing dimensions")
            if any(value.dtype.kind not in "fiu" or not np.isfinite(value).all() for value in model.values()):
                raise ValueError("Peak checkpoint contains invalid values")
            if float(model["scale"]) <= 0:
                raise ValueError("Peak checkpoint scale must be positive")
            raw = bundle["diagnostic_json"]
            if raw.shape != () or raw.dtype.kind != "U":
                raise ValueError("Invalid peak diagnostics")
            def reject_constant(value):
                raise ValueError(f"Non-finite peak diagnostic: {value}")
            diagnostic = json.loads(str(raw), parse_constant=reject_constant)
            if (not isinstance(diagnostic, dict) or diagnostic.get("task") != "ucnp_peaks"
                    or diagnostic.get("full_dimension") != dimension or diagnostic.get("bands") != list(BANDS)):
                raise ValueError("Peak diagnostics do not match the checkpoint")
        fitted = cls()
        model["scale"] = float(model["scale"])
        fitted.model_, fitted.diagnostic_ = model, diagnostic
        return fitted
