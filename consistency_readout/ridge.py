"""Consistency-regularized linear readout with exact QR reduction and model persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional
from zipfile import BadZipFile

import numpy as np
from scipy.linalg import eigh, qr

ALPHAS = np.logspace(-6, 6, 25)
LAMBDAS = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
SCHEMA_VERSION = 1


def prepare_stats(x: np.ndarray, y: np.ndarray, available: np.ndarray,
                  reduce: bool = True) -> dict:
    """Build the original-prompt regression and rewrite-consistency quadratic terms."""
    mu = x[:, 0].mean(axis=0)
    scale = max(float(np.sqrt(np.mean(np.sum((x[:, 0] - mu) ** 2, axis=1)))), 1e-12)
    basis = None
    u = x[available]
    if reduce and x.shape[2] > 128 and len(u) < x.shape[2]:
        basis = qr(u.T, mode="economic", check_finite=True)[0]
        z = (x @ basis) / scale
    else:
        z = x / scale
    xx = z[:, 0]
    weights = np.ones(len(y))
    xm = np.average(xx, axis=0, weights=weights)
    ym = float(np.average(y, weights=weights))
    xc = xx - xm
    gram = xc.T @ (xc * weights[:, None])
    rhs = xc.T @ (weights * (y - ym))
    reg = np.zeros_like(gram)
    rewrite_weights = 1.0 / np.maximum(available[:, 1:].sum(axis=1), 1)
    for column in range(1, x.shape[1]):
        keep = available[:, column]
        diff = z[keep, column] - z[keep, 0]
        reg += diff.T @ (diff * rewrite_weights[keep, None])
    return {"gram": gram, "rhs": rhs, "center": xm, "full_center": mu / scale,
            "target_center": ym, "scale": scale, "reg": reg, "basis": basis,
            "dimension": x.shape[2]}


def solve_candidates(st: dict, lam: float, alphas: np.ndarray = ALPHAS):
    """Solve the same consistency objective for all ridge strengths."""
    val, q = eigh(st["gram"] + lam * st["reg"], check_finite=True)
    if val.min() < -1e-7 * max(1.0, float(val.max())):
        raise ValueError("Non-PSD training Gram matrix")
    val = np.maximum(val, 0.0)
    coefficients = q @ ((q.T @ st["rhs"])[:, None] / (val[:, None] + alphas[None, :]))
    return coefficients, val, q


def predict_from_basis(st: dict, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    z = x if st["basis"] is None else x @ st["basis"]
    return (z / st["scale"] - st["center"]) @ coefficients + st["target_center"]


def _real_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    return array if array.dtype in (np.float32, np.float64) else array.astype(np.float64)


class ConsistencyRidge:
    """Fit a single consistency readout, then predict, save, or reload it.

    Training X has shape (samples, variants, features), with the original in
    variant 0. An optional boolean mask marks missing rewrites. If omitted,
    entirely NaN rewrite vectors are inferred as missing.
    """

    def __init__(self, alphas: np.ndarray = ALPHAS, lambdas: np.ndarray = LAMBDAS,
                 reduce: bool = True):
        self.alphas = np.asarray(alphas, dtype=float).copy()
        self.lambdas = np.asarray(lambdas, dtype=float).copy()
        for name, grid, positive in [("alphas", self.alphas, True), ("lambdas", self.lambdas, False)]:
            invalid_sign = (grid <= 0).any() if positive else (grid < 0).any()
            if (grid.ndim != 1 or not grid.size or not np.isfinite(grid).all()
                    or invalid_sign):
                raise ValueError(f"{name} must be a nonempty finite 1D grid with "
                                 f"{'positive' if positive else 'nonnegative'} values")
        self.reduce = bool(reduce)
        self.model_ = {}
        self.diagnostic_ = {}

    def fit(self, x: np.ndarray, y: np.ndarray, available: Optional[np.ndarray] = None,
            inner_splits: Optional[np.ndarray] = None,
            progress: Optional[Callable] = None) -> "ConsistencyRidge":
        """Select alpha/lambda on training-only inner folds and fit all training data."""
        x = _real_array(x, "X")
        y = _real_array(y, "y")
        if x.ndim != 3 or min(x.shape) < 1 or x.shape[0] < 2:
            raise ValueError("X must have shape (at least 2 samples, variants, features)")
        if y.shape != (len(x),) or not np.isfinite(y).all():
            raise ValueError("y must be a finite vector with one value per sample")
        if available is None:
            available = ~np.isnan(x).all(axis=-1)
        else:
            available = np.asarray(available)
        if available.dtype.kind != "b" or available.shape != x.shape[:2]:
            raise ValueError("available must be a boolean mask with shape (samples, variants)")
        if not available[:, 0].all():
            raise ValueError("Every sample must have an original representation")
        if not np.isfinite(x[available]).all():
            raise ValueError("Available representations must contain finite values")
        x = np.where(available[:, :, None], x, 0)
        if inner_splits is None:
            from .folds import inner_folds
            inner_splits = inner_folds(np.arange(len(y)), n=3)
        inner_splits = np.asarray(inner_splits)
        if (inner_splits.shape != (len(y),) or inner_splits.dtype.kind not in "iu"
                or (inner_splits < 0).any()):
            raise ValueError("inner_splits must be a nonnegative integer vector, one fold per sample")
        unique = np.unique(inner_splits)
        if len(unique) < 2 or not np.array_equal(unique, np.arange(len(unique))):
            raise ValueError("inner_splits must cover consecutive folds 0..k-1, with at least two folds")
        n_inner = len(unique)
        scores = np.zeros((len(self.lambdas), len(self.alphas)))
        for fold in range(n_inner):
            train = inner_splits != fold
            valid = ~train
            st = prepare_stats(x[train], y[train], available[train], self.reduce)
            for li, lam in enumerate(self.lambdas):
                coefficients, _, _ = solve_candidates(st, float(lam), self.alphas)
                predictions = predict_from_basis(st, coefficients, x[valid])
                errors = np.where(available[valid, :, None],
                                  np.abs(predictions - y[valid, None, None]), 0.0)
                scores[li] += (errors.sum(axis=1) / available[valid].sum(axis=1)[:, None]).sum(axis=0) / len(y)
                if progress:
                    progress(fold, float(lam), st["gram"].shape[0])
        # Lowest MAE, then largest alpha, then smallest lambda.
        _, _, _, li, ai = min((scores[i, j], -float(a), float(l), i, j)
                             for i, l in enumerate(self.lambdas)
                             for j, a in enumerate(self.alphas))
        best_alpha, best_lambda = float(self.alphas[ai]), float(self.lambdas[li])
        st = prepare_stats(x, y, available, self.reduce)
        coefficients, val, q = solve_candidates(st, best_lambda, self.alphas)
        beta = coefficients[:, ai]
        full_beta = beta if st["basis"] is None else st["basis"] @ beta
        diagonal = np.einsum("ij,ij->j", q, st["gram"] @ q)
        self.diagnostic_ = {
            "alpha": best_alpha, "lambda": best_lambda, "inner_mae": float(scores[li, ai]),
            "scale": st["scale"], "coefficient_l2": float(np.linalg.norm(full_beta)),
            "effective_df": float(np.sum(diagonal / (val + best_alpha))),
            "training_target_mean": float(y.mean()), "full_dimension": x.shape[2],
            "solve_dimension": st["gram"].shape[0],
            "reduction": "untruncated orthonormal training-row-space" if st["basis"] is not None else "none",
        }
        self.model_ = {"beta": full_beta, "feature_center": st["full_center"],
                       "target_center": st["target_center"], "scale": st["scale"],
                       "alpha": best_alpha, "consistency_lambda": best_lambda}
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from 2D original/new representations or a 3D variant array."""
        if not self.model_:
            raise ValueError("Model is not fitted; call fit() or load() first")
        x = _real_array(x, "X")
        if x.ndim not in (2, 3) or x.shape[-1] != len(self.model_["beta"]):
            raise ValueError(f"X must be 2D or 3D with final dimension {len(self.model_['beta'])}")
        if not np.isfinite(x).all():
            raise ValueError("Prediction representations must contain finite values")
        return (x / self.model_["scale"] - self.model_["feature_center"]) @ self.model_["beta"] + self.model_["target_center"]

    def save(self, path: str | Path) -> None:
        """Save all fitted prediction parameters to a non-pickle NPZ checkpoint."""
        if not self.model_:
            raise ValueError("Model is not fitted; call fit() before save()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, schema_version=np.asarray(SCHEMA_VERSION),
                                model_type=np.asarray("consistency_ridge"),
                                diagnostic_json=np.asarray(json.dumps(self.diagnostic_, allow_nan=False)),
                                **self.model_)

    @classmethod
    def load(cls, path: str | Path) -> "ConsistencyRidge":
        """Load and validate a fitted readout without loading pickled Python objects."""
        try:
            with np.load(path, allow_pickle=False) as bundle:
                required = {"schema_version", "beta", "feature_center", "target_center",
                            "scale", "alpha", "consistency_lambda", "diagnostic_json"}
                if not required.issubset(bundle.files):
                    raise ValueError("Checkpoint is missing required readout parameters")
                version = bundle["schema_version"]
                if version.shape != () or version.dtype.kind not in "iu" or int(version) != SCHEMA_VERSION:
                    raise ValueError("Unsupported readout checkpoint schema version")
                beta, center = bundle["beta"], bundle["feature_center"]
                if (beta.ndim != 1 or not beta.size or center.shape != beta.shape
                        or beta.dtype.kind != "f" or center.dtype.kind != "f"
                        or not np.isfinite(beta).all() or not np.isfinite(center).all()):
                    raise ValueError("Invalid checkpoint coefficient or feature-center dimensions/values")
                values = {}
                for key in ["target_center", "scale", "alpha", "consistency_lambda"]:
                    value = bundle[key]
                    if value.shape != () or value.dtype.kind not in "fiu" or not np.isfinite(value):
                        raise ValueError(f"Invalid checkpoint scalar: {key}")
                    values[key] = float(value)
                if values["scale"] <= 0 or values["alpha"] <= 0 or values["consistency_lambda"] < 0:
                    raise ValueError("Checkpoint scale/alpha must be positive and lambda nonnegative")
                raw_diagnostic = bundle["diagnostic_json"]
                if raw_diagnostic.shape != () or raw_diagnostic.dtype.kind != "U":
                    raise ValueError("Invalid checkpoint diagnostic_json")
                def reject_constant(value):
                    raise ValueError(f"Invalid non-finite diagnostic value: {value}")
                diagnostic = json.loads(str(raw_diagnostic), parse_constant=reject_constant)
                if not isinstance(diagnostic, dict) or diagnostic.get("full_dimension") != len(beta):
                    raise ValueError("Checkpoint diagnostics do not match coefficient dimension")
                if diagnostic.get("alpha") != values["alpha"] or diagnostic.get("lambda") != values["consistency_lambda"]:
                    raise ValueError("Checkpoint diagnostics do not match selected parameters")
            fitted = cls()
            fitted.model_ = {"beta": beta, "feature_center": center, **values}
            fitted.diagnostic_ = diagnostic
            return fitted
        except FileNotFoundError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, BadZipFile) as error:
            raise ValueError(f"Invalid readout checkpoint: {error}") from error
