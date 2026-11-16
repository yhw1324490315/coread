"""Seven-band consistency readout with train-only spectral shrinkage."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from scipy.special import expit


SCHEMA_VERSION = 1
MODEL_TYPE = "consistency_detection"
N_BANDS = 7


def _matrix(values, name, dimension=None):
    values = np.asarray(values)
    if (values.ndim != 2 or values.dtype.kind not in "fiu" or not len(values)
            or not values.shape[1] or not np.isfinite(values).all()):
        raise ValueError(f"{name} must be a nonempty finite 2D real array")
    if dimension is not None and values.shape[1] != dimension:
        raise ValueError(f"{name} must have {dimension} features")
    return values.astype(np.float64, copy=False)


def _parents(values, size, n_samples, projection_mask, name):
    values = np.asarray(values)
    if (values.shape != (size,) or values.dtype.kind not in "iu"
            or (values < 0).any() or (values >= n_samples).any()):
        raise ValueError(f"{name} must contain one valid training-parent index per rewrite")
    values = values.astype(int, copy=False)
    if not projection_mask[values].all():
        raise ValueError(f"{name} contains a parent outside the projection training pool")
    return values


class ConsistencyDetection:
    """Fit the seven-band detection readout without using validation representations.

    ``projection_mask`` identifies the original training pool. Supplemental rows
    contribute to supervised fitting but not to centering or consistency penalties.
    Each paraphrase is retained as a separate pair, including repeated parent IDs.
    """

    def __init__(self, C: float = 0.1, kappa: float = 100.0,
                 max_iter: int = 4000, tol: float = 1e-6):
        if not np.isfinite(C) or C <= 0 or not np.isfinite(kappa) or kappa < 0:
            raise ValueError("C must be positive and kappa must be nonnegative")
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
            raise ValueError("max_iter and tol must be positive")
        self.C, self.kappa = float(C), float(kappa)
        self.max_iter, self.tol = int(max_iter), float(tol)
        self.model_ = {}
        self.diagnostic_ = {}

    def fit(self, X_original, Y_masked, projection_mask, X_reorder,
            reorder_parent_indices, X_paraphrase, paraphrase_parent_indices,
            progress=None):
        """Fit from training rows only; masked labels are -1, observed labels 0/1.

        There must be exactly one reorder for every projection-pool parent, while
        any number of paraphrases may reference the same projection-pool parent.
        ``progress``, when supplied, receives bounded stage-description strings.
        """
        original = _matrix(X_original, "X_original")
        n_samples, dimension = original.shape
        labels = np.asarray(Y_masked)
        if (labels.shape != (n_samples, N_BANDS) or labels.dtype.kind not in "fiu"
                or not np.isin(labels, [-1, 0, 1]).all()):
            raise ValueError("Y_masked must have seven columns containing only -1, 0 or 1")
        labels = labels.astype(int)
        projection_mask = np.asarray(projection_mask)
        if (projection_mask.dtype.kind != "b" or projection_mask.shape != (n_samples,)
                or projection_mask.sum() < 2):
            raise ValueError("projection_mask must select at least two training originals")
        reorder = _matrix(X_reorder, "X_reorder", dimension)
        para = _matrix(X_paraphrase, "X_paraphrase", dimension)
        reorder_parents = _parents(reorder_parent_indices, len(reorder), n_samples,
                                   projection_mask, "reorder_parent_indices")
        para_parents = _parents(paraphrase_parent_indices, len(para), n_samples,
                                projection_mask, "paraphrase_parent_indices")
        expected = np.flatnonzero(projection_mask)
        if not np.array_equal(np.sort(reorder_parents), expected):
            raise ValueError("Reorders must cover every projection-pool parent exactly once")
        for band in range(N_BANDS):
            observed = labels[:, band] >= 0
            if not np.array_equal(np.unique(labels[observed, band]), [0, 1]):
                raise ValueError(f"Band B{band + 1} needs both observed label classes")

        def notify(message):
            if progress is not None:
                progress(message)

        notify("scale: estimate center and standard deviation from the original training pool")
        pool = original[projection_mask]
        mu, sd = pool.mean(axis=0), pool.std(axis=0) + 1e-6
        notify("penalty: compute reordered and paraphrased training-pair covariance")
        difference = (original[reorder_parents] - reorder) / sd
        reorder_penalty = difference.T @ difference / len(difference)
        difference = (original[para_parents] - para) / sd
        para_penalty = difference.T @ difference / len(difference)
        del difference
        reorder_size = np.trace(reorder_penalty) / dimension
        para_size = np.trace(para_penalty) / dimension
        if (not np.isfinite(reorder_size) or not np.isfinite(para_size)
                or reorder_size <= 0 or para_size <= 0):
            raise ValueError("Both rewrite families need nonzero finite consistency variance")
        penalty = 0.5 * reorder_penalty / reorder_size + 0.5 * para_penalty / para_size
        del reorder_penalty, para_penalty
        notify(f"eigh: spectral shrinkage in {dimension} dimensions")
        eigenvalues, projection = np.linalg.eigh(penalty)
        scale = 1.0 / np.sqrt(1.0 + self.kappa * np.clip(eigenvalues, 0, None))
        del penalty
        transformed = ((original - mu) / sd) @ projection * scale

        from sklearn.linear_model import LogisticRegression

        coef, intercept, counts, iterations = [], [], [], []
        for band in range(N_BANDS):
            observed = labels[:, band] >= 0
            notify(f"logistic: fit band B{band + 1}/{N_BANDS} on {int(observed.sum())} observed labels")
            fitted = LogisticRegression(C=self.C, max_iter=self.max_iter, tol=self.tol).fit(
                transformed[observed], labels[observed, band])
            coef.append(fitted.coef_[0])
            intercept.append(fitted.intercept_[0])
            counts.append(int(observed.sum()))
            iterations.append(int(fitted.n_iter_[0]))
        self.model_ = {"mu": mu, "sd": sd, "projection": projection, "scale": scale,
                       "coef": np.asarray(coef), "intercept": np.asarray(intercept)}
        self.diagnostic_ = {
            "C": self.C, "kappa": self.kappa, "max_iter": self.max_iter, "tol": self.tol,
            "training_samples": n_samples, "projection_samples": int(projection_mask.sum()),
            "supplemental_samples": int((~projection_mask).sum()),
            "reorder_pairs": len(reorder), "paraphrase_pairs": len(para),
            "n_train_per_band": counts, "iterations_per_band": iterations,
            "full_dimension": dimension, "threshold": 0.5,
        }
        notify("complete: seven-band consistency detection model fitted")
        return self

    def predict_proba(self, X):
        """Return one detection probability per sample and band, in B1..B7 order."""
        if not self.model_:
            raise ValueError("Model is not fitted; call fit() or load() first")
        x = _matrix(X, "X", len(self.model_["mu"]))
        transformed = ((x - self.model_["mu"]) / self.model_["sd"]) @ self.model_["projection"] * self.model_["scale"]
        return expit(transformed @ self.model_["coef"].T + self.model_["intercept"])

    def predict(self, X):
        """Return binary band detections at the fixed probability threshold 0.5."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def save(self, path: str | Path):
        """Save all prediction parameters without pickle, refusing to overwrite."""
        if not self.model_:
            raise ValueError("Model is not fitted; call fit() before save()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, schema_version=np.asarray(SCHEMA_VERSION),
                                model_type=np.asarray(MODEL_TYPE),
                                diagnostic_json=np.asarray(json.dumps(self.diagnostic_, allow_nan=False)),
                                **self.model_)

    @classmethod
    def load(cls, path: str | Path):
        """Load a validated numeric seven-band checkpoint; pickled arrays are forbidden."""
        try:
            with np.load(path, allow_pickle=False) as bundle:
                required = {"schema_version", "model_type", "diagnostic_json", "mu", "sd",
                            "projection", "scale", "coef", "intercept"}
                if not required.issubset(bundle.files):
                    raise ValueError("Checkpoint is missing detection parameters")
                version, kind = bundle["schema_version"], bundle["model_type"]
                if version.shape != () or version.dtype.kind not in "iu" or int(version) != SCHEMA_VERSION:
                    raise ValueError("Unsupported detection checkpoint schema version")
                if kind.shape != () or kind.dtype.kind != "U" or str(kind) != MODEL_TYPE:
                    raise ValueError("Checkpoint is not a consistency detection model")
                values = {key: bundle[key] for key in ["mu", "sd", "projection", "scale", "coef", "intercept"]}
                mu = values["mu"]
                if mu.ndim != 1 or not len(mu):
                    raise ValueError("Invalid detection feature dimension")
                d = len(mu)
                expected = {"mu": (d,), "sd": (d,), "projection": (d, d),
                            "scale": (d,), "coef": (N_BANDS, d), "intercept": (N_BANDS,)}
                for key, shape in expected.items():
                    value = values[key]
                    if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                        raise ValueError(f"Invalid checkpoint shape or values: {key}")
                if (values["sd"] <= 0).any() or (values["scale"] <= 0).any() or (values["scale"] > 1).any():
                    raise ValueError("Checkpoint standard deviations/scales must be positive, with scales <= 1")
                raw = bundle["diagnostic_json"]
                if raw.shape != () or raw.dtype.kind != "U":
                    raise ValueError("Invalid checkpoint diagnostic_json")
                def reject_constant(value):
                    raise ValueError(f"Non-finite diagnostic value: {value}")
                diagnostic = json.loads(str(raw), parse_constant=reject_constant)
                if not isinstance(diagnostic, dict) or diagnostic.get("full_dimension") != d:
                    raise ValueError("Checkpoint diagnostics disagree with feature dimension")
            fitted = cls(C=diagnostic["C"], kappa=diagnostic["kappa"],
                         max_iter=diagnostic["max_iter"], tol=diagnostic["tol"])
            fitted.model_, fitted.diagnostic_ = values, diagnostic
            return fitted
        except FileNotFoundError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, BadZipFile) as error:
            raise ValueError(f"Invalid detection checkpoint: {error}") from error
