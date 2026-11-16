"""Train, save, reload, and apply consistency readouts for scientific targets."""

from pathlib import Path
import numpy as np

__version__ = "0.1.0"

from .ridge import ConsistencyRidge, ALPHAS, LAMBDAS
from .folds import inner_folds
from .peaks import ConsistencyPeaks
from .detection import ConsistencyDetection


def load_model(path: str | Path):
    """Load a saved scalar, UCNP peak, or UCNP detection readout."""
    with np.load(path, allow_pickle=False) as bundle:
        kind = str(bundle["model_type"]) if "model_type" in bundle else "consistency_ridge"
    models = {"consistency_ridge": ConsistencyRidge,
              "consistency_peaks": ConsistencyPeaks,
              "consistency_detection": ConsistencyDetection}
    if kind not in models:
        raise ValueError(f"Unknown saved readout type: {kind}")
    return models[kind].load(path)

__all__ = [
    "ConsistencyRidge",
    "ConsistencyPeaks",
    "ConsistencyDetection",
    "load_model",
    "ALPHAS",
    "LAMBDAS",
    "inner_folds",
    "__version__",
]
