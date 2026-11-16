"""Extract local representations, then train, save and apply consistency readouts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from consistency_readout import (ConsistencyRidge, ConsistencyPeaks, ConsistencyDetection,
                                 ALPHAS, LAMBDAS, inner_folds, load_model)
from consistency_readout.data import TrainingData, load_data, load_features
from consistency_readout.ucnp import BANDS, load_peaks, load_detection, task_for_path
from consistency_readout.peaks import scientific_weights

ROOT = Path(__file__).resolve().parent
DATASETS = ("carbon_dot", "biochar", "perovskite", "ucnp_peaks", "ucnp_detection")
INNER_SEED = 20260923


def write_csv(path: Path, rows: list[dict]) -> None:
    """Save a new result table without replacing an existing file."""
    if not rows:
        raise ValueError("No predictions or metrics to save")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reserve_directory(path: Path) -> Path:
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite a nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def fit_readout(data: TrainingData, keep: np.ndarray, prefix: str) -> ConsistencyRidge:
    """Select parameters inside the training partition, then fit that partition."""
    print(f"{prefix}: fit {int(keep.sum())} samples", flush=True)
    inner = inner_folds(data.ids[keep], n=3, seed=INNER_SEED)
    model = ConsistencyRidge()
    reported = set()

    def progress(fold, lam, dimension):
        if fold not in reported:
            print(f"{prefix}: inner fold {fold + 1}/3, solve dimension {dimension}", flush=True)
            reported.add(fold)
        if fold == 2 and lam == model.lambdas[-1]:
            print(f"{prefix}: parameter search complete; final fit", flush=True)

    model.fit(data.x[keep], data.targets[keep], data.available[keep],
              inner_splits=inner, progress=progress)
    print(f"{prefix}: fitted, alpha={model.diagnostic_['alpha']:g}, "
          f"lambda={model.diagnostic_['lambda']:g}", flush=True)
    return model


def train(data_path: Path, model_path: Path, threads: int = 2) -> Path:
    """Fit one reusable readout from all supplied training samples."""
    if threads < 1:
        raise ValueError("threads must be positive")
    model_path = Path(model_path)
    if model_path.exists():
        raise FileExistsError(f"Refusing to replace an existing model: {model_path}")
    print("[train] load training data", flush=True)
    task = task_for_path(data_path)
    data = load_peaks(data_path) if task == "peaks" else load_detection(data_path) if task == "detection" else load_data(data_path)
    with threadpool_limits(limits=threads):
        if task == "peaks":
            model = fit_peaks(data, np.ones(len(data.ids), dtype=bool), "[train]")
        elif task == "detection":
            model = fit_detection(data, "[train]")
        else:
            model = fit_readout(data, np.ones(len(data.ids), dtype=bool), "[train]")
    model.diagnostic_["input_hashes"] = data.inputs
    if task == "scalar":
        model.diagnostic_["inner_validation"] = "three training-only folds grouped by sample_id"
    model.diagnostic_["training_samples"] = len(data.train_ids) if task == "detection" else len(data.ids)
    model.save(model_path)
    print(f"[train] saved model: {model_path.resolve()}", flush=True)
    return model_path


def predict(model_path: Path, features_path: Path, output_path: Path, threads: int = 2) -> Path:
    """Load a saved readout and predict new, dimension-compatible features."""
    if threads < 1:
        raise ValueError("threads must be positive")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing predictions: {output_path}")
    print("[predict] load model and features", flush=True)
    model = load_model(Path(model_path))
    ids, features = load_features(Path(features_path))
    with threadpool_limits(limits=threads):
        predictions = model.predict_proba(features) if isinstance(model, ConsistencyDetection) else model.predict(features)
    if predictions.ndim == 1:
        rows = [dict(sample_id=str(sid), prediction=float(value)) for sid, value in zip(ids, predictions)]
    elif isinstance(model, ConsistencyDetection):
        rows = [dict(sample_id=str(sid), **{f"prob_{band}": float(value[j]) for j, band in enumerate(BANDS)},
                     **{f"present_{band}": int(value[j] >= .5) for j, band in enumerate(BANDS)})
                for sid, value in zip(ids, predictions)]
    else:
        rows = [dict(sample_id=str(sid), **{f"prediction_{band}": float(value[j]) for j, band in enumerate(BANDS)})
                for sid, value in zip(ids, predictions)]
    write_csv(output_path, rows)
    print(f"[predict] saved {len(rows)} predictions: {output_path.resolve()}", flush=True)
    return output_path


def fit_peaks(data, keep, prefix, outer_fold="full"):
    reported = set()

    def progress(band, inner, lam, dimension):
        key = (band, inner)
        if key not in reported:
            stage = "final fit" if inner is None else f"inner fold {inner + 1}/3"
            print(f"{prefix}: {BANDS[band]} {stage}, solve dimension {dimension}", flush=True)
            reported.add(key)

    print(f"{prefix}: seven-band peak fit, {int(keep.sum())} training samples", flush=True)
    groups = [keys for keys, selected in zip(data.scientific_groups, keep) if selected]
    return ConsistencyPeaks().fit(data.x[keep], data.targets[keep], data.available[keep],
                                  data.components[keep], groups, progress=progress, outer_fold=outer_fold)


def fit_detection(data, prefix):
    print(f"{prefix}: fit {len(data.train_ids)} training samples; validation excluded", flush=True)
    return ConsistencyDetection().fit(
        data.x_train, data.y_train, data.projection_mask, data.reorder,
        data.reorder_parents, data.paraphrase, data.paraphrase_parents,
        progress=lambda stage: print(f"{prefix}: {stage}", flush=True))


def demo_peaks(name, data, output, model_path):
    predictions = np.full((*data.available.shape, len(BANDS)), np.nan)
    selections = []
    for fold in range(5):
        keep = data.folds != fold
        model = fit_peaks(data, keep, f"[{name}] outer fold {fold + 1}/5", outer_fold=fold)
        predictions[~keep] = model.predict(data.x[~keep])
        selections.append({"outer_fold": fold, **model.diagnostic_})
    rows, metrics = [], []
    for i, sid in enumerate(data.ids):
        for variant_index, variant in enumerate(data.variants):
            if data.available[i, variant_index]:
                for band_index, band in enumerate(BANDS):
                    target = data.targets[i, band_index]
                    rows.append(dict(sample_id=str(sid), outer_fold=int(data.folds[i]), band=band,
                                     variant=variant, target=float(target) if np.isfinite(target) else None,
                                     prediction=float(predictions[i, variant_index, band_index])))
    for band_index, band in enumerate(BANDS):
        for variant_index, variant in enumerate(data.variants):
            keep = np.isfinite(data.targets[:, band_index]) & data.available[:, variant_index]
            if not keep.any():
                continue
            weights = scientific_weights(data.scientific_groups, keep)[keep]
            truth = data.targets[keep, band_index]
            pred = predictions[keep, variant_index, band_index]
            if not np.isfinite(pred).all():
                raise ValueError("Incomplete UCNP peak predictions")
            error = pred - truth
            total = np.sum(weights * (truth - np.sum(weights * truth)) ** 2)
            metrics.append(dict(dataset=name, band=band, variant=variant, n=int(keep.sum()),
                                mae=float(np.sum(weights * abs(error))),
                                rmse=float(np.sqrt(np.sum(weights * error ** 2))),
                                r2=float(1 - np.sum(weights * error ** 2) / total) if total else None,
                                rewrite_gap_mean=float(np.sum(weights * abs(pred - predictions[keep, 0, band_index]))) if variant_index else None))
    output.mkdir()
    write_csv(output / "predictions.csv", rows)
    fitted = fit_peaks(data, np.ones(len(data.ids), dtype=bool), f"[{name}] reusable model")
    fitted.diagnostic_["input_hashes"] = data.inputs
    fitted.save(model_path)
    return metrics, dict(task="peak_regression", inputs=data.inputs, samples=len(data.ids),
                         outer_validation="fixed component folds; scientific-group-equal scoring",
                         outer_fit_parameters=selections, full_fit_parameters=fitted.diagnostic_)


def demo_detection(name, data, output, model_path):
    from sklearn.metrics import roc_auc_score

    if not len(data.valid_ids):
        raise ValueError("Detection demo needs the held-out validation partition")
    fitted = fit_detection(data, f"[{name}]")
    probabilities = fitted.predict_proba(data.x_valid)
    metrics, rows = [], []
    for j, band in enumerate(BANDS):
        known = data.y_valid[:, j] >= 0
        truth, prob = data.y_valid[known, j], probabilities[known, j]
        present = prob >= .5
        tp = int(np.sum(present & (truth == 1)))
        fp = int(np.sum(present & (truth == 0)))
        fn = int(np.sum(~present & (truth == 1)))
        denominator = 2 * tp + fp + fn
        metrics.append(dict(dataset=name, band=band, variant="original", n=int(known.sum()),
                            f1=(float(2 * tp / denominator) if denominator else 0.) if known.any() else None,
                            auc=float(roc_auc_score(truth, prob)) if len(np.unique(truth)) == 2 else None))
    aucs = [row["auc"] for row in metrics if row["auc"] is not None]
    f1s = [row["f1"] for row in metrics if row["f1"] is not None]
    metrics.append(dict(dataset=name, band="macro", variant="original", n=len(data.valid_ids),
                        f1=float(np.mean(f1s)) if f1s else None,
                        auc=float(np.mean(aucs)) if aucs else None))
    for i, sid in enumerate(data.valid_ids):
        for j, band in enumerate(BANDS):
            rows.append(dict(sample_id=str(sid), paper_id=str(data.valid_papers[i]), band=band,
                             target=int(data.y_valid[i, j]) if data.y_valid[i, j] >= 0 else None,
                             probability=float(probabilities[i, j]), present=int(probabilities[i, j] >= .5)))
    output.mkdir()
    write_csv(output / "predictions.csv", rows)
    fitted.diagnostic_["input_hashes"] = data.inputs
    fitted.save(model_path)
    return metrics, dict(task="band_detection", inputs=data.inputs,
                         training_samples=len(data.train_ids), validation_samples=len(data.valid_ids),
                         projection_samples=int(data.projection_mask.sum()),
                         validation="fixed external-paper holdout; masked cells excluded",
                         full_fit_parameters=fitted.diagnostic_)


def evaluate(name: str, data: TrainingData, predictions: np.ndarray) -> list[dict]:
    rows = []
    for column, variant in enumerate(data.variants):
        keep = data.available[:, column]
        if not keep.any():
            continue
        truth = data.targets[keep]
        predicted = predictions[keep, column]
        error = predicted - truth
        total = float(np.sum((truth - truth.mean()) ** 2))
        rows.append({
            "dataset": name,
            "variant": variant,
            "n": int(keep.sum()),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": float(1.0 - np.sum(error ** 2) / total) if total else None,
            "rewrite_gap_mean": float(np.mean(np.abs(predicted - predictions[keep, 0]))) if column else None,
        })
    return rows


def demo_dataset(name: str, data: TrainingData, output: Path, model_path: Path):
    """Evaluate held-out predictions, then save a model fitted to all samples."""
    folds = data.folds
    if folds is None:
        folds = inner_folds(data.ids, n=5, seed=INNER_SEED)
        outer_protocol = "five deterministic sample_id folds generated from input sample IDs"
    else:
        if set(folds) != set(range(5)):
            raise ValueError(f"{name}: demo outer_fold must cover all five folds (0..4)")
        outer_protocol = "fixed five outer_fold assignments supplied with the data"
    predictions = np.full(data.available.shape, np.nan)
    selections = []
    for fold in range(5):
        keep = folds != fold
        model = fit_readout(data, keep, f"[{name}] outer fold {fold + 1}/5")
        held_out = ~keep
        predicted = model.predict(data.x[held_out])
        if not np.isfinite(predicted[data.available[held_out]]).all():
            raise ValueError(f"{name}: non-finite held-out predictions")
        predictions[held_out] = predicted
        selections.append({"outer_fold": fold, **model.diagnostic_})
    rows = []
    for i, sid in enumerate(data.ids):
        for j, variant in enumerate(data.variants):
            if data.available[i, j]:
                rows.append({"sample_id": str(sid), "outer_fold": int(folds[i]),
                             "variant": variant, "target": float(data.targets[i]),
                             "prediction": float(predictions[i, j])})
    output.mkdir()
    write_csv(output / "predictions.csv", rows)
    fitted = fit_readout(data, np.ones(len(data.ids), dtype=bool), f"[{name}] reusable model")
    fitted.diagnostic_["input_hashes"] = data.inputs
    fitted.diagnostic_["training_samples"] = len(data.ids)
    fitted.save(model_path)
    print(f"[{name}] saved reusable model: {model_path.resolve()}", flush=True)
    return evaluate(name, data, predictions), {
        "outer_validation": outer_protocol,
        "inputs": data.inputs,
        "samples": len(data.ids),
        "variants": list(data.variants),
        "outer_fit_parameters": selections,
        "full_fit_parameters": fitted.diagnostic_,
    }


def demo(datasets: list[str], data_dir: Path, output_dir: Path | None = None,
         model_dir: Path | None = None, threads: int = 2) -> Path:
    if not datasets or len(set(datasets)) != len(datasets) or any(name not in DATASETS for name in datasets):
        raise ValueError(f"Select unique datasets from: {', '.join(DATASETS)}")
    if threads < 1:
        raise ValueError("threads must be positive")
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output = reserve_directory(output_dir if output_dir is not None else ROOT / "results" / run_id)
    models = reserve_directory(model_dir if model_dir is not None else ROOT / "models" / run_id)
    manifest_path = output / "manifest.json"
    start = time.perf_counter()
    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "readout": "consistency",
        "inner_validation": "three folds grouped by sample_id inside each outer training partition",
        "inner_seed": INNER_SEED,
        "alphas": ALPHAS.tolist(),
        "lambdas": LAMBDAS.tolist(),
        "blas_threads": threads,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "datasets": {},
    }

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    metrics = []
    try:
        with threadpool_limits(limits=threads):
            for name in datasets:
                print(f"[{name}] load training data and frozen features", flush=True)
                folder = Path(data_dir) / name
                task = task_for_path(folder)
                data = load_peaks(folder) if task == "peaks" else load_detection(folder) if task == "detection" else load_data(folder)
                model_path = models / f"{name}.npz"
                if task == "peaks":
                    measured, details = demo_peaks(name, data, output / name, model_path)
                elif task == "detection":
                    measured, details = demo_detection(name, data, output / name, model_path)
                else:
                    measured, details = demo_dataset(name, data, output / name, model_path)
                details["model_file"] = Path(os.path.relpath(model_path, output)).as_posix()
                manifest["datasets"][name] = details
                metrics.extend(measured)
                del data
        write_csv(output / "metrics.csv", metrics)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        save_manifest()
    print(f"Completed. Results: {output.resolve()}", flush=True)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Extract local features, then train and use consistency readout models.")
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract", help="Generate feature caches from prompts and a local pretrained LLM")
    extraction.add_argument("--data", type=Path, required=True, help="Directory containing prompts.jsonl.gz; labels are not required")
    extraction.add_argument("--model", type=Path, required=True, help="Existing local pretrained LLM directory, not a trained readout NPZ")
    extraction.add_argument("--output-dir", type=Path, default=None, help="Feature cache directory (default: --data)")
    extraction.add_argument("--batch-size", type=int, default=2)
    extraction.add_argument("--chunk-size", type=int, default=6000)
    extraction.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    extraction.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=None,
                            help="Default: bfloat16 on CUDA, float32 on CPU")
    extraction.add_argument("--resume", action="store_true", help="Resume only matching, verified extraction checkpoints")
    training = commands.add_parser("train", help="Train and save one reusable readout")
    training.add_argument("--data", type=Path, required=True, help="Dataset folder or training NPZ with X and y")
    training.add_argument("--model", type=Path, required=True, help="New trained readout NPZ output file")
    training.add_argument("--threads", type=int, default=2)
    prediction = commands.add_parser("predict", help="Predict new features using a saved readout")
    prediction.add_argument("--model", type=Path, required=True, help="Saved trained readout NPZ file")
    prediction.add_argument("--features", type=Path, required=True, help="2D NPY/NPZ or directory containing extracted feature chunks")
    prediction.add_argument("--output", type=Path, required=True, help="New predictions CSV")
    prediction.add_argument("--threads", type=int, default=2)
    example = commands.add_parser("demo", help="Run bundled examples and save reusable readouts")
    example.add_argument("--datasets", nargs="+", choices=DATASETS, default=["carbon_dot", "biochar"])
    example.add_argument("--data-dir", type=Path, default=ROOT / "data")
    example.add_argument("--output-dir", type=Path, default=None)
    example.add_argument("--model-dir", type=Path, default=None)
    example.add_argument("--threads", type=int, default=2)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments or ["demo"])
    try:
        if args.command == "extract":
            # Optional extraction libraries stay unloaded for CPU-only readout users.
            from consistency_readout.extraction import extract_features
            extract_features(args.data, args.model, output_dir=args.output_dir,
                             batch_size=args.batch_size, chunk_size=args.chunk_size,
                             device=args.device, dtype=args.dtype, resume=args.resume)
        elif args.command == "train":
            train(args.data, args.model, args.threads)
        elif args.command == "predict":
            predict(args.model, args.features, args.output, args.threads)
        else:
            demo(args.datasets, args.data_dir, args.output_dir, args.model_dir, args.threads)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
