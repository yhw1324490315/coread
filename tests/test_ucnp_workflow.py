"""Complete UCNP workflows on small, explicitly synthetic schema fixtures."""

import contextlib
import csv
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run as runner
from consistency_readout import ConsistencyDetection, ConsistencyPeaks, load_model
from consistency_readout.ucnp import BANDS, load_detection, load_peaks


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_fixture(folder, labels, prompts, features):
    folder.mkdir(parents=True)
    with (folder / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    with gzip.open(folder / "prompts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in prompts:
            stream.write(json.dumps(row) + "\n")
    np.savez_compressed(folder / "features.npz",
                        sample_id=np.asarray([row["sample_id"] for row in prompts]),
                        features=np.asarray(features, dtype=np.float32))


class UCNPWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".verification"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="ucnp_workflow_", dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.data = self.work / "data"
        self.make_peaks()
        self.make_detection()

    def make_peaks(self):
        rng = np.random.default_rng(712)
        originals = rng.normal(size=(45, 6)).astype(np.float32)
        weights = rng.normal(size=(6, 7))
        targets = originals @ weights + np.array([360, 450, 475, 520, 545, 655, 800])
        labels, prompts, features = [], [], []
        for i, original in enumerate(originals):
            component = i // 3
            row = dict(sample_id=f"peak_{i}", component_id=f"component_{component}",
                       outer_fold=component % 5,
                       scientific_group_ids=json.dumps([f"paper_{component:03d}", f"group_{component:03d}"]))
            for j, band in enumerate(BANDS):
                row[f"target_{band}"] = "" if (i + j) % 11 == 0 else float(targets[i, j])
            labels.append(row)
            for j, variant in enumerate(["original", "equivalent_order", "equivalent_units"]):
                if j == 2 and i % 4 == 0:
                    continue
                prompts.append(dict(sample_id=f"peak_{i}:{variant}", parent_id=f"peak_{i}", variant=variant))
                features.append(original if j == 0 else original + rng.normal(scale=.12, size=6))
        self.peak_labels = labels
        self.peak_originals = originals
        save_fixture(self.data / "ucnp_peaks", labels, prompts, features)

    def make_detection(self):
        rng = np.random.default_rng(809)
        originals = rng.normal(size=(40, 6)).astype(np.float32)
        originals[30:34] += 20  # Supplements are supervised but excluded from projection statistics.
        originals[34:] += 1000  # Held-out representations must never enter fitting.
        labels, prompts, features = [], [], []
        for i, original in enumerate(originals):
            partition = "train" if i < 34 else "valid"
            subpartition = "cohort" if i < 15 else "extra" if i < 30 else "supp" if i < 34 else "external"
            row = dict(sample_id=f"detection_{i}", partition=partition, subpartition=subpartition,
                       paper_id=f"paper_{i:03d}")
            for j, band in enumerate(BANDS):
                row[f"mlabel_{band}"] = -1 if (i + j) % 7 == 0 else (i + j) % 2
            labels.append(row)
            prompts.append(dict(sample_id=f"detection_{i}:original", parent_id=f"detection_{i}", variant="original"))
            features.append(original)
            if i < 30:
                prompts.append(dict(sample_id=f"detection_{i}:order", parent_id=f"detection_{i}", variant="equivalent_order"))
                features.append(original + rng.normal(scale=.2, size=6))
                for k in range(2 if i < 6 else 1):
                    prompts.append(dict(sample_id=f"detection_{i}:para_{k}", parent_id=f"detection_{i}", variant="paraphrase"))
                    features.append(original + rng.normal(scale=.25, size=6))
        self.detection_originals = originals
        self.detection_labels = np.array([[row[f"mlabel_{band}"] for band in BANDS] for row in labels])
        save_fixture(self.data / "ucnp_detection", labels, prompts, features)

    def invoke(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
            status = runner.main(arguments)
        self.assertEqual(status, 0, error.getvalue())

    def test_loader_preserves_all_paraphrases_and_unknown_labels(self):
        detection = load_detection(self.data / "ucnp_detection")
        self.assertEqual(len(detection.paraphrase), 36)
        self.assertEqual(len(np.unique(detection.paraphrase_parents)), 30)
        self.assertEqual(int(detection.projection_mask.sum()), 30)
        np.testing.assert_array_equal(detection.y_train, self.detection_labels[:34])
        np.testing.assert_array_equal(detection.y_valid, self.detection_labels[34:])
        np.testing.assert_array_equal(detection.x_valid, self.detection_originals[34:])
        peaks = load_peaks(self.data / "ucnp_peaks")
        self.assertTrue(np.isnan(peaks.targets).any())
        self.assertTrue((~peaks.available[:, 2]).any())

    def test_component_crossing_outer_folds_is_rejected(self):
        self.peak_labels[1]["outer_fold"] = 1
        path = self.data / "ucnp_peaks" / "labels.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.peak_labels[0]))
            writer.writeheader()
            writer.writerows(self.peak_labels)
        with self.assertRaisesRegex(ValueError, "component crosses"):
            load_peaks(path.parent)

    def test_detection_paper_crossing_train_validation_is_rejected(self):
        path = self.data / "ucnp_detection" / "labels.csv"
        rows = read_rows(path)
        first_training_paper = next(row["paper_id"] for row in rows if row["partition"] == "train")
        next(row for row in rows if row["partition"] == "valid")["paper_id"] = first_training_paper
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "papers overlap"):
            load_detection(path.parent)

    def test_train_reload_predict_both_task_types(self):
        for name, kind, originals in [
                ("ucnp_peaks", ConsistencyPeaks, self.peak_originals),
                ("ucnp_detection", ConsistencyDetection, self.detection_originals)]:
            with self.subTest(dataset=name):
                model_path = self.work / f"{name}.npz"
                self.invoke(["train", "--data", str(self.data / name), "--model", str(model_path), "--threads", "1"])
                model = load_model(model_path)
                self.assertIsInstance(model, kind)
                features_path = self.work / f"{name}_new.npy"
                np.save(features_path, originals[:4])
                output_path = self.work / f"{name}_predictions.csv"
                self.invoke(["predict", "--model", str(model_path), "--features", str(features_path),
                             "--output", str(output_path), "--threads", "1"])
                rows = read_rows(output_path)
                self.assertEqual(len(rows), 4)
                prefix = "prob" if name == "ucnp_detection" else "prediction"
                actual = np.array([[float(row[f"{prefix}_{band}"]) for band in BANDS] for row in rows])
                expected = model.predict_proba(originals[:4]) if name == "ucnp_detection" else model.predict(originals[:4])
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
                if name == "ucnp_detection":
                    np.testing.assert_allclose(model.model_["mu"], self.detection_originals[:30].mean(0), rtol=1e-6)
                    self.assertEqual(model.diagnostic_["supplemental_samples"], 4)
                    self.assertEqual(model.diagnostic_["n_train_per_band"], (self.detection_labels[:34] >= 0).sum(0).tolist())

    def test_demo_outputs_masked_metrics_and_reusable_models(self):
        output, models = self.work / "results", self.work / "models"
        self.invoke(["demo", "--datasets", "ucnp_peaks", "ucnp_detection", "--data-dir", str(self.data),
                     "--output-dir", str(output), "--model-dir", str(models), "--threads", "1"])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        metrics = read_rows(output / "metrics.csv")
        self.assertEqual({row["dataset"] for row in metrics}, {"ucnp_peaks", "ucnp_detection"})
        self.assertNotIn("method", metrics[0])
        detection_rows = read_rows(output / "ucnp_detection" / "predictions.csv")
        self.assertEqual(len(detection_rows), 6 * 7)
        self.assertTrue(any(row["target"] == "" for row in detection_rows))
        detection_metrics = [row for row in metrics if row["dataset"] == "ucnp_detection" and row["band"] != "macro"]
        for measured in detection_metrics:
            observed = [row for row in detection_rows if row["band"] == measured["band"] and row["target"] != ""]
            truth = np.array([int(row["target"]) for row in observed])
            predicted = np.array([int(row["present"]) for row in observed])
            tp, fp, fn = ((predicted == 1) & (truth == 1)).sum(), ((predicted == 1) & (truth == 0)).sum(), ((predicted == 0) & (truth == 1)).sum()
            denominator = 2 * tp + fp + fn
            self.assertEqual(int(measured["n"]), len(observed))
            self.assertAlmostEqual(float(measured["f1"]), 2 * tp / denominator if denominator else 0)
        peak_rows = read_rows(output / "ucnp_peaks" / "predictions.csv")
        self.assertTrue(any(row["target"] == "" for row in peak_rows))
        self.assertTrue(all(np.isfinite(float(row["prediction"])) for row in peak_rows))
        for name, kind in [("ucnp_peaks", ConsistencyPeaks), ("ucnp_detection", ConsistencyDetection)]:
            self.assertIsInstance(load_model(models / f"{name}.npz"), kind)


if __name__ == "__main__":
    unittest.main()
