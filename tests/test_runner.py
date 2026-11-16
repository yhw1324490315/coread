"""End-to-end checks for training, reloading and applying one consistency readout."""

import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run as runner
from consistency_readout import ConsistencyRidge
from consistency_readout.data import load_data


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".verification"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="readout_cli_", dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.data_dir = self.work / "data"
        self.folder = self.data_dir / "carbon_dot"
        self.folder.mkdir(parents=True)
        rng = np.random.default_rng(41)
        n, dimension = 20, 8
        base = rng.normal(size=(n, dimension)).astype(np.float32)
        self.x = np.stack([base, base + rng.normal(scale=0.1, size=base.shape),
                           base + rng.normal(scale=0.1, size=base.shape)], axis=1).astype(np.float32)
        self.y = base @ rng.normal(size=dimension) + rng.normal(scale=0.1, size=n)
        self.available = np.ones((n, 3), dtype=bool)
        self.available[0, 2] = False
        self.x[0, 2] = np.nan
        self.ids = np.asarray([f"unit_{i}" for i in range(n)])
        self.variants = ("original", "equivalent_order", "equivalent_units")
        labels = [dict(sample_id=sid, outer_fold=i % 5, target=self.y[i]) for i, sid in enumerate(self.ids)]
        with (self.folder / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(labels[0]))
            writer.writeheader()
            writer.writerows(labels)
        self.prompts, features = [], []
        for i, sid in enumerate(self.ids):
            for j, variant in enumerate(self.variants):
                if self.available[i, j]:
                    self.prompts.append(dict(sample_id=f"{sid}:{variant}", parent_id=str(sid), variant=variant))
                    features.append(self.x[i, j])
        self.feature_ids = np.asarray([row["sample_id"] for row in self.prompts])
        self.features = np.asarray(features)
        with gzip.open(self.folder / "prompts.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row in self.prompts:
                stream.write(json.dumps(row) + "\n")
        np.savez_compressed(self.folder / "features.npz", sample_id=self.feature_ids, features=self.features)
        self.custom = self.work / "training.npz"
        np.savez_compressed(self.custom, X=self.x, y=self.y, available=self.available, sample_id=self.ids)

    def read_csv(self, path):
        with Path(path).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_demo_saves_one_readout_and_metrics_recompute(self):
        output, models = self.work / "results", self.work / "models"
        runner.demo(["carbon_dot"], self.data_dir, output, models, threads=1)
        predictions = self.read_csv(output / "carbon_dot/predictions.csv")
        metrics = self.read_csv(output / "metrics.csv")
        self.assertEqual(len(predictions), 59)
        self.assertEqual(len(metrics), 3)
        self.assertEqual(set(predictions[0]), {"sample_id", "outer_fold", "variant", "target", "prediction"})
        self.assertEqual(set(metrics[0]), {"dataset", "variant", "n", "mae", "rmse", "r2", "rewrite_gap_mean"})
        originals = {r["sample_id"]: float(r["prediction"]) for r in predictions if r["variant"] == "original"}
        for measured in metrics:
            rows = [r for r in predictions if r["variant"] == measured["variant"]]
            truth = np.array([float(r["target"]) for r in rows])
            predicted = np.array([float(r["prediction"]) for r in rows])
            self.assertEqual(int(measured["n"]), len(rows))
            self.assertAlmostEqual(float(measured["mae"]), abs(truth - predicted).mean())
            self.assertAlmostEqual(float(measured["rmse"]), np.sqrt(np.mean((truth - predicted) ** 2)))
            self.assertAlmostEqual(float(measured["r2"]), 1 - np.sum((truth - predicted) ** 2) / np.sum((truth - truth.mean()) ** 2))
            if measured["variant"] != "original":
                gap = np.mean([abs(float(r["prediction"]) - originals[r["sample_id"]]) for r in rows])
                self.assertAlmostEqual(float(measured["rewrite_gap_mean"]), gap)
        loaded = ConsistencyRidge.load(models / "carbon_dot.npz")
        self.assertTrue(np.isfinite(loaded.predict(self.x[:, 0])).all())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        details = manifest["datasets"]["carbon_dot"]
        self.assertEqual(len(details["outer_fit_parameters"]), 5)
        self.assertEqual((output / details["model_file"]).resolve(), (models / "carbon_dot.npz").resolve())
        self.assertEqual(len(details["inputs"]), 3)
        self.assertFalse((output / "selections.csv").exists())
        before = (output / "metrics.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            runner.demo(["carbon_dot"], self.data_dir, output, models)
        self.assertEqual(before, (output / "metrics.csv").read_bytes())

    def test_train_custom_npz_and_predict_reuses_saved_weights(self):
        model = self.work / "readout.npz"
        runner.train(self.custom, model, threads=1)
        features = self.work / "new_features.npz"
        np.savez_compressed(features, features=self.x[:4, 0], sample_id=self.ids[:4])
        output = self.work / "predictions.csv"
        runner.predict(model, features, output, threads=1)
        rows = self.read_csv(output)
        self.assertEqual(set(rows[0]), {"sample_id", "prediction"})
        self.assertEqual([r["sample_id"] for r in rows], self.ids[:4].tolist())
        expected = ConsistencyRidge.load(model).predict(self.x[:4, 0])
        np.testing.assert_array_equal([float(r["prediction"]) for r in rows], expected)
        with self.assertRaises(FileExistsError):
            runner.train(self.custom, model)
        with self.assertRaises(FileExistsError):
            runner.predict(model, features, output)

    def test_cli_training_and_prediction_from_another_directory(self):
        model, predictions = self.work / "cli_model.npz", self.work / "cli_predictions.csv"
        features = self.work / "new_features.npy"
        np.save(features, self.x[:3, 0])
        commands = [
            ["train", "--data", str(self.custom), "--model", str(model), "--threads", "1"],
            ["predict", "--model", str(model), "--features", str(features), "--output", str(predictions)],
        ]
        for args in commands:
            completed = subprocess.run([sys.executable, "-B", str(ROOT / "run.py"), *args],
                                       cwd=self.work, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(len(self.read_csv(predictions)), 3)

    def test_missing_rewrite_is_masked_in_both_input_formats(self):
        for path in [self.folder, self.custom]:
            data = load_data(path)
            self.assertFalse(data.available[0, 2])
            self.assertTrue(np.isfinite(data.x).all())
            self.assertEqual(int(data.available.sum()), 59)

    def test_demo_generates_folds_when_labels_omit_them(self):
        labels = self.read_csv(self.folder / "labels.csv")
        with (self.folder / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "target"])
            writer.writeheader()
            writer.writerows({"sample_id": r["sample_id"], "target": r["target"]} for r in labels)
        output = self.work / "generated_fold_results"
        runner.demo(["carbon_dot"], self.data_dir, output, self.work / "generated_fold_models", threads=1)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertIn("generated", manifest["datasets"]["carbon_dot"]["outer_validation"])
        self.assertEqual({int(r["outer_fold"]) for r in self.read_csv(output / "carbon_dot/predictions.csv")}, set(range(5)))

    def test_failure_is_recorded_and_cli_errors_are_nonzero(self):
        output = self.work / "failed_run"
        with self.assertRaises(FileNotFoundError):
            runner.demo(["biochar"], self.data_dir, output, self.work / "failed_models", threads=1)
        self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "failed")
        self.assertEqual(runner.main(["train", "--data", str(self.work / "missing.npz"),
                                      "--model", str(self.work / "never_written.npz")]), 1)


if __name__ == "__main__":
    unittest.main()
