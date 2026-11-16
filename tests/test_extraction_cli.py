"""Extraction CLI routing and safe consumption of locally generated feature chunks."""

import contextlib
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run as runner
from consistency_readout import ConsistencyRidge
from consistency_readout.data import _feature_paths, load_data, load_features
from consistency_readout.ucnp import load_peaks


class ExtractionCLITests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".verification"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="extraction_cli_", dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        rng = np.random.default_rng(812)
        self.features = rng.normal(size=(12, 4)).astype(np.float32)
        self.ids = np.asarray([f"sample_{index}" for index in range(12)])

    def chunks(self, name="chunks"):
        folder = self.work / name
        folder.mkdir()
        for index, start in enumerate([0, 6]):
            np.savez_compressed(folder / f"features_{index:03d}.npz",
                                sample_id=self.ids[start:start + 6], features=self.features[start:start + 6])
        return folder

    def test_extract_dispatches_explicit_and_automatic_dtype_options(self):
        fake = types.ModuleType("consistency_readout.extraction")
        fake.extract_features = Mock(return_value=self.work)
        with patch.dict(sys.modules, {fake.__name__: fake}):
            status = runner.main(["extract", "--data", str(self.work), "--model", "models/pretrained/local",
                                  "--output-dir", str(self.work / "cache"), "--batch-size", "3",
                                  "--chunk-size", "9", "--device", "cpu", "--dtype", "float32", "--resume"])
            self.assertEqual(status, 0)
            fake.extract_features.assert_called_once_with(
                self.work, Path("models/pretrained/local"), output_dir=self.work / "cache",
                batch_size=3, chunk_size=9, device="cpu", dtype="float32", resume=True)
            fake.extract_features.reset_mock()
            self.assertEqual(runner.main(["extract", "--data", str(self.work), "--model", "local", "--device", "cpu"]), 0)
            self.assertIsNone(fake.extract_features.call_args.kwargs["dtype"])
            self.assertFalse(fake.extract_features.call_args.kwargs["resume"])

    def test_missing_optional_extractor_dependency_is_a_clear_nonzero_error(self):
        fake = types.ModuleType("consistency_readout.extraction")
        fake.extract_features = Mock(side_effect=ImportError("Install with python -m pip install '.[extract]'"))
        error = io.StringIO()
        with patch.dict(sys.modules, {fake.__name__: fake}), contextlib.redirect_stderr(error):
            self.assertEqual(runner.main(["extract", "--data", str(self.work), "--model", "local"]), 1)
        self.assertIn(".[extract]", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())

    def test_cpu_training_does_not_import_optional_extraction_libraries(self):
        training = self.work / "training.npz"
        np.savez_compressed(training, X=np.stack([self.features, self.features + .02], axis=1),
                            y=self.features[:, 0] * 2)
        code = """import importlib.abc,sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','transformers'}:
            raise AssertionError('CPU training imported optional dependency: '+fullname)
sys.meta_path.insert(0,BlockOptional())
sys.path.insert(0,sys.argv[1])
import run
raise SystemExit(run.main(['train','--data',sys.argv[2],'--model',sys.argv[3],'--threads','1']))
"""
        output = self.work / "cpu_readout.npz"
        result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT), str(training), str(output)],
                                cwd=self.work, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(output.is_file())

    def test_chunk_directory_matches_single_file_and_saved_model_prediction(self):
        folder = self.chunks()
        single = self.work / "single.npz"
        np.savez_compressed(single, sample_id=self.ids, features=self.features)
        chunk_ids, chunk_features = load_features(folder)
        single_ids, single_features = load_features(single)
        np.testing.assert_array_equal(chunk_ids, single_ids)
        np.testing.assert_array_equal(chunk_features, single_features)
        model = ConsistencyRidge(alphas=np.array([1.]), lambdas=np.array([1.])).fit(
            np.stack([self.features, self.features + .03], axis=1), self.features[:, 0] * 3)
        checkpoint = self.work / "readout.npz"
        model.save(checkpoint)
        output = self.work / "predictions.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            runner.predict(checkpoint, folder, output, threads=1)
        with output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["sample_id"] for row in rows], self.ids.tolist())
        np.testing.assert_array_equal([float(row["prediction"]) for row in rows], model.predict(self.features))

    def test_chunk_conflicts_missing_indices_and_dtype_dimension_errors(self):
        cases = {
            "duplicate": (self.ids[:6], self.features[6:]),
            "dimension": (self.ids[6:], self.features[6:, :3]),
            "dtype": (self.ids[6:], self.features[6:].astype(np.float64)),
            "nonfinite": (self.ids[6:], np.full((6, 4), np.nan, dtype=np.float32)),
        }
        for name, (ids, features) in cases.items():
            with self.subTest(case=name):
                folder = self.chunks(name)
                np.savez_compressed(folder / "features_001.npz", sample_id=ids, features=features)
                with self.assertRaises(ValueError):
                    load_features(folder)
        folder = self.work / "gap"
        folder.mkdir()
        np.savez_compressed(folder / "features_001.npz", sample_id=self.ids, features=self.features)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            load_features(folder)
        folder = self.chunks("conflict")
        np.savez_compressed(folder / "features.npz", sample_id=self.ids, features=self.features)
        with self.assertRaisesRegex(ValueError, "not both"):
            load_features(folder)

    def test_prompt_directory_requires_complete_id_coverage(self):
        folder = self.chunks()
        with gzip.open(folder / "prompts.jsonl.gz", "wt", encoding="utf-8") as stream:
            for sid in [*self.ids, "missing_prompt"]:
                stream.write(json.dumps({"sample_id": str(sid)}) + "\n")
        with self.assertRaisesRegex(ValueError, "exactly match"):
            load_features(folder)

    def test_incomplete_or_changed_extraction_cache_is_rejected(self):
        folder = self.chunks()
        manifest = {"schema_version": 1, "status": "complete", "configuration": {"prompt_count": 12},
                    "completed_rows": 12, "feature_dimension": 4, "chunks": []}
        for index, path in enumerate(sorted(folder.glob("features_*.npz"))):
            manifest["chunks"].append({"file": path.name, "start": index * 6, "rows": 6,
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        receipt = folder / "features_manifest.json"
        receipt.write_text(json.dumps(manifest), encoding="utf-8")
        np.testing.assert_array_equal(load_features(folder)[1], self.features)
        manifest["status"] = "running"
        receipt.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_features(folder)
        manifest["status"] = "complete"
        manifest["chunks"][0]["sha256"] = "0" * 64
        receipt.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_features(folder)

    def test_missing_cache_guidance_for_scalar_and_ucnp_inputs(self):
        folder = self.work / "raw_only"
        folder.mkdir()
        with (folder / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "target"])
            writer.writeheader()
            writer.writerow({"sample_id": "original", "target": 1.})
        with gzip.open(folder / "prompts.jsonl.gz", "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"sample_id": "original", "parent_id": "original", "variant": "original"}) + "\n")
        for loader in [load_data, load_peaks, _feature_paths]:
            with self.subTest(loader=loader.__name__), self.assertRaises(FileNotFoundError) as caught:
                loader(folder)
            message = str(caught.exception)
            self.assertIn(".[extract]", message)
            self.assertIn("run.py extract", message)
            self.assertIn("models/pretrained/Qwen2.5-7B-Instruct", message)

    def test_changed_prompt_text_is_rejected_even_when_sample_ids_match(self):
        folder = self.chunks()
        prompts = [{"sample_id": str(sid), "parent_id": str(sid), "variant": "original", "text": "original text"}
                   for sid in self.ids]
        prompt_path = folder / "prompts.jsonl.gz"

        def write_prompts():
            with gzip.open(prompt_path, "wt", encoding="utf-8") as stream:
                for row in prompts:
                    stream.write(json.dumps(row) + "\n")

        write_prompts()
        manifest = {"schema_version": 1, "status": "complete", "configuration": {
            "prompt_count": len(prompts), "prompts_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest()},
            "completed_rows": 12, "feature_dimension": 4, "chunks": []}
        for index, path in enumerate(sorted(folder.glob("features_*.npz"))):
            manifest["chunks"].append({"file": path.name, "start": index * 6, "rows": 6,
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (folder / "features_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        np.testing.assert_array_equal(load_features(folder)[0], self.ids)
        prompts[0]["text"] = "changed text with unchanged sample ID"
        write_prompts()
        with self.assertRaisesRegex(ValueError, "prompt checksum"):
            load_features(folder)


if __name__ == "__main__":
    unittest.main()
