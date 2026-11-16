"""Check reusable training, inference and checkpoint round trips."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from consistency_readout import ConsistencyRidge

ROOT = Path(__file__).resolve().parents[1]


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        original = rng.normal(size=(18, 12)).astype(np.float32)
        self.x = np.stack([original, original + rng.normal(scale=.1, size=original.shape),
                           original + rng.normal(scale=.2, size=original.shape)], axis=1).astype(np.float32)
        self.y = original @ rng.normal(size=12)
        self.available = np.ones((18, 3), dtype=bool)
        self.fold = np.arange(18) % 3
        self.pool = threadpool_limits(limits=1)
        self.pool.__enter__()
        self.addCleanup(self.pool.__exit__, None, None, None)
        temporary_root = ROOT / ".verification"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="readout_test_", dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "models" / "readout.npz"

    def fit_model(self, x=None, available=None):
        return ConsistencyRidge(alphas=[.01, .1, 1.0], lambdas=[0, 1, 10]).fit(
            self.x if x is None else x, self.y,
            self.available if available is None else available, self.fold)

    def test_save_load_preserves_predictions_and_metadata(self):
        fitted = self.fit_model()
        before = fitted.predict(self.x)
        fitted.save(self.path)
        with np.load(self.path, allow_pickle=False) as stored:
            self.assertEqual(int(stored["schema_version"]), 1)
            self.assertEqual(stored["beta"].shape, (12,))
            self.assertIsInstance(json.loads(str(stored["diagnostic_json"])), dict)
        loaded = ConsistencyRidge.load(self.path)
        np.testing.assert_array_equal(before, loaded.predict(self.x))
        np.testing.assert_array_equal(fitted.predict(self.x[:, 0]), loaded.predict(self.x[:, 0]))
        self.assertEqual(fitted.diagnostic_, loaded.diagnostic_)
        with self.assertRaises(FileExistsError):
            fitted.save(self.path)
        np.testing.assert_array_equal(before, ConsistencyRidge.load(self.path).predict(self.x))

    def test_missing_rewrites_are_excluded_and_can_be_inferred(self):
        available = self.available.copy()
        available[:7, 1] = False
        missing = self.x.copy()
        missing[~available] = np.nan
        zeroed = np.where(available[:, :, None], self.x, 0)
        explicit = self.fit_model(missing, available)
        reference = self.fit_model(zeroed, available)
        inferred = ConsistencyRidge(alphas=[.01, .1, 1], lambdas=[0, 1, 10]).fit(missing, self.y, inner_splits=self.fold)
        np.testing.assert_array_equal(explicit.predict(self.x), reference.predict(self.x))
        np.testing.assert_array_equal(explicit.predict(self.x), inferred.predict(self.x))

    def test_invalid_training_inputs_fail(self):
        missing_original = self.available.copy()
        missing_original[0, 0] = False
        bad_x = self.x.copy()
        bad_x[0, 1, 0] = np.inf
        cases = [
            (self.x[:, 0], self.y, self.available, self.fold),
            (bad_x, self.y, None, self.fold),
            (self.x, np.full(18, np.nan), self.available, self.fold),
            (self.x, self.y, missing_original, self.fold),
            (self.x, self.y, self.available.astype(int), self.fold),
            (self.x, self.y, self.available, np.zeros(18, dtype=int)),
            (self.x, self.y, self.available, np.arange(18) % 3 + 1),
            (self.x, self.y, self.available, np.ones(17, dtype=int)),
        ]
        for x, y, mask, folds in cases:
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                ConsistencyRidge().fit(x, y, mask, folds)

    def test_invalid_parameter_grids_fail(self):
        for args in [dict(alphas=[]), dict(alphas=[0]), dict(lambdas=[-1]),
                     dict(alphas=[[1]]), dict(lambdas=[np.inf])]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                ConsistencyRidge(**args)

    def test_prediction_requires_fitted_compatible_finite_input(self):
        with self.assertRaisesRegex(ValueError, "not fitted"):
            ConsistencyRidge().predict(self.x[:, 0])
        fitted = self.fit_model()
        for values in [np.ones((2, 11)), np.ones(12), np.full((2, 12), np.nan)]:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                fitted.predict(values)

    def test_checkpoint_validation_blocks_corrupted_parameters(self):
        fitted = self.fit_model()
        fitted.save(self.path)
        with np.load(self.path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        corruptions = [dict(schema_version=np.asarray(99)), dict(scale=np.asarray(0.0)),
                       dict(feature_center=np.zeros(11)), dict(beta=np.full(12, np.nan)),
                       dict(beta=np.asarray([None] * 12, dtype=object)),
                       dict(diagnostic_json=np.asarray("{}"))]
        for index, changed in enumerate(corruptions):
            path = self.path.parent / f"invalid_{index}.npz"
            np.savez_compressed(path, **{**arrays, **changed})
            with self.subTest(changed=list(changed)), self.assertRaisesRegex(ValueError, "Invalid readout checkpoint"):
                ConsistencyRidge.load(path)
        empty_path = self.path.parent / "empty.npz"
        empty_path.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Invalid readout checkpoint"):
            ConsistencyRidge.load(empty_path)


if __name__ == "__main__":
    unittest.main()
