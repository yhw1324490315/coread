"""UCNP peak readout checks preserve missing-label and scientific-group semantics."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from consistency_readout.peaks import (
    ConsistencyPeaks, component_inner, predict_path, prepare_peak_stats,
    scientific_weights, solution_path,
)


class PeakTests(unittest.TestCase):
    def setUp(self):
        self.limits = threadpool_limits(limits=2)
        self.addCleanup(self.limits.restore_original_limits)

    def fixture(self):
        rng = np.random.default_rng(72)
        x = rng.normal(size=(18, 3, 8))
        available = np.ones((18, 3), dtype=bool)
        available[0, 2] = False
        x[0, 2] = np.nan
        y = x[:, 0] @ rng.normal(size=(8, 7)) + rng.normal(scale=0.02, size=(18, 7))
        y[-2:, :] = np.nan
        components = np.asarray([f"component_{i // 2}" for i in range(18)])
        groups = [(f"scientific_{i // 2}",) for i in range(18)]
        return x, y, available, components, groups

    def test_normal_equation_keeps_unlabelled_parent_pairs(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(7, 3, 64))
        y = rng.normal(size=7)
        y[-3:] = np.nan
        available = np.ones((7, 3), dtype=bool)
        available[1, 2] = False
        primal = prepare_peak_stats(x, y, available, reduce=False)
        reduced = prepare_peak_stats(x, y, available, reduce=True)
        self.assertEqual(primal["unlabeled_pairs"], 6)
        self.assertEqual(primal["pairs"], 13)
        np.testing.assert_array_equal(primal["mu"], x[:, 0].mean(axis=0))
        # An independent full-width normal equation includes every unlabelled pair.
        labeled = np.isfinite(y)
        z = (x - x[:, 0].mean(axis=0)) / primal["scale"]
        a = z[labeled, 0] - z[labeled, 0].mean(axis=0)
        differences = np.concatenate([z[available[:, j], 0] - z[available[:, j], j] for j in (1, 2)])
        alphas = np.asarray([0.1, 3.0])
        testing = rng.normal(size=(4, 3, 64))
        for lam in [0.0, 5.0]:
            expected = np.column_stack([
                np.linalg.solve(a.T @ a + lam * differences.T @ differences + alpha * np.eye(64),
                                a.T @ (y[labeled] - y[labeled].mean()))
                for alpha in alphas
            ])
            np.testing.assert_allclose(solution_path(primal, lam, alphas), expected, atol=2e-12)
            np.testing.assert_allclose(predict_path(primal, expected, testing),
                                       predict_path(reduced, solution_path(reduced, lam, alphas), testing),
                                       atol=2e-12)
        changed = x.copy()
        changed[-1, 1] += 20
        altered = prepare_peak_stats(changed, y, available, reduce=False)
        self.assertGreater(np.linalg.norm(primal["reg"] - altered["reg"]), 0.1)

    def test_scientific_key_average_includes_multigroup_memberships(self):
        groups = [("a",), ("a", "b"), ("b",), ("ignored",)]
        weights = scientific_weights(groups, np.asarray([True, True, True, False]))
        np.testing.assert_allclose(weights, [0.25, 0.5, 0.25, 0])
        errors = np.asarray([2, 8, 5, 100])
        expected = ((2 + 8) / 2 + (8 + 5) / 2) / 2
        self.assertAlmostEqual(float(weights @ errors), expected)

    def test_components_never_cross_inner_folds(self):
        components = np.repeat(["a", "b", "c", "d", "e"], [8, 5, 4, 2, 1])
        folds = component_inner(components, outer_fold=2)
        self.assertEqual(set(folds), {0, 1, 2})
        for component in set(components):
            self.assertEqual(len(set(folds[components == component])), 1)
        np.testing.assert_array_equal(folds, component_inner(components, outer_fold=2))

    def test_seven_bands_save_load_predict_and_no_overwrite(self):
        x, y, available, components, groups = self.fixture()
        events = []
        fitted = ConsistencyPeaks(alphas=[0.1, 1.0], lambdas=[0, 3]).fit(
            x, y, available, components, groups, progress=lambda *event: events.append(event))
        new = np.nan_to_num(x[:3])
        predicted = fitted.predict(new)
        self.assertEqual(predicted.shape, (3, 3, 7))
        self.assertTrue(np.isfinite(predicted).all())
        self.assertEqual(len(fitted.diagnostic_["band_parameters"]), 7)
        self.assertTrue(all(item["consistency_pairs_with_missing_target"] == 4
                            for item in fitted.diagnostic_["band_parameters"]))
        self.assertEqual(sum(event[1] is None for event in events), 7)
        temporary = ROOT / ".verification"
        temporary.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary, prefix="peaks_") as folder:
            path = Path(folder) / "model.npz"
            fitted.save(path)
            restored = ConsistencyPeaks.load(path)
            np.testing.assert_array_equal(restored.predict(new), predicted)
            with self.assertRaises(FileExistsError):
                fitted.save(path)

    def test_missing_supervision_and_cross_component_groups_fail(self):
        x, y, available, components, groups = self.fixture()
        y[:, 4] = np.nan
        with self.assertRaisesRegex(ValueError, "no measured"):
            ConsistencyPeaks(alphas=[1], lambdas=[0]).fit(x, y, available, components, groups)
        y[:, 4] = 500
        groups[0], groups[2] = ("same",), ("same",)
        with self.assertRaisesRegex(ValueError, "crosses component"):
            ConsistencyPeaks(alphas=[1], lambdas=[0]).fit(x, y, available, components, groups)


if __name__ == "__main__":
    unittest.main()
