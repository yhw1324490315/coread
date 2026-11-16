"""Check the train-only spectral detection readout and reusable checkpoints."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from consistency_readout.detection import ConsistencyDetection

ROOT = Path(__file__).resolve().parents[1]


class DetectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(142)
        self.x = rng.normal(size=(36, 8))
        self.labels = np.asarray([[(i + j) % 2 for j in range(7)] for i in range(36)])
        self.labels[::5, ::2] = -1
        self.labels[-1] = -1
        self.pool = np.arange(36) < 28
        self.reorder_parents = np.flatnonzero(self.pool)
        self.reorder = self.x[self.reorder_parents] + rng.normal(scale=.1, size=(28, 8))
        self.para_parents = np.concatenate([np.arange(28), np.arange(10)])
        self.para = self.x[self.para_parents] + rng.normal(scale=.2, size=(38, 8))
        self.new = rng.normal(size=(5, 8))
        threads = threadpool_limits(limits=1)
        threads.__enter__()
        self.addCleanup(threads.__exit__, None, None, None)
        temporary_root = ROOT / ".verification"
        temporary_root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="detection_test_", dir=temporary_root)
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "model.npz"

    def fit(self, x=None, labels=None):
        return ConsistencyDetection().fit(
            self.x if x is None else x, self.labels if labels is None else labels,
            self.pool, self.reorder, self.reorder_parents, self.para, self.para_parents)

    def test_probabilities_match_explicit_spectral_reference(self):
        fitted = self.fit()
        mu, sd = self.x[self.pool].mean(0), self.x[self.pool].std(0) + 1e-6
        dr = (self.x[self.reorder_parents] - self.reorder) / sd
        dp = (self.x[self.para_parents] - self.para) / sd
        sr, sp = dr.T @ dr / len(dr), dp.T @ dp / len(dp)
        penalty = .5 * sr / (np.trace(sr) / len(sr)) + .5 * sp / (np.trace(sp) / len(sp))
        eigenvalues, basis = np.linalg.eigh(penalty)
        scale = 1 / np.sqrt(1 + 100 * np.clip(eigenvalues, 0, None))
        projected = ((self.x - mu) / sd) @ basis * scale
        expected = []
        for band in range(7):
            mask = self.labels[:, band] >= 0
            lr = LogisticRegression(C=.1, max_iter=4000, tol=1e-6).fit(projected[mask], self.labels[mask, band])
            weight = (basis * scale) @ lr.coef_[0] / sd
            intercept = lr.intercept_[0] - weight @ mu
            expected.append(expit(self.new @ weight + intercept))
        np.testing.assert_allclose(fitted.predict_proba(self.new), np.column_stack(expected), atol=1e-13, rtol=1e-13)
        self.assertEqual(fitted.diagnostic_["paraphrase_pairs"], 38)
        self.assertEqual(fitted.diagnostic_["n_train_per_band"], (self.labels >= 0).sum(0).tolist())

    def test_supplemental_rows_do_not_change_projection_statistics(self):
        original = self.fit()
        changed = self.x.copy()
        changed[~self.pool] += 3
        updated = self.fit(changed)
        for key in ["mu", "sd", "projection", "scale"]:
            np.testing.assert_array_equal(original.model_[key], updated.model_[key])
        self.assertGreater(np.max(np.abs(original.predict_proba(self.new) - updated.predict_proba(self.new))), 1e-6)

    def test_fully_masked_supplemental_row_has_no_supervised_effect(self):
        original = self.fit()
        changed = self.x.copy()
        changed[-1] += 100
        updated = self.fit(changed)
        np.testing.assert_array_equal(original.predict_proba(self.new), updated.predict_proba(self.new))

    def test_save_load_round_trip_and_no_overwrite(self):
        fitted = self.fit()
        fitted.save(self.path)
        with np.load(self.path, allow_pickle=False) as bundle:
            self.assertEqual(str(bundle["model_type"]), "consistency_detection")
        restored = ConsistencyDetection.load(self.path)
        np.testing.assert_array_equal(fitted.predict_proba(self.new), restored.predict_proba(self.new))
        self.assertEqual(fitted.diagnostic_, restored.diagnostic_)
        self.assertTrue(((restored.predict_proba(self.new) >= 0) & (restored.predict_proba(self.new) <= 1)).all())
        np.testing.assert_array_equal(restored.predict(self.new), restored.predict_proba(self.new) >= .5)
        with self.assertRaises(FileExistsError):
            fitted.save(self.path)

    def test_bad_inputs_are_rejected(self):
        invalid_labels = self.labels.copy()
        invalid_labels[0, 0] = 2
        one_class = self.labels.copy()
        one_class[:, 0] = 0
        for labels in [invalid_labels, one_class, self.labels[:, :-1]]:
            with self.subTest(shape=labels.shape), self.assertRaises(ValueError):
                self.fit(labels=labels)
        bad_parents = self.para_parents.copy()
        bad_parents[0] = 35
        with self.assertRaisesRegex(ValueError, "outside the projection"):
            ConsistencyDetection().fit(self.x, self.labels, self.pool, self.reorder,
                                       self.reorder_parents, self.para, bad_parents)
        with self.assertRaises(ValueError):
            ConsistencyDetection().predict_proba(self.new)
        fitted = self.fit()
        with self.assertRaises(ValueError):
            fitted.predict_proba(self.new[:, :-1])

    def test_invalid_checkpoint_is_rejected_without_pickle(self):
        fitted = self.fit()
        fitted.save(self.path)
        with np.load(self.path, allow_pickle=False) as bundle:
            arrays = {key: bundle[key] for key in bundle.files}
        arrays["mu"] = np.asarray([{"unsafe": True}], dtype=object)
        bad = self.path.with_name("bad.npz")
        np.savez_compressed(bad, **arrays)
        with self.assertRaisesRegex(ValueError, "Invalid detection checkpoint"):
            ConsistencyDetection.load(bad)


if __name__ == "__main__":
    unittest.main()
