"""The exact consistency objective agrees in reduced and full coordinates."""

import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from consistency_readout.ridge import prepare_stats, solve_candidates, predict_from_basis, ALPHAS


class QRTests(unittest.TestCase):
    def test_qr_reduction_matches_full_objective(self):
        rng = np.random.default_rng(2026)
        x = rng.normal(size=(40, 3, 256))
        y = rng.normal(size=40)
        available = np.ones((40, 3), dtype=bool)
        available[:5, 2] = False
        test_x = rng.normal(size=(10, 3, 256))
        with threadpool_limits(limits=2):
            reduced = prepare_stats(x, y, available, reduce=True)
            full = prepare_stats(x, y, available, reduce=False)
            self.assertIsNotNone(reduced["basis"])
            self.assertEqual(reduced["gram"].shape, (115, 115))
            self.assertEqual(full["gram"].shape, (256, 256))
            for lam in [0.0, 1.0, 100.0]:
                red, _, _ = solve_candidates(reduced, lam, ALPHAS)
                primal, _, _ = solve_candidates(full, lam, ALPHAS)
                np.testing.assert_allclose(predict_from_basis(reduced, red, test_x),
                                           predict_from_basis(full, primal, test_x), atol=1e-6, rtol=1e-6)
                # Independently solve one penalized normal equation.
                alpha = ALPHAS[12]
                direct = np.linalg.solve(full["gram"] + lam * full["reg"] + alpha * np.eye(256), full["rhs"])
                np.testing.assert_allclose(primal[:, 12], direct, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
