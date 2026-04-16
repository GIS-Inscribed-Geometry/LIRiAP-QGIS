import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

try:
    from LIRiAP_pack import approximation_fast_worker as approx_fast
    from LIRiAP_pack import approximation_standard_worker as approx_std
    from LIRiAP_pack import contained_standard_worker as contained_std
    HAS_RUNTIME_IMPORTS = True
except Exception:
    HAS_RUNTIME_IMPORTS = False


class TestTuningConstantsStatic(unittest.TestCase):
    def test_approximation_standard_constants_declared(self):
        src = (ROOT / "LIRiAP_pack" / "approximation_standard_worker.py").read_text(encoding="utf-8")
        self.assertIn("_EDGE_KERNEL =", src)
        self.assertIn("_UPPER_BOUND_FACTOR =", src)
        self.assertIn("_BRENT_XATOL =", src)

    def test_approximation_fast_constants_declared(self):
        src = (ROOT / "LIRiAP_pack" / "approximation_fast_worker.py").read_text(encoding="utf-8")
        self.assertIn("_EDGE_KERNEL =", src)
        self.assertIn("_UPPER_BOUND_FACTOR =", src)
        self.assertIn("_BRENT_XATOL =", src)

    def test_contained_standard_constants_declared(self):
        src = (ROOT / "LIRiAP_pack" / "contained_standard_worker.py").read_text(encoding="utf-8")
        self.assertIn("_PRUNE_MARGIN =", src)
        self.assertIn("_MIN_STAGE1_CANDIDATES =", src)
        self.assertIn("_ANGLE_DEDUP_DEG =", src)
        self.assertIn("_FALLBACK_INSET_FRACS =", src)
        self.assertIn("_BEST_EFFORT_SCALES =", src)


@unittest.skipUnless(HAS_RUNTIME_IMPORTS, "runtime imports unavailable for worker modules")
class TestTuningConstantsRuntime(unittest.TestCase):
    def test_approximation_constants_are_exposed(self):
        self.assertGreater(approx_std._BRENT_XATOL, 0.0)
        self.assertGreater(approx_std._HALF_WINDOW_MAX, approx_std._HALF_WINDOW_MIN)
        self.assertEqual(len(approx_std._EDGE_KERNEL), 5)
        self.assertAlmostEqual(float(approx_std._UPPER_BOUND_FACTOR), 0.5, places=7)

    def test_fast_and_standard_approximation_match_core_tunings(self):
        self.assertEqual(approx_std._BRENT_XATOL, approx_fast._BRENT_XATOL)
        self.assertEqual(approx_std._HALF_WINDOW_FALLBACK, approx_fast._HALF_WINDOW_FALLBACK)
        self.assertEqual(approx_std._UPPER_BOUND_FACTOR, approx_fast._UPPER_BOUND_FACTOR)

    def test_contained_standard_constants_are_documented(self):
        self.assertGreater(contained_std._PHASE_A_XATOL, 0.0)
        self.assertGreater(contained_std._PRUNE_MARGIN, 0.0)
        self.assertLess(contained_std._PRUNE_MARGIN, 1.0)
        self.assertGreaterEqual(contained_std._MIN_STAGE1_CANDIDATES, 1)
        self.assertGreaterEqual(len(contained_std._FALLBACK_INSET_FRACS), 1)
        self.assertGreaterEqual(len(contained_std._BEST_EFFORT_SCALES), 1)


if __name__ == "__main__":
    unittest.main()
