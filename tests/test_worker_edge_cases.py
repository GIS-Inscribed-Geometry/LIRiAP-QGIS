import unittest

try:
    from shapely.geometry import MultiPolygon, Polygon
    from LIRiAP_pack import approximation_standard_worker as approx_std
    HAS_GEOM_STACK = True
except Exception:
    HAS_GEOM_STACK = False


@unittest.skipUnless(HAS_GEOM_STACK, "requires numpy/scipy/shapely")
class TestApproximationWorkerEdgeCases(unittest.TestCase):
    def test_edge_candidates_include_anchor_angles(self):
        poly = Polygon([(0, 0), (4, 0), (4, 1), (0, 1), (0, 0)])
        angles = approx_std._edge_candidate_angles(poly)
        self.assertIn(0.0, angles.tolist())
        self.assertIn(45.0, angles.tolist())
        self.assertTrue((angles >= 0.0).all())
        self.assertTrue((angles <= 90.0).all())

    def test_search_handles_multipolygon(self):
        p1 = Polygon([(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)])
        p2 = Polygon([(3, 0), (8, 0), (8, 2), (3, 2), (3, 0)])
        mp = MultiPolygon([p1, p2])
        out = approx_std._search(
            mp,
            angle_step=15,
            grid_steps_coarse=20,
            grid_steps_fine=30,
            max_ratio=0.0,
            buf_enabled=False,
            buf_value=0.0,
        )
        self.assertIsNotNone(out)
        rect, area, _, ratio = out
        self.assertGreater(area, 0.0)
        self.assertGreater(rect.area, 0.0)
        self.assertGreaterEqual(ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
