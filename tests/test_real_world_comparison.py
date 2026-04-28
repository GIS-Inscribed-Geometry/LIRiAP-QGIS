"""
Real-world data comparison tests for all LIR algorithms.

Compares result size, failure rate, containment, and execution speed across:
- axis_aligned
- contained_standard
- contained_fast
- bcrs_standard
- bcrs_fast
- approximation_standard
- approximation_fast

Data sources:
- tests/real_world_data/test2.geojson
- tests/real_world_data/test3.gpkg

Test Functions
==============
test_axis_aligned_performance: Axis-Aligned LIR metrics
test_contained_standard_performance: Contained Standard metrics
test_contained_fast_performance: Contained Fast metrics
test_bcrs_standard_performance: BCRS Standard metrics
test_bcrs_fast_performance: BCRS Fast metrics
test_approximation_standard_performance: Approximation Standard metrics
test_approximation_fast_performance: Approximation Fast metrics

Running
=======
pytest tests/test_real_world_comparison.py -v

See Also
========
*_worker.py: Modules under test
"""

import sys
import time
from pathlib import Path

import pytest
from shapely.geometry import shape
from shapely.wkb import dumps as wkb_dumps
from shapely import Polygon, from_wkt

sys.path.insert(0, str(Path(__file__).parent.parent / "LIRiAP_pack"))

from axis_aligned_lir_worker import _worker_process_feature as aa_worker
from contained_standard_worker import _worker_process_feature as cont_std_worker
from contained_fast_worker import _worker_process_feature as cont_fast_worker
from bcrs_worker import _worker_process_feature as bcrs_std_worker
from bcrs_fast_worker import _worker_process_feature as bcrs_fast_worker
from approximation_standard_worker import _worker_process_feature as approx_std_worker
from approximation_fast_worker import _worker_process_feature as approx_fast_worker


DATA_DIR = Path(__file__).parent / "real_world_data"


def load_features(limit=None):
    features = []
    geojson_path = DATA_DIR / "test2.geojson"
    if geojson_path.exists():
        import json
        with open(geojson_path) as f:
            data = json.load(f)
        for feat in data["features"][:limit]:
            props = feat.get("properties", {})
            geom = shape(feat["geometry"])
            features.append({
                "feat_id": props.get("feat_id", 0),
                "polygon": geom,
                "expected_area": props.get("area"),
            })

    gpkg_path = DATA_DIR / "test3.gpkg"
    if gpkg_path.exists():
        import fiona
        with fiona.open(gpkg_path) as src:
            for feat in src[:limit]:
                props = feat["properties"]
                geom = shape(feat["geometry"])
                features.append({
                    "feat_id": props.get("feat_id", 0),
                    "polygon": geom,
                    "expected_area": props.get("area"),
                })
    return features


WORKER_CONFIGS = {
    "axis_aligned": (aa_worker, [
        0.0,   # axis_angle
        10,     # grid_fine
        0.0,    # max_ratio
        False,   # buf_enabled
        0.0,    # buf_value
        True,    # always_return
    ]),
    "contained_standard": (cont_std_worker, [
        15,      # angle_step
        20,      # grid_coarse
        20,      # grid_fine
        0.0,     # max_ratio
        False,    # buf_enabled
        0.0,     # buf_value
        5,       # top_k
        True,    # always_return
    ]),
    "contained_fast": (cont_fast_worker, [
        15,      # angle_step
        20,      # grid_coarse
        20,      # grid_fine
        0.0,     # max_ratio
        False,   # buf_enabled
        0.0,     # buf_value
        5,       # top_k
        True,    # always_return
    ]),
    "bcrs_standard": (bcrs_std_worker, [
        15,      # angle_step
        20,      # grid_coarse
        20,      # grid_fine
        0.0,     # max_ratio
        False,    # buf_enabled
        0.0,     # buf_value
        5,       # top_k
        True,    # always_return
    ]),
    "bcrs_fast": (bcrs_fast_worker, [
        15,      # angle_step
        20,      # grid_coarse
        20,      # grid_fine
        0.0,     # max_ratio
        False,    # buf_enabled
        0.0,     # buf_value
        5,       # top_k
        True,    # always_return
    ]),
    "approximation_standard": (approx_std_worker, [
        15,      # angle_step
        20,      # grid_steps_coarse
        20,      # grid_steps_fine (unused for approx)
        0.0,     # max_ratio
        False,    # buf_enabled
        0.0,     # buf_value
    ]),
    "approximation_fast": (approx_fast_worker, [
        15,      # angle_step
        20,      # grid_steps_coarse
        20,      # grid_steps_fine (unused for approx)
        0.0,     # max_ratio
        False,    # buf_enabled
        0.0,     # buf_value
    ]),
}


@pytest.fixture(scope="module")
def features():
    return load_features(limit=20)


@pytest.fixture(scope="module")
def results(features):
    results_data = {}
    for feat in features:
        feat_id = feat["feat_id"]
        polygon = feat["polygon"]
        wkb = wkb_dumps(polygon)
        results_data[feat_id] = {}

        for algo_name, (worker, args) in WORKER_CONFIGS.items():
            start_time = time.perf_counter()
            try:
                result = worker((feat_id, wkb, *args))
                exec_time = (time.perf_counter() - start_time) * 1000
                if result is None:
                    results_data[feat_id][algo_name] = {"success": False, "error": "returned None", "time_ms": exec_time}
                else:
                    f_id, wkt, area, *rest = result
                    rect = from_wkt(wkt)
                    covered = polygon.covers(rect)
                    results_data[feat_id][algo_name] = {
                        "success": True,
                        "area": area,
                        "wkt": wkt,
                        "covered": covered,
                        "rest": rest,
                        "time_ms": exec_time,
                    }
            except Exception as e:
                exec_time = (time.perf_counter() - start_time) * 1000
                results_data[feat_id][algo_name] = {"success": False, "error": str(e), "time_ms": exec_time}
    return results_data


class TestRealWorldComparison:
    def test_algorithms_run(self, results, features):
        assert len(results) > 0, "No features loaded"

    def test_summary(self, results, features):
        print("\n" + "=" * 90)
        print("REAL-WORLD DATA COMPARISON")
        print("=" * 90)

        summary = {}
        for algo_name in WORKER_CONFIGS:
            successes = sum(1 for r in results.values() if r.get(algo_name, {}).get("success"))
            total = len(results)
            fail_rate = (total - successes) / total * 100
            total_area = sum(
                r.get(algo_name, {}).get("area", 0)
                for r in results.values()
                if r.get(algo_name, {}).get("success")
            )
            total_time = sum(
                r.get(algo_name, {}).get("time_ms", 0)
                for r in results.values()
            )
            avg_time = total_time / total if total_time > 0 else 0
            summary[algo_name] = {
                "successes": successes,
                "fail_rate": fail_rate,
                "total_area": total_area,
                "total_time": total_time,
                "avg_time": avg_time,
            }

        print(f"\n{'Algorithm':<25} {'Success':>8} {'Fail%':>8} {'Total Area':>14} {'Avg Time (ms)':>12}")
        print("-" * 90)
        for algo_name, stats in summary.items():
            print(f"{algo_name:<25} {stats['successes']:>8}/{len(results)} {stats['fail_rate']:>7.1f}% {stats['total_area']:>14,.2f} {stats['avg_time']:>11.2f}")
        print("=" * 90)

        assert len(results) > 0

    def test_containment(self, results, features):
        print("\n" + "=" * 90)
        print("CONTAINMENT CHECK")
        print("=" * 90)

        for feat in features[:5]:
            feat_id = feat["feat_id"]
            print(f"\nfeat_id={feat_id}:")
            for algo_name in WORKER_CONFIGS:
                res = results.get(feat_id, {}).get(algo_name, {})
                if res.get("success"):
                    area = res.get("area", 0)
                    covered = res.get("covered", False)
                    status = "OK" if covered else "FAIL"
                    print(f"  {algo_name:<25} area={area:12,.2f} containment={status}")
                else:
                    print(f"  {algo_name:<25} error: {res.get('error', 'unknown')}")