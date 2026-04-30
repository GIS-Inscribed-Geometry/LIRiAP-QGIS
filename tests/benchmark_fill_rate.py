"""
Fill rate benchmark for all LIR algorithms.
"""

import json
import time
from pathlib import Path

import numpy as np
from shapely.geometry import shape

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "LIRiAP_pack"))

from axis_aligned_lir_worker import _worker_process_feature as aa_worker
from bcrs_worker import _worker_process_feature as bcrs_std_worker
from bcrs_fast_worker import _worker_process_feature as bcrs_fast_worker
from skeleton_worker import _worker_process_feature as skeleton_worker


def main():
    DATA_FILE = Path(__file__).parent / "real_world_data" / "realworld.geojson"
    with open(DATA_FILE) as f:
        features = [shape(g) for g in [feat["geometry"] for feat in json.load(f)["features"]]]
    features = [f for f in features if f.is_valid and not f.is_empty]
    print(f"Features: {len(features)}")

    results = {}

    # Axis-Aligned
    print("Running Axis-Aligned...")
    t0 = time.perf_counter()
    r_aa = []
    for poly in features:
        r = aa_worker((0, poly.wkb, 0.0, 120, 0.0, False, 0.0, True))
        if r:
            r_aa.append(float(r[2]) / poly.area * 100)
    aa_time = time.perf_counter() - t0
    results["Axis-Aligned"] = {"mean": np.mean(r_aa), "median": np.median(r_aa), "min": np.min(r_aa), "max": np.max(r_aa), "std": np.std(r_aa), "time": aa_time, "n": len(r_aa)}
    print(f"  Done: {aa_time:.1f}s")

    # Skeleton
    print("Running Skeleton...")
    t0 = time.perf_counter()
    r_sk = []
    for poly in features:
        r = skeleton_worker((0, poly.wkb, 5.0, 40, 40, 0.0, False, 0.0, 5, True))
        if r:
            r_sk.append(float(r[2]) / poly.area * 100)
    sk_time = time.perf_counter() - t0
    results["Skeleton"] = {"mean": np.mean(r_sk), "median": np.median(r_sk), "min": np.min(r_sk), "max": np.max(r_sk), "std": np.std(r_sk), "time": sk_time, "n": len(r_sk)}
    print(f"  Done: {sk_time:.1f}s")

    # BCRS Fast
    print("Running BCRS Fast...")
    t0 = time.perf_counter()
    r_bf = []
    for poly in features:
        r = bcrs_fast_worker((0, poly.wkb, 5.0, 40, 120, 0.0, False, 0.0, 3, True))
        if r:
            r_bf.append(float(r[2]) / poly.area * 100)
    bf_time = time.perf_counter() - t0
    results["BCRS Fast"] = {"mean": np.mean(r_bf), "median": np.median(r_bf), "min": np.min(r_bf), "max": np.max(r_bf), "std": np.std(r_bf), "time": bf_time, "n": len(r_bf)}
    print(f"  Done: {bf_time:.1f}s")

    # BCRS Standard
    print("Running BCRS Standard...")
    t0 = time.perf_counter()
    r_bs = []
    for poly in features:
        r = bcrs_std_worker((0, poly.wkb, 5.0, 40, 120, 0.0, False, 0.0, 3, True))
        if r:
            r_bs.append(float(r[2]) / poly.area * 100)
    bs_time = time.perf_counter() - t0
    results["BCRS"] = {"mean": np.mean(r_bs), "median": np.median(r_bs), "min": np.min(r_bs), "max": np.max(r_bs), "std": np.std(r_bs), "time": bs_time, "n": len(r_bs)}
    print(f"  Done: {bs_time:.1f}s")

    print()
    print("=" * 80)
    print("Algorithm               Mean%    Med%     Min%     Max%     Std%     Time(s)")
    print("=" * 80)
    for name, d in results.items():
        print(f"{name:<22} {d['mean']:>7.2f} {d['median']:>7.2f} {d['min']:>7.2f} {d['max']:>7.2f} {d['std']:>7.2f} {d['time']:>10.2f}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()