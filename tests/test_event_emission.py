"""
Tests for the TraceEmitter event system integrated into workers.

Verifies that the event emission system correctly:
- Accumulates structured events
- Produces valid JSON-serializable traces
- Handles verbosity levels
- Truncates labels and narrations appropriately

Test Classes
============
TestTraceEmitterBasics: Verify TraceEmitter class itself
TestEmitterWithAxisAlignedWorker: Verify emission from workers
TestEmitterWithContainedWorker: Verify Contained family events
TestEmitterWithBCRSWorker: Verify BCRS family events
TestEmitterWithApproximationWorker: Verify Approximation family events

Running
=======
pytest tests/test_event_emission.py -v

See Also
========
event_emitter.py: Module under test
LIRiAP_visualize.py: Visualizer that consumes traces
"""
import json
import math
import sys
from pathlib import Path

import pytest
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).parent.parent / "LIRiAP_pack"))
from event_emitter import TraceEmitter, VERBOSITY_NORMAL, VERBOSITY_FULL


class TestTraceEmitterBasics:
    """Verify the TraceEmitter class itself."""

    def test_emit_appends_event(self):
        emitter = TraceEmitter("test", "1", "test_poly", {})
        emitter.emit("SETUP", "polygon_loaded", "Test", "Test narration",
                     key="value")
        assert len(emitter.events) == 1
        ev = emitter.events[0]
        assert ev["seq"] == 0
        assert ev["phase"] == "SETUP"
        assert ev["type"] == "polygon_loaded"
        assert ev["label"] == "Test"
        assert ev["narration"] == "Test narration"
        assert ev["ext"] == {"key": "value"}

    def test_seq_increments(self):
        emitter = TraceEmitter("test", "1", "test_poly", {})
        emitter.emit("A", "a", "", "")
        emitter.emit("B", "b", "", "")
        assert emitter.events[0]["seq"] == 0
        assert emitter.events[1]["seq"] == 1

    def test_to_trace_structure(self):
        emitter = TraceEmitter("test_algo", "42", "MyPoly", {"step": 5})
        emitter.emit("SETUP", "polygon_loaded", "Test", "Narration",
                     exterior=[[0, 0], [1, 0], [1, 1], [0, 1]],
                     holes=[], bbox=[0, 0, 1, 1])
        trace = emitter.to_trace()
        assert trace["schema_version"] == "1.0"
        assert trace["algorithm"] == "test_algo"
        assert trace["polygon_id"] == "42"
        assert trace["polygon_name"] == "MyPoly"
        assert trace["params"] == {"step": 5}
        assert "trace_id" in trace
        assert "elapsed_ms" in trace
        assert len(trace["events"]) == 1

    def test_label_truncation(self):
        emitter = TraceEmitter("test", "1", "p", {})
        emitter.emit("P", "t", "x" * 50, "y" * 250)
        assert len(emitter.events[0]["label"]) == 40
        assert len(emitter.events[0]["narration"]) == 200

    def test_verbosity_levels(self):
        full = TraceEmitter("test", "1", "p", {}, verbosity=VERBOSITY_FULL)
        normal = TraceEmitter("test", "1", "p", {}, verbosity=VERBOSITY_NORMAL)
        assert full.verbosity == "FULL"
        assert normal.verbosity == "NORMAL"

    def test_json_serializable(self):
        emitter = TraceEmitter("test", "1", "p", {"key": "val"})
        emitter.emit("R", "final_result", "Done", "The end",
                     rect=[0.0, 0.0, 1.0, 1.0],
                     area=1.0,
                     angle_deg=0.0)
        trace = emitter.to_trace()
        dumped = json.dumps(trace)
        loaded = json.loads(dumped)
        assert loaded["algorithm"] == "test"


class TestEmitterWithAxisAlignedWorker:
    """Verify event emission from the axis-aligned LIR worker."""

    def _make_square_polygon(self):
        return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    def _make_rotated_polygon(self):
        return Polygon([(5, 0), (10, 5), (5, 10), (0, 5)])

    def test_emitter_passed_to_worker(self):
        from axis_aligned_lir_worker import _worker_process_feature
        poly = self._make_square_polygon()
        from shapely.wkb import dumps as wkb_dumps
        wkb_bytes = wkb_dumps(poly)

        emitter = TraceEmitter("axis_aligned_lir", "1", "test_square",
                               {"axis_angle": 0.0, "grid_fine": 50,
                                "max_ratio": 0.0, "always_return": True})
        args = (1, wkb_bytes, 0.0, 50, 0.0, False, 0.0, True)
        result = _worker_process_feature(args, emitter=emitter)

        assert result is not None
        events = emitter.events
        event_types = [e["type"] for e in events]

        assert "polygon_loaded" in event_types, f"Missing polygon_loaded. Events: {event_types}"
        assert "final_result" in event_types, f"Missing final_result. Events: {event_types}"

    def test_no_emitter_still_works(self):
        from axis_aligned_lir_worker import _worker_process_feature
        poly = self._make_square_polygon()
        from shapely.wkb import dumps as wkb_dumps
        wkb_bytes = wkb_dumps(poly)

        args = (1, wkb_bytes, 0.0, 50, 0.0, False, 0.0, True)
        result = _worker_process_feature(args, emitter=None)
        assert result is not None

    def test_rotation_events(self):
        from axis_aligned_lir_worker import _worker_process_feature
        poly = self._make_rotated_polygon()
        from shapely.wkb import dumps as wkb_dumps
        wkb_bytes = wkb_dumps(poly)

        emitter = TraceEmitter("axis_aligned_lir", "1", "test_diamond",
                               {"axis_angle": 45.0})
        args = (1, wkb_bytes, 45.0, 50, 0.0, False, 0.0, True)
        result = _worker_process_feature(args, emitter=emitter)

        assert result is not None
        event_types = [e["type"] for e in emitter.events]
        assert "rotation_applied" in event_types
        assert "rotation_removed" in event_types


class TestEmitterWithBCRSWorker:
    """Verify event emission from the BCRS worker."""

    def _make_test_polygon(self):
        return Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])

    def test_bcrs_candidates_events(self):
        from bcrs_worker import _worker_process_feature
        poly = self._make_test_polygon()
        from shapely.wkb import dumps as wkb_dumps
        wkb_bytes = wkb_dumps(poly)

        emitter = TraceEmitter("bcrs_standard", "1", "test_square",
                               {"angle_step": 10})
        args = (1, wkb_bytes, 10, 20, 50, 0.0, False, 0.0, 5, True)
        result = _worker_process_feature(args, emitter=emitter)

        assert result is not None
        event_types = [e["type"] for e in emitter.events]
        assert "polygon_loaded" in event_types
        assert "edge_angles_found" in event_types


class TestEmitterWithApproxWorker:
    """Verify event emission from approximation workers."""

    def _make_test_polygon(self):
        return Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])

    def test_approximation_standard_events(self):
        from approximation_standard_worker import _worker_process_feature
        from shapely.wkb import dumps as wkb_dumps
        wkb_bytes = wkb_dumps(self._make_test_polygon())

        emitter = TraceEmitter("approximation_standard", "1", "test",
                               {"angle_step": 10})
        args = (1, wkb_bytes, 10, 20, 50, 0.0, False, 0.0)
        result = _worker_process_feature(args, emitter=emitter)

        assert result is not None
        event_types = [e["type"] for e in emitter.events]
        assert "final_result" in event_types

    def test_approximation_fast_events(self):
        from approximation_fast_worker import _worker_process_feature
        from shapely.wkb import dumps as wkb_dumps
        wkb_bytes = wkb_dumps(self._make_test_polygon())

        emitter = TraceEmitter("approximation_fast", "1", "test",
                               {"angle_step": 10})
        args = (1, wkb_bytes, 10, 20, 50, 0.0, False, 0.0)
        result = _worker_process_feature(args, emitter=emitter)

        assert result is not None
        event_types = [e["type"] for e in emitter.events]
        assert "final_result" in event_types


class TestEmitterWithContainedWorker:
    """Verify event emission from contained workers."""

    def _make_test_polygon(self):
        return Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])

    def test_contained_standard_events(self):
        from contained_standard_worker import _worker_process_feature
        from shapely.wkb import dumps as wkb_dumps
        wkb_bytes = wkb_dumps(self._make_test_polygon())

        emitter = TraceEmitter("contained_standard", "1", "test",
                               {"angle_step": 10})
        args = (1, wkb_bytes, 10, 10, 50, 0.0, False, 0.0, 3, True)
        result = _worker_process_feature(args, emitter=emitter)

        assert result is not None
        event_types = [e["type"] for e in emitter.events]
        assert "final_result" in event_types
        assert "edge_angles_found" in event_types

    def test_contained_fast_events(self):
        from contained_fast_worker import _worker_process_feature
        from shapely.wkb import dumps as wkb_dumps
        wkb_bytes = wkb_dumps(self._make_test_polygon())

        emitter = TraceEmitter("contained_fast", "1", "test",
                               {"angle_step": 10})
        args = (1, wkb_bytes, 10, 10, 50, 0.0, False, 0.0, 3, True)
        result = _worker_process_feature(args, emitter=emitter)

        assert result is not None
        event_types = [e["type"] for e in emitter.events]
        assert "final_result" in event_types


class TestEventCompleteness:
    """Verify that all required event types from the schema can be produced."""

    def test_all_mandatory_event_types(self):
        emitter = TraceEmitter("test", "1", "p", {})

        emitter.emit("SETUP", "polygon_loaded", "", "", **{"exterior": [], "holes": [], "bbox": [0,0,1,1], "area": 1.0, "vertex_count": 4, "poly_type": "convex", "is_valid": True})
        emitter.emit("SETUP", "rotation_applied", "", "", **{"angle_deg": 45.0, "origin": [0,0], "exterior": [], "holes": []})
        emitter.emit("SETUP", "rotation_removed", "", "", **{"angle_deg": 45.0})
        emitter.emit("GRID", "grid_built", "", "", **{"xs_vertex": [], "ys_vertex": [], "xs_augmented": [], "ys_augmented": [], "n_cols": 5, "n_rows": 5, "n_cells": 25})
        emitter.emit("MASK", "mask_row_started", "", "", **{"row_idx": 0, "y0": 0.0, "y1": 1.0, "y_mid": 0.5})
        emitter.emit("MASK", "mask_row_intervals", "", "", **{"row_idx": 0, "y_mid": 0.5, "intervals": []})
        emitter.emit("MASK", "mask_cells_set", "", "", **{"row_idx": 0, "valid_cols": [], "invalid_cols": [], "row_valid_count": 0, "total_valid_so_far": 0})
        emitter.emit("MASK", "mask_complete", "", "", **{"total_valid": 0, "total_cells": 25, "fill_ratio": 0.0, "rle_rows": []})
        emitter.emit("CANDIDATES", "edge_angles_found", "", "", **{"angles_deg": [], "edge_lengths": [], "smoothed": True})
        emitter.emit("CANDIDATES", "upper_bound_computed", "", "", **{"angle_deg": 0.0, "upper_bound": 100.0, "pruned": False, "prune_threshold": 50.0})
        emitter.emit("CANDIDATES", "candidate_found", "", "", **{"angle_deg": 0.0, "rect": [0,0,1,1], "area": 1.0, "source": "grid", "rank": 0})
        emitter.emit("ANGLE_SEARCH", "brent_bracket_set", "", "", **{"center_deg": 0.0, "bracket_deg": [-3, 3], "half_width": 3.0})
        emitter.emit("ANGLE_SEARCH", "angle_polished", "", "", **{"angle_deg": 0.0, "area": 1.0, "rect": [0,0,1,1], "iterations_used": 5})
        emitter.emit("BCRS_SOLVE", "bcrs_seed_set", "", "", **{"angle_deg": 0.0, "seed_bounds": [0,0,1,1], "seed_area": 1.0})
        emitter.emit("BCRS_SOLVE", "bcrs_boundary_expand", "", "", **{"side": "left", "from_coord": 0.0, "to_coord": -0.5, "rect_after": [0,0,1,1], "area_after": 1.0})
        emitter.emit("CABF", "cabf_iteration_started", "", "", **{"iteration": 0, "rect_in": [0,0,1,1], "area_in": 1.0})
        emitter.emit("CABF", "cabf_iteration_done", "", "", **{"iteration": 0, "rect_out": [0,0,1,1], "area_out": 1.0, "delta_area": 0.0})
        emitter.emit("CERT", "cert_started", "", "", **{"rect": [0,0,1,1], "area": 1.0, "method": "covers"})
        emitter.emit("CERT", "cert_passed", "", "", **{"rect": [0,0,1,1], "area": 1.0, "inset": 0.0})
        emitter.emit("CERT", "cert_failed_shrink", "", "", **{"attempt": 1, "rect_before": [0,0,1,1], "rect_after": [0,0,0.5,0.5], "eps": 1e-5})
        emitter.emit("CERT", "cert_fallback", "", "", **{"reason": "shrink_exhausted", "fallback": "conservative_inner"})
        emitter.emit("RESULT", "best_updated", "", "", **{"rect": [0,0,1,1], "area": 1.0, "pct_polygon": 50.0, "angle_deg": 0.0, "source": "HISTOGRAM", "prev_area": 0.0})
        emitter.emit("RESULT", "final_result", "", "", **{"rect": [0,0,1,1], "area": 1.0, "pct_polygon": 50.0, "angle_deg": 0.0, "algorithm": "test", "total_events": 22, "elapsed_ms": 1.0})

        expected_count = 23
        assert len(emitter.events) == expected_count
