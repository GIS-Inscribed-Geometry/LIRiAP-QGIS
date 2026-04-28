"""
LIRiAP TraceEmitter — structured event logging for the visualisation module.

Every solver function accepts an optional ``emitter`` keyword argument
(default ``None``). When an emitter is supplied, key algorithmic steps
record JSON-serialisable events that the ``LIRiAP_visualize.py`` viewer
renders as an animated trace.

The emitter is entirely opt-in — existing production call sites need no
changes because ``emitter=None`` is the default everywhere.

Classes
=======
TraceEmitter: Accumulates structured events during a single polygon solve

Verbosity Levels
================
VERBOSITY_FULL: Record all events
VERBOSITY_NORMAL: Standard verbosity
VERBOSITY_SUMMARY: Summary only

Event Structure
===============
{
    "seq": int,          # Sequence number
    "phase": str,        # Algorithm phase (SETUP, SEARCH, etc.)
    "type": str,         # Event type
    "label": str,        # Short label (truncated to 40 chars)
    "narration": str,    # Detailed description (truncated to 200 chars)
    "ext": dict          # Extended fields
}

Output
======
to_trace() returns JSON-serializable dict with schema_version, trace_id,
algorithm, polygon_id, polygon_name, params, elapsed_ms, events

See Also
========
LIRiAP_visualize.py: Visualization module that renders traces
"""

from __future__ import annotations

import time
import uuid
from typing import Any


VERBOSITY_FULL = "FULL"
VERBOSITY_NORMAL = "NORMAL"
VERBOSITY_SUMMARY = "SUMMARY"


class TraceEmitter:
    """Accumulate structured events during a single polygon solve.

    Parameters
    ----------
    algorithm : str
        Algorithm identifier (e.g. ``"axis_aligned_lir"``).
    polygon_id : str
        Source feature identifier.
    polygon_name : str
        Human-readable label for the polygon.
    params : dict
        Algorithm parameters as passed by the user.
    verbosity : str
        One of ``"FULL"``, ``"NORMAL"``, ``"SUMMARY"``.
    """

    def __init__(
        self,
        algorithm: str,
        polygon_id: str,
        polygon_name: str,
        params: dict,
        verbosity: str = VERBOSITY_NORMAL,
    ):
        self._algorithm = algorithm
        self._polygon_id = polygon_id
        self._polygon_name = polygon_name
        self._params = params
        self._verbosity = verbosity
        self._seq = 0
        self._events: list[dict] = []
        self._start_ms = time.monotonic() * 1000
        self._best_area = 0.0
        self._seq_data: dict = {}

    @property
    def verbosity(self) -> str:
        return self._verbosity

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def emit(
        self,
        phase: str,
        type_: str,
        label: str,
        narration: str,
        **ext_fields: Any,
    ) -> None:
        """Append one event to the internal buffer."""
        self._events.append({
            "seq": self._seq,
            "phase": phase,
            "type": type_,
            "label": label[:40],
            "narration": narration[:200],
            "ext": ext_fields,
        })
        self._seq += 1

    def to_trace(self) -> dict:
        """Return the complete trace dict ready for ``json.dumps()``."""
        elapsed = time.monotonic() * 1000 - self._start_ms
        return {
            "schema_version": "1.0",
            "trace_id": str(uuid.uuid4()),
            "algorithm": self._algorithm,
            "polygon_id": self._polygon_id,
            "polygon_name": self._polygon_name,
            "params": self._params,
            "elapsed_ms": round(elapsed, 2),
            "events": self._events,
        }
