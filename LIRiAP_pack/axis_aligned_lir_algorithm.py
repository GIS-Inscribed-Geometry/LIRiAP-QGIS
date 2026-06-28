"""
LIRiAP Axis-Aligned LIR algorithm wrapper.

Implements a QGIS Processing algorithm for exact fixed-axis Largest Inscribed
Rectangle solving with vertex-coordinate precision.

Algorithm Overview
==================
Exposes the exact axis-aligned LIR worker (``axis_aligned_lir_worker``) as a
QGIS Processing algorithm.

Four exact solvers dispatched based on polygon topology:
- convex_no_holes: Alt/Amenta O(n²) vertex-pair enumeration
- convex_with_holes / concave_no_holes / concave_with_holes:
  Daniels et al. O(n²) vertex-coordinate-grid + LRH scanline

Supports optional AXIS_ANGLE parameter to rotate the axis frame.

Execution Modes
===============
- Serial (N_WORKERS=1): Single-threaded execution
- Parallel (N_WORKERS>1): Per-feature parallel
- Chunked (N_WORKERS>1 + USE_CHUNKING=True): Chunk-based parallel

Output Fields
=============
- feat_id: Source feature ID
- area: Rectangle area in CRS map units
- axis_angle: Axis angle (echoes input)
- poly_type: convex_no_holes / convex_with_holes / concave_no_holes / concave_with_holes
- ratio: Aspect ratio (long:short)
- best_effort: 1 if fallback used, 0 otherwise

Parameters
==========
AXIS_ANGLE: Rotation of axis-aligned frame (0=horizontal)
GRID_FINE: Fallback grid resolution (if vertices > 500)
MAX_RATIO: Aspect ratio constraint (0=unlimited)
ALWAYS_RETURN : Enable best-effort fallback
USE_BUFFER: Apply containment buffer
BUFFER_VALUE: Buffer distance
N_WORKERS: Parallel workers (0=auto, 1=serial)
USE_CHUNKING: Chunked parallel mode
AUTO_INSTALL_NUMBA: Auto-install Numba JIT

References
==========
Alt & Amenta (1999): Convex polygon LIR
Daniels et al. (1997): Axis-aligned rectangle in polygons
Klingel (1986): Largest rectangle in histogram

See Also
========
axis_aligned_lir_worker: Geometric solver
"""

import concurrent.futures as _cf
import os
import sys

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
    QgsProcessingParameterVectorLayer,
    QgsWkbTypes,
)

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
for p in [script_dir, parent_dir]:
    if p not in sys.path:
        sys.path.append(p)

from axis_aligned_lir_worker import _worker_process_feature, _NUMBA_AVAILABLE
from help_descriptions import build_short_help
from numba_bootstrap import ensure_numba

# ---------------------------------------------------------------------------
# Chunk-based parallel execution helper
# ---------------------------------------------------------------------------


def _process_slice(job_array, start, end, shared_params):
    """
    Process a contiguous slice of *job_array* in a single thread.

    Parameters
    ----------
    job_array : list of (feat_id, wkb_bytes)
    start, end : int
        Half-open slice indices.
    shared_params : tuple
        Positional parameters forwarded to ``_worker_process_feature`` after
        ``(feat_id, wkb_bytes)``.

    Returns
    -------
    results : dict  {feat_id: (wkt, area, axis_angle, poly_type, ratio, best_effort)}
    best_effort_count : int
    """
    out = {}
    best_effort_count = 0
    for i in range(start, end):
        feat_id, wkb_bytes = job_array[i]
        res = _worker_process_feature((feat_id, wkb_bytes, *shared_params))
        if res is None:
            continue
        _, wkt, area, axis_angle, poly_type, ratio, best_effort = res
        out[feat_id] = (wkt, area, axis_angle, poly_type, ratio, best_effort)
        best_effort_count += int(best_effort)
    return out, best_effort_count


# ---------------------------------------------------------------------------
# QGIS Processing algorithm class
# ---------------------------------------------------------------------------


class InscribedRectangleAxisAligned(QgsProcessingAlgorithm):
    """
    QGIS Processing algorithm that wraps the exact axis-aligned LIR solver.

    Follows the same parameter layout and execution model as
    ``InscribedRectangleBCRS`` (``bcrs_algorithm.py``).
    """

    # ── Parameter name constants ────────────────────────────────────────────
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    AXIS_ANGLE = "AXIS_ANGLE"
    GRID_FINE = "GRID_FINE"
    MAX_RATIO = "MAX_RATIO"
    USE_BUFFER = "USE_BUFFER"
    BUFFER_VALUE = "BUFFER_VALUE"
    ALWAYS_RETURN = "ALWAYS_RETURN"
    N_WORKERS = "N_WORKERS"
    USE_CHUNKING = "USE_CHUNKING"
    AUTO_INSTALL_NUMBA = "AUTO_INSTALL_NUMBA"

    # ── Parameter declaration ───────────────────────────────────────────────

    def initAlgorithm(self, config=None):
        """Declare all algorithm parameters."""
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT,
                self.tr("Input layer (polygons)"),
                [QgsProcessing.TypeVectorPolygon],
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.AXIS_ANGLE,
                self.tr(
                    "Axis angle [°] — rotation of the 'axis-aligned' frame (0 = horizontal)"
                ),
                QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=-360.0,
                maxValue=360.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.GRID_FINE,
                self.tr(
                    "Fallback uniform grid resolution (used only when vertex density > 500)"
                ),
                QgsProcessingParameterNumber.Integer,
                defaultValue=120,
                minValue=10,
                maxValue=2000,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_RATIO,
                self.tr("Max aspect ratio long:short (0 = unlimited)"),
                QgsProcessingParameterNumber.Double,
                defaultValue=1.6,
                minValue=0.0,
                maxValue=20.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ALWAYS_RETURN,
                self.tr(
                    "Always return a best-effort rectangle if epsilon-inset certification fails"
                ),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_BUFFER,
                self.tr("Apply containment buffer after certification"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BUFFER_VALUE,
                self.tr("Buffer value in map units (negative = inward safety margin)"),
                QgsProcessingParameterNumber.Double,
                defaultValue=-0.5,
                minValue=-1e9,
                maxValue=1e9,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_WORKERS,
                self.tr("Workers (0 = auto, 1 = serial, >1 = custom)"),
                QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=0,
                maxValue=512,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_CHUNKING,
                self.tr("Enable chunked parallel execution"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.AUTO_INSTALL_NUMBA,
                self.tr(
                    "Attempt safe Numba auto-install if missing (JIT accelerates the LRH kernel)"
                ),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr("Axis-aligned inscribed rectangles"),
            )
        )

    # ── Main processing ─────────────────────────────────────────────────────

    def processAlgorithm(self, parameters, context, feedback):
        """
        Execute the axis-aligned LIR algorithm on all polygon features.

        Steps:
        1. Read parameters and optionally bootstrap Numba.
        2. Define output schema (feat_id, area, axis_angle, poly_type,
           ratio, best_effort).
        3. Serialise source features to WKB for worker-safe transmission.
        4. Resolve execution mode (serial / chunked-parallel /
           per-feature-parallel) from N_WORKERS and USE_CHUNKING.
        5. Dispatch to ``_worker_process_feature`` (serial) or
           ``_process_slice`` (chunked) or a per-feature future pool.
        6. Write results to the output sink in original feature order.
        """
        # 1) Parameters
        layer = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        axis_angle = self.parameterAsDouble(parameters, self.AXIS_ANGLE, context)
        grid_fine = self.parameterAsInt(parameters, self.GRID_FINE, context)
        max_ratio = self.parameterAsDouble(parameters, self.MAX_RATIO, context)
        always_return = self.parameterAsBoolean(parameters, self.ALWAYS_RETURN, context)
        use_buffer = self.parameterAsBoolean(parameters, self.USE_BUFFER, context)
        buf_value = (
            self.parameterAsDouble(parameters, self.BUFFER_VALUE, context)
            if use_buffer
            else 0.0
        )
        n_workers_in = self.parameterAsInt(parameters, self.N_WORKERS, context)
        use_chunking = self.parameterAsBoolean(parameters, self.USE_CHUNKING, context)
        auto_install_numba = self.parameterAsBoolean(
            parameters, self.AUTO_INSTALL_NUMBA, context
        )

        _, installed_now = ensure_numba(feedback, auto_install_numba)
        if installed_now:
            feedback.pushInfo(
                "Re-run the algorithm to activate Numba JIT acceleration."
            )

        # 2) Output schema
        fields = QgsFields()
        fields.append(QgsField("feat_id", QVariant.Int))
        fields.append(QgsField("area", QVariant.Double))
        fields.append(QgsField("axis_angle", QVariant.Double))
        fields.append(QgsField("poly_type", QVariant.String))
        fields.append(QgsField("ratio", QVariant.Double))
        fields.append(QgsField("best_effort", QVariant.Int))

        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields, QgsWkbTypes.Polygon, layer.crs()
        )
        if sink is None:
            raise QgsProcessingException("Could not create output layer.")

        # 3) Serialise features
        job_array = []
        feat_order = []
        for feat in layer.getFeatures():
            if feedback.isCanceled():
                return {self.OUTPUT: dest_id}
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            job_array.append((feat.id(), bytes(geom.asWkb())))
            feat_order.append(feat.id())

        if not job_array:
            return {self.OUTPUT: dest_id}

        # 4) Resolve execution mode
        total = len(job_array)
        cpu = os.cpu_count() or 1
        n_workers = n_workers_in if n_workers_in > 0 else cpu
        n_workers = max(1, min(n_workers, total))
        use_parallel = n_workers > 1 and total > 4

        shared_params = (
            axis_angle,
            grid_fine,
            max_ratio,
            use_buffer,
            buf_value,
            always_return,
        )

        # 5) Execute
        results: dict = {}
        best_effort_count = 0

        if not use_parallel or n_workers == 1:
            # ── Serial ───────────────────────────────────────────────────
            done = 0
            for feat_id, wkb_bytes in job_array:
                if feedback.isCanceled():
                    return {self.OUTPUT: dest_id}
                res = _worker_process_feature((feat_id, wkb_bytes, *shared_params))
                if res is not None:
                    _, wkt, area, ang, poly_type, ratio, best_effort = res
                    results[feat_id] = (wkt, area, ang, poly_type, ratio, best_effort)
                    best_effort_count += int(best_effort)
                done += 1
                feedback.setProgress(int(done / total * 100))

        elif use_chunking:
            # ── Chunked parallel ─────────────────────────────────────────
            base = total // n_workers
            rem = total % n_workers
            slices = []
            idx = 0
            for i in range(n_workers):
                size = base + (1 if i < rem else 0)
                if size:
                    slices.append((idx, idx + size))
                    idx += size

            with _cf.ThreadPoolExecutor(max_workers=n_workers) as exe:
                future_to_size = {
                    exe.submit(_process_slice, job_array, s, e, shared_params): (e - s)
                    for s, e in slices
                }
                done = 0
                for fut in _cf.as_completed(future_to_size):
                    if feedback.isCanceled():
                        exe.shutdown(wait=False, cancel_futures=True)
                        return {self.OUTPUT: dest_id}
                    chunk_size = future_to_size[fut]
                    res_dict, chunk_be = fut.result()
                    results.update(res_dict)
                    best_effort_count += chunk_be
                    done += chunk_size
                    feedback.setProgress(int(done / total * 100))

        else:
            # ── Per-feature parallel ──────────────────────────────────────
            with _cf.ThreadPoolExecutor(max_workers=n_workers) as exe:
                futures = {
                    exe.submit(_worker_process_feature, (fid, wkb, *shared_params)): fid
                    for fid, wkb in job_array
                }
                done = 0
                for fut in _cf.as_completed(futures):
                    if feedback.isCanceled():
                        exe.shutdown(wait=False, cancel_futures=True)
                        return {self.OUTPUT: dest_id}
                    res = fut.result()
                    if res is not None:
                        fid = futures[fut]
                        _, wkt, area, ang, poly_type, ratio, best_effort = res
                        results[fid] = (wkt, area, ang, poly_type, ratio, best_effort)
                        best_effort_count += int(best_effort)
                    done += 1
                    feedback.setProgress(int(done / total * 100))

        # 6) Write results in original feature order
        for fid in feat_order:
            if fid not in results:
                continue
            wkt, area, ang, poly_type, ratio, best_effort = results[fid]
            f_out = QgsFeature(fields)
            f_out.setGeometry(QgsGeometry.fromWkt(wkt))
            f_out.setAttributes([fid, area, ang, poly_type, ratio, best_effort])
            sink.addFeature(f_out, QgsFeatureSink.FastInsert)

        feedback.pushInfo(
            f"Best-effort fallback used on {best_effort_count} feature(s)."
        )
        return {self.OUTPUT: dest_id}

    # ── Algorithm metadata ───────────────────────────────────────────────────

    def name(self):
        """Unique processing algorithm identifier."""
        return "lir_axis_aligned"

    def displayName(self):
        """Human-readable algorithm name shown in the Processing toolbox."""
        return self.tr("Axis-Aligned LIR")

    def group(self):
        """Processing group name."""
        return self.tr("LIRiAP")

    def groupId(self):
        """Processing group identifier."""
        return "liriap"

    def createInstance(self):
        """Return a fresh instance of this algorithm (required by QGIS)."""
        return InscribedRectangleAxisAligned()

    def tr(self, s):
        """Translate *s* via the Qt translation system."""
        return QCoreApplication.translate("Processing", s)

    def shortHelpString(self):
        """Return the algorithm's short help HTML string."""
        return build_short_help("Axis-Aligned LIR", "axis_aligned", _NUMBA_AVAILABLE)
