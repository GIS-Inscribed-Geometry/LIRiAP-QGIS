"""
LIRiAP Contained Fast algorithm wrapper.

This file keeps the geometric logic in `contained_fast_worker.py` and provides
a consistent QGIS-facing interface for workers, chunking, and optional Numba
bootstrap.
"""

import concurrent.futures as _cf
import os
import sys

from PyQt5.QtCore import QVariant
from qgis.PyQt.QtCore import QCoreApplication
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
if script_dir not in sys.path:
    sys.path.append(script_dir)

from contained_fast_worker import _worker_process_feature, _NUMBA_AVAILABLE
from help_descriptions import build_short_help
from numba_bootstrap import ensure_numba


    # Process a contiguous feature slice using the worker's native logic.
def _process_slice(job_array, start, end, shared_params):
    out = {}
    best_effort_count = 0
    for i in range(start, end):
        feat_id, wkb_bytes = job_array[i]
        res = _worker_process_feature((feat_id, wkb_bytes, *shared_params))
        if res is None:
            continue
        (_, wkt, area, angle, ratio,
         cand_rank, s2_gain, best_effort) = res
        out[feat_id] = (wkt, area, angle, ratio, cand_rank, s2_gain, best_effort)
        best_effort_count += int(best_effort)
    return out, best_effort_count


class InscribedRectangleContainedFast(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    ALWAYS_RETURN = "ALWAYS_RETURN"
    ANGLE_STEP = "ANGLE_STEP"
    GRID_COARSE = "GRID_COARSE"
    GRID_FINE = "GRID_FINE"
    MAX_RATIO = "MAX_RATIO"
    TOP_K = "TOP_K"
    USE_BUFFER = "USE_BUFFER"
    BUFFER_VALUE = "BUFFER_VALUE"
    N_WORKERS = "N_WORKERS"
    USE_CHUNKING = "USE_CHUNKING"
    AUTO_INSTALL_NUMBA = "AUTO_INSTALL_NUMBA"

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.INPUT, self.tr("Input layer (polygons)"), [QgsProcessing.TypeVectorPolygon]
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.ALWAYS_RETURN,
            self.tr("Always return a best-effort rectangle if strict certification fails"),
            defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.ANGLE_STEP, self.tr("Rotation angle step [°] — fallback"),
            QgsProcessingParameterNumber.Integer, defaultValue=5, minValue=1, maxValue=45
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.GRID_COARSE, self.tr("Coarse grid resolution"),
            QgsProcessingParameterNumber.Integer, defaultValue=40, minValue=10, maxValue=1000
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.GRID_FINE, self.tr("Fine grid resolution"),
            QgsProcessingParameterNumber.Integer, defaultValue=120, minValue=30, maxValue=1000
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_RATIO, self.tr("Max aspect ratio long:short (0=unlimited)"),
            QgsProcessingParameterNumber.Double, defaultValue=1.6, minValue=0.0, maxValue=20.0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.TOP_K, self.tr("Stage 1 top-K candidates"),
            QgsProcessingParameterNumber.Integer, defaultValue=3, minValue=1, maxValue=20
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_BUFFER, self.tr("Apply containment buffer"), defaultValue=False
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.BUFFER_VALUE, self.tr("Buffer value in map units"),
            QgsProcessingParameterNumber.Double, defaultValue=-0.5,
            minValue=-1e9, maxValue=1e9, optional=True
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_CHUNKING, self.tr("Enable chunking (parallel mode)"), defaultValue=False
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.AUTO_INSTALL_NUMBA, self.tr("Attempt to install numba if missing"), defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.N_WORKERS, self.tr("Workers (0 = auto, 1 = serial, >1 = custom)"),
            QgsProcessingParameterNumber.Integer, defaultValue=1, minValue=0, maxValue=512
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, self.tr("Inscribed rectangles")
        ))

    def processAlgorithm(self, parameters, context, feedback):
        # 1) Read user parameters and optional runtime bootstrap flags.
        layer = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        always_return = self.parameterAsBoolean(parameters, self.ALWAYS_RETURN, context)
        angle_step = self.parameterAsInt(parameters, self.ANGLE_STEP, context)
        grid_coarse = self.parameterAsInt(parameters, self.GRID_COARSE, context)
        grid_fine = self.parameterAsInt(parameters, self.GRID_FINE, context)
        max_ratio = self.parameterAsDouble(parameters, self.MAX_RATIO, context)
        top_k = self.parameterAsInt(parameters, self.TOP_K, context)
        use_buffer = self.parameterAsBoolean(parameters, self.USE_BUFFER, context)
        buf_value = self.parameterAsDouble(parameters, self.BUFFER_VALUE, context) if use_buffer else 0.0
        use_chunking = self.parameterAsBoolean(parameters, self.USE_CHUNKING, context)
        auto_install_numba = self.parameterAsBoolean(parameters, self.AUTO_INSTALL_NUMBA, context)
        n_workers_in = self.parameterAsInt(parameters, self.N_WORKERS, context)
        _, installed_now = ensure_numba(feedback, auto_install_numba)
        if installed_now:
            feedback.pushInfo("Re-run the algorithm to activate numba acceleration.")

        # 2) Define the output schema.
        fields = QgsFields()
        fields.append(QgsField("feat_id", QVariant.Int))
        fields.append(QgsField("area", QVariant.Double))
        fields.append(QgsField("angle_deg", QVariant.Double))
        fields.append(QgsField("ratio", QVariant.Double))
        fields.append(QgsField("cand_rank", QVariant.Int))
        fields.append(QgsField("s2_gain", QVariant.Double))
        fields.append(QgsField("best_effort", QVariant.Int))

        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields, QgsWkbTypes.Polygon, layer.crs()
        )
        if sink is None:
            raise QgsProcessingException("Could not create output layer.")

        # 3) Serialize source features once for safe worker execution.
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

        # 4) Resolve execution mode from worker count.
        total = len(job_array)
        cpu = os.cpu_count() or 1
        n_workers = n_workers_in if n_workers_in > 0 else cpu
        n_workers = max(1, min(n_workers, total))
        use_parallel = n_workers > 1 and total > 4
        shared_params = (
            angle_step, grid_coarse, grid_fine, max_ratio,
            use_buffer, buf_value, top_k, always_return,
        )

        # 5) Execute serially, chunked-parallel, or per-feature parallel.
        results = {}
        best_effort_count = 0

        if not use_parallel or n_workers == 1:
            done = 0
            for feat_id, wkb_bytes in job_array:
                if feedback.isCanceled():
                    return {self.OUTPUT: dest_id}
                res = _worker_process_feature((feat_id, wkb_bytes, *shared_params))
                if res is not None:
                    (_, wkt, area, angle, ratio,
                     cand_rank, s2_gain, best_effort) = res
                    results[feat_id] = (wkt, area, angle, ratio, cand_rank, s2_gain, best_effort)
                    best_effort_count += int(best_effort)
                done += 1
                feedback.setProgress(int(done / total * 100))
        elif use_chunking:
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
                    res_dict, chunk_best_effort = fut.result()
                    results.update(res_dict)
                    best_effort_count += chunk_best_effort
                    done += chunk_size
                    feedback.setProgress(int(done / total * 100))
        else:
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
                        (fid, wkt, area, angle, ratio,
                         cand_rank, s2_gain, best_effort) = res
                        results[fid] = (wkt, area, angle, ratio, cand_rank, s2_gain, best_effort)
                        best_effort_count += int(best_effort)
                    done += 1
                    feedback.setProgress(int(done / total * 100))

        # 6) Write results in original feature order for deterministic output.
        for fid in feat_order:
            if fid not in results:
                continue
            wkt, area, angle, ratio, cand_rank, s2_gain, best_effort = results[fid]
            f_out = QgsFeature(fields)
            f_out.setGeometry(QgsGeometry.fromWkt(wkt))
            f_out.setAttributes([fid, area, angle, ratio, cand_rank, s2_gain, best_effort])
            sink.addFeature(f_out, QgsFeatureSink.FastInsert)

        feedback.pushInfo(f"Best-effort fallback used on {best_effort_count} feature(s).")
        return {self.OUTPUT: dest_id}

    def name(self):
        return "lir_contained_fast"

    def displayName(self):
        return self.tr("Contained Fast")

    def group(self):
        return self.tr("LIRiAP")

    def groupId(self):
        return "liriap"

    def createInstance(self):
        return InscribedRectangleContainedFast()

    def tr(self, s):
        return QCoreApplication.translate("Processing", s)

    def shortHelpString(self):
        return build_short_help("Contained Fast", "contained_fast", _NUMBA_AVAILABLE)

