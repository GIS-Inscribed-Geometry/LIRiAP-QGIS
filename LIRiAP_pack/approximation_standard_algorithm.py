"""
LIRiAP Approximation Standard algorithm wrapper.

This file keeps the geometric logic in `approximation_standard_worker.py` and
provides a consistent QGIS-facing interface for workers, chunking, and optional
Numba bootstrap.
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

from approximation_standard_worker import _worker_process_feature, _NUMBA_AVAILABLE
from help_descriptions import build_short_help
from numba_bootstrap import ensure_numba


    # Process a contiguous feature slice using the worker's native logic.
def _process_slice(job_array, start, end, shared_params):
    out = {}
    for i in range(start, end):
        feat_id, wkb_bytes = job_array[i]
        res = _worker_process_feature((feat_id, wkb_bytes, *shared_params))
        if res is None:
            continue
        _, wkt, area, angle, ratio = res
        out[feat_id] = (wkt, area, angle, ratio)
    return out


class InscribedRectangleApproximationStandard(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    ANGLE_STEP = "ANGLE_STEP"
    GRID_STEPS_COARSE = "GRID_STEPS_COARSE"
    GRID_STEPS_FINE = "GRID_STEPS_FINE"
    MAX_RATIO = "MAX_RATIO"
    REFINE_BUFFER = "REFINE_BUFFER"
    REFINE_BUFFER_VALUE = "REFINE_BUFFER_VALUE"
    N_WORKERS = "N_WORKERS"
    USE_CHUNKING = "USE_CHUNKING"
    AUTO_INSTALL_NUMBA = "AUTO_INSTALL_NUMBA"

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.INPUT, "Input layer (polygons)", [QgsProcessing.TypeVectorPolygon]
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.ANGLE_STEP, "Fallback angle step [deg]",
            QgsProcessingParameterNumber.Integer, defaultValue=5, minValue=1, maxValue=45
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.GRID_STEPS_COARSE, "Coarse grid resolution",
            QgsProcessingParameterNumber.Integer, defaultValue=40, minValue=10, maxValue=300
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.GRID_STEPS_FINE, "Fine grid resolution",
            QgsProcessingParameterNumber.Integer, defaultValue=100, minValue=20, maxValue=500
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_RATIO, "Max aspect ratio long:short (0=unlimited)",
            QgsProcessingParameterNumber.Double, defaultValue=1.6, minValue=0.0, maxValue=20.0
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.REFINE_BUFFER, "Enable containment refinement buffer", defaultValue=False
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.REFINE_BUFFER_VALUE, "Buffer distance in CRS units",
            QgsProcessingParameterNumber.Double, defaultValue=-0.5,
            minValue=-1e9, maxValue=1e9, optional=True
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_CHUNKING, "Enable chunking (parallel mode)", defaultValue=False
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.AUTO_INSTALL_NUMBA, "Attempt to install numba if missing", defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.N_WORKERS, "Workers (0 = auto, 1 = serial, >1 = custom)",
            QgsProcessingParameterNumber.Integer, defaultValue=1, minValue=0, maxValue=128
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Inscribed rectangles"))

    def processAlgorithm(self, parameters, context, feedback):
        # 1) Read user parameters and optional runtime bootstrap flags.
        layer = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        angle_step = self.parameterAsInt(parameters, self.ANGLE_STEP, context)
        grid_coarse = self.parameterAsInt(parameters, self.GRID_STEPS_COARSE, context)
        grid_fine = self.parameterAsInt(parameters, self.GRID_STEPS_FINE, context)
        max_ratio = self.parameterAsDouble(parameters, self.MAX_RATIO, context)
        buf_enabled = self.parameterAsBoolean(parameters, self.REFINE_BUFFER, context)
        buf_value = self.parameterAsDouble(parameters, self.REFINE_BUFFER_VALUE, context)
        use_chunking = self.parameterAsBoolean(parameters, self.USE_CHUNKING, context)
        auto_install_numba = self.parameterAsBoolean(parameters, self.AUTO_INSTALL_NUMBA, context)
        n_workers_param = self.parameterAsInt(parameters, self.N_WORKERS, context)
        _, installed_now = ensure_numba(feedback, auto_install_numba)
        if installed_now:
            feedback.pushInfo("Re-run the algorithm to activate numba acceleration.")

        # 2) Define the output schema.
        fields = QgsFields()
        fields.append(QgsField("feat_id", QVariant.Int))
        fields.append(QgsField("area", QVariant.Double))
        fields.append(QgsField("angle_deg", QVariant.Double))
        fields.append(QgsField("ratio", QVariant.Double))

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
        n_workers = n_workers_param if n_workers_param > 0 else cpu
        n_workers = max(1, min(n_workers, total))
        use_parallel = n_workers > 1 and total > 4
        shared_params = (angle_step, grid_coarse, grid_fine, max_ratio, buf_enabled, buf_value)

        # 5) Execute serially, chunked-parallel, or per-feature parallel.
        results = {}
        if not use_parallel or n_workers == 1:
            done = 0
            for feat_id, wkb_bytes in job_array:
                if feedback.isCanceled():
                    return {self.OUTPUT: dest_id}
                res = _worker_process_feature((feat_id, wkb_bytes, *shared_params))
                if res is not None:
                    _, wkt, area, angle, ratio = res
                    results[feat_id] = (wkt, area, angle, ratio)
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
                    results.update(fut.result())
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
                        fid, wkt, area, angle, ratio = res
                        results[fid] = (wkt, area, angle, ratio)
                    done += 1
                    feedback.setProgress(int(done / total * 100))

        # 6) Write results in original feature order for deterministic output.
        for fid in feat_order:
            if fid not in results:
                continue
            wkt_rect, area, angle, ratio = results[fid]
            f_out = QgsFeature(fields)
            f_out.setGeometry(QgsGeometry.fromWkt(wkt_rect))
            f_out.setAttributes([fid, area, angle, ratio])
            sink.addFeature(f_out, QgsFeatureSink.FastInsert)

        return {self.OUTPUT: dest_id}

    def name(self):
        return "lir_approximation_standard"

    def displayName(self):
        return "Approximation Standard"

    def group(self):
        return "LIRiAP"

    def groupId(self):
        return "liriap"

    def createInstance(self):
        return InscribedRectangleApproximationStandard()

    def tr(self, s):
        return QCoreApplication.translate("Processing", s)

    def shortHelpString(self):
        return build_short_help("Approximation Standard", "approximation_standard", _NUMBA_AVAILABLE)

