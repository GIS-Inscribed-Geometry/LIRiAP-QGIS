"""
LIRiAP QGIS Plugin Main File
"""

import os
import sys

from qgis.PyQt.QtCore import QCoreAlgorithm
from qgis.core import QgsApplication, QgsProcessingProvider

# Import our provider
from LIRiAP_pack.LIRiAP_provider import LIRiAPProvider


class LIRiAPPlugin:
    """LIRiAP QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Initialize the plugin."""
        self.iface = iface
        self.provider = None
        self.plugin_dir = os.path.dirname(__file__)

    def initProcessing(self):
        """Initialize Processing provider for QGIS >= 3.8."""
        self.provider = LIRiAPProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        """Initialize the plugin GUI components."""
        self.initProcessing()

    def unload(self):
        """Unload the plugin."""
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)