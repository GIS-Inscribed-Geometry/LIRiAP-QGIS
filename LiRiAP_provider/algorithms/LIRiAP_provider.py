"""
LIRiAP Provider for QGIS Processing
"""

import os
import sys

from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProcessingProvider

# Add the current directory to the path so we can import the algorithms
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.append(script_dir)


class LIRiAPProvider(QgsProcessingProvider):
    """LIRiAP Processing Provider."""

    def __init__(self):
        super().__init__()
        self._algorithms = None

    def id(self):
        """Returns the provider ID."""
        return "liriap"

    def name(self):
        """Returns the provider name."""
        return "LIRiAP"

    def icon(self):
        """Returns the provider icon."""
        # Use the BCRS image as the provider icon
        icon_path = os.path.join(os.path.dirname(__file__), "media", "BCRS.png")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
         
        # Fallback to a default icon if the specific one doesn't exist
        return QIcon()

    def longDescription(self):
        """Returns the provider long description."""
        return (
            "LIRiAP (Largest Inscribed Rectangle in Arbitrary Polygon) provides "
            "algorithms for computing the largest inscribed rectangles approximations "
            "for polygon features."
        )

    def loadAlgorithms(self):
        """Load the algorithms."""
        # Import algorithm classes here to avoid circular imports
        from .approximation_standard_algorithm import InscribedRectangleApproximationStandard
        from .approximation_fast_algorithm import InscribedRectangleApproximationFast
        from .contained_standard_algorithm import InscribedRectangleContainedStandard
        from .contained_fast_algorithm import InscribedRectangleContainedFast
        from .bcrs_algorithm import InscribedRectangleBCRS
        from .bcrs_fast_algorithm import InscribedRectangleBCRSFast
        
        # Clear any existing algorithms
        self._algorithms = []
        
        # Add all algorithms
        self._algorithms.append(InscribedRectangleApproximationStandard())
        self._algorithms.append(InscribedRectangleApproximationFast())
        self._algorithms.append(InscribedRectangleContainedStandard())
        self._algorithms.append(InscribedRectangleContainedFast())
        self._algorithms.append(InscribedRectangleBCRS())
        self._algorithms.append(InscribedRectangleBCRSFast())

    def algorithms(self):
        """Returns the list of algorithms."""
        if self._algorithms is None:
            self.loadAlgorithms()
        return self._algorithms

    def unload(self):
        """Unload the provider."""
        pass