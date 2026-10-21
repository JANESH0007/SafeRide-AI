"""
SafeRide AI — PlateReader re-export for the plate_ocr subpackage.

This simply re-exports the PlateReader from detection so both
``from src.detection import PlateReader`` and
``from src.plate_ocr import PlateReader`` work.
"""

from src.detection.plate_reader import PlateReader  # noqa: F401
