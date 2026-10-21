"""
Number-plate reader — EasyOCR (pretrained).

Pipeline:
  1. Crop the lower region of the motorcycle bounding box.
  2. Pre-process (greyscale, CLAHE, sharpen).
  3. Run EasyOCR to find text regions.
  4. Filter / clean results with an Indian-plate regex.
"""

from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np

import config.settings as cfg


class PlateReader:
    """Reads number plates from motorcycle bounding-box crops using EasyOCR."""

    _plate_re = re.compile(cfg.PLATE_REGEX)

    def __init__(self):
        """Eagerly initialise EasyOCR so model download happens at startup."""
        import easyocr
        gpu = False
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            pass
        self._reader = easyocr.Reader(
            cfg.PLATE_LANGUAGES, gpu=gpu, verbose=False
        )

    # ──────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(region: np.ndarray) -> np.ndarray:
        """Enhance the plate crop for better OCR accuracy."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # CLAHE for local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Slight sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return sharpened

    # ──────────────────────────────────────────────────────
    def read(
        self,
        frame: np.ndarray,
        motorcycle_bbox: tuple,
    ) -> tuple[Optional[str], Optional[np.ndarray]]:
        """
        Attempt to read a number plate near the motorcycle.

        Returns
        -------
        plate_text : str | None
            Cleaned plate string, or None if nothing plausible found.
        plate_crop : np.ndarray | None
            The image region that was scanned.
        """
        x1, y1, x2, y2 = motorcycle_bbox
        h, w = frame.shape[:2]
        bh = y2 - y1  # bbox height

        # Search the lower 50 % of the motorcycle box + a little below
        crop_y1 = max(0, int(y1 + bh * 0.5))
        crop_y2 = min(h, int(y2 + bh * 0.25))
        crop_x1 = max(0, x1)
        crop_x2 = min(w, x2)

        plate_region = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if plate_region.size == 0 or plate_region.shape[0] < 10 or plate_region.shape[1] < 20:
            return None, None

        preprocessed = self._preprocess(plate_region)

        reader = self._reader
        try:
            results = reader.readtext(preprocessed, detail=1, paragraph=False)
        except Exception:
            return None, plate_region

        # Combine all detected text fragments
        raw_parts: list[str] = []
        for _bbox, text, _conf in results:
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
            if len(cleaned) >= 2:
                raw_parts.append(cleaned)

        combined = "".join(raw_parts)

        if len(combined) < cfg.PLATE_MIN_CHARS:
            return None, plate_region

        # Try to match Indian-plate pattern
        match = self._plate_re.search(combined)
        if match:
            return match.group(), plate_region

        # Fall back to the full cleaned string (might still be useful)
        return combined if len(combined) >= cfg.PLATE_MIN_CHARS else None, plate_region
