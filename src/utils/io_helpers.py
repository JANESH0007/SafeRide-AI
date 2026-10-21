"""
I/O helpers — CSV logging, crop saving, video writing.
"""

from __future__ import annotations

import csv
import os
from datetime import timedelta
from typing import Optional

import cv2
import numpy as np

import config.settings as cfg


class IOHelper:
    """Handles violation logging, image crop saving, and output video."""

    CSV_HEADER = [
        "violation_id",
        "timestamp",
        "frame_idx",
        "violation_type",
        "rider_count",
        "helmet_prob",
        "plate_text",
        "crop_path",
        "track_id",
    ]

    def __init__(self, csv_path: str | None = None):
        self.csv_path = csv_path or cfg.VIOLATIONS_CSV
        self._violation_counter = 0
        self._writer = None
        self._csv_file = None
        self._ensure_csv()

    # ──────────────────────────────────────────────────────
    def _ensure_csv(self):
        """Create the CSV with a header if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.CSV_HEADER)

    def _open_writer(self):
        if self._writer is None:
            self._csv_file = open(self.csv_path, "a", newline="")
            self._writer = csv.writer(self._csv_file)

    # ──────────────────────────────────────────────────────
    def log_violation(self, violation: dict) -> str:
        """Append one violation row to the CSV.  Returns the violation ID."""
        self._open_writer()
        self._violation_counter += 1
        vid = f"V{self._violation_counter:04d}"

        ts = violation.get("timestamp", 0)
        ts_str = str(timedelta(seconds=int(ts)))

        row = [
            vid,
            ts_str,
            violation.get("frame_idx", ""),
            violation.get("type", ""),
            violation.get("rider_count", 1),
            violation.get("helmet_prob", ""),
            violation.get("plate_text", ""),
            violation.get("crop_path", ""),
            violation.get("track_id", ""),
        ]
        self._writer.writerow(row)
        self._csv_file.flush()
        return vid

    # ──────────────────────────────────────────────────────
    @staticmethod
    def save_crop(
        frame: np.ndarray,
        bbox: tuple,
        track_id,
        frame_idx: int,
        label: str = "violation",
    ) -> str:
        """Save a cropped image of the violation area.  Returns the path."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        # Add padding
        pad = 30
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = frame[y1:y2, x1:x2]
        fname = f"{label}_track{track_id}_f{frame_idx}.jpg"
        path = os.path.join(cfg.CROPS_DIR, fname)
        cv2.imwrite(path, crop)
        return path

    # ──────────────────────────────────────────────────────
    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None


class VideoWriter:
    """Context manager for writing an annotated output video."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        self.path = path

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
