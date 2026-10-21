"""
Vehicle tracker — DeepSORT wrapper (IoU-only mode for speed).

Tracks motorcycles across frames so that each violating vehicle is
counted **once**, not in every single frame.

Uses IoU-only association (no appearance embedder) for maximum speed.
"""

from __future__ import annotations

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

import config.settings as cfg


class VehicleTracker:
    """Maintains motorcycle tracks using DeepSORT (motion-only)."""

    def __init__(self):
        # No embedder → IoU / Kalman-only tracking (MUCH faster)
        self.tracker = DeepSort(
            max_age=cfg.TRACK_MAX_AGE,
            n_init=cfg.TRACK_N_INIT,
            max_iou_distance=0.7,
        )

    # ──────────────────────────────────────────────────────
    def update(
        self,
        detections: list[dict],
        frame: np.ndarray,
    ) -> list[dict]:
        """
        Feed motorcycle detections for one frame and get back
        confirmed tracks with stable IDs.

        Parameters
        ----------
        detections : list[dict]
            Each dict must have ``bbox`` (x1, y1, x2, y2) and ``confidence``.
        frame : np.ndarray
            Current BGR frame (kept for API compatibility).

        Returns
        -------
        list[dict]
            Each dict: ``track_id`` (int), ``bbox`` (x1,y1,x2,y2).
        """
        # DeepSORT expects ([left, top, w, h], confidence, class_name)
        raw: list[tuple] = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            raw.append(([x1, y1, w, h], det["confidence"], "motorcycle"))

        tracks = self.tracker.update_tracks(raw, frame=frame)

        active: list[dict] = []
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            ltrb = trk.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            active.append({
                "track_id": trk.track_id,
                "bbox": (x1, y1, x2, y2),
            })

        return active
