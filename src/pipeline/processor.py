"""
SafeRide AI — main processing pipeline.

Orchestrates:
  detection → tracking → rule logic → plate OCR → logging
for every frame of a video.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config.settings as cfg
from src.detection.vehicle_person_detector import VehiclePersonDetector
from src.detection.helmet_classifier import HelmetClassifier
from src.detection.plate_reader import PlateReader
from src.tracking.vehicle_tracker import VehicleTracker
from src.utils.violation_engine import ViolationEngine
from src.utils.drawing import Annotator
from src.utils.io_helpers import IOHelper, VideoWriter


class SafeRidePipeline:
    """End-to-end pipeline: video in → annotated video + violation CSV out."""

    def __init__(self, enable_ocr: bool = True):
        print("🏍️  SafeRide AI — initialising models …")

        self.detector = VehiclePersonDetector()
        print("  ✓ YOLOv8 vehicle/person detector loaded")

        self.helmet_clf = HelmetClassifier()
        print("  ✓ Helmet classifier ready (face-detection proxy)")

        self.tracker = VehicleTracker()
        print("  ✓ DeepSORT tracker ready (IoU mode)")

        self.rules = ViolationEngine()
        self.annotator = Annotator()
        self.io = IOHelper()

        self.enable_ocr = enable_ocr
        self.plate_reader: Optional[PlateReader] = None
        if enable_ocr:
            self.plate_reader = PlateReader()
            print("  ✓ EasyOCR plate reader loaded")

        self._stats = {
            "frames_processed": 0,
            "total_violations": 0,
            "no_helmet": 0,
            "triple_riding": 0,
            "plates_read": 0,
        }
        print("  ✓ Pipeline ready\n")

    # ──────────────────────────────────────────────────────
    #  Single-frame processing
    # ──────────────────────────────────────────────────────
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Process one BGR frame.

        Returns
        -------
        annotated : np.ndarray
            Frame with drawn annotations.
        new_violations : list[dict]
            Only *first-time* violations (de-duplicated by track ID).
        """
        # 1. Detect persons & motorcycles
        persons, motorcycles = self.detector.detect(frame)

        if not motorcycles:
            # Nothing to do — just overlay the stats bar
            out = frame.copy()
            Annotator._draw_stats_bar(out, [])
            return out, []

        # 2. Track motorcycles → stable IDs
        tracked = self.tracker.update(motorcycles, frame)

        # Build a combined motorcycle list with track IDs
        # (match tracked boxes back to detected boxes by IoU)
        enriched_motos = self._enrich_with_tracks(motorcycles, tracked)

        # 3. Match riders to motorcycles
        moto_riders = self.rules.match_riders(persons, enriched_motos)

        # 4. Classify helmets for every matched rider
        helmet_results: dict[int, tuple[bool, float]] = {}
        for m_idx, riders in moto_riders.items():
            for rider in riders:
                has_helmet, prob = self.helmet_clf.classify(frame, rider["bbox"])
                helmet_results[id(rider)] = (has_helmet, prob)

        # 5. Check violations
        violations = self.rules.check_violations(
            moto_riders, enriched_motos, helmet_results,
        )

        # 6. De-dup and log new violations
        new_violations: list[dict] = []
        for v in violations:
            if not self.rules.is_new(v.get("track_id"), v["type"]):
                continue

            # Try to read the plate
            plate_text = None
            if self.plate_reader:
                plate_text, _ = self.plate_reader.read(frame, v["motorcycle_bbox"])
                if plate_text:
                    self._stats["plates_read"] += 1

            v["plate_text"] = plate_text
            v["timestamp"] = timestamp
            v["frame_idx"] = frame_idx

            # Save crop
            crop_path = IOHelper.save_crop(
                frame,
                v["motorcycle_bbox"],
                v.get("track_id", "unk"),
                frame_idx,
                label=v["type"],
            )
            v["crop_path"] = crop_path

            # Log to CSV
            vid = self.io.log_violation(v)
            v["violation_id"] = vid
            new_violations.append(v)

            # Update stats
            self._stats["total_violations"] += 1
            self._stats[v["type"]] = self._stats.get(v["type"], 0) + 1

        self._stats["frames_processed"] += 1

        # 7. Annotate
        annotated = self.annotator.annotate_frame(
            frame, persons, enriched_motos, tracked,
            violations, helmet_results,
        )

        return annotated, new_violations

    # ──────────────────────────────────────────────────────
    #  Full-video processing
    # ──────────────────────────────────────────────────────
    def process_video(
        self,
        input_path: str,
        output_path: str | None = None,
        display: bool = False,
        max_frames: int | None = None,
        skip_frames: int = 0,
    ) -> dict:
        """
        Process an entire video file.

        Parameters
        ----------
        input_path : str
            Path to the input video (or ``"0"`` for webcam).
        output_path : str | None
            If given, save the annotated video here.
        display : bool
            Show a live OpenCV window.
        max_frames : int | None
            Stop after this many frames (useful for testing).
        skip_frames : int
            Process every N-th frame (0 = process all).

        Returns
        -------
        dict   — summary statistics.
        """
        # Reset state
        self.rules.reset()
        self._stats = {k: 0 for k in self._stats}

        src = int(input_path) if input_path.isdigit() else input_path
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if max_frames and total:
            total = min(total, max_frames)

        writer: Optional[VideoWriter] = None
        if output_path:
            writer = VideoWriter(output_path, fps, w, h)

        pbar = tqdm(total=total or None, desc="Processing", unit="frame")
        frame_idx = 0
        t0 = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                # Optional frame skipping for speed
                if skip_frames and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                timestamp = frame_idx / fps

                annotated, new_violations = self.process_frame(
                    frame, frame_idx, timestamp,
                )

                # Print new violations to console
                for v in new_violations:
                    plate = v.get("plate_text") or "—"
                    print(
                        f"  🚨 [{v['violation_id']}] {v['type'].upper()}"
                        f"  @ {timestamp:.1f}s  plate={plate}"
                        f"  track={v.get('track_id', '?')}"
                    )

                if writer:
                    writer.write(annotated)

                if display:
                    cv2.imshow("SafeRide AI", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n⏹  Stopped by user.")
                        break

                frame_idx += 1
                pbar.update(1)

        finally:
            pbar.close()
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            self.io.close()

        elapsed = time.time() - t0
        self._stats["elapsed_seconds"] = round(elapsed, 2)
        self._stats["avg_fps"] = (
            round(self._stats["frames_processed"] / elapsed, 1) if elapsed > 0 else 0
        )

        return self._stats

    # ──────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _enrich_with_tracks(
        detections: list[dict],
        tracked: list[dict],
    ) -> list[dict]:
        """
        Match tracked motorcycle bboxes back to raw detections so
        every detection carries a ``track_id`` (if confirmed).
        """

        def _iou(a, b):
            ix1 = max(a[0], b[0])
            iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2])
            iy2 = min(a[3], b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            aa = (a[2] - a[0]) * (a[3] - a[1])
            ab = (b[2] - b[0]) * (b[3] - b[1])
            union = aa + ab - inter
            return inter / union if union > 0 else 0.0

        enriched = []
        for det in detections:
            best_id = None
            best_iou = 0.3  # minimum to match
            for trk in tracked:
                iou = _iou(det["bbox"], trk["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = trk["track_id"]
            enriched.append({**det, "track_id": best_id})
        return enriched
