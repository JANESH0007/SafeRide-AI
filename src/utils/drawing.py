"""
Drawing / annotation utilities for SafeRide AI.

Draws bounding boxes, labels, track IDs and violation badges on frames.
"""

from __future__ import annotations

import cv2
import numpy as np

import config.settings as cfg


class Annotator:
    """Draw detections and violations onto a frame (in-place by default)."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.55
    THICKNESS = 2
    LABEL_PAD = 4

    # ──────────────────────────────────────────────────────
    @classmethod
    def draw_box(
        cls,
        img: np.ndarray,
        bbox: tuple,
        color: tuple,
        label: str = "",
        thickness: int = 2,
    ):
        """Draw a rectangle + optional label above it."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cls.FONT, cls.FONT_SCALE, 1)
            # Label background
            cv2.rectangle(
                img,
                (x1, y1 - th - 2 * cls.LABEL_PAD),
                (x1 + tw + 2 * cls.LABEL_PAD, y1),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                img,
                label,
                (x1 + cls.LABEL_PAD, y1 - cls.LABEL_PAD),
                cls.FONT,
                cls.FONT_SCALE,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    # ──────────────────────────────────────────────────────
    @classmethod
    def annotate_frame(
        cls,
        frame: np.ndarray,
        persons: list[dict],
        motorcycles: list[dict],
        tracked: list[dict],
        violations: list[dict],
        helmet_results: dict[int, tuple[bool, float]],
        *,
        copy: bool = True,
    ) -> np.ndarray:
        """
        Full-frame annotation with colour-coded boxes and violation badges.

        Colours
        -------
        - Motorcycle …… cyan-blue
        - Helmet OK …… green
        - No helmet …… red
        - Triple ride … orange
        """
        out = frame.copy() if copy else frame

        # 1. Draw tracked motorcycles
        for trk in tracked:
            label = f"ID:{trk['track_id']}"
            cls.draw_box(out, trk["bbox"], cfg.COLOR_MOTORCYCLE, label)

        # 2. Draw persons (colour depends on helmet result)
        for person in persons:
            key = id(person)
            if key in helmet_results:
                has_helmet, prob = helmet_results[key]
                if has_helmet:
                    color = cfg.COLOR_HELMET_OK
                    lbl = f"Helmet {prob:.0%}"
                else:
                    color = cfg.COLOR_NO_HELMET
                    lbl = "NO HELMET"
            else:
                color = (180, 180, 180)
                lbl = "Person"
            cls.draw_box(out, person["bbox"], color, lbl)

        # 3. Draw violation badges
        for v in violations:
            if v["type"] == "triple_riding":
                bx = v["motorcycle_bbox"]
                badge = f"TRIPLE RIDING ({v['rider_count']} riders)"
                # Big warning box
                cx = (bx[0] + bx[2]) // 2
                cy = bx[1] - 30
                (tw, th), _ = cv2.getTextSize(badge, cls.FONT, 0.7, 2)
                cv2.rectangle(
                    out,
                    (cx - tw // 2 - 8, cy - th - 8),
                    (cx + tw // 2 + 8, cy + 8),
                    cfg.COLOR_TRIPLE,
                    cv2.FILLED,
                )
                cv2.putText(
                    out, badge,
                    (cx - tw // 2, cy),
                    cls.FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
                )

            elif v["type"] == "no_helmet":
                bx = v.get("rider_bbox", v["motorcycle_bbox"])
                badge = "⚠ NO HELMET"
                cls.draw_box(out, bx, cfg.COLOR_NO_HELMET, badge, thickness=3)

        # 4. Overlay stats bar
        cls._draw_stats_bar(out, violations)

        return out

    # ──────────────────────────────────────────────────────
    @classmethod
    def _draw_stats_bar(cls, img: np.ndarray, violations: list[dict]):
        """Semi-transparent status bar at the top."""
        h, w = img.shape[:2]
        bar_h = 36
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        n_helmet = sum(1 for v in violations if v["type"] == "no_helmet")
        n_triple = sum(1 for v in violations if v["type"] == "triple_riding")

        text = f"SafeRide AI  |  Frame violations: {len(violations)}  |  No-Helmet: {n_helmet}  |  Triple-Riding: {n_triple}"
        cv2.putText(
            img, text, (10, 24),
            cls.FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
