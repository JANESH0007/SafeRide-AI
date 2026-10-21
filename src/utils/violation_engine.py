"""
Violation engine — rule logic for SafeRide AI.

Rules
-----
1. **No helmet** — a rider on a motorcycle whose head-crop is classified
   as "no helmet" by the HelmetClassifier.
2. **Triple riding** — more than ``MAX_LEGAL_RIDERS`` persons matched to
   the same motorcycle.

De-duplication: violations are tracked per ``track_id × violation_type``
so the same offence is logged only once per tracked vehicle.
"""

from __future__ import annotations

import config.settings as cfg


class ViolationEngine:
    """Match riders → motorcycles and detect violations."""

    def __init__(self):
        # {track_id: {violation_type, ...}}
        self._seen: dict[str, set[str]] = {}

    # ────────────────── geometry helpers ──────────────────
    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        """Intersection-over-union of two (x1,y1,x2,y2) boxes."""
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        union = aa + ab - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _overlap_ratio(person: tuple, motorcycle: tuple) -> float:
        """
        Fraction of the *person* box that overlaps with the motorcycle box.
        Better than pure IoU when the person is small relative to the bike.
        """
        ix1 = max(person[0], motorcycle[0])
        iy1 = max(person[1], motorcycle[1])
        ix2 = min(person[2], motorcycle[2])
        iy2 = min(person[3], motorcycle[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        person_area = (person[2] - person[0]) * (person[3] - person[1])
        return inter / person_area if person_area > 0 else 0.0

    # ─────────── rider ↔ motorcycle matching ─────────────
    def match_riders(
        self,
        persons: list[dict],
        motorcycles: list[dict],
    ) -> dict[int, list[dict]]:
        """
        Greedily assign each person to the motorcycle with the highest
        overlap.  A person is matched only if their overlap exceeds
        ``RIDER_OVERLAP_THRESHOLD``.

        Parameters
        ----------
        persons, motorcycles : list[dict]
            Detections with ``bbox`` keys.  Motorcycles may also carry
            ``track_id``.

        Returns
        -------
        dict[int, list[dict]]
            motorcycle-index → list of matched person dicts.
        """
        groups: dict[int, list[dict]] = {i: [] for i in range(len(motorcycles))}

        for person in persons:
            best_idx, best_score = -1, 0.0
            for m_idx, moto in enumerate(motorcycles):
                # Use overlap ratio (how much of the person overlaps the bike)
                score = self._overlap_ratio(person["bbox"], moto["bbox"])
                # Also give bonus if vertical positions are compatible
                # (person bottom near motorcycle bottom)
                p_bot = person["bbox"][3]
                m_top = moto["bbox"][1]
                m_bot = moto["bbox"][3]
                m_h = m_bot - m_top
                if m_top - m_h * 0.2 <= p_bot <= m_bot + m_h * 0.3:
                    score += 0.05  # small bonus for vertical compatibility

                if score > best_score:
                    best_score = score
                    best_idx = m_idx

            if best_idx >= 0 and best_score >= cfg.RIDER_OVERLAP_THRESHOLD:
                groups[best_idx].append(person)

        return groups

    # ───────────────── violation checks ──────────────────
    def check_violations(
        self,
        motorcycle_riders: dict[int, list[dict]],
        motorcycles: list[dict],
        helmet_results: dict[int, tuple[bool, float]],
    ) -> list[dict]:
        """
        Return a list of violation dicts for this frame.

        Parameters
        ----------
        motorcycle_riders : dict[int, list[dict]]
            Output of ``match_riders``.
        motorcycles : list[dict]
            Motorcycle detections (may contain ``track_id``).
        helmet_results : dict[int, tuple[bool, float]]
            ``id(person) → (has_helmet, prob)``.
        """
        violations: list[dict] = []

        for m_idx, riders in motorcycle_riders.items():
            if not riders:
                continue

            moto = motorcycles[m_idx]
            track_id = moto.get("track_id")

            # ---- Triple riding ----
            if len(riders) > cfg.MAX_LEGAL_RIDERS:
                violations.append({
                    "type": "triple_riding",
                    "motorcycle_bbox": moto["bbox"],
                    "rider_count": len(riders),
                    "riders": riders,
                    "track_id": track_id,
                })

            # ---- No helmet ----
            for rider in riders:
                key = id(rider)
                if key in helmet_results:
                    has_helmet, prob = helmet_results[key]
                    if not has_helmet:
                        violations.append({
                            "type": "no_helmet",
                            "motorcycle_bbox": moto["bbox"],
                            "rider_bbox": rider["bbox"],
                            "helmet_prob": prob,
                            "track_id": track_id,
                        })

        return violations

    # ───────── de-duplication across frames ──────────────
    def is_new(self, track_id, violation_type: str) -> bool:
        """Return True the *first* time this (track, type) pair is seen."""
        if track_id is None:
            return True  # can't de-dup without a track id
        key = str(track_id)
        if key not in self._seen:
            self._seen[key] = set()
        if violation_type in self._seen[key]:
            return False
        self._seen[key].add(violation_type)
        return True

    def reset(self):
        """Clear all seen violations (e.g. when processing a new video)."""
        self._seen.clear()
