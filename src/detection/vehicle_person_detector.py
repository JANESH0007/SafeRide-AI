"""
Vehicle & person detector — YOLOv8 pretrained on COCO.

Detects:
  • class 0  → person
  • class 3  → motorcycle
"""

from __future__ import annotations

from ultralytics import YOLO

import config.settings as cfg


class VehiclePersonDetector:
    """Wraps a YOLOv8-nano model for person + motorcycle detection."""

    def __init__(self, model_path: str | None = None):
        path = model_path or cfg.YOLO_MODEL
        self.model = YOLO(path)
        self._target_classes = [cfg.PERSON_CLASS, cfg.MOTORCYCLE_CLASS]

    # ──────────────────────────────────────────────────────
    def detect(self, frame):
        """
        Run inference on a single BGR frame.

        Returns
        -------
        persons : list[dict]
            Each dict has keys  bbox (x1,y1,x2,y2), confidence, class_id.
        motorcycles : list[dict]
            Same schema.
        """
        results = self.model.predict(
            frame,
            conf=cfg.YOLO_CONFIDENCE,
            iou=cfg.YOLO_IOU,
            classes=self._target_classes,
            verbose=False,
        )

        persons: list[dict] = []
        motorcycles: list[dict] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                det = {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id": cls_id,
                }

                if cls_id == cfg.PERSON_CLASS:
                    persons.append(det)
                elif cls_id == cfg.MOTORCYCLE_CLASS:
                    motorcycles.append(det)

        return persons, motorcycles
