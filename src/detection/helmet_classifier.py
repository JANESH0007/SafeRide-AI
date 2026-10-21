"""
Helmet classifier — face-detection proxy (zero training).

Strategy
--------
Instead of classifying head crops with ImageNet (unreliable on small
surveillance crops), we use **face visibility as a proxy**:

  • Face clearly detected  → rider likely does NOT have a full helmet
  • Face NOT detected      → helmet is likely covering the face

Uses OpenCV 5's ``FaceDetectorYN`` (YuNet DNN model, ~230 KB ONNX).
Runs in < 5 ms per crop on CPU.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

import config.settings as cfg

# Path to the YuNet ONNX model (downloaded at setup time)
_YUNET_MODEL = os.path.join(cfg.MODELS_DIR, "face_detection_yunet.onnx")


class HelmetClassifier:
    """Head-crop → YuNet face detection → helmet / no-helmet."""

    def __init__(self):
        if not os.path.isfile(_YUNET_MODEL):
            raise FileNotFoundError(
                f"YuNet face-detection model not found at {_YUNET_MODEL}. "
                "Download it with:\n"
                "  curl -L https://github.com/opencv/opencv_zoo/raw/main/"
                "models/face_detection_yunet/face_detection_yunet_2023mar.onnx "
                f"-o {_YUNET_MODEL}"
            )

        # Create the face detector — input size will be set per-crop
        self._detector = cv2.FaceDetectorYN.create(
            model=_YUNET_MODEL,
            config="",
            input_size=(120, 120),       # default, overridden per crop
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5,
        )

    # ──────────────────────────────────────────────────────
    def classify(
        self, frame: np.ndarray, person_bbox: tuple
    ) -> tuple[bool, float]:
        """
        Classify whether the rider in *person_bbox* is wearing a helmet.

        Returns
        -------
        has_helmet : bool
        confidence : float   (0-1, higher = more certain)
        """
        x1, y1, x2, y2 = person_bbox
        h = y2 - y1

        # Crop head region (top portion of person bbox)
        head_h = int(h * cfg.HEAD_CROP_RATIO)
        cy1 = max(0, y1)
        cy2 = max(0, y1 + head_h)
        cx1 = max(0, x1)
        cx2 = max(0, x2)
        head_crop = frame[cy1:cy2, cx1:cx2]

        if (
            head_crop.size == 0
            or head_crop.shape[0] < 10
            or head_crop.shape[1] < 10
        ):
            # Crop too small to analyse — conservatively assume helmet
            return True, 0.5

        crop_h, crop_w = head_crop.shape[:2]
        crop_area = crop_h * crop_w

        # Set input size to match the crop
        self._detector.setInputSize((crop_w, crop_h))

        # Detect faces in the head crop
        _, faces = self._detector.detect(head_crop)

        if faces is not None and len(faces) > 0:
            # faces is an Nx15 array; columns 0-3 are x,y,w,h; col 14 is score
            for face in faces:
                fw, fh = face[2], face[3]
                score = face[14] if face.shape[0] > 14 else face[-1]
                face_area = fw * fh
                face_ratio = face_area / crop_area if crop_area > 0 else 0

                if face_ratio > 0.03 and score > 0.5:
                    # Significant face visible → likely NO helmet
                    return False, round(float(min(score, 1.0)), 4)

        # No face detected → assume helmet is covering the face
        return True, 0.8
