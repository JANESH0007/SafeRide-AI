"""
SafeRide AI — centralised configuration.

All tuneable knobs live here so nothing is hard-coded in the modules.
"""

import os

# ───────────────────────── paths ─────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
VIOLATIONS_CSV = os.path.join(OUTPUT_DIR, "violations.csv")

# ───────────────────── YOLO detection ────────────────────
YOLO_MODEL = "yolov8n.pt"                # auto-downloaded by ultralytics
YOLO_CONFIDENCE = 0.40
YOLO_IOU = 0.50

# COCO class IDs we care about
PERSON_CLASS = 0
MOTORCYCLE_CLASS = 3
BICYCLE_CLASS = 1

# ──────────────── rider ↔ motorcycle matching ────────────
RIDER_OVERLAP_THRESHOLD = 0.10           # min IoU to pair a person to a bike
MAX_LEGAL_RIDERS = 2                     # >2 = triple riding

# ─────────────── helmet classification ───────────────────
# Uses MobileNetV2 pretrained on ImageNet.
# ImageNet index 518 = "crash helmet"
HELMET_IMAGENET_INDICES = [518]          # class indices treated as "helmet"
HELMET_CONFIDENCE = 0.15                 # probability above this → has helmet
HEAD_CROP_RATIO = 0.35                   # top 35 % of person bbox = head region

# ──────────────── plate reading (EasyOCR) ────────────────
PLATE_LANGUAGES = ["en"]
PLATE_MIN_CHARS = 4                      # ignore OCR results shorter than this
# Indian plate regex (loose): 2 letters, 1-2 digits, 1-2 letters, 4 digits
PLATE_REGEX = r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}"

# ───────────────── DeepSORT tracking ─────────────────────
TRACK_MAX_AGE = 30                       # frames to keep a lost track
TRACK_N_INIT = 3                         # detections to confirm a track
TRACK_MAX_COSINE_DIST = 0.3
TRACK_NN_BUDGET = 100

# ──────────────── visualisation colors (BGR) ─────────────
COLOR_MOTORCYCLE = (255, 180, 0)         # cyan-blue
COLOR_HELMET_OK = (0, 220, 0)            # green
COLOR_NO_HELMET = (0, 0, 255)            # red
COLOR_TRIPLE = (0, 165, 255)             # orange
COLOR_PLATE = (255, 255, 0)              # yellow
COLOR_TRACK_ID = (255, 255, 255)         # white

# ──────────────── dashboard ──────────────────────────────
DASHBOARD_PAGE_TITLE = "SafeRide AI"
DASHBOARD_ICON = "🏍️"

# ──────────────── create dirs on import ──────────────────
for _d in (OUTPUT_DIR, CROPS_DIR, MODELS_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)
