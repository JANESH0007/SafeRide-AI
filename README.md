# SafeRide AI 🏍️

**Automated detection of helmetless two-wheeler riders and triple riding from traffic camera footage, with number plate extraction to generate a violation report.**

---

## Architecture

```
Traffic Video
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  YOLOv8n    │────▶│  DeepSORT    │────▶│  Rule Logic │
│  (COCO)     │     │  Tracker     │     │  Engine     │
│  person +   │     │  stable IDs  │     │  helmet?    │
│  motorcycle │     │  per vehicle │     │  triple?    │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                         ┌──────────────────────┤
                         ▼                      ▼
                  ┌─────────────┐     ┌──────────────┐
                  │ MobileNetV2 │     │   EasyOCR    │
                  │ (ImageNet)  │     │  plate text  │
                  │ helmet      │     │  extraction  │
                  │ classifier  │     └──────────────┘
                  └─────────────┘              │
                                               ▼
                                     ┌──────────────┐
                                     │  CSV log +   │
                                     │  crop images │
                                     │  + Streamlit │
                                     │  dashboard   │
                                     └──────────────┘
```

## Pipeline Steps

| Step | Component | Model / Method | What it does |
|------|-----------|----------------|-------------|
| 1 | **Detection** | YOLOv8-nano (COCO pretrained) | Detects persons (class 0) and motorcycles (class 3) in each frame |
| 2 | **Tracking** | DeepSORT | Assigns stable IDs to motorcycles across frames so each is counted once |
| 3 | **Rider Matching** | IoU + overlap ratio | Matches persons to motorcycles by bounding-box overlap |
| 4 | **Helmet Check** | MobileNetV2 (ImageNet class 518: "crash helmet") | Crops the top 35% of person bbox (head region) and classifies helmet presence |
| 5 | **Triple Riding** | Rule logic | Flags motorcycles with > 2 matched riders |
| 6 | **Plate OCR** | EasyOCR | Reads number plates from the lower region of the motorcycle bbox |
| 7 | **Output** | CSV + Streamlit | Logs violations with timestamps, crops, plate text; displays in dashboard |

## Quick Start

### 1. Install dependencies

```bash
cd "SafeRide AI"
pip install -r requirements.txt
```

### 2. Process a video (CLI)

```bash
# With live display window (press 'q' to quit)
python main.py --input path/to/traffic_video.mp4 --display

# Save annotated output video
python main.py --input path/to/traffic_video.mp4 --output output/result.mp4

# Webcam
python main.py --input 0 --display

# Fast mode: skip frames, disable OCR
python main.py --input video.mp4 --skip 2 --no-ocr --display
```

### 3. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
SafeRide AI/
├── main.py                          # CLI entry point
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py                  # All tuneable parameters
├── src/
│   ├── detection/
│   │   ├── vehicle_person_detector.py   # YOLOv8 COCO detector
│   │   ├── helmet_classifier.py         # MobileNetV2 helmet classifier
│   │   └── plate_reader.py             # EasyOCR plate reader
│   ├── tracking/
│   │   └── vehicle_tracker.py          # DeepSORT wrapper
│   ├── pipeline/
│   │   └── processor.py               # Main processing pipeline
│   ├── plate_ocr/
│   │   └── plate_reader.py            # PlateReader re-export
│   └── utils/
│       ├── violation_engine.py         # Rule logic + de-duplication
│       ├── drawing.py                  # Frame annotation
│       └── io_helpers.py              # CSV, crops, video writer
├── dashboard/
│   └── app.py                         # Streamlit dashboard
├── output/
│   ├── violations.csv                 # Violation log
│   └── crops/                         # Cropped violation images
├── models/                            # Auto-downloaded model weights
└── data/                              # Place test videos here
```

## Models Used (All Pretrained — No Training Required)

| Model | Size | Purpose | Source |
|-------|------|---------|--------|
| YOLOv8-nano | ~6 MB | Person + motorcycle detection | Ultralytics (COCO) |
| MobileNetV2 | ~14 MB | Helmet classification | torchvision (ImageNet) |
| DeepSORT | ~5 MB | Multi-object tracking | deep-sort-realtime |
| EasyOCR | ~100 MB | Number plate text recognition | easyocr (pretrained) |

All models download automatically on first run.

## Configuration

Edit [`config/settings.py`](config/settings.py) to tune:

- **Detection thresholds**: `YOLO_CONFIDENCE`, `YOLO_IOU`
- **Rider matching**: `RIDER_OVERLAP_THRESHOLD`, `MAX_LEGAL_RIDERS`
- **Helmet sensitivity**: `HELMET_CONFIDENCE`, `HEAD_CROP_RATIO`
- **Tracking**: `TRACK_MAX_AGE`, `TRACK_N_INIT`
- **Plate OCR**: `PLATE_REGEX`, `PLATE_MIN_CHARS`

## Output Format

### violations.csv

| Column | Description |
|--------|-------------|
| violation_id | Unique ID (V0001, V0002, …) |
| timestamp | Time in video (HH:MM:SS) |
| frame_idx | Frame number |
| violation_type | `no_helmet` or `triple_riding` |
| rider_count | Number of riders on the motorcycle |
| helmet_prob | MobileNetV2 helmet probability |
| plate_text | Extracted plate number (if readable) |
| crop_path | Path to cropped violation image |
| track_id | DeepSORT track ID |

## Evaluation Metrics

- **Detection**: mAP@0.5, precision/recall for person + motorcycle classes (via YOLOv8 built-in metrics)
- **Helmet classification**: Accuracy on head-crop test set
- **Plate OCR**: Character-level and plate-level accuracy
- **Overall**: Violation detection rate on test videos + FPS

## Limitations & Future Work

- **Helmet detection**: Uses ImageNet's "crash helmet" class — works for many helmets but may miss unusual styles. Fine-tuning a dedicated helmet detector (e.g., on Roboflow datasets) would improve accuracy.
- **Plate OCR**: Best results on clear, well-lit, front/rear-facing plates. Indian plates with non-standard fonts may need a dedicated CRNN model.
- **Angle sensitivity**: Overhead cameras work best; steep angles reduce person-motorcycle overlap matching.

## Tech Stack

`YOLOv8` · `DeepSORT` · `MobileNetV2` · `EasyOCR` · `OpenCV` · `PyTorch` · `Streamlit` · `Plotly`
