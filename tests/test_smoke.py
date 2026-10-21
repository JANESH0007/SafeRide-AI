#!/usr/bin/env python3
"""
Quick smoke test — validates that all SafeRide AI modules import
and initialise correctly.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

def test_imports():
    print("🔍 Testing imports …")
    
    print("  ➤ config … ", end="")
    import config.settings as cfg
    assert os.path.isdir(cfg.OUTPUT_DIR)
    print("✓")

    print("  ➤ VehiclePersonDetector … ", end="")
    from src.detection.vehicle_person_detector import VehiclePersonDetector
    print("✓")

    print("  ➤ HelmetClassifier … ", end="")
    from src.detection.helmet_classifier import HelmetClassifier
    print("✓")

    print("  ➤ PlateReader … ", end="")
    from src.detection.plate_reader import PlateReader
    print("✓")

    print("  ➤ VehicleTracker … ", end="")
    from src.tracking.vehicle_tracker import VehicleTracker
    print("✓")

    print("  ➤ ViolationEngine … ", end="")
    from src.utils.violation_engine import ViolationEngine
    print("✓")

    print("  ➤ Annotator … ", end="")
    from src.utils.drawing import Annotator
    print("✓")

    print("  ➤ IOHelper … ", end="")
    from src.utils.io_helpers import IOHelper
    print("✓")

    print("  ➤ SafeRidePipeline … ", end="")
    from src.pipeline.processor import SafeRidePipeline
    print("✓")

    print("\n✅ All imports passed!")


def test_instantiation():
    print("\n🔍 Testing model instantiation …")

    import numpy as np
    from src.detection.vehicle_person_detector import VehiclePersonDetector
    from src.detection.helmet_classifier import HelmetClassifier
    from src.tracking.vehicle_tracker import VehicleTracker
    from src.utils.violation_engine import ViolationEngine

    print("  ➤ Loading YOLOv8 … ", end="", flush=True)
    detector = VehiclePersonDetector()
    print("✓")

    print("  ➤ Loading MobileNetV2 … ", end="", flush=True)
    helmet = HelmetClassifier()
    print("✓")

    print("  ➤ Initialising DeepSORT … ", end="", flush=True)
    tracker = VehicleTracker()
    print("✓")

    # Quick inference on a dummy frame
    print("  ➤ Running dummy inference … ", end="", flush=True)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    persons, motos = detector.detect(dummy)
    assert isinstance(persons, list)
    assert isinstance(motos, list)
    print(f"✓  (detected {len(persons)} persons, {len(motos)} motorcycles)")

    print("  ➤ Testing helmet classifier … ", end="", flush=True)
    has_helmet, prob = helmet.classify(dummy, (100, 100, 200, 300))
    print(f"✓  (has_helmet={has_helmet}, prob={prob})")

    # Test violation engine
    print("  ➤ Testing violation engine … ", end="", flush=True)
    engine = ViolationEngine()
    groups = engine.match_riders([], [])
    assert isinstance(groups, dict)
    assert engine.is_new("track_1", "no_helmet") is True
    assert engine.is_new("track_1", "no_helmet") is False  # duplicate
    print("✓")

    print("\n✅ All instantiation tests passed!")


if __name__ == "__main__":
    test_imports()
    test_instantiation()
    print("\n🎉 SafeRide AI is ready to go!")
