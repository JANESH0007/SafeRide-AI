#!/usr/bin/env python3
"""
SafeRide AI — CLI entry point.

Usage
-----
  # Process a video file (with display window)
  python main.py --input path/to/video.mp4 --display

  # Process and save annotated output
  python main.py --input path/to/video.mp4 --output output/annotated.mp4

  # Webcam (live)
  python main.py --input 0 --display

  # Fast mode: skip every other frame, disable OCR
  python main.py --input video.mp4 --skip 1 --no-ocr --display
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="SafeRide AI — Automated helmetless rider & triple-riding detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video file, or '0' for webcam.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save annotated output video (e.g. output/result.mp4).",
    )
    parser.add_argument(
        "--display", "-d",
        action="store_true",
        help="Show live OpenCV window (press 'q' to quit).",
    )
    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Process at most N frames (for testing).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Process every N-th frame (0 = all frames).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable number-plate OCR (faster processing).",
    )

    args = parser.parse_args()

    from src.pipeline.processor import SafeRidePipeline

    pipeline = SafeRidePipeline(enable_ocr=not args.no_ocr)

    stats = pipeline.process_video(
        input_path=args.input,
        output_path=args.output,
        display=args.display,
        max_frames=args.max_frames,
        skip_frames=args.skip,
    )

    # ── Summary ──
    print("\n" + "═" * 50)
    print("  SafeRide AI — Processing Complete")
    print("═" * 50)
    print(f"  Frames processed : {stats['frames_processed']}")
    print(f"  Elapsed time     : {stats['elapsed_seconds']} s")
    print(f"  Average FPS      : {stats['avg_fps']}")
    print(f"  ─────────────────────────────────")
    print(f"  Total violations : {stats['total_violations']}")
    print(f"    No helmet      : {stats.get('no_helmet', 0)}")
    print(f"    Triple riding  : {stats.get('triple_riding', 0)}")
    print(f"  Plates read      : {stats['plates_read']}")
    print(f"  ─────────────────────────────────")
    print(f"  CSV log          : {os.path.abspath('output/violations.csv')}")
    print(f"  Image crops      : {os.path.abspath('output/crops/')}")
    if args.output:
        print(f"  Output video     : {os.path.abspath(args.output)}")
    print("═" * 50)


if __name__ == "__main__":
    main()
