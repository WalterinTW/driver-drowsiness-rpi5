"""Application entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(prog="driver-drowsiness")
    parser.add_argument(
        "--demo",
        choices=("webcam",),
        help="Run a development demo mode.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index for webcam demo mode.",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        help="Path to MediaPipe face_landmarker.task for MediaPipe Tasks API.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected app mode."""
    args = build_parser().parse_args(argv)
    if args.demo == "webcam":
        from driver_drowsiness.demo.webcam import run_webcam_demo

        run_webcam_demo(camera_index=args.camera_index, face_model=args.face_model)
        return

    print("Driver drowsiness detection app starting...")


if __name__ == "__main__":
    main()
