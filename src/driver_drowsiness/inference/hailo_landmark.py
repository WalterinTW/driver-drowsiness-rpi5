"""Hailo landmark backend skeleton for Raspberry Pi 5 deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from driver_drowsiness.inference.types import FaceLandmarks


@dataclass
class HailoLandmarkBackend:
    """Skeleton backend matching the MediaPipe landmark detect interface."""

    model_path: Path | str | None = None
    device_name: str | None = None

    def detect(self, frame_bgr: Any) -> list[FaceLandmarks]:
        """Return face landmarks detected in a BGR frame."""
        raise NotImplementedError(
            "Hailo landmark inference is not implemented yet. TODO: adapt the "
            "working Hailo object detection example by replacing its model "
            "load, preprocessing, inference call, and output decoding with "
            "face-landmark equivalents that return FaceLandmarks."
        )

    def close(self) -> None:
        """Release Hailo resources."""
        return None

    def __enter__(self) -> HailoLandmarkBackend:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["HailoLandmarkBackend"]
