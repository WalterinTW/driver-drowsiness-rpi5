"""OpenCV drawing helpers for development demos."""

from __future__ import annotations

from typing import Any

from driver_drowsiness.features.face_metrics import (
    LEFT_EYE_INDICES,
    MOUTH_INDICES,
    RIGHT_EYE_INDICES,
)
from driver_drowsiness.inference.types import FaceLandmarks


def draw_demo_overlay(
    frame: Any,
    *,
    ear: float | None,
    mar: float | None,
    face_detected: bool,
) -> Any:
    """Draw EAR/MAR status text on an OpenCV frame."""
    cv2 = _import_cv2()
    status = "Face detected" if face_detected else "No face detected"
    ear_text = f"EAR: {ear:.3f}" if ear is not None else "EAR: --"
    mar_text = f"MAR: {mar:.3f}" if mar is not None else "MAR: --"

    _put_text(cv2, frame, status, (20, 30), (0, 255, 0) if face_detected else (0, 0, 255))
    _put_text(cv2, frame, ear_text, (20, 60), (255, 255, 255))
    _put_text(cv2, frame, mar_text, (20, 90), (255, 255, 255))
    _put_text(cv2, frame, "Press q to quit", (20, 120), (200, 200, 200))
    return frame


def draw_metric_landmarks(frame: Any, landmarks: FaceLandmarks) -> Any:
    """Draw only the landmarks used by EAR and MAR."""
    cv2 = _import_cv2()
    height, width = frame.shape[:2]
    for index in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
        _draw_point(cv2, frame, landmarks, index, width, height, (0, 255, 255))
    for index in MOUTH_INDICES:
        _draw_point(cv2, frame, landmarks, index, width, height, (255, 0, 255))
    return frame


def _put_text(
    cv2: Any,
    frame: Any,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _draw_point(
    cv2: Any,
    frame: Any,
    landmarks: FaceLandmarks,
    index: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    point = landmarks.point(index)
    cv2.circle(frame, (int(point.x * width), int(point.y * height)), 2, color, -1)


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for drawing demo overlays. Install requirements-dev.txt."
        ) from exc
    return cv2


__all__ = ["draw_demo_overlay", "draw_metric_landmarks"]
