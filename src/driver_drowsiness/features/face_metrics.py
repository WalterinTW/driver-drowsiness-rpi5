"""Face metric extraction from normalized landmarks."""

from __future__ import annotations

from dataclasses import dataclass

from driver_drowsiness.features.eye import calculate_ear
from driver_drowsiness.features.mouth import calculate_mar
from driver_drowsiness.inference.types import FaceLandmarks

LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
MOUTH_INDICES = (61, 81, 13, 311, 291, 402, 14, 178)


@dataclass(frozen=True)
class FaceMetrics:
    """EAR and MAR metrics derived from one detected face."""

    ear: float
    mar: float


def calculate_face_metrics(landmarks: FaceLandmarks) -> FaceMetrics:
    """Calculate EAR and MAR from MediaPipe Face Mesh landmark indices."""
    left_ear = calculate_ear(_points_for_indices(landmarks, LEFT_EYE_INDICES))
    right_ear = calculate_ear(_points_for_indices(landmarks, RIGHT_EYE_INDICES))
    mar = calculate_mar(_points_for_indices(landmarks, MOUTH_INDICES))
    return FaceMetrics(ear=(left_ear + right_ear) / 2.0, mar=mar)


def _points_for_indices(
    landmarks: FaceLandmarks,
    indices: tuple[int, ...],
) -> list[tuple[float, float, float]]:
    return [landmarks.point(index).as_tuple() for index in indices]


__all__ = [
    "FaceMetrics",
    "LEFT_EYE_INDICES",
    "MOUTH_INDICES",
    "RIGHT_EYE_INDICES",
    "calculate_face_metrics",
]
