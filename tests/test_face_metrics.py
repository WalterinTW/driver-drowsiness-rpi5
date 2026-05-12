import pytest

from driver_drowsiness.features.face_metrics import calculate_face_metrics
from driver_drowsiness.inference.types import FaceLandmarks, LandmarkPoint


def test_calculate_face_metrics_uses_mediapipe_eye_and_mouth_indices():
    landmarks = FaceLandmarks(
        points={
            33: LandmarkPoint(0.0, 0.0),
            160: LandmarkPoint(1.0, 1.0),
            158: LandmarkPoint(3.0, 1.0),
            133: LandmarkPoint(4.0, 0.0),
            153: LandmarkPoint(3.0, -1.0),
            144: LandmarkPoint(1.0, -1.0),
            362: LandmarkPoint(10.0, 0.0),
            385: LandmarkPoint(11.0, 1.0),
            387: LandmarkPoint(13.0, 1.0),
            263: LandmarkPoint(14.0, 0.0),
            373: LandmarkPoint(13.0, -1.0),
            380: LandmarkPoint(11.0, -1.0),
            61: LandmarkPoint(0.0, 10.0),
            81: LandmarkPoint(1.0, 11.0),
            13: LandmarkPoint(3.0, 12.0),
            311: LandmarkPoint(5.0, 11.0),
            291: LandmarkPoint(6.0, 10.0),
            402: LandmarkPoint(5.0, 9.0),
            14: LandmarkPoint(3.0, 8.0),
            178: LandmarkPoint(1.0, 9.0),
        }
    )

    metrics = calculate_face_metrics(landmarks)

    assert metrics.ear == pytest.approx(0.5)
    assert metrics.mar == pytest.approx(2.0 / 3.0)


def test_calculate_face_metrics_rejects_missing_required_landmarks():
    landmarks = FaceLandmarks(points={33: LandmarkPoint(0.0, 0.0)})

    with pytest.raises(ValueError, match="landmark"):
        calculate_face_metrics(landmarks)
