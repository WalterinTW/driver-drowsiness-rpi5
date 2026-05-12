import pytest

from driver_drowsiness.inference.hailo_landmark import HailoLandmarkBackend


def test_hailo_landmark_backend_matches_detect_interface():
    backend = HailoLandmarkBackend(model_path="models/face_landmark.hef")

    with pytest.raises(NotImplementedError, match="Hailo"):
        backend.detect(object())


def test_hailo_landmark_backend_supports_context_manager():
    backend = HailoLandmarkBackend()

    with backend as active_backend:
        assert active_backend is backend
