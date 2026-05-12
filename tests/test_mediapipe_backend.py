from pathlib import Path
from types import SimpleNamespace

import pytest

from driver_drowsiness.inference import mediapipe_face_mesh
from driver_drowsiness.inference.mediapipe_face_mesh import MediaPipeFaceMeshBackend


def test_tasks_api_requires_face_landmarker_model(monkeypatch, tmp_path):
    fake_mediapipe = SimpleNamespace(tasks=SimpleNamespace(vision=SimpleNamespace()))
    monkeypatch.setattr(
        mediapipe_face_mesh,
        "_import_mediapipe",
        lambda: fake_mediapipe,
    )

    missing_model = tmp_path / "face_landmarker.task"

    with pytest.raises(RuntimeError, match="face_landmarker.task"):
        MediaPipeFaceMeshBackend(model_path=missing_model)


def test_tasks_api_detect_returns_landmarks(monkeypatch, tmp_path):
    model_path = tmp_path / "face_landmarker.task"
    model_path.write_bytes(b"fake model")

    class FakeBaseOptions:
        def __init__(self, model_asset_path):
            self.model_asset_path = model_asset_path

    class FakeFaceLandmarkerOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeFaceLandmarker:
        @classmethod
        def create_from_options(cls, options):
            return cls()

        def detect_for_video(self, image, timestamp_ms):
            return SimpleNamespace(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.1, y=0.2, z=0.3),
                        SimpleNamespace(x=0.4, y=0.5, z=0.6),
                    ]
                ]
            )

        def close(self):
            pass

    fake_vision = SimpleNamespace(
        FaceLandmarker=FakeFaceLandmarker,
        FaceLandmarkerOptions=FakeFaceLandmarkerOptions,
        RunningMode=SimpleNamespace(VIDEO="VIDEO"),
    )
    fake_mediapipe = SimpleNamespace(
        Image=lambda image_format, data: data,
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
        tasks=SimpleNamespace(BaseOptions=FakeBaseOptions, vision=fake_vision),
    )
    fake_cv2 = SimpleNamespace(COLOR_BGR2RGB="COLOR_BGR2RGB", cvtColor=lambda frame, _: frame)

    monkeypatch.setattr(mediapipe_face_mesh, "_import_mediapipe", lambda: fake_mediapipe)
    monkeypatch.setattr(mediapipe_face_mesh, "_import_cv2", lambda: fake_cv2)

    backend = MediaPipeFaceMeshBackend(model_path=model_path)

    faces = backend.detect(object())

    assert len(faces) == 1
    assert faces[0].point(0).as_tuple() == pytest.approx((0.1, 0.2, 0.3))
    assert faces[0].point(1).as_tuple() == pytest.approx((0.4, 0.5, 0.6))
