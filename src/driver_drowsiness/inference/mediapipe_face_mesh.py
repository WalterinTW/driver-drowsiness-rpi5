"""MediaPipe Face Mesh landmark backend for development."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from driver_drowsiness.inference.types import FaceLandmarks, LandmarkPoint

DEFAULT_FACE_LANDMARKER_MODEL = Path("models") / "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass
class MediaPipeFaceMeshBackend:
    """Detect face landmarks with MediaPipe Face Mesh or Face Landmarker."""

    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: Path | str | None = None
    _face_mesh: Any | None = None
    _landmarker: Any | None = None
    _mediapipe: Any | None = None
    _timestamp_ms: int = 0

    def __post_init__(self) -> None:
        mediapipe = _import_mediapipe()
        self._mediapipe = mediapipe
        if _has_legacy_face_mesh(mediapipe):
            self._init_legacy_face_mesh(mediapipe)
            return

        self._init_tasks_face_landmarker(mediapipe)

    def _init_legacy_face_mesh(self, mediapipe: Any) -> None:
        self._face_mesh = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def _init_tasks_face_landmarker(self, mediapipe: Any) -> None:
        model_path = Path(self.model_path) if self.model_path is not None else DEFAULT_FACE_LANDMARKER_MODEL
        if not model_path.exists():
            raise RuntimeError(
                "MediaPipe 0.10.35 uses the Tasks Face Landmarker API and needs "
                f"a local model file. Download {FACE_LANDMARKER_MODEL_URL} to "
                f"{model_path}, or pass --face-model PATH."
            )

        base_options = mediapipe.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mediapipe.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
            num_faces=self.max_num_faces,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = mediapipe.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: Any) -> list[FaceLandmarks]:
        """Return face landmarks detected in a BGR OpenCV frame."""
        if self._face_mesh is None and self._landmarker is None:
            raise RuntimeError("MediaPipe Face Mesh backend is closed.")

        cv2 = _import_cv2()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._face_mesh is not None:
            return self._detect_with_legacy_face_mesh(frame_rgb)

        return self._detect_with_tasks_face_landmarker(frame_rgb)

    def _detect_with_legacy_face_mesh(self, frame_rgb: Any) -> list[FaceLandmarks]:
        if self._face_mesh is None:
            raise RuntimeError("Legacy MediaPipe Face Mesh backend is closed.")

        result = self._face_mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return []

        faces: list[FaceLandmarks] = []
        for face in result.multi_face_landmarks:
            faces.append(
                FaceLandmarks(
                    points={
                        index: LandmarkPoint(
                            x=landmark.x,
                            y=landmark.y,
                            z=landmark.z,
                        )
                        for index, landmark in enumerate(face.landmark)
                    }
                )
            )
        return faces

    def _detect_with_tasks_face_landmarker(self, frame_rgb: Any) -> list[FaceLandmarks]:
        if self._landmarker is None or self._mediapipe is None:
            raise RuntimeError("MediaPipe Face Landmarker backend is closed.")

        image = self._mediapipe.Image(
            image_format=self._mediapipe.ImageFormat.SRGB,
            data=frame_rgb,
        )
        self._timestamp_ms += 1
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        faces: list[FaceLandmarks] = []
        for face in result.face_landmarks:
            faces.append(
                FaceLandmarks(
                    points={
                        index: LandmarkPoint(
                            x=landmark.x,
                            y=landmark.y,
                            z=landmark.z,
                        )
                        for index, landmark in enumerate(face)
                    }
                )
            )
        return faces

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> MediaPipeFaceMeshBackend:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for MediaPipe webcam mode. Install requirements-dev.txt."
        ) from exc
    return cv2


def _import_mediapipe() -> Any:
    try:
        import mediapipe  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required for webcam mode. Install requirements-dev.txt."
        ) from exc
    return mediapipe


def _has_legacy_face_mesh(mediapipe: Any) -> bool:
    solutions = getattr(mediapipe, "solutions", None)
    return solutions is not None and hasattr(solutions, "face_mesh")


__all__ = [
    "DEFAULT_FACE_LANDMARKER_MODEL",
    "FACE_LANDMARKER_MODEL_URL",
    "MediaPipeFaceMeshBackend",
]
