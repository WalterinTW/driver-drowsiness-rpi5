"""macOS webcam + MediaPipe development demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from driver_drowsiness.camera.opencv_camera import OpenCVCamera
from driver_drowsiness.camera.opencv_video import OpenCVVideoSource
from driver_drowsiness.camera.rpi_camera import RaspberryPiCamera
from driver_drowsiness.features.face_metrics import calculate_face_metrics
from driver_drowsiness.inference.hailo_landmark import HailoLandmarkBackend
from driver_drowsiness.inference.mediapipe_face_mesh import MediaPipeFaceMeshBackend
from driver_drowsiness.inference.mock_landmark import MockLandmarkBackend
from driver_drowsiness.utils.drawing import draw_demo_overlay, draw_metric_landmarks


def run_webcam_demo(
    camera_index: int = 0,
    face_model: Path | None = None,
    *,
    source: str = "webcam",
    backend: str = "mediapipe",
    video_path: Path | None = None,
    hailo_model: Path | None = None,
) -> None:
    """Run a simple real-time webcam demo until q is pressed."""
    import cv2  # type: ignore[import-not-found]

    frame_source = create_frame_source(
        source=source,
        camera_index=camera_index,
        video_path=video_path,
    )
    landmark_backend = create_landmark_backend(
        backend=backend,
        face_model=face_model,
        hailo_model=hailo_model,
    )

    with frame_source as camera, landmark_backend as detector:
        while True:
            try:
                frame = camera.read()
            except StopIteration:
                break

            faces = detector.detect(frame)

            ear = None
            mar = None
            if faces:
                metrics = calculate_face_metrics(faces[0])
                ear = metrics.ear
                mar = metrics.mar
                draw_metric_landmarks(frame, faces[0])

            draw_demo_overlay(frame, ear=ear, mar=mar, face_detected=bool(faces))
            cv2.imshow("Driver Drowsiness Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


def create_frame_source(
    *,
    source: str,
    camera_index: int,
    video_path: Path | None,
) -> Any:
    """Create a frame source for the demo."""
    if source == "webcam":
        return OpenCVCamera(camera_index=camera_index)
    if source == "video":
        if video_path is None:
            raise ValueError("--video-path is required when --source video is used.")
        return OpenCVVideoSource(video_path=video_path)
    if source == "rpi-camera":
        return RaspberryPiCamera()
    raise ValueError(f"Unsupported source: {source}")


def create_landmark_backend(
    *,
    backend: str,
    face_model: Path | None,
    hailo_model: Path | None,
) -> Any:
    """Create a landmark backend for the demo."""
    if backend == "mediapipe":
        return MediaPipeFaceMeshBackend(model_path=face_model)
    if backend == "mock":
        return MockLandmarkBackend()
    if backend == "hailo":
        return HailoLandmarkBackend(model_path=hailo_model)
    raise ValueError(f"Unsupported backend: {backend}")


__all__ = ["create_frame_source", "create_landmark_backend", "run_webcam_demo"]
