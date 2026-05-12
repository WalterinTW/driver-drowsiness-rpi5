"""macOS webcam + MediaPipe development demo."""

from __future__ import annotations

from pathlib import Path

from driver_drowsiness.camera.opencv_camera import OpenCVCamera
from driver_drowsiness.features.face_metrics import calculate_face_metrics
from driver_drowsiness.inference.mediapipe_face_mesh import MediaPipeFaceMeshBackend
from driver_drowsiness.utils.drawing import draw_demo_overlay, draw_metric_landmarks


def run_webcam_demo(camera_index: int = 0, face_model: Path | None = None) -> None:
    """Run a simple real-time webcam demo until q is pressed."""
    import cv2  # type: ignore[import-not-found]

    with OpenCVCamera(camera_index=camera_index) as camera, MediaPipeFaceMeshBackend(
        model_path=face_model
    ) as backend:
        while True:
            frame = camera.read()
            faces = backend.detect(frame)

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


__all__ = ["run_webcam_demo"]
