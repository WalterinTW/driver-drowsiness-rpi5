"""OpenCV webcam camera backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OpenCVCamera:
    """Small wrapper around OpenCV VideoCapture."""

    camera_index: int = 0
    width: int | None = None
    height: int | None = None
    _capture: Any | None = None

    def open(self) -> None:
        """Open the webcam."""
        cv2 = _import_cv2()
        self._capture = cv2.VideoCapture(self.camera_index)
        if self.width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._capture.isOpened():
            self.release()
            raise RuntimeError(f"Could not open webcam at index {self.camera_index}.")

    def read(self) -> Any:
        """Read one frame from the webcam."""
        if self._capture is None:
            raise RuntimeError("Camera is not open.")
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("Could not read frame from webcam.")
        return frame

    def release(self) -> None:
        """Release the webcam if it is open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> OpenCVCamera:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for webcam mode. Install requirements-dev.txt."
        ) from exc
    return cv2


__all__ = ["OpenCVCamera"]
