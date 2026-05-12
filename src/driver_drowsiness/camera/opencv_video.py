"""OpenCV video file frame source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from driver_drowsiness.camera.opencv_camera import _import_cv2


@dataclass
class OpenCVVideoSource:
    """Small wrapper around OpenCV VideoCapture for video files."""

    video_path: Path | str
    _capture: Any | None = None

    def open(self) -> None:
        """Open the video file."""
        path = Path(self.video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        cv2 = _import_cv2()
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            self.release()
            raise RuntimeError(f"Could not open video file: {path}")

    def read(self) -> Any:
        """Read one frame from the video file."""
        if self._capture is None:
            raise RuntimeError("Video source is not open.")
        ok, frame = self._capture.read()
        if not ok:
            raise StopIteration("End of video file.")
        return frame

    def release(self) -> None:
        """Release the video source if it is open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> OpenCVVideoSource:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


__all__ = ["OpenCVVideoSource"]
