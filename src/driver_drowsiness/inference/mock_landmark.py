"""Mock landmark backend for development and smoke tests."""

from __future__ import annotations

from typing import Any

from driver_drowsiness.inference.types import FaceLandmarks


class MockLandmarkBackend:
    """Landmark backend that returns no faces."""

    def detect(self, frame_bgr: Any) -> list[FaceLandmarks]:
        """Return an empty face list for every frame."""
        return []

    def close(self) -> None:
        """Release mock resources."""
        return None

    def __enter__(self) -> MockLandmarkBackend:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["MockLandmarkBackend"]
