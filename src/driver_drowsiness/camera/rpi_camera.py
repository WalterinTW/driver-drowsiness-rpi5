"""Raspberry Pi camera frame source placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RaspberryPiCamera:
    """Placeholder for the Raspberry Pi camera backend."""

    width: int = 1280
    height: int = 720
    framerate: int = 30

    def open(self) -> None:
        """Open the Raspberry Pi camera."""
        raise NotImplementedError(
            "Raspberry Pi camera support is a placeholder. TODO: adapt this "
            "to the camera stack used on the Raspberry Pi 5 deployment image."
        )

    def read(self) -> Any:
        """Read one frame from the Raspberry Pi camera."""
        raise NotImplementedError(
            "Raspberry Pi camera support must be tested on Raspberry Pi 5."
        )

    def release(self) -> None:
        """Release Raspberry Pi camera resources."""
        return None

    def __enter__(self) -> RaspberryPiCamera:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


__all__ = ["RaspberryPiCamera"]
