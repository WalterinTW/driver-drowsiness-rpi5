"""Shared inference data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class LandmarkPoint:
    """Normalized face landmark point."""

    x: float
    y: float
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        """Return the point as an x, y, z tuple."""
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class FaceLandmarks:
    """Face landmarks keyed by backend-specific landmark index."""

    points: Mapping[int, LandmarkPoint]

    def point(self, index: int) -> LandmarkPoint:
        """Return a landmark point by index."""
        try:
            return self.points[index]
        except KeyError as exc:
            raise ValueError(f"Missing required face landmark: {index}") from exc


__all__ = ["FaceLandmarks", "LandmarkPoint"]
