"""Mouth feature calculations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Point = Sequence[float]


def calculate_mar(points: Iterable[Point]) -> float:
    """Calculate mouth aspect ratio from eight ordered mouth landmark points."""
    landmarks = _validate_points(points, expected_count=8)
    horizontal = _distance(landmarks[0], landmarks[4])
    if horizontal == 0.0:
        raise ValueError("Mouth horizontal distance must be greater than zero.")

    vertical_1 = _distance(landmarks[1], landmarks[7])
    vertical_2 = _distance(landmarks[2], landmarks[6])
    vertical_3 = _distance(landmarks[3], landmarks[5])
    return (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)


def _validate_points(points: Iterable[Point], expected_count: int) -> list[tuple[float, ...]]:
    landmarks = [tuple(float(value) for value in point) for point in points]
    if len(landmarks) != expected_count:
        raise ValueError(f"MAR calculation requires exactly {expected_count} points.")

    for point in landmarks:
        if len(point) < 2:
            raise ValueError("Each point must contain at least x and y coordinates.")
        if not all(math.isfinite(value) for value in point):
            raise ValueError("Point coordinates must be finite numbers.")

    return landmarks


def _distance(point_a: tuple[float, ...], point_b: tuple[float, ...]) -> float:
    return math.dist(point_a, point_b)


__all__ = ["calculate_mar"]
