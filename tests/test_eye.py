import math

import pytest

from driver_drowsiness.features.eye import calculate_ear


def test_calculate_ear_returns_expected_ratio_for_six_eye_points():
    points = [
        (0.0, 0.0),
        (1.0, 1.0),
        (3.0, 1.0),
        (4.0, 0.0),
        (3.0, -1.0),
        (1.0, -1.0),
    ]

    assert calculate_ear(points) == pytest.approx(0.5)


def test_calculate_ear_accepts_three_dimensional_points():
    points = [
        (0.0, 0.0, 2.0),
        (1.0, 1.0, 2.0),
        (3.0, 1.0, 2.0),
        (4.0, 0.0, 2.0),
        (3.0, -1.0, 2.0),
        (1.0, -1.0, 2.0),
    ]

    assert calculate_ear(points) == pytest.approx(0.5)


def test_calculate_ear_rejects_wrong_number_of_points():
    with pytest.raises(ValueError, match="6"):
        calculate_ear([(0.0, 0.0)] * 5)


def test_calculate_ear_rejects_zero_eye_width():
    points = [
        (0.0, 0.0),
        (1.0, 1.0),
        (3.0, 1.0),
        (0.0, 0.0),
        (3.0, -1.0),
        (1.0, -1.0),
    ]

    with pytest.raises(ValueError, match="horizontal"):
        calculate_ear(points)


def test_calculate_ear_rejects_non_finite_coordinates():
    points = [
        (0.0, 0.0),
        (1.0, math.nan),
        (3.0, 1.0),
        (4.0, 0.0),
        (3.0, -1.0),
        (1.0, -1.0),
    ]

    with pytest.raises(ValueError, match="finite"):
        calculate_ear(points)
