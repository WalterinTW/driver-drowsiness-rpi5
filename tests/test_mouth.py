import pytest

from driver_drowsiness.features.mouth import calculate_mar


def test_calculate_mar_returns_expected_ratio_for_eight_mouth_points():
    points = [
        (0.0, 0.0),
        (1.0, 1.0),
        (3.0, 2.0),
        (5.0, 1.0),
        (6.0, 0.0),
        (5.0, -1.0),
        (3.0, -2.0),
        (1.0, -1.0),
    ]

    assert calculate_mar(points) == pytest.approx(2.0 / 3.0)


def test_calculate_mar_rejects_wrong_number_of_points():
    with pytest.raises(ValueError, match="8"):
        calculate_mar([(0.0, 0.0)] * 7)


def test_calculate_mar_rejects_zero_mouth_width():
    points = [
        (0.0, 0.0),
        (1.0, 1.0),
        (3.0, 2.0),
        (5.0, 1.0),
        (0.0, 0.0),
        (5.0, -1.0),
        (3.0, -2.0),
        (1.0, -1.0),
    ]

    with pytest.raises(ValueError, match="horizontal"):
        calculate_mar(points)
