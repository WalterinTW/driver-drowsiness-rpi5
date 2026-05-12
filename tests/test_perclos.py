import pytest

from driver_drowsiness.features.perclos import PerclosCalculator


def test_perclos_returns_zero_when_window_has_no_samples():
    calculator = PerclosCalculator(window_seconds=10.0)

    assert calculator.ratio() == 0.0


def test_perclos_returns_closed_sample_ratio_in_current_window():
    calculator = PerclosCalculator(window_seconds=10.0)

    calculator.update(timestamp=0.0, is_closed=True)
    calculator.update(timestamp=2.0, is_closed=False)
    ratio = calculator.update(timestamp=5.0, is_closed=True)

    assert ratio == pytest.approx(2.0 / 3.0)
    assert calculator.ratio() == pytest.approx(2.0 / 3.0)


def test_perclos_prunes_samples_older_than_window():
    calculator = PerclosCalculator(window_seconds=10.0)

    calculator.update(timestamp=0.0, is_closed=True)
    calculator.update(timestamp=2.0, is_closed=False)
    calculator.update(timestamp=5.0, is_closed=True)
    ratio = calculator.update(timestamp=12.0, is_closed=False)

    assert ratio == pytest.approx(1.0 / 3.0)


def test_perclos_rejects_non_positive_window():
    with pytest.raises(ValueError, match="positive"):
        PerclosCalculator(window_seconds=0.0)


def test_perclos_rejects_out_of_order_samples():
    calculator = PerclosCalculator(window_seconds=10.0)
    calculator.update(timestamp=3.0, is_closed=False)

    with pytest.raises(ValueError, match="monotonic"):
        calculator.update(timestamp=2.0, is_closed=True)
