import pytest

from driver_drowsiness.decision.fatigue_score import (
    FatigueFeatures,
    FatigueScorer,
)


def test_score_stays_low_for_normal_driver_features():
    scorer = FatigueScorer()

    features = FatigueFeatures(
        ear=0.32,
        eye_closed=False,
        eye_closed_duration=0.0,
        perclos=0.05,
        mar=0.25,
        yawning_detected=False,
        head_pitch=0.0,
        head_yaw=0.0,
    )

    assert scorer.score(features) == pytest.approx(0.0)


def test_score_increases_for_long_eye_closure():
    scorer = FatigueScorer()

    features = FatigueFeatures(
        ear=0.15,
        eye_closed=True,
        eye_closed_duration=2.5,
        perclos=0.20,
        mar=0.25,
        yawning_detected=False,
        head_pitch=0.0,
        head_yaw=0.0,
    )

    assert scorer.score(features) >= 0.5


def test_score_rejects_negative_feature_values():
    scorer = FatigueScorer()

    features = FatigueFeatures(
        ear=-0.1,
        eye_closed=False,
        eye_closed_duration=0.0,
        perclos=0.0,
        mar=0.0,
        yawning_detected=False,
        head_pitch=0.0,
        head_yaw=0.0,
    )

    with pytest.raises(ValueError, match="ear"):
        scorer.score(features)
