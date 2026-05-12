"""Fatigue scoring from simulated feature values."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FatigueFeatures:
    """Feature-only input for fatigue decisions."""

    ear: float
    eye_closed: bool
    eye_closed_duration: float
    perclos: float
    mar: float
    yawning_detected: bool
    head_pitch: float
    head_yaw: float


@dataclass(frozen=True)
class FatigueScoreConfig:
    """Thresholds and weights for the simple fatigue score."""

    ear_warning: float = 0.25
    ear_danger: float = 0.20
    eye_closed_duration_warning: float = 1.0
    eye_closed_duration_danger: float = 2.0
    perclos_warning: float = 0.30
    perclos_danger: float = 0.45
    mar_warning: float = 0.60
    mar_danger: float = 0.75
    head_pitch_warning: float = 20.0
    head_pitch_danger: float = 35.0
    head_yaw_warning: float = 25.0
    head_yaw_danger: float = 45.0
    ear_weight: float = 0.20
    eye_closed_duration_weight: float = 0.30
    perclos_weight: float = 0.30
    yawn_weight: float = 0.20
    head_pose_weight: float = 0.20


class FatigueScorer:
    """Calculate a normalized fatigue score between 0.0 and 1.0."""

    def __init__(self, config: FatigueScoreConfig | None = None) -> None:
        # TODO: Load these defaults from config/default.yaml after a shared config
        # loader exists.
        self.config = config or FatigueScoreConfig()

    def score(self, features: FatigueFeatures) -> float:
        """Return a weighted fatigue score for one feature sample."""
        self._validate_features(features)
        config = self.config

        ear_risk = _inverse_threshold_risk(
            value=features.ear,
            warning=config.ear_warning,
            danger=config.ear_danger,
        )
        closed_duration_risk = _threshold_risk(
            value=features.eye_closed_duration if features.eye_closed else 0.0,
            warning=config.eye_closed_duration_warning,
            danger=config.eye_closed_duration_danger,
        )
        perclos_risk = _threshold_risk(
            value=features.perclos,
            warning=config.perclos_warning,
            danger=config.perclos_danger,
        )
        mar_risk = _threshold_risk(
            value=features.mar,
            warning=config.mar_warning,
            danger=config.mar_danger,
        )
        yawn_risk = max(mar_risk, 1.0 if features.yawning_detected else 0.0)
        head_pose_risk = max(
            _threshold_risk(
                value=abs(features.head_pitch),
                warning=config.head_pitch_warning,
                danger=config.head_pitch_danger,
            ),
            _threshold_risk(
                value=abs(features.head_yaw),
                warning=config.head_yaw_warning,
                danger=config.head_yaw_danger,
            ),
        )

        weighted_score = (
            config.ear_weight * ear_risk
            + config.eye_closed_duration_weight * closed_duration_risk
            + config.perclos_weight * perclos_risk
            + config.yawn_weight * yawn_risk
            + config.head_pose_weight * head_pose_risk
        )
        return _clamp(weighted_score)

    @staticmethod
    def _validate_features(features: FatigueFeatures) -> None:
        values = {
            "ear": features.ear,
            "eye_closed_duration": features.eye_closed_duration,
            "perclos": features.perclos,
            "mar": features.mar,
            "head_pitch": features.head_pitch,
            "head_yaw": features.head_yaw,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        for name in ("ear", "eye_closed_duration", "perclos", "mar"):
            if values[name] < 0.0:
                raise ValueError(f"{name} must be non-negative.")


def _threshold_risk(value: float, warning: float, danger: float) -> float:
    if value < warning:
        return 0.0
    if value >= danger:
        return 1.0
    return (value - warning) / (danger - warning)


def _inverse_threshold_risk(value: float, warning: float, danger: float) -> float:
    if value > warning:
        return 0.0
    if value <= danger:
        return 1.0
    return (warning - value) / (warning - danger)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


__all__ = ["FatigueFeatures", "FatigueScoreConfig", "FatigueScorer"]
