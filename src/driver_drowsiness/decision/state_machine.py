"""Fatigue state machine with temporal smoothing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from driver_drowsiness.decision.fatigue_score import FatigueFeatures, FatigueScorer


class FatigueState(Enum):
    """Discrete fatigue states."""

    NORMAL = "NORMAL"
    WATCH = "WATCH"
    WARNING = "WARNING"
    DANGER = "DANGER"


@dataclass(frozen=True)
class FatigueStateSnapshot:
    """State-machine output for one feature sample."""

    state: FatigueState
    score: float
    smoothed_score: float


@dataclass(frozen=True)
class FatigueStateMachineConfig:
    """State thresholds and hysteresis settings."""

    watch_score: float = 0.15
    warning_score: float = 0.35
    danger_score: float = 0.65
    smoothing_window: int = 3
    escalation_confirmations: int = 2


class FatigueStateMachine:
    """Convert feature samples into smoothed fatigue states."""

    _STATE_ORDER = [
        FatigueState.NORMAL,
        FatigueState.WATCH,
        FatigueState.WARNING,
        FatigueState.DANGER,
    ]

    def __init__(
        self,
        scorer: FatigueScorer | None = None,
        config: FatigueStateMachineConfig | None = None,
    ) -> None:
        self.scorer = scorer or FatigueScorer()
        self.config = config or FatigueStateMachineConfig()
        if self.config.smoothing_window <= 0:
            raise ValueError("smoothing_window must be positive.")
        if self.config.escalation_confirmations <= 0:
            raise ValueError("escalation_confirmations must be positive.")

        self._scores: deque[float] = deque(maxlen=self.config.smoothing_window)
        self._state = FatigueState.NORMAL
        self._pending_state = FatigueState.NORMAL
        self._pending_count = 0

    def update(self, features: FatigueFeatures) -> FatigueStateSnapshot:
        """Evaluate one feature sample and return the current fatigue state."""
        score = self.scorer.score(features)
        self._scores.append(score)
        smoothed_score = sum(self._scores) / len(self._scores)
        target_state = self._state_for_score(smoothed_score)
        self._state = self._next_state(target_state)
        return FatigueStateSnapshot(
            state=self._state,
            score=score,
            smoothed_score=smoothed_score,
        )

    def _state_for_score(self, score: float) -> FatigueState:
        if score >= self.config.danger_score:
            return FatigueState.DANGER
        if score >= self.config.warning_score:
            return FatigueState.WARNING
        if score >= self.config.watch_score:
            return FatigueState.WATCH
        return FatigueState.NORMAL

    def _next_state(self, target_state: FatigueState) -> FatigueState:
        current_level = self._level(self._state)
        target_level = self._level(target_state)

        if target_level <= current_level:
            self._pending_state = target_state
            self._pending_count = 0
            return target_state

        if target_state is self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = target_state
            self._pending_count = 1

        if self._pending_count >= self.config.escalation_confirmations:
            return target_state
        return self._state

    def _level(self, state: FatigueState) -> int:
        return self._STATE_ORDER.index(state)


__all__ = [
    "FatigueState",
    "FatigueStateMachine",
    "FatigueStateMachineConfig",
    "FatigueStateSnapshot",
]
