"""PERCLOS sliding-window calculation."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PerclosCalculator:
    """Track closed-eye ratio over a monotonic timestamp window."""

    window_seconds: float
    _samples: deque[tuple[float, bool]] = field(default_factory=deque, init=False)
    _last_timestamp: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.window_seconds = float(self.window_seconds)
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0.0:
            raise ValueError("PERCLOS window_seconds must be positive.")

    def update(self, timestamp: float, is_closed: bool) -> float:
        """Add one timestamped eye state and return the current closed ratio."""
        timestamp = self._validate_timestamp(timestamp)
        self._samples.append((timestamp, bool(is_closed)))
        self._last_timestamp = timestamp
        self._prune(timestamp)
        return self.ratio()

    def ratio(self, current_timestamp: float | None = None) -> float:
        """Return the closed-eye ratio in the current window."""
        if current_timestamp is not None:
            current_timestamp = self._validate_timestamp(current_timestamp)
            self._prune(current_timestamp)

        if not self._samples:
            return 0.0

        closed_count = sum(1 for _, is_closed in self._samples if is_closed)
        return closed_count / len(self._samples)

    def _validate_timestamp(self, timestamp: float) -> float:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("PERCLOS timestamp must be finite.")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("PERCLOS timestamps must be monotonic.")
        return timestamp

    def _prune(self, current_timestamp: float) -> None:
        cutoff = current_timestamp - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()


__all__ = ["PerclosCalculator"]
