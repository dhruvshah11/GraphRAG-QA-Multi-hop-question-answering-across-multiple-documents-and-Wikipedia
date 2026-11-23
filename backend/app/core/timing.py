"""Small utilities for measuring latency of pipeline stages.

Every measurement shown in the UI comes from these timers; nothing is estimated.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class StageTimer:
    """Collects wall-clock durations (milliseconds) for named stages."""

    durations_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.durations_ms[name] = self.durations_ms.get(name, 0.0) + elapsed

    def add(self, name: str, ms: float) -> None:
        self.durations_ms[name] = self.durations_ms.get(name, 0.0) + ms

    @property
    def total_ms(self) -> float:
        return sum(self.durations_ms.values())

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self.durations_ms.items()}


def now_ms() -> float:
    return time.perf_counter() * 1000.0
