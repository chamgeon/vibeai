"""Metric base class, deepeval-style: measure() returns a score + reason,
is_successful() checks it against a threshold."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    score: float  # normalized to 0-1
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class Metric(ABC):
    name: str = "metric"
    threshold: float = 0.7

    @abstractmethod
    def measure(self, test_case) -> MetricResult:
        ...

    def is_successful(self, result: MetricResult) -> bool:
        return result.score >= self.threshold
