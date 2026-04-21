"""Judge protocol — pluggable scorers that return a SampleScore."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import EvalSample, SampleScore


class Judge(ABC):
    """A Judge scores one EvalSample and returns a SampleScore in [0, 1]."""

    name: str = "base"

    @abstractmethod
    def score(self, sample: EvalSample) -> SampleScore: ...
