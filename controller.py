import math
from typing import Sequence


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class StepController:
    def __init__(
        self,
        safety: float = 0.9,
        damping: float = 1.0,
        min_ratio: float = 0.5,
        max_ratio: float = 2.0,
    ) -> None:
        self.safety: float = safety
        self.damping: float = damping
        self.min_ratio: float = min_ratio
        self.max_ratio: float = max_ratio

    def growth_ratio(self, error: float, order: int, target: float) -> float:
        if error <= 0.0 or target <= 0.0:
            return self.max_ratio
        exponent: float = self.damping / (order + 1)
        return clamp(
            self.safety * (target / error) ** exponent,
            self.min_ratio,
            self.max_ratio,
        )

    def next_step(
        self,
        h: float,
        error: float,
        order: int,
        remaining: float,
        steps_left: int,
    ) -> float:
        raise NotImplementedError


class FixedStepController(StepController):
    def __init__(self, schedule: Sequence[float]) -> None:
        super().__init__()
        self.schedule: list[float] = [float(value) for value in schedule]
        self.index: int = 0

    def next_step(
        self,
        h: float,
        error: float,
        order: int,
        remaining: float,
        steps_left: int,
    ) -> float:
        self.index += 1
        if self.index + 1 >= len(self.schedule):
            return remaining
        return self.schedule[self.index + 1] - self.schedule[self.index]


class ToleranceStepController(StepController):
    def next_step(
        self,
        h: float,
        error: float,
        order: int,
        remaining: float,
        steps_left: int,
    ) -> float:
        if steps_left <= 1:
            return remaining
        return h * self.growth_ratio(error, order, 1.0)


class BudgetStepController(StepController):
    def __init__(
        self,
        anchor: Sequence[float],
        spread: float = 3.0,
        smoothing: float = 0.5,
        safety: float = 1.0,
        damping: float = 1.0,
        min_ratio: float = 0.5,
        max_ratio: float = 2.0,
    ) -> None:
        super().__init__(
            safety=safety,
            damping=damping,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
        )
        self.spread: float = spread
        self.smoothing: float = smoothing
        self.reference: float | None = None
        self.anchor: list[float] = [float(value) for value in anchor]
        self.budget: float = float(len(self.anchor) - 1)
        self.position: float = 0.0
        self.synchronised: bool = False

    def update_reference(self, error: float) -> float:
        if self.reference is None:
            self.reference = error
        else:
            blend: float = self.smoothing
            self.reference = (1.0 - blend) * self.reference + blend * error
        return self.reference

    def sigma_at(self, position: float) -> float:
        clamped: float = clamp(position, 0.0, self.budget)
        low: int = int(math.floor(clamped))
        high: int = min(low + 1, len(self.anchor) - 1)
        weight: float = clamped - low
        return self.anchor[low] * (1.0 - weight) + self.anchor[high] * weight

    def next_step(
        self,
        h: float,
        error: float,
        order: int,
        remaining: float,
        steps_left: int,
    ) -> float:
        if not self.synchronised:
            self.position = self.budget - steps_left
            self.synchronised = True
        if steps_left <= 1:
            self.position = self.budget
            return remaining
        target: float = self.update_reference(error)
        available: float = self.budget - self.position
        pace: float = available / steps_left
        modulation: float = self.growth_ratio(error, order, target)
        reserve: float = available * (steps_left - 1) / (steps_left * self.spread)
        advance: float = clamp(
            pace * modulation,
            pace / self.spread,
            max(min(pace * self.spread, available - reserve), pace / self.spread),
        )
        self.position = self.position + advance
        return self.sigma_at(self.position) - (self.anchor[-1] - remaining)
