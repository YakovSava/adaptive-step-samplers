from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .controller import BudgetStepController, FixedStepController, StepController
from .pc_core import (
    MAX_ORDER,
    corrector_order,
    corrector_weights,
    milne_factor,
    predictor_weights,
)

BUDGET_MODE: str = "budget"
FIXED_MODE: str = "fixed"
MODES: list[str] = [BUDGET_MODE, FIXED_MODE]


@dataclass
class AdaptiveConfig:
    order: int = 2
    mode: str = BUDGET_MODE
    rtol: float = 0.05
    atol: float = 0.0078
    damping: float = 1.0
    min_ratio: float = 0.5
    max_ratio: float = 2.0
    spread: float = 3.0
    smoothing: float = 0.5
    use_corrector: bool = True
    global_weighting: bool = False
    anchor_to_schedule: bool = True
    sigma_floor: float = 1e-5

    def __post_init__(self) -> None:
        self.order = max(1, min(int(self.order), MAX_ORDER))
        if self.mode not in MODES:
            self.mode = BUDGET_MODE


def build_controller(config: AdaptiveConfig, schedule: Sequence[float]) -> StepController:
    if config.mode == FIXED_MODE:
        return FixedStepController(schedule)
    anchor: Sequence[float] = schedule
    if not config.anchor_to_schedule:
        count: int = len(schedule) - 1
        span: float = schedule[-1] - schedule[0]
        anchor = [schedule[0] + span * i / count for i in range(count + 1)]
    return BudgetStepController(
        spread=config.spread,
        smoothing=config.smoothing,
        damping=config.damping,
        min_ratio=config.min_ratio,
        max_ratio=config.max_ratio,
        anchor=anchor,
    )


def combine(base: Any, weights: np.ndarray, values: Sequence[Any]) -> Any:
    result: Any = base
    for weight, value in zip(weights, values):
        result = result + float(weight) * value
    return result


def relative_error(residual: Any, reference: Any, rtol: float, atol: float) -> float:
    scale: Any = atol + rtol * abs(reference)
    ratio: Any = residual / scale
    squared: Any = ratio * ratio
    if hasattr(squared, "mean"):
        return float(squared.mean() ** 0.5)
    return float(squared**0.5)


def integrate_adaptive_pc(
    denoise_fn: Callable[[Any, float], Any],
    x: Any,
    schedule: Sequence[float],
    config: AdaptiveConfig,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    controller: StepController = build_controller(config, schedule)
    budget: int = len(schedule) - 1
    sigma_target: float = float(schedule[-1])
    t: float = float(schedule[0])
    h: float = float(schedule[1]) - t
    derivatives: list[Any] = []
    steps: list[float] = []
    previous_state: Any = None
    predicted_state: Any = None

    for index in range(budget):
        sigma_eval: float = max(t, config.sigma_floor)
        denoised: Any = denoise_fn(x, sigma_eval)
        derivative: Any = (x - denoised) / sigma_eval

        error: float | None = None
        order_used: int = 1
        if predicted_state is not None:
            past: list[float] = steps[1:]
            order_used = corrector_order(config.order, past)
            weights: np.ndarray = corrector_weights(steps[0], past, config.order)
            corrected: Any = combine(previous_state, weights, [derivative] + derivatives)
            factor: float = milne_factor(order_used, config.use_corrector)
            residual: Any = (predicted_state - corrected) * factor
            error = relative_error(residual, x, config.rtol, config.atol)
            if config.global_weighting:
                error = error * max(sigma_target, config.sigma_floor) / sigma_eval
            if config.use_corrector:
                x = corrected

        steps_left: int = budget - index
        remaining: float = sigma_target - t
        if error is None:
            h_next: float = h
        else:
            h_next = controller.next_step(h, error, order_used, remaining, steps_left)

        final: bool = steps_left <= 1 or abs(h_next) >= abs(remaining)
        if final:
            h_next = remaining

        if on_step is not None:
            on_step(
                {
                    "index": index,
                    "sigma": t,
                    "x": x,
                    "denoised": denoised,
                    "derivative": derivative,
                    "error": error,
                    "order": order_used,
                    "step_size": h_next,
                    "final": final,
                }
            )

        if final:
            if sigma_target <= 0.0:
                return denoised
            return combine(
                x,
                predictor_weights(h_next, steps, config.order),
                [derivative] + derivatives,
            )

        previous_state = x
        predicted_state = combine(
            x,
            predictor_weights(h_next, steps, config.order),
            [derivative] + derivatives,
        )
        derivatives.insert(0, derivative)
        steps.insert(0, h_next)
        del derivatives[config.order:]
        del steps[config.order:]
        x = predicted_state
        t = t + h_next
        h = h_next

    return x
