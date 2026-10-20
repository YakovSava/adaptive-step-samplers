from typing import Any, Callable

import numpy as np

import mixture_problem as mp
from _bootstrap import module

integrator = module("integrator")

BUDGETS: list[int] = [6, 8, 10, 12, 16, 20, 25]
REFERENCE_STEPS: int = 8000
ScheduleFactory = Callable[[int], list[float]]


def fixed_config(order: int) -> Any:
    return integrator.AdaptiveConfig(order=order, mode="fixed", use_corrector=True)


def budget_config(
    order: int,
    spread: float = 3.0,
    anchored: bool = True,
    weighting: bool = False,
    smoothing: float = 0.5,
) -> Any:
    return integrator.AdaptiveConfig(
        order=order,
        mode="budget",
        use_corrector=True,
        spread=spread,
        smoothing=smoothing,
        anchor_to_schedule=anchored,
        global_weighting=weighting,
    )


def run(config: Any, schedule: list[float], x0: np.ndarray) -> tuple[np.ndarray, int]:
    counter: dict[str, int] = {"calls": 0}

    def denoise_fn(state: np.ndarray, sigma: float) -> np.ndarray:
        counter["calls"] += 1
        return mp.denoiser(state, sigma)

    result = integrator.integrate_adaptive_pc(
        denoise_fn,
        np.array(x0, dtype=np.float64),
        schedule,
        config,
    )
    return result, counter["calls"]


def mean_error(
    config_factory: Callable[[], Any],
    schedule_factory: ScheduleFactory,
    steps: int,
    states: list[np.ndarray],
    truths: list[np.ndarray],
) -> tuple[float, int]:
    errors: list[float] = []
    calls: int = 0
    for state, truth in zip(states, truths):
        value, calls = run(config_factory(), schedule_factory(steps), state)
        errors.append(float(np.sqrt(np.mean((value - truth) ** 2))))
    return float(np.mean(errors)), calls


def report_grid(steps: int, schedule_factory: ScheduleFactory, state: np.ndarray) -> None:
    visited: list[float] = []

    def observer(record: dict[str, Any]) -> None:
        visited.append(record["sigma"])

    schedule: list[float] = schedule_factory(steps)
    integrator.integrate_adaptive_pc(
        lambda current, sigma: mp.denoiser(current, sigma),
        np.array(state, dtype=np.float64),
        schedule,
        budget_config(2),
        on_step=observer,
    )
    print("    base     " + " ".join(f"{value:6.3f}" for value in schedule[:-1]))
    print("    adaptive " + " ".join(f"{value:6.3f}" for value in visited))


if __name__ == "__main__":
    states: list[np.ndarray] = mp.initial_states()
    truths: list[np.ndarray] = [
        mp.rk4_reference(mp.SIGMA_START, mp.SIGMA_TARGET, state, REFERENCE_STEPS)
        for state in states
    ]

    variants: list[tuple[str, Callable[[], Any], ScheduleFactory]] = [
        ("uniform", lambda: fixed_config(2), mp.uniform_schedule),
        ("karras", lambda: fixed_config(2), mp.karras_schedule),
        ("exponential", lambda: fixed_config(2), mp.exponential_schedule),
        ("shift rho=3", lambda: fixed_config(2), lambda n: mp.shift_schedule(n, 3.0)),
        ("uniform+apc", lambda: budget_config(2), mp.uniform_schedule),
        ("karras+apc", lambda: budget_config(2), mp.karras_schedule),
        ("karras+apc p3", lambda: budget_config(3), mp.karras_schedule),
        ("exp+apc", lambda: budget_config(2), mp.exponential_schedule),
    ]

    header: str = f"{'variant':>14} | " + " | ".join(f"nfe={n:<3d}" for n in BUDGETS)
    print("mean rms error against the reference trajectory, equal NFE per column")
    print(header)
    print("-" * len(header))

    table: dict[str, list[float]] = {}
    for label, config_factory, schedule_factory in variants:
        row: list[float] = []
        for steps in BUDGETS:
            error, used = mean_error(config_factory, schedule_factory, steps, states, truths)
            row.append(error)
            if used != steps:
                print(f"  nfe mismatch for {label} at {steps}: used {used}")
        table[label] = row
        print(f"{label:>14} | " + " | ".join(f"{value:.2e}" for value in row))

    print("\ngain from adding the free adaptive controller to a static schedule")
    for adaptive, static in [
        ("uniform+apc", "uniform"),
        ("karras+apc", "karras"),
        ("exp+apc", "exponential"),
    ]:
        gains = [table[static][i] / table[adaptive][i] for i in range(len(BUDGETS))]
        print(f"{adaptive:>14} | " + " | ".join(f"{value:6.2f}x" for value in gains))

    print("\nstep placement at nfe=12, starting from the uniform schedule")
    report_grid(12, mp.uniform_schedule, states[0])
    print("\nstep placement at nfe=12, starting from the karras schedule")
    report_grid(12, mp.karras_schedule, states[0])
