import math

import numpy as np

from _bootstrap import module

pc_core = module("pc_core")
integrator = module("integrator")

TEXTBOOK_PREDICTOR: dict[int, list[float]] = {
    1: [1.0],
    2: [3.0 / 2.0, -1.0 / 2.0],
    3: [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0],
    4: [55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
}

TEXTBOOK_CORRECTOR: dict[int, list[float]] = {
    1: [1.0],
    2: [1.0 / 2.0, 1.0 / 2.0],
    3: [5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0],
    4: [9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0],
}


def check_uniform_weights() -> bool:
    ok: bool = True
    past: list[float] = [1.0, 1.0, 1.0]
    for order in range(1, pc_core.MAX_ORDER + 1):
        predictor = pc_core.predictor_weights(1.0, past, order)
        corrector = pc_core.corrector_weights(1.0, past, order)
        predictor_gap = float(np.max(np.abs(predictor - np.array(TEXTBOOK_PREDICTOR[order]))))
        corrector_gap = float(np.max(np.abs(corrector - np.array(TEXTBOOK_CORRECTOR[order]))))
        ok = ok and predictor_gap < 1e-12 and corrector_gap < 1e-12
        print(
            f"  order {order}: predictor gap {predictor_gap:.2e}"
            f"  corrector gap {corrector_gap:.2e}"
        )
    return ok


def check_variable_weights() -> bool:
    past: list[float] = [0.37, 0.11, 0.53]
    ok: bool = True
    for order in range(1, pc_core.MAX_ORDER + 1):
        for h in [0.19, -0.19]:
            predictor = pc_core.predictor_weights(h, past, order)
            corrector = pc_core.corrector_weights(h, past, order)
            predictor_sum = float(np.sum(predictor))
            corrector_sum = float(np.sum(corrector))
            ok = ok and abs(predictor_sum - h) < 1e-10 and abs(corrector_sum - h) < 1e-10
        print(f"  order {order}: weight sums match h")
    return ok


def smooth_denoised(sigma: float) -> float:
    return 1.0 + 0.5 * math.sin(3.0 * sigma) + 0.2 * math.cos(5.0 * sigma)


def reference_value(start: float, target: float, x0: float, steps: int) -> float:
    h: float = (target - start) / steps
    sigma: float = start
    state: float = x0
    for _ in range(steps):
        slope_a: float = (state - smooth_denoised(sigma)) / sigma
        mid: float = state + 0.5 * h * slope_a
        slope_b: float = (mid - smooth_denoised(sigma + 0.5 * h)) / (sigma + 0.5 * h)
        mid2: float = state + 0.5 * h * slope_b
        slope_c: float = (mid2 - smooth_denoised(sigma + 0.5 * h)) / (sigma + 0.5 * h)
        end: float = state + h * slope_c
        slope_d: float = (end - smooth_denoised(sigma + h)) / (sigma + h)
        state = state + h * (slope_a + 2.0 * slope_b + 2.0 * slope_c + slope_d) / 6.0
        sigma = sigma + h
    return state


START: float = 1.0
TARGET: float = 0.1
INITIAL_STATE: float = 1.7


def run_fixed(order: int, steps: int, use_corrector: bool) -> float:
    schedule: list[float] = [START + (TARGET - START) * i / steps for i in range(steps + 1)]
    config = integrator.AdaptiveConfig(
        order=order,
        mode="fixed",
        use_corrector=use_corrector,
    )
    return integrator.integrate_adaptive_pc(
        lambda state, sigma: smooth_denoised(sigma),
        INITIAL_STATE,
        schedule,
        config,
    )


def observed_rates(order: int, use_corrector: bool, grids: list[int], truth: float) -> list[float]:
    errors: list[float] = [abs(run_fixed(order, steps, use_corrector) - truth) for steps in grids]
    rates: list[float] = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    formatted = "  ".join(f"{value:.2e}" for value in errors)
    rate_text = "  ".join(f"{value:.2f}" for value in rates)
    print(f"  order {order}: errors {formatted} | rates {rate_text}")
    return rates


def exact_state(sigma: float) -> float:
    if abs(sigma - START) < 1e-15:
        return INITIAL_STATE
    return reference_value(START, sigma, INITIAL_STATE, 20000)


def check_local_order() -> bool:
    base: float = 0.6
    ok: bool = True
    for order in range(1, pc_core.MAX_ORDER + 1):
        errors: list[float] = []
        deltas: list[float] = [0.02, 0.01, 0.005, 0.0025]
        for delta in deltas:
            past: list[float] = [-delta] * (order - 1)
            nodes: list[float] = [base + index * delta for index in range(order)]
            values: list[float] = [
                (exact_state(node) - smooth_denoised(node)) / node for node in nodes
            ]
            predicted: float = integrator.combine(
                exact_state(base),
                pc_core.predictor_weights(-delta, past, order),
                values,
            )
            errors.append(abs(predicted - exact_state(base - delta)))
        rates: list[float] = [
            math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)
        ]
        ok = ok and rates[-1] > order + 0.6
        formatted = "  ".join(f"{value:.2e}" for value in errors)
        rate_text = "  ".join(f"{value:.2f}" for value in rates)
        print(f"  order {order}: local errors {formatted} | rates {rate_text}")
    return ok


def check_convergence() -> bool:
    truth: float = reference_value(START, TARGET, INITIAL_STATE, 40000)
    grids: list[int] = [160, 320, 640, 1280]
    ok: bool = True
    for order in range(1, pc_core.MAX_ORDER + 1):
        rates = observed_rates(order, False, grids, truth)
        ok = ok and rates[-1] > min(order, 2) - 0.1
    return ok


def check_corrector_order_cap() -> bool:
    truth: float = reference_value(START, TARGET, INITIAL_STATE, 40000)
    grids: list[int] = [640, 1280, 2560, 5120]
    ok: bool = True
    for order in range(2, pc_core.MAX_ORDER + 1):
        rates = observed_rates(order, True, grids, truth)
        ok = ok and 1.7 < rates[-1] < 2.3
    return ok


def check_corrector_coarse_gain() -> bool:
    truth: float = reference_value(START, TARGET, INITIAL_STATE, 40000)
    ok: bool = True
    for order in range(2, pc_core.MAX_ORDER + 1):
        ratios: list[float] = []
        for steps in [8, 12, 16, 24, 32]:
            predictor_error = abs(run_fixed(order, steps, False) - truth)
            corrector_error = abs(run_fixed(order, steps, True) - truth)
            ratios.append(predictor_error / corrector_error)
        if order <= 3:
            ok = ok and min(ratios) > 1.0
        text = "  ".join(f"{value:5.2f}x" for value in ratios)
        print(f"  order {order}: predictor error over corrector error {text}")
    return ok


def check_milne_factor() -> bool:
    corrected: dict[int, float] = {1: 0.5, 2: 1.0 / 6.0, 3: 1.0 / 10.0, 4: 19.0 / 270.0}
    predicted: dict[int, float] = {1: 0.5, 2: 5.0 / 6.0, 3: 9.0 / 10.0, 4: 251.0 / 270.0}
    ok: bool = True
    for order in range(1, pc_core.MAX_ORDER + 1):
        on_corrector: float = pc_core.milne_factor(order, True)
        on_predictor: float = pc_core.milne_factor(order, False)
        ok = ok and abs(on_corrector - corrected[order]) < 1e-12
        ok = ok and abs(on_predictor - predicted[order]) < 1e-12
        ok = ok and abs(on_corrector + on_predictor - 1.0) < 1e-12
        print(f"  order {order}: corrector {on_corrector:.6f}  predictor {on_predictor:.6f}")
    return ok


def check_error_estimate_quality() -> bool:
    truth_grid: int = 40000
    ok: bool = True
    for order in [2, 3]:
        print(f"  order {order}")
        for steps in [16, 32, 64]:
            schedule: list[float] = [
                START + (TARGET - START) * i / steps for i in range(steps + 1)
            ]
            config = integrator.AdaptiveConfig(order=order, mode="fixed", use_corrector=False)
            estimates: list[float] = []
            actual: list[float] = []
            states: list[tuple[float, float, float]] = []

            def observer(record: dict) -> None:
                if record["error"] is not None:
                    estimates.append(record["error"])
                states.append((record["sigma"], record["x"], record["step_size"]))

            integrator.integrate_adaptive_pc(
                lambda state, sigma: smooth_denoised(sigma),
                INITIAL_STATE,
                schedule,
                config,
                on_step=observer,
            )

            for index in range(1, len(states)):
                sigma_from, state_from, step = states[index - 1]
                sigma_to, state_to, _ = states[index]
                exact: float = reference_value(
                    sigma_from, sigma_to, state_from, max(8, truth_grid // steps)
                )
                scale: float = config.atol + config.rtol * abs(state_to)
                actual.append(abs(state_to - exact) / scale)

            paired = [
                (estimates[i], actual[i])
                for i in range(len(estimates))
                if i < len(actual)
            ]
            if not paired:
                continue
            ratios = [a / e for e, a in paired if e > 0.0]
            median = sorted(ratios)[len(ratios) // 2]
            ok = ok and 0.1 < median < 10.0
            print(f"    n={steps:3d}: median actual/estimated local error {median:.2f}")
    return ok


if __name__ == "__main__":
    print("uniform-step weights versus textbook Adams coefficients")
    weights_ok = check_uniform_weights()
    print("\nvariable-step weight consistency (sum of weights equals h)")
    variable_ok = check_variable_weights()
    print("\nMilne device constants")
    milne_ok = check_milne_factor()
    print("\nlocal truncation order with exact history (expect order + 1)")
    local_ok = check_local_order()
    print("\nglobal order, predictor trajectory (capped near 2 by the Euler startup step)")
    convergence_ok = check_convergence()
    print("\norder cap when the corrector is written back into the state")
    cap_ok = check_corrector_order_cap()
    print("\ncorrector gain in the low-step regime")
    gain_ok = check_corrector_coarse_gain()
    print("\ncalibration of the free local error estimate")
    estimate_ok = check_error_estimate_quality()
    print("\nresult")
    for label, passed in [
        ("uniform weights", weights_ok),
        ("variable weights", variable_ok),
        ("milne constants", milne_ok),
        ("local truncation order", local_ok),
        ("global order floor", convergence_ok),
        ("corrector order cap at 2", cap_ok),
        ("corrector coarse-grid gain", gain_ok),
        ("error estimate calibration", estimate_ok),
    ]:
        print(f"  {label}: {'pass' if passed else 'FAIL'}")
