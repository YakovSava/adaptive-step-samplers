import numpy as np

DIMENSION: int = 8
MODE_COUNT: int = 6
SIGMA_START: float = 10.0
SIGMA_TARGET: float = 0.01


def build_modes(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    modes: np.ndarray = rng.normal(0.0, 1.0, (MODE_COUNT, DIMENSION))
    weights: np.ndarray = rng.uniform(0.5, 1.5, MODE_COUNT)
    return modes, weights / weights.sum()


MODES, WEIGHTS = build_modes()
LOG_WEIGHTS: np.ndarray = np.log(WEIGHTS)


def denoiser(x: np.ndarray, sigma: float) -> np.ndarray:
    difference: np.ndarray = MODES - x
    logits: np.ndarray = -0.5 * np.sum(difference * difference, axis=1) / sigma**2
    logits = logits + LOG_WEIGHTS
    logits = logits - logits.max()
    posterior: np.ndarray = np.exp(logits)
    posterior = posterior / posterior.sum()
    return posterior @ MODES


def slope(sigma: float, x: np.ndarray) -> np.ndarray:
    return (x - denoiser(x, sigma)) / sigma


def rk4_reference(start: float, target: float, x0: np.ndarray, steps: int) -> np.ndarray:
    h: float = (target - start) / steps
    sigma: float = start
    state: np.ndarray = np.array(x0, dtype=np.float64)
    for _ in range(steps):
        a: np.ndarray = slope(sigma, state)
        b: np.ndarray = slope(sigma + 0.5 * h, state + 0.5 * h * a)
        c: np.ndarray = slope(sigma + 0.5 * h, state + 0.5 * h * b)
        d: np.ndarray = slope(sigma + h, state + h * c)
        state = state + h * (a + 2.0 * b + 2.0 * c + d) / 6.0
        sigma = sigma + h
    return state


def initial_states(count: int = 6, seed: int = 11) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, SIGMA_START, DIMENSION) for _ in range(count)]


def uniform_schedule(steps: int) -> list[float]:
    span: float = SIGMA_TARGET - SIGMA_START
    return [SIGMA_START + span * i / steps for i in range(steps + 1)]


def karras_schedule(steps: int, rho: float = 7.0) -> list[float]:
    low: float = SIGMA_TARGET ** (1.0 / rho)
    high: float = SIGMA_START ** (1.0 / rho)
    return [(high + (low - high) * i / steps) ** rho for i in range(steps + 1)]


def exponential_schedule(steps: int) -> list[float]:
    low: float = np.log(SIGMA_TARGET)
    high: float = np.log(SIGMA_START)
    return [float(np.exp(high + (low - high) * i / steps)) for i in range(steps + 1)]


def shift_schedule(steps: int, rho: float) -> list[float]:
    span: float = SIGMA_START - SIGMA_TARGET
    return [SIGMA_TARGET + span * (1.0 - i / steps) ** rho for i in range(steps + 1)]
