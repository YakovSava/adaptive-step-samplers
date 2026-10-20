from typing import Sequence

import numpy as np

MAX_ORDER: int = 4

ADAMS_BASHFORTH_LTE: dict[int, float] = {
    1: 1.0 / 2.0,
    2: 5.0 / 12.0,
    3: 3.0 / 8.0,
    4: 251.0 / 720.0,
}

ADAMS_MOULTON_LTE: dict[int, float] = {
    1: 1.0 / 2.0,
    2: 1.0 / 12.0,
    3: 1.0 / 24.0,
    4: 19.0 / 720.0,
}


def milne_factor(order: int, corrected: bool = True) -> float:
    predictor_lte: float = ADAMS_BASHFORTH_LTE[order]
    corrector_lte: float = ADAMS_MOULTON_LTE[order]
    target: float = corrector_lte if corrected else predictor_lte
    return target / (predictor_lte + corrector_lte)


def divided_difference_functionals(taus: Sequence[float]) -> np.ndarray:
    count: int = len(taus)
    table: np.ndarray = np.eye(count, dtype=np.float64)
    functionals: np.ndarray = np.zeros((count, count), dtype=np.float64)
    functionals[0] = table[0]
    for level in range(1, count):
        for index in range(count - 1, level - 1, -1):
            spacing: float = taus[index] - taus[index - level]
            table[index] = (table[index] - table[index - 1]) / spacing
        functionals[level] = table[level]
    return functionals


def node_product_polynomial(taus: Sequence[float], degree: int) -> np.ndarray:
    poly: np.ndarray = np.ones(1, dtype=np.float64)
    for index in range(degree):
        poly = np.convolve(poly, np.array([1.0, -taus[index]], dtype=np.float64))
    return poly


def integrate_polynomial(poly: np.ndarray, h: float) -> float:
    powers: np.ndarray = np.arange(poly.size - 1, -1, -1)
    return float(np.sum(poly * h ** (powers + 1) / (powers + 1)))


def adams_weights(taus: Sequence[float], h: float) -> np.ndarray:
    functionals: np.ndarray = divided_difference_functionals(taus)
    weights: np.ndarray = np.zeros(len(taus), dtype=np.float64)
    for level in range(len(taus)):
        integral: float = integrate_polynomial(node_product_polynomial(taus, level), h)
        weights = weights + integral * functionals[level]
    return weights


def history_taus(past_steps: Sequence[float]) -> list[float]:
    taus: list[float] = [0.0]
    offset: float = 0.0
    for step in past_steps:
        offset -= step
        taus.append(offset)
    return taus


def predictor_order(order: int, past_steps: Sequence[float]) -> int:
    return max(1, min(order, 1 + len(past_steps)))


def corrector_order(order: int, past_steps: Sequence[float]) -> int:
    return max(1, min(order, 1 + len(past_steps)))


def predictor_weights(h: float, past_steps: Sequence[float], order: int) -> np.ndarray:
    count: int = predictor_order(order, past_steps)
    return adams_weights(history_taus(list(past_steps)[: count - 1]), h)


def corrector_weights(h: float, past_steps: Sequence[float], order: int) -> np.ndarray:
    count: int = corrector_order(order, past_steps)
    if count < 2:
        return adams_weights([h], h)
    return adams_weights([h] + history_taus(list(past_steps)[: count - 2]), h)
