from typing import Any, Callable

import torch
from comfy.utils import model_trange as trange

from .integrator import AdaptiveConfig, integrate_adaptive_pc


@torch.no_grad()
def sample_adaptive_pc(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Callable[..., Any] | None = None,
    disable: bool | None = None,
    order: int = 2,
    mode: str = "budget",
    rtol: float = 0.05,
    atol: float = 0.0078,
    safety: float = 0.9,
    damping: float = 1.0,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
    spread: float = 3.0,
    smoothing: float = 0.5,
    use_corrector: bool = True,
) -> torch.Tensor:
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    config: AdaptiveConfig = AdaptiveConfig(
        order=order,
        mode=mode,
        rtol=rtol,
        atol=atol,
        safety=safety,
        damping=damping,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        spread=spread,
        smoothing=smoothing,
        use_corrector=use_corrector,
    )
    budget: int = len(sigmas) - 1
    s_in: torch.Tensor = x.new_ones([x.shape[0]])
    reference: torch.Tensor = sigmas[0]
    progress = trange(budget, disable=disable)

    def denoise_fn(state: torch.Tensor, sigma: float) -> torch.Tensor:
        return model(state, sigma * s_in, **extra_args)

    def on_step(record: dict[str, Any]) -> None:
        progress.update(1)
        if callback is not None:
            sigma_value: torch.Tensor = reference.new_full((), record["sigma"])
            callback(
                {
                    "x": record["x"],
                    "i": record["index"],
                    "sigma": sigma_value,
                    "sigma_hat": sigma_value,
                    "denoised": record["denoised"],
                }
            )

    result: torch.Tensor = integrate_adaptive_pc(
        denoise_fn,
        x,
        [float(value) for value in sigmas],
        config,
        on_step=on_step,
    )
    progress.close()
    return result
