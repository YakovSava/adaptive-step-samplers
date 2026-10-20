from typing import Any, Callable

import torch

import comfy.k_diffusion.sa_solver as sa_solver
from comfy.k_diffusion.sampling import (
    default_noise_sampler,
    offset_first_sigma_for_snr,
    sigma_to_half_log_snr,
)
from comfy.utils import model_trange as trange

from .controller import NoiseLevelController
from .integrator import relative_error

STOCHASTIC_START: float = 0.2
STOCHASTIC_END: float = 0.8


@torch.no_grad()
def sample_adaptive_sa(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Callable[..., Any] | None = None,
    disable: bool | None = None,
    noise_sampler: Callable[..., torch.Tensor] | None = None,
    s_noise: float = 1.0,
    predictor_order: int = 3,
    corrector_order: int = 4,
    simple_order_2: bool = False,
    eta: float = 1.0,
    stochastic_start: float = STOCHASTIC_START,
    stochastic_end: float = STOCHASTIC_END,
    response: float = 0.0,
    spread: float = 3.0,
    smoothing: float = 0.5,
    tau_max: float = 2.0,
    rtol: float = 0.05,
    atol: float = 0.0078,
    tau_observer: Callable[[int, float], None] | None = None,
) -> torch.Tensor:
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in: torch.Tensor = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    lambdas: torch.Tensor = sigma_to_half_log_snr(sigmas, model_sampling=model_sampling)

    base_tau: Callable[[Any], float] = sa_solver.get_tau_interval_func(
        model_sampling.percent_to_sigma(stochastic_start),
        model_sampling.percent_to_sigma(stochastic_end),
        eta=eta,
    )
    controller: NoiseLevelController = NoiseLevelController(
        response=response,
        spread=spread,
        smoothing=smoothing,
        maximum=tau_max,
    )

    max_used_order: int = max(predictor_order, corrector_order)
    x_pred: torch.Tensor = x
    h: float = 0.0
    tau_t: float = 0.0
    noise: Any = 0.0
    pred_list: list[torch.Tensor] = []
    lower_order_to_end: bool = sigmas[-1].item() == 0

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised: torch.Tensor = model(x_pred, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x_pred,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        pred_list.append(denoised)
        pred_list = pred_list[-max_used_order:]

        predictor_order_used: int = min(predictor_order, len(pred_list))
        if i == 0 or sigmas[i + 1] == 0:
            corrector_order_used: int = 0
        else:
            corrector_order_used = min(corrector_order, len(pred_list))

        if lower_order_to_end:
            predictor_order_used = min(predictor_order_used, len(sigmas) - 2 - i)
            corrector_order_used = min(corrector_order_used, len(sigmas) - 1 - i)

        drift_error: float | None = None
        if corrector_order_used == 0:
            x = x_pred
        else:
            curr_lambdas: torch.Tensor = lambdas[i - corrector_order_used + 1:i + 1]
            b_coeffs: torch.Tensor = sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i],
                curr_lambdas,
                lambdas[i - 1],
                lambdas[i],
                tau_t,
                simple_order_2,
                is_corrector_step=True,
            )
            pred_mat: torch.Tensor = torch.stack(pred_list[-corrector_order_used:], dim=1)
            corr_res: torch.Tensor = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))
            x = sigmas[i] / sigmas[i - 1] * (-(tau_t**2) * h).exp() * x + corr_res

            if tau_t > 0 and s_noise > 0:
                x = x + noise

            drift_error = relative_error(x_pred - x, x, rtol, atol)

        if sigmas[i + 1] == 0:
            x_pred = denoised
        else:
            tau_t = controller.level(base_tau(sigmas[i + 1]), drift_error)
            if tau_observer is not None:
                tau_observer(i, tau_t)
            curr_lambdas = lambdas[i - predictor_order_used + 1:i + 1]
            b_coeffs = sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i + 1],
                curr_lambdas,
                lambdas[i],
                lambdas[i + 1],
                tau_t,
                simple_order_2,
                is_corrector_step=False,
            )
            pred_mat = torch.stack(pred_list[-predictor_order_used:], dim=1)
            pred_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))
            h = lambdas[i + 1] - lambdas[i]
            x_pred = sigmas[i + 1] / sigmas[i] * (-(tau_t**2) * h).exp() * x + pred_res

            if tau_t > 0 and s_noise > 0:
                noise = (
                    noise_sampler(sigmas[i], sigmas[i + 1])
                    * sigmas[i + 1]
                    * (-2 * tau_t**2 * h).expm1().neg().sqrt()
                    * s_noise
                )
                x_pred = x_pred + noise

    return x_pred
