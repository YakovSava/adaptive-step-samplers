import json
import os
from typing import Any, Callable

import torch

import comfy.samplers
import folder_paths
from comfy.utils import save_torch_file

from .integrator import combine, relative_error
from .pc_core import corrector_order, corrector_weights, milne_factor, predictor_weights

EXPERIMENT_SUBDIR: str = "exp"
SIGMA_FLOOR: float = 1e-5


def rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.to(torch.float32) ** 2)))


class TrajectoryRecorder:
    def __init__(
        self,
        tag: str,
        order: int = 2,
        rtol: float = 0.05,
        atol: float = 0.0078,
        latent_stride: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.tag: str = tag
        self.order: int = order
        self.rtol: float = rtol
        self.atol: float = atol
        self.latent_stride: int = latent_stride
        self.metadata: dict[str, Any] = {} if metadata is None else metadata
        self.sigmas: list[float] = []
        self.state_rms: list[float] = []
        self.denoised_rms: list[float] = []
        self.derivative_rms: list[float] = []
        self.curvature: list[float] = []
        self.pc_error: list[float] = []
        self.pc_order: list[float] = []
        self.latents: list[torch.Tensor] = []
        self.latent_indices: list[float] = []
        self.derivatives: list[torch.Tensor] = []
        self.steps: list[float] = []
        self.previous_state: torch.Tensor | None = None
        self.calls: int = 0

    def observe(self, x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> None:
        sigma_value: float = float(sigma.flatten()[0])
        sigma_eval: float = max(sigma_value, SIGMA_FLOOR)
        derivative: torch.Tensor = (x - denoised) / sigma_eval

        self.sigmas.append(sigma_value)
        self.state_rms.append(rms(x))
        self.denoised_rms.append(rms(denoised))
        self.derivative_rms.append(rms(derivative))

        last_step: float = 0.0
        if len(self.sigmas) >= 2:
            last_step = self.sigmas[-1] - self.sigmas[-2]
        usable: bool = last_step != 0.0 and self.previous_state is not None

        if self.derivatives and usable:
            self.curvature.append(rms(derivative - self.derivatives[0]) / abs(last_step))
        else:
            self.curvature.append(0.0)

        if usable:
            order_used: int = corrector_order(self.order, self.steps)
            predicted: torch.Tensor = combine(
                self.previous_state,
                predictor_weights(last_step, self.steps, self.order),
                self.derivatives,
            )
            corrected: torch.Tensor = combine(
                self.previous_state,
                corrector_weights(last_step, self.steps, self.order),
                [derivative] + self.derivatives,
            )
            residual: torch.Tensor = (predicted - corrected) * milne_factor(order_used, False)
            self.pc_error.append(relative_error(residual, x, self.rtol, self.atol))
            self.pc_order.append(float(order_used))
        else:
            self.pc_error.append(0.0)
            self.pc_order.append(0.0)

        if self.latent_stride > 0 and self.calls % self.latent_stride == 0:
            self.latents.append(x.detach().to(torch.float16).cpu())
            self.latent_indices.append(float(self.calls))

        if usable:
            self.steps.insert(0, last_step)
        self.derivatives.insert(0, derivative)
        del self.derivatives[self.order:]
        del self.steps[self.order:]
        self.previous_state = x
        self.calls += 1

    def dump(self, sigmas: torch.Tensor, result: torch.Tensor) -> str:
        payload: dict[str, torch.Tensor] = {
            "sigma": torch.tensor(self.sigmas, dtype=torch.float32),
            "state_rms": torch.tensor(self.state_rms, dtype=torch.float32),
            "denoised_rms": torch.tensor(self.denoised_rms, dtype=torch.float32),
            "derivative_rms": torch.tensor(self.derivative_rms, dtype=torch.float32),
            "curvature": torch.tensor(self.curvature, dtype=torch.float32),
            "pc_error": torch.tensor(self.pc_error, dtype=torch.float32),
            "pc_order": torch.tensor(self.pc_order, dtype=torch.float32),
            "schedule": sigmas.detach().to(torch.float32).cpu(),
            "final_latent": result.detach().to(torch.float32).cpu(),
        }
        if self.latents:
            payload["latent_stack"] = torch.stack(self.latents)
            payload["latent_index"] = torch.tensor(self.latent_indices, dtype=torch.float32)

        info: dict[str, str] = {
            "tag": self.tag,
            "nfe": str(self.calls),
            "order": str(self.order),
            "rtol": str(self.rtol),
            "atol": str(self.atol),
        }
        for key, value in self.metadata.items():
            info[key] = str(value)

        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            os.path.join(EXPERIMENT_SUBDIR, self.tag),
            folder_paths.get_output_directory(),
        )
        path: str = os.path.join(folder, f"{filename}_{counter:05}_.safetensors")
        save_torch_file(payload, path, metadata={"trajectory": json.dumps(info)})
        return path


class RecordingModel:
    def __init__(self, inner: Any, recorder: TrajectoryRecorder) -> None:
        self.inner: Any = inner
        self.recorder: TrajectoryRecorder = recorder

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        denoised: torch.Tensor = self.inner(x, sigma, **kwargs)
        self.recorder.observe(x, sigma, denoised)
        return denoised

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["inner"], name)


def wrap_sampler(
    inner: comfy.samplers.KSAMPLER,
    recorder_factory: Callable[[], TrajectoryRecorder],
) -> comfy.samplers.KSAMPLER:
    inner_options: dict[str, Any] = dict(inner.extra_options)

    def probe_function(
        model: Any,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: dict[str, Any] | None = None,
        callback: Callable[..., Any] | None = None,
        disable: bool | None = None,
    ) -> torch.Tensor:
        recorder: TrajectoryRecorder = recorder_factory()
        result: torch.Tensor = inner.sampler_function(
            RecordingModel(model, recorder),
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            **inner_options,
        )
        recorder.dump(sigmas, result)
        return result

    return comfy.samplers.KSAMPLER(probe_function, {}, inner.inpaint_options)
