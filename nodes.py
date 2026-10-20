import functools
import json
from typing import Any

import comfy.samplers

from .integrator import MAX_ORDER, MODES
from .probe import TrajectoryRecorder, wrap_sampler
from .samplers import sample_adaptive_pc
from .stochastic import sample_adaptive_sa


class AdaptivePCSampler:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "order": ("INT", {"default": 2, "min": 1, "max": MAX_ORDER}),
                "mode": (MODES, {"default": MODES[0]}),
                "use_corrector": ("BOOLEAN", {"default": True}),
                "damping": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "min_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "max_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "spread": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 16.0, "step": 0.25}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rtol": ("FLOAT", {"default": 0.05, "min": 1e-05, "max": 1.0, "step": 0.001}),
                "atol": ("FLOAT", {"default": 0.0078, "min": 1e-06, "max": 1.0, "step": 0.0001}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/adaptive"

    def build(
        self,
        order: int,
        mode: str,
        use_corrector: bool,
        damping: float,
        min_ratio: float,
        max_ratio: float,
        spread: float,
        smoothing: float,
        rtol: float,
        atol: float,
    ) -> tuple[Any]:
        sampler_function = functools.partial(
            sample_adaptive_pc,
            order=order,
            mode=mode,
            use_corrector=use_corrector,
            damping=damping,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            spread=spread,
            smoothing=smoothing,
            rtol=rtol,
            atol=atol,
        )
        return (comfy.samplers.KSAMPLER(sampler_function),)


class AdaptiveSASampler:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "predictor_order": ("INT", {"default": 3, "min": 1, "max": 6}),
                "corrector_order": ("INT", {"default": 4, "min": 0, "max": 6}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "stochastic_start": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "stochastic_end": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "response": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "spread": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 16.0, "step": 0.25}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "tau_max": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "simple_order_2": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/adaptive"

    def build(
        self,
        predictor_order: int,
        corrector_order: int,
        eta: float,
        stochastic_start: float,
        stochastic_end: float,
        response: float,
        spread: float,
        smoothing: float,
        tau_max: float,
        s_noise: float,
        simple_order_2: bool,
    ) -> tuple[Any]:
        sampler_function = functools.partial(
            sample_adaptive_sa,
            predictor_order=predictor_order,
            corrector_order=corrector_order,
            eta=eta,
            stochastic_start=stochastic_start,
            stochastic_end=stochastic_end,
            response=response,
            spread=spread,
            smoothing=smoothing,
            tau_max=tau_max,
            s_noise=s_noise,
            simple_order_2=simple_order_2,
        )
        return (comfy.samplers.KSAMPLER(sampler_function),)


class TrajectoryProbe:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "tag": ("STRING", {"default": "run"}),
                "order": ("INT", {"default": 2, "min": 1, "max": MAX_ORDER}),
                "rtol": ("FLOAT", {"default": 0.05, "min": 1e-05, "max": 1.0, "step": 0.001}),
                "atol": ("FLOAT", {"default": 0.0078, "min": 1e-06, "max": 1.0, "step": 0.0001}),
                "latent_stride": ("INT", {"default": 0, "min": 0, "max": 512}),
            },
            "optional": {
                "note": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/adaptive"

    def build(
        self,
        sampler: Any,
        tag: str,
        order: int,
        rtol: float,
        atol: float,
        latent_stride: int,
        note: str = "",
        prompt: Any = None,
    ) -> tuple[Any]:
        metadata: dict[str, Any] = {"note": note}
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)

        factory = functools.partial(
            TrajectoryRecorder,
            tag,
            order=order,
            rtol=rtol,
            atol=atol,
            latent_stride=latent_stride,
            metadata=metadata,
        )
        return (wrap_sampler(sampler, factory),)


NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "AdaptivePCSampler": AdaptivePCSampler,
    "AdaptiveSASampler": AdaptiveSASampler,
    "TrajectoryProbe": TrajectoryProbe,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "AdaptivePCSampler": "Adaptive PC Sampler",
    "AdaptiveSASampler": "Adaptive SA Sampler",
    "TrajectoryProbe": "Trajectory Probe",
}
