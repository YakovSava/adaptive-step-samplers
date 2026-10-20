from typing import Any, Callable

import comfy.k_diffusion.sampling as k_diffusion_sampling
import comfy.samplers

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .samplers import sample_adaptive_pc

PRESETS: dict[str, dict[str, Any]] = {
    "apc_2": {"order": 2, "mode": "budget", "use_corrector": True},
    "apc_3": {"order": 3, "mode": "budget", "use_corrector": True},
    "apc_2_pred": {"order": 2, "mode": "budget", "use_corrector": False},
    "apc_2_fixed": {"order": 2, "mode": "fixed", "use_corrector": True},
    "apc_3_fixed": {"order": 3, "mode": "fixed", "use_corrector": True},
    "apc_tol_2": {"order": 2, "mode": "tolerance", "use_corrector": True},
}


def make_preset_sampler(name: str, options: dict[str, Any]) -> Callable[..., Any]:
    def preset_sampler(
        model: Any,
        x: Any,
        sigmas: Any,
        extra_args: dict[str, Any] | None = None,
        callback: Any = None,
        disable: bool | None = None,
        **overrides: Any,
    ) -> Any:
        merged: dict[str, Any] = dict(options)
        merged.update(overrides)
        return sample_adaptive_pc(
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            **merged,
        )

    preset_sampler.__name__ = f"sample_{name}"
    return preset_sampler


def register_presets() -> None:
    for name, options in PRESETS.items():
        setattr(k_diffusion_sampling, f"sample_{name}", make_preset_sampler(name, options))
        if name not in comfy.samplers.KSAMPLER_NAMES:
            comfy.samplers.KSAMPLER_NAMES.append(name)
        if name not in comfy.samplers.SAMPLER_NAMES:
            comfy.samplers.SAMPLER_NAMES.append(name)


register_presets()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
