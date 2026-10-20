import importlib.util
import pathlib
import sys
from typing import Callable

COMFY_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMFY_ROOT))

import torch  # noqa: E402

import comfy.samplers  # noqa: E402

PACKAGE_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "adaptive_samplers",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules["adaptive_samplers"] = package
spec.loader.exec_module(package)

from adaptive_samplers.probe import TrajectoryRecorder, wrap_sampler  # noqa: E402

SHAPE: tuple[int, int, int, int] = (1, 16, 32, 32)
MODE_COUNT: int = 6
MODE_SCALE: float = 3.0
MODES: torch.Tensor = torch.randn(
    (MODE_COUNT,) + SHAPE[1:], generator=torch.Generator().manual_seed(3)
) * (MODE_SCALE / (SHAPE[1] * SHAPE[2] * SHAPE[3]) ** 0.5)


def fake_model(x: torch.Tensor, sigma: torch.Tensor, **kwargs: object) -> torch.Tensor:
    level: float = float(sigma.flatten()[0])
    distance: torch.Tensor = ((MODES - x) ** 2).flatten(1).sum(dim=1)
    logits: torch.Tensor = -0.5 * distance / max(level, 1e-5) ** 2
    posterior: torch.Tensor = torch.softmax(logits, dim=0).reshape(-1, 1, 1, 1)
    return (posterior * MODES).sum(dim=0, keepdim=True)


def make_sigmas(steps: int) -> torch.Tensor:
    ramp: torch.Tensor = torch.linspace(0.0, 1.0, steps)
    high: float = 10.0 ** (1.0 / 7.0)
    low: float = 0.02 ** (1.0 / 7.0)
    sigmas: torch.Tensor = (high + ramp * (low - high)) ** 7.0
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def run_sampler(name: str, steps: int) -> tuple[torch.Tensor, int]:
    sampler = comfy.samplers.sampler_object(name)
    sigmas: torch.Tensor = make_sigmas(steps)
    generator = torch.Generator().manual_seed(1)
    noise: torch.Tensor = torch.randn(SHAPE, generator=generator) * sigmas[0]
    calls: dict[str, int] = {"count": 0}

    def counting_model(x: torch.Tensor, sigma: torch.Tensor, **kwargs: object) -> torch.Tensor:
        calls["count"] += 1
        return fake_model(x, sigma, **kwargs)

    result = sampler.sampler_function(
        counting_model, noise, sigmas, extra_args={}, callback=None, disable=True
    )
    return result, calls["count"]


EXPECTED_NFE: dict[str, Callable[[int], bool]] = {
    "euler": lambda steps, used: used == steps,
    "heun": lambda steps, used: used == 2 * steps - 1,
    "ipndm": lambda steps, used: used == steps,
    "deis": lambda steps, used: used == steps,
    "dpmpp_2m": lambda steps, used: used == steps,
    "apc_2_fixed": lambda steps, used: used == steps,
    "apc_2_pred": lambda steps, used: used == steps,
    "apc_2": lambda steps, used: used == steps,
    "apc_3": lambda steps, used: used == steps,
}


if __name__ == "__main__":
    reference, _ = run_sampler("euler", 400)
    steps: int = 12

    print(f"torch path check on {SHAPE} tensors, {steps} requested steps")
    print("the synthetic mixture denoiser collapses to one mode in this many")
    print("dimensions, so the deviation column is diagnostic only, not a benchmark\n")
    print(f"{'sampler':>14} | {'nfe':>4} | {'budget':>7} | {'deviation':>10}")
    print("-" * 44)
    ok: bool = True
    for name, expectation in EXPECTED_NFE.items():
        result, used = run_sampler(name, steps)
        deviation: float = float(torch.sqrt(torch.mean((result - reference) ** 2)))
        passed: bool = expectation(steps, used)
        ok = ok and passed and torch.isfinite(result).all().item()
        print(f"{name:>14} | {used:4d} | {'ok' if passed else 'FAIL':>7} | {deviation:10.3e}")

    print(f"\nnfe accounting and finite output: {'pass' if ok else 'FAIL'}")
    print("\ntrajectory probe over euler")
    probe = wrap_sampler(
        comfy.samplers.sampler_object("euler"),
        lambda: TrajectoryRecorder("smoke", order=2, latent_stride=4),
    )
    sigmas: torch.Tensor = make_sigmas(steps)
    generator = torch.Generator().manual_seed(1)
    noise: torch.Tensor = torch.randn(SHAPE, generator=generator) * sigmas[0]
    probe.sampler_function(fake_model, noise, sigmas, extra_args={}, callback=None, disable=True)
    print("  dump written under the output exp directory")
