import importlib.util
import pathlib
import sys

COMFY_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMFY_ROOT))

import torch  # noqa: E402

import comfy.k_diffusion.sampling as kds  # noqa: E402
import comfy.model_sampling  # noqa: E402

PACKAGE_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "adaptive_samplers",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules["adaptive_samplers"] = package
spec.loader.exec_module(package)

from adaptive_samplers.stochastic import sample_adaptive_sa  # noqa: E402

SHAPE: tuple[int, int, int, int] = (1, 16, 32, 32)
SIGMA_FLOOR: float = 1e-4


class FlowSampling(
    comfy.model_sampling.ModelSamplingDiscreteFlow,
    comfy.model_sampling.CONST,
):
    pass


class Patcher:
    def __init__(self, sampling: FlowSampling) -> None:
        self.sampling: FlowSampling = sampling

    def get_model_object(self, name: str) -> FlowSampling:
        return self.sampling


class Inner:
    def __init__(self, sampling: FlowSampling) -> None:
        self.model_patcher: Patcher = Patcher(sampling)
        self.model_sampling: FlowSampling = sampling


class FlowModel:
    def __init__(self) -> None:
        self.sampling: FlowSampling = FlowSampling()
        self.inner_model: Inner = Inner(self.sampling)
        self.calls: int = 0

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.calls += 1
        level: torch.Tensor = sigma.reshape(-1, 1, 1, 1).clamp(min=SIGMA_FLOOR)
        return torch.tanh((1.0 - level) * x / level**2)


def flow_sigmas(steps: int) -> torch.Tensor:
    ramp: torch.Tensor = torch.linspace(1.0, 0.0, steps + 1)
    return ramp.clone()


def fixed_noise_sampler(seed: int) -> object:
    generator = torch.Generator().manual_seed(seed)

    def sampler(sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
        return torch.randn(SHAPE, generator=generator)

    return sampler


def run(function: object, steps: int, **kwargs: object) -> tuple[torch.Tensor, int]:
    model = FlowModel()
    sigmas: torch.Tensor = flow_sigmas(steps)
    noise: torch.Tensor = torch.randn(SHAPE, generator=torch.Generator().manual_seed(1))
    result = function(
        model,
        noise,
        sigmas,
        extra_args={},
        callback=None,
        disable=True,
        noise_sampler=fixed_noise_sampler(5),
        **kwargs,
    )
    return result, model.calls


def structure_agreement(result: torch.Tensor, truth: torch.Tensor) -> float:
    return float((torch.sign(result) == torch.sign(truth)).to(torch.float32).mean())


def texture_energy(result: torch.Tensor) -> float:
    high_pass: torch.Tensor = result[..., 1:, :] - result[..., :-1, :]
    return float(torch.sqrt(torch.mean(high_pass**2)))


if __name__ == "__main__":
    steps: int = 12
    print(f"per-coordinate rademacher prior under flow matching, {steps} steps\n")

    truth, _ = run(sample_adaptive_sa, 200, eta=0.0)

    reference, reference_calls = run(kds.sample_sa_solver, steps)
    neutral, neutral_calls = run(sample_adaptive_sa, steps, response=0.0)
    gap: float = float((reference - neutral).abs().max())
    print(f"sa_solver            nfe={reference_calls}")
    print(f"adaptive response=0  nfe={neutral_calls}")
    print(f"max absolute difference against sa_solver: {gap:.3e}")
    equivalent: bool = gap == 0.0 and reference_calls == neutral_calls
    print(f"exact reproduction of sa_solver: {'pass' if equivalent else 'FAIL'}")

    ode, _ = run(sample_adaptive_sa, steps, eta=0.0)
    print(f"\nreference ode at 200 steps, texture energy {texture_energy(truth):.4f}")
    print(f"{'variant':>20} | {'nfe':>4} | {'structure':>9} | {'texture':>8} | {'rms err':>9}")
    print("-" * 62)

    rows: list[tuple[str, dict[str, object]]] = [
        ("ode eta=0", {"eta": 0.0}),
        ("sa_solver fixed tau", {"response": 0.0}),
        ("response -1.0", {"response": -1.0}),
        ("response -0.5", {"response": -0.5}),
        ("response -0.25", {"response": -0.25}),
        ("response +0.25", {"response": 0.25}),
        ("response +0.5", {"response": 0.5}),
        ("response +1.0", {"response": 1.0}),
    ]
    for label, options in rows:
        result, calls = run(sample_adaptive_sa, steps, **options)
        error: float = float(torch.sqrt(torch.mean((result - truth) ** 2)))
        print(
            f"{label:>20} | {calls:4d} | {structure_agreement(result, truth):9.4f}"
            f" | {texture_energy(result):8.4f} | {error:9.4f}"
        )

    print("\nnfe accounting at one call per step")
    for budget in [8, 10, 12, 14]:
        _, calls = run(sample_adaptive_sa, budget, response=0.5)
        status: str = "ok" if calls == budget else "FAIL"
        print(f"  steps={budget:3d} -> nfe={calls:3d}  {status}")
