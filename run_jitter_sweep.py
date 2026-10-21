import argparse
import importlib.util
import json
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sde_test", ROOT / "run_sde_test.py")
sde = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sde)

STEPS: int = 10
ETAS: list[float] = [1.0, 1.4, 1.8]
WINDOWS: list[tuple[float, float]] = [(0.2, 0.8), (0.1, 0.9), (0.05, 0.95)]


def build(
    template: dict[str, Any],
    prompt: str,
    tag: str,
    seed: int,
    eta: float,
    window: tuple[float, float],
) -> dict[str, Any]:
    graph = json.loads(json.dumps(template))
    graph["37"]["inputs"]["text"] = f"{sde.TRIGGER}, {prompt}"
    graph["51"]["inputs"]["steps"] = STEPS
    graph["53"]["inputs"]["tag"] = tag
    graph["54"]["inputs"]["noise_seed"] = seed
    graph["54"]["inputs"]["cfg"] = sde.CFG
    graph["12"]["inputs"]["filename_prefix"] = "exp/jit_"
    graph["52"] = {
        "class_type": "AdaptiveSASampler",
        "inputs": {
            "predictor_order": 3,
            "corrector_order": 4,
            "eta": eta,
            "stochastic_start": window[0],
            "stochastic_end": window[1],
            "response": 0.0,
            "spread": 3.0,
            "smoothing": 0.5,
            "tau_max": 4.0,
            "s_noise": 1.0,
            "simple_order_2": False,
        },
    }
    return graph


def tag_for(name: str, seed: int, eta: float, window: tuple[float, float]) -> str:
    return f"jit_{name}_s{seed}_e{round(eta * 10):02d}_w{round(window[0] * 100):02d}"


def texture_reference() -> dict[str, float]:
    reference: dict[str, float] = {}
    for name, _ in sde.PROMPTS:
        for seed in sde.SEEDS:
            values: list[float] = []
            for sampler in sde.STOCHASTIC:
                latent = sde.latent_of(
                    sde.tag_for(name, seed, sde.TEXTURE_REFERENCE_STEPS, sampler)
                )
                if latent is not None:
                    values.append(sde.highpass_energy(latent))
            if values:
                reference[f"{name}_{seed}"] = float(np.mean(values))
    return reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyse-only", action="store_true")
    arguments = parser.parse_args()

    template: dict[str, Any] = json.loads(sde.WORKFLOW.read_text(encoding="utf-8"))
    jobs: list[tuple[str, str, int, float, tuple[float, float]]] = []
    for name, prompt in sde.PROMPTS:
        for seed in sde.SEEDS:
            for eta in ETAS:
                for window in WINDOWS:
                    jobs.append((tag_for(name, seed, eta, window), prompt, seed, eta, window))

    if not arguments.analyse_only:
        print(f"{len(jobs)} runs at {STEPS} nfe, cfg={sde.CFG}")
        started: float = time.time()
        for position, (tag, prompt, seed, eta, window) in enumerate(jobs, start=1):
            graph = build(template, prompt, tag, seed, eta, window)
            try:
                prompt_id = sde.post(graph)
            except urllib.error.HTTPError as error:
                print(f"  [{position}/{len(jobs)}] {tag}: rejected {error.read()[:160]!r}")
                continue
            if not sde.wait_for(prompt_id):
                print(f"  [{position}/{len(jobs)}] {tag}: FAILED")
            if position % 20 == 0:
                print(f"  [{position}/{len(jobs)}] {(time.time() - started) / 60:.1f} min", flush=True)
        print(f"\nsampling done in {(time.time() - started) / 60:.1f} min\n")

    reference = texture_reference()
    target: float = float(np.mean(list(reference.values())))
    print(f"converged sde texture target: {target:.4f}")

    baseline_macro: list[float] = []
    for name, _ in sde.PROMPTS:
        for seed in sde.SEEDS:
            value = sde.latent_of(sde.tag_for(name, seed, STEPS, "sa_solver"))
            geometry = sde.latent_of(f"sdegeo_{name}_s{seed}")
            if value is not None and geometry is not None:
                baseline_macro.append(sde.macro_error(value, geometry))
    print(f"sa_solver baseline at {STEPS} nfe: macro {np.mean(baseline_macro):.4f}\n")

    print(f"{'eta':>5} {'window':>12} | {'macro err':>9} | {'texture':>8} | {'gap':>7} | {'vs base':>8}")
    print("-" * 62)
    for eta in ETAS:
        for window in WINDOWS:
            macro: list[float] = []
            texture: list[float] = []
            for name, _ in sde.PROMPTS:
                for seed in sde.SEEDS:
                    value = sde.latent_of(tag_for(name, seed, eta, window))
                    geometry = sde.latent_of(f"sdegeo_{name}_s{seed}")
                    if value is None or geometry is None:
                        continue
                    macro.append(sde.macro_error(value, geometry))
                    texture.append(sde.highpass_energy(value))
            if not macro:
                continue
            gap: float = np.mean(texture) - target
            ratio: float = np.mean(macro) / np.mean(baseline_macro)
            label = f"{window[0]:.2f}-{window[1]:.2f}"
            print(f"{eta:5.1f} {label:>12} | {np.mean(macro):9.4f} | "
                  f"{np.mean(texture):8.4f} | {gap:+7.4f} | {ratio:7.3f}x")
        print()


if __name__ == "__main__":
    main()
