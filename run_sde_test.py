import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from safetensors import safe_open

SERVER: str = "http://127.0.0.1:8188"
ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent
WORKFLOW: pathlib.Path = ROOT / "probe_variance.json"
DUMPS: pathlib.Path = pathlib.Path("D:/ComfyUI/output/exp")

TRIGGER: str = "@firedotinc"
PROMPTS: list[tuple[str, str]] = [
    ("flat", "1girl, flat color illustration, plain pastel background, simple shading"),
    ("face", "extreme close-up portrait of a girl's face, soft skin, detailed eyes"),
    ("fur", "dense forest foliage, fur texture, thousands of leaves"),
    ("grid", "checkerboard tiled floor, perspective grid, geometric architecture"),
]
SEEDS: list[int] = [42, 1337]
CFG: float = 5.0
BUDGETS: list[int] = [8, 10, 12]
TEXTURE_REFERENCE_STEPS: int = 30
GEOMETRY_REFERENCE_STEPS: int = 100
POOL: int = 8

STOCHASTIC: list[str] = [
    "euler_ancestral",
    "sa_solver",
    "res_multistep_ancestral",
    "dpmpp_3m_sde",
    "clyb_4m_sde_momentumized",
    "asa_adaptive",
]
DETERMINISTIC: list[str] = ["euler"]


def post(graph: dict[str, Any]) -> str:
    request = urllib.request.Request(
        f"{SERVER}/prompt",
        data=json.dumps({"prompt": graph}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())["prompt_id"]


def wait_for(prompt_id: str, timeout: float = 1200.0) -> bool:
    deadline: float = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{SERVER}/history/{prompt_id}") as response:
            history = json.loads(response.read())
        if prompt_id in history:
            return history[prompt_id].get("status", {}).get("status_str") != "error"
        time.sleep(0.5)
    return False


def build(
    template: dict[str, Any],
    prompt: str,
    tag: str,
    seed: int,
    steps: int,
    sampler: str,
) -> dict[str, Any]:
    graph = json.loads(json.dumps(template))
    graph["37"]["inputs"]["text"] = f"{TRIGGER}, {prompt}"
    graph["51"]["inputs"]["steps"] = steps
    graph["52"]["inputs"]["sampler_name"] = sampler
    graph["53"]["inputs"]["tag"] = tag
    graph["54"]["inputs"]["noise_seed"] = seed
    graph["54"]["inputs"]["cfg"] = CFG
    graph["12"]["inputs"]["filename_prefix"] = "exp/sde_"
    return graph


def latent_of(tag: str) -> np.ndarray | None:
    best: tuple[float, np.ndarray] | None = None
    for path in DUMPS.glob("*.safetensors"):
        with safe_open(str(path), "pt") as handle:
            info = json.loads((handle.metadata() or {}).get("trajectory", "{}"))
            if info.get("tag") != tag:
                continue
            stamp = path.stat().st_mtime
            if best is None or stamp > best[0]:
                best = (stamp, handle.get_tensor("final_latent").numpy().astype(np.float64))
    return None if best is None else best[1]


def lowpass(x: np.ndarray) -> np.ndarray:
    height, width = x.shape[-2], x.shape[-1]
    trimmed = x[..., : height // POOL * POOL, : width // POOL * POOL]
    shape = trimmed.shape[:-2] + (trimmed.shape[-2] // POOL, POOL, trimmed.shape[-1] // POOL, POOL)
    return trimmed.reshape(shape).mean(axis=(-3, -1))


def highpass_energy(x: np.ndarray) -> float:
    coarse = lowpass(x)
    expanded = np.repeat(np.repeat(coarse, POOL, axis=-2), POOL, axis=-1)
    height, width = expanded.shape[-2], expanded.shape[-1]
    return float(np.sqrt(np.mean((x[..., :height, :width] - expanded) ** 2)))


def macro_error(x: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((lowpass(x) - lowpass(reference)) ** 2)))


def tag_for(name: str, seed: int, steps: int, sampler: str) -> str:
    return f"sde_{name}_s{seed}_n{steps:02d}_{sampler}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyse-only", action="store_true")
    arguments = parser.parse_args()

    template: dict[str, Any] = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    samplers: list[str] = DETERMINISTIC + STOCHASTIC

    jobs: list[tuple[str, str, int, int, str]] = []
    for name, prompt in PROMPTS:
        for seed in SEEDS:
            jobs.append((f"sdegeo_{name}_s{seed}", prompt, seed, GEOMETRY_REFERENCE_STEPS, "euler"))
            for sampler in STOCHASTIC:
                jobs.append(
                    (
                        tag_for(name, seed, TEXTURE_REFERENCE_STEPS, sampler),
                        prompt,
                        seed,
                        TEXTURE_REFERENCE_STEPS,
                        sampler,
                    )
                )
            for steps in BUDGETS:
                for sampler in samplers:
                    jobs.append((tag_for(name, seed, steps, sampler), prompt, seed, steps, sampler))

    if not arguments.analyse_only:
        print(f"{len(jobs)} runs at cfg={CFG}")
        started: float = time.time()
        for position, (tag, prompt, seed, steps, sampler) in enumerate(jobs, start=1):
            graph = build(template, prompt, tag, seed, steps, sampler)
            try:
                prompt_id = post(graph)
            except urllib.error.HTTPError as error:
                print(f"  [{position}/{len(jobs)}] {tag}: rejected {error.read()[:160]!r}")
                continue
            if not wait_for(prompt_id):
                print(f"  [{position}/{len(jobs)}] {tag}: FAILED")
            if position % 25 == 0:
                print(f"  [{position}/{len(jobs)}] {(time.time() - started) / 60:.1f} min", flush=True)
        print(f"\nsampling done in {(time.time() - started) / 60:.1f} min\n")

    reference_texture: dict[str, float] = {}
    for name, _ in PROMPTS:
        for seed in SEEDS:
            values = [
                highpass_energy(latent_of(tag_for(name, seed, TEXTURE_REFERENCE_STEPS, sampler)))
                for sampler in STOCHASTIC
                if latent_of(tag_for(name, seed, TEXTURE_REFERENCE_STEPS, sampler)) is not None
            ]
            if values:
                reference_texture[f"{name}_{seed}"] = float(np.mean(values))

    print(f"converged sde texture reference at {TEXTURE_REFERENCE_STEPS} steps: "
          f"{np.mean(list(reference_texture.values())):.4f}")
    print(f"\n{'nfe':>4} {'sampler':>26} | {'macro err':>9} | {'texture':>8} | {'excess':>8}")
    print("-" * 64)
    for steps in BUDGETS:
        for sampler in samplers:
            macro: list[float] = []
            texture: list[float] = []
            excess: list[float] = []
            for name, _ in PROMPTS:
                for seed in SEEDS:
                    value = latent_of(tag_for(name, seed, steps, sampler))
                    geometry = latent_of(f"sdegeo_{name}_s{seed}")
                    if value is None or geometry is None:
                        continue
                    macro.append(macro_error(value, geometry))
                    energy = highpass_energy(value)
                    texture.append(energy)
                    target = reference_texture.get(f"{name}_{seed}")
                    if target is not None:
                        excess.append(energy - target)
            if not macro:
                continue
            print(f"{steps:4d} {sampler:>26} | {np.mean(macro):9.4f} | "
                  f"{np.mean(texture):8.4f} | {np.mean(excess):+8.4f}")
        print()


if __name__ == "__main__":
    main()
