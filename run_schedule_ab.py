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
    ("city", "1girl standing in a crowded neon city street at night, detailed background"),
    ("face", "extreme close-up portrait of a girl's face, soft skin, detailed eyes"),
    ("grid", "checkerboard tiled floor, perspective grid, geometric architecture"),
]
CFG_VALUES: list[float] = [1.0, 8.0]
BUDGETS: list[int] = [6, 8, 10, 12]
SEED: int = 42
REFERENCE_STEPS: int = 100
ORDER: int = 2


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


def measured_profiles() -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    pattern = __import__("re").compile(r"^p(\d+)([a-z]+)_s(\d+)_c(\d+)$")
    grouped: dict[float, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for path in DUMPS.glob("*.safetensors"):
        with safe_open(str(path), "pt") as handle:
            info = json.loads((handle.metadata() or {}).get("trajectory", "{}"))
            match = pattern.match(info.get("tag", ""))
            if match is None:
                continue
            sigma = handle.get_tensor("sigma").numpy().astype(np.float64)
            if sigma.size != 24:
                continue
            grouped.setdefault(int(match.group(4)) / 10.0, []).append(
                (
                    sigma,
                    handle.get_tensor("pc_error").numpy().astype(np.float64),
                    handle.get_tensor("schedule").numpy().astype(np.float64),
                )
            )
    return {cfg: items[0] if False else items for cfg, items in grouped.items()}


def derive_schedule(samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]], steps: int) -> list[float]:
    sigma: np.ndarray = samples[0][0]
    schedule: np.ndarray = samples[0][2]
    errors: np.ndarray = np.median(np.stack([item[1] for item in samples]), axis=0)
    incoming: np.ndarray = np.abs(np.diff(schedule))
    widths: np.ndarray = np.concatenate([[0.0], incoming])[: errors.size]
    usable: np.ndarray = (errors > 0.0) & (widths > 0.0)
    difficulty: np.ndarray = errors[usable] / widths[usable] ** (ORDER + 1)
    nodes: np.ndarray = sigma[usable]
    density: np.ndarray = difficulty ** (1.0 / (ORDER + 1))
    order_index: np.ndarray = np.argsort(nodes)
    nodes, density = nodes[order_index], density[order_index]
    cumulative: np.ndarray = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(nodes))]
    )
    cumulative = cumulative / cumulative[-1]
    grid: np.ndarray = np.interp(np.linspace(1.0, 0.0, steps), cumulative, nodes)
    grid[0] = float(schedule[0])
    grid[-1] = float(schedule[-2])
    return [float(value) for value in grid] + [float(schedule[-1])]


def static_schedule(name: str, steps: int, reference: np.ndarray) -> list[float] | None:
    if name == "normal":
        return None
    if name == "karras":
        high: float = float(reference[0]) ** (1.0 / 7.0)
        low: float = float(reference[-2]) ** (1.0 / 7.0)
        ramp = np.linspace(0.0, 1.0, steps)
        values = (high + ramp * (low - high)) ** 7.0
        return [float(v) for v in values] + [0.0]
    raise ValueError(name)


def build(
    template: dict[str, Any],
    prompt: str,
    tag: str,
    cfg: float,
    steps: int,
    sigmas: list[float] | None,
    sampler: str,
) -> dict[str, Any]:
    graph = json.loads(json.dumps(template))
    graph["37"]["inputs"]["text"] = f"{TRIGGER}, {prompt}"
    graph["52"]["inputs"]["sampler_name"] = sampler
    graph["53"]["inputs"]["tag"] = tag
    graph["54"]["inputs"]["noise_seed"] = SEED
    graph["54"]["inputs"]["cfg"] = cfg
    graph["12"]["inputs"]["filename_prefix"] = "exp/ab_"
    if sigmas is None:
        graph["51"]["inputs"]["steps"] = steps
    else:
        graph["55"] = {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": ", ".join(f"{v:.6f}" for v in sigmas)},
        }
        graph["54"]["inputs"]["sigmas"] = ["55", 0]
        del graph["51"]
    return graph


def final_latent(tag: str) -> np.ndarray | None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    template: dict[str, Any] = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    profiles = measured_profiles()
    print(f"measured profiles available for cfg: {sorted(profiles)}")

    derived: dict[tuple[float, int], list[float]] = {}
    for cfg in CFG_VALUES:
        for steps in BUDGETS:
            derived[(cfg, steps)] = derive_schedule(profiles[cfg], steps)

    reference_sigma: np.ndarray = profiles[CFG_VALUES[0]][0][2]
    jobs: list[tuple[str, str, float, int, list[float] | None, str]] = []
    for name, prompt in PROMPTS:
        for cfg in CFG_VALUES:
            jobs.append(
                (f"ab_{name}_c{round(cfg * 10):03d}_ref", prompt, cfg, REFERENCE_STEPS, None, "euler")
            )
    for name, prompt in PROMPTS:
        for cfg in CFG_VALUES:
            for steps in BUDGETS:
                for label in ["normal", "karras", "derived"]:
                    if label == "derived":
                        sigmas = derived[(cfg, steps)]
                    else:
                        sigmas = static_schedule(label, steps, reference_sigma)
                    tag = f"ab_{name}_c{round(cfg * 10):03d}_n{steps:02d}_{label}"
                    jobs.append((tag, prompt, cfg, steps, sigmas, arguments.sampler))

    print(f"{len(jobs)} runs total ({len(PROMPTS) * len(CFG_VALUES)} references)")
    if arguments.dry_run:
        for cfg in CFG_VALUES:
            print(f"  derived cfg={cfg} n=10: "
                  + " ".join(f"{v:.3f}" for v in derived[(cfg, 10)]))
        return

    started: float = time.time()
    for position, (tag, prompt, cfg, steps, sigmas, sampler) in enumerate(jobs, start=1):
        graph = build(template, prompt, tag, cfg, steps, sigmas, sampler)
        try:
            prompt_id = post(graph)
        except urllib.error.HTTPError as error:
            print(f"  [{position}/{len(jobs)}] {tag}: rejected {error.read()[:200]!r}")
            continue
        ok = wait_for(prompt_id)
        if position % 12 == 0 or not ok:
            print(f"  [{position}/{len(jobs)}] {tag:42s} "
                  f"{'ok' if ok else 'FAILED'}  {(time.time() - started) / 60:.1f} min")

    print(f"\nsampling done in {(time.time() - started) / 60:.1f} min\n")

    print(f"{'cfg':>5} {'nfe':>4} | " + " | ".join(f"{label:>10}" for label in
                                                   ["normal", "karras", "derived"]))
    print("-" * 52)
    for cfg in CFG_VALUES:
        for steps in BUDGETS:
            row: list[str] = []
            for label in ["normal", "karras", "derived"]:
                errors: list[float] = []
                for name, _ in PROMPTS:
                    truth = final_latent(f"ab_{name}_c{round(cfg * 10):03d}_ref")
                    value = final_latent(f"ab_{name}_c{round(cfg * 10):03d}_n{steps:02d}_{label}")
                    if truth is None or value is None:
                        continue
                    errors.append(float(np.sqrt(np.mean((value - truth) ** 2))))
                row.append(f"{np.mean(errors):.4f}" if errors else "n/a")
            print(f"{cfg:5.1f} {steps:4d} | " + " | ".join(f"{value:>10}" for value in row))


if __name__ == "__main__":
    main()
