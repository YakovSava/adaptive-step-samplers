import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request

SERVER: str = "http://127.0.0.1:8188"
WORKFLOW: pathlib.Path = pathlib.Path(__file__).resolve().parent / "probe_variance.json"

PROMPTS: list[tuple[str, str]] = [
    ("flat", "1girl, flat color illustration, plain pastel background, simple shading"),
    ("city", "1girl standing in a crowded neon city street at night, detailed background"),
    ("face", "extreme close-up portrait of a girl's face, soft skin, detailed eyes"),
    ("fur", "dense forest foliage, fur texture, thousands of leaves"),
    ("dark", "dark dimly lit room, single candle, deep shadows, low contrast"),
    ("bright", "overexposed white studio, backlit silhouette, blown highlights"),
    ("grid", "checkerboard tiled floor, perspective grid, geometric architecture"),
    ("group", "three girls group photo, full body, complex overlapping composition"),
]

TRIGGER: str = "@firedotinc"
SEEDS: list[int] = [42, 1337]
CFG_VALUES: list[float] = [1.0, 3.5, 5.0, 8.0]


def post_prompt(graph: dict[str, object]) -> str:
    payload: bytes = json.dumps({"prompt": graph}).encode("utf-8")
    request = urllib.request.Request(
        f"{SERVER}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())["prompt_id"]


def wait_for(prompt_id: str, timeout: float) -> bool:
    deadline: float = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{SERVER}/history/{prompt_id}") as response:
            history = json.loads(response.read())
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            return status.get("status_str") != "error"
        time.sleep(1.0)
    return False


def build_graph(
    template: dict[str, dict[str, object]],
    prompt: str,
    tag: str,
    seed: int,
    cfg: float,
    steps: int,
    scheduler: str,
    sampler: str,
) -> dict[str, dict[str, object]]:
    graph = json.loads(json.dumps(template))
    graph["37"]["inputs"]["text"] = f"{TRIGGER}, {prompt}"
    graph["51"]["inputs"]["steps"] = steps
    graph["51"]["inputs"]["scheduler"] = scheduler
    graph["52"]["inputs"]["sampler_name"] = sampler
    graph["53"]["inputs"]["tag"] = tag
    graph["54"]["inputs"]["noise_seed"] = seed
    graph["54"]["inputs"]["cfg"] = cfg
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--scheduler", default="normal")
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    template: dict[str, dict[str, object]] = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    jobs: list[tuple[str, str, int, float]] = []
    for index, (name, prompt) in enumerate(PROMPTS):
        for seed in SEEDS:
            for cfg in CFG_VALUES:
                tag: str = f"p{index:02d}{name}_s{seed}_c{round(cfg * 10):03d}"
                jobs.append((tag, prompt, seed, cfg))

    print(f"{len(jobs)} runs, sampler={arguments.sampler} "
          f"scheduler={arguments.scheduler} steps={arguments.steps}")
    if arguments.dry_run:
        for tag, prompt, seed, cfg in jobs[:6]:
            print(f"  {tag:34s} cfg={cfg:<4} seed={seed}  {prompt[:44]}")
        print("  ...")
        return

    started: float = time.time()
    failures: list[str] = []
    for position, (tag, prompt, seed, cfg) in enumerate(jobs, start=1):
        graph = build_graph(
            template,
            prompt,
            tag,
            seed,
            cfg,
            arguments.steps,
            arguments.scheduler,
            arguments.sampler,
        )
        try:
            prompt_id: str = post_prompt(graph)
        except urllib.error.HTTPError as error:
            print(f"  [{position}/{len(jobs)}] {tag}: rejected {error.read()[:200]!r}")
            failures.append(tag)
            continue
        ok: bool = wait_for(prompt_id, arguments.timeout)
        elapsed: float = time.time() - started
        state: str = "ok" if ok else "FAILED"
        print(f"  [{position}/{len(jobs)}] {tag:34s} {state}  elapsed {elapsed / 60:.1f} min")
        if not ok:
            failures.append(tag)

    print(f"\ndone in {(time.time() - started) / 60:.1f} min, {len(failures)} failures")
    for tag in failures:
        print(f"  failed: {tag}")


if __name__ == "__main__":
    main()
