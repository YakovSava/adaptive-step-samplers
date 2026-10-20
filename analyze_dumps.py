import argparse
import json
import pathlib
import re
from typing import Any

import numpy as np
from safetensors import safe_open

TAG_PATTERN: re.Pattern[str] = re.compile(r"^p(\d+)([a-z]+)_s(\d+)_c(\d+)$")


def load_dumps(directory: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), "pt") as handle:
            metadata = handle.metadata() or {}
            info = json.loads(metadata.get("trajectory", "{}"))
            tag: str = info.get("tag", "")
            match = TAG_PATTERN.match(tag)
            if match is None:
                continue
            graph = json.loads(info.get("prompt", "{}"))
            samplers = [
                node["inputs"]["sampler_name"]
                for node in graph.values()
                if isinstance(node, dict) and "sampler_name" in node.get("inputs", {})
            ]
            records.append(
                {
                    "path": path,
                    "sampler": samplers[0] if samplers else "unknown",
                    "prompt": match.group(2),
                    "seed": int(match.group(3)),
                    "cfg": int(match.group(4)) / 10.0,
                    "pc_error": handle.get_tensor("pc_error").numpy().astype(np.float64),
                    "curvature": handle.get_tensor("curvature").numpy().astype(np.float64),
                    "sigma": handle.get_tensor("sigma").numpy().astype(np.float64),
                }
            )
    return records


def select_comparable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lengths = [record["sigma"].size for record in records]
    modal_length: int = max(set(lengths), key=lengths.count)
    kept = [record for record in records if record["sigma"].size == modal_length]
    samplers = [record["sampler"] for record in kept]
    modal_sampler: str = max(set(samplers), key=samplers.count)
    return [record for record in kept if record["sampler"] == modal_sampler]


def build_matrix(records: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    length: int = min(record[key].size for record in records)
    stacked: np.ndarray = np.stack([record[key][:length] for record in records])
    usable: np.ndarray = np.all(stacked > 0.0, axis=0)
    return stacked[:, usable], np.flatnonzero(usable)


def group_means(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centred: np.ndarray = np.empty_like(values)
    for label in np.unique(labels):
        mask: np.ndarray = labels == label
        centred[mask] = values[mask] - values[mask].mean(axis=0, keepdims=True)
    return centred


def decompose(values: np.ndarray, cfg_labels: np.ndarray) -> dict[str, float]:
    total: float = float(values.var())
    after_step: np.ndarray = values - values.mean(axis=0, keepdims=True)
    step_residual: float = float(after_step.var())
    after_cfg: np.ndarray = group_means(values, cfg_labels)
    cfg_residual: float = float(after_cfg.var())
    return {
        "total": total,
        "sigma_explained": 1.0 - step_residual / total,
        "cfg_additional": (step_residual - cfg_residual) / total,
        "instance_residual": cfg_residual / total,
    }


def report(records: list[dict[str, Any]], key: str) -> None:
    matrix, steps = build_matrix(records, key)
    values: np.ndarray = np.log(matrix)
    cfg_labels: np.ndarray = np.array([record["cfg"] for record in records])
    shares = decompose(values, cfg_labels)

    print(f"\n{key}: {matrix.shape[0]} runs x {matrix.shape[1]} usable steps")
    print(f"  variance of log {key} explained by step index alone : "
          f"{shares['sigma_explained'] * 100:6.2f} %")
    print(f"  additional variance explained by cfg                : "
          f"{shares['cfg_additional'] * 100:6.2f} %")
    print(f"  remaining instance-specific variance                : "
          f"{shares['instance_residual'] * 100:6.2f} %")

    spread: np.ndarray = matrix.max(axis=0) / matrix.min(axis=0)
    print(f"  per-step spread across runs, median {np.median(spread):.2f}x "
          f"max {spread.max():.2f}x")

    profile: np.ndarray = matrix.mean(axis=0)
    span: int = min(record["sigma"].size for record in records)
    sigma_profile: np.ndarray = np.stack(
        [record["sigma"][:span] for record in records]
    ).mean(axis=0)
    print("  mean profile (step, sigma, value):")
    for position in range(0, len(steps), max(1, len(steps) // 8)):
        index = steps[position]
        print(f"    {index:3d}  sigma={sigma_profile[index]:7.4f}  {profile[position]:.4f}")


def report_by_cfg(records: list[dict[str, Any]], key: str) -> None:
    print(f"\n{key} by cfg, instance-specific share within each cfg group")
    for cfg in sorted({record["cfg"] for record in records}):
        subset = [record for record in records if record["cfg"] == cfg]
        if len(subset) < 3:
            continue
        matrix, _ = build_matrix(subset, key)
        values: np.ndarray = np.log(matrix)
        residual: np.ndarray = values - values.mean(axis=0, keepdims=True)
        share: float = float(residual.var()) / float(values.var())
        spread: np.ndarray = matrix.max(axis=0) / matrix.min(axis=0)
        print(f"  cfg={cfg:<5} runs={len(subset):3d}  instance share "
              f"{share * 100:6.2f} %  median spread {np.median(spread):.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        default=str(pathlib.Path("D:/ComfyUI/output/exp")),
    )
    arguments = parser.parse_args()

    directory = pathlib.Path(arguments.directory)
    loaded = load_dumps(directory)
    if not loaded:
        print(f"no tagged dumps found in {directory}")
        return

    records = select_comparable(loaded)
    dropped: int = len(loaded) - len(records)
    schedules = {tuple(np.round(record["sigma"], 6)) for record in records}
    print(f"loaded {len(loaded)} runs, kept {len(records)} comparable "
          f"(sampler={records[0]['sampler']}, steps={records[0]['sigma'].size})")
    if dropped:
        print(f"  dropped {dropped} runs with a different sampler or step count")
    if len(schedules) > 1:
        print("  warning: grids differ across runs, step indices are not comparable")

    for key in ["pc_error", "curvature"]:
        report(records, key)
        report_by_cfg(records, key)

    print("\nreading: a high instance-specific share means adaptive control has")
    print("headroom; a share near zero means a static schedule already captures")
    print("everything the free error estimate can see")


if __name__ == "__main__":
    main()
