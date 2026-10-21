# Adaptive Samplers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23529356.svg)](https://doi.org/10.5281/zenodo.23529356)

ComfyUI samplers built around one idea: in a multistep predictor–corrector scheme the gap
between the predicted and the corrected state is a local truncation error estimate that costs
**zero extra network evaluations**, because the evaluation that produces the correction is the one
the next step needs anyway.

This package exposes that estimate, and — more usefully in practice — exposes the stochastic
strength of SA-Solver, which ComfyUI hardcodes.

[Русский](README_ru.md) · [Марий йылме](README_chr.md) · [中文](README_zh.md)

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. No extra dependencies; it uses `torch`,
`numpy` and `safetensors`, all already required by ComfyUI.

## Nodes

| Node | Output | Purpose |
| --- | --- | --- |
| `Adaptive SA Sampler` | `SAMPLER` | SA-Solver with exposed stochastic strength (`eta`) and noise window |
| `Adaptive PC Sampler` | `SAMPLER` | Deterministic Adams predictor–corrector with free error estimate |
| `Trajectory Probe` | `SAMPLER` | Wraps any sampler and dumps per-step diagnostics to `output/exp` |

Sampler names are also registered for the plain `KSampler` dropdown and for API workflows:
`asa_fixed`, `asa_adaptive`, `apc_2`, `apc_3`, `apc_2_pred`, `apc_2_fixed`, `apc_3_fixed`.

## The useful part: stochastic strength

ComfyUI ships SA-Solver (Xue et al., NeurIPS 2023) as `sa_solver`. It applies a multistep Adams
predictor–corrector to the **drift** only, keeps the diffusion term separate, and costs exactly
**1 NFE per step**. ComfyUI calls it with `tau_func=None`, which pins the stochastic strength to
`eta = 1.0` inside a fixed 20–80 % window. There is no node to change that.

`Adaptive SA Sampler` exposes it. At `eta = 1.0`, `response = 0.0` and window `0.2–0.8` it
reproduces `sa_solver` bit-for-bit (verified: maximum absolute difference `0.000e+00`).

### Measured behaviour

Anima (flow-matching DiT), 768×1024, cfg 5.0, `normal` scheduler, 10 NFE, 4 prompts × 2 seeds.
*Geometry* is the RMS deviation from a 100-step euler reference after 8× average pooling;
*texture* is the RMS of the high-pass residual. A converged 30-step SDE scores `0.3335` on texture.

| Sampler | Geometry (lower is better) | Texture |
| --- | --- | --- |
| `euler` (ODE) | 0.3677 | 0.1368 |
| `euler_ancestral` (SDE, no drift corrector) | 0.4016 | 0.1318 |
| `sa_solver` (SDE + drift corrector) | **0.3516** | **0.2244** |

The drift corrector gives a stochastic sampler *better* macro geometry than the deterministic
solver at the same budget, while keeping 64 % more micro-relief.

### Choosing `eta`

| `eta` | Geometry | Texture | Texture gap closed |
| --- | --- | --- | --- |
| 1.0 | 0.3516 | 0.2244 | 0 % |
| 1.4 | 0.3865 | 0.2503 | 24 % |
| **1.8** | **0.4225** | **0.2885** | **59 %** |
| 2.0 | 0.4541 | 0.2975 | 67 % |

Texture bought per unit of geometry lost: `0.74` from 1.0→1.4, `1.06` from 1.4→1.8, `0.28` from
1.8→2.0. The exchange rate collapses after 1.8, and fine structures (eyelashes, leaf edges) start
turning into scribble. **Useful range is 1.6–1.8.**

Raising `eta` also shifts composition, not only surface detail — expect a different image, not the
same image with more texture.

### Do not widen the noise window

Moving `stochastic_start`/`stochastic_end` from `0.2–0.8` to `0.1–0.9` costs about 20 % of macro
geometry and yields **no** texture at all (0.2244 → 0.2225). Widening further to `0.05–0.95`
changes nothing. Noise injected at very high sigma is integrated away before it becomes structure;
noise injected at very low sigma has no steps left to resolve. Only the middle band converts
stochasticity into detail.

## Trajectory Probe

Wraps any `SAMPLER` and writes a `.safetensors` per run into `output/exp`, containing per-step
`sigma`, `pc_error` (the free local error estimate), `curvature`, state/denoised/derivative norms,
the schedule, the final latent, and optionally a stack of intermediate latents (`latent_stride`).
The generating prompt graph is stored in the file metadata.

`analyze_dumps.py` decomposes the variance of `log(pc_error)` into shares explained by step index,
by cfg, and by the individual sample.

## What did not work

Reported because it was measured, not assumed.

- **Adaptive step size.** `apc_*` in `budget` mode redistributes steps using the free error
  estimate. It beats a uniform grid by a wide margin but **loses to ComfyUI's `normal` scheduler**
  (0.81–0.99× its accuracy). Schedules derived offline by equidistributing the measured error also
  lose to `normal`. Kept for ablation, not recommended for production.
- **Error-driven noise control** (`response` ≠ 0). On every test it matched, within noise, simply
  lowering `eta` by the same average amount. Keep `response = 0.0`.
- **Solver order at production cfg.** At cfg 1.0 a 4th-order method beats euler by 1.30–1.52×. At
  cfg 5.0 the best of `dpmpp_2m` / `ipndm` / `deis` beats euler by only 1.01–1.04×, and at cfg 8.0
  not at all. Truncation error is not the dominant error term at usual guidance scales.

## Notes on other samplers

Measured on the same model, worth knowing before building on them:

- `uni_pc` is hardcoded to a VP/eps parametrisation (`noise / sqrt(1 + sigma²)`, `model_type =
  "noise"`). It is not valid for flow-matching models and scores 6.86 against 0.63 for euler at
  cfg 8. Do not use it with Flux/SD3/Anima-class models.
- NFE per step, measured: `sa_solver`, `euler_ancestral`, `res_multistep_ancestral`,
  `dpmpp_3m_sde`, `er_sde`, `deis`, `clyb_4m_sde_momentumized` — 1.0. `heun`, `seeds_2`,
  `dpmpp_sde` — 1.9. `res_3s_ode` — 3.0, `res_5s_ode` — 5.0, `res_6s_ode` — 6.0. `dpm_adaptive` —
  3.0. Only 1-NFE samplers are comparable at an 8–14 NFE budget.
- At 8 NFE, high-order stochastic multistep without a corrector destabilises badly:
  `res_multistep_ancestral` scores 1.93 and `clyb_4m_sde_momentumized` 7.74 on geometry, against
  0.40 for `sa_solver`.

## Tests

Standalone, no GPU required:

```
python test_pc_core.py          # Adams weights, convergence orders, error calibration
python test_adaptive_step.py    # adaptive vs static schedules on a nonlinear benchmark
```

Requires ComfyUI's environment, run from the ComfyUI root:

```
venv/Scripts/python.exe custom_nodes/adaptive-samplers/test_comfy_path.py   # NFE accounting
venv/Scripts/python.exe custom_nodes/adaptive-samplers/test_stochastic.py   # sa_solver parity
```

`test_pc_core.py` verifies that the Adams predictor/corrector weights match textbook coefficients
to 1e-16, that local truncation orders are 2/3/4/5 for orders 1–4, and that the free error estimate
tracks the true local error with a median ratio of 0.93–0.98.

## Experiment scripts

`run_grid.py`, `run_schedule_ab.py`, `run_sde_test.py`, `run_jitter_sweep.py` drive ComfyUI over its
HTTP API using `probe_variance.json` as the graph template. They are the scripts that produced the
numbers above; adapt the model, LoRA and VAE names in the template before use.

## Citation

```bibtex
@software{savelev_yuzvynsky_2026_adaptivesamplers,
  author       = {Iakov Savelev and Yehor Yuzvynsky},
  title        = {Adaptive Step Samplers: Zero-Cost Truncation Error Estimation and Exposed Stochastic Control for ComfyUI},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23529356},
  url          = {https://doi.org/10.5281/zenodo.23529356}
}
```

## License

MIT License, see `LICENSE`.
