# Adaptive Samplers

一套 ComfyUI 采样器，核心只有一个想法：在多步预测–校正格式中，预测状态与校正状态之差本身就是局部截断
误差的估计，而且**不需要任何额外的网络调用**——因为产生校正的那次调用，正是下一步本来就要做的。

本插件把这个估计暴露出来；而在实用层面更重要的是，它开放了 SA-Solver 的随机强度，这个参数在 ComfyUI
中是写死的。

[English](README.md) · [Русский](README_ru.md)

## 安装

克隆到 `ComfyUI/custom_nodes/` 并重启 ComfyUI。无需额外依赖，仅使用 ComfyUI 本身已有的 `torch`、
`numpy` 和 `safetensors`。

## 节点

| 节点 | 输出 | 用途 |
| --- | --- | --- |
| `Adaptive SA Sampler` | `SAMPLER` | 开放了随机强度（`eta`）与噪声窗口的 SA-Solver |
| `Adaptive PC Sampler` | `SAMPLER` | 带免费误差估计的确定性 Adams 预测–校正采样器 |
| `Trajectory Probe` | `SAMPLER` | 包裹任意采样器，把逐步诊断数据写入 `output/exp` |

采样器名称同时注册到普通 `KSampler` 下拉列表和 API 工作流中：
`asa_fixed`、`asa_adaptive`、`apc_2`、`apc_3`、`apc_2_pred`、`apc_2_fixed`、`apc_3_fixed`。

## 真正有用的部分：随机强度

ComfyUI 内置了 SA-Solver（Xue 等，NeurIPS 2023），名为 `sa_solver`。它只对**漂移项**施加多步 Adams
预测–校正，把扩散项单独保留，代价恰好是**每步 1 次网络调用**。ComfyUI 以 `tau_func=None` 调用它，
因此随机强度被固定为 `eta = 1.0`，窗口固定在 20–80 %，并且没有任何节点可以修改。

`Adaptive SA Sampler` 将其开放。当 `eta = 1.0`、`response = 0.0`、窗口为 `0.2–0.8` 时，它与
`sa_solver` 逐位一致（已验证：最大绝对差为 `0.000e+00`）。

### 实测结果

Anima（flow matching DiT），768×1024，cfg 5.0，`normal` 调度，10 NFE，4 个提示词 × 2 个随机种子。
*几何*指经 8 倍平均池化后相对 100 步 euler 参考解的 RMS 偏差；*纹理*指高通残差的 RMS。收敛的 30 步
SDE 的纹理值为 `0.3335`。

| 采样器 | 几何（越低越好） | 纹理 |
| --- | --- | --- |
| `euler`（ODE） | 0.3677 | 0.1368 |
| `euler_ancestral`（SDE，无漂移校正器） | 0.4016 | 0.1318 |
| `sa_solver`（SDE + 漂移校正器） | **0.3516** | **0.2244** |

在相同预算下，漂移校正器让随机采样器的宏观几何*优于*确定性求解器，同时多保留 64 % 的微观纹理。

### 如何选择 `eta`

| `eta` | 几何 | 纹理 | 纹理差距缩小 |
| --- | --- | --- | --- |
| 1.0 | 0.3516 | 0.2244 | 0 % |
| 1.4 | 0.3865 | 0.2503 | 24 % |
| **1.8** | **0.4225** | **0.2885** | **59 %** |
| 2.0 | 0.4541 | 0.2975 | 67 % |

以损失单位几何换得的纹理：1.0→1.4 为 `0.74`，1.4→1.8 为 `1.06`，1.8→2.0 为 `0.28`。超过 1.8 之后
兑换率急剧恶化，细小结构（睫毛、叶缘）开始变成乱涂。**建议范围为 1.6–1.8。**

提高 `eta` 还会改变构图，而不只是表面质感：请预期得到一张不同的图，而不是同一张图加上更多细节。

### 不要放宽噪声窗口

把 `stochastic_start`/`stochastic_end` 从 `0.2–0.8` 改为 `0.1–0.9`，会损失约 20 % 的宏观几何，而纹
理**完全没有**增加（0.2244 → 0.2225）。继续放宽到 `0.05–0.95` 则毫无变化。在极高 sigma 处注入的噪
声，还没来得及形成结构就被重新积分掉了；在极低 sigma 处注入的噪声，已没有剩余步数可供其解析。只有
中间区段才能把随机性转化为细节。

## Trajectory Probe

包裹任意 `SAMPLER`，每次运行向 `output/exp` 写入一个 `.safetensors`，内含逐步的 `sigma`、
`pc_error`（即那个免费的局部误差估计）、`curvature`、状态/denoised/导数的范数、调度表、最终 latent，
以及可选的中间 latent 堆栈（`latent_stride`）。生成该结果的提示词图保存在文件元数据中。

`analyze_dumps.py` 将 `log(pc_error)` 的方差分解为由步序号、cfg 和具体样本分别解释的份额。

## 未能奏效的部分

之所以写出来，是因为它们被实测过，而非凭推测。

- **自适应步长。** `budget` 模式下的 `apc_*` 利用免费误差估计重新分配步长。它大幅优于均匀网格，但
  **不敌 ComfyUI 自带的 `normal` 调度**（精度为其 0.81–0.99 倍）。通过离线均衡实测误差导出的调度同样
  不敌 `normal`。保留它们仅用于消融对比，不建议用于生产。
- **由误差驱动的噪声控制**（`response` ≠ 0）。在所有测试中，其效果与直接把 `eta` 降低相同平均幅度在
  噪声范围内一致。请保持 `response = 0.0`。
- **常用 cfg 下的求解器阶数。** 在 cfg 1.0 时四阶方法比 euler 好 1.30–1.52 倍；在 cfg 5.0 时
  `dpmpp_2m` / `ipndm` / `deis` 中最好的也只领先 1.01–1.04 倍；在 cfg 8.0 时则毫无优势。在常用的引导
  强度下，截断误差并不是误差的主导项。

## 关于其他采样器的说明

在同一模型上实测，在基于它们开展工作之前值得了解：

- `uni_pc` 被写死为 VP/eps 参数化（`noise / sqrt(1 + sigma²)`、`model_type = "noise"`）。它对
  flow matching 模型并不成立，在 cfg 8 下得分 6.86，而 euler 为 0.63。请勿用于 Flux/SD3/Anima 这类
  模型。
- 实测每步 NFE：`sa_solver`、`euler_ancestral`、`res_multistep_ancestral`、`dpmpp_3m_sde`、
  `er_sde`、`deis`、`clyb_4m_sde_momentumized` 为 1.0；`heun`、`seeds_2`、`dpmpp_sde` 为 1.9；
  `res_3s_ode` 为 3.0，`res_5s_ode` 为 5.0，`res_6s_ode` 为 6.0；`dpm_adaptive` 为 3.0。在 8–14 NFE
  的预算下，只有每步 1 次调用的采样器之间才具可比性。
- 在 8 NFE 下，没有校正器的高阶随机多步方法会严重失稳：`res_multistep_ancestral` 的几何得分为
  1.93，`clyb_4m_sde_momentumized` 为 7.74，而 `sa_solver` 为 0.40。

## 测试

独立运行，无需 GPU：

```
python test_pc_core.py          # Adams 权重、收敛阶、误差估计标定
python test_adaptive_step.py    # 非线性基准上自适应调度与静态调度的对比
```

需要 ComfyUI 环境，从 ComfyUI 根目录运行：

```
venv/Scripts/python.exe custom_nodes/adaptive-samplers/test_comfy_path.py   # NFE 计数
venv/Scripts/python.exe custom_nodes/adaptive-samplers/test_stochastic.py   # 与 sa_solver 的一致性
```

`test_pc_core.py` 验证 Adams 预测器/校正器权重与教科书系数在 1e-16 内一致、1–4 阶方法的局部截断阶为
2/3/4/5，以及免费误差估计跟踪真实局部误差的中位比值为 0.93–0.98。

## 实验脚本

`run_grid.py`、`run_schedule_ab.py`、`run_sde_test.py`、`run_jitter_sweep.py` 通过 HTTP API 驱动
ComfyUI，以 `probe_variance.json` 作为图模板。上文的数据正是由它们得出；使用前请先修改模板中的模型、
LoRA 和 VAE 名称。

## 许可证

BSD 3-Clause，见 `LICENSE`。
