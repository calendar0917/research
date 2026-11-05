# REPORT — LCRC PCBA 数据管线性能诊断与定向优化

Round: `lcrc-pcba-pipeline-perf-v1`
结果标签：**`PERF_OPTIMIZATION_ACCEPTED`**（工程决策标签，不是科学性能指标）
上游冻结对象：`tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1`（只读，未修改）
测量主机/环境：`res`（a100-2），pool `res-gpu0`，A100-SXM4-40GB，driver 550.163.01，
torch 2.5.1+cu124，python 3.12.14，FP32（TF32 关闭）

---

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| 位置 | 冻结 PCBA seed0 训练 step（batch=128，LCRC-C1，245,184 参数） |
| 基线 step wall | **71.34 ms p50**（p90 73.55），14.83 step/s，300 个固定 step |
| 最大已测瓶颈 | **CPU→GPU 传输**：29.07 ms/step（38.5%），其中 21 次逐张量 pageable 拷贝每次固定 ~1.27 ms 主机阻塞 |
| 唯一优化 | `PinnedStager`：pinned 暂存 + `non_blocking` 拷贝（`code/pf_pipeline.py`） |
| 优化后 step wall | **41.22 ms p50**（p90 45.30），23.92 step/s |
| 端到端加速比 | **1.612×**（总墙钟 300 step）；p50 口径 1.731× |
| 纯训练 epoch 投影 | 184.6 s → **114.5 s**（2738 step） |
| 正确性 | 6 组门槛（逐字段/逐字节相等、单图vs批量、logits、loss/梯度/单次更新、10 step、标签隔离）**全部通过** |
| 显存 | 峰值 allocated 80.5 MB / reserved 116.0 MB，**优化前后完全相同（Δ=0）** |
| GPU 预算 | **≈0.11 A100 GPU-hours**（上限 0.25） |
| 官方 Test | 未读取、未评估；标签工件中 Test 行全 NaN，且批次行全部来自 Train split |
| 后续训练 | **未启动**（按要求在本轮结束后立即停止） |

一句话：这条 PCBA 训练 step 96% 的时间不是花在算子上，而是花在**主机侧等待**；
最大的单块等待是「21 个逐张量 pageable H2D 拷贝」，把它换成 pinned + 非阻塞拷贝即可
在完全不改数学、模型、协议的前提下把 step 从 71.3 ms 压到 41.2 ms（1.61×）。

## 2. 性能分解（A 阶段）

测的是**冻结的** `pc_train.py` 内循环，固定 batch 顺序
（`rows_sha256` 记录在 `analysis/profile.json` 的 `protocol.batch_plan`），
20 step warmup 后计时 300 step，三个 pass：

* `coarse`：基线循环原样 + phase 边界 CUDA event + 每 step 一次 `synchronize()`，
  给出**吞吐基准**（不做额外同步，不扰动）；
* `isolated`：同样算术，phase 边界全部 `synchronize()`，给出**无歧义归因**；
* `profiler`：`torch.profiler`（CPU+CUDA）覆盖 20 step，给 kernel 级归因。

CUDA event 一律在 stream 内 `record`、在 step 末尾 `synchronize()` **之后**才读
`elapsed_time`，因此表中的 GPU 段是 kernel 执行跨度，**不是 enqueue 时间**；
主机侧 enqueue 用 `perf_counter` 单独记录。

### 2.1 主分解（`isolated` pass，300 step，in-loop，权威口径）

| 阶段 | 总耗时 ms | p50 ms | p90 ms | 占 step wall |
|---|---:|---:|---:|---:|
| **CPU→GPU 传输**（`to_device` + labels） | **7930.4** | **29.07** | **29.96** | **38.5%** |
| CPU memmap 读取 + `collate` | 3873.8 | 13.12 | 14.56 | 18.8% |
| backward（含 `zero_grad`） | 3642.5 | 12.36 | 12.44 | 17.7% |
| loss（masked BCE + `isfinite` 隐式同步） | 2779.1 | 10.24 | 10.31 | 13.5% |
| optimizer / scheduler（含 `clip_grad_norm_` + `float()` 同步） | 1459.8 | 5.26 | 5.31 | 7.1% |
| GPU forward | 883.6 | 3.10 | 3.13 | 4.3% |
| 标量读取 `float(loss.item())` | 14.1 | 0.05 | 0.05 | 0.1% |
| 未归因（每 phase 的 sync 延迟 + Python 循环） | 9.2 | — | — | 0.04% |
| **step wall 合计** | **20 590** | **72.03** | **74.20** | 100% |

`isolated` 的 step wall 比 `coarse` 高 0.6 ms（多出来的每 phase `synchronize()`），
所以占比用 `isolated`、吞吐用 `coarse`。

### 2.2 无扰动吞吐（`coarse` pass）

| 指标 | 值 |
|---|---|
| 300 step 总墙钟 | 20.75 s |
| step/s | 14.46 |
| step wall p50 / p90 | 71.39 / 73.66 ms |
| CUDA event 覆盖的「stream 忙」占比 | 0.80 |

`coarse` 的 GPU event span（p50/step）：H2D **31.77 ms**、backward 14.55、loss 7.73、
optimizer 2.76、forward 0.60；主机侧 enqueue（p50/step）：`to_device` **26.72**、
`collate` 14.06、backward 12.30、loss+sync 9.42、optimizer+sync 5.05、labels 2.53、
forward 1.28。注意 forward 的 **enqueue 1.28 ms > GPU 执行 0.60 ms**：前向是小 kernel
密集、完全被发射延迟支配的「碎」负载。

### 2.3 kernel 级观测（`torch.profiler`，20 step）

| 指标 | 值 |
|---|---|
| GPU kernel 时间 | **6.36 ms/step**（占 71.4 ms 的 8.9%） |
| kernel 启动次数 | **3 079 次/step**（平均每个 kernel 2.1 µs） |
| 阶段 CPU 自耗时 | 62.6 ms/step（插桩抬高，仅供相对比较） |
| 头部 kernel | `embedding_dense_backward` 1.06 ms、CUB radix-sort（来自 `nonzero`）0.63 ms、`aten::mm` 0.37 ms、`aten::addmm` 0.26 ms |

结论很清楚：**这条 step 是主机延迟瓶颈，不是算力瓶颈**。GPU 真正在算的时间只有 6.4 ms，
其中还有 3.6 ms 与「算子之间的等待」混在一起。这也解释了 NVML 观测到的低占用：
GPU util 均值 49.4%（250 ms 采样，粗观测）、SM 时钟 810 MHz（仅为额定 1410 MHz 的 57%）、
平均功耗 93.7 W（A100 满载 250 W+）。

### 2.4 数据管线自身（微基准，与 loop 分离）

`collate` 单独跑 100 batch：p50 **9.26 ms**；拆开看：

| 组件 | p50 ms | 说明 |
|---|---:|---|
| `index_build_arange_loops` | **8.334** | `_ranges()` 的逐分子 `np.arange`+`np.concatenate`，以及 nbr/path 的逐原子 `np.arange` 循环 |
| `field_gather_and_tensor_wrap` | 0.479 | memmap 逐字段 fancy-index + `torch.from_numpy` |
| `gather_from_mmap` | 0.412 | `bond_index`/`nbr_code`/`path_end` 的 gather |
| `ptr_read` | 0.043 | 指针数组读取 |

也就是说 `collate` 的 **90% 是 ragged index 的逐元素数组创建**，而不是 memmap I/O。
同一 batch 在训练循环内测得 13.1 ms、独立微基准 9.3 ms（差 1.4×）——loop 内的 CPU 状态
（紧随上一 step 的 GPU 工作、频率/页缓存竞争）会让它更慢；**报告以 in-loop 值为准**。

## 3. 瓶颈定位：为什么是 H2D，而不是 `collate`

`analysis/h2d_probe.json`（同一 batch、同一设备、每种变体后都 `synchronize()`，
单位 ms/batch，1.746 MB / 21 个张量）：

| 变体 | p50 ms | 说明 |
|---|---:|---|
| `pageable_per_tensor`（**现状**） | **26.605** | 21 次 `.to(device)`，每次约 1.27 ms 主机阻塞 |
| `pinned_per_tensor`（**采用**） | **0.245** | pinned 暂存 + `non_blocking=True` |
| `pinned_bulk`（1 次拷贝） | 0.239 | 单次 bulk |
| `pageable_bulk`（1 次 pageable 拷贝） | 0.317 | 说明**不是带宽问题**：同样 1.75 MB，一次拷只要 0.32 ms |
| `pin_memory_only` | 0.450 | pinned 暂存本身的 host 成本 |
| `sync_only`（裸 `cudaStreamSynchronize`） | 0.005 | |

`torch.profiler` 的交叉验证：`Memcpy HtoD (Pageable -> Device)` 21 次/step 只占
**0.171 ms/step** 的 GPU 时间，而同一批拷贝周围的 `cudaStreamSynchronize`（28 次/step）
占 **39.7 ms/step** 的阶段 CPU 时间。即：**拷贝引擎很快，等它的是主机**。
有效带宽 1.75 MB / 26.6 ms ≈ 0.066 GB/s，pinned 后约 7 GB/s（106×）。

因此按 `PROTOCOL.md` §5「只优化一个最大已测瓶颈」改为优化 H2D。若反过来只优化
`collate`（第二大，13.1 ms），上限约 70.9/(70.9−17) ≈ 1.31×，达不到 1.5× 门槛——
这是选择 H2D 的工程理由，不是事后合理化：A 阶段先测，位移后再定优化对象。

## 4. 优化内容（`code/pf_pipeline.py`）

改动**只有一处语义**：CPU→GPU 传输的机制。

| 路径 | 实现 |
|---|---|
| A（冻结） | `pc_model.to_device`：`{k: v.to(device)}`，pageable、逐张量、阻塞；labels 用 `torch.as_tensor(..., device=)` |
| B（本轮） | `PinnedStager`：每个张量先 `pin_memory()`（pinned 暂存），再 `to(device, non_blocking=True)`；pinned 源用深度 2 的环形引用保活；labels 走同一条 pinned 路径 |

明确**未改动**：`lcrc_data.collate` 一行未改（仍是被测基线）、batch 顺序、batch size 128、
模型结构与参数初始化（同 seed 同 `build_c1`）、masked BCE、AdamW（betas/eps/wd）、
per-step cosine（T_max = steps×360、eta_min 1e-5）、grad clip 5.0、FP32 配置、
Train/Valid 划分、任务数 128、官方 Test 隔离。没有新增张量常驻显存，没有预先把全量数据
搬到 GPU，没有改优化算法（AdamW 一步不差）。

## 5. 正确性门槛（全部通过）

| gate | 内容 | 结果 |
|---|---|---|
| G1 | 新旧 `collate` 输出**逐字段**一致（20 个张量字段 + `n_graphs`，shape/dtype/顺序/字节） | 6 个固定 batch（原子 3 093–3 603、键 3 322–3 882、S2 9 242–10 798）全部 bitwise 相等 |
| G1 | 传到设备后的 batch 与 labels 逐字节一致 | `max_abs_diff = 0.0`（全部字段） |
| G2 | 单图 vs 批量推理一致性 | 1.67e-6 ~ 1.91e-6（阈值 1e-5，与历史 gate 同口径） |
| G3 | 新旧 forward logits | `max_abs_diff = 2.86e-6`；**同路径 A/A 噪声底 3.81e-6、B/B 2.86e-6** → 差异处于同输入同代码的 kernel 归约顺序噪声内 |
| G4 | loss / 梯度 / 单次优化更新 | loss 差 **0.0**；grad-norm 差 **0.0**；最大梯度差 2.98e-8；单步后权重最大差 1.19e-7（阈值 1e-6） |
| G5 | 固定 10 step 的 loss 序列与权重 | loss 最大差 **5.96e-8**；权重最大差 1.19e-7（阈值 1e-5） |
| G6 | 标签 mask / 任务数 / Test 隔离 | 128 任务；4 177 692 观测 + 2 703 588 NaN（NaN 严格排除，未当 0）；Test 行全 NaN 且**批次中从未出现** Test/Valid 行 |

G3 的容差说明：`index_add_` 的归约顺序使同一模型、同一输入、同一 transfer 路径的两次
forward 就能差到 3.8e-6，因此这里用的是上游冻结轮次已注册的 `INFERENCE_TOL = 1e-5`，
并同时报告 A/A、B/B 噪声底作为对照。真正的强证据是 G4/G5：loss 完全相同、梯度与权重
差异在 1 ulp 量级。**没有任何一条门槛是靠"放宽到刚好通过"设的**。

## 6. 性能对照（同环境、同 300 step、同 batch 顺序）

| 指标 | A 原版（pageable） | B 新版（pinned） | 加速比 |
|---|---:|---:|---:|
| step wall p50 | 71.34 ms | **41.22 ms** | **1.731×** |
| step wall p90 | 73.55 ms | 45.30 ms | 1.624× |
| step wall mean | 67.41 ms | 41.81 ms | 1.612× |
| step/s（总墙钟） | 14.83 | **23.92** | **1.612×** |
| 2738-step epoch 投影（纯训练） | 184.6 s | **114.5 s** | 1.612× |
| 完整 epoch 投影（train+valid 10 s） | 194.6 s | 124.5 s | 1.56×（投影） |
| GPU 峰值 allocated | 80.5 MB | 80.5 MB | Δ = 0 |
| GPU 峰值 reserved | 116.0 MB | 116.0 MB | Δ = 0 |
| 300 step 的 loss 序列 | 见 `throughput_comparison.json` | 同上 | 最大差 1.49e-8 |

* 漂移控制：A 在 B 之后重跑了 100 step，p50 = 71.66 ms（相对 A 首轮 +0.45%），
  说明 1.6× 不是时钟漂移造成的。
* 基线一致性：A 的 71.3 ms/step × 2738 = 195 s，与 pilot 实测 train 188 s 同量级
  （pilot 的 188 s 是整 epoch 平均，含逐 epoch 日志开销，本报告 300 step 均值为 67.4 ms
  对应 184.6 s）；优化前的 `coarse` 300 step 也复现了同一量级（71.39 ms p50）。
* 端到端加速比用**总墙钟**口径（1.612×），p50 口径更高（1.731×）；两者都 ≥ 1.5 门槛。
* 显存零增长：pinned 只占主机内存（~1.75 MB × 2 深度），没有把任何数据提前搬到 GPU。

## 7. CPU 耗时分解（对照）

| 组成 | A 原版 | B 新版 |
|---|---|---|
| 主机阻塞在 H2D 拷贝 | 26.7 ms/step（21 次拷贝） | ~0.7 ms/step（21 次 pinned 暂存 + 21 次非阻塞 enqueue） |
| `collate`（纯 CPU） | 13.1–14.1 ms | 同一函数，未改 |
| 其余（backward/loss/opt enqueue） | 与 GPU 侧同步等待交织 | 未改 |
| 进程 CPU 时间（300 step，`getrusage`） | user 20.6 s + sys 0.21 s | 未单独统计（同量级） |

（`isolated` pass 的 `h2d_and_labels` 为 29.07 ms，其中 `to_device` 26.7 ms + labels 2.5 ms；
B 之后该 phase 降到亚毫秒级，剩余时间主要由 `collate` 与 GPU 侧 30 ms 决定。）

## 8. 无法可靠测量的指标（预先声明 + 实际处理）

| 指标 | 处理 |
|---|---|
| page-fault 时间 | 无法按 phase 归因；只记录 `getrusage`（300 step 内 major faults = 0，minor = 44 169，说明缓存已在页缓存中、无磁盘读放大） |
| NVML 利用率 | 250 ms 采样，只作粗观测（util 49.4% 均值、SM 时钟 810/1410 MHz、功耗 93.7 W）；不用它做归因 |
| kernel 级占比 | `torch.profiler` 有插桩开销（CPU 自耗时 62.6 ms vs 实际 71.4 ms 墙钟），只用于相对排序；CUDA kernel 时间口径可信 |
| 主机共租 / 频率 | 无法排除；用 p50/p90 + A 重跑做漂移控制（+0.45%） |
| `collate` 的 in-loop vs 独立差 | 13.1 ms vs 9.3 ms，未做因果归因，仅报告两者 |

## 9. 交付物

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 预先注册的测量协议、阶段定义、不可测指标、验收门槛（§9 为测量后的追加说明） |
| `code/pf_common.py` | 冻结输入装载、固定 batch 顺序与指纹、缓存身份校验 |
| `code/pf_collate.py` | 冻结 collate 引用 + 计时镜像（**未改 collate**） |
| `code/pf_pipeline.py` | `PageableTransfer`（冻结路径）与 `PinnedStager`（本轮唯一优化） |
| `code/pf_profile.py` | 300-step 三 pass 性能分解 |
| `code/pf_h2d_probe.py` | H2D 机制探针（pageable/pinned/bulk） |
| `code/pf_bench.py` | 6 组正确性门槛 + 300-step A/B 吞吐 |
| `tests/test_pf_collate.py` | CPU-only 逐字段/逐字节等价、传输保活、dtype/形状测试（6 passed / 2 skipped） |
| `analysis/profile.json` | 性能分解原始结果（含 profiler 表、NVML、rusage） |
| `analysis/h2d_probe.json` | H2D 机制探针原始结果 |
| `analysis/throughput_comparison.json` | 正确性门槛 + A/B 吞吐 + 决策字段 |

## 10. 复现

```bash
# 本地 CPU 单测（需要 pilot 的 4.5 GB memmap 缓存）
uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1/tests

# 远端（res / pool res-gpu0）——本轮实际执行的三条命令
rr run res pcba-h2d-probe --pool res-gpu0 \
  -- python -m tracks.structural-sufficiency.lcrc_pcba_pipeline_perf_v1.code.pf_h2d_probe
rr run res pcba-perf-profile2 --pool res-gpu0 \
  -- python -m tracks.structural-sufficiency.lcrc_pcba_pipeline_perf_v1.code.pf_profile
rr run res pcba-perf-bench --pool res-gpu0 \
  -- python -m tracks.structural-sufficiency.lcrc_pcba_pipeline_perf_v1.code.pf_bench
```

| 项 | 值 |
|---|---|
| 代码 commit（最终测量所用） | `a3a3453521cf`（bench）、`56d97e54f0a2`（profile2）、`750609d73120`（h2d probe） |
| rr run ID（bench，最终） | `pcba-perf-bench-20261105-184829-7d0c68da` |
| rr run ID（profile） | `pcba-perf-profile2-20261105-183445-617bb9f5` |
| rr run ID（probe） | `pcba-h2d-probe-20261105-182845-633fd8bf` |
| 早期（作废）run | `pcba-perf-profile-20261105-181641-344a8578`、`…182011-9c6e8e24`（导入路径 bug）、`pcba-perf-bench-…54ebe0cb`（G3 容差过紧）、`…e0147a27`（S2 计数口径 bug），均保留在 rr 记录中 |
| GPU 预算 | 8 个 run 的进程墙钟合计 ≈ 320 s + 启动开销 ≈ **0.11 A100 GPU-hours**（上限 0.25） |

## 11. 局限与"不做"的边界

* 本轮只测了 **train step**；Validation（10 s/epoch，pilot 实测）未优化、未重测，
  1.56× 的完整 epoch 数字是**投影**而非测量。
* 加速比是在 `res` 单机、单 A100、共享主机的条件下测的；`res-2` 属于另一个执行 regime，
  本报告的数字**不可直接移植**（regime 记录见 rr `status`）。
* pageable 拷贝每次 1.27 ms 的主机阻塞是**这台主机/驱动栈的性质**；换主机后该数值可能变化，
  但优化本身（pinned + 非阻塞）在任何主机上都不劣化，且已用 A/A 噪声底与漂移检查保护结论。
* `collate` 仍有 8.3 ms/step 的逐元素 `np.arange`/`concatenate`（可向量化），
  本轮按"只优化一个最大已测瓶颈"的规则**未动**；它是下一轮若继续做性能工作的头号候选。
* 数值等价的证据强度：G4 的 loss 完全相同、梯度/权重差在 1 ulp；G3 的 logits 差
  （2.9e-6）与同代码重复运行的噪声底同量级，属于 GPU 归约顺序的非确定性，与本次改动无关。
* 未做：正式训练、低学习率实验、多 seed、其他数据集、模型消融、架构探索、组合式性能搜索。
  本轮到此为止。
