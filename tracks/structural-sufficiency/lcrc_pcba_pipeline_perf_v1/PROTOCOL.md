# PROTOCOL — LCRC PCBA 数据管线性能诊断与定向优化

Round: `lcrc-pcba-pipeline-perf-v1`
结果标签：`PERF_OPTIMIZATION_ACCEPTED` 或 `NO_CLEAR_SPEEDUP`（工程决策标签，不是科学指标）
上游（只读）：`tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1`

本文件在**第一次 GPU 测量之前**固定测量协议。任何改动都必须重新记录并重跑。

---

## 1. 被测对象

* 数据集/缓存：pilot 的 `cache/pcba_v1`（`npy-directory-v1`，4.5 GB，memmap 读取），
  `analysis/pcba_labels_trainvalid.npy`（Test 行全 NaN），官方 split（`pcba_split.json`）。
  本轮**不重建**缓存、不重采样、不换 split。
* 模型：冻结 LCRC-C1（`bp_model.build_c1(cont_slice=None)`），245,184 参数，宽度 64。
* 训练配方：AdamW（peak lr 3e-4、wd 1e-3、betas 0.9/0.999、eps 1e-8）、
  per-step cosine（T_max = steps_per_epoch × 360，eta_min 1e-5）、
  grad clip 5.0、masked BCE over observed entries、FP32。
* batch size = 128，seed = 0。

## 2. 测量协议（固定）

| 项 | 值 |
|---|---|
| batch 顺序 | `torch.Generator().manual_seed(0)` + `torch.randperm(len(train_rows))`，即 `pc_train._batches` 在第 1 个 epoch 产生的前缀；记录 `rows_sha256` |
| warmup | 前 20 step 不计时 |
| 计时 step | 300（记录 p50 / p90 / 累计） |
| 设备 | `res` 主机，pool `res-gpu0`，单进程 |
| 精度 | FP32；不改 TF32 / cudnn 设置（实测值记录在 `profile.json`） |

三个 pass，全部在同一进程、同一 batch 顺序上：

1. `coarse` —— 基线 step 循环原样执行（只在 phase 边界插 CUDA event + 每 step 一次 sync）。
   **CUDA event 在 stream 内记录、且在 step 末尾 `synchronize()` 之后才读 `elapsed_time`**，
   因此 GPU 段是 kernel 执行跨度，不是 enqueue 时间。此 pass 给出吞吐基准。
2. `isolated` —— 同样算术，每个 phase 边界 `torch.cuda.synchronize()`，用于无歧义归因
   （原循环本来就被 `isfinite` / `float(clip)` / `.item()` 串行化，代价很小）。
3. `profiler` —— `torch.profiler`（CPU+CUDA activities）覆盖 20 step，取 kernel 级 top 表。

另外做纯数据管线微基准：只 `collate`、`collate`+`to_device`（各 100 batch），
以及带计时探针的 collate 逐段分解（`pf_collate.collate_instrumented`，输出必须与冻结实现逐字段相等）。

## 3. 阶段定义

| 阶段 | 含义 |
|---|---|
| `collate_cpu` | `np.asarray(...)[...]` 从 memmap gather + ragged index 构造 + `torch.from_numpy` |
| `h2d_and_labels` | 20 个 batch 张量的 `.to(device)`（pageable 内存，非 pinned）+ `labels[rows]` 的 CPU gather 与上传 |
| `forward_gpu` | `model(batch)`（C1：atom/bond encoder + local MLP + rho readout） |
| `loss_gpu` | masked BCE + `torch.isfinite(loss)` 的隐式同步 |
| `backward_gpu` | `zero_grad` + `loss.backward()` |
| `optimizer_gpu` | `clip_grad_norm_`（+ `float()` 同步）、`optimizer.step()`、`scheduler.step()` |
| `scalar_read_sync` | `float(loss.detach().cpu().item())`（基线循环里的标量读取） |
| `_unaccounted` | step wall 中未被上述 phase 覆盖的部分（每 phase sync 的延迟 + Python 循环本身） |

## 4. 无法可靠测量的指标（预先声明）

* **page-fault 时间**无法按 phase 归因：只记录 `getrusage` 的 minor/major fault 计数增量。
* **NVML 利用率**只有 250 ms 采样粒度，是粗观测，不是 per-step 度量。
* **主机共租**、CPU 频率/睿频、页缓存首轮冷热差异不可控：只报告分布（p50/p90）与冷热两轮，
  不把单个数字当作精确值。
* **kernel 级占比**来自 `torch.profiler`，对极短 kernel 有插桩开销，只用于相对排序。

## 5. 优化约束（不可违反）

* 输入分子与顺序不变；atom / bond / S1 / S2 内容不变；batch size = 128 不变；
  模型与参数初始化不变；loss / optimizer / scheduler 不变；FP32 配置不变；
  train/valid 划分不变。
* 不改模型结构、不改采样、不改标签、不增减任务、不改学习率、
  不使用官方 Test、不启动新 seed。
* 不预先生成占大量内存的全量 GPU 张量。
* **只优化一个最大已测瓶颈**，不做组合式性能搜索。

## 6. 验收门槛

1. 新旧 `collate` 全部字段在 shape / order / index / value 上逐项相等（`torch.equal`）。
2. 单图 vs 批量推理一致性不退化。
3. 新旧 forward logits 差异在 FP32 容差内。
4. 新旧 loss、参数梯度、单次优化更新在容差内。
5. 固定 10 step 的 loss 序列与模型权重一致（不允许通过改变优化算法取速）。
6. 标签 mask、任务数（128）、官方 Test 隔离规则不变。
7. **端到端训练吞吐提升 ≥ 1.5×** 才标记 `PERF_OPTIMIZATION_ACCEPTED`，否则 `NO_CLEAR_SPEEDUP`。

## 7. 资源预算

GPU 测量总预算 **0.25 A100 GPU-hours**（`res-gpu0`）。

## 8. 停止条件

本轮只做：性能分解 → 单一瓶颈优化 → 正确性门槛 → 同环境 300 step A/B。
完成后立即停止；不启动 PCBA 低学习率实验、正式训练、多 seed、其他数据集、
模型消融或任何新架构探索。

---

## 9. 测量后追加（post-hoc，未改动上文预先注册的规则）

A 阶段结果（`analysis/profile.json`、`analysis/h2d_probe.json`）显示：

| 阶段 | ms/step | 占 step wall |
|---|---|---|
| CPU→GPU 传输（`to_device` + labels） | **29.1** | **39.6%** |
| CPU memmap + `collate` | 13.1 | 17.8% |
| loss + 隐式同步 | 10.2 | 13.9% |
| backward | 12.4 | 16.9% |
| forward | 3.1 | 4.2% |
| optimizer / scheduler（含 clip + 同步） | 5.3 | 7.2% |
| 标量读取 `.item()` | 0.05 | 0.07% |
| 未归因（sync 延迟 + Python 循环） | 0.01 | 0.0% |

最大已测瓶颈是 **CPU→GPU 传输**，不是 `collate`（后者为第二大）。机制已由
`pf_h2d_probe.py` 定位：21 个逐张量 pageable 拷贝每次固定 ~1.27 ms 主机阻塞，
而 memcpy 引擎本身只要 0.16 ms/step；pin 后同类拷贝 0.245 ms。

因此按 §5「只优化一个最大已测瓶颈」改为优化 H2D：`code/pf_pipeline.py` 的
`PinnedStager`（pinned 暂存 + 非阻塞拷贝），**`collate` 一行未改**。
若只优化 `collate`，投影上限约 70.9/(70.9−17) ≈ 1.31×，达不到 §6 的 1.5× 门槛，
这也是选择 H2D 的工程理由。
