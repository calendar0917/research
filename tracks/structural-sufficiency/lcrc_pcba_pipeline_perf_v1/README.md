# lcrc_pcba_pipeline_perf_v1 — PCBA 数据管线性能诊断与定向优化

Round: `lcrc-pcba-pipeline-perf-v1`
上游冻结对象：`tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1`（不改动）

## 这个 round 做什么

1. 在 `res-gpu0` 上对**冻结的** PCBA 训练 step 做性能分解（300 个固定 step，先 warmup
   再计时）：CPU memmap 读取 + `collate`、CPU→GPU 传输、GPU forward、loss/backward、
   optimizer/scheduler、Python 调度与同步。GPU 段用 CUDA event 在 stream 内测量并在
   `synchronize()` 之后读取，不把 enqueue 时间当成 kernel 时间。
2. 只优化**实测最大**的那一个瓶颈，保持数学计算与训练协议不变。
3. 正确性门槛（逐字段相等 / logits / loss / 梯度 / 单次更新 / 固定 10 step / 标签隔离）。
4. 同环境、同 300 step 的原版 vs 新版吞吐对照。

## 入口

```bash
# 性能分解（写 analysis/profile.json）
uv run python tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1/code/pf_profile.py --device cuda

# H2D 机制探针（写 analysis/h2d_probe.json）
uv run python tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1/code/pf_h2d_probe.py --device cuda

# 正确性门槛 + 300-step A/B 吞吐（写 analysis/throughput_comparison.json）
uv run python tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1/code/pf_bench.py --device cuda

# CPU 单元测试（需要 pilot 的 4.5 GB 缓存）
uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1/tests
```

远端通过 `rr` 执行（`res` + pool `res-gpu0`），run ID 记录在 `REPORT.md`。

## 文件

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 测量协议、阶段定义、预先声明的不可测指标、验收门槛 |
| `code/pf_common.py` | 冻结输入装载、固定 batch 顺序、环境/指纹 |
| `code/pf_collate.py` | 冻结 collate 引用、计时镜像（未改 collate 本身） |
| `code/pf_pipeline.py` | 设备传输：冻结的 pageable 路径 vs `PinnedStager`（本轮唯一优化） |
| `code/pf_profile.py` | 300-step 性能分解（coarse / isolated / profiler 三 pass） |
| `code/pf_h2d_probe.py` | H2D 机制探针（pageable vs pinned vs bulk，解释瓶颈成因） |
| `code/pf_bench.py` | 正确性门槛 + 原版 vs 新版 300-step 吞吐对照 |
| `tests/test_pf_collate.py` | CPU-only 逐字段/逐字节等价测试 |
| `analysis/profile.json` | 性能分解结果 |
| `analysis/h2d_probe.json` | H2D 机制探针结果 |
| `analysis/throughput_comparison.json` | 吞吐对照、显存、数值一致性 |
| `REPORT.md` | 结论、结果表、局限性 |

## 边界

本轮只做工程提速与正确性验证：不启动新训练、不换 seed、不碰官方 Test、
不改超参/架构/标签，也不做组合式性能搜索。结论标签只有
`PERF_OPTIMIZATION_ACCEPTED` 或 `NO_CLEAR_SPEEDUP`。
