# 工程日志 — `fixed-source-scaffold-balanced-learning-v0`

本地执行（用户指示：`res` / `res-2` 均被占用，直接本地跑）。本机无 CUDA 设备
（`torch.cuda.is_available() == False`，仅 AMD 集成显卡），全部计算在 CPU（8 线程）上完成。
所有时间戳为 UTC，2026-11-26。

## 执行顺序（实际）

| 时间 | 动作 | 结果 |
|---|---|---|
| 13:34 | 建立 round 目录、写 `PROTOCOL.md`、写 `code/` | — |
| ~13:37 | 吞吐预检（临时脚本，未提交）：单 step ≈ 0.045 s、eval Train ≈ 0.42 s、eval Valid ≈ 0.05 s、峰值 RSS ≈ 0.68 GB | 360 epoch 投影 ≈ 10 min，预算可行 |
| 13:38:11 | `sb_freeze.py` | 首次运行因 `np.asarray(..., reverse=...)` 笔误报错；修正后通过：K=1604、N=3360、max/min 权重 76、ESS/N 0.609、门禁 `insufficient=False` |
| 13:38:39–13:38:51 | `sb_checks.py` | 10/10 通过（12.0 s），投影 360 epoch 611.95 s |
| 13:38:51 | `sb_train.py` 尝试 1 | epoch 1 结束即被自检断言拦下（`int(step_w)` 与 3360 差 1，浮点累加截断）——未写任何 checkpoint/曲线 |
| 13:39:23 | `sb_train.py` 尝试 2 | 同上（断言阈值仍过严）——未写任何 checkpoint/曲线 |
| 13:40:14 | `sb_train.py` 尝试 3 | 诊断版报错信息显示 `weight_sum = 3359.99992`（float32 权重求和），确认是数值容差而非逻辑错误 |
| 13:40:33–13:50:08 | `sb_train.py` 正式运行（唯一完整训练） | `COMPLETED`，360 epoch / 9720 step，best epoch 136，best Valid RMSE 0.7274001，墙钟 565.06 s |
| 13:50:15 起 | `sb_analyze.py`（多次重跑以修正表格列映射与权重归一化口径） | verdict `NO_CLEAR_GAIN` |

三次中止的尝试各自在第一个 epoch 内退出（合计约 10 s 计算），没有产生任何 checkpoint、
预测或曲线行；`runs/scaffold_balanced_seed0/run.json` 中只剩正式运行（`COMPLETED`）的记录。
正式训练只发生一次。

## 未执行

- `run_all.py` 没有端到端跑（那会重复训练，违反"只训练一个新模型"）。四个阶段按
  freeze → checks → train → analyze 的顺序**逐条**执行，与 `run_all.py` 一致。
- 没有做第二 seed、没有搜索其他权重公式、没有读取/评分 Test。

## 复现

```bash
cd /home/calendar/code/research
uv run python tracks/structural-sufficiency/fixed_source_scaffold_balanced_learning_v0/code/sb_freeze.py
uv run python tracks/structural-sufficiency/fixed_source_scaffold_balanced_learning_v0/code/sb_checks.py
uv run python tracks/structural-sufficiency/fixed_source_scaffold_balanced_learning_v0/code/sb_train.py
uv run python tracks/structural-sufficiency/fixed_source_scaffold_balanced_learning_v0/code/sb_analyze.py
# 或一次执行 freeze -> checks -> train -> analyze:
uv run python tracks/structural-sufficiency/fixed_source_scaffold_balanced_learning_v0/code/run_all.py
```
