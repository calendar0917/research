# Kaggle 分版本证据（promoted）

存放从 Kaggle `calendar917/hod-gnn-replication` 各版本下载的有效实验证据。

> 仓库 `results/` 默认 gitignore；此 `kaggle/` 子目录通过 `.gitignore` 放行，纳入版本控制，
> 保证换机器 / 清空工作区后这些原始证据仍在。

## 备份物

| 目录 | 来源 | 内容 | 用途 |
|------|------|------|------|
| `v16/` | Kaggle version 16（2026-08-24） | `gps-molhiv-{0,1}.log` + registry + csv + run 状态 | GPS/MOLHIV **seed 0,1** |
| `v22/` | Kaggle version 22（2026-09-11） | `gps-molhiv-{2,3}.log` + registry + csv + run 状态 | GPS/MOLHIV **seed 2,3** |

每个版本含：
- `registry.jsonl` —— 逐单元解析结果（本目录唯一数字源）
- `paper_vs_reproduced.csv` —— 该版本单独判定
- `run_state.json` / `run.log` / `events.jsonl` —— 环境、命令、心跳审计

## 汇总（GPS + MOLHIV，paper-seeds-provisional）

| seed | best epoch | valid AUC | test AUC | 来源 |
|------|-----------|-----------|----------|------|
| 0 | 43 | 0.81136 | **0.78850** | v16 |
| 1 | 19 | 0.80730 | **0.77454** | v16 |
| 2 | 63 | 0.81982 | **0.78382** | v22 |
| 3 | 44 | 0.83384 | **0.76889** | v22 |

**4-seed 聚合：`0.7789 ± 0.0089`（n=4）** vs 论文 `0.788 ± 0.0101` → **reproduced**
（判定规则：`|mean−paper| = 0.0091 ≤ sqrt(0.0089²+0.0101²) = 0.0134`）