# analysis/ — 本轮所有分析产物的提升副本

这里是从 `runs/_analysis/`（gitignored，随 run 目录生成）复制过来的**只读副本**，
用于在没有 checkpoint 的情况下复读结论。内容与 run 时的输出逐字节相同。

| 文件 | 内容 | 生成命令 |
|---|---|---|
| `freeze_{a,a2,b,a3}.json` | 每个 run 的 best/final 预测 sha256、train.json sha256、参数、GPU·h、初值 hash、以及 config/PLAN/gate 的 sha256 冻结 | `uv run python code/aas_freeze.py --stage <x> --tags ... [--config configs/aas_a3.json]` |
| `summary_{a,a2,b,a3}.json` | 主表 + 分组指标 + 配对 bootstrap 对照 + 预注册判定 + `paired_by_seed` / `same_arm_seed_spread` / `contradiction_verdicts` | `uv run python code/aas_analyze.py --stage <x> --pairs "A1:A0,A1:B_s0,A0:B_s0,A1:B,A0:B,A0M0:A0"` |
| `tables_{a,a2,b,a3}.md` | 人类可读的同内容表格（含 `drop-row172`、`drop-top-1-contributor`） | 同上 |
| `per_molecule_{a,a2,b,a3}.csv` | 1,000 行 valid 的逐分子 \|AE\|（best 与 final） | 同上 |
| `cpu_checks{,_b,_c}.json` | 33 条 CPU gate（11 + 12 + 10） | `uv run python code/aas_checks{,_b,_c}.py` |

阶段对应：

* `a`  = 方向 A 的 seed-0 初判（A1_s0 / A0_s0 vs B_s0）
* `a2` = 补上 seed 1 与 B_s1 匹配基线后的最终裁定（**STOP_A**）
* `b`  = 方向 B（SQBC S1 vs S2 vs B_s0，**STOP_B**）
* `a3` = moment-bank 消融 A0M0 vs A0（**MOMENT_ESSENTIAL**）

原始 checkpoint / 预测 / curve 在 `runs/`（gitignored）；重新生成需要对应的
`rr pull` 结果。**没有任何文件包含 official test 的预测或标签。**
