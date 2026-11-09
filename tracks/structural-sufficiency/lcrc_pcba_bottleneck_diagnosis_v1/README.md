# LCRC MolPCBA 性能瓶颈诊断 v1

`POST_HOC_EXPOSED_TEST_DIAGNOSIS` — 只读、CPU、描述性诊断。

解释为什么 W2 / S3 / RC-REAL / BOC-FULL 在引入不同结构计算后官方 Test AP 仍相近
（0.2506–0.2559），并区分三个竞争解释：预测趋同（H1）、结构迁移失效（H2）、
任务稀疏性/评价不稳定性（H3）。

**本轮不训练、不调参、不选 checkpoint、不改模型、不在 Test 上拟合任何东西。**
GPU 使用：0。结构计算与统计全部在 CPU 上完成（约 6 分钟，16 核）。

---

## 1. 数据与来源

| 项 | 位置 |
|---|---|
| 官方 MolPCBA mapping / split / labels | `data/ogb/ogbg_molpcba/` |
| W2 / RC-REAL / BOC-FULL / BOC-BOND Test logits | `../lcrc_pcba_boc_v1/runs/freeze/test_predictions/` |
| S3 Test logits | `../lcrc_pcba_s3_selective_v1/analysis/test_predictions/` |
| Validation logits | 本轮 `_local/valid_predictions/`（由冻结 checkpoint 逐次只读前向得到；`*.npy` 被 gitignore） |
| 官方 Test 标签读取 | 只经历史 `cc_data.read_official_test_labels` 门禁 |

完整身份链（checkpoint SHA256 → logits SHA256 → 行 id → AP 重放）见
`analysis/prediction_inventory.json` 与 `analysis/alignment_report.json`。

## 2. 复现

```bash
cd tracks/structural-sufficiency/lcrc_pcba_bottleneck_diagnosis_v1

# 全流程（Validation 重放 → 库存/对齐 → P1 → P2 → P3 → P4 → 报告表格）
uv run python code/run_all.py

# 正确性审计
uv run pytest -q tests
```

可分开执行：

```bash
uv run python code/lc_valid.py run        # 5 个冻结 checkpoint 的 Validation 前向（CPU，~90 s）
uv run python code/lc_inventory.py        # 库存 + 行对齐 + AP 重放门禁
uv run python code/lc_rank.py             # P1 排序趋同
uv run python code/lc_structures.py       # P2 Train-only WL2 支持度（~4 min）
uv run python code/lc_gain.py             # P3 结构支持度 vs 收益
uv run python code/lc_macro.py            # P4 宏 AP 分解
uv run python code/lc_tables.py           # 生成 analysis/report_tables.md
```

`run_all.py --skip-valid --skip-structures` 可复用已生成的 Validation logits 与
WL2 支持字典（两者都是确定性的）。

## 3. 输出

| 文件 | 内容 |
|---|---|
| `analysis/prediction_inventory.json` | 每个模型的 checkpoint/epoch/SHA、logits 形状与 SHA、行 id、历史 AP、重放 AP |
| `analysis/alignment_report.json` | 行对齐、重放、官方 vs 独立 AP、身份链检查 |
| `analysis/task_sets.json` | `ALL` / `HIGH_SUPPORT` / `LOW_SUPPORT` / 边界任务 |
| `analysis/ranking_by_task.csv` | 每个 (pair, split, task) 的 Spearman rho |
| `analysis/topk_by_task.csv` | Top-1%/Top-5% Jaccard、Top-K 内正例、互斥正例、排名百分位 |
| `analysis/prediction_similarity_summary.json` | P1 汇总与预注册判读标签 |
| `analysis/train_wl2_frequency_summary.json` | Train-only WL2 字典与频次统计 |
| `analysis/structure_support_{valid,test}.csv` | 逐分子结构支持度（不读标签） |
| `analysis/structure_support_summary.json` | 分组计数、重原子数、rare 比例 |
| `analysis/novelty_vs_gain_by_task.csv` | 每个 (pair, split, task, novelty group) 的组内 AP 与 ΔAP |
| `analysis/novelty_rank_change_by_task.csv` | 正例归一化排名变化的组内汇总 |
| `analysis/novelty_vs_gain_summary.json` | P3 汇总、任务匹配趋势、尺寸混杂检查 |
| `analysis/task_support_effects.csv` | 逐任务 support、AP、ΔAP |
| `analysis/macro_ap_decomposition.json` | P4 宏 AP 分解与敏感性分析 |
| `analysis/report_tables.md` | 报告用紧凑表格（自动生成） |
| `analysis/run_log.json` | 各步骤耗时；GPU=0、训练步=0 |

研究结论见 `REPORT.md`；冻结的分析定义见 `PROTOCOL.md`（含 §12 事后修订声明）。

## 4. 注意

- Test 已暴露，所有 Test 数字都是**事后工程分析**，不是独立确认。
- P1/P3 的排序统计直接使用 logits；AP 使用历史评分路径（`sigmoid` float32），
  因此 P4 的宏 AP 与历史冻结值一致到 1e-7。
- Validation logits 是 CPU 重放，与历史 GPU 前向不 bitwise 可比；聚合 AP 门禁 1e-4。
- WL2 只是"局部结构是否被 Train 支持"的代理，不是同构证明，也不是网络真实输入。
