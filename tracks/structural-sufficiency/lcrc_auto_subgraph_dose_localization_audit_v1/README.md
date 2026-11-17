# lcrc-auto-subgraph-dose-localization-audit-v1

`lcrc-auto-subgraph-faithfulness-pilot-v1`（交付提交 `147b1d81`）的**只读审计轮**。

回答三个问题：

1. 原轮是否用“二跳路径两条键都在子图内”（`n_s2_internal`）匹配了随机对照的扰动剂量？
2. 修正为实际被屏蔽的 S2 occurrence 数（`n_s2_affected`）后，Top-1 相对随机对照的优势是否仍在？
3. 高亮子图引起的局部预测响应主要落在子图内部根节点，还是外部根节点？

## 运行

```bash
# 身份门禁 + 8 分子全链路 smoke（两个数据集）
bash tracks/structural-sufficiency/lcrc_auto_subgraph_dose_localization_audit_v1/run_local.sh \
    --purpose "preflight" --set stage=preflight --set dataset=all
# 正式审计（每个数据集一个 worker；CPU 足够）
bash .../run_local.sh --purpose "audit molesol" --set stage=audit --set dataset=ogbg-molesol
bash .../run_local.sh --purpose "audit mollipo" --set stage=audit --set dataset=ogbg-mollipo
# 聚合：表格、bootstrap、REPORT/DECISION
bash .../run_local.sh --purpose "aggregate" --set stage=aggregate
```

测试：

```bash
uv run pytest -q tracks/structural-sufficiency/lcrc_auto_subgraph_dose_localization_audit_v1/tests
```

## 只读边界

* 上游轮次目录、`tracks/structural-query-learning`、`lcrc_cross_task_preparation_v1`、
  `lcrc_lipo_model_dissection_v1`、`lcrc_readout_context_pilot_small_v1` 全部只读复用；
  本轮只在自己的 `code/`、`results/`、`analysis/`、`REPORT.md`、`DECISION.md` 下写文件。
* 不重新选择 Top-1，不重新训练，不换 seed，不改架构，不改搜索范围，不新增数据集。
* 两种对照协议（`pilot_internal_matched` / `corrected_affected_matched`）永远分列，不混池。

## 产物

* `PROTOCOL.md`、`REPORT.md`、`DECISION.md`
* `analysis/`: `old_vs_corrected.csv`、`matched_controls.csv`、`old_control_dose.csv`、
  `localization.csv`、`localization_summary.json`、`bootstrap.json`、`decision.json`、
  `preflight.json`、`run_manifest.json`、`protocol_freeze.json`
* `results/<dataset>/<split>/`: `records.json`、`candidates.npz`（全部候选的三口径计数，
  含被拒候选）、`atom_response.csv`、`summary.csv`

## 内存与运行预算

所有前向都分块（`MAX_VARIANTS_PER_FORWARD=8`、单次前向原子/键数上限 200000、torch 线程 4），
逐 split 落盘，不在内存里累积全部候选的图。实测峰值 RSS：molesol **704 MB**、mollipo
**797 MB**（192 分子/数据集，CPU，31 s / 58 s wall）；未使用 GPU，因此没有占用 `res`。
预算与内存守卫参数记录在 `results/*/budget.json` 与 `analysis/run_manifest.json`。

## 代码结构

| 模块 | 作用 |
|---|---|
| `code/au_common.py` | 身份、路径、冻结常量、哈希与统计小工具 |
| `code/au_s2.py` | internal / boundary / affected 剂量账目 + 真实门控等价检查 |
| `code/au_controls.py` | corrected affected-matched 随机对照（复用上游生成器） |
| `code/au_intervene.py` | 分块冻结前向、效果记录、根节点定位 |
| `code/au_worker.py` | 数据集 worker：身份核验 + 逐分子审计（按 split 落盘） |
| `code/au_aggregate.py` | CPU 聚合：并排表格、bootstrap、判定 |
| `code/au_report.py` | 由 analysis 产物生成 `REPORT.md` / `DECISION.md` |
| `code/au_runner.py` | 控制面 runner（`preflight` / `audit` / `aggregate`） |

## 工程修复记录（只改实现，不改科研规则）

1. `UPSTREAM_ROUND_DIR` 曾用 round id（连字符）拼目录，实际仓库目录用下划线；改为
   `UPSTREAM_ROUND_ID.replace("-", "_")`。此前 preflight 直接 FileNotFoundError，未产生任何数字。
2. preflight smoke 记录缺少 `occurrence_mapping_n_failed` / `occurrence_mapping_checks`，
   触发 `KeyError`；补上这两个字段。此前没有写出任何正式审计记录。
3. 发现分数重放容差拆成两件事：`score_A` 必须等于“上游自己存的梯度”上的公式值（1e-9，
   精确）；“重算梯度”的分数用 CPU↔GPU 重放宽容差（`1e-5 + 1e-5|x|`）核对。此前
   2.955e-07 的浮点差被误当失败；没有修改任何上游数字或判定规则。
