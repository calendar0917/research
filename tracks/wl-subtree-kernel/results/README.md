# results/

本轨实验数字唯一入口（JSON）。

## 目录布局（v2 起）

| 目录 | 内容 |
|------|------|
| `results/`（顶层） | **v1 官方结果（degree 节点标签 + 旧网格）**，保留不动，供对照/追踪 |
| `results/v2-categorical/` | **v2 官方结果（categorical/constant/degree 节点标签按数据集选择；h∈{1..6}；C log 网格 0.001..1000；nested 默认 cv5）** |
| `results/decomp/` | v1↔v2 分解实验（seed 0 × 10×1，inner cv5）：`deg_oldgrid/`、`cat_oldgrid/`、`deg_newgrid/` |

## 文件命名（每个目录内）

- `{DATASET}__seed{SEED}.json`：strict-fixed 单 (数据集, 种子) 的 fold 明细与聚合
- `{DATASET}_summary.json`：strict-fixed 跨种子汇总
- `{DATASET}__wlnest_seed{SEED}.json` / `{DATASET}__wlnest_summary.json`：nested
- `{DATASET}__wlpaper_seed{SEED}.json` / `{DATASET}__wlpaper_summary.json`：paperlike
- `{DATASET}__wlopt_seed{SEED}.json` / `{DATASET}__wlopt_summary.json`：optimistic

每个 summary.json 含：

- 主统计：`across_seed_mean_acc` / `across_seed_std_acc`（= `across_seed_std_of_mean_acc`）
  —— 每个 seed 的 repeated-CV mean accuracy 在 seeds 之间的 mean±std
- 池化统计：`all_outer_fold_mean_acc` / `all_outer_fold_std_acc` —— 全部 outer fold accuracy 直接合并
- 选参统计（nested/paperlike/optimistic）：`param_stats.n_iter`、`param_stats.C` 频次 +
  `param_stats.grid_boundary_hits`（是否撞网格边界）
- 每 seed 记录：`mean_acc`/`std_acc`（该 seed 内 fold 级 mean±std）、`fold_acc` 数组、所选参数

## 统计口径说明

GIN Table 1 报告 10-fold validation mean ± fold std；本轨主统计是
"across-seed mean of repeated-CV means" ± "std across seed-level means"。
两者 ± 不是同一种不确定性指标，只能比较 central accuracy（见 docs/README.md）。

## v1 首次结果（保留于顶层，10 种子 × 10 折 × 10 重复，degree 标签，旧网格）

| 数据集 | strict | nested | optimistic |
|--------|--------|--------|------------|
| IMDB-BINARY | 0.7283 ± 0.0022 | 0.7228 ± 0.0026 | 0.7541 ± 0.0021 |
| IMDB-MULTI | 0.5113 ± 0.0009 | 0.5067 ± 0.0021 | 0.5262 ± 0.0010 |
| REDDIT-BINARY | 0.7640 ± 0.0011 | 0.7599 ± 0.0020 | 0.7704 ± 0.0011 |
| COLLAB | 0.7742 ± 0.0006 | 0.7813 ± 0.0009 | 0.7873 ± 0.0006 |

bio 数据集 v1（degree 标签，旧网格，nested 默认 holdout 0.2）：

| 数据集 | strict | nested(doc 值) | optimistic |
|--------|--------|----------------|------------|
| MUTAG | 0.8431 | 0.8612 | 0.9091 |
| PROTEINS | 0.7159 | 0.7165 | 0.7443 |
| DD | 0.7362 | 0.7421 | 0.7569 |
| NCI1 | 0.7802 | 0.8101 | 0.8126 |
| ENZYMES | 0.4122 | 0.4019 | 0.4294 |

---

## v2 正式结果（10 种子 × 10 折 × 10 重复；categorical 标签；h∈1..6 × C∈0.001..1000；nested=cv5）

| 数据集 | strict-fixed (h3,C10) | **nested(推荐)** | paperlike | optimistic | GIN WL-subtree |
|---|---|---|---|---|---|
| MUTAG | 0.8696 ± 0.0043 | **0.8600** ± 0.0025 | 0.8803 ± 0.0023 | 0.9213 ± 0.0020 | 0.904 |
| PROTEINS | 0.7483 ± 0.0012 | **0.7566** ± 0.0011 | 0.7620 ± 0.0007 | 0.7785 ± 0.0007 | 0.750 |
| DD | 0.7922 ± 0.0009 | **0.7921** ± 0.0006 | 0.7942 ± 0.0005 | 0.8070 ± 0.0005 | N/A |
| NCI1 | 0.8408 ± 0.0005 | **0.8516** ± 0.0006 | 0.8528 ± 0.0006 | 0.8566 ± 0.0005 | 0.860 |
| ENZYMES | 0.5275 ± 0.0021 | **0.5315** ± 0.0035 | 0.5415 ± 0.0039 | 0.5623 ± 0.0025 | N/A |

± = std across seed-level means（每个 seed 为 100 个 outer fold 的 mean）。
GIN 的 ± 为 10-fold validation 的 fold std，与上表 ± 不同构，只能比较 central accuracy。
GIN Table 1 未报告 DD / ENZYMES（N/A）；GIN did not disclose the exact C search grid。

## 与 v1（degree 标签 + 旧网格 h∈{1,2,3,5} × C∈{0.1,1,10,100}）对比

| 数据集 | v1 strict | v2 strict | Δstrict | v1 nested | v2 nested | Δnested | v1 optimistic | v2 optimistic | Δopt |
|---|---|---|---|---|---|---|---|---|---|
| MUTAG | 0.8431 | 0.8696 | +0.0265 | 0.8612 | 0.8600 | -0.0012 | 0.9091 | 0.9213 | +0.0122 |
| PROTEINS | 0.7159 | 0.7483 | +0.0324 | 0.7165 | 0.7566 | +0.0401 | 0.7443 | 0.7785 | +0.0342 |
| DD | 0.7362 | 0.7922 | +0.0559 | 0.7421 | 0.7921 | +0.0500 | 0.7569 | 0.8070 | +0.0501 |
| NCI1 | 0.7802 | 0.8408 | +0.0606 | 0.8101 | 0.8516 | +0.0415 | 0.8126 | 0.8566 | +0.0440 |
| ENZYMES | 0.4122 | 0.5275 | +0.1152 | 0.4019 | 0.5315 | +0.1296 | 0.4294 | 0.5623 | +0.1330 |

## 与 GIN 差距（central accuracy）

| 数据集 | GIN WL-subtree | v2 nested | Δ(nested−GIN) | v2 paperlike | Δ(paperlike−GIN) |
|---|---|---|---|---|---|
| MUTAG | 0.904 | 0.8600 | -0.0440 | 0.8803 | -0.0237 |
| PROTEINS | 0.750 | 0.7566 | +0.0066 | 0.7620 | +0.0120 |
| DD | N/A | 0.7921 | N/A | 0.7942 | N/A |
| NCI1 | 0.860 | 0.8516 | -0.0084 | 0.8528 | -0.0072 |
| ENZYMES | N/A | 0.5315 | N/A | 0.5415 | N/A |

---

## 统一 9 数据集 v2 结果（2026-09-09 追加；social 数据集补跑 nested+paperlike）

### 表 A：mean ± std（± = std across seed-level means；10 seeds × 10 折 × 10 重复）

| 数据集 | v1 nested (旧配置) | **v2 nested** | v2 paperlike | Δ(v2nested−v1) | paperlike−nested | GIN WL-subtree |
|---|---|---|---|---|---|---|
| MUTAG | 0.8612 ± 0.0064 | **0.8600 ± 0.0025** | 0.8803 ± 0.0023 | -0.0012 | +0.0203 | 0.904 ± 0.057 |
| PROTEINS | 0.7165 ± 0.0026 | **0.7566 ± 0.0011** | 0.7620 ± 0.0007 | +0.0401 | +0.0054 | 0.750 ± 0.031 |
| DD | 0.7421 ± 0.0016 | **0.7921 ± 0.0006** | 0.7942 ± 0.0005 | +0.0500 | +0.0021 | N/A |
| NCI1 | 0.8101 ± 0.0007 | **0.8516 ± 0.0006** | 0.8528 ± 0.0006 | +0.0415 | +0.0011 | 0.860 ± 0.018 |
| ENZYMES | 0.4019 ± 0.0039 | **0.5315 ± 0.0035** | 0.5415 ± 0.0039 | +0.1296 | +0.0100 | N/A |
| IMDB-BINARY | 0.7228 ± 0.0026 | **0.7227 ± 0.0021** | 0.7307 ± 0.0022 | -0.0001 | +0.0080 | 0.738 ± 0.018 |
| IMDB-MULTI | 0.5067 ± 0.0021 | **0.5090 ± 0.0009** | 0.5125 ± 0.0008 | +0.0023 | +0.0035 | 0.509 ± 0.016 |
| REDDIT-BINARY | 0.7599 ± 0.0020 | **0.7633 ± 0.0013** | 0.7640 ± 0.0011 | +0.0034 | +0.0007 | 0.780 ± 0.014 |
| COLLAB | 0.7813 ± 0.0009 | **0.7826 ± 0.0005** | 0.7833 ± 0.0005 | +0.0013 | +0.0007 | 0.789 ± 0.019 |

### 表 B：pooled all-outer-fold mean ± std（10 seeds × 100 folds = 1000 folds 合并计算，
最接近 GIN 的 fold-std 量级的参考量；仍与 GIN 单次 10-fold 的 fold std 不同构）

| 数据集 | v1 nested pooled | v2 nested pooled | v2 paperlike pooled |
|---|---|---|---|
| MUTAG | 0.8612 ± 0.0759 | 0.8600 ± 0.0771 | 0.8803 ± 0.0712 |
| PROTEINS | 0.7165 ± 0.0387 | 0.7566 ± 0.0377 | 0.7620 ± 0.0370 |
| DD | 0.7421 ± 0.0356 | 0.7921 ± 0.0342 | 0.7942 ± 0.0347 |
| NCI1 | 0.8101 ± 0.0193 | 0.8516 ± 0.0175 | 0.8528 ± 0.0172 |
| ENZYMES | 0.4019 ± 0.0621 | 0.5315 ± 0.0639 | 0.5415 ± 0.0632 |
| IMDB-BINARY | 0.7228 ± 0.0406 | 0.7227 ± 0.0419 | 0.7307 ± 0.0427 |
| IMDB-MULTI | 0.5067 ± 0.0381 | 0.5090 ± 0.0371 | 0.5125 ± 0.0367 |
| REDDIT-BINARY | 0.7599 ± 0.0297 | 0.7633 ± 0.0288 | 0.7640 ± 0.0286 |
| COLLAB | 0.7813 ± 0.0178 | 0.7826 ± 0.0174 | 0.7833 ± 0.0173 |

注：v1 nested = degree/constant 标签 + h∈{1,2,3,5}×C∈{0.1..100} + 内层 holdout(0.2)；
v2 = categorical/constant/degree 标签按数据集 + h∈{1..6}×C∈{0.001..1000} + 内层 cv5。
GIN WL-subtree 列：MUTAG/PROTEINS/NCI1 为用户提供的 Table 1 值；social 四行为论文常见引用值
（本机未持有原文逐字核对）；DD/ENZYMES Table 1 未报告。GIN 的 ± 是 10-fold validation 的
fold std，与表 A 的 ±（std across seed-level means）不是同一种不确定性指标，只能比较 central accuracy。

