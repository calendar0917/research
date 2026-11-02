# LCRC TU quick screen v1

**状态**: `EXPLORATORY_SINGLE_SEED_RANDOM_811`（单 seed、单划分、80/10/10 探索性筛查）

固定执行 prompt 的实现。数据集、模型、输入、划分和训练配方在运行前已经确定；
本轮只做实现、最小正确性验证、12 个正式 run 和实测结果汇总。

## 范围

* 4 个 TUDataset 二/三分类数据集：NCI1、PROTEINS、IMDB-BINARY、IMDB-MULTI；
* 3 个模型：`LCRC-FULL`、`LCRC-C1`（仅删除 `b_mean` 块）、`GIN+global`（固定应用参照）；
* 12 个正式 run = 4 数据集 × 3 模型，每个只训练一次 `seed=0`；
* 唯一划分：8:1:1，`split_seed=0`，两次分层 `train_test_split(random_state=0)`；
* 额外报告 Train 多数类常数预测器（零训练成本，不计入 12 个 run）。

不做多 seed、交叉验证、调参、消融、早停、EMA/SWA/soup、增强、SSL 或额外 loss。

## 输入（TU 适配，不含化学字段）

* 官方接口：`torch_geometric.datasets.TUDataset(cleaned=False, use_node_attr=False, use_edge_attr=False)`；
  官方数据说明：<https://chrsmrrs.github.io/datasets/docs/datasets/>，
  PyG 文档：<https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.TUDataset.html>。
* 图按无向简单图处理：去自环、合并重复方向边（记录变更计数），保留孤立点与不连通图。
* 节点类型：NCI1/PROTEINS 用原始离散节点标签（词表只由 Train 建立，未见标签 → UNK）；
  两个 IMDB 用度数桶 `0,1,…,15,>=16` + 预留 UNK（K=18）。
* 边类型：所有物理边 `EDGE=1`，`UNK=2` 预留（B=2）。
* `node_feat` = 节点类型 one-hot (K) + 度数桶 one-hot (17) + in_cycle + 最小环长 one-hot (7)
  + log1p(cycle_degree) + cycle_degree/max(degree,1) + degree==1 + cycle_degree>=3（总宽 K+29）。
* `edge_feat` = `[1, is_cycle_edge]`；9 个纯拓扑 scalar；28 维 global（lipo_v1 前 20 个拓扑项
  + 6 个最小环长比例 + `fused_ring_atom_fraction` + `distinct_atom_types`）。
* S1/S2 沿用 LCRC 的有向非回退二跳路径枚举（root→middle→end，end≠root），不采样、不截断；
  `nbr_code = type*B + (bt-1)`，`path_cat = type(middle)*B² + (bt1-1)*B + (bt2-1)`，
  `path_end_code = type(end)*B + (bt2-1)`。
* scalar / global 的均值与尺度只用 Train 拟合，零方差尺度置 1。

复用（只读）：`lcrc_cross_task_preparation_v1/code/{feature_spec,molecular_graph,lcrc_data}.py` 的
图算法、缓存布局、`collate` 与 `fit_norm`；`lcrc_lipo_model_dissection_v1/code/dissect_model.py` /
`dissect_common.py` 的 FULL / C1 结构语义（删除 `b_mean`、共享初始化映射）。

## 模型与配方

| 模型 | 说明 |
|---|---|
| `FULL` | LCRC-A 主体：atom 128→96，bond 160→96，stat_rank 32，local_hidden 192，h=64，rho_hidden 64；`logits = Σ_v f(h_v) + rho(global, mean h, mean h², log1p n)`；h 上 dropout 0.10，每 forward 一次 |
| `C1` | 与 FULL 完全一致，只是删除 96 维 `b_mean = b_sum/deg`；共享参数按 TU 实际列布局从 seed-0 FULL 初始化映射 |
| `GIN` | 节点线性→64，5 层 GINConv（train_eps、eps=0、MLP 64-128-64、LayerNorm→SiLU→dropout 0.10 残差），`[sum_pool, mean_pool, global28, log1p n]`→157-64-C |

配方（三模型一致）：fp32，AdamW lr=1e-3 / wd=1e-4 / betas=(0.9,0.999) / eps=1e-8，
batch 64，200 epoch，per-step cosine 到 1e-5，梯度裁剪 5.0，C 维 CrossEntropyLoss。
checkpoint 按 Valid accuracy（并列取更低 Valid CE，再并列取更早 epoch）选择；
Test 在每个 run 的 checkpoint 哈希写盘后只评估一次。

## 目录

```
lcrc_tu_quick_screen_v1/
├── code/
│   ├── tu_common.py     # 常量、路径、固定配方、12-run 矩阵、seed/环境记录
│   ├── tu_adapter.py    # TUDataset 加载/清洗、节点类型、TU FeatureSpec、特征构造
│   ├── tu_data.py       # 划分、结构缓存、Train-only 归一化、清单写盘
│   ├── tu_model.py      # LCRC-FULL / LCRC-C1 / GIN+global、共享初始化映射
│   ├── tu_prep.py       # 数据准备 CLI（manifest + split + caches + norm）
│   ├── tu_gates.py      # 最小正确性 gates（划分/手算/置换/batch/共享初始化/梯度）
│   ├── tu_smoke.py      # 短 CUDA smoke（前向/反向/保存/重载，不读 Test）
│   ├── tu_train.py      # 单个正式 run
│   ├── tu_aggregate.py  # results.csv + 汇总表 + 完整性校验
│   └── tu_report.py     # 由 run.json/summary.json 生成 REPORT.md
├── configs/protocol.json
├── tests/               # CPU、无数据的 gate 测试
├── analysis/            # 提交的证据 JSON（划分、数据来源、gates、汇总）
├── cache/               # 可重建结构缓存（git-ignored）
└── runs/                # 每个 run 的 checkpoint/curve/predictions（git-ignored）
```

## 运行命令

```bash
# 0) 数据位置（官方 raw 文件；可用 LCX_TU_ROOT 覆盖）
#    <root>/<NAME>/<NAME>/raw/ 或 <root>/<NAME>/raw/

# 1) 测试（CPU、无数据）
uv run pytest -q tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/tests

# 2) 数据准备（划分 + 缓存 + Train-only norm），只跑一次
uv run python tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/code/tu_prep.py

# 3) 正确性 gates（写 analysis/gates.json）
uv run python tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/code/tu_gates.py

# 4) 远端短 CUDA smoke（不读 Test）
rr run res lcx-smoke --pool res-gpu0 --result tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/analysis \
  -- python tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/code/tu_smoke.py --device cuda:0

# 5) 12 个正式 run（2 卡并行，一任务一卡）
rr run res lcx-NCI1-FULL --pool res-gpu0 --result tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/runs \
  -- python tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/code/tu_train.py \
       --dataset NCI1 --model FULL --seed 0 --device cuda:0
# ... 其余 11 个 run 见 REPORT.md 的运行记录

# 6) 汇总（本地读 12 个 run.json）
uv run python tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/code/tu_aggregate.py
uv run python tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/code/tu_report.py
```

本轮实测结果与逐 run 记录见 `REPORT.md`、`results.csv`、`analysis/summary.json`、
`analysis/comparison.md`、`analysis/run_registry.json`。

缓存和权重按仓库既有规则不进入 Git；`analysis/` 的 JSON、`results.csv` 与 `REPORT.md` 提交。
