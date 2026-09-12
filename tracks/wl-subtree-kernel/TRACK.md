# Track: WL-Subtree-Kernel 实验（v2）

| 项 | 内容 |
|----|------|
| slug | `wl-subtree-kernel` |
| 状态 | `active` |
| 创建 | 2026-09-08（v2 修订同日） |
| 主线位置 | Graph Kernel（Kernel → GNN → GSN 谱系的最老基线） |

## 问题（一句话）

把旧 keyan 仓的 **WL subtree kernel + SVM**（Method D）作为标准基线复现，
并修正 TU bioinformatics 数据集的节点标签处理，使其贴近标准 WL-subtree / GIN Table 1 设置。

## 范围（v2）

- 核：grakel `WeisfeilerLehman(n_iter∈{1..6}, normalize=True)`（base=VertexHistogram）+ `SVC(kernel='precomputed')`
- 节点标签（显式 `pyg_to_grakel(data, dataset_name)`）：
  REDDIT* = constant；MUTAG/PROTEINS/DD/NCI1/ENZYMES = TU categorical labels；
  IMDB-*/COLLAB = degree
- C 网格：{0.001..1000} log 网格（**不是 GIN 原始 grid**，GIN 未公开）
- 协议四档（同折叠，可逐 fold 配对）：`strict-fixed`（h=3,C=10 先验固定）、
  `nested`（内层 cv5 默认；主结果）、`paperlike`（non-nested CV 选参，approximation）、
  `optimistic`（per-fold 乐观，诊断）
- 统计：across-seed mean of repeated-CV means ± std-of-means；另给全体 outer fold 池化 mean±std
  **（与 GIN 的 fold std 不同构，只能比 central accuracy）**

## 协议 / 评估

- 每 (数据集, 种子) 独立跑 RepeatedStratifiedKFold(10,10)；种子
  `0, 42, 123, 1024, 2026, 777, 3407, 999, 111, 888`
- 指标：acc 与 macro-F1
- run 参数即配置 + 默认输出 `results/v2-categorical/`；v1 结果保留在 `results/` 顶层

## 进度

| 阶段 | 状态 |
|------|------|
| 数据/标签核对（5 bio 数据集 one-hot 核验、DD 89 维原始标签） | 完成 |
| 重复边 sanity check（核矩阵 max\|Δ\|=0） | 完成 |
| v1 语义恢复 + 逐位一致性验证（MUTAG strict seed0 max diff=0.0） | 完成 |
| phase-1 sanity（seed0 × 10×1 × 4 协议 × 5 数据集） | 完成 |
| 分解实验（deg/cat × old/new grid，seed0 × 10×1） | 完成 |
| 正式 10 种子 × 10×10：strict / nested(cv5) / paperlike / optimistic | 完成/进行中 |
| 与 GIN Table 1 对比 + 结论 | 进行中 |

## 入口

| 路径 | 用途 |
|------|------|
| `code/run_wl_subtree_kernel.py` | v2 唯一实现（单文件） |
| `code/run_wl_subtree_kernel_v1_deg.py` | v1-semantics 变体（分解用，已验证一致） |
| `results/v2-categorical/` | v2 官方结果 |
| `results/decomp/` | v1↔v2 分解实验 |
| `results/`（顶层） | v1 官方结果（保留不动） |
| `notes/README.md` | 决策与来源 |
| `docs/README.md` | 协议与跑法 |

## 来源

- 旧 keyan 仓（备份：`~/life/archive/科研/keyan.zip`）
  `phase2_structure_classification/src/models.py::WLKernelClassifier` +
  `experiments/archive/run_step2.py`（Method D）
- GIN：K. Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019, Table 1
  （WL subtree 参考行；其中 DD 与 ENZYMES 未报告）

## 禁止

- 声称 exact GIN reproduction（GIN 未公开 C grid / 归一化 / 预处理细节）
- 用 strict-fixed 或 optimistic 声称复现 GIN
- 人为挑 seed / 删 seed 拟合数字
