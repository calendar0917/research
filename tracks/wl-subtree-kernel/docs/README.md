# WL-Subtree-Kernel — 协议与跑法

> 轨入口：[TRACK.md](../TRACK.md)

## 复现对象（what）

- **核**：grakel `WeisfeilerLehman(n_iter, normalize=True)`，base 为 VertexHistogram（即 WL 子树核）
- **节点标签（v2，按数据集显式选择）**：
  - REDDIT* → constant 0（无属性社交图惯例）
  - MUTAG / PROTEINS / DD / NCI1 / ENZYMES → TU 原始 categorical node labels
    （`TUDataset(use_node_attr=False)`，data.x one-hot → argmax）
  - IMDB-* / COLLAB 等 → degree（`edge_index[0].bincount`）
- **SVM**：sklearn `SVC(kernel='precomputed', C=...)`
- **CV**：`RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=seed)`，
  每 fold `set_seed(seed+fold_idx)`
- **指标**：accuracy、macro-F1（每 fold 计算后聚合）

## 四档协议（同折叠、可逐 fold 配对）

| | strict-fixed | **nested（荐，严格）** | paperlike | optimistic（诊断） |
|---|---|---|---|---|
| 选参 | 无（h=3, C=10 先验固定） | 训练折内 cv5（**默认**；holdout 可选 1/9） | 同一批 folds 上 global CV：theta*=argmax mean_cv(theta) | 每折在留出折（验证集）直接选参 |
| 报告 | 外层留出折 | 选定后 90% 重训，外层留出折**只报告一次** | theta* 在同一批 folds 上的 mean±std | 同折选择+报告 |
| 偏差 | 无（但参数先验固定，可能非最优） | 无（选择与报告分离） | 有（non-nested 选择偏差） | 强乐观（选择+报告泄漏） |
| 用途 | 固定参数基线 | **主结果（推荐）** | 研究 non-nested selection bias；paper-like approximation，不是 exact GIN | 乐观上界 / 自检 |

> GIN（Xu et al. 2019）Table 1 的 WL subtree 行：**没有公开其具体选参实现**，
> C grid、kernel 归一化与预处理细节均未完全公开；因此本轨任何一档都不能声称
> "exact GIN reproduction"，paperlike 也明确标为 approximation。

## 超参网格（v2）

- `h ∈ {1,2,3,4,5,6}` —— GIN 对 WL 的明确范围
- `C ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000}` —— log 网格。
  **GIN did not disclose the exact C search grid. 这不是 GIN 原始 C grid**，
  只是避免最优值撞搜索边界的扩展。

## 统计口径（重要）

- 主统计：`across_seed_mean_acc` = 10 个 seed 的 repeated-CV mean accuracy 的均值；
  `across_seed_std_acc`（= `across_seed_std_of_mean_acc`）= 这些 seed-level means 的 std。
- 池化统计：`all_outer_fold_mean_acc/std_acc` = 所有 seed 所有 outer fold accuracy 直接合并。
- **GIN 报告 10-fold validation mean ± fold std** → 与本轨 ± 不是同一种不确定性指标，
  只能比较 central accuracy，不要把两个 ± 直接对比。

## 种子

每数据集固定 10 个 CV 种子：`0, 42, 123, 1024, 2026, 777, 3407, 999, 111, 888`。
每个种子 = 100 次「10 折」评估（`--n-splits 10 --n-repeats 10`）。

## 部署命令

```bash
# nested（主结果）
uv run --group wl-kernel python code/run_wl_subtree_kernel.py \
    --datasets MUTAG PROTEINS DD NCI1 ENZYMES \
    --protocol nested --inner-strategy cv5 \
    --grid-iters 1 2 3 4 5 6 \
    --grid-C 0.001 0.01 0.1 1 10 100 1000 \
    --seeds 0 42 123 1024 2026 777 3407 999 111 888 \
    --n-splits 10 --n-repeats 10 --force

# strict-fixed（h=3, C=10 基线）
uv run --group wl-kernel python code/run_wl_subtree_kernel.py \
    --datasets MUTAG PROTEINS DD NCI1 ENZYMES \
    --protocol strict --wl-iter 3 --wl-C 10 \
    --seeds 0 42 123 1024 2026 777 3407 999 111 888 \
    --n-splits 10 --n-repeats 10 --force

# paperlike / optimistic 同参（--protocol 切换）
```

结果默认写 `../results/v2-categorical/`；v1（degree 标签 + 旧网格）结果保留在
`../results/` 顶层供对照。

## 更新日志

- **v2（2026-09-08+）**：bio 数据集改用原始 categorical node labels；网格扩为
  h∈{1..6} × C∈{0.001..1000}；nested 默认 cv5（holdout 修正为 1/9 → overall 80/10/10）；
  新增 paperlike 协议；metadata 统一记录 node_label_mode / kernel / base_kernel；
  重复边影响数值验证为 0；统计口径增加 pooled 与 std-of-mean 字段。
- v1（2026-09-08）:注意 nested 默认实际是 holdout(0.2)（→72/18/10）而非文档写的
  8:1:1，v2 已修正。
