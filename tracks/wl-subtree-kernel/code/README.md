# code/

本轨实现（单文件，v2 起两个变体）：

- `run_wl_subtree_kernel.py` — **v2 主脚本**：
  - 节点标签按数据集显式选择（`pyg_to_grakel(data, dataset_name)`）：
    REDDIT* = constant 0；MUTAG/PROTEINS/DD/NCI1/ENZYMES = TU 原始 categorical labels
    （`TUDataset(use_node_attr=False)`；data.x 为 one-hot，argmax 还原）；
    IMDB-*/COLLAB 等 = degree
  - 网格：`h ∈ {1..6}`（GIN 对 WL 的明确范围）× `C ∈ {0.001..1000}`（log 网格；
    GIN 未公开 C grid，这不是 GIN 原始网格）
  - 协议：`strict-fixed`（h=3, C=10 先验固定）、`nested`（内层默认 **cv5**；
    holdout 可选且内层占比默认 1/9 → overall 80/10/10）、`paperlike`
    （non-nested CV 选参，标为 paper-like approximation）、`optimistic`
    （per-fold 乐观选择，仅诊断）
  - 默认输出：`../results/v2-categorical/`（v1 旧结果保留在 `results/`）
- `run_wl_subtree_kernel_v1_deg.py` — **v1-semantics 变体**（degree 标签 + 旧网格 + holdout 0.2），
  仅用于 v1↔v2 分解实验；已在 MUTAG strict seed0 上验证与 v1 存档结果逐位一致
  （max fold diff = 0.0）。v1 单文件脚本当时未被 git 追踪，被 v2 覆盖前没有历史提交，
  本文件由 v2 框架按 v1 文档语义恢复（上游源实现见
  `~/life/archive/科研/keyan.zip::phase2_structure_classification`）。

## 跑法（v2）

```bash
# 默认：REDDIT-BINARY + COLLAB，10 种子 × 10 折 × 10 重复
uv run --group wl-kernel python code/run_wl_subtree_kernel.py

# 正式 bio 实验：nested（推荐主结果）
uv run --group wl-kernel python code/run_wl_subtree_kernel.py \
    --datasets MUTAG PROTEINS DD NCI1 ENZYMES \
    --protocol nested --inner-strategy cv5 \
    --grid-iters 1 2 3 4 5 6 \
    --grid-C 0.001 0.01 0.1 1 10 100 1000 \
    --seeds 0 42 123 1024 2026 777 3407 999 111 888 \
    --n-splits 10 --n-repeats 10 --force

# 其余协议同参（--protocol strict / paperlike / optimistic）
```

环境：依赖走 `pyproject.toml` 的 `wl-kernel` group（grakel==0.1.10）。
注意 grakel 0.1.10 与 numpy 2.x 有 `ComplexWarning` 导入兼容问题，脚本内已带 shim。

数据：TUDataset 首次运行自动下载到 `<repo>/data/TUD/<NAME>`（默认 `--data-root`）。
全部 bio 数据集已预下载。
