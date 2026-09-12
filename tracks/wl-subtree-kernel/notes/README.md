# notes/

轨内决策与来源记录。结果数字在 `../results/`，不放这里。

## 源实现出处

- 旧 keyan 仓（备份：`~/life/archive/科研/keyan.zip`，2026-06-16 快照）
  - `phase2_structure_classification/src/models.py::WLKernelClassifier`
  - `phase2_structure_classification/experiments/archive/run_step2.py` Method D
- 原版流程：每 fold 新建 `WeisfeilerLehman(n_iter=3, normalize=True)`，
  `fit_transform(train)` + `transform(test)`，再 `SVC(kernel='precomputed', C=10)`
- 原版数据根：`/home/calendar/code/keyan/data/TUDataset`（旧路径，本轨改用 `<repo>/data/TUD`）

## v2 关键决策（2026-09-08+）

1. **节点标签按数据集显式选择**（`pyg_to_grakel(data, dataset_name)`，不再依赖隐式
   全局 `_DS_NAME`）：
   - REDDIT* → constant 0（v1 已定：无属性社交图，度标签哈希空间过稀疏；
     v1 改常数标签后 REDDIT-BINARY 0.7369→0.7640）
   - MUTAG/PROTEINS/DD/NCI1/ENZYMES → TU 原始 categorical labels
   - IMDB-*/COLLAB 等 → degree
2. **bio 数据核对**（`TUDataset(use_node_attr=False)`，data.x 为 one-hot）：
   - MUTAG: 188 graphs, x=(17,7), 7 类（与文献一致）
   - PROTEINS: 1113 graphs, x=(42,3), 3 类
   - NCI1: 4110 graphs, x=(21,37), 37 类
   - ENZYMES: 600 graphs, x=(37,3), 3 类
   - DD: 1178 graphs, x=(327,89)，one-hot 89 维；原始 `DD_node_labels.txt` 取值 1..89，
     实际出现 82 个不同值（21-30、84+ 等少数离群标签，其余多为 0 列）——
     这是 TU 原始文件的内容，不是 bug；argmax 等价还原（校验：行和=1、二值）。
3. **重复边 sanity check**：PyG TUDataset 的 `process` 已对 edge_index `coalesce`（无向
   双方向各一条）；当前代码再加 reversed 产生重复 tuple，但 grakel 以
   `nested_dict_add` 覆盖写导入，邻接不变。验证：MUTAG 前 20 图，
   A（edges+reversed）vs B（set 去重后对称展开）的 WL 核矩阵（n_iter∈{1,3,6}，
   normalize=True/False）max|Δ| = 0.0 → 保留现状，并把检查写入每次运行的日志。
4. **网格**：h ∈ {1..6}（GIN 对 WL 明确给出的范围）；C ∈ {0.001..1000} log 网格。
   **GIN 论文未公开其 C grid 的确切取值**，本网格只是避免最优解撞边界，
   不要声称是 GIN 原始 C grid。
5. **nested 内层默认 cv5**；holdout 保留为可选项，`--inner-val-frac` 默认 1/9
   （相对 outer-train 90%）→ overall 80/10/10（修正 v1 文档说 8:1:1 而实际 72/18/10 的不一致）。
6. **paperlike 协议**：non-nested CV model selection——theta* = argmax_(h,C) mean_cv(theta)
   在**同一批** folds 上选出，再报告 theta* 在这批 folds 上的 mean±std。
   有 selection bias；标为 "paper-like approximation"，不是 exact GIN protocol
   （GIN 未公开 WL 的具体选参实现）。
7. **统计口径**：主统计 = across-seed mean of repeated-CV means（± = std across
   seed-level means）；另给 all-outer-fold pooled mean±std。GIN 的 ± 是 fold std，
   两者不是同一种统计量，只能比较 central accuracy。
8. **结果目录**：v2 默认写 `results/v2-categorical/`；v1 旧结果保留在 `results/` 顶层
   （不覆盖，可追踪）。分解实验写 `results/decomp/`。

## 已知注意点

- grakel 0.1.10 与 numpy 2.x：`from numpy import ComplexWarning` 在 import 期失败，
  脚本内置 shim（`np.ComplexWarning = np.exceptions.ComplexWarning`）。
- TUDataset 下载源为 chrsmrrs.com（TU 官方镜像）；网络不可达时需代理或预置数据。
- 继承的事件：v1 单文件脚本未被 git 追踪，被 v2 覆盖前无历史提交。已用
  `run_wl_subtree_kernel_v1_deg.py`（v2 框架 + v1 语义）恢复，并验证 MUTAG strict
  seed0（10×10 完整跑）与 v1 存档结果逐位一致（max fold diff = 0.0）。

## social 数据集补跑（2026-09-09，nested + paperlike，见 run_logs/social_*）

- 前四个 social 数据集此前只有 v1 三档；本次用 v2 配置补跑 nested(cv5) 与 paperlike
  （10 种子 × 10 折 × 10 重复，grid h∈{1..6} × C∈{0.001..1000}）。
- 运行事故与处置：COLLAB（5000 图）nested 在 8 workers 时因每个 worker 临时
  4500×4500 核矩阵（×42 参数）导致内存碎片化/换页（swap 1.9G、load 9.5、
  worker CPU ~2%），seed1024 慢到 6.8h 才完成；结果确定性不受影响（已落盘），
  但后续 COLLAB 剩余 seeds 改用 **--workers 4** 恢复正常（~880s/seed）。
  结论：大图数据集（≥~4000 图）+ 多 worker 时每个 fold 的核矩阵切片拷贝会累积，
  建议 workers ≤ 4 并观察内存。
- REDDIT-BINARY nested 选参非常稳定：h=3 占 902/1000、C=10 占 999/1000
  （常数标签下 WL 迭代深度与 C 几乎不敏感）；COLLAB：h=2 占 44%、C=1 占 999/1000。
