# MolHIV radius-2 atom-centered typed proxy

日期：2026-08-30

> **2026-08-31 审计更正：** 本页数值保留为历史协议结果，但原 52D
> patch 不能继续作为有效图表示做机制归因。其 28D 邻接槽位受 node-ID
> tie-break 影响；重标号后冻结模型的单图预测最大漂移达到 `0.3454`。
> 此外，14.12% patch 的邻接只保留 8 节点而 atom/bond histogram 使用完整
> ego，原子类别 `%16` 也会合并真实类别。因此 `0.8307` 不能再作为
> “K-SVD/纯结构机制成立”的证据。修正后的 invariant full-ego screen 见
> [`INVARIANT_PATCH_MECHANISM_SCREEN_20260831.md`](INVARIANT_PATCH_MECHANISM_SCREEN_20260831.md)。

## 结论先行

保留每个分子的全部 radius-2 原子中心 patch 后，624D typed readout 与
显式统计 S 产生了稳定增量；在 official validation 上，12-trial Optuna
（参数搜索只使用 official-train 内部 scaffold folds）得到：

| view | dim | valid ROC-AUC（5 seeds） |
|---|---:|---:|
| `S` | 205 | 0.7916 ± 0.0063 |
| `R_raw` | 624 | fixed classifier 0.7988 ± 0.0058 |
| `S + R_raw` | 829 | **0.8307 ± 0.0069** |
| `R_final` | 624 | fixed classifier 0.7935 ± 0.0115 |
| `S + R_final` | 829 | 0.8207 ± 0.0058 |

固定分类器结果和 Optuna 结果必须分开看：固定参数时 S=0.7817、
S+R_raw=0.8129；Optuna 只在 train 内部搜索后，最终 valid 为
S=0.7916、S+R_raw=0.8307。没有读取 official test（`test_evaluated=false`）。

## 协议

- 数据：OGBG-MolHIV official train/valid/test split；本阶段只评估 train/valid。
- patch：每个原子为中心，距离 ≤2 的诱导 ego 子图；`max_nodes=8`。
- 52D patch：中心优先 BFS 的 28D padded upper adjacency、16D atom-type
  histogram、8D bond-type histogram；直方图使用 OGB 的第一个 atom/bond
  类别通道（不是导师未知的完整化学 schema），每个 patch 做列归一化。
- 图级 patch 数：`max_patches_per_graph=null`，平均 25.51、最大 222；
  截到 8 个中心的版本作为对照，不能与本协议混报。
- 字典：K=64、T=8、T_min=2、6 次 K-SVD；24,000 个 train-only patch
  上限。由于 64>52，使用显式 overcomplete 初始化，而不是旧默认路径的
  K≤输入维度截断。
- typed readout：对 52D patch 矩阵逐坐标计算 12 个固定统计块，得到
  `12×52=624D` 的 `R_raw/R_init/R_final`。
- 下游：XGBoost binary logistic；screen 用 3 个 official-train-only
  Bemis–Murcko scaffold folds，晋级阈值为相对 S 不低于 -0.005 且至少赢
  2/3 folds；晋级后做 12-trial Optuna 和 5 个 classifier seeds。

配置与输出：

- [`mentor_r2_atom_chem_k64_s8_all_official_valid.yaml`](../../configs/luyin16/mentor_r2_atom_chem_k64_s8_all_official_valid.yaml)
- [`xgb_scaffold_screen.json`](mentor_r2_atom_chem_k64_s8_all_official_valid/xgb_scaffold_screen.json)
- [`xgb_optuna_promoted_12.json`](mentor_r2_atom_chem_k64_s8_all_official_valid/xgb_optuna_promoted_12.json)

## 逐步证据

1. 截断到每图 8 个中心的 full run 中，S=0.7817，S+R_raw≈0.7818；
   radius-2 patch 本身没有显出增量。
2. 全部中心的 300 图 smoke 只作方向检查；它提示 patch 数量可能是瓶颈，
   但小 validation 不能作为最终数字。
3. 全部中心的 full fixed run 中，S=0.7817，R_raw=0.7988，
   S+R_raw=0.8129；scaffold screen 中 `S+R_raw` 和 `S+R_final` 均 3/3
   folds 胜 S。
4. 正式 12-trial 结果中 `S+R_raw` 达到 0.8307，五个 seed 的 valid AUC
   为 0.8378、0.8262、0.8221、0.8298、0.8375；相对同 seed 的 S，五次
   全部提升，平均增量 +0.0391。

## 解释与边界

- 关键变化不是继续增大 K/T，而是从“每图 8 个中心”改为“所有中心的
  局部 patch 分布”。这使 624D 统计对象真正代表分子内部局部结构的
  分布，而不是少量随机中心。
- K-SVD 的重建误差从 INIT `0.1046` 降至 FINAL `0.0520`，但 raw 的
  valid 通常高于 final：压缩/重建更好不等于保留了最有分类信息的坐标。
  因此当前最强证据是 `S+R_raw`，不是 `S+R_final`。
- 这仍是独立概念 proxy，不是导师真实 69/624D 上游实现。结果达到并超过
  导师口述的约 0.81，只能说明该类“统计特征 + radius-2 typed local
  distribution + XGBoost”路线可行；不能声称已复现导师代码。

## 停止条件与下一步

当前应冻结 `S+R_raw` 的特征与 Optuna 参数，不再盲扫 K/T、Beam 或融合
深度。若要继续，优先向导师索取真实 69D/624D schema 和 payload，用本结果
作为候选机制对照；冻结前不读取 test，也不把本 proxy 数字写成最终论文主
结果。

## 冻结后 official test 评估

随后使用已冻结的 view/Optuna 参数查看 test；test 没有参与任何选择。下面的
`train+valid refit` 是最终模型常用口径，`train-only` 则是与 valid 阶段完全
同口径的 holdout 对照：

| view | train-only test | train+valid refit test |
|---|---:|---:|
| `S` | 0.7432 | 0.7540 |
| `S+R_raw`（valid 阶段选中的主候选） | 0.7805 | 0.7804 |
| `S+R_final`（预先保留的诊断候选） | 0.7965 | **0.8022** |

五个 seed 中，`S+R_raw` 和 `S+R_final` 都逐次超过 S；但 valid 上选出的
`R_raw` 与 test 上最好的 `R_final` 排名不同，说明单一官方 split 仍有明显
方差。因此 test 数字只能作为冻结后的外部检查，不能反过来把 `R_final`
改选成 valid 主模型，也不能据此继续调参。完整记录见
`mentor_r2_atom_chem_k64_s8_all_official_valid/frozen_test_evaluation.json`。

## pure-topology 对照

为区分局部拓扑和局部化学的贡献，保持 radius、全中心采样、K64/T8 和
XGBoost 搜索协议完全不变，只将每个 patch 改为 28D rooted adjacency，
因此 readout 为 `12×28=336D`：

| view | valid ROC-AUC（5 seeds） | train+valid refit test |
|---|---:|---:|
| `S` | 0.7916 ± 0.0063 | 0.7540 |
| `S+R_raw`（topology） | 0.8100 ± 0.0078 | 0.7627 |
| `S+R_final`（topology） | **0.8177 ± 0.0040** | **0.7751** |

typed 版本对应的 valid/test 是 `0.8307/0.7804`（raw）和
`0.8207/0.8022`（final）。因此当前更稳妥的归因是：radius-2 局部拓扑
分布本身有增量，局部 atom/bond 类型分布进一步提供增益；K-SVD 更新的
任务贡献依赖 split，不能单独视为全部提升来源。
