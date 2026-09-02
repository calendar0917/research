# ZINC 长程、统计对象与 K-SVD 归因

## 结论

当前路线已经正式进入 `luyin16` 所说的 ZINC 长程问题域，但结果也确认：
现有 graph-level proxy 离导师描述的 `0.22` 以及论文 `0.0x` 量级仍很远。
在本协议内，扩大局部半径确实有稳定增益；然而 K-SVD 重建优化没有稳定转化为
下游收益，纯结构也不是 ZINC 的独立强预测器。当前最合理的定位是：

```text
全中心局部对象 + 分布统计提供主信号
更大 radius 提供条件结构增量
K-SVD code 只能作为可能的 side channel，不能替代原对象
属性决定主要任务信号，结构主要改变属性的组织方式
```

## 协议

- 数据：PyG `ZINC(subset=True)` official train/val/test，`10000/1000/1000`；
- 数据 URL：PyG ZINC 的 Dropbox 数据文件与 `benchmarking-gnns` split index；
- 数据预检：8 个 raw 文件全部记录大小和 SHA-256；
- 采样：每个原子作为中心，保留全部中心 patch；
- 半径：`1/2/3`，固定 `max_nodes=12`；另做 radius-3 `max_nodes=20` matched full；
- 字典：仅 official train patches，`K=24, T=3, 6 iterations`；
- 下游：XGBoost regression，MAE 越低越好；
- test：只对 validation 晋级视图评估，并在冻结后用 train+valid refit；
- suite：8/8 tasks success，可断点续跑，独立日志和状态文件齐全。

这不是导师未知上游 schema 的 exact replication。

## 主要结果

固定参数、多 seed 均值：

| radius / view | valid MAE | test MAE |
|---|---:|---:|
| global statistics + attributes | 0.6122 | 0.6382 |
| r1 local typed raw | 0.5920 | 0.6055 |
| r1 global + raw + KSVD-final | 0.5436 | 0.5674 |
| r2 local typed raw | 0.6032 | 0.6208 |
| r2 global + raw + KSVD-final | 0.5430 | 0.5599 |
| r3 local typed raw | 0.5836 | 0.5928 |
| r3 global + raw + KSVD-final | **0.5225** | 0.5400 |
| r3-wide global + raw + KSVD-final | 0.5256 | **0.5276** |

6-trial Optuna 只用于 baseline 与固定参数 validation 排名前列的候选：

| radius / selected view | valid MAE | test MAE after refit |
|---|---:|---:|
| r1 global + raw + KSVD-final | 0.5151 | 0.5475 |
| r2 global + raw + KSVD-final | 0.5139 | 0.5478 |
| r3 global + raw + KSVD-final | **0.4953** | 0.5271 |
| r3-wide global + raw + KSVD-final | 0.4967 | **0.5080** |

test 只用于本冻结 suite 的终端核对；后续不得根据这些 test 数字继续选择 view 或参数。

## 四个矛盾的当前答案

### 1. 统计应该统计什么

单独的全局长程统计（距离、直径、偏心率等）很弱，纯全局结构 valid MAE 为
`1.3844`；全局属性为 `0.6507`。但 radius-3 typed patch 的逐坐标分布统计为
`0.5836`，与全局信息融合后为 `0.5602`。

因此不是“多算几个全局标量”就能解决长程问题。当前有效对象是带属性的局部结构
经验分布；下一步更值得保留 block-wise joint/co-occurrence，而不是继续增加互相高度
相关的边际统计量。

### 2. patch 应该怎么采样

所有半径均使用全中心，平均每图约 `23.17` 个 patch，节点覆盖率均为 `1.0`。
真正变化的是节点对覆盖率：

| radius | mean pair coverage | truncation |
|---|---:|---:|
| 1 | 0.2391 | 0.0000 |
| 2 | 0.4952 | 0.0001 |
| 3, max_nodes=12 | 0.6787 | 0.0615 |
| 3, max_nodes=20 | 0.6987 | 0.0000 |

radius-3 比 radius-1/2 更好，说明长程/关系覆盖确实重要。将 `max_nodes` 从 12
增至 20 消除了截断，但 valid 基本不变，test 更稳一些；所以截断不是主瓶颈，
半径带来的关系覆盖才是主要变化。后续采样目标应从“中心数量”转向“关系/节点对覆盖”。

### 3. K-SVD 怎么用、是否有用

K-SVD 始终明显降低训练 patch 重建误差：

| radius | typed INIT | typed FINAL |
|---|---:|---:|
| 1 | 0.1414 | 0.0592 |
| 2 | 0.2728 | 0.1760 |
| 3 | 0.4267 | 0.3220 |
| 3-wide | 0.4329 | 0.3306 |

但任务指标不一致。radius-3 standalone typed code 从 INIT `1.0330` 变为 FINAL
`1.0613`，反而更差；融合时 FINAL 在 valid 略好，但 INIT 在 test 明显更好。
topology code 也出现相同现象。结论是：

- 重建优化与回归目标不对齐；
- K-SVD code 不能作为原始局部对象的充分替代；
- 训练列初始化得到的真实 patch prototype 可能比更新后的抽象原子更稳；
- 当前最多能说 KSVD-derived view 在融合中有条件增量，不能说 K-SVD 更新本身有效。

### 4. 属性和结构怎么解耦、融合

纯属性显著强于纯结构：global attributes `0.6507`，global structure `1.3844`，
radius-3 local topology raw `1.2739`。但是 global all 与 radius-3 topology code
融合能到 valid `0.5261`（INIT）或 `0.5380`（FINAL）。

这说明结构在 ZINC 上主要是条件信息：它单独难以预测目标，却能描述原子/键属性如何
被组织。typed object 的优势不是“属性块 + 结构块维度更多”，而是保留两者在同一
patch 中的绑定。当前简单拼接已能产生增量，但还没有真正建模跨 patch 的关系。

## 对长程问题的判断

导师提出的“当前方法局部、长程不足”得到部分支持：radius-3 在完全相同的预算下
稳定优于 radius-1/2。不过显式全局距离统计本身没有解决问题，并且最好结果仍约
`0.50`。因此缺口不只是感受野大小，更可能来自：

1. 当前 rooted adjacency slot 只部分 canonical，tie-break 仍使用节点编号；
2. 逐坐标边际统计丢失 atom/bond/topology 的联合出现关系；
3. 图级 readout 没有表示不同中心 patch 之间的距离、重叠和相对位置；
4. K-SVD 仅优化无监督重建，没有任务对齐；
5. 缺少导师真实 69/624D schema 和上游 payload。

## 下一步停止与晋级规则

- 冻结本 suite，不再基于 official test 调参；
- 停止盲扫 K/T 和单纯增加边际统计；
- 下一优先级是 permutation-invariant 的局部对象与 block-wise joint statistics；
- 随后构造 patch-relation graph 或中心对距离/重叠统计，直接检验跨 patch 长程关系；
- K-SVD 只做 matched `INIT/FINAL/raw/residual` 归因；若多 train folds 上 FINAL 仍不稳，
  将它定位为压缩/诊断工具，而非任务表征学习器。

完整 JSON、日志、状态和自动摘要见
`suite_zinc_long_range_factorial_20260830/`。
