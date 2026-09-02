# Clean rooted-WL 与导师 patch 统计差距诊断

日期：2026-09-01  
协议：`luyin16-clean-local-distribution-readout-probe-v1`  
数据：OGBG-MolHIV official train/valid（37,014 图；train 32,901，valid 4,113）  
官方 test：本实验未编码、未评估。

## 结论

当前路线的主要问题不是结构—属性融合原则错误，而是 graph-level readout
把所有中心 patch 只压成了一个一阶均值。导师 proxy 保留的是每个中心的
patch population distribution（mean/std/分位数/极值/稀疏率）。在完全相同的
clean rooted-WL radius-2 局部对象上，仅恢复这一读出就能带来约 +0.023 ROC-AUC；
再加入 205D 全局 composition/topology block 后，已经接近此前导师 proxy 的
validation 数字。

## 受控比较

局部对象固定为：radius-2 all-centre induced ego、2-round rooted-WL、64 node
role bins、32 edge role bins、strict chemistry（40D atom + 13D bond）。
XGBoost 使用 frozen `T+A` 参数，5 个模型 seeds（0--4）；只改变 graph-level
readout/是否加入全局 `S`。

| view | dim | valid ROC-AUC |
|---|---:|---:|
| `T+A mean` | 154 | 0.794468 ± 0.003303 |
| `T+A distribution` | 1793 | **0.817198 ± 0.002827** |
| `S + T+A mean` | 359 | 0.802208 ± 0.002013 |
| `S + T+A distribution` | 1998 | **0.827574 ± 0.004311** |

这里的 `distribution` 是对每个中心的 149D marginal row 逐坐标计算导师式
12 个统计块，再附加一次 5D graph context；不是把 context 重复 12 次。
其 mean block 与原 `T+A` primitive block 的最大误差小于 `2e-6`。

### 统计块的快速归因

3 个模型 seeds 的固定参数方向探测：

| readout blocks | valid ROC-AUC |
|---|---:|
| mean | 0.793649 |
| mean + std | **0.818747** |
| mean + min/max | 0.801988 |
| mean + q10--q90 | 0.794001 |
| mean + abs-mean/RMS/nonzero | 0.807486 |
| all 12 blocks | 0.817243 |

当前增量主要由跨中心 `std`（局部环境异质性）提供；并非单纯维度越高越好。

## 为什么之前的 centered binding 没有解决它

当前 centered feature 是每个 patch 内的

```text
joint(role, attribute) - marginal(role) × marginal(attribute)
```

但随后仍执行

```text
mean over centres
```

因此它测试的是“单个 patch 内结构与属性是否耦合”，却没有保留“一个分子中
这些 patch 是均匀分布还是少数特殊环境 + 大量普通环境”的信息。二者是两个
不同轴：

```text
within-patch relation   ≠   across-centre population distribution
```

导师式 `R_raw` 的主要优势很可能来自后者；它不必依赖显式 centered binding。

## 与导师 proxy 的公平比较

此前导师-shaped radius-2 proxy 的 official-valid 主候选 `S+R_raw` 为 0.8307，
但其 `R_raw` 仍含 rooted adjacency slot、node-ID tie-break、8 节点截断与完整
histogram 不一致等审计问题。当前 clean `S+T+A distribution` 用不变的 WL
角色对象达到 0.8276，剩余约 0.003 的差距可能来自更直接的 rooted adjacency
slot 信息、属性 schema 差异或 XGBoost 参数尚未针对新维度重调，而不是原则性
失败。

需要特别区分历史 test 数字：旧 `S+R_raw` train+valid refit test 是 0.7804，
当前 clean late-fusion test 是 0.7839；旧的 0.8022 属于 validation 排名较低的
`S+R_final` 诊断候选，不能拿来作为 clean 主路线的直接基准。

## 不变性检查

对 64 个图各做 2 次随机重标号：distribution graph-level 最大漂移
`2.38e-7`，平均 `8.68e-8`；中心 row 按坐标排序后的最大漂移为 0。该读出
没有引入 node-ID 顺序依赖。

## 下一步判定

1. 将“先定义 invariant local patch object，再保留 patch population distribution”
   作为主路线；不要继续把 mean readout 当作结构—属性融合的最终形式。
2. 在新的 outer scaffold splits 上对 `T+A distribution`、`S+T+A distribution`
   和固定 50/50 late fusion 做一次等预算 Optuna；当前 official test 不再用于
   任何选择。
3. 若分布读出在重复 splits 上稳定，再研究更精细的对象：保留 local count、
   typed edge/context 条件分布，或用低维 block-wise binding distribution；暂不
   优先 K-SVD、attention 或手工 cycle。

## 冻结调参与终端 test

随后预注册并冻结三个 `S+local` 候选，在 official-train scaffold folds 上做 8-trial
Optuna；test 没有参与调参或候选选择。结果如下：

| view | train scaffold CV | official-valid | train+valid refit test |
|---|---:|---:|---:|
| `S+T+A mean` | 0.7776 | 0.8128 | 0.7697 |
| `S+T+A mean+std` | 0.7856 | 0.8266 | 0.7723 |
| `S+T+A all12` | **0.7895** | **0.8341** | 0.7779 |
| fixed 50/50 mean+std/all12 | — | — | **0.7814** |

五模型/seed 的 all-model ensemble test 为 `0.7849`。因此分布读出在 test 上仍有
小幅增量，但没有把 validation 的 +0.04 完整迁移；主要表现为官方 scaffold split
之间的分布差异，而不是 readout 机制被证伪。该 test 已消费，不能再据此修改参数、
统计块或融合权重。

完整冻结记录：

- [`frozen_manifest.md`](clean_distribution_terminal_audit/frozen_manifest.md)
- [`terminal_test.md`](clean_distribution_terminal_audit/terminal_test.md)
- [`terminal_test.json`](clean_distribution_terminal_audit/terminal_test.json)

实验入口：

- [`clean_patch_distribution_readout.py`](../../experiments/luyin16/clean_patch_distribution_readout.py)
- [`clean_distribution_features.npz`](clean_patch_distribution_readout/clean_distribution_features.npz)
- [`summary.json`](clean_patch_distribution_readout/summary.json)
- [`readout_block_ablation.json`](clean_patch_distribution_readout/readout_block_ablation.json)

实现文件在本次工作区中的 SHA-256：

```text
clean_patch_distribution_readout.py       72feda09e924e9836cc79b28527a749289230e9de129adc1318426a171aeb46f
structural_role_fusion_screen.py          fbea87d5d592ac2b876a129b98fc9600b3b0b3252b392d56c5b349908a9809cc
```
