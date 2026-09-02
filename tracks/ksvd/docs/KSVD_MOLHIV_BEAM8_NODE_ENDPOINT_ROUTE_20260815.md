# MolHIV Beam8 节点级 endpoint 路线报告（2026-08-15）

## 结论摘要

此前“Beam8 路线全部关闭”的结论需要修正。原始 chain、普通 atom--patch incidence、overlap graph、patch-size 搜索和 bond-edge update 仍然应关闭，但 **bond-endpoint 节点级回写**在完整 `ogbg-molhiv` official-train 内部 scaffold folds 上出现了可重复的 Beam-specific AUC 信号。

当前最值得保留的实现不是 patch 均值读出，而是：

1. 第一层 GINE 得到 atom states；
2. Beam8 patch 保留 canonical slot、内部 typed bond、completion/base role 和位置特征；
3. patch 内每条 bond 产生两个定向 `patch -> endpoint atom` 消息；
4. 消息带有 bond type、canonical pair、endpoint slot；
5. patch 上下文回写到具体节点，再进入后续 GINE。

当前最佳候选是 `bond_endpoint_structure_context`：只使用 patch 内部 bond summary 与 patch position/role，不再把 patch 内 atom slot states 粗粒度汇总进上下文。

不过它尚未稳定胜过强全局 `BAG` 分支：seed-0 三折均值略高，但只在 fold 0 胜出。因此现在应将其定义为“已证实存在的 Beam-specific 局部结构信号”，而不是“已完成的 MolHIV 最优模型”。

## 实验边界

- 数据：`ogbg-molhiv` 前 8000 图。
- 划分：仅 official-train 内部的 3-fold scaffold split。
- 没有编码、调参或评估 official-valid / official-test。
- 完整 fold 每次约 4100 fit / 2300 heldout，heldout positives 约 80--90。
- 主配置：EDGE100、patch size 8、overlap 2、retained beam 8、10 epochs、hidden 64、3-layer GINE。
- 所有 matched variants 在模型构造前重新使用相同 seed，minibatch 顺序一致。
- 512 图筛选会被约 20 个正例强烈误导；路线判断以完整 folds 为准。

## 完整三折 seed-0 结果

数值为 `ROC-AUC / AP`。

| Variant | Fold 0 | Fold 1 | Fold 2 | 三折均值 |
|---|---:|---:|---:|---:|
| GINE | .6993 / .1458 | .5894 / .0552 | .6043 / .0733 | .6310 / .0914 |
| BAG | .6939 / .1655 | .6613 / .1613 | .7022 / .1902 | .6858 / .1724 |
| endpoint full | .7297 / .2009 | .6625 / .1393 | .6599 / .1246 | .6840 / .1549 |
| endpoint shuffled | .7065 / .1976 | .6460 / .1385 | .6425 / .1709 | .6650 / .1690 |
| endpoint no-patch | .6282 / .1382 | .5999 / .0813 | .6319 / .0837 | .6200 / .1011 |
| endpoint graph-context | .6531 / .1583 | .6211 / .0862 | .6073 / .1073 | .6272 / .1173 |
| endpoint leave-one-out | .6895 / .1990 | .6453 / .1181 | .7194 / .2438 | .6847 / .1870 |
| endpoint slot-context only | .7172 / .1794 | .6002 / .1086 | .6090 / .0824 | .6422 / .1235 |
| endpoint structure-context | .7443 / .2221 | .6391 / .1248 | .6823 / .1746 | **.6886 / .1738** |
| structure-context shuffled | .7426 / .2050 | .6564 / .1317 | .6097 / .0910 | .6695 / .1426 |
| endpoint without canonical role | .6979 / .1966 | .6416 / .1372 | .6727 / .1533 | .6707 / .1624 |

## 关键归因

### 1. Beam patch 内容不是额外 atom MLP 的假象

完整 endpoint 相对 no-patch 的增益：

| Fold | AUC delta | AP delta |
|---|---:|---:|
| 0 | +.1015 | +.0627 |
| 1 | +.0626 | +.0580 |
| 2 | +.0280 | +.0409 |

三折全部为正。小样本 screen 曾得到相反排序，说明 MolHIV 不应继续用 512 图 prescreen 作为 go/no-go 依据。

### 2. 任意分子级上下文不能替代局部 patch

`graph_context` 三折平均 AUC `.6272`，接近 no-patch 和 GINE，明显低于 endpoint `.6840`。收益不是简单增加全局 molecule context。

### 3. 当前端点自身不是必要的信息泄漏

leave-one-out 从 patch slot 汇总中移除当前 endpoint atom，三折平均仍为 `.6847 / .1870`。这支持“邻域 patch 环境”解释，而不是重复编码 endpoint 自身。

但 fold 0 三 seed 中 leave-one-out AUC 平均 `.7027`，低于 full endpoint `.7177`；其优势更偏 AP，暂不作为主 AUC 路线。

### 4. 粗粒度 atom-slot 汇总是噪声，结构/位置更有效

seed-0 三折：

- slot-context only：`.6422 / .1235`
- structure-context：`.6886 / .1738`
- full endpoint：`.6840 / .1549`

fold 0 进一步拆解：

- internal-bond only：`.7197 / .1831`
- position only：`.7228 / .2068`
- internal + position：`.7443 / .2221`

两者都贡献信息，合并后最好。这与“原读出粒度太大、平均 atom content 会冲淡结构”一致。

### 5. canonical endpoint role 有效，但不是 AP 的唯一来源

移除 canonical pair/slot 后三折平均 AUC 从 `.6840` 降至 `.6707`。因此 canonical pair + oriented endpoint slot 应保留。

### 6. structure-context 的 Beam 对齐信号存在，但有 fold 异质性

aligned 相对 matched shuffled：

| Fold | AUC delta | AP delta |
|---|---:|---:|
| 0 | +.0018 | +.0171 |
| 1 | -.0173 | -.0070 |
| 2 | +.0726 | +.0836 |
| mean | +.0190 | +.0312 |

均值为正，但主要由 fold 2 驱动；必须继续用多 seed / 多 fold 约束，不能只报告平均值。

## Fold 0 三 seed 稳定性

| Variant | Seed 0 | Seed 1 | Seed 2 | 平均 |
|---|---:|---:|---:|---:|
| endpoint full | .7297 / .2009 | .6965 / .1682 | .7270 / .1730 | .7177 / .1807 |
| structure-context | .7443 / .2221 | .6988 / .1644 | .7301 / .1676 | **.7244 / .1847** |
| leave-one-out | .6895 / .1990 | .7041 / .1882 | .7145 / .1738 | .7027 / .1870 |

`structure-context` 在三个 seeds 上 AUC 都高于 full endpoint，平均 `+.0066 AUC / +.0040 AP`，因此它是当前主候选。

## 已关闭路线

- 原 chain / chain mapping：真实映射未稳定胜 mapping shuffle。
- 普通 atom--patch incidence：真实 incidence 未稳定胜 patch shuffle。
- all-overlap / base-overlap patch graph：平均收益小或退化。
- patch size 6/8/10 搜索：size 10 偶有 mapping signal，但整体性能下降。
- bond-edge update：未稳定胜 matched shuffle。
- endpoint + BAG 简单相加：完整 fold 0 低于纯 endpoint。
- base-only endpoint：不稳定；completion patches 不是纯噪声。
- graph-context / no-patch：不能解释完整 endpoint 收益。
- 仅 slot/atom-content patch context：明显弱于 structure-context。

## 当前路线判定

### 保留

`bond_endpoint_structure_context`：节点级、typed、canonical、局部 patch structure/position 回写。

### 次候选

`bond_endpoint_leave_one_out`：更偏 AP，可作为结构分支的低相关补充，但需更多 seeds。

### 不应再投入

继续调 chain、overlap graph、粗粒度 patch mean、单纯扩大 patch size，或者回到纯 graph-level Beam readout。

## 下一轮优先级

1. 对 `structure_context` 与其 matched shuffle 跑 folds 1--2 的额外 seeds，确认 fold 2 增益不是单 seed 偶然。
2. 将 internal-bond 与 position 做可学习的低容量门控，而不是直接相加；门控必须零初始化并配 matched shuffle。
3. 对 completion/base role 做 structure-context 专属消融，判断位置收益是否主要来自 completion coverage metadata。
4. 只在内部 folds 冻结结构后，再考虑 class weighting、fusion scale、branch dropout；不要用 512 图筛选超参。
5. 只有当 aligned structure-context 在多 fold / 多 seed 上稳定胜 shuffled，并至少不弱于 BAG，才冻结配置并进入 official-valid；official-test 仍保持封存。

## 实现与结果

- Runner：`tracks/ksvd/code/run_molhiv_beam8_incidence.py`
- 数据构建：`tracks/ksvd/code/molhiv_beam8_incidence.py`
- 单测：`tracks/ksvd/code/test_molhiv_beam8_incidence.py`
- 结果：`tracks/ksvd/results/molhiv/beam8_bond_endpoint_*.json`

注意：`beam8_bond_endpoint_structure_subcomponents_full_fold0_seed0_20260815.json` 是修复 variant 路由前的无效运行，不得引用。有效结果文件带有 `_fixed_20260815.json`。
