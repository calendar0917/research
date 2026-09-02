# 导师 50-node 真实子图：无标签 relation-token（关系令牌）pilot

> 日期：2026-08-06
> 协议：`tracks/ksvd/docs/KSVD_MENTOR_SUBGRAPHS_RELATION_TOKEN_PILOT_PROTOCOL_20260806.md`
> 上游协议：`tracks/ksvd/docs/KSVD_MENTOR_SUBGRAPHS_BEAM8_PILOT_PROTOCOL_20260806.md`；`tracks/ksvd/docs/KSVD_MENTOR_SUBGRAPHS_GROUPED_RECONSTRUCTION_PROTOCOL_20260806.md`
> 判定：`MENTOR_RELATION_SIGNAL_BELOW_GATE`（导师数据上的关系信号低于门槛）

## 1. 数据与 selection（样本选择）

- 源数据 SHA-256：`936134e7876740d5efc478f6bdf9663e31440ab506ecee507b00f8e32a7f05cb`；
- 缓存图数：`10000`；pilot 子集：`100`；
- pilot 根节点数：`93`；密度配额：`{'lt5': 20, 'd5_10': 20, 'd10_15': 20, 'd15_25': 20, 'ge25': 20}`（五个平均度分层各 20 张）；
- 几何配置：`s8_o2`（patch 尺寸 8、目标重叠 2，BASE 档）；cover seed：`970201`；
- 最大 patch 数：`60`；Beam `8`/R`1`；rooted-canonical slots（以根为锚的规范标号槽位）；
- 划分：`3` 折、`root_candidate_grouped`（根候选分组，同一根不跨折），seed `20260807`；组间泄漏：`0`；
- 源节点暴露（仅审计、不进特征）：node seen=`0.6766`，edge seen=`0.5862`；
- KSVD：K=`24`，T=`3`，T_min=`1`，updates=`25`；
  centering=train-only（仅训练折居中），INIT=deterministic maximin（确定性最远点初始化），FINAL=same-init（同起点学习后）；ridge alpha=`0.01`。
- token 族（token families）：`['INVARIANT', 'INIT', 'FINAL']`；分支（branches）：`['BAG', 'TRUE_RELATION', 'SHUFFLED_RELATION']`；目标 = 不变 patch 描述符（2·s+2 维）。
- 标签使用：`False`；特征中的源 ID：`False`；残差令牌：`False`；密度分层用作标签：`False`。

## 2. 总判定

- 完整性校验 gate：`True`；
- FINAL 关系 gate：`False`（未通过）；
- 不变描述符 gate：`True`；
- FINAL_TRUE vs INIT_TRUE gate：`True`（学习字典优于初始字典）；
- FINAL TRUE vs BAG 的 RMSE 改善：`0.2614`（26.14%）；
- FINAL TRUE vs SHUFFLED 的 RMSE 改善：`0.0108`（1.08%，不足 2%）；
- FINAL TRUE 逐折同时胜过两者：`[False, True, True]`（仅 2/3 折）；
- FINAL TRUE 逐密度层同时胜过两者：`3/5`（`{'lt5': True, 'd5_10': True, 'd10_15': True, 'd15_25': False, 'ge25': False}`，高密度层反超）；
- FINAL TRUE vs INIT TRUE 差值：`0.0043`（0.43%）；
- 不变描述符 TRUE vs SHUFFLED 改善：`0.0218`（2.18%，过线）；
- 预注册检查项：`{'integrity_passed': True, 'final_true_vs_bag_gain_at_least_002': True, 'final_true_vs_shuffled_gain_at_least_002': False, 'final_true_wins_both_all_folds': False, 'final_true_wins_both_strata_at_least_4_of_5': False, 'final_true_improves_init_true': True, 'invariant_true_vs_shuffled_gain_at_least_002': True}`。

> 关键解释：FINAL 字典在**词袋**（不看关系）下比**真实关系**绑定差（说明关系绑定本身有信息），但"真实关系 vs 打乱关系"只差 1.08%，低于 2% 的预注册门槛 → 关系绑定的增量在 KSVD 令牌上不足以成立。

## 3. Masked prediction（掩码预测，图级平衡，各折均值）

> 实验：把一张图里的某个 patch 整个藏起来（掩码），用其他 patch 的令牌去预测它的局部结构（度/谱/密度三角三个块）。RMSE 越低越好。

| 令牌族 | 分支 | 综合 RMSE | 度块 RMSE | 谱块 RMSE | 密度/三角块 RMSE |
|---|---:|---:|---:|---:|---:|
| INVARIANT | BAG | 0.16453 | 0.20201 | 0.08636 | 0.22149 |
| INVARIANT | TRUE_RELATION | 0.12960 | 0.15999 | 0.07106 | 0.16643 |
| INVARIANT | SHUFFLED_RELATION | 0.13249 | 0.16412 | 0.07423 | 0.16451 |
| INIT | BAG | 0.18611 | 0.22667 | 0.09779 | 0.25666 |
| INIT | TRUE_RELATION | 0.15013 | 0.18319 | 0.08356 | 0.19651 |
| INIT | SHUFFLED_RELATION | 0.14768 | 0.18030 | 0.08337 | 0.19081 |
| FINAL | BAG | 0.19739 | 0.23801 | 0.10499 | 0.27848 |
| FINAL | TRUE_RELATION | 0.14579 | 0.17625 | 0.08474 | 0.19000 |
| FINAL | SHUFFLED_RELATION | 0.14738 | 0.17979 | 0.08454 | 0.18817 |

> 读法：同一族里 `TRUE_RELATION < BAG`（关系绑定有用）、`TRUE_RELATION < SHUFFLED_RELATION`（真实关系优于打乱关系）。FINAL 的 true vs shuffled 差距极小（0.14579 vs 0.14738），即信号没有越过门槛。

## 4. 密度分层（FINAL，各折合并，按测试图加权）

| 密度层 | 图数 | BAG | TRUE | SHUFFLED | TRUE 同时优于 BAG 和 SHUFFLED |
|---|---:|---:|---:|---:|---:|
| lt5（度<5） | 20 | 0.18561 | 0.13372 | 0.14070 | True |
| d5_10（度5–10） | 20 | 0.25482 | 0.17340 | 0.18302 | True |
| d10_15（度10–15） | 20 | 0.16430 | 0.13590 | 0.13621 | True |
| d15_25（度15–25） | 20 | 0.19843 | 0.14253 | 0.13614 | False |
| ge25（度≥25） | 20 | 0.18434 | 0.14400 | 0.14120 | False |

> 高密度层（d15_25、ge25）上打乱关系反而更准 → 关系绑定信号在密集图上不稳定，是未通过的主因。

## 5. 逐图明细

每张图、每个令牌族、每个分支的 overall/degree/spectrum/density_triangle RMSE 见 JSON `folds[].branches[family][branch].graphs`（共 100 张测试图 × 3 族 × 3 分支行）。

## 6. 边界（本实验不可推而广之的点）

- 密度分层与 root_candidate 只用于划分/条件报告，不是标签；数值源 ID 不进入任何特征。
- 目标是不变 patch 描述符（2·s+2 维）；目标 patch 令牌从自身上下文中排除（BAG 删除目标行，TRUE/SHUFFLED 只使用其他 patch 下标）。
- TRUE/SHUFFLED 只差上下文的令牌对齐方式（overlap 比例与中心最短路径距离相同）；SHUFFLED 用循环平移打乱对齐。
- 完全不用残差令牌：被掩码的目标是不变描述符，残差会把目标内容泄漏进上下文。
- 本 pilot 不训练 Transformer、不使用图标签；只判定 FINAL 字典的关系令牌是否通过冻结门槛。
