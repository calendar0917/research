# Rooted-canonical KSVD 下游归因协议

> 日期：2026-08-02  
> 状态：结果前冻结。

## 1. 问题

rooted-canonical 45D KSVD 已通过 reconstruction 与 relabel audit。下一问题是：若 graph-level readout 最终仍需 pooling，KSVD 是否只是把直接图统计换成另一套统计量？

masked-patch 任务不能公平加入 whole-graph statistics，因为它会直接读取被 mask patch 的边。本轮改用真正 graph-level synthetic downstream：预测 generator cell `family × target degree`。

## 2. 数据与 targets

- 72张50-node graphs；
- families：`regular / small_world / block`；
- degree cells：`15 / 20 / 25`；
- primary：9-class joint cell balanced accuracy；
- secondary：3-class family 与3-class degree balanced accuracy；
- 3 folds 沿用 replicate-index split，所有9 cells 在各 fold 均出现；
- graph labels 只进入最终 readout，不进入 sampler、canonicalizer、dictionary或PCA。

## 3. Branches

1. `GLOBAL_STATS`：whole-graph sorted normalized degrees、sorted normalized adjacency spectrum、density、triangle density；
2. `ROOTED_RAW_BAG`：rooted-canonical 45D patch vectors的 mean/std/max；
3. `ROOTED_RAW_TRUE_RELATION`：raw patch tokens + true overlap/distance relation summary；
4. `ROOTED_KSVD_BAG`：K24/T3 sparse codes的 mean/std/max；
5. `ROOTED_KSVD_TRUE_RELATION`：codes + true relations；
6. `ROOTED_KSVD_SHUFFLED_RELATION`：相同维度但错绑 relations。

## 4. Matched readout capacity

每 fold、每 branch：

1. train-only feature standardization；
2. train-only PCA 到12维；
3. fixed-alpha ridge one-hot decoder；
4. argmax classification。

所有 branches 使用相同12维 graph representation 和同一 readout，避免高维 relation branch 获得额外容量。

## 5. Attribution

- `GLOBAL_STATS` vs others：直接统计是否足够；
- `RAW_BAG` vs `KSVD_BAG`：dictionary/compression 是否增加可线性读取信号；
- `TRUE_RELATION` vs BAG：组合关系是否增加信号；
- KSVD TRUE vs RAW TRUE：KSVD 是否在相同关系机制下有增量；
- TRUE vs SHUFFLED：排除只靠维度或 relation marginals。

同时报告 relabel+Beam8 resampling 后12D projected representation cosine。

## 6. 判定

Primary 使用 joint balanced accuracy：

- KSVD TRUE 相对 GLOBAL_STATS、KSVD BAG、KSVD SHUFFLED均提高至少0.02，且3/3 folds 对 BAG与SHUFFLED同时更好：`ROOTED_KSVD_RELATIONS_ADD_DOWNSTREAM_VALUE`；
- GLOBAL_STATS 不低于所有 patch branches超过0.02：`DIRECT_STATS_SUFFICIENT_FOR_SYNTHETIC_FACTORS`；
- relations有增量但KSVD TRUE不优于RAW TRUE：`RELATIONS_USEFUL_KSVD_NOT_NEEDED_FOR_READOUT`；
- 其余：`NO_CLEAR_DOWNSTREAM_ATTRIBUTION`。

## 7. 边界

- synthetic family/degree targets不是导师真实任务；
- ridge + handcrafted relation summary不是最终 attention model；
- direct statistics胜出只表示这些生成因素可被宏观统计解释，不否定KSVD reconstruction价值；
- patch branch胜出也必须在真实50-node labels上复验。
