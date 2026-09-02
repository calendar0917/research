# Attributed Beam8 content-aware relation Stage B1

> 日期：2026-08-14  
> 状态：INIT-incidence follow-up 后冻结，content-relation 分类结果不可见。

## 1. 动机

已有结果表明 localized INIT incidence 优于 BAG，但 slot/position/relation-degree metadata 反而
降低分类表现。旧 relation 只表达“有多少邻居、重叠多少”，没有表达“哪一种 patch 内容与
哪一种 patch 内容相连”。

本轮固定 INIT dictionary，不运行普通 KSVD updates，不使用 attention，只加入内容感知的
patch-to-patch message。

## 2. 固定表示

- 数据：TU Mutagenicity，split seed 0，3 folds，model seed 0；
- Beam8：合法 attributed s8/o2 BASE；
- dictionary：outer-train `K24/T3` deterministic maximin INIT，最多 3000 patches；
- local patch token：24D `|INIT code|` + 14D patch atom histogram；
- relations：
  - `CHAIN`：同 segment 中前后连续 patches；
  - `NONCHAIN_OVERLAP`：有 node overlap、但不是连续 chain 的 patches。

对每个 relation `r`：

`m_p^r = row_normalize(R^r) Z`

relation feature 为 `[m_p^r, z_p ⊙ m_p^r, |z_p-m_p^r|]`；没有该 relation 的 patch 对应
channel 全零。最终 patch token 为 local token 加两个 relation channels，再经 orbit-safe
node incidence mean/max 回写节点。旧 slot/center/position/relation-degree metadata 全部置零，
只保留 incidence count/coverage。

## 3. 变体

- `GINE_ONLY`；
- `INIT_LOCAL_CONTENT`：localized INIT patch content，无 patch message；
- `INIT_RELATION_TRUE`：真实 CHAIN/NONCHAIN_OVERLAP message；
- `INIT_RELATION_SHUFFLED`：保持 local token、patch graph、incidence 与 message-code multiset，
  只在同图内打乱用于邻居消息的 patch codes。

## 4. 分类与 checkpoint

共同 3-layer GINE、hidden64、edge attributes、zero-init FiLM。outer-train 内 80/20 inner
validation，最多 80 epochs、patience20；重置后在完整 outer-train 训练 selected epoch；
outer-test 只评估一次。

## 5. Gate

- classification：RELATION TRUE−GINE `≥+1pt`，至少 2/3 folds 正；
- relation increment：RELATION TRUE−LOCAL `≥+0.5pt`，至少 2/3 folds 正；
- relation binding：RELATION TRUE−SHUFFLED `≥+0.5pt`，至少 2/3 folds 正。

三项全通过才扩展 model seeds 1/2。仅 LOCAL 胜 GINE 时，保留 localized INIT route、停止
patch relation；TRUE 胜 SHUFFLED但不胜 LOCAL时，只能说明 binding 可检测，不能说明 relation
提高分类。
