# Invariant-space KSVD capacity 跟进协议

> 日期：2026-08-02  
> 状态：K24/T3 invariant-space 结果后、capacity 结果前冻结。

## 1. 动机

K24/T3 invariant-space KSVD 已通过 relation 与 relabel stability gates，但 masked RMSE 相对 raw invariant TRUE 恶化 `9.23%`，超过注册的5%上限。需要判断这是方法失败还是稀疏容量过紧。

## 2. 冻结网格

```text
K = 16 / 24 / 32
T = 3 / 4
updates = 25
```

其他数据、Beam8 covers、22D invariant features、folds、relations、ridge 和 relabel permutations 全部相同。

## 3. 单 cell gate

沿用：

1. TRUE vs BAG >=2%；
2. TRUE vs SHUFFLED >=2%；
3. 3/3 folds simultaneous wins；
4. TRUE relabel cosine >=0.90；
5. TRUE RMSE 不比 raw invariant TRUE `0.1247372` 恶化超过5%；
6. invariants 通过。

## 4. 选择规则

若有 passing cells，按以下 lexicographic rule 选择：

```text
smaller T → smaller K → lower TRUE RMSE
```

即优先保持每 patch sparse code 更短，再减少 dictionary，最后比较误差。

有 cell 通过：`ADOPT_INVARIANT_KSVD_CAPACITY`；全部失败：`INVARIANT_KSVD_CAPACITY_GRID_FAILS_ERROR_GATE`。

## 5. 边界

- 增加 T/K 是明确容量代价；
- 本轮只寻找 synthetic masked-structure knee，不重新优化 invariant descriptor；
- 通过仍不等于真实下游胜过 raw invariant token。
