# KSVD 从零路线：IMDB-BINARY R1-B consensus basis atlas

> 日期：2026-08-01  
> 状态：**descriptive protocol frozen before execution**  
> 角色：R1-A gate 存在结构性失配后，用无 gate atlas 直接展示 learned atoms 对应的真实 held-out patches。

## 1. 输入

只复用 R0-D 的 raw/exact-isomorphism-grouped 5-fold artifacts：

```text
split seed = 731301
patch sampling seed = 20260731
s = 7, d = 21, K = 12, T = 2
updates = 25, deterministic INIT, restarts = 0
```

不重新训练、不使用 labels 选择 atom/fold/patch。

## 2. Atom alignment

固定 grouped fold 0 为 reference，不按任何质量指标选 reference。对 fold 1–4：

1. 计算 `abs(D_ref.T @ D_fold)`；
2. 用 exact maximum-sum assignment 对齐 12 个 atoms；
3. 用 dot-product sign 对齐方向；
4. 报告每个 reference atom 在其他 folds 的 cosine mean/min；
5. aligned atoms 归一化平均，仅用于 continuous consensus 描述。

## 3. Held-out exemplars

每个 fold 只编码该 fold 的 grouped outer-test patches。对每个 aligned atom 固定取 absolute coefficient top-5；五 folds 合并后每个 consensus atom 最多 25 个 held-out exemplars。

每个 atom 报告：

- cross-fold matched cosine mean/min；
- exemplar graph count；
- edge-count mean/std/histogram；
- unique WALK vectors；
- unique rooted-canonical signatures；
- canonical effective count 与 dominant mass；
- top-3 canonical representatives 的 edge list；
- descriptive-only label composition；
- consensus continuous atom 的 L1、positive-mass fraction、maximum coordinate。

## 4. 边界

R1-B 没有 PASS/FAIL gate，不修正 R1-A 的 formal classification。它只回答：

> 跨 fold 可对齐的 latent atoms，在真实 held-out patches 上最强响应到哪些结构族？

允许使用 `latent basis atom/component`。只有当 exemplars 呈稳定窄结构族时才使用 `motif-like`，不能把 continuous atom 本身阈值化后宣布为合法 motif。

Labels 只作 descriptive metadata，不得用于 atom 排序、命名或路线结论。
