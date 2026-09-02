# MolHIV invariant patch-level joint reconstruction：2026-09-01

本轮只检查历史 `0.8022` 中最可能可迁移的机制：结构与属性是否必须在图级
读出之前进入同一个 patch 坐标，以及有效部分究竟来自普通低秩变换、经验原型
稀疏投影，还是 K-SVD 的迭代更新。

协议只使用 official-train 的三个 scaffold folds；每折 3000 train / 1500 valid，
固定 XGBoost，model seeds 0/1/2。official validation/test 均未编码或评估。

## 表示与对照

每个原子中心使用完整 radius-2 induced ego：

```text
96D topology-only rooted-WL roles
+53D strict atom/bond attributes（去掉 degree/is_in_ring）
=149D invariant patch row
```

四个 patch block 先按 categorical field mass 平衡，再统一做整行 L2。图级读出
固定为 mean+std，最后与冻结的 205D `S` 拼接。

候选：

- `marginal_mean_std`：原始四块分别保留，直接 mean+std；
- `joint_l2`：整行共享 L2 后读出，作为最小隐式融合；
- `bilinear`：train-only structure-PCA 8D 与 attribute-PCA 8D 的 patch-level
  outer product，再与 marginal readout 拼接；
- `pca_reconstruction`：train-only rank-32 PCA reconstruction；
- `ksvd_initial`：24 个真实 train patch 初始化原子、T=3 的零更新稀疏重构；
- `ksvd_final`：同一初始化做两次 K-SVD update 后重构。

matched null 将完整 attribute patch rows 在同一图内置乱，严格保留结构 patch
分布与属性 patch 分布，只破坏它们在同一中心 patch 上的对应关系。true 与 shuffle
的 patch multiset 均通过 24 图 × 2 次重标号审计，最大漂移为 0。

## 三 seed scaffold-fold 结果

| view | mean ROC-AUC | 相对 marginal |
|---|---:|---:|
| `S` | 0.6810 | — |
| `S+marginal mean+std` | 0.6811 | +0.0000 |
| `S+joint L2` | 0.6920 | +0.0109 |
| `S+bilinear` | 0.7087 | +0.0276 |
| `S+PCA reconstruction` | 0.7145 | +0.0334 |
| `S+K-SVD INIT reconstruction` | **0.7323** | **+0.0512** |
| `S+K-SVD FINAL reconstruction` | 0.7184 | +0.0373 |

主要机制 gate：

- shared `joint L2 − marginal`：`+0.0109`，2/3 wins；
- bilinear − marginal：`+0.0276`，3/3 wins；
- bilinear true − matched shuffle：`+0.0061`，2/3 wins；
- PCA true − matched shuffle：`+0.0087`，3/3 wins；
- K-SVD INIT − joint L2：`+0.0403`，3/3 wins；
- K-SVD INIT true − matched shuffle：`+0.0199`，3/3 wins；
- K-SVD FINAL true − matched shuffle：`+0.0123`，3/3 wins；
- K-SVD FINAL − INIT：`-0.0139`，1/3 wins；
- K-SVD FINAL − PCA：`+0.0038`，仅 1/3 wins。

## 重构误差与任务指标分离

三个 fold 上：

| transform | valid relative reconstruction error |
|---|---|
| PCA rank-32 | 0.140–0.145 |
| K-SVD INIT | 0.403–0.509 |
| K-SVD FINAL | 0.308–0.339 |

PCA 已保留约 `96.7%–97.0%` 的 train patch variance；K-SVD update 也显著降低
了 INIT 的重构误差。但二者都没有因此超过 INIT 的分类结果。尤其 FINAL 的
重构更好而 AUC 平均下降 `1.39pt`，再次说明无监督 reconstruction objective 与
MolHIV 标签目标并不对齐。

## 机制结论

1. **读出前融合成立。** 仅把四块放进同一个共享归一化坐标，就比独立 marginal
   读出高 `1.09pt`；true-shuffle 也为正，说明 patch-level 共现不是纯容量现象。
2. **显式低秩 binding 有真实但较小的净增量。** bilinear 相对 marginal 的大增量
   包含额外非线性容量；matched shuffle 后可归因于真实配对的部分约 `0.61pt`。
3. **当前最强机制是经验原型稀疏投影。** 零更新 INIT 字典由真实 train patch
   组成；它相对 joint L2 `+4.03pt`，相对 matched shuffle `+1.99pt`，均 3/3 wins。
4. **普通 K-SVD update 仍不成立。** FINAL 降低重构误差，却稳定不如 INIT；因此
   不能把本轮增量归因于 K-SVD 优化。
5. **历史 `0.8022` 的可迁移核心得到部分解释。** 旧方案在读出前对联合 patch 做
   共享归一化和原型重构，这一机制在干净不变对象上确实重新出现；但旧 FINAL
   的特殊 test 增益仍未复现，更可能混有 ordered-slot 捷径与 split 方差。

准确的一句话结论是：

> 结构与属性应先在同一个 invariant patch 坐标中形成联合原型，再做图级统计；
> 当前有用的是稀疏经验原型/量化带来的任务友好坐标，而不是继续最小化 K-SVD
> reconstruction error。

## 下一步边界

这只是 train-only 机制 screen，绝对 AUC 不与 official-valid/test 数字横比。候选若
进入下一阶段，应冻结为 `joint L2 / bilinear / PCA / empirical INIT` 四个家族，先在
完整 official-train scaffold CV 做等预算确认；只有 INIT 同时保持相对 marginal、
matched shuffle 和 PCA 的优势，才进入一次冻结 official-valid。不要扫描 K/T 或
继续增加 K-SVD iterations。

原始记录：

- [`invariant_joint_reconstruction_screen/summary.json`](invariant_joint_reconstruction_screen/summary.json)
- [`invariant_joint_reconstruction_screen_seedconfirm/summary.json`](invariant_joint_reconstruction_screen_seedconfirm/summary.json)
- [`invariant_joint_reconstruction_screen_seedconfirm/summary.md`](invariant_joint_reconstruction_screen_seedconfirm/summary.md)
