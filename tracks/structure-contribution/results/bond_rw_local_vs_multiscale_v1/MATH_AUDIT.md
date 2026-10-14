# 数学核查 — cscl-rw-local-vs-multiscale-v1（Phase 0）

- 协议 `cscl-rw-local-vs-multiscale-v1` · commit `fdcbbb43872fb20941623576fc05239fb8a0674c` · 500 个 split-D Fit 分子（标签盲）

## 1. K 与 Ks=(K+Kᵀ)/2：全图双重求和相等，逐键 r_e 不等

设 `r_e = Σ_f K_ef φ_ef`，`r'_e = Σ_f Ks_ef φ_ef`，`Ks=(K+Kᵀ)/2`；φ 取对称测试函数
`φ(b_e,b_f)=B_e+B_f+B_e⊙B_f`。

**全图双重求和相等**：每个无序对 {e,f} 在 Σ_e r_e 与 Σ_e r'_e 中都贡献
`(K_ef+K_fe)·φ_ef`，因此

```
Σ_e r'_e = Σ_e Σ_f Ks_ef φ_ef = Σ_e r_e = Σ_{e,f} K_ef φ_ef
```

数值核查（500 个分子）最大偏差 **3.55e-14**（K123 为 2.49e-14）。

**逐键 r_e 并不相等**（本轮修正）：上一轮“φ 对称 ⇒ 使用 K 与使用 Ks 相同”的说法
只在全图求和层面成立，而模型消费的正是**逐键** r_e。逐键有精确恒等式

```
r'_e = Σ_f (K_ef+K_fe)/2·φ_ef = ( r_e(K) + Σ_f K_fe φ_ef ) / 2 = ( r_e(K) + (Kᵀφ)_e ) / 2
```

数值核查：平均 `max_e|r_e−r'_e|` = **1.355**（K123 为 1.523），恒等式残差最大 3.55e-15（P 为 1.78e-15）。

历史影响：上一轮的 unit test `test_k_and_symmetrised_k_are_model_equivalent` 比较的是
`Ks` 与 `symmetrize(Ks)`（恒等），因此是空洞测试；上一轮模型实际使用 triplet of Ks
（`symmetrize` 后才建 triplet），历史预测/训练记录不受影响。本轮仅修订解释：
R1/R123 两臂使用的都是**对称化核** Ks，而不是原始 K；两者逐键消息不同。

**行权重被重新分配**：`Σ_f Ks_ef = (rowsum_e(K)+colsum_e(K))/2`。
对行随机的 lazy walk（每行和=1），Ks 的行和不再恒为 1：

| 量 | 均值 |
|---|---:|
| mean \|rowsum(symK)−1\| | 0.0728 |
| rowsum(symK) CV | 0.089 |
| rowsum(symK) min / max | 0.836 / 1.174 |
| corr(线图度, rowsum(symK)) | 0.888 |
| corr(线图度, colsum(symK)) | 0.888 |
| mean \|rowsum(symP)−1\| | 0.0573 |
| corr(线图度, rowsum(symP)) | 0.836 |

**解释边界**：因为 Pair MLP 带偏置，对称化后的行和变化等价于给每条键叠加一个与线图度/中心性相关的偏置项；这与“哪对键被组织在一起”是两种可能的效果来源。R1 与 R123 都使用 Ks，都含此效应；U 的行和恒为 1。因此 R1/R123 相对 U 的优势**不能**排除行和/度效应的贡献，需要在科学解释中与“真实成对组织”分开陈述。

结论：{"double_sum_equality_exact": true, "double_sum_equality_exact_K123": true, "per_bond_equality_holds": false, "per_bond_identity_verified": true, "symmetrisation_changes_rows": true, "row_sum_centrality_signal": true}

## 2. 历史合成任务 MAE 不一致的核查

- 历史观测：{"local_bond_random_walk_dry_run_0.3025": 0.3025, "remote_part_a_synthetic_witnesses_0.3755": 0.3755, "remote_mechanism_audit_0.7174": 0.7174}
- 同一函数 `synthetic_relation_task(seed=0)` 在三次不同进程中给出不同 real-K MAE。
- 本次对照：相同 `torch.manual_seed(0)` 两次 → 结果逐位一致？**True**；不同 `torch.manual_seed(12345)` → 结果不同？**True**。

原因：该函数的 numpy 数据采样有 seed，但 `_TinyPairRegressor`/`_TinyLinearRegressor`由全局 torch RNG 初始化，且函数内不调用 `torch.manual_seed`；不同进程（或此前是否调用过 `build_model`/`torch.manual_seed`）会得到不同初始化。

| torch 手动 seed | real-K MAE | uniform-K MAE | linear MAE |
|---:|---:|---:|---:|
| 0 | 0.2649 | 4.2790 | 7.6431 |
| 0 | 0.2649 | 4.2790 | 7.6431 |
| 12345 | 0.5908 | 4.5873 | 7.6431 |

**结论**：the historical spread of the real-K synthetic MAE is explained by the unseeded global torch RNG used for the tiny regressors; the qualitative ordering real-K < uniform-K < linear summary is unchanged in every run

历史 `REPORT.md`（0.7174，mechanism 进程）与 `SYNTHETIC_WITNESSES.md`（0.3755，synthetic 进程）均来自同一函数同一 seed 参数，差异是全局 torch RNG 初始化所致；两者作为定性 witness 仍然成立（真实 K 显著优于 uniform 与线性），但不应把具体数值当作可复现的定量结果。历史记录不改动。
