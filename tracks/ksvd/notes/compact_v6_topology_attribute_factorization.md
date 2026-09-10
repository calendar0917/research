# Compact-v6: Topology–Attribute Factorization

> 实验对象：`luyin16-zinc-hierarchical-patch-relation-context-compact-v6-*`
> 基线：historical compact-v4-hinge（`...-compact-v4-topology-hinge`；
> selection 99,613 params；seed-0 valid `0.17006561887910357 @53`）
> 协议：`zinc-context-gap`（PyG ZINC `subset=True` official split；train→valid
> selection→frozen checkpoint；本阶段仅 seed-0/1/2，**test 未加载**）
> 代码：`experiments/luyin16/compact_v6_attribute_roles.py`（role primitives）、
> `zinc_patch_path_pooling.py`（`_AttributeEncoder` + patch 输入扩展）、
> `compact_v6_diagnostics.py`（shuffle/norm 诊断）、`compact_v6_analysis.py`（表/图）
> configs：`configs/luyin16/zinc_compact_v6_{factorized_role,count_control,capacity_control}.yaml`
> 测试：`tests/test_compact_v6_attribute_roles.py`（11 tests）
> 数据/表：`results/compact_v6/`、`results/compact_v6_diagnostics/`

---

## 1. Motivation：为什么停止 tokenizer coarse/fine 分支

前三轮已经把 exact typed-token 方向做成 negative：

- **correctness repair**（`typed_patch_tokenizer_correctness_repair.md`，Case C）：
  historical `pynauty.certificate` 实际是 **rooted uncolored topology token**；
  corrected complete invariant 语义正确，但 v4 valid **−0.0046** / test −0.0040。
- **fragmentation & rarity audit**（`corrected_token_fragmentation_rarity_audit.md`，
  Case D）：corrected 确实把 r2 vocab 6,784→15,218、parent 31→512 切碎，但 paired
  degradation 与任何 fragmentation metric `|ρ|≤0.06`，由 **baseline-difficulty
  redistribution**（`Spearman(baseline error, degradation) = −0.352`）主导。
- **baseline-difficulty / typed-refinement audit**
  （`baseline_difficulty_typed_refinement_redistribution_audit.md`，Case D）：退化为
  difficulty-locked **centre(bias)** shift；typed-child informativeness **无独立预测力**
  （partial −0.036）。

因此：**tokenizer-derived architecture 分支已关闭**（不做 coarse-to-fine、coarse
prior、adaptive gate）。本轮不再问"token 该粗还是细"，而问一个结构不同的问题：

> topology 与 chemistry 是否应该 **factorize**，而不是被做成笛卡尔积式离散 vocab？

## 2. Hypothesis

historical 表示之所以有效，是因为 coarse topology token → shared embedding 提供了
**统计共享**，而 chemistry 主要经 146D continuous descriptor 进入 MLP（并因此被
aliasing 意外共享）。corrected exact typed token 把 chemistry 也压成 **one-hot
identity lookup**，vocab 爆炸且无收益。

v6 假设：

> 保留 shared coarse topology embedding 不变；chemistry 改为一个**共享、可组合、
> permutation-invariant、conditioned on coarse rooted topological role** 的小 encoder，
> 而不是 exact typed-token embedding。

核心区分不是 coarse vs exact，而是 **shared topology vs compositional attribute
placement**。

## 3. Difference from historical tokenizer

**不改**。historical coarse token（`typed_tokenizer_v1_historical` / `pynauty.certificate`）
的 lookup、parent token、146D descriptor、pair/path pooling、center-context、graph head
全部逐位保持。v6 只在 patch encoder 输入上**追加** `e_attribute ∈ R^8`。

## 4. Difference from corrected exact tokenizer

v6 **不使用** `e_corrected_exact`。attribute 分支只查
`Embedding(28,4)`（atom type）与 `Embedding(4,2)`（bond type）两张小表，加结构 role
descriptor；没有 15k 行 lookup、没有 exact typed key 进入模型。corrected tokenizer
仅用于（未在本轮启用的）semantic/correctness oracle 角色。

## 5. Topological-role definition

对 radius-2 patch，在 **coarse rooted topology** 上（只保留 root designation、
incidence 结构、root distance；**丢弃** atom/bond type）：

```
incidence 图着色：
  cell 0        = {root}
  cell 1..d     = {distance = 1}, {distance = 2}
  cell d+1      = 所有 (untyped) bond-incidence 顶点
```

取 `pynauty.autgrp` 的 **automorphism orbits**，再用 `pynauty.canon_label` 的
canonical position 对 orbit 重新编号（`_canonical_orbit_ids`），使 orbit 编号与
node ID 无关。两个 atom 同 role ⇔ 存在保持 root、保持 dist 的 coarse automorphism
互换它们。

每个 atom 的 **structural role descriptor**（≤8 raw features）：

| idx | feature | 语义 |
|----:|---------|------|
| 0 | root_flag | 是否 patch root |
| 1 | distance_from_root / radius | root distance |
| 2 | degree_within_patch / 5 | 局部度（branch） |
| 3 | orbit_size / n_patch_nodes | orbit 大小 |
| 4 | is_boundary | distance == radius |
| 5 | n_farther / degree | 外向邻居比例 |
| 6 | n_same / degree | 同距邻居比例（ring closure） |
| 7 | n_closer / degree | 内向邻居比例 |

**不做**全局 orbit-id embedding（避免再造一个 vocabulary）；role 是 structural,
bounded, permutation-invariant。

## 6. Attribute primitive definition

- atom：`(atom_type, role_descriptor)`
- bond：`(bond_type, role_descriptor_u, role_descriptor_v)`
  - 两端按 **canonical position** 排序（`positions[u] <= positions[v]`），
    消除 node-ID 方向依赖。

encoder（共享、跨 patch 参数共享）：

```
x_atom = [ Embedding(28,4)(atom_type) ; role_desc ]            # 12D
x_bond = [ Embedding(4,2)(bond_type) ; role_u ; role_v ]       # 18D
h_atom = MLP_atom(x_atom)   (Linear(12,10)-ReLU-Linear(10,10)-ReLU)
h_bond = MLP_bond(x_bond)
atom_pool = mean_over_patch(h_atom)
bond_pool = mean_over_patch(h_bond)
e_attribute = MLP_attr([atom_pool ; bond_pool]) -> R^8
patch_input = [ patch_cont(146) ; patch_context(0) ; e_token(16) ;
                parent_emb(8) ; e_attribute(8) ]  -> patch_encoder
```

实现上 per-occurrence 张量经 `attribute_{atom,bond}_patch_index`（PyG 对含 `index`
的 key 自动按 `num_nodes`=patch 数偏移）scatter-mean 回 patch。

final fusion projection **零初始化** → 初始 `e_attribute = 0`，完整模型在
initialisation 处**逐位等价于 v4**（见 §12 Tests / Fig）。

## 7. Permutation invariance

role、primitive、pooling 全部 permutation-invariant：
- orbit 用 `autgrp` + canonical position 规范编号；
- bond 端点用 canonical position 排序；
- pooling 是 scatter-mean。

`tests/test_compact_v6_attribute_roles.py`：
- Test 1：branched / cyclic / symmetric 三类图，每图每个 center **≥120 次**随机
  relabel，pooled `(type, role)` 与 `(bond_type, role_u, role_v)` multiset **完全相等**；
- Test 2：same topology same attributes under permutation 相等；
- Test 3/4：adversarial（同 topology、同 atom/bond multiset、N 在 leaf vs branch）
  count-view 相同、factorized-view 不同；
- Test 5：count-control 对 placement permutation 不变；
- Test 6：`attribute_mode=none` 逐位 = v4（params 99,613）；
- Test 7：attribute 路径无 corrected exact-token lookup（embedding 仅 28/4 行）；
- Test 8：role fingerprint 版本化、稳定、随 radius 变化。

## 8. Model architecture

```
v4-hinge model (unchanged)
   +
_AttributeEncoder (shared across patches):
   Embedding(28,4) + Embedding(4,2)
   MLP_atom(12->10->10) , MLP_bond(18->10->10) , fusion(20->10->8, zero-init)
   scatter-mean pooling -> e_attribute (8D)
   -> concatenated to patch_encoder input (170 -> 178)
```

**不含**：message passing / GNN / Transformer / ring nodes / dictionary / KSVD /
frequency gate / coarse-to-fine gate / target-derived 特征。

## 9. Parameter audit

| component | params |
|---|---:|
| historical v4-hinge (selection) | 99,613 |
| atom type embedding (28×4) | 112 |
| bond type embedding (4×2) | 8 |
| atom MLP (12→10→10) | 240 |
| bond MLP (18→10→10) | 300 |
| attr fusion (20→10→8) | 298 |
| extra patch_encoder input (+8 × 64) | 512 |
| **v6 total (selection)** | **101,083 (+1,470)** |

+1,470 ≤ 2k 目标；≤ 105k budget（PASS）。count/capacity 与 factorized **参数量完全相同**
（同一 `_AttributeEncoder`）→ 三者是干净的等参数对照。

## 10. Controls

- **capacity-control**：同参数 branch，输入 = `e_patch`（既有 coarse token embedding）
  的确定性 tile/truncate。给"额外 MLP 容量 + 既有重复信息"，**不给**新 attribute 信息。
- **count-control**：同参数 branch，role descriptor / bond role 全部置零。给
  **chemistry composition**（atom/bond type multiset），**不给** placement。
- **adversarial**：同 topology、同 type multiset、不同 role placement → count-view
  相同，factorized-view 不同（Test 3/4）。

## 11. Seed0 results

| model | params | valid MAE | Δ vs v4 |
|---|---:|---:|---:|
| v4 historical | 99,613 | 0.1700656 @53 | — |
| capacity-control | 101,083 | 0.1625751 @57 | **+0.007491** |
| count-control | 101,083 | 0.1709307 @47 | −0.000865 |
| factorized-role | 101,083 | 0.1564208 @58 | **+0.013645** |

seed0：factorized ≈ **Strong GO vs baseline**（+0.0136 ≥ 0.008），且
factorized > capacity（**+0.006154**）、factorized > count（**+0.014510**）。
**但 capacity-control 本身 +0.0075** —— 容量/优化本身已经给出 GO 级收益，这是
seed0 结果的第一个警告信号。

## 12. Multi-seed results

（seeds 0–2；seed 3 因时间预算提前停止，未运行。paired Δ = v4 − v6，正 = v6 更好）

| seed | v4 valid | factorized valid | Δ |
|---:|---:|---:|---:|
| 0 | 0.170066 | 0.156421 | **+0.013645** |
| 1 | 0.163167 | 0.180184 | **−0.017017** |
| 2 | 0.174149 | 0.171662 | **+0.002487** |

paired mean **−0.000295**，median +0.002487，std 0.012671，**2/3** seeds 正向
（seed 1 factorized 明显变差）。

对照 multi-seed（Δ vs v4）：

| model | seed0 | seed1 | seed2 | mean | 正向 |
|---|---:|---:|---:|---:|---:|
| capacity | +0.007491 | −0.024601 | −0.002315 | **−0.006475** | 1/3 |
| count | −0.000865 | −0.013333 | +0.007469 | **−0.002243** | 1/3 |
| factorized | +0.013645 | −0.017017 | +0.002487 | **−0.000295** | 2/3 |

机制比较（正 = factorized 更好）：

| 对比 | seed0 | seed1 | seed2 | mean | 正向 |
|---|---:|---:|---:|---:|---:|
| factorized − capacity | +0.006154 | +0.007583 | +0.004802 | **+0.006180** | **3/3** |
| factorized − count | +0.014510 | −0.003684 | −0.004982 | **+0.001948** | 1/3 |

**判定（预注册 §32）**：paired mean −0.0003 < +0.003 → **NO-GO**。seed-0 的
"Strong GO" **不复现**，是 seed/init lottery（seed 1 反转）。

## 13. Easy/hard redistribution

复用上一轮冻结的 historical difficulty quintiles（
`baseline_difficulty_typed_refinement_audit/per_molecule_analysis.csv` 的
`difficulty_quintile`）。seed-0，n=200/quintile：

| quintile | v4 MAE | factorized MAE | Δ (v4−fac) | count MAE | capacity MAE |
|---:|---:|---:|---:|---:|---:|
| Q1 | 0.034839 | 0.063601 | **−0.028762** | 0.072466 | 0.065997 |
| Q2 | 0.070616 | 0.076383 | **−0.005767** | 0.086565 | 0.076266 |
| Q3 | 0.095517 | 0.093707 | +0.001811 | 0.100617 | 0.089232 |
| Q4 | 0.144247 | 0.135013 | +0.009234 | 0.132120 | 0.127984 |
| Q5 | 0.505109 | 0.413400 | **+0.091709** | 0.462885 | 0.453397 |

**Bulk safety gate（§34：Q1/Q2 combined degradation ≤ +0.002）：FAIL。**
Q1+Q2 combined v4 `0.052727` → factorized `0.069992`，**Δ = −0.017265**（factorized
在 easy bulk 明显更差）。seed-0 的全部收益来自 **Q5 hard tail**，而 Q1/Q2 退化 ——
这与 corrected exact tokenizer 的 **同一 redistribution 指纹**（easy bulk 变差、
hard tail 变好）一致。即 seed-0 的"GO"不是 bulk 改进，而是 tail-for-bulk trade。

## 14. Shuffle mechanism test

冻结 seed-0 factorized checkpoint（train-only vocab/standardizer，`save_state_dict`），
在 official valid 上重前向：

| diagnostic | clean MAE | shuffled MAE | effect |
|---|---:|---:|---:|
| attribute-type shuffle | 0.156421 | 0.156421 | **0.000000** |
| role-association shuffle | 0.156421 | 0.156421 | **0.000000** |
| count-control（同上两 shuffle） | 0.170931 | 0.170931 | 0.000000 |

- shuffle = **patch 内**置换（type shuffle 置换 atom/bond **type** assignment；
  role shuffle 置换 role descriptor assignment），保持每 patch 的 type/role multiset。
- per-molecule 预测 `max |clean − type| = 0.0`，`max |clean − role| = 0.0`（精确）。
- `e_attribute` 范数：mean **0.002749**，std **2.3e-10**，p50==p95 → 分支输出
  **近乎常数**（训练后 branch 权重被压到 ~1e-3–1e-6，fusion 输出前后依赖被抹平）。
- 对照：把整个 branch 输出强制置零，预测变化也仅 `2.4e-4`。

**结论**：冻结 v6 模型的预测**完全不依赖 attribute placement**；shuffle 不破坏
（也不改变）收益。机制 gate（§28/§39"role shuffle 破坏收益"）**不成立**。

## 15. Final verdict

**NO-GO — TOPOLOGY–ATTRIBUTE FACTORIZATION（Case D at the benchmark gate）。**

判定链：

1. **benchmark gate（§32）**：multi-seed paired mean **−0.0003 < +0.003** → NO-GO。
   seed-0 +0.0136 不复现（seed 1 −0.0170）。
2. **bulk gate（§34）**：seed-0 gain 是 tail-for-bulk trade（Q1/Q2 Δ −0.0173，
   FAIL），与 corrected tokenizer 同指纹。
3. **mechanism gate（§39）**：frozen factorized 预测对 attribute/role shuffle
   **精确不变**；branch 输出近乎常数（norm std 2e-10）→ 无 placement 使用证据。
4. **capacity**：seed-0 capacity-control 自身 +0.0075 → seed-0 的相当部分是
   容量/优化效应，不是 attribute 信息。
5. **count**：factorized−count mean **+0.0019 < 0.003** → 即使只看机制，
   "attribute placement 相对 chemistry composition 的额外价值"**未达 gate**。

唯一一致的信号：factorized **在 3/3 seeds 上优于等参数 capacity-control**
（mean +0.0062）。但由于 branch 已 collapse（shuffle-invariant），该差距来自
训练轨线而非 attribute 使用，**不构成 factorization 机制证据**。

按 §53 Case D：**关闭该分支**；不要继续增大 attr dim / 加深 MLP / 增加 role
features。可重开条件见 decision record。

---

## Tables / Figures

- Table A：架构（`results/compact_v6/summary.json`）
- Table B（seed0）：§11
- Table C（multi-seed）：§12
- Table D（difficulty quintiles）：§13 · `results/compact_v6/table_d_difficulty_quintiles.csv`
- Table E（mechanism）：§14 · `results/compact_v6_diagnostics/diagnostics.json`
- Figure 1：v4 vs factorized by quintile · `results/compact_v6/figures/fig1_difficulty_quintile.png`
- Figure 2：factorized vs count vs capacity vs seed · `fig2_variants.png`
- Figure 3：attribute branch norm（collapsed）· `fig3_attribute_norm.png`
- Figure 4：clean vs role-shuffled error · `fig4_role_shuffle.png`

## Q1–Q14

**Q1** historical v4 中 topology 与 chemical semantics 如何进入 patch encoder？
→ historical token = rooted **uncolored** topology 的 canonical adjacency（coarse
topology 的共享 lookup，token_width 16）；chemistry 经 146D continuous shell
descriptor（atomic shell proportions、shell-pair bond proportions、root/incident
chemistry、size/cycle scalars）进入 patch encoder。即 topology 走 discrete shared
lookup，chemistry 走 continuous descriptor。

**Q2** v6 改变了哪一条数据流？
→ 只改 **patch encoder 的输入**：追加 8D `e_attribute`（shared compositional
encoder over (type, topological-role) primitives）。其余（token lookup、parent、
pair、pooling、center-context、head、loss、optimizer）完全不动。

**Q3** topological role 如何定义，为什么 permutation-invariant？
→ coarse rooted incidence graph 的 **automorphism orbits**（`pynauty.autgrp`），用
canonical position 规范编号；node 描述用 8 个 bounded structural features。orbit
与 canonical position 对 relabel 不变 → permutation-invariant（Test 1，≥120 次
relabel 精确相等）。

**Q4** v6 是否完全避免 corrected exact typed-token lookup？
→ 是。attribute 路径只有 `Embedding(28,4)` + `Embedding(4,2)`；无 15k exact table、
无 `corrected_canonical_key` 进入模型（Test 7）。

**Q5** 新增参数多少？
→ **+1,470**（99,613 → 101,083 selection）；见 §9。

**Q6** capacity control 结果？
→ seed0 +0.007491（比 baseline 好）；multi-seed mean −0.006475（1/3）。即
"额外容量/既有信息"本身在 seed0 就能给出 GO 级收益 → seed0 结果被污染。

**Q7** count-only chemistry control 结果？
→ seed0 −0.000865；multi-seed mean −0.002243（1/3）。**composition alone 无收益**。

**Q8** factorized role-aware 是否优于 count control？
→ multi-seed mean **+0.001948**（1/3，< 0.003 gate）→ **未达"placement matters"
门槛**（Case B 方向）。

**Q9** seed0 valid improvement？
→ **+0.013645**（0.1700656 → 0.1564208）。

**Q10** multi-seed paired mean，几/4 同向？
→ seeds 0–2（seed 3 未运行，时间预算提前停止）：mean **−0.000295**，**2/3** 正向
（若按用户"只跑两个 seed"口径取 seeds 0–1：mean −0.001686，1/2）。

**Q11** Q1/Q2 easy bulk 是否稳定？
→ **否**。Q1/Q2 combined Δ **−0.017265**（FAIL ≤ +0.002 gate），与 corrected
tokenizer 的 redistribution 同向。

**Q12** role-association shuffle 是否破坏收益？
→ **否**。clean == role-shuffle，MAE 差 **0.000000**（精确），per-molecule 预测
max 差 0.0。模型不使用 placement。

**Q13** 当前数据支持 A/B/C/D 哪个？
→ **D. no useful signal**（在 benchmark gate）；机制上接近 B（composition-only,
placement unproven），但 factorized 甚至没有稳定的 composition 收益。
A（placement factorization）不支持；C（纯 capacity）也不完全——factorized 稳定
优于 capacity，但 branch 已 collapse，故该差异非机制。

**Q14** 唯一推荐下一阶段？
→ **关闭 topology–attribute factorization 分支**。不要增大 attr dim / 加深 MLP /
增加 role features / 打开 dictionary / KSVD。若要重开，唯一预注册路径是
**先修掉 branch collapse** 的初始化/正则问题并用一个 graph-side-observable 的
预注册 gate 重新检验（见 decision record `revisit_if`）。
