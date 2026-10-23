# zinc_native_relation_v1：一轮原生关系交互能否稳定改善未见局部环境？

**Protocol:** `ssrc-zinc-native-relation-v1`
**Frozen code:** `85b849734587`（之后只有 tracked manifest 变化）
**Formal run code:** `da86150a9e337928671069a827294dd482160ebb`（与 frozen code 相同；六个 `train.json` 均记录此 commit）
**状态:** 完成。六个正式 run（P/R/N × seed 0/1）全部完成，official test **未读取**。
**自动 verdict:** `UNSTABLE_OR_MIXED`。这个字符串主要指 R−N 两个 seed 方向不一致；对 H1 的 R−P，两个 seed **都为负**，方向一致。

---

## 0. 一句话结论

在 P3 的原始计算图和原始向量幅度通道上加入**一轮、共享参数、rank-16 的原生关系交互**，没有带来可重复的 overall 或 225-unseen-ordinary 改善：
R 相对同轮 P 在两个 seed 上分别差 **−0.006380 / −0.001278**（均值 **−0.003829**），225-unseen-ordinary 分别差 **−0.006039 / −0.002054**。
参数完全相同的 permuted-label 控制 N 也几乎同样差于 P（−0.006181 / −0.005345），说明损失主要来自“多出一个可训练 message-passing 分支”本身，而不是关系标签的真假。关系标签确实被模型读到了（固定 R 权重，把 real 换成 null 会变差 0.00066/0.00050），但这点标签拟合不足以抵消分支带来的优化/容量代价，也没有超过 N 对 permuted labels 的拟合。official test 因冻结触发条件不成立而没有打开，因此本轮**不能**声称达到 test raw-y MAE < 0.060（0.05×）。

---

## 1. 问题、冻结与不可越过的边界

### 1.1 本轮问题

在保留 P3 预训练 encoder、原始 atom/bond/junction 对象、原始 `group_to_bonds → Null composer → reader` 计算图，以及“主路径 h0 不被强制归一化”的向量幅度通道的前提下：

> 只加入**一个真实的结构关系交互**，能否稳定改善 valid 上的 225 个 unseen-local 分子和 overall MAE？

三个假设预先分开：

- **H1:** R 优于同轮 P，且在两个 seed 上可重复；overall 均值 ≥ +0.001，225-unseen-ordinary 均值 ≥ +0.005，且 seeded W20/T100 不出现 >0.001 的反向。
- **H2:** R 优于参数完全相同、只把关系标签在 molecule × role stratum 内置换的 N；两个 seed 都为正，均值 ≥ +0.001，W20/T100 不反向。H2 控制的是“关系类型 ↔ token pair 的对应是否有价值”，不控制 pair set 的值，也不控制 message passing 是否必要。
- **H3:** 冻结候选在 official test 上 <0.060。只有 H1 overall 成立且两个 seed 都完成时才允许终端读取；否则不读。

### 1.2 与旧轮的区别

- 旧 HR v1 的失败发生在“在 junction/bond 编码之后加 LayerNorm/refine、再做关系”的路径上；本轮**不**在 junction 编码前加归一化，也**不**在 h1 后加 LayerNorm。
- 本轮 P 是**同轮、同 recipe、同数据、同 trainer** 的 P3 控制，不用历史 P3B 数字替代。历史 `xBssl_P3N_s0` 的 0.076224 只作为外部参照；同轮 P 的 seed 方差本身就有 0.076428–0.078611。
- 没有 checkpoint soup、没有 seed averaging、没有按 valid 曲线中途挑点以外的选择；best-valid 严格 `<`，平局保留更早 epoch。

### 1.3 冻结的文件

- `PROTOCOL.md`
- `configs/nr_v1.json`
- `analysis/frozen_manifest.json`（protocol/config/code/cache/SSL/timing 的 hash 绑定）
- `analysis/cpu_checks.json`（训练前全部 CPU gate）
- `analysis/timing.json`（GPU 计时与 600-epoch 外推）

---

## 2. 实现与强制检查

### 2.1 三个 arm

**P（同轮 P3 控制）**
`ad2_models.P3NullModel` 原样：P3 encoders → `group_to_bonds` → Null composer → reader，413,505 参数。构造时使用与 R/N 相同的 supervised seed 和同一 SSL checkpoint。

**R（real relation）**
P3 encoders 完全不动，不加 junction 前 LayerNorm/refine。已有 token 集合为“所有 junction token + 原始 bond token”，按 junction 在前、bond 在后沿 token 维拼接：

```
h0 = concat(w0, b0)            # (n_j + n_b) × 96，每个 token 仍是原始幅度
u  = LayerNorm(h0 + role_emb)   # 只在分支输入上归一化
message = segment_softmax(q(u), k(u), rel_emb[r]) · v(u)
h1 = h0 + Wo · message          # h1 后不再 LayerNorm
w1, b1 = split(h1)              # 原 P3 group_to_bonds/composer/reader 不变
```

关系模块：rank 16，q/k/v 无 bias，`rel_emb` 33×16，`rel_bias` 33，role embedding 2×96，Wo 初始化 std 1e-3，**没有 zero-init gate**。新增 7,185 参数，总 420,690（<500k）。

关系定义沿用 HR 已验证的锚点映射（P3 `dist_flat` 重算，300/300 分子与 HR cache 的 `r_u/r_v/r_rel` 完全一致）：

- 候选 pair：clipped anchor distance ≤3，或共享 biconnected ring block；
- `rel = 1 + 8·min(d,3) + 4·same_ring_block + min(A²,3)`；
- self loop = rel 0，每 token 恰一条；每条非 self 边双向各一条；
- 关系只在分子内，不跨 molecule。

**N（parameter-identical null）**
与 R 完全相同（参数、初始化、优化器、数据顺序、pair set、结构统计），唯一区别是 `r_rel` 在**每个 molecule × role stratum（JJ/JB/BB）内**独立置换；固定 RNG，缓存一次，**从不读取 y 或任何模型预测**。不是全局 rel-id 改名。

train cache 的置换统计：changed fraction **0.8406**；valid **0.8408**；熵均值 1.921/1.926；没有 homogeneous stratum（全部 30,000/3,000 个组都可置换）。这保证 N 的关系标签既非恒等，也不是简单重命名。

### 2.2 训练前检查（`analysis/cpu_checks.json`，全部 PASS）

| gate | 结果 |
|---|---|
| P 参数 = 413,505 | PASS |
| R/N ≤ 500k | PASS（420,690） |
| mature P3 构造 state 与 P 逐 tensor 相等 | max abs diff **0.0** |
| P/R/N 公共初始化完全相同 | `c58e7567…` 三者相同 |
| R/N 关系模块初始化完全相同 | `aab16068…` 相同 |
| SSL encoder hash 相同 | `006fabc1…` |
| 原生 encoder replay（R/N vs P） | a/b/w max abs diff **0.0** |
| bypass replay（跳过关系分支） | R/N vs P **0.0** |
| atom renumbering | P 0.0；R 1.19e-7；N 2.38e-7 |
| bond direction | P/R/N 均 **0.0** |
| batch vs single（R） | 3.58e-7 |
| synthetic empty-graph / 1-token | 有限 |
| 关系不跨分子 | true |
| self loop 每 token 一次、rel=0 | true |
| segment softmax 每 row sum=1 | max dev 5.36e-7 |
| real/null pair set 相同 | true；50,550/59,938 条 directed non-self 标签改变 |
| null transport 在 renumbering 下 | pair/label 完全相同；N 预测差 2.38e-7 |
| 关系分支梯度 | 10/10 tensor 非零、有限；common 22/22 有限 |
| real vs null 关系梯度不同 | max abs diff 1.07e-4 |
| HR relation replay | 10,000 train + 1,000 valid，0 mismatch |

这些检查回答的是“实现是否真实、是否等价、是否读到了关系”，不回答“是否有效”。

---

## 3. 结果

### 3.1 主表（best-valid checkpoint；row 172 始终保留）

| run | best valid | epoch | final valid | W20 | T100 | train@best | params | wall h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P_s0 | 0.078611 | 502 | 0.079085 | 0.079046 | 0.079079 | 0.004795 | 413,505 | 0.312 |
| R_s0 | 0.084991 | 395 | 0.085834 | 0.085805 | 0.085833 | 0.012363 | 420,690 | 0.587 |
| N_s0 | 0.084792 | 392 | 0.085819 | 0.085794 | 0.085879 | 0.014152 | 420,690 | 0.519 |
| P_s1 | 0.076428 | 375 | 0.078378 | 0.078317 | 0.078387 | 0.016524 | 413,505 | 0.314 |
| R_s1 | 0.077706 | 294 | 0.079916 | 0.079894 | 0.079925 | 0.025250 | 420,690 | 0.571 |
| N_s1 | 0.081773 | 485 | 0.082980 | 0.082896 | 0.082927 | 0.006754 | 420,690 | 0.510 |

`W20` = 任意连续 20 epoch 的 valid median 的最小值；`T100` = 最后 100 epoch 的 median。它们不是另一个可挑的 checkpoint，只是持续性诊断。

### 3.2 H1：R vs 同轮 P

| seed | R−P best gain | 95% paired bootstrap CI | W20 gain | T100 gain | 225 unseen-ordinary gain |
|---:|---:|---|---:|---:|---:|
| 0 | **−0.006380** | [−0.012819, −0.001231] | −0.006759 | −0.006754 | **−0.006039** |
| 1 | **−0.001278** | [−0.006467, +0.005009] | −0.001577 | −0.001538 | **−0.002054** |
| mean | **−0.003829** | — | −0.004168 | −0.004146 | −0.004047 |

正号 = R 更好。两个 seed 的 overall 和 225-unseen-ordinary 都**不是**正号；H1 overall 与 H1 unseen 均不成立。seed 1 的 overall 点估计接近 0，但配对 bootstrap CI 跨 0，且 W20/T100 仍为负；这不是“可重复的改善”。

注意一个容易误读的绝对数：R_s1 的 0.077706 与 P_s0 的 0.078611 看起来接近，但配对比较必须同 seed（R_s1 vs P_s1 = −0.001278），不能拿 R_s1 去比 P_s0。

### 3.3 H2：R vs 参数完全相同的 N

| seed | R−N best gain | W20 gain | T100 gain | 225 unseen-ordinary gain |
|---:|---:|---:|---:|---:|
| 0 | **−0.000199** | −0.000011 | +0.000046 | −0.000673 |
| 1 | **+0.004067** | +0.003003 | +0.003002 | +0.002939 |
| mean | +0.001934 | +0.001496 | +0.001524 | +0.001133 |

H2 要求**两个 seed 都为正**；seed 0 为负，因此 H2 不成立。均值 +0.001934 全部来自 seed 1；seed 0 的 R 与 N 实际上不可区分（W20 只差 1.1e-5）。不能把“seed 1 上 real 比 null 好”写成可重复的关系语义优势。

### 3.4 固定旧分组（best-valid；MAE，越低越好）

| run | all | seen_ord | unseen_ord | seen_tail | unseen_tail | ordinary | tail | tail−172 | ring-rich | ring-rich-unseen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P_s0 | 0.07861 | 0.04716 | 0.13487 | 0.19274 | 0.27942 | 0.06827 | 0.22741 | 0.17436 | 0.11639 | 0.31124 |
| R_s0 | 0.08499 | 0.04967 | 0.14091 | 0.24088 | 0.33176 | 0.07163 | 0.27723 | 0.19026 | 0.11858 | 0.34341 |
| N_s0 | 0.08479 | 0.05074 | 0.14024 | 0.24698 | 0.29154 | 0.07228 | 0.26480 | 0.18251 | 0.11565 | 0.31884 |
| P_s1 | 0.07643 | 0.04759 | 0.12966 | 0.16791 | 0.26598 | 0.06734 | 0.20714 | 0.16050 | 0.10636 | 0.29236 |
| R_s1 | 0.07771 | 0.04910 | 0.13171 | 0.11967 | 0.32848 | 0.06898 | 0.20319 | 0.19024 | 0.11499 | 0.31932 |
| N_s1 | 0.08177 | 0.04602 | 0.13465 | 0.25396 | 0.34223 | 0.06735 | 0.28927 | 0.19300 | 0.11394 | 0.34369 |

- 225-unseen-ordinary：R 两个 seed 都更差（0.14091/0.13171 vs P 0.13487/0.12966）。
- ring-rich-unseen-47：R 两个 seed 都更差（0.34341/0.31932 vs P 0.31124/0.29236）。
- seen_ordinary：R_s0 更差，R_s1 也略差；没有“只牺牲 tail 换 ordinary”的迹象。

### 3.5 row 172、tail 与“表面接近”的来源

row 172 的 y = −20.340498，始终保留在 headline 中。把 R−P 的 overall gain 拆开：

| seed | R−P gain | drop-row172 gain | row172 AE R / P | row172 对 gain 的贡献 | tail gain | tail drop172 gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | −0.006380 | **−0.004163** | 5.8437 / 3.6225 | −0.002217 | −0.049819 | **−0.015890** |
| 1 | −0.001278 | **−0.003441** | 1.0320 / 3.1922 | +0.002160 | +0.003949 | **−0.029742** |

两个 seed 的结论方向不同，但都不是“R 变好”：

- seed 0：R 在 row 172 上也更差；去掉 row 172 后仍然差 −0.004163，tail 去掉 row 172 后差 −0.015890。row 172 只解释约 35% 的 overall 损失。
- seed 1：R 在 row 172 上好很多（1.03 vs 3.19），把 overall 从“去掉 row 172 的 −0.003441”拉回到 −0.001278；tail 的 +0.003949 也完全来自这一分子，去掉后 tail 差 −0.029742。

因此，**row 172 不是 R 负结果的制造者；它反而掩盖了 seed 1 的负结果**。这正是“headline 永远保留 row 172、tail−172 只做诊断”的原因。任何“R_s1 的 tail 变好”或“R_s1 overall 接近 P”的说法，在去掉这一个分子后都消失。

### 3.6 机制 probe：分支不是死的，但标签收益很小

对 R/N 的 best checkpoint 做固定权重、real/null/bypass 三种前向（CPU，完整 1000 valid）：

| run | real MAE | null MAE | bypass MAE | real−null（负=R real 更好） | branch/main norm | α max mean | α entropy mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| R_s0 | 0.084991 | 0.085651 | 0.131925 | **−0.000660** | 0.268 | 0.478 | 1.730 |
| R_s1 | 0.077706 | 0.078206 | 0.116393 | **−0.000501** | 0.265 | 0.381 | 2.065 |
| N_s0 | 0.084871 | 0.084792 | 0.126174 | +0.000079 | 0.268 | 0.467 | 1.753 |
| N_s1 | 0.082004 | 0.081773 | 0.118057 | +0.000231 | 0.292 | 0.440 | 1.830 |

- 分支**没有塌缩**：branch/main norm ≈0.27，message norm ≈1.6–1.8，α entropy ≈1.7–2.1，relation gradient norm ≈0.09–0.13（占 common grad 的 2.4–3.2%），10/10 关系 tensor 有非零有限梯度。
- R 固定权重下 real 比 null 好 0.00066/0.00050：标签不是完全没被读。
- 但 N（训练时只见过 permuted labels）在 real 标签下并不比 null 差，而且 R−N 的 best gain 在 seed 0 为负、seed 1 为正。**“网络能拟合 real 标签”没有转化为“real 标签比 permuted 标签更有泛化价值”。**
- bypass 在训练后大幅变差（0.116–0.132），是因为 common 权重已经与分支共同适配；它不能当作“没有分支的 P”来读，只能说明分支在训练后的前向中是必需的。真正的无分支控制是 P。

---

## 4. 预算、执行与可复现性

### 4.1 GPU ledger（含失败）

| item | run id | state | GPU h |
|---|---|---|---:|
| prep（CPU） | `nr-prep-20261023-154127-87c58275` | completed | 0 |
| timing（config key 失败） | `nr-timing-20261023-154329-1e0df3bb` | failed | 0.0031 |
| timing（save path 失败） | `nr-timing-20261023-154612-0b6e2ab2` | failed | 0.0042 |
| timing（完成） | `nr-timing-20261023-154815-168a15b9` | completed | 0.0058 |
| timing2（最终 code 完成） | `nr-timing2-20261023-160019-8aeed9f8` | completed | 0.0058 |
| formal P_s0 | `nr-P_s0-20261023-160259-b026d171` | completed | 0.3122 |
| formal R_s0 | `nr-R_s0-20261023-160310-3f7697d8` | completed | 0.5872 |
| formal N_s0 | `nr-N_s0-20261023-162246-eadacc22` | completed | 0.5189 |
| formal P_s1 | `nr-P_s1-20261023-165538-7d7596ca` | completed | 0.3139 |
| formal R_s1 | `nr-R_s1-20261023-165552-f1290488` | completed | 0.5714 |
| formal N_s1 | `nr-N_s1-20261023-171546-0cb950e9` | completed | 0.5097 |

**Total ≈ 2.832 GPU·h / 5.0 上限**（含失败；analysis、score、probe 全在 CPU）。六个正式 job 都落在空闲 pool 上，queue wait 可忽略。从第一个 remote job（15:41）到最后一个完成（17:46）约 **2 h 05 min**，远低于 5 h wall cap。详细 JSON 见 `analysis/gpu_ledger.json`。

### 4.2 运行环境与 provenance

- host `res` / node `a100-2`；2× A100-SXM4-40GB；每个 job 1 GPU；driver **550.163.01**；torch **2.5.1+cu124**；CUDA 12.4。
- 所有正式 run 的 `train.json.git_commit = da86150a9e33…`，`git_dirty_tracked = false`，`official_test_loaded = false`。
- SSL：`tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`，sha256 `006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97`。
- Relation cache：train `f22059ea…`（92 MB，10,000 分子，595,637 tokens，9,040,772 non-self pairs）；valid `b5e457a8…`（1,000 分子，59,366 tokens，897,166 non-self pairs）。远程 prep 与本机 hash 一致。
- 每个 run 的 best/final checkpoint、valid_best/valid_final 预测、曲线、初始化 hash、关系 cache hash、null fingerprint、data-order stream hash 都记录在 `analysis/run_manifest.json` 与各 `run_extra.json` 中。权重本体保留在 `runs/<arm>_s<seed>/{best,final}.pt`（按 repo 约定 `runs/` 不入 git）。

### 4.3 选择与统计

- best-valid：严格 `<`，平局取更早 epoch；`W20` 与 `T100` 只作为持续性诊断，不参与 checkpoint 选择。
- paired bootstrap：10,000 次 molecule-level 重采样，seed 20261108；只估计 valid 分子抽样不确定性，不估计 training-seed 方差。CI 正号 = 第一个 arm 更好。
- 没有 test 读取；没有 checkpoint soup；没有 seed averaging；没有按 test 调参。

---

## 5. 对六个问题的直接回答

### Q1. P3 的原生计算图是否被真实保留？关系是否真的进入前向？

**是，而且可逐 tensor 复现。** P 就是 `P3NullModel` 原样（413,505 参数）；R/N 的 P3 encoders 与 P 在 eval 模式下 replay max abs diff **0.0**；跳过关系分支的 bypass 与 P 预测 max abs diff **0.0**；公共初始化三者 hash 相同；关系模块初始化 R/N 相同；SSL 三者相同；atom renumbering、bond direction、batch-vs-single、空图/单 token、关系不跨分子、self loop、softmax 归一化全部通过；关系分支 10/10 tensor 有非零有限梯度。关系不是“接上但没跑”。

### Q2. R 相对同轮 P，在两个 seed 上是否可重复改善 overall 或 unseen？row 172 是否支配结果？

**没有可重复改善；两个 seed 的 overall 和 225-unseen-ordinary 都为负。** R−P = −0.006380 / −0.001278（mean −0.003829）；225 组 = −0.006039 / −0.002054。W20/T100 两个 seed 也都为负。row 172 **不支配**这个负结论：seed 0 去掉 row 172 仍差 −0.004163；seed 1 去掉 row 172 差得更多（−0.003441），是 row 172 上一个异常好的预测把 overall 拉回到接近 0。tail 同理：seed 1 的 tail 表面 +0.003949 在 tail−172 上变成 −0.029742。row 172 必须保留，否则会得到相反的错误叙事。

### Q3. R 相对 N 如何？N 控制了什么、没控制什么？

R−N = −0.000199（seed 0）/ +0.004067（seed 1），均值 +0.001934，但**方向不一致**，H2 不成立。N 控制了：参数量、公共初始化、关系模块初始化、优化器/recipe、数据顺序、pair set、结构统计、训练步数；只改变 molecule × role stratum 内的关系标签（84% 标签改变，熵 ~1.92，无 homogeneous 组），且从不读 y/预测。N **没有**控制 pair set 的值本身，也**没有**控制“是否需要 message passing”（R/N 都有分支，P 没有）。因此 R−N 只能回答“关系类型与 pair 的对应是否有额外价值”，不能回答“关系交互是否必要”；后者由 R−P/N−P 回答，而两者都输给 P。

### Q4. 225 unseen + ordinary 如何移动？overall 代价多大？

两个 seed 的 225-unseen-ordinary 都变差：+0.006039 / +0.002054 的 MAE 损失；overall 损失 +0.006380 / +0.001278。没有出现“unseen 改善、overall 只付很小代价”的交换；unseen、ordinary、ring-rich-unseen 47、tail 的诊断方向一致（除 seed 1 被 row 172 翻转的 tail 均值）。ordinary_935 也变差（−0.003360 / −0.001641 gain）。所以这不是“局部环境变好但全局略贵”，而是**局部和全局一起变差**。

### Q5. official test 的 0.05×（<0.060）达到了吗？没达到的话是否未测？

**没有达到，也没有测。** 冻结触发条件要求 H1 overall 成立且两个 seed 都完成；H1 overall 为 false，因此 `test_trigger=false`，official test 从未加载（所有 run 的 `official_test_loaded=false`，analysis 中没有 `test_terminal.json`）。valid 上最好的 R 是 0.077706（seed 1），P 是 0.076428；历史 P3B 参照 0.076224；没有任何候选接近 0.060。不能把“未测”写成“可能达到”，也不能把 valid 数字外推成 test 0.05×。

### Q6. 这条线值得继续吗？如果只问一个最重要的问题，是什么？

**按本轮配置，不值得直接继续加同一个分支。** 证据是：分支确实在跑、确实读到了关系标签（real−null 在固定 R 上 −0.00066/−0.00050），但 R 仍然输给 P，且 R 与 N 的差异方向不稳定；损失主要来自“给 P3 增加一个 7k 参数的可训练 message-passing 分支”本身，而不是标签真假。继续微调 rank、层数、初始化或关系粒度，等于在已经观察到优化干扰的同一设计空间里搜索，容易过拟合 seed/valid。

如果只保留一个最重要的问题，我会问：

> **能否把关系信息注入 P3 而不新增一个会扰动 P3 优化轨迹的可训练 message-passing 分支？**

具体地说：关系信号是否可以作为一个**零初始化/参数免费的门控或条件**，作用在已有的 `group_to_bonds`/composer/reader 上，使得 step 0 与 P 完全等价、训练中只学习“是否/如何读取关系”，而不是让一个 7k 参数分支与 P3 共同重新适配？本轮不能回答这个问题；本轮能确定的是：**“原生图 + 一轮共享 rank-16 分支”这个具体形式，在两个 seed 上没有给出可重复的 overall 或 unseen 改善。** 在回答上面的问题之前，不应再花 GPU 在同一分支形式上做超参搜索。

---

## 6. 限制与非声明

- 只测了**一轮**共享关系交互、rank 16、33 个关系类型、固定的 distance/ring-block 关系定义；没有测多层、不同 rank、不同关系粒度、不同聚合方式。
- 只测了 2 个 supervised seed；paired bootstrap 只给 valid 分子抽样区间，不给 training-seed 区间。seed 1 的 R−P CI 跨 0，不能据此声称“R 一定更差”，但两个 seed 的点估计都为负，且 W20/T100 都为负，足以拒绝 H1 的“可重复改善”。
- 没有做 test 读取；没有用 test 做任何选择。
- P 与历史 P3B 的 0.076224 有 0.0004–0.0024 的差异；这是同轮 trainer/code 版本与 seed 方差的混合，本轮不把它解释成机制。所有 H1/H2 结论只用同轮 P。
- N 的置换是“分子内 × role stratum”的标签置换，保留了每个 stratum 的标签多重集和 pair set；它不检验“关系标签完全随机”的更强 null，也不检验“pair set 本身错”的 null。
- 关系分支的 bypass 在训练后不等价于 P；这是 co-adaptation 的预期结果，不能把 bypass 当无分支控制。
- 本轮没有做 representation probing、attention 可视化或关系类型逐类分析；这些是机制解释的下一步，不是本轮结论的依据。

---

## 7. 文件索引

- `PROTOCOL.md` — 冻结协议与判定规则
- `configs/nr_v1.json` — 冻结配置
- `analysis/frozen_manifest.json` — 训练前 hash/预算/时间绑定
- `analysis/cpu_checks.json` — 全部训练前 gate
- `analysis/hr_relation_replay.json` — 与旧 HR cache 的 300/300 关系重放
- `analysis/timing.json` — GPU 计时与外推
- `analysis/stage1_decision.md` — seed 0 后的判定便签
- `analysis/summary.json` — 六 run 汇总、contrast、冻结决策
- `analysis/main_table.md` / `groups.md` / `decision.md` — 自动生成的表
- `analysis/run_manifest.json` — 每 run 的 checkpoint/预测/初始化/cache hash 与执行环境
- `analysis/probe_summary.json` — real/null/bypass 与幅度/梯度诊断
- `analysis/gpu_ledger.json` — 含失败的 GPU ledger
- `analysis/scores/*.json`、`analysis/per_molecule/*.csv` — 每 run 指标与逐分子预测
- `runs/<arm>_s<seed>/{best,final}.pt` — 权重本体（git-ignored，按 repo 约定）
