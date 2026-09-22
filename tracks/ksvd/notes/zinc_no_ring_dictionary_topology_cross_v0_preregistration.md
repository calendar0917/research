# DTX-v0 预注册 — No-Ring Dictionary × Generic Topology Cross

Round: **zinc-no-ring-dictionary-topology-cross-v0** (DTX-v0).
Dataset: ZINC strict-static（PyG `ZINC(subset=True)`；official train 10000 /
official valid 1000）。
Official ZINC **test 在本轮任何阶段都不得加载**（`test_policy = no_test`）。

本文件在**正式 run 之前冻结**；一旦 formal run 开始，以下任何定义、宽度、
阈值、协议都不得修改。

---

## 0. 本轮的唯一因果问题

> 一个共享 dictionary 从局部 static descriptor 学到的可复用 local
> environment，若与一个**完全 generic、untyped、label-free** 的 topology role
> vector 做 **aligned joint statistics**，是否比只保留两者 marginals 的
> matched control 更有预测价值？

关键点：新的 cross branch **不读取任何显式 ring/cycle 信息**；模型从未被告知
“这是 ring”。同时，本轮**不声称整个模型 ring-blind**：既有已经证明有效的
global topology hinge、raw patch descriptor、raw pair backbone 均保留。

---

## 1. 与已知教训的对应

1. dictionary 不替代 raw local / raw pair（SRDA 结论：raw pair 必须保留）。
2. SRDA 中 dictionary load-bearing 但绝对性能恶化 → 本轮 dictionary **只**服务
   graph-level cross statistics，不再改写 pair algebra。
3. aligned structure×attribute effect 必须与 marginal/capacity control 比较 →
   Arm M vs Arm A。
4. 不因 branch 被使用就宣布机制成功 → 必须同时看 mechanism intervention 与
   absolute MAE。
5. 显式 ring context 有信号但削弱“自发现”假设 → 本轮新 branch 完全禁止。
6. 导师 no-ring = 没有 handcrafted ring-context blocks，不是 ring-blind →
   本轮新 branch 比它更干净：**不允许任何 explicit ring label / cycle
   enumeration**。

---

## 2. 删除的 identity 通道（永久）

本轮不重新引入：

* typed embedding 16D；
* parent embedding 8D；
* historical / corrected token lookup、certificate lookup、fixed-code lookup。

模型唯一的 `nn.Embedding` 是 5 行 distance-bucket table。local state 只使用
既有 static continuous patch descriptor。

---

## 3. 冻结架构

### 3.1 local state（identity-free，S0 local width）

```text
x_i   = [patch_cont | patch_context]        # runtime width 146 (patch_context = 0)
h_i   = SiLU MLP(146 -> 64 -> 48)           # Linear, SiLU, Dropout(0.05), Linear
```

`h_i ∈ R^48`，是 unary 与 raw pair 的唯一 local 输入。

### 3.2 dictionary（local environment）

dictionary 输入 `x_dict_i` = `patch_cont` 删除**显式 cycle-rank 坐标**后的 145D。

审计结论（来自真实 `zpp._shell_descriptor` 实现，见
`cycle_rank_coordinate_audit`，合成图逐例验证）：

`patch_cont` 布局 = `atom_shell(3×28) + bond_shell(6×4) + root_atom(28) +
incident_bonds(4) + scalars(6)`；六个 scalar 为

| index | 名称 | 是否 cycle-derived |
|---|---|---|
| 140 | `log1p(n_nodes)` | 否（size） |
| 141 | `log1p(n_edges)` | 否（size） |
| 142 | boundary fraction（distance == radius） | 否（shell occupancy） |
| **143** | **`max(E−V+1,0)/V`（cycle rank / cyclomatic number）** | **是 → 删除** |
| 144 | root degree / 4 | 否（degree） |
| 145 | mean degree / 4 | 否（degree） |

因此 `CYCLE_RANK_FEATURE_INDEX = 143`，`DICT_INPUT_WIDTH = 145`。保留全部
atom shell / bond shell / root chemistry / incident-bond chemistry / size /
degree 坐标；不删除一般 shell topology 与 edge/bond structure。

dictionary encoder 与 assignment：

```text
z_dict_i = SiLU MLP(145 -> 64 -> 32)
q_i      = l2_normalize(z_dict_i)                 # R^32
D_k      = l2_normalize(dictionary_atom_k)        # K = 64
tau      = 0.05 + 0.95 * sigmoid(tau_logit)       # tau_init = 0.20
alpha_i  = softmax((q_i @ D_norm^T) / tau)        # R^64
```

全部由最终 ZINC MAE end-to-end 学习。禁止：K-SVD pretrain、OMP、reconstruction
loss、entropy loss、balance loss、orthogonality、top-k、sparsity penalty、
dictionary residual into `h_i`、`gamma`。**`alpha_i` 永不修改 `h_i`。**

dictionary 不读取 topology role vector。

### 3.3 generic topology role `s_i ∈ R^8`（绝对不枚举 ring）

对每个 patch root，在其 **untyped radius-2 induced patch** 内（`S0={root}`,
`S1=distance 1`, `S2=distance 2`, `n_patch=|patch|`）：

| idx | 名称 | 公式 |
|---|---|---|
| t0 | root_degree_fraction | `deg(root) / max(1, n_patch−1)` |
| t1 | shell1_occupancy | `|S1| / max(1, n_patch−1)` |
| t2 | shell2_occupancy | `|S2| / max(1, n_patch−1)` |
| t3 | shell1_edge_density | `#edges(S1,S1) / max(1, C(|S1|,2))` |
| t4 | shell1_shell2_edge_density | `#edges(S1,S2) / max(1, |S1|·|S2|)` |
| t5 | shell2_edge_density | `#edges(S2,S2) / max(1, C(|S2|,2))` |
| t6 | shell2_multi_parent_fraction | `mean_{v∈S2} 1[#neighbors(v)∩S1 ≥ 2]`；S2 空为 0 |
| t7 | same_shell_incidence_fraction | `mean_{v≠root} same_shell_neighbors(v)/max(1, patch_degree(v))` |

所有量均为整数计数之比，取值有界于 [0,1]，purely graph-topological、label-free、
deterministic、permutation-invariant。t7 的 float 累加使用节点无关的**排序后
multiset**，因此对节点重标号**逐位不变**（测试验证 50 次 relabel 严格相等）。

禁止在 `s_i` 中偷加：cycle rank、number of cycles、cycle length、ring
membership、cycle basis、simple cycles、induced-cycle list。

### 3.4 raw pair backbone（S0 风格，严格保留）

```text
u_i        = pair_projection(h_i)                 # 48 -> 16, no bias
gate_ij    = 1 + tanh(distance_gate_embedding(bucket_ij))   # Embedding(5,16)
rel_ij     = relation_encoder(pair_relation_ij)   # _MLPBlock(23 -> 32 -> 16)
pair_input = [u_i+u_j | |u_i-u_j| | (u_i*u_j)*gate_ij | rel_ij]   # 64
q_ij       = _MLPBlock(64 -> 64 -> 16)            # 每个 occurrence pair 只算一次
```

graph pooling 沿用现有 5-distance-bucket mean/std/log-count：
`pair_pool = 5 × (2·16+1) = 165`。

### 3.5 graph-level cross（本轮唯一新计算）

```text
mu_alpha = (1/N) sum_i alpha_i                    # R^64
mu_s     = (1/N) sum_i s_i                        # R^8
J_align  = (1/N) sum_i alpha_i outer s_i          # R^{64×8} = 512
J_indep  = mu_alpha outer mu_s                    # R^{64×8} = 512
e_cross  = SiLU Linear(512 -> 32)
```

`J` 是 graph-level sufficient statistic，**不能写回任何 local state**。

### 3.6 graph representation

```text
R_final = [ R_unary(97) | R_rawpair(165) | R_global(32) | R_topology_hinge(8)
          | mu_alpha(64) | mu_s(8) | e_cross(32) ]            # runtime width 406
y_hat   = GenericReader(406 -> 16 -> 16 -> 1)
```

existing global topology hinge（25→16→8）保留，但硬保证它**不进入**
`dict_encoder` / `alpha_i` / `s_i` / `J`；它只在最终 graph readout 与其它分支并列。

运行时宽度以真实 tensor 审计为准：`local 146→48`，`dict 145→32`，`alpha 64`，
`s 8`，`J 512`，`e_cross 32`，`pair_input 64`，`R 406`。
参数总量 runtime 审计 = **60,442**。

### 3.7 strict-static contract

```text
NO message passing
NO pair -> centre
NO pair -> local
NO topology/cross -> local update
NO recurrence
NO second pair evaluation
NO attention
```

---

## 4. 两个正式 arms（完全同参数）

* **Arm M — MARGINAL**：`cross = J_indep`
* **Arm A — ALIGNED**：`cross = J_align`

两者都提供 `mu_alpha`(64D) 与 `mu_s`(8D)。因此 dictionary composition 与
generic topology composition 都不是 Arm A 独占；Arm A 唯一额外信息是
`alpha_i <-> s_i` 的 within-graph local alignment。

Arm A/M 必须：相同参数总量、相同 tensor 名称/形状、shared init 逐位相同；仅
forward 中选择 `J_align` 或 `J_indep` 不同。

---

## 5. 硬性测试（formal run 前必须通过）

* A. `center_context=False`、`center_update=None`；把
  `zpp.PatchPathModel._pool_pairs_to_centres` monkeypatch 成 raise，forward 仍成功。
* B. 每个 forward：`pair_encoder` / `relation_encoder` / `dict_encoder` /
  `dictionary` call count 各 = 1。
* C. 改变 `s_i`：`h_i`、`alpha_i` 逐位不变；prediction 在 Arm A 中改变
  （非 vacuous）。
* D. within-graph random permutation `s_i -> s_perm(i)`：
  * Arm M：`J_indep` 与 prediction 变化 ≤ `1e-5`（CPU 实测 `7.5e-9`），且
    `J_indep` **按构造精确等于** `mu_alpha outer mu_s`；
  * Arm A：`J_align` 变化 ≥ `1e-2`，prediction 变化 ≥ `max(1e-6, 10 × Arm-M shift)`
    （随机 init 实测 `1.14e-2` / `2.3e-5`；训练后干预数值见 §8）。
* E. A/M 参数与 init parity（见 §4）。
* F. pair order / endpoint swap / within-graph patch relabel：prediction 变化
  < `1e-4`；role extractor 对 relabel **逐位不变**。
* G. 新增 `s_i` extractor 与 `DTXModel.encode` 的 AST 源级审计：不得出现
  cycle/ring/basis/aromatic/fused/spiro 符号，不得读取 typed/parent/structural
  token；`cycle_rank` 坐标由真实实现定位并删除（合成图验证）。
* H. real batch backward：`local_encoder`、`dict_encoder`、dictionary atoms、
  `cross_projection`、`pair_projection`、`relation_encoder`、`pair_encoder`、
  `head` grad > 0 且 finite；tau grad finite。

---

## 6. 训练协议（两 arm 完全一致）

```text
seed                 = 0
optimizer            = Adam
lr                   = 1e-3
weight_decay         = 1e-5
batch_size           = 128
loss                 = L1 / MAE
gradient clip        = 5.0
scheduler            = none
epochs               = 240（跑满，不 early stop）
checkpoint selection = best official-valid MAE
soup                 = fixed Top-5（valid MAE 最低 5 个，并列取最早 epoch）
                       equal parameter averaging（定义与既有实现完全一致）
```

记录：best valid MAE、best epoch、Top-5 soup MAE、soup members、每 epoch curve。

---

## 7. Deterministic A100 regime

正式 run 必须：

```text
torch.use_deterministic_algorithms(True)
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

（repo 既有 deterministic convention）。formal smoke 前确认 same seed、same
checkpoint init 下 GPU0 vs GPU1 的 short trace（2 epoch / 256 训练 / 128 valid）
逐位一致。

---

## 8. Alignment-use interventions（训练后，inference-only）

对 Arm A 的 soup checkpoint（official valid）：

1. **alignment removal**：`J_align -> J_indep`，其余不变；
2. **within-graph topology shuffle**：随机 permute `s_i`，保持 `mu_s`/`mu_alpha`
   的 multiset 不变。

记录：`mean |Δpred|`、`max |Δpred|`、valid MAE after、repeat-forward noise floor。

“机制明确”的门槛（frozen）：

```text
mean |Δpred| >= 0.010
且 mean |Δpred| >= 20 × repeat-forward noise floor
```

两种 intervention 都必须满足，才记为 `mechanism_clear = true`。

---

## 9. 预注册 outcome gates

```text
M_M = Arm M Top-5 soup MAE
M_A = Arm A Top-5 soup MAE
Delta_align = M_M - M_A          (lower MAE better)

Intervention_clear = §8 两种 intervention 均 mechanism_clear
```

* **Case A — STRONG ALIGNED SIGNAL**
  `Delta_align >= 0.004` 且 `best improvement (M_best - A_best) >= 0.003`
  且 `M_A <= 0.134` 且 `Intervention_clear`。
  → `NO_RING_DICT_TOPOLOGY_ALIGNMENT_SIGNAL`；允许下一轮 seed1 paired
  replication；本轮停止。
* **Case B — MECHANISM SIGNAL, ABSOLUTE WEAK**
  `Delta_align >= 0.003` 但 `M_A > 0.134`
  → `ALIGNMENT_SIGNAL_BUT_BASE_NOT_COMPETITIVE`。本轮不 rescue。
* **Case C — DIRECTION ONLY**
  `0 < Delta_align < 0.003` → `DIRECTIONAL_ONLY`；不购买 seed1。
* **Case D — NO SIGNAL / NEGATIVE**
  `Delta_align <= 0` → `NO_ALIGNED_CROSS_SIGNAL`；关闭 DTX-v0。

额外的 absolute safety check：

```text
若 M_M > 0.145  ->  报告 BASE_REGRESSION_CONFOUND
```

即使 aligned 有小 gain，也不得写成 architecture success。

---

## 10. 预算

```text
MAX FULL TRAINING RUNS = 2
GPU0: MARGINAL seed0
GPU1: ALIGNED  seed0
并行执行
```

不购买：seed1/2/3、K sweep、tau sweep、cross width sweep、topology feature
sweep、任何 rescue。

---

## 11. 参考性能（absolute context only；不重训旧模型）

```text
S0 seed0:      best 0.145674  soup 0.140794
S0 seed1:      best 0.139389  soup 0.136423
SDPK-v0 seed0: best 0.142193  soup 0.139735
SRDA-v0 seed0: best 0.155449  soup 0.149873
```

主因果比较是 **MARGINAL vs ALIGNED**，不是旧 S0。

---

## 12. 训练后 diagnostics（不得用于选择 / 不得回灌）

* Dictionary：active atoms、argmax-used、assignment entropy、effective atom
  count、top-8 mass、max avg atom mass、tau final、coherence。
* Alignment-use：§8 两种 intervention。
* Post-hoc ring（**允许显式 ring，仅解释**，`post_hoc_only=true`、
  `gradient_used=false`、`used_for_selection=false`）：
  * dictionary atom enrichment：`mean alpha_k | ring_any / ring5 / ring6 /
    multi_fused / ring_boundary` vs complement；
  * 8 个 `s` coordinate 在 ring vs non-ring 下的分布；
  * `J_align[k,p]` 高值组合在 post-hoc ring strata 中的富集。
  * 定义：`ring_any = belongs to an induced simple cycle (≤8)`；`ring5/ring6 =
    belongs to a 5-/6-cycle`；`multi_fused = #covering cycles >= 2`（fused/spiro
    节点必满足）；`ring_boundary = 非 ring 节点且邻接 ring 节点`。
  * 不得根据这些 diagnostics 改 architecture 或选择 checkpoint。

---

## 13. 禁止清单（本轮）

```text
explicit ring context input / ring enumeration in the new cross branch
ring5/ring6/fused labels / aromatic-ring context / ring dictionary
second topology dictionary / learned topology prototypes
K^2 dictionary pair matrix / dictionary pair algebra / SRDA tensor contraction
dictionary residual into h_i / gamma
message passing / centre update / attention / relation refresh / second pair pass
entropy / sparsity / reconstruction auxiliary losses
K / tau / cross width / topology feature sweep
seed1+ / official test
```

---

## 14. Durable artifacts

```text
tracks/ksvd/experiments/luyin16/zinc_no_ring_dictionary_topology_cross.py
tracks/ksvd/tests/test_zinc_no_ring_dictionary_topology_cross.py
tracks/ksvd/results/zinc_no_ring_dictionary_topology_cross_v0/   (local evidence)
tracks/ksvd/notes/zinc_no_ring_dictionary_topology_cross_v0_analysis.md
records/claims/... , records/decisions/... , STATE.yaml
```

必须记录：prereg commit、implementation commit、record commit、remote commit、
GPU IDs、deterministic settings、runtime、peak GPU memory、parameter parity、
shared init equality、两 arm best/epoch/soup、Delta_align、dictionary diagnostics、
alignment intervention、topology shuffle intervention、post-hoc ring enrichment、
strict-static tests、`official_test_loaded=false`。
