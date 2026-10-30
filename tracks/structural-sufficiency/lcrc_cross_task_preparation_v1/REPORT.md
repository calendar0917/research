# LCRC 跨数据集准备 v1 — CPU 审计报告

- **Round**: `lcrc-cross-task-preparation-v1`
- **Scope**: `official-train-only`（官方 train 3360 行 = Fit 2352 + Monitor 504 + Reserved 504）
- **Device**: CPU only（`cuda_available=false`）
- **Training / optimizer / backward**: 未执行
- **Labels read**: 未读取
- **Final status**: **`READY`**（仅表示数据/特征/模型构造接口通过本轮审计；不表示模型在 Lipophilicity 上有效）

> 本报告回答 A–G。`READY` 的科学含义严格限定为“该接口可以进入下一阶段设计”，不是“LCRC-A 已在 Lipophilicity 上被验证”。

---

## A. 数据来源、划分完整性与标签隔离

### A.1 只读来源

| 来源 | SHA256 | 本轮用途 |
|---|---|---|
| `lipo_raw_v1.json.gz` | `8416837ea3e56e3e5218865391c88913bb16948b48b7a61be1ee890ea4dda6dd` | 唯一结构来源；schema `bjoc-lipo-rawgraph-v1`，4200 行，`label_free=true` |
| `split_phase1a.json` | `9fa96d49efd94cd2ff381922f83a07f963d832064c75469657b4ce5f68c332b6` | 唯一划分来源；protocol `ssrc-phase1a-lipo-v1` |
| `row_manifest.json` | `2c638e4814cb6d21ccec9d8c8af357d28ce210fb2c20c3d09b9c19afd476073d` | 行级 `compound_id` / canonical-SMILES 哈希 / 原子数 / 组分数 / scaffold 哈希 |
| `Lipophilicity.csv` | `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd` | **仅按字节流计算 SHA**；没有解析 CSV，没有读 `exp` 列 |
| `split_lipo_v1.json` | `7f0e0b991b0a0465bfe5fa30dc2ca77e40239aeecbdd0bc87322c9621c2393f8` | **仅用流式 reader 读取 `train_rows` / `valid_rows` / `test_rows` 三个行号数组**；标签尾部不解析 |

逐行核对结果：

- Fit/Monitor/Reserved 行哈希分别为
  `c474e38c...` / `fe81cc79...` / `bc28c5d3...`，与冻结值逐一相符。
- 三者并集大小 3360，且**与官方 `train_rows` 完全相等**
  （`union_minus_official_train=0`，`official_train_minus_union=0`）。
- 三者两两行重叠为 0；scaffold 组数 725 / 421 / 422，scaffold 组重叠也为 0。
- 对 3360 行逐一核对 `row_manifest`：`compound_id`、`n_atoms`、`n_components`
  全部一致；canonical-SMILES SHA256 **3360/3360 匹配，0 mismatch**。
- 所有 3360 行的 `n_components` 均为 1；本轮实际数据中**没有孤立原子**
  （degree 0 = 0），也没有 zero-path 分子。因此“disconnected / isolated atom /
  zero-path”支持是由合成边界用例验证的，不是由本轮真实数据验证的。
- raw artifact 物理上包含 4200 行；loader 会校验其 label-free schema 和
  `mol_index`，但只有 3360 个 in-scope 行被转换成 graph/cache。scope 外行
  没有转换、没有特征统计，也没有进入任何 gate。`cpu_audit.scope` 明确记录
  `raw_artifact_rows_loaded=4200`、`in_scope_rows_processed=3360`。

### A.2 scope 外样本

- `row 1646` **不在** 3360 行 union 中；官方归属为 `valid`。本轮**没有转换或
  处理它的结构，也没有读取它的标签**。
- 官方 valid/test 的结构与标签均不在本轮范围。scope 外样本不参与本轮统计、
  词表拟合或结论；disconnected/UNK 能力只用合成用例验证。
- 之所以不把它们“补”进 scope：本轮协议只允许官方 train 3360 行；把
  valid/test 结构拉进来会破坏划分纪律。disconnected/UNK 能力用合成用例单独覆盖。

### A.3 标签隔离声明

- `labels_fit_monitor.json`、`labels.npy`、`labels_meta.json`、官方 valid/test
  标签、CSV `exp` 列**均未打开/解析**。
- `audit_cpu.json` 的 `label_isolation` 明确记录：
  `labels_fit_monitor_opened=false`、`reserved_labels_opened=false`、
  `official_valid_test_labels_opened=false`、`csv_exp_column_parsed=false`、
  `label_values_parsed=false`。
- 需要如实说明：**早期环境检查时曾意外看到 `split_lipo_v1.json` 标签尾部约
  20 个 `train_y` 数值**。这些数值没有被复制进任何代码、缓存、特征、统计或
  报告，也没有参与任何计算。正式审计改用只读行号数组的流式 reader，其审计
  记录 `label_values_parsed=false`；`CPU audit` 与 `ZINC replay` 都没有标签字段。

---

## B. 显式 FeatureSpec、语义、词表、哈希与整数范围

### B.1 两个独立 schema

`analysis/feature_spec_lipo.json` 与 `analysis/feature_spec_legacy_zinc.json` 是
可序列化、可哈希的完整规格。两者不是补零/截断关系。

| 项目 | `legacy_zinc` | `lipo_v1` |
|---|---:|---:|
| atom feature dim | 108 | 51 |
| bond feature dim | 7 | 13 |
| atom scalars | 9 | 9 |
| global features | 33 | 42 |
| atom types | 21 | 16（15 个化学类 + `UNK`） |
| bond types | 3（1-based） | 5（`SINGLE,DOUBLE,TRIPLE,AROMATIC,UNK`） |
| S1 vocabulary | 63 | 80 |
| S2 vocabulary | 189 | 400 |
| structural code dtype | `int32` | `int32` |
| disconnected | reject | supported, finite statistics |

`lipo_v1` 的 51 维 atom 特征由以下块按固定顺序组成：

`atom_type(16) + formal_charge(6) + n_hydrogens(6) + degree(6) + in_ring(1)
+ min_ring_size(7) + ring_bond_count/4(1) + ring_bond_fraction(1)
+ incident_bond_class_fraction(5) + is_terminal(1) + is_fused_or_spiro(1) = 51`

13 维 bond 特征：

`bond_class(5) + is_ring(1) + exactly_one_endpoint_in_ring(1)
+ both_endpoints_in_ring(1) + stereo(5) = 13`

9 个 atom scalar：

`log1p_n_d2, log1p_n_paths2, closure_ratio, log1p_degree, in_ring,
min_ring_size_over_8, ring_bond_count_over_4, is_terminal, is_fused_or_spiro`

42 个 global 特征包括：原子/键/路径/组分计数、环与稠合环统计、直径与有限
距离统计、不可达对比例、键类型比例、环大小比例、formal charge/H 统计、
distinct type 数量等。每项的字段名、维度、dtype、化学/拓扑含义和来源均写入
JSON；没有 `exp`、label、target、penalty logP、SA score 或任何标签拟合量。

### B.2 化学名映射与 unknown 策略

- 原子类：0..14 为固定化学名（B, Br, C, C_aromatic, Cl, F, I, N, N_aromatic,
  O, O_aromatic, P, S, S_aromatic, Si），15 为 `UNK`。raw `atom_class=16`
  是历史 MASK sentinel，**显式拒绝**，不静默映射。
- 键类：raw 0..3 → 1..4（`SINGLE,DOUBLE,TRIPLE,AROMATIC`），raw 4 → 5 (`UNK`)。
- Fit 结构扫描实际观察到 atom class `0..14`、bond class `0..3`；没有观察到
  UNK。也就是说：**UNK 分支存在于 spec 并由合成用例验证，但本轮真实训练
  结构没有触发它**。这是明确记录的事实，不是把 UNK 隐藏掉。
- 词表策略是“事先固定的 label-free source vocabulary”，并由 Fit 结构扫描
  证明覆盖；没有从 Monitor/Reserved 或标签建立词表。

### B.3 S1/S2 联合码

- S1：`atom_type[neighbour] * 5 + (bond_type - 1)`，范围 0..79。
- S2 category：`atom_type[middle] * 25 + (bond_type_1 - 1) * 5 + (bond_type_2 - 1)`，范围 0..399。
- S2 end code：`atom_type[end] * 5 + (bond_type_2 - 1)`，范围 0..79。
- S2 closure：end 与 root 相邻则为 1，否则 0。
- 路径是 **ordered non-backtracking 2-path**；立即回退 `end == root` 被排除；
  三角形路径保留且 closure=1。

### B.4 dtype、shape 与整数溢出

`lipo_v1` 的 `N_PATH_CODES = 16 * 5 * 5 = 400 > 255`，因此：

| 数组 | dtype | 说明 |
|---|---|---|
| `atom_type`, `bond_type` | `int64` | categorical id |
| `nbr_code`, `path_cat`, `path_end63` | `int32` | 至少 4 字节 |
| `path_end` | `int64` | batch 内绝对 atom index |
| `path_closure` | `int8` | 0/1 |
| `atom_feat`, `bond_feat`, `atom_scalar`, `global` | `float32` | 特征 |

溢出证据：

- 合成 UNK 最大路径码 = `399`；`int32` 精确保存。
- 若用 `uint8`，`399` 会变成 `143`；测试显式记录该差异并禁止 `uint8`。
- 真实 Fit 的 `max_path_cat=350`，Monitor/Reserved 为 343；真实
  `max_nbr_code` 为 70/68。真实值虽未超过 255，但不能据此把 dtype 降为
  `uint8`，因为 spec 的合法上界是 399。
- 所有路径/S1 码均通过 `0 <= code < vocab` 检查。

### B.5 序列化与哈希

- `lipo_v1` spec hash：`7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee`
- `analysis/feature_spec_lipo.json` SHA256：
  `a65c67f9595d9190e7c913d2594ba984a07b51300d63916de865dccca434206c`
- `legacy_zinc` spec hash：
  `06710a5f003935a1fb3a1f27abf73265d370c03a984aa6ab839ae6220eecb5e0`
- provenance 记录：RDKit `2026.09.1`、Python `3.12.15`、NumPy `2.1.3`、
  Torch `2.5.1+cu124`、冻结来源哈希、Fit 观察到的原子/键类、protocol id。
- `source_provenance` 不含标签或标签统计；spec 中没有任何 label field。

---

## C. 结构正确性 Gates（G0–G7）

所有 gate 由 `code/cpu_gates.py` 实现，audit 与 pytest 共用同一套函数。
`analysis/cpu_audit.json` 中状态全部为 `PASS`。

| Gate | 检查 | 状态 | 关键证据 |
|---|---|---|---|
| G0 | Fit cache 重复构建确定性 | PASS | 两次 `content_fingerprint` 均为 `0a73407f...` |
| G1 | 来源/划分/scaffold/row-manifest 对齐 | PASS | 3360 行 canonical hash 全匹配；scaffold overlap 0；row 1646 不在 union |
| G2 | 特征语义 + 原子/键/路径不变性 | PASS | atom perm、bond swap/shuffle、path shuffle 全部通过 |
| G3 | 独立参考枚举 S1/S2 | PASS | 32 个真实分子 + 11 个边界用例逐项一致 |
| G4 | 数值有效性 + 整数溢出 | PASS | 3 个 split cache 有限、索引合法；399 的 dtype 测试通过 |
| G5 | batching/collation 隔离 | PASS | 单分子 vs batch 最大差 `4.77e-7`；索引 cross-graph 检查全真 |
| G6 | 边界分子 | PASS | 单原子、双原子、链、环、支链、charge、双组分、UNK 全部有限且分解正确 |
| G7 | LCRC-A 前向与 `Σ f(h_v)+rho` 分解 | PASS | 三个 split 的 `decomposition_max_abs_error=0.0` |

### G2 细节

- 16 个代表性真实分子，随机原子排列：atom feature、scalar、global 的
  `max_abs` 均为 0；S1/S2 多重集一致；预测最大差 `1.43e-6`。
- 16 个分子，随机交换每条键的两个端点并打乱键列表顺序：所有结构数组
  **精确相等**；预测最大差 0。
- 16 个分子，按 root 打乱 S2 记录顺序：S1 不变、S2 记录多重集不变；
  预测最大差 `1.43e-6`。
- 没有 attention、relation head、junction module 或额外 loss；模型
  `state_dict` 中不存在 `rel.`、`junction`、`attention`、`pair` 键。

### G3 细节

- 32 个真实代表性分子：共 1954 个 S1 code、2714 个 S2 path，与独立
  reference enumerator 完全一致；degree、bond 数也一致。
- 11 个合成边界用例：46 个 S1、32 个 S2，全部一致。
- reference enumerator 直接由邻接表嵌套循环实现，不复用生产枚举函数。

### G4 细节

- Fit/Monitor/Reserved 三个 cache 均通过 `LD.validate_cache`：
  finite、索引范围、指针单调、S1 度数和 = 2×bond 数、码范围。
- `path_cat` / `nbr_code` / `path_end63` 的 itemsize 均为 4；
  `path_end` 为 8；`path_closure` 为 1。
- 真实 Fit 的最大 path code 为 350；合成最大合法 code 为 399。
- 本轮真实数据：zero-path 分子 0、孤立原子 0、负计数 0。

### G5 细节

- 24 个分子组成 batch：单分子预测与整 batch 预测最大差 `4.77e-7`；
  子 batch（2/3/5）与整 batch 差为 0。
- 检查 `atom_batch[bond_index[:,0]] == bond_batch`、
  `atom_batch[bond_index[:,1]] == bond_batch`、
  `atom_batch[path_end] == path_graph`，全部通过。
- 没有跨分子边或跨分子路径污染。

### G6 边界用例

| 用例 | atoms | bonds | components | S1 | S2 | 参考一致 | 有限 | 分解误差 |
|---|---:|---:|---:|---:|---:|---|---|---:|
| single_atom | 1 | 0 | 1 | 0 | 0 | 是 | 是 | 0 |
| two_atom_single | 2 | 1 | 1 | 2 | 0 | 是 | 是 | 0 |
| three_atom_path | 3 | 2 | 1 | 4 | 2 | 是 | 是 | 0 |
| four_atom_linear | 4 | 3 | 1 | 6 | 4 | 是 | 是 | 0 |
| triangle | 3 | 3 | 1 | 6 | 6 | 是 | 是 | 0 |
| benzene | 6 | 6 | 1 | 12 | 12 | 是 | 是 | 0 |
| branched | 4 | 3 | 1 | 6 | 6 | 是 | 是 | 0 |
| charged | 2 | 1 | 1 | 2 | 0 | 是 | 是 | 0 |
| two_components | 3 | 1 | 2 | 2 | 0 | 是 | 是 | 0 |
| isolated_in_component | 3 | 1 | 2 | 2 | 0 | 是 | 是 | 0 |
| unknown_atom_and_bond | 3 | 2 | 1 | 4 | 2 | 是 | 是 | 0 |

注意：`triangle`/`benzene` 的 S2 含 closure=1 的路径；`two_components` 的
global 特征中不可达对被排除在有限距离之外，`frac_unreachable_pairs=2/3`，
`frac_finite_pairs_ge3=0`，所有值有限。

### G7 细节

- Fit 32 个、Monitor 16 个、Reserved 16 个代表性分子。
- 三个 split 的 `decomposition_max_abs_error=0.0`；
  `Σ contrib_atom` 与 `contrib` 的最大差分别为 `1.10e-6`、`2.89e-6`、
  `1.27e-6`（float32 累加误差）。
- `y`、`h`、`mean_h`、`mean_h2` 全部 finite。
- 在 cache 中注入 dummy `labels`/`exp` 后，collate 输出逐 tensor 不变；
  batch 中没有 label/target/exp 键。
- 源码静态检查没有发现打开 `labels_fit_monitor.json` / `labels.npy` 的调用，
  也没有 `optimizer.step` / `loss.backward`。

---

## D. 全量 CPU 审计覆盖

`analysis/cpu_audit.json` 处理了全部 3360 行；没有分子/原子/键/路径被静默丢弃。

| split | molecules | atoms | bonds | S1 | S2 | max path | max S1 | unknown atoms/bonds | isolated | disconnected | ring atoms | fused/spiro | charged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fit | 2352 | 63231 | 68688 | 137376 | 193288 | 350 | 70 | 0 / 0 | 0 | 0 | 41893 | 3137 | 117 |
| Monitor | 504 | 13400 | 14667 | 29334 | 41226 | 343 | 68 | 0 / 0 | 0 | 0 | 9407 | 747 | 12 |
| Reserved | 504 | 13874 | 15199 | 30398 | 42794 | 343 | 68 | 0 / 0 | 0 | 0 | 9727 | 778 | 28 |
| **合计** | **3360** | **90505** | **98554** | **197108** | **277308** | — | — | **0 / 0** | **0** | **0** | **61027** | **4662** | **157** |

Cache 证据：

| split | cache file | bytes | content fingerprint |
|---|---|---:|---|
| Fit | `cache/lipo_fit.pt` | 27,718,948 | `0a73407fe3b20bb5158b521bcd8dd94bac69d0857ec5c5bd31314c378e9990d9` |
| Monitor | `cache/lipo_monitor.pt` | 5,884,596 | `db48a9c237d628025185f76497a024dac35537c77413c511b1d76fb3721919b8` |
| Reserved | `cache/lipo_reserved.pt` | 6,091,320 | `8018c9f56ce713a15e82e72dc5a5e0d0ddf32ffd7ed4b30634d3068ec1b7c7a4` |

`cache/*.pt` 是本地可重建结构缓存，已被 `.gitignore` 排除；其中
`meta.labels_present=false`，没有 label 字段。

### Fit-only normalization

- `analysis/norm_lipo.json` 只使用 Fit cache；`valid_or_test_used=false`。
- 9 个 atom scalar 在 Fit 上均有非零方差。
- 6 个 global 特征在 Fit 上为常量，按显式策略使用 `scale=1.0`，不除以
  `1e-6`：
  indices `4,10,24,33,40,41`，分别对应 `n_components`、
  `frac_unreachable_pairs`、`bond_unknown_fraction`、`unknown_atom_fraction`、
  `unknown_charge_fraction`、`unknown_hydrogen_fraction`。
- 该策略避免了 scope 外/合成样本的常量特征被放大 10^6 倍；它只影响
  normalization 的数值稳定，不改变 LCRC-A 科学机制。
- norm fingerprint：`9c54f31ca26b6bc4b303958dfb465f16512fb6c576d23817aff2117d872d4286`。

### 运行成本

- Lipophilicity 全量审计：约 10 s CPU；peak RSS 约 653 MB。
- ZINC full replay：约 23.6 s CPU；peak RSS 约 1104 MB。
- 无 GPU；`cuda_available=false`。

---

## E. LCRC-A 前向与逐原子贡献分解

本轮没有修改 LCRC-A 的科学机制。模型仍是：

```
a_v = AtomEncoder(atom_feat_v)
b_e = BondEncoder([a_u + a_v, |a_u - a_v|, a_u ⊙ a_v, bond_feat_e])
h_v = LayerNorm(MLP([a_v, S1_v, mean_e b_e, S1joint_v, S2_v, scalars_v])) ∈ R^64
y   = Σ_v f(h_v) + rho(global_norm, mean_v h_v, mean_v h_v², log1p(n_atoms))
```

- 没有 attention。
- 没有递归 hidden-state message passing。
- 没有 junction head。
- 没有额外辅助 loss。
- `f(h_v)` 是逐原子贡献；`rho` 是全局修正；`y = Σ f(h_v) + rho` 在 float32
  下分解误差为 0.0。

`lipo_v1` 的随机初始化模型（seed 0，CPU，eval）：

| 项目 | 值 |
|---|---|
| parameters | 247,106 |
| breakdown | atom_encoder 19,040; bond_encoder 63,776; local 148,992; f 4,225; rho 11,073 |
| state hash | `e7a688ff91f29489e33568065f5fb85e17dfad5bc09b364a1ae8a08d76ecba6d` |
| input widths | atom 51; bond 13; global 42; scalars 9; nbr 80; path 400 |
| relation/junction/attention | 均无 |
| SSL loaded | 无（`ssl=null`） |

该模型**只是接口探针**：随机初始化、没有在 Lipophilicity 上训练、没有读标签，
因此不能把它的输出解释为 lipophilicity 预测。报告不比较 DCC-L/GINE/指纹，
也不声称 SSL 有收益。

为了说明 schema 驱动而非硬编码：同一 LCRC 构造器在 `legacy_zinc` 下参数为
237,519，在 `lipo_v1` 下为 247,106；差异来自不同的 atom/bond/global/path
词表宽度，不是性能比较。

ZINC 兼容性回放中的 `best.pt` 仅做 CPU eval forward；没有 optimizer step，
没有 backward，没有读取 ZINC 标签，也没有计算 MAE/RMSE。回放产物只保留
arm/seed/epoch 等 checkpoint 标识，不记录历史度量字段，也不将其用于结论。

---

## F. ZINC legacy 兼容回放

`analysis/zinc_compatibility.json`（full run，非 quick）：

### F.1 输入回放（P3 `MolObjects`，10,000 个）

- P3 train objects：10,000；SHA256 与冻结值
  `14326a7ae6f9bb673a5c8b35c7d54171a4008f77edb702f0591d3d1f21338e8d` 相符。
- 旧 `lc_data` 构造 vs 新 `legacy_zinc` adapter：17 个字段全部逐分子精确
  一致（14 个 LCRC-A 字段 + 3 个 pair 字段）：
  `atom_feat, atom_type, atom_scalar, bond_feat, bond_type, bond_index,
  nbr_ptr, nbr_code, path_ptr, path_cat, path_end63, path_end, path_closure,
  global, pair_i, pair_j, pair_cat`。
- `n_exact=10000`，`n_shape_mismatch=0`，`max_abs_diff=0`，`first_mismatch=null`。
- 耗时约 20.2 s；没有标签、没有训练。

### F.2 Collate 回放（256 个）

- 旧 `_concat_molecules` + `collate` vs 新 `concat_molecules` + `collate`：
  19 个共同字段全部精确一致（atom/bond/global、batch 索引、S1/S2 指针、
  counts 等）。
- `all_fields_exact=true`，无 mismatch。

### F.3 Checkpoint 回放（256 个）

| 项目 | 结果 |
|---|---|
| checkpoint | `zinc_local_relation_composition_v1/runs/lcrcA_s0/best.pt` |
| arm / seed / epoch | `A` / 0 / 476 |
| state dict keys | 新旧完全一致 |
| parameters old / new | 237,519 / 237,519 |
| state hash old / new | `d086a69e82bd73995e2ef820fcf456258dc9dff8ac4e086dd4fd904b827ef899` |
| `y`, `contrib`, `rho`, `h` | 全部精确一致 |
| `decomposition_max_abs_error` | 0.0 |
| training / optimizer steps | false / 0 |
| labels read | false |
| device | cpu |

Legacy schema 仍为 108/7/9/33、21 atom types、3 bond types、63 S1、189 S2；
因为 189 < 255，历史 `uint8` 路径码没有溢出。`lipo_v1` 的 400 类路径码则
独立使用 `int32`；没有把 ZINC 的 `uint8` 缓存复用到 Lipophilicity。

**结论边界**：F 只证明“冻结 ZINC 输入/模型接口可以被新代码精确复现”，
不证明 ZINC checkpoint 适用于 Lipophilicity，也不构成跨数据集迁移结果。

---

## G. 最终判断、限制与下一步边界

### G.1 最终状态

**`READY`**

理由：A–F 的数据/特征/结构/批处理/前向/ZINC 兼容性 gate 全部 PASS，没有
`FAIL`，没有 `BLOCKED`。`analysis/cpu_audit.json` 的 `final_status` 为
`READY`，`analysis/zinc_compatibility.json` 的 `status` 为 `PASS`。

### G.2 这个 `READY` 不表示什么

- 不表示 LCRC-A 在 Lipophilicity 上有效。
- 不表示已经训练、微调、调参或做过性能比较。
- 不表示 ZINC SSL checkpoint 对 Lipophilicity 有迁移收益。
- 不表示 DCC-L/GINE/指纹等基线已被比较。
- 不表示 scope 外（官方 valid/test）结构或标签已被审计。

### G.3 已知限制

1. 真实 3360 行全部是单组分、无孤立原子、无 zero-path、无 UNK；因此
   disconnected/isolated/zero-path/UNK 分支只有合成用例证据，没有真实数据
   覆盖。它们不能被称为“在本轮 Lipophilicity 数据上验证过”。
2. `row 1646` 属于官方 valid，被排除在 scope 外；没有为了“覆盖 disconnected”
   而把它拉进来，也没有转换它的结构。
3. 随机初始化模型的有限输出只说明构造/数值接口可用；不能解读为预测。
4. Fit 上 6 个 global 特征为常量，normalization 采用 scale=1.0 的显式策略；
   这是 interface 决策，需要在下一阶段设计时继续审计其影响。
5. spec hash 包含来源/环境 provenance（RDKit/Python/NumPy/Torch），环境变化
   会改变 hash；这是可复现性要求，不是数据变化。
6. 早期意外看到约 20 个 `train_y` 数值；未使用、未记录进工件。若未来阶段
   需要标签，必须重新建立显式的标签访问协议。

### G.4 没有做的事

- 没有训练/预训练/微调/超参搜索。
- 没有 optimizer step 或 backward。
- 没有 GPU。
- 没有读取 Reserved / 官方 valid/test 标签。
- 没有修改冻结 ZINC 模型/checkpoint/split/历史缓存/实验结论。
- 没有把 ZINC SSL checkpoint 直接用于 Lipophilicity。
- 没有补零/截断/任意映射制造假兼容。
- 没有降低测试条件或伪造工件。
- 没有自动开展下一阶段。

### G.5 复现命令

```bash
# Full Lipophilicity audit (all 3360 rows) + 10k ZINC replay
uv run python tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/audit_cpu.py \
  --dataset lipo --scope official-train-only --device cpu

# ZINC only, full 10k input replay
uv run python tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/audit_cpu.py \
  --dataset zinc-legacy --scope official-train-only --device cpu --zinc-max 10000

# Tests
uv run python -m pytest tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/tests -q
# -> 42 passed
```

### G.6 证据文件

| 文件 | SHA256 |
|---|---|
| `analysis/cpu_audit.json` | `cb902f1b97f62bfe88491f552d218d7d5bfd85dff7d440c70e11420d7e656369` |
| `analysis/source_manifest.json` | `c96972361d8c204c81b3accd4d70a7962ea5ed0af975f98528ea41c440df9093` |
| `analysis/feature_spec_lipo.json` | `a65c67f9595d9190e7c913d2594ba984a07b51300d63916de865dccca434206c` |
| `analysis/feature_spec_legacy_zinc.json` | `57d16ec3b20f0cf12ea806f6ca88d78fe94e9fce4e6caf7ee3db56696ad06570` |
| `analysis/norm_lipo.json` | `084c129c1731ea92a67d5e9bdce0d8d3bb335cf75ef45595ee690da288394d77` |
| `analysis/zinc_compatibility.json` | `cae1f38ec4224f38fa73bda8ac37fcf46995455148812661152b8a7d3c1cdf59` |

（文件 SHA 会随 `generated_utc` 等非科学 metadata 改变；上述为本次运行产物。）
