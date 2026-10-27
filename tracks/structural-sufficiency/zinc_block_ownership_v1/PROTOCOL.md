# PROTOCOL — `zinc_block_ownership_v1`（GPU 之前冻结）

**协议 ID：** `zinc-block-ownership-v1`
**设计：** 同目录 `DESIGN.md`（本文件不改写它）
**状态：** Phase 0 CPU gate 完成后冻结；`configs/zbo_v1.json` 记录冻结时刻与全部 hash。
**边界：** official test 全程不构造、不加载、不预测、不评分。official valid 只用于
每 epoch 评估与 strict-minimum 选点，不进入特征、归一化、SSL、结构统计或模型选择
以外的任何用途。

---

## 1. 问句

> 物理键的真实 biconnected 块归属，能否让一种共享的非递归结构计算，在未见局部
> 环境与未见块组合上，比相同内容的全图均场统计具有更好的泛化能力？

本轮把问题拆成 A/B/C，优先检验 A/B；C（共享块级非线性本身的效率）需要额外控制，
本轮不宣称已检验。

---

## 2. 数据角色（官方 split，不新建划分）

| 角色 | 行数 | 来源 | 用途 |
|---|---:|---|---|
| official train | 10,000 | `zinc_arch_discovery_v1/cache/ad_train.pt`（sha256 见 config） | 监督训练 + global-33 标准化；块缓存只读其结构字段 |
| official valid | 1,000 | `zinc_arch_discovery_v1/cache/ad_valid.pt` | 每 epoch 完整评估 + best checkpoint 选择 + 开发评估 |
| official test | — | **不加载** | — |

- `norm.json` 来自 `zinc_arch_discovery_v1/analysis/norm.json`：official train 10k
  计算的 global-33 mean/scale，`valid_or_test_used=false`；本轮重算核对。
- SSL：只从 `tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s{seed}.pt`
  加载 `atom_encoder` 与 `bond_encoder`。seed 0/1 各自独立；不加载 junction encoder，
  不加载任何监督 DCC/BOSI checkpoint。
- 块/关系缓存是 label-free、确定性的；构建函数不得读取 `labels` / `y` / `smiles`。

---

## 3. 模型与参数（实际 `named_parameters()`）

| 模块 | 形状 | 参数 |
|---|---|---:|
| atom encoder | 108→128→96 | 26,336 |
| bond encoder | 295→160→96 | 62,816 |
| half-edge encoder | 295→128→32（tanh） | 42,016 |
| side | 617→128→96 | 91,488 |
| bond composer | 391→160→96 | 78,176 |
| graph reader（旧部分） | 417→256→256→1 | 173,057 |
| **BOSI-B 合计** | | **473,889** |
| block LayerNorm(196, affine) | 196 | 392 |
| block MLP | 196→64→32 (SiLU) | 14,688 |
| reader_ext | 32→256（bias=False，零初始化） | 8,192 |
| **总参数** | | **497,161** |

- `reader_ext` 与旧 `reader.lins[0]` 相加，等价于 reader 输入 417→449、32 个新列
  权重为 0；这样初始化时与原始 BOSI-B 前向逐位相同。
- 参数上限 500,000；Phase 0 用 `named_parameters()` 重算，不允许用 DESIGN 估计值
  代替。
- 每块 `LayerNorm(196)` 是共享的；`phi` 是共享的；无 attention、无递归、无 junction
  token、无 ensemble、无蒸馏。

---

## 4. 块构造（label-free）

对每个分子 `G`，令 `E(G)` 为无向物理键集合。

1. 用独立 Tarjan 边栈算法求 biconnected components；每条桥边是单键块。
2. 用 `networkx.biconnected_component_edges` 作为独立参考；两套划分按“边集合的
   frozenset 多重集”逐分子比较，必须完全一致。
3. 每个 `e ∈ E(G)` 恰好属于一个块；所有块边集合的并集等于 `E(G)`，且两两不交。
4. 块顺序只用于缓存；模型对块顺序不变（`Σ_B phi`）。
5. 块标量：
   - `log1p(n_edges)`
   - `log1p(n_atoms)`
   - `cyclomatic_rank = n_edges - n_atoms + 1`
   - `log1p(n_articulations)`，其中 articulation points 取**原分子**的割点集合。
6. 孤立原子（没有键）不构造内容块；ZINC subset 实测没有孤立原子。

Phase 0 会将本实现与旧 `zinc_arch_discovery_v1` 缓存里的 `block_ptr` /
`block_bond_idx` / `block_bond_ptr` 做第三方交叉核对，但旧 `block_joint`、
`block_atom_type`、`block_bond_type`、`block_dist`、`block_overlap` 与旧
`BlockModel` / `GlobalAttention` 不进入 forward。

---

## 5. 三臂公式（唯一变量是内容归属）

记 `c_e ∈ R^96` 为 BOSI-B compose 后的物理键状态，`first_B = Σ_{e∈B} c_e`，
`second_B = Σ_{e∈B} c_e²`。

```
u_B = [ first_B / max(n_edges(B),1),
        second_B / max(n_edges(B),1),
        log1p(n_edges(B)),
        log1p(n_atoms(B)),
        cyclomatic_rank(B),
        log1p(n_articulations(B)) ]                       # R^196
h_block[G] = Σ_{B∈G} phi(LayerNorm(u_B))                  # R^32
yhat = reader_ext(z_old, h_block[G])                      # 449 维等价输入
```

### R（Real ownership）

```
content_first_B  = first_B
content_second_B = second_B
```

### N（Mean-field null）

```
S1[G] = Σ_{e∈G} c_e
S2[G] = Σ_{e∈G} c_e²
E[G]  = |E(G)|
content_first_B  = (n_edges(B)/E[G]) * S1[G]
content_second_B = (n_edges(B)/E[G]) * S2[G]
```

### S（Singleton-preserving null）

```
MULTI[G]  = {B∈G : n_edges(B) > 1}
E_multi[G] = Σ_{B∈MULTI[G]} n_edges(B)
S1_multi[G] = Σ_{B∈MULTI[G]} first_B
S2_multi[G] = Σ_{B∈MULTI[G]} second_B

B 单键: content_first_B = first_B,  content_second_B = second_B
B 多键: content_first_B  = (n_edges(B)/E_multi[G]) * S1_multi[G]
        content_second_B = (n_edges(B)/E_multi[G]) * S2_multi[G]
```

### 5.1 守恒（CPU gate 必须逐分子证明）

- R/N/S 的归一化前 `Σ_B content_first_B` 与 `Σ_B content_second_B` 逐分子相等，
  都等于 `S1[G]` / `S2[G]`。
- N 的 `Σ_B n_edges(B)/E[G] = 1`；S 的 `Σ_{B∈MULTI} n_edges(B)/E_multi[G] = 1`。
- S 的单键块内容与 R 精确相同（float64 参考下 0.0；float32 下只允许加法顺序误差）。
- 三臂的 `u_B` 形状、`phi`、reader 参数、初始化、batch 顺序、loss、优化器、
  updates 完全相同。

---

## 6. 配方（与 BOSI/DCC R0 逐项相同）

```
AdamW, lr 1e-3 -> 1e-5 per-step cosine, warmup 500 updates,
weight_decay 1e-5, dropout 0, grad clip 5, FP32, no AMP,
raw-y L1, batch 128, eval batch 256,
official train 10k -> 79 steps/epoch, 600 epochs, 47,400 updates,
batch order: torch.Generator().manual_seed(seed+101) -> randperm(10000) each epoch,
best checkpoint: strict minimum official-valid MAE, earlier epoch wins ties.
```

不缩短 updates；不改 lr/wd/warmup/cosine/clip/batch/selection。训练代码
`zbo_train.py` 在 CPU gate 中核对 step 0 的 batch rows、初始 state hash、generator
state hash 与同 seed 的 BOSI 历史流程一致。

---

## 7. Phase 0 CPU gate（先于 GPU）

`analysis/cpu_checks.json` 必须全部 PASS，才允许 GPU。门控条目：

| 门 | 内容 | 通过条件 |
|---|---|---|
| C1 | Tarjan vs NetworkX vs 旧缓存块划分；每条键一次；全键覆盖 | 0 mismatch，0 未覆盖，0 重复 |
| C2 | 原子重编号 / 边反向 / 边顺序 / batch 重组 | 块特征与前向 max\|Δ\| ≤1e-5（实测目标 0–1e-6） |
| C3 | R/N/S 一阶、二阶总量守恒；N/S 分配比例和 1；S 单键精确 | float64 max\|Δ\| ≤1e-9（合成）/ ≤1e-5（float32 真实 batch） |
| C4 | 合成反例：相同多重集/块大小，不同归属 | R 的两个输出差 >1e-6；N 的两个输出差 ==0；S 的输出差 ==0 |
| C5 | 真实 10k 统计与 R/S 覆盖率 | 报告全量；结构覆盖率 ≥20%；数值覆盖率满足 §7.1 |
| C6 | 与旧 BlockModel 区分 | 静态审计 + 参数/特征表；旧 attention/type-histogram 未被 import/调用 |
| C7 | 参数、初始化、前向等价 | 497,161 ≤500k；三臂完整 state 逐张量 0.0；vs 原始 BOSI-B 公共 state 0.0、前向 0.0 |
| C8 | 梯度/激活/数值 | step 2 block encoder 梯度 >1e-8；block 输出非零有限；无 NaN/Inf；loss 下降 |
| C9 | 标签隔离 | 块/关系构建不读 labels；valid labels 设 NaN 后前向/预测不变；代码无 test cache 访问 |
| C10 | 缓存可复现 | 重建两次内容 hash 相同；relation 缓存内容 hash 与历史 `c698a2f3…` 一致 |
| C11 | 合成小图精确性 | 链/三角/稠环/双环+桥的 n_edges/n_atoms/rank/articulation 手算一致 |
| C12 | checkpoint 往返 | 保存/加载后预测逐位相同 |

### 7.1 GPU 停止规则（Phase 0 之后）

若以下任一成立，停止并报告，不进入 GPU：

- train 10k 中含 ≥2 个多键块的分子比例 **<20%**；
- 在含 ≥2 个多键块的分子中，多键块的 R−S 相对内容差 **<1%** 的比例 >50%，
  或平均相对差 <1%；
- C1–C12 任一失败；
- 参数 >500,000 或三臂初始化不逐位相同。

Phase 0 实测值写入 `analysis/cpu_checks.json` 与 REPORT；本文件不根据实测值改阈值。

---

## 8. GPU 批次与停止规则

### 8.1 执行纪律

- 正式部署前：本地 commit；`rr doctor res`；`rr deploy res`。
- 一律 `rr run res <tag> --pool res-gpu0|res-gpu1 -- python .../zbo_train.py ...`；
  不手工 SSH 后台作业。
- 先 smoke（少量 updates）：核对初值、batch rows、梯度和形状；smoke 数据不用于决策。
- 正式 run 使用唯一 tag 和独立 `runs/{arm}_s{seed}` 输出目录；不覆盖历史。
- 远端分析 JSON 写到 gitignored 目录（`ZBO_ANALYSIS_DIR`），curated 副本再提交。

### 8.2 第一批（seed 0）

`R_seed0` / `N_seed0` 两卡并行。seed-0 判决：

```
d_best = N_best_valid - R_best_valid
d_W20  = N_W20        - R_W20
d_T100 = N_T100       - R_T100

PASS iff
  d_best >= 0.005
  and R_best_valid <= 0.082
  and d_W20 >= -0.002 and d_T100 >= -0.002
  and (d_W20 > 0 or d_T100 > 0)
  and ordinary (y > -3) paired ΔMAE >= 0
  and fixed-worst-10-removed paired ΔMAE > 0
  and mechanism_active
```

`mechanism_active`：

- step 2 的 block encoder 梯度范数 >1e-8；
- valid 上 `h_block` 标准差 >1e-6；
- best checkpoint 的 `reader_ext.weight` Frobenius 范数 >1e-6；
- R 权重 + N 内容的 valid 预测 mean|Δpred| >1e-6。

若失败：**停止**。不调宽度/层数/lr/wd/归一化/块类型，不买 seed 1，不跑 S。

### 8.3 第二批（条件）

`R_seed1` / `N_seed1` 两卡并行。两 seed 的 `d_best` 都 ≥0.005 且方向一致，才进入
第三批；否则停止并按 §9 归类。

### 8.4 第三批（条件）

`S_seed0` / `S_seed1` 两卡并行。按 §9 判决。

### 8.5 预算

软预算 **4 A100 GPU-hours**，硬上限 **6**。含 smoke、失败作业、正式训练。
单 run 预计 0.45–0.55 h；6 条正式轨迹 + smoke 预计 3.0–3.5 h。超过 6 h 立即停止，
不挪用预算继续超参搜索。

---

## 9. 判决（预注册）

先检查 `mechanism_active`；若失败：

- **`MECHANISM_INACTIVE`**：不把负结果当作科学否定；停止，不救援。

机制激活后：

1. **`OWNERSHIP_NOT_USEFUL`**：R−N 无正向实用收益（两 seed 均 <0.005 或方向相反），
   或 R 在两 seed 都不优于 N。
2. **`SINGLETON_NONLINEARITY_ONLY`**：R−N 两 seed 正向且 ≥0.005，但 R−S 不稳定/
   不正向；收益主要来自单键块上的逐键内容与非线性。
3. **`STRUCTURAL_GAIN_BUT_NOT_COMPETITIVE`**：R−S 两 seed 同向、平均 ≥0.003，
   但 R 的 seed-mean valid >0.077（不接近 DCC-L 0.075990）。
4. **`PROMISING_COMPOSITION_MECHANISM`**：R−S 两 seed 同向、平均 ≥0.003，
   且 R 的 seed-mean valid ≤0.077。
5. **`UNRESOLVED_DUE_TO_VARIANCE`**：符号/CI 不能区分，或 seed 间冲突；
   不追加 seed、不改设计。

附加判定条件：

- `d_best >=0.005` 必须同时满足 W20/T100 不反向（≥−0.002）且至少一个同向为正；
- 改善不能只来自固定 worst-10 或单行：drop-fixed-worst-10 后仍为正，且单行最大
  贡献 <50% 的 ΔMAE；
- 若 R 的改善集中在 `max_ring_size >= 7` 的受罚代理子集，而不在 ordinary/一般
  结构组合上出现，不宣称通用组合泛化；
- 两个 seed 共用同一对 SSL 编码器（s0/s1），不是独立 SSL 初始化族；`PROMISING`
  只表示“值得进一步确认”，不是确认。

---

## 10. 分析产物

- `analysis/cpu_checks.json`：Phase 0 全部门控。
- `analysis/prep.json`：块/关系缓存 hash、统计、数据来源。
- `analysis/frozen_manifest.json`：协议/配置/代码/缓存/SSL 的 sha256。
- `analysis/timing.json`：GPU smoke 与稳态 s/epoch。
- `analysis/freeze.json`：checkpoint/valid 预测/参数量/初值 hash。
- `analysis/summary.json`、`analysis/groups.json`、`analysis/decision.json`、
  `analysis/tables.md`、`analysis/per_molecule_valid.csv`。
- `analysis/probe.json`：冻结权重干预（arm switch、zero channel、shuffle、
  等尺寸块内容置换）。
- `analysis/run_manifest.tsv`、`analysis/gpu_ledger.json`。
- `REPORT.md`、`DECISION.md`。

---

## 11. 禁止清单

official test 访问；official valid 用于特征/归一化/SSL/结构统计；新的随机划分；
额外分子/外部标签/新标签公式分量；knowledge distillation；ensemble / seed 平均 /
checkpoint soup / TTA；递归隐藏状态更新；learned junction token；attention；
block-cut-tree attention；旧 BlockModel 的 `block_joint/block_dist/block_overlap`；
加宽 reader、增加块层数/宽度/秩来救援；改 lr/wd/warmup/horizon/selection；
用 valid 反复调参后仍称预注册；把 R−N 写成严格同信息对照；把 C 写成已检验；
把代码正确性/机制活跃性/局部显著性写成泛化成功。
