# MODEL — LCRC-A / Lipophilicity 冻结模型的计算图、参数量与解读边界

本轮对象是 **冻结的 LCRC-A Lipophilicity 模型**（`lcrc_cross_task_preparation_v1` +
`lcrc_molnet_benchmark_pilot_v1`，R3 配方）。本文件只描述**实际 forward 执行的
计算**，区分「预处理算好的图统计」与「训练学到的参数」，并给出本轮可支持的
回溯范围。所有公式与 `code/dissect_model.py` 的实现逐项对应；FULL（A0）与冻结
上游 `lcrc_model.LCRC` 在 CPU 上逐位一致（见 `analysis/audit_cpu.json`）。

参照点：public122 / FULL / seed0，Valid RMSE 0.7108803968，Test RMSE 0.5727470871，
best epoch 240（旧 checkpoint，仅用于等价复播；本轮 54 个 run 全部重训）。

## 1. 数据与输入

| 项 | 值 |
|---|---|
| 任务 | Lipophilicity（logD，实验值） |
| 源 | ChemBench `Lipop`，commit `0b10d4c1ac72b6929503694a5f60446824736391` |
| CSV | `Lipophilicity.csv`，SHA256 `34280f0a…cd758`，n=4200，标签列 `exp` |
| 划分 | `Lipop.random{122,123,124}.ind.pkl`，3360/420/420，行索引哈希见 `analysis/protocol_freeze.json` |
| 结构 | 只依赖标签无关的 RDKit 图（`raw_attributes_v1.json.gz`，RDKit 2026.09.1） |
| 归一化 | `g`/`atom_scalar` 用 **Train 结构** 的 mean/scale；目标用 **Train 标签** 的 mean/std；Valid/Test 不参与 |
| 目标单位 | 原始 logD；训练在标准化空间做 MSE，报告一律换算回原始单位 |

Test 目标不在训练/选择链路中解码；只在 stage freeze 之后由
`code/dissect_score.py` 一次性评分（`EXPOSED_TEST_EXPLORATORY`）。

## 2. 计算图（逐行）

记原子 $v$、键 $e=(u,v)$、根原子 $r$、二跳路径 occurrence $p=(r\to w\to u)$。

```
a_v   = AtomEncoder(atom_feat_v)                        51  -> 128 -> 96
b_e   = BondEncoder([a_u+a_v, |a_u-a_v|, a_u*a_v, bf_e]) 301 -> 160 -> 96

# --- S1：一跳邻域，来自 bond encoder 的入射和 ---
b_sum_v  = Σ_{e∋v} b_e
b_mean_v = b_sum_v / deg(v)
nbr_v    = Σ_{(v,u)} Emb_nbr[ atom_type(u) * 5 + (bt_uv - 1) ]       (32-d)

# --- S2：二跳 occurrence，枚举在预处理完成，forward 只做查表+求和 ---
path_cat_p    = atom_type(w) * 25 + (bt1-1) * 5 + (bt2-1)           (400 类)
path_end63_p  = atom_type(u) * 5 + (bt2-1)                          (80 类)

J_r = Σ_{p: root=r} W_j ( U[path_cat_p] ⊙ V[path_end63_p] )         (32-d, 学习)
T_r = Σ_{p: root=r} Emb_type[path_cat_p]                            (32-d)
E_r = Σ_{p: root=r} Emb_end[path_end63_p]                           (32-d)
K_r = Σ_{p: root=r} Emb_close[closure_p]                            (8-d)
C_r = Σ_{p: root=r} gate[path_cat_p] * a_u                          (96-d, 学习 gate)

x_v = [a_v, b_sum_v, b_mean_v, nbr_v, T_v, E_v, J_v, K_v, C_v, scalars_v]   (529-d)
h_v = LayerNorm( LocalMLP(529 -> 192 -> 64)(x_v) )                  (64-d)
h_v = Dropout_0.10(h_v)                                             # 训练时，每次 forward 仅一次

L      = Σ_v f(h_v)                                                 (f: 64 -> 64 -> 1)
P_g    = rho([ g, mean_v h_v, mean_v h_v², log1p(n_atoms) ])        (171 -> 64 -> 1)
ŷ      = L + P_g
```

要点：

* **只有一次编码**。`a_v`、`b_e` 各过一个小 MLP；S1/S2 都是**固定结构汇总**
  （index_add / 求和 / 按根求和），没有沿边的递归传播，没有 attention，没有迭代。
  模块名里的 `path_*`/`nbr_*` 是查表参数，不是消息传递层。
* **dropout 只有一个 module，作用在两个读出的共同 h 上**：`L` 用 `f(h_drop)`，
  `rho` 用 `mean(h_drop)`/`mean(h_drop²)`。这保证训练时两条读出看到同一份噪声。
  `analysis/audit_cpu.json` 与 `tests/test_dissection.py` 都验证了「一次调用」与
  「两个读出共用」。
* **S2 的 200 维按根原子求和**，不跨根、不跨分子；`closure_p=1` 表示该二跳路径的
  末端与根相邻（即三角形），它是 S2 枚举的副产物，不是图卷积。
* `g` 是 **42 维全图统计**（原子/键数、环系统、环大小分布、距离、键型分数、
  形式电荷、氢数等，逐项见 `feature_spec.py::_LIPO_GLOBAL_FIELDS`），由预处理
  从图直接计算；`rho` 把 `g` 与池化的 `h` 一起送入一个小 MLP。
* `atom_scalar` 的 9 个分量是**逐原子标量**：`log1p_n_d2`、`log1p_n_paths2`、
  `closure_ratio`、`log1p_degree`、`in_ring`、`min_ring_size/8`、
  `ring_bond_count/4`、`is_terminal`、`is_fused_or_spiro`。其中 `n_paths2` 与
  `closure_ratio` 来自 S2 枚举，`in_ring`/`min_ring_size`/`ring_bond_count` 来自
  RDKit 环分析。**因此「删除 typed-S2」不等于「模型看不到任何二跳/环信息」**。

## 3. 字段来源与学习边界

| 字段 | 维度 | 来源 | 是否学习 |
|---|---|---|---|
| `atom_feat` | 51 | RDKit 原子类/电荷/氢 + 物理键图（环、度、半径 1/2 类型直方图等） | 否（预处理） |
| `bond_feat` | 13 | 原始 edge_class / stereo + 环/端点标志 | 否 |
| `atom_scalar` | 9 | 图分析 + S2 枚举 | 否（Train 标准化） |
| `global` | 42 | 全图统计 | 否（Train 标准化） |
| `nbr_code` | int | `atom_type(u)*5+(bt-1)` | 否 |
| `path_cat` / `path_end63` | int | 二跳枚举编码 | 否 |
| `path_closure` | int | 二跳末端是否与根相邻 | 否 |
| AtomEncoder / BondEncoder | 96 / 96 | — | **是** |
| `Emb_nbr`, `Emb_type`, `Emb_end`, `Emb_close` | 32/32/32/8 | — | **是** |
| `path_u`, `path_v`, `joint_proj` | 32/32/32 | — | **是** |
| `path_gate` | 1 | 初始化为 1 | **是** |
| LocalMLP + LayerNorm | 64 | — | **是** |
| `f` | 1 | — | **是** |
| `rho` | 1 | — | **是** |

## 4. 参数量

FULL = **247,106** trainable（= nominal；无 buffer 计入）。

| 模块 | 参数 |
|---|---|
| `atom_encoder` (51→128→96) | 19,040 |
| `bond_encoder` (301→160→96) | 63,776 |
| `local.mlp` (529→192→64) | 114,112 |
| `local.norm` | 128 |
| `nbr_emb` (80×32) | 2,560 |
| `path_emb` (400×32) | 12,800 |
| `end_emb` (80×32) | 2,560 |
| `path_u` (400×32) | 12,800 |
| `path_v` (80×32) | 2,560 |
| `joint_proj` (32→32) | 1,056 |
| `close_emb` (2×8) | 16 |
| `path_gate` (400×1) | 400 |
| `f` (64→64→1) | 4,225 |
| `rho` (171→64→1) | 11,073 |
| **合计** | **247,106** |

18 个变体的期望参数量与实际逐项核对一致（`analysis/audit_cpu.json`）：

| 变体 | 名称 | 参数 | Δ vs FULL |
|---|---|---|---|
| A0 | FULL | 247,106 | 0 |
| A1 | G_ONLY | 2,817 | −244,289 |
| A2 | NO_G | 244,418 | −2,688 |
| A3 | NO_ADD | 242,881 | −4,225 |
| A4 | POOL_ONLY | 240,193 | −6,913 |
| A5 | ADD_ONLY | 236,034 | −11,072 |
| B0 | TYPED_S2_OFF | 176,514 | −70,592 |
| B1 | NO_JOINT | 224,546 | −22,560 |
| B2 | NO_CONTENT | 228,274 | −18,832 |
| B3 | NO_JOINT_CONTENT | 205,714 | −41,392 |
| B4 | MARGINAL_ALL | 247,106 | 0 |
| B5 | MARGINAL_BT2 | 247,106 | 0 |
| B6 | NO_CLOSURE_CHANNELS | 245,362 | −1,744 |
| C0 | NO_S1JOINT | 238,402 | −8,704 |
| C1 | NO_BOND_MEAN | 228,674 | −18,432 |
| C2 | NO_SCALARS | 245,378 | −1,728 |
| C3 | NO_M2 | 243,010 | −4,096 |
| C4 | MEAN_ADD | 247,106 | 0 |

所有变体共享同一份 FULL 初始 state（按固定列映射复制），新参数只有 A5 的
`b0`（初始 0）；被删模块的参数不存在，删除后没有不可达参数。

## 5. 训练配方（R3，冻结）

* AdamW，peak lr 3e-4 → 1e-5（**per-step** cosine），betas (0.9, 0.999)，eps 1e-8，
  weight decay 1e-3；batch 128；fp32。
* 每个 epoch 27 步，360 epochs = **9,720 updates**；无 early stopping、无 EMA/SWA、
  无梯度累积、无增广。
* 损失：Train 标准化目标上的 MSE；梯度裁剪 5.0（global norm）。
* 模型选择：**best Validation RMSE，严格更小，平局取更早 epoch**；只保存 best
  checkpoint；Test 从不参与选择。
* 随机性：每个 seed 先用 CPU `torch.manual_seed(seed)` 构造 FULL 初始 state，再
  列删除映射到变体；shuffle 顺序由 `(seed, epoch)` 决定（固定 offset 917,000），
  同一 seed 的所有变体看到完全相同的 batch 序列（`batch_order_probe` 记录在
  `train.json` 中并可复核）。
* 训练/推理时间、GPU 小时、梯度裁剪次数、非有限值次数逐 run 记录于 `train.json`。

## 6. 变体切换点（只做删除/替换，不新增模块）

| 变体 | 改动 | 可声称 | 不可声称 |
|---|---|---|---|
| A2 NO_G | rho 不再读 `g` | 该输入对当前函数的条件性贡献 | `g` 信息对任务无用 |
| A3 NO_ADD | 去掉 `f`/`L`，只用 rho | 原子求和读出的必要性 | logD 不服从可加性 |
| A4 POOL_ONLY | 去 `g` 去 `f` | 池化读出 vs 原子求和读出的条件性差异 | 唯一因果分解 |
| A1 G_ONLY | 只有 `g`→64→1 | 全图统计单独的可预测性 | 局部表示无用 |
| A5 ADD_ONLY | 只有 `L`+常数 `b0` | 纯原子求和读出的上限 | 无 |
| B0 TYPED_S2_OFF | 删 200 维 S2 块 | 该块的条件性收益 | 模型不含任何二跳/环信息（scalars、g 仍在） |
| B1 NO_JOINT | 只删 `S2_joint` | 联合项的条件性收益 | 只删了类型/末端内容 |
| B2 NO_CONTENT | 只删 `S2_content`+gate | gate×末端内容项的条件性收益 | 只删了联合项 |
| B3 | 同时删 B1+B2 | 两项合并的条件性收益 | 单删可替代性可外推到联删 |
| B4 MARGINAL_ALL | 联合/内容聚合改为按根边际乘积 | 联合统计 vs 边际统计的差别 | 生成合法分子/化学反事实 |
| B5 MARGINAL_BT2 | 同上，再按 (root, bt2) 分组 | 保留第二条键型相容关系后剩余的配对价值 | 化学因果 |
| B6 NO_CLOSURE_CHANNELS | 删 `close_emb` 与 scalar 的 `closure_ratio` | 显式 closure 通道的条件性收益 | 去掉图中所有环/三角形信息 |
| C0 NO_S1JOINT | 删 `nbr_emb` | S1 类型联合项的条件性收益 | 去掉一跳信息（`b_sum`/`b_mean` 仍在） |
| C1 NO_BOND_MEAN | 删 `b_mean` | 均值化入射和的参数化收益 | 去掉度信息 |
| C2 NO_SCALARS | 删 9 维 `atom_scalar` | 这组逐原子标量的条件性收益 | 去掉环/度/二跳信息 |
| C3 NO_M2 | rho 不读 `mean_h²` | 二阶池化的条件性收益 | 去掉所有二阶信息（局部 h 仍在） |
| C4 MEAN_ADD | `L=mean_v f(h_v)` | 原子求和 vs 平均的读出偏置 | logD 的可加化学定律 |

## 7. 回溯接口（本轮实际支持的范围）

对 122 Valid 的 8 个固定例子（按 FULL seed0 的 |error| 排名 0/1/209/210/416/417/418/419）：

* 原始单位分解：`pred = target_mean + Σ_v c_v_raw + r_raw`，其中
  `c_v_raw = target_std · f(h_v)`，`r_raw = target_std · rho`。逐例核对恒等式残差。
* 逐原子项 `c_v_raw` 与元素、原子索引、`f(h)` 一起保存（`analysis/attribution/`）。
* **leave-one-S2-occurrence-out**：只从该根的 typed-S2 输入中删除一条 occurrence
  记录，图/S1/raw scalars/`g` 不变，模型 eval。记录 root/middle/end 原子与两条键型、
  根 `h`、根 `f`、其他原子 `f`、`rho`、最终预测差；并核对
  `Δpred = ΔΣc + Δrho`（原始单位）。
* 这是**模型输入通道干预**，不是分子编辑，也不是化学因果；路径之间的非线性交互
  使得 leave-one-out 差不能相加当作「唯一路径贡献分解」。
* 无局部求和头的变体（A1/A3/A4/A5）不产生原子贡献图；`ADD_ONLY` 的 `b0` 记为
  全图常数项。

## 8. 明确不支持的解读

* 真实标签只有分子级 logD；本轮**没有**把真实误差唯一分配到某个原子/路径，也
  **没有**发现化学因果规律。
* 固定权重干预只说明「当前训练好的函数依赖该输入」，不说明训练前的信息必要性；
  删减后重训保持/变差才是条件性收益陈述。
* B4/B5 是代数联合统计对照，不是生成新分子的反事实；其差距不能单独排除价态、
  度、环等简单相容性解释。
* 全图 `g` 与池化 `h` 的信息高度重叠（`rho` 同时读两者）；A 组的差值不能被解释为
  「信息的唯一物理贡献」。
