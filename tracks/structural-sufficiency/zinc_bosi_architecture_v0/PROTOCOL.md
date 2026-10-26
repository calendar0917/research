# PROTOCOL — `zinc_bosi_architecture_v0`（GPU 之前冻结）

**协议 ID：** `zinc-bosi-architecture-v0`
**设计基准：** `calendar0917/research@5950260` 的 `DESIGN.md`（本文件不改写它）
**架构审计：** `AUDIT.md`（同目录，CPU only）
**CPU gate：** `analysis/cpu_checks.json` — **12/12 PASS → `CPU_GO`**
**冻结时刻：** 见 `configs/bosi_v0.json` 的 `frozen_utc`；本文件在任何正式 GPU 训练之前提交。

**边界：** official test 全程不构造、不加载、不预测。official valid 也**不加载**
（`zinc_dccl_bottleneck_v1` 的官方 valid 数字只作为历史引用，不参与本轮的选点或比较）。
所有评估集都是 official-train 10k 内的冻结行划分。

---

## 1. 问句

> 当原子只负责锚定局部环境、物理键承载预测、显式结构统计表达"目标键与邻键的关系"时，
> 能否在**不使用 learned junction token** 的条件下得到一个更清晰、泛化更好的架构？

两个等参数模型，唯一差别是关系统计的分配方式：

* **BOSI-R** — 真实按 `(v,e)` 的伙伴关系分组，计算每组 `T1,T2,T3` 矩。
* **BOSI-B** — 保留同一组关系计数 `n[v,e,r]`、保留相同的伙伴总量，把**中心（leave-one-out）
  统计均值**按计数广播到各组，替代真实分配。

不作 Stage 2 attention；`ATTENTION_NEXT.md` 只做设计，不训练。

## 2. 数据角色（预注册，与 `zinc_additive_reader_v1` 完全同一划分）

| 角色 | 行数 | 用途 | 行号 SHA-256 |
|---|---:|---|---|
| Fit8k | 8,000 | 训练 + Fit-only 标准化 | `1cf6a2ec73ff5dfc182d72a7d4cd0148d31f3ae2488ee9943132f1aa51d7455a` |
| Monitor | 1,000 | 每 epoch 评估 + best checkpoint 选择 | `d493dc7fc547e062ee16301bb42f350d5eb6d5d773215254c0d84098a410933b` |
| Locked | 1,000 | 冻结后**一次性**读取（历史已暴露，不是新盲测） | `9cbefd5e84cf7f4e8f7314280f35f636a0b18753a2f93a724b935e65a208cf6d` |
| official valid | — | **不加载** | — |
| official test | — | **不构造、不加载** | — |

来源：`split_a.json`（`bjoc_predictive_bottleneck_v1`），数据缓存为
`zinc_arch_discovery_v1/cache/ad_train.pt`（sha256 `d85d9875…`），只读其 `split="train"` 内容。

## 3. 模型（`code/bosi_models.py`，实际 `named_parameters()` 计数）

| 模块 | 形状 | 参数 |
|---|---|---:|
| atom encoder | 108→128→96 | 26,336 |
| bond encoder | 295→160→96 | 62,816 |
| half-edge encoder | 295→128→32（tanh） | 42,016 |
| side（目标键条件状态） | 617→128→96 | 91,488 |
| bond composer | 391→160→96 | 78,176 |
| graph reader | 417→256→256→1 | 173,057 |
| **合计** | | **473,889** |

* 上限 500,000；DESIGN 预算 473,889 被**实际重算命中**（`G7`）。
* `side` 输入 617 = `a_v(96) + b_e(96) + x(32) + M(4×3×32=384) + n(4) + closure(4) + log1p(deg)(1)`。
  `closure = [Σ_f log1p(c_f), log1p(max c_f), log1p(#{c≥1}), log1p(#{c≥2})]`，
  `c_f` = 伙伴 f 的两侧远端在去掉中心后的公共邻居数（原始闭合计数，priority 不吞掉它们）。
* 只从 `pretrain_encoder_s{seed}.pt`（train-only masked-SSL，sha256 `006fabc1…` / s1）
  加载 `atom_encoder` 与 `bond_encoder`；**不实例化 junction encoder**，不加载任何监督 DCC-L 权重。
* 无 residual bypass、无递归、无 attention、无查表、无 ensemble、无蒸馏。
* 每个 seed **只有一份共享初始状态**；`arm` 是纯非参数开关，R/B 构造走同一 RNG 流，
  `G7` 实测 `initial_state_sha256` 相同、逐张量最大差 0.0。

### 3.1 关系定义（label-free，与 DCC-L 冻结对象逐项对齐）

对 `(v,e)`，`e={v,u}`，伙伴 `f={v,w}`，优先级 T→Q→R→N：

1. `T`：`u,w` 直接相连（三角闭合）。
2. `Q`：否则 `u,w` 在去掉 `v` 后仍有公共邻居。
3. `R`：否则 `e,f` 同属一个含 `v` 的 biconnected block（`|E(block)| ≥ 3`）。
4. `N`：其余。

`networkx.biconnected_component_edges` 是冻结对象构建器 `structural_relation_objects_v1`
使用的同一个参考实现，因此 block 定义不会漂移。`G3` 逐分子核对：
T 伙伴行 = `2 × #junctions(endpoints bonded)`（观察 168 = 期望 168），
Q 伙伴行 = `2 × #junctions(common>0 且非三角)`（观察 40 = 期望 40），
三角/环键/block 划分与 networkx 及 `atom_feat`/`bond_feat` 的零不一致。

## 4. 配方（与 `zinc_dccl_bottleneck_v1` Fit8k **逐项相同**）

AdamW，lr 1e-3 → 1e-5（per-step cosine），warmup 500 updates，wd 1e-5，dropout 0，
grad clip 5，raw-y L1，batch 128，eval batch 256，**47,400 optimizer updates**
（`ceil(8000/128)=63` steps/epoch → 753 epochs），FP32，无 AMP。
batch 顺序：`torch.Generator().manual_seed(seed+101)` → 每 epoch `randperm(8000)`。
best checkpoint = **严格最小 Monitor MAE**（ties 保留更早 epoch）。
标准化：只用 Fit8k 行计算 global33 均值/方差（`valid_or_test_used=false`）。

## 5. 预注册修订 A1（在任何 GPU 训练之前）

**DESIGN.md §3 的字面 B 公式是** `M_B[r] = n[r]/deg(v) · Sv`，`Sv = Σ_{f∈Inc(v)} T_k(x_f)`。
它与 R 有**两处**差别：(i) 关系分配；(ii) `Sv` 含目标键自身、并且被 `(deg−1)/deg` 收缩。
因此它**不是**"唯一的架构区别是分配方式"，且 R−B 的差会混入 leave-one-out 与自含项。

**本轮采用纯分配版本（单变量对照）：**

```
Sloo(v,e) = Σ_{f ∈ P(v,e)} T_k(x_f)                       # 与 R 完全相同的伙伴总量
M_B[v,e,r,k] = n[v,e,r] / max(deg(v)-1, 1) · Sloo(v,e,k)
```

于是 `Σ_r M_R[r] = Σ_r M_B[r] = Sloo`，两臂**唯一差别**是 Sloo 如何分到 4 个组。
`G9` 给出两个字面版本的数值差以便审计：

| 版本 | R−B 相对差（有伙伴的半键） | 受影响半键比例 |
|---|---:|---:|
| A1 纯分配（**本轮训练**）| mean 2.82% / median 0 | 21.2% |
| DESIGN 字面 `Bfull`（**不训练**）| mean 10.73% / median 10.27% | 99.4% |

A1 的代价与收益必须一起读：**分配只在 ≥2 个关系组非空时才有定义**，实测覆盖率
**19.50%**（`G9`）；在分配有定义的半键里，`M` 的相对差 mean **13.3%**，且 **100%** 的半键
发生改变。换句话说，R/B 是一个**只作用于 1/5 查询**的精确干预，效应在整图 MAE 上会被
稀释约 5 倍。这是设计本身的覆盖率事实，不能用加宽/加阶绕过。

## 6. CPU gate（强制，已完成）

`analysis/cpu_checks.json`，12/12 PASS：

| gate | 内容 | 关键数字 |
|---|---|---|
| G1 | Fit8k 度上界，不截断 | max deg 4，max 伙伴 3（hist 1/2/3/4 = 32,959/188,028/168,477/10,112）|
| G2 | 四类互斥完备；`Σ_r n = deg−1` | 0 违反，0 重复伙伴 |
| G3 | 环/三角/额外公共邻居/block 与原始图一致 | 零不一致；T/Q 行数与冻结 junction 精确相等 |
| G4 | 三阶矩反例 + 跨通道配对反例 | `ΣT3` 0 vs 0.5773502692；投影二阶矩 4 vs 2 |
| G5 | 独立 NumPy/Python 实现 | max\|ΔM\| = 1.1e-7（float64 参考）|
| G6 | 原子重编号/键向反转/边序/batch 重排 | max\|Δy\| = 2.4e-7（tol 1e-5）|
| G7 | R/B 参数与初始化逐位相同 | 473,889 / 相同 sha256 / 逐张量 0.0 |
| G8 | R/B 各自进入真实 forward | 直方图保持置换：R 动 4.2e-4，B **精确 0.0** |
| G9 | Fit8k 使用率与 R/B 差分布 | 见 §5，覆盖率 19.50% |
| G10 | 六个模块梯度均有效 | 全非零；M 置零 ⇒ \|Δy\| 5.1e-3 |
| G11 | 无分支归零/尺度爆炸 | M std 0.399、max 3.0；z max 26.2；无 NaN |
| G12 | official valid/test 未触碰 | 无任何 valid/test loader；只读 `ad_train.pt` |

**判据通过：`CPU_GO`。** 未发现 R≡B、关系构造错误、参数超标、训练路径不变或关键反例失败。

## 7. GPU Stage 1（条件执行，最多四条正式轨迹）

先跑 **R_s0、B_s0**（`res` 两卡并行，`rr`）。

**买第二个 seed 的条件（预注册，看完 seed 0 不得改设计）：**

```
if (R_s0.Monitor > DCCL_s0.Monitor + 0.012) and (R_s0.Monitor - B_s0.Monitor > -0.003):
        STOP  # R 没有实际价值，也不比 B 好
else:
        跑 R_s1、B_s1
```

DCC-L 历史（`zinc_dccl_bottleneck_v1/runs/fit8k_s{0,1}`，**不重训、只复用**）：

| seed | Monitor MAE | Locked MAE (best-monitor) | Locked MAE (final) |
|---:|---:|---:|---:|
| 0 | 0.12315087235253304 | 0.08357349264295771 | 0.08254802366392687 |
| 1 | 0.12777156261634082 | 0.08342303324723616 | 0.08369429864967241 |

**预算：2.5 A100·h 硬上限**，含 smoke、失败作业与正式训练。先做实测计时（1 个完整 epoch
+ 完整 Monitor 评估），若四条轨迹的投影超过 2.5 h，则**停止并记录**，不得缩短 47,400 updates。

## 8. 冻结与 Locked 一次性读取

1. 四条（或两条）轨迹完成后，`bosi_freeze.py` 重算 Monitor 预测、写
   `analysis/freeze.json`：config sha、checkpoint sha、Monitor 预测 sha、参数、run id、
   GPU-hour、verdict 输入。**冻结之后不得再改模型/配方/epoch。**
2. `bosi_locked.py` 校验 freeze 的每个 hash，然后**一次性**读取 Locked（1000 行），
   写 `analysis/locked_read.json` + `analysis/locked_preds/*.npz`。已存在则拒绝重写。
3. `bosi_analyze.py` 只用冻结文件产出所有表与判决。

**Locked 在历史研究中已暴露（DCC-L 4 seeds、ESOC、additive-reader 都读过），
本轮的读取不是新的盲测确认，不得这样表述。**

## 9. 评价与判决（预注册）

指标（Locked, n=1000）：overall MAE、ordinary（`y>−3`）、tail（`y≤−3`）、
去掉最差 10 个分子（按配对 `max(AE_A, AE_B)`）、low_support、any_unseen、all_supported。
support 用 Fit8k-only 精确 typed radius-2 签名（与 `zinc_dccl_bottleneck_v1` 同构造）。
配对 bootstrap（10,000 次，固定行重采样）只描述分子抽样，不作为跨 seed 置信度。

判决（严格按任务定义）：

* **`PERFORMANCE_PROMISING`** — R 在两个 seed 的 Locked 上都优于历史 DCC-L；R 的 Locked
  seed-mean 改善 ≥ 0.005；ordinary 与 any-unseen 同向为正；去掉 top-10 后仍为正。
* **`STRUCTURE_MECHANISM_ONLY`** — R 稳定优于 B（两 seed 同向，seed-mean ≥ 0.003），
  但没有超过 DCC-L。不自动做 attention 补救。
* **`NO_GO`** — R 不优于 B，或相对 DCC-L 明显退化，或效果只由尾部/单 seed 驱动。
* **`INCONCLUSIVE`** — 两 seed 或区间不能区分关键比较。不扩样本、不改划分、不追加 seed。
* 只有 `PERFORMANCE_PROMISING` 才讨论 official 10k；0.001–0.002 的弱信号不触发 0.05x 讨论。

附加诊断（预注册，全部在冻结权重上推理，不训练）：
R 权重改用 B 统计（`arm` 切换）；M 块置零；直方图保持的 relation 置换；
按"分配有定义的半键占比"四分位分层；T/Q 组使用率与含 T/Q 分子的子组 MAE。

## 10. 禁止清单

重训 DCC-L；official test/valid 访问；新的随机划分；用 Monitor/Locked 标签选架构/超参/epoch
（Monitor 只按上述规则选 checkpoint）；第三个 seed；learned junction encoder；递归 MP；
Transformer/attention 层；新的统计查表；外部预训练；新的标签公式分量；教师蒸馏；ensemble；
加宽网络、加矩阶数、增删关系类型来绕过失败。
