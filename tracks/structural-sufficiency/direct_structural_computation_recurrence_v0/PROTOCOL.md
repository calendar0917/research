# DSCR v0 — Direct Structural Computation and Recurrence

- Round: `tracks/structural-sufficiency/direct_structural_computation_recurrence_v0/`
- Protocol id: `dscr-v0`
- 数据：`ogbg-molpcba` 官方 scaffold split（Train 350343 / Valid 43793 / Test 43793，128 任务，masked BCE，官方 OGA macro AP）
- Test 角色：`DEVELOPMENT_TEST / EXPOSED_TEST_ENGINEERING_COMPARISON_SET`（每 epoch 全量评价，但**从不**进入 checkpoint / horizon / 配置选择；本轮结果不构成盲测确认）
- 历史资产：`lcrc_pcba_backbone_scaling_v1`（W2 994324）、`lcrc_pcba_dynamic_v1`（W2-DYNAMIC 1128468）、`lcrc_pcba_structural_gain_sprint_v1`（`MessageFFN`）、`lcrc_pcba_continuation_recipe_v1`（Test 标签读取与对齐证明）
- T0（本轮开始）：`2026-11-25T00:33+08:00`；停止新训练：T0+7h15min；硬截止：T0+8h

本文件在**任何正式训练开始之前**冻结。下文所有公式、参数、对照与判定规则不得在结果出现之后修改。

---

## 1. 科学问题

较强图预测性能是否真正依赖迭代更新后的邻居状态？能否用固定初始局部状态 `h0` 和真实结构组织关系，建立同样有效、更明确的直接结构计算？

DYNAMIC 只作为能力边界参照，不是教师模型、不做蒸馏、不作为预定最终架构。

## 2. 共享基础模型（只读复用）

`lcrc_pcba_backbone_scaling_v1/code/bs_model.py::BSModel`（历史 W2 构造器与宽度），

```python
a, b, scal = base._encode(batch)
h0 = base.drop(base.local(batch, a, b, scal))       # [N_atoms, 128]
logits = base.post_local(h2, batch)
```

W2 基线为 `post_local(h0, batch)`。atom encoder、bond encoder、S1/S2 `LocalContent`、dropout、per-atom `f`、graph `rho`、四头 attention readout、全部图描述符一律不改；不增加预测头、不改 readout、不删任何原始输入。全部参数从零训练、同一个 LR group。

## 3. 统一新增模块（五个臂完全相同）

```
bond_proj : Linear(13, 128)                     1792
blocks    : [MessageFFN(128,256), MessageFFN(128,256)]  (2 x 66176)
```

`MessageFFN` = `sg_model.MessageFFN`（= `ResidualFFN` + 纯键消息项），

```
FFN_l(z, m) = fc2_l(SiLU(fc1_l(z + m)))
```

`bond_proj` 由两个块共享；每个物理无向键展开为两个真实有向键；`edge_msg = bond_proj(bond_feat_directed)`。
所有臂 `h_next = h_current + FFN_l(z, m)`。总参数 **1,128,468**（= 994,324 + 132,352 + 1,792）。

初始化（历史 DYNAMIC 规则，逐字复用 `dy_model.initialize_dynamic_modules`）：新模块私有 generator `seed*1_000_003+17`，Linear `U(±1/sqrt(fan_in))`，LayerNorm 1/0，两个 `fc2.weight/bias` 置零，不推进全局 RNG。因此 epoch 0 eval-mode 下所有臂与同 seed W2 的 logits 完全一致。

## 4. 五个计算机制

记 `src=j, dst=i` 为 `j→i`；`e_ji = edge_msg[j,i]`；`LN_l` 为第 l 块 LayerNorm。

**DYNAMIC**（历史 `W2DynamicModel`，禁止重新设计）

```
z1 = LN1(h0);  m1[i] = Σ_j SiLU(z1[j] + e_ji);  h1 = h0 + FFN1(z1, m1)
z2 = LN2(h1);  m2[i] = Σ_j SiLU(z2[j] + e_ji);  h2 = h1 + FFN2(z2, m2)
```

**SELF**（点态深度 / 容量控制；不读任何其他节点隐藏状态）

```
z1 = LN1(h0);  m1[i] = Σ_j SiLU(z1[i] + e_ji);  h1 = h0 + FFN1(z1, m1)
z2 = LN2(h1);  m2[i] = Σ_j SiLU(z2[i] + e_ji);  h2 = h1 + FFN2(z2, m2)
```

`j` 只用于遍历真实入射键与其键属性；不读 `h[j]`，不造虚构边。

**FIXED1**（固定来源单跳）

```
z1, m1, h1 与 DYNAMIC 相同
z_target = LN2(h1);  z_source = LN2(h0)            # 同一个 LN2 模块/参数
m2[i] = Σ_j SiLU(z_source[j] + e_ji)
h2 = h1 + FFN2(z_target, m2)
```

**REAL2**（真实两跳分支的直接计算）

```
第一阶段同上；z_target = LN2(h1)；z = LN2(h0)
t[k,j] = SiLU(z[k] + e_kj);  c[j] = Σ_k t[k,j] / max(degree[j], 1)      # 真实有向键求和
m2[i] = Σ_j SiLU(z[j] + e_ji + γ c[j]),   γ = 0.5
h2 = h1 + FFN2(z_target, m2)
```

`c[j]` 只由 `h0` 得到；不读 `h1[j]`；不新增参数；不移除 backtracking。

**BAG2**（取消真实对应关系的多重集组合控制）

```
第一阶段、z_target、z、t、c、γ 与 REAL2 完全相同
m2[i] = ( Σ_{j∈N(i)} Σ_{l∈N(i)} SiLU(z[j] + e_ji + γ c[l]) ) / max(degree[i], 1)
h2 = h1 + FFN2(z_target, m2)
```

外层 `(z[j], e_ji)` 配对不变，`c[l]` 取同一根节点的真实分支集合，保留全部 occurrence 与重数，无随机打乱、无截断、无跨分子取 `c`。`degree=0 → m2=0`。

实现用 `bag2_pair_index` 精确枚举 `Σ_i deg_i²` 个 pair（分块求和），不物化完整路径。

## 5. 六个实验臂与主要对照

| Arm | 参数 | 角色 |
|---|---:|---|
| W2 | 994,324 | 原始非 MP 性能基准 |
| SELF | 1,128,468 | 点态计算与容量控制 |
| FIXED1 | 1,128,468 | 固定来源单跳结构计算 |
| REAL2 | 1,128,468 | 真实两跳分支组织 |
| BAG2 | 1,128,468 | 分支对应关系控制 |
| DYNAMIC | 1,128,468 | 迭代邻居状态边界 |

预注册主要比较：SELF−W2、FIXED1−SELF、REAL2−FIXED1、BAG2−FIXED1、REAL2−BAG2、DYNAMIC−FIXED1、DYNAMIC−REAL2。

## 6. 训练协议（历史 scratch recipe 逐字复用）

官方完整 Train；batch 128；每 epoch 2738 updates（历史最后一个 batch 规则）；AdamW `beta=(0.9,0.999)`、`eps=1e-8`、`wd=1e-3`；dropout 0.10；clip 5；FP32（沿用历史 TF32 默认配置）；单一 LR group；masked observed-entry BCE；首 epoch 线性 warmup `3e-5→3e-4`，之后按**历史 40 epoch** per-step cosine 到 `1e-5`；无辅助 loss；无早停；不改 batch size。

共同 horizon `H ∈ {40,30,25}`：所有臂永远只使用 40-epoch schedule 的**前缀**，不重新压缩 cosine。H 由 preflight 计时 + 双卡调度模拟（≥15% 余量、含报告预留）决定，不看任何精度。

seed 0 使用冻结的 W2 40-epoch plan（`global_plan_sha256 = 06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6`）；seed 1 用同一生成规则（`make_order_plan(seed=1)`）并在正式训练前冻结到 `configs/plan_seed1.json`。同 seed 的所有臂共享同一 plan 前缀。

每 epoch 完成后 eval-mode 全量评价：Train / Valid / Test 的官方 macro AP 与 masked BCE（不允许用 online batch AP 冒充）。评价与训练 RNG 隔离（代码中逐 epoch 断言 RNG 哈希不变）。Test 单独存储，不作为任何自动决策输入。

checkpoint：共同 horizon 内最高官方 Valid macro AP（平局取更早 epoch）为主 checkpoint；同时保存 last（可恢复 optimizer/scheduler/RNG）、horizon 终点、best 的完整预测与逐 epoch 曲线。

## 7. 单元队列与预算降级规则

优先完整计划（8 个新训练单元，seed-0 已有历史结果的不重训）：

1 FIXED1 s0、2 REAL2 s0、3 SELF s0、4 BAG2 s0、5 W2 s1、6 DYNAMIC s1、7 C\* s1、8 K\* s1。

选择规则（逐字按协议）：A 全 8 单元可行则取最大 H；否则 B 去掉第 8 项取 7 单元最大 H；否则 C 再去掉 W2 s1 取 6 单元；否则 D 保证 seed-0 四个新臂完成。计划冻结在 `analysis/schedule.json`。不因中间精度改变 H，也不因某臂成绩差而提前停止。

`C*` = seed-0 四个非迭代候选（SELF/FIXED1/REAL2/BAG2）中 Valid macro AP 最高者；差距 ≤0.001 AP 时按 `SELF > FIXED1 > REAL2 > BAG2`（工程简化顺序，不代表表达能力）取更简单者。`K*` = 固定映射的机制控制（SELF↔FIXED1，REAL2↔BAG2）。选择只看 Valid，不看 Test。

## 8. Phase E（仅在核心计划完成且确有完整运行预算时）

E1 补齐 K\*/W2 seed 1；E2 成对从可恢复 checkpoint 延长 C\*/DYNAMIC seed 1 到 40 epoch（必须两者都做，恢复 optimizer/scheduler/RNG，结果单独报告，不覆盖主结果）；E3 仅在 H=40 且 C\* 属于 FIXED1/REAL2/BAG2 时允许复用原始 S3 结构跑 S3 seed 1；E4 仅在两个 seed 与 DYNAMIC 方向冲突或差距 <0.005 AP 时成对运行 C\*/DYNAMIC seed 2。预算富余本身不是新增实验的理由。

## 9. 判定规则（阈值预先冻结，不允许为得到正标签而修改）

- Q1 `STRICT_LOCAL_COMPUTATION_SIGNAL`：SELF 在两个可配对 seed 上均 ≥ +0.005 Valid AP 且 Test 同向、高支持任务广泛改善。
- Q2 `FIXED_SOURCE_COMPUTATION_SIGNAL`：FIXED1 明显优于 SELF 且与 DYNAMIC 差距约 0.003–0.005 AP（结合两 seed、终点与高支持任务）。
- Q3：REAL2/BAG2 相对 FIXED1 的收益决定“两跳结构信息 + 组合运算”是否有价值。
- Q4 `REAL_BRANCH_BINDING_SIGNAL`：REAL2 在可用配对 seed 上一致优于 BAG2（且高支持任务/Test 无冲突）；若只有冻结干预（REAL2→BAG2）有变化而独立训练无优势，则 `BINDING_USED_WITHOUT_PREDICTIVE_GAIN`。
- Q5 `ITERATIVE_ADVANTAGE_REMAINS` / `COMPETITIVE_NONITERATIVE_COMPUTATION` / `DIRECT_COMPUTATION_ADVANCE`：C\* 与 DYNAMIC 在两 seed 上的差距（约 0.005 AP 尺度、高支持任务广泛性）。
- Q6：只有“有实际增益 + 跨 seed 可重复 + 非少数低支持任务 + 满足声明的信息依赖边界 + 研究价值明确”同时成立才建议继续该路线；否则 `NO_NEW_COMPETITIVE_DIRECT_MECHANISM`，并允许叠加 `SEED_CONFLICT` / `SPLIT_CONFLICT` / `BUDGET_LIMITED` / `MECHANISM_NOT_IDENTIFIED`。

任务级评价：官方全部可评分任务 macro AP + 预定义高支持集合（Valid 正/负例均 ≥20；Test 自身分层；Valid∩Test 高支持集合）。所有 AP 比较使用共同可评分任务集合，并报告任务数、每任务差值、改善/退化计数、中位数、高支持平均差值、低支持贡献、去除最大绝对差 1/3 个任务的敏感性。主 checkpoint 仍按官方全任务 Valid AP 选择。

不确定性：主要配对的任务重采样 2000 次；主要配对另做约 200 次分子（若可用则 scaffold-cluster）paired bootstrap。不以 bootstrap 冒充 seed 方差。

## 10. 有限机制解释（只对 C\*、DYNAMIC 与其关键控制）

G1 固定 hash 选出的 Valid 分子上记录 `R1 = RMS(h1−h0)/RMS(h0)`、`R2 = RMS(h2−h1)/RMS(h1)`、新模块梯度/输出尺度、实际 degree 与 pair 数、REAL2/BAG2 计算量与运行时间。
G2 冻结 REAL2 权重，把第二阶段真实对应关系替换为 BAG2 组合计算（其余完全不变），报告 AP 与 logit 变化；明确区分它与独立训练 REAL2 vs BAG2。
G3 最多 3 个真实 Valid 分子记录 `logits = Σ_v f(h2_v) + rho + attention_residual` 的精确账目与节点/边支持。

不做真实分子替换响应研究、不重建匹配分子目录、不训练解释器、不加解释 loss。

## 11. 交付物

`PROTOCOL.md`、`configs/`、`code/`、`analysis/{checks,preflight_timing,schedule,main_table,per_task_metrics,paired_comparisons,mechanism_diagnostics,summary}.json|csv`、`results/`（曲线、逐任务指标、manifest；大体积 checkpoint/预测按仓库规范留在本地与 res，并记录路径+SHA256）、`REPORT.md`（中文，首页回答十个问题 + 路线判定表）。
