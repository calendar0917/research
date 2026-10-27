# DECISION — `zinc_block_ownership_v1`

**决策标签：`UNRESOLVED_DUE_TO_VARIANCE`**

**预注册 seed-0 主门：FAIL（唯一失败项 `d_best`，差 0.000116）。**

**执行动作：停止。不跑 seed 1，不跑 S，不调任何超参/宽度/层数/块类型。**

---

## 1. 判定依据（全部为预注册量）

```
d_best = N_best_valid − R_best_valid
       = 0.081591541 − 0.076707141
       = 0.004884399
threshold = 0.005
shortfall = 0.000115601
```

| seed-0 条件 | 实测 | 结果 |
|---|---:|---|
| `d_best >= 0.005` | **0.004884** | **FAIL** |
| `R_best_valid <= 0.082` | 0.076707 | PASS |
| `d_W20 >= −0.002` | +0.006204 | PASS |
| `d_T100 >= −0.002` | +0.006302 | PASS |
| `d_W20 > 0 or d_T100 > 0` | 两者均正 | PASS |
| ordinary (y>−3) ΔMAE ≥ 0 | +0.002089 | PASS |
| fixed-worst-10-removed ΔMAE > 0 | +0.003676 | PASS |
| 单行最大贡献 < 50% ΔMAE | 0.419 | PASS |
| `mechanism_active` | True | PASS |

- 配对 ΔMAE（best）：**+0.004884**，95% CI **[−0.000306, +0.011554]**，win rate 53.1%。
- final-valid：R 0.077741 / N 0.084134，**+0.006394**，95% CI
  **[−0.000847, +0.017782]**；W20/T100 也约 +0.0062/+0.0063。
- 机制：R block grad step 1 = 1.14e-4；`h_block` std = 1.439；reader_ext norm = 4.785；
  R 内容 vs N 内容的 valid mean|Δpred| = **0.1016**。不是 `MECHANISM_INACTIVE`。

**结论：方向性为正、稳定指标也正向，但主端点未过实用门槛，且配对 CI 跨零。**

---

## 2. 为什么是 `UNRESOLVED_DUE_TO_VARIANCE`，而不是其他标签

| 候选标签 | 是否适用 | 理由 |
|---|---|---|
| `MECHANISM_INACTIVE` | 否 | 第二步块梯度非零，块通道被读出；机制明确激活 |
| `OWNERSHIP_NOT_USEFUL` | **不作为主标签** | 预注册定义要求“两 seed 均 <0.005 或方向相反”；本轮只跑了 seed 0。且点估计为正、W20/T100/final 正向、drop-fixed 为正，不能诚实地说“无用” |
| `SINGLETON_NONLINEARITY_ONLY` | 否（不能判定） | S 未跑；没有 R−S 证据 |
| `STRUCTURAL_GAIN_BUT_NOT_COMPETITIVE` | 否 | 要求 R−S 两 seed 同向、平均 ≥0.003；S 未跑 |
| `PROMISING_COMPOSITION_MECHANISM` | 否 | 要求 R−S 两 seed、R seed-mean ≤0.077；S/seed1 均未跑，且主门未过 |
| `UNRESOLVED_DUE_TO_VARIANCE` | **是** | 符号/CI 不能区分：点估计 +0.004884 低于 0.005，CI 跨零；没有第二 seed 解决 |

**保守读法**：`d_best` 未过 0.005，因此不能把 R 的 0.076707 写成“块归属带来可用泛化”
或“接近 DCC-L 0.075990 的竞争力”；也不能反过来写成“ownership 无用”。这一轮买到的是
一个**接近门槛但未过**的机制信号，以及一组可用于下一轮预注册的分组事实。

---

## 3. 停止规则执行记录

`PROTOCOL.md` §8.2：

> 若失败：**停止**。不调宽度/层数/lr/wd/归一化/块类型，不买 seed 1，不跑 S。

执行情况：

- 未启动 `R_seed1` / `N_seed1`；
- 未启动 `S_seed0` / `S_seed1`；
- 未改任何超参、宽度、层数、归一化、块类型、readout；
- 未读 official test；
- 未做 test 调参、蒸馏、ensemble、seed 平均推理；
- GPU 总计 **1.079 A100·h**（R 0.493 + N 0.586），加 smoke 0.002 h，远低于 4 h 软预算。

---

## 4. 本轮可保留的事实（供下一轮预注册引用）

1. **结构覆盖率足够**：train 88.32% 分子含 ≥2 个多键块；valid 88.90%；
   121,808 / 12,155 个块与 NetworkX/旧缓存完全一致。
2. **机制可激活**：R/N 同 init、同 batch、同参数 497,161；R 的块梯度、reader_ext、
   h_block、R/N 内容替换敏感度均非零。
3. **方向性信号**：
   - best-valid R−N = +0.004884（未过 0.005）；
   - W20 +0.006204、T100 +0.006302、final +0.006394；
   - low-support ΔMAE +0.010286，CI [+0.000958, +0.024114]；
   - 块数 13–14 / ≥15 的 ΔMAE 分别 +0.007385 / +0.007593，CI 不跨零；
   - drop-fixed-worst-10 +0.003676，CI [+0.000125, +0.007408]；drop-top10
     +0.004997，CI [+0.001486, +0.008772]；单行最大贡献 0.419 < 0.5。
4. **不支持“未见组合泛化”**：
   - `any_unseen` ΔMAE −0.000252（CI 跨零）；
   - block-size combo novel n=47，点估计 +0.0636，CI [−0.0140, +0.1768]；
   - block-signature combo novel n=370，+0.005636，CI [−0.007059, +0.021336]。
5. **普通分子证据弱**：ordinary n=935，+0.002089，CI [−0.001847, +0.006091]；
   tail n=65，+0.045089，CI 极宽。
6. **S 未跑**：不能判断收益是否主要来自单键块逐键内容与非线性；不能把 R−N 的
   点估计解释为“多块组合泛化”。

---

## 5. 下一轮的唯一允许方向（新预注册，不是本轮救援）

本轮不自动开新实验。若重开，必须先写新预注册，且至少满足：

1. **主端点仍是 best-valid R−N 的实用门槛**，但把 0.005 的判定与配对 CI 联合起来；
   不因为本轮 0.004884 接近 0.005 就降低阈值。
2. **至少 2 个 seed**（各自独立 SSL，不共用 s1），先 R/N 两臂；只有两 seed 的
   R−N 都 ≥0.005 且同向，才允许跑 S。
3. **S 的作用是机制拆分**，不是“再买一个性能点”：若 R−S 不稳定，只能报
   `SINGLETON_NONLINEARITY_ONLY`，不能报组合泛化。
4. **预注册 any_unseen / low-support / 块数分层**，并把 ordinary 与 any_unseen 作为
   主要外推组；不能只引用 low-support 正 CI。
5. **不读 test**；不调宽度/层数/lr/wd/归一化/块类型；不追加 seed 3+ 来解决符号冲突。
6. 若下一轮仍只有单 seed 且 d_best 在 [0.004, 0.005) 区间，预先声明：
   标签仍是 `UNRESOLVED_DUE_TO_VARIANCE`，不是 `OWNERSHIP_NOT_USEFUL`，也不是成功。

**不建议**：直接加宽块编码器、换 attention/递归 MP、加 learned junction token、
把 N 改成另一种均场、或用 test 结果反推块设计。这些都不是本轮预注册的问题。

---

## 6. 最终一句话

**R 的机制是活的，方向性为正（best +0.004884，W20/T100/final ≈ +0.0062…+0.0064），
但 seed-0 的 best-valid 主门差 0.000116 未过；按预注册停止，不买 seed 1、不跑 S。
最终 `UNRESOLVED_DUE_TO_VARIANCE`：不能宣称 ownership 有用，也不能宣称 ownership 无用。**
