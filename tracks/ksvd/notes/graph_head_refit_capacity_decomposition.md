# Graph Head Refit & Capacity Decomposition Audit — why does a small post-hoc MLP beat the original joint head?

> 问题：在同一 frozen compact-v4 graph representation `R ∈ R^302` 上，上一轮观察到
> 一个**小的 post-hoc MLP head**（`302→13→13→1`, H2）pooled outer-heldout MAE
> `0.170280` 稳定优于**原 jointly-trained graph head**（`302→64→32→1`, H0）
> `0.175031`，差 `+0.00475`。本阶段把 H0 与 H2 之间的 confound 拆开，回答
> **这个 gain 到底来自 capacity、frozen refit、late adaptation、initialisation，
> 还是 standardisation/optimisation protocol**。
>
> 模块：`experiments/luyin16/zinc_graph_head_refit_capacity_decomposition.py`
> 结果：`results/graph_head_refit_capacity_decomposition/`
> 测试：`tests/test_graph_head_refit_capacity_decomposition.py`（13 tests，全过）
> 结论：**LATE-READOUT-ADAPTATION GO（Case C，并带 refit 分量）**。原 architecture
> 本身足够；joint training 结束时 head 对最终 frozen `R` **under-adapted**。
> small capacity **不是**机制（Δcap = +0.00025，95% CI 跨 0，标准输入下 2/5 fold，
> raw 输入下符号反转）。**不授权** end-to-end small-head 替换。

---

## 1. Motivation

连续多条 representation/statistic/function-family 路线 NO-GO 之后，唯一还活着的
正向现象就是这个 small-MLP refit signal：

| reader | pooled outer-heldout MAE |
|---|---:|
| original jointly-trained H0 | 0.175031 |
| H2 `302→13→13→1` | **0.170280** |
| rank-4 FM | 0.174048 |

`0.175031 − 0.170280 = 0.004751`，且在两个 frozen backbone seed 上方向一致
（seed0 +0.005757，seed1 +0.003745）。本阶段的目标不是找最好的 head，而是解释
这个已观察到的 gap 来自哪里。

## 2. Why the FM verdict is separate

上一轮 `Graph-Head Function Family Audit` 的 verdict 是
**INCONCLUSIVE（Case E）**：rank-4 FM vs 同预算 generic MLP 的 `Δsmall` pooled
`+0.00190 < +0.0025`，fold-averaged 3/5，CI 跨 0。FM 这条线因此**关闭**：不扫 rank、
不做 DeepFM/CrossNet/bilinear、不做 E2E。本轮不重开 FM，也不引入新 head family；
只解释更强、更稳的 small-MLP signal。

## 3. The unexplained small-MLP signal

H0 与 H2 同时混着五个变量：capacity（64→32 vs 13→13）、frozen refit、late
adaptation、initialisation、以及 post-hoc 的 optimisation protocol（fit-only
standardisation、mini-batch 512、800 epochs、wd=0、dedicated head selection）。
§4 的四个 hypothesis 必须被拆开，而不是笼统说「小 head 更好」。

## 4. Competing hypotheses

- **A Capacity / regularisation**：原 64→32 head 对 7200 样本过复杂；13→13 有更好的
  inductive bias。
- **B Frozen-representation refit**：真正有价值的是 backbone 训练结束后重新拟合
  最终 MAE readout。
- **C Head under-adaptation**：原 head 不差，只是 joint training 结束时没充分适应
  最终 `R`；固定 `R` 后继续优化原 head 即可改善。
- **D Optimisation/standardisation protocol**：gain 来自 fit-only standardisation /
  mini-batch / longer head training / dedicated checkpoint。
- **E Unstable**：上一轮信号本身被高估或不可复现。

## 5. Frozen OOF protocol

完全复用上一轮已导出的 frozen-R cache（read-only）：

```
results/graph_head_function_family/state_exports/
    graph_head_R_cache_v1_fold{0..4}_seed{0,1}.npz
```

每个 fold 的真实 nested structure：

```
7200 head-fit  /  800 head-selection  /  2000 untouched outer-heldout
```

- 2 个 frozen backbone seed × 5 folds，一次跑完（§5：直接使用两个 seed，不做
  逐步解锁，也不新增 backbone training）。
- 只使用 official train universe；official valid / official test **从未加载**。
- head init 只用一个 `head_seed=0`（§22）。第二 init 仅在 borderline 时触发，本轮
  **未触发**（|Δcap| = 0.00025 < 0.001）。

## 6. Exact standardization reparameterization

所有 refit head 输入 fit-only standardized `z = (R−μ)/max(σ, ε)`（只由 7200 fit
计算，degenerate 坐标 `z=0`）。原 H0 吃 raw `R`，所以 L-warm 必须做**精确 first-layer
reparameterization**，否则 step 0 function 就变了。令原第一层 `a = W R + b`，
`R = μ + σ⊙z`：

```
W' = W diag(scale)
b' = b + W μ
```

则 `W' z + b' = W R + b`。对 `std < 1e-6` 的 degenerate 坐标沿用既有规则
（`z=0`，`scale=1`），并把 `W μ` 烘进 `b'`。Gate（§15）：

| quantity | max abs diff |
|---|---:|
| float64 identity（在 reconstruction `μ+scale·z` 上） | **≤ 6.2e-15** |
| float32 deployment（`original(R)` vs `transformed(z)`） | **≤ 4.4e-5 < 1e-4** |

float32 残差全部来自 frozen standardizer 自己的 degenerate 约定（fit-degenerate
坐标在该 fold 的 selection/holdout 上仍可能带极小残余变化，被 `z=0` 丢掉），不是
transform 错误。因此 L-warm 的 step 0 与 H0 在数值容差内**是同一个函数**。

## 7. Head matrix

| head | architecture | init | input | 是否训练 |
|---|---|---|---|---|
| **H0** | original `302→64→32→1` + LayerNorm(64) + Dropout(0.05) | checkpoint | raw `R` | 否（reference） |
| **S-scratch** | `302→13→13→1` pure ReLU MLP | scratch seed0 | `z` | 是，800 ep |
| **L-scratch** | `302→64→32→1` pure ReLU MLP | scratch seed0 | `z` | 是，800 ep |
| **L-warm** | original head module（含 LayerNorm/Dropout） | checkpoint + 精确 reparameterization | `z` | 是，800 ep |
| **S-raw**（control） | `302→13→13→1` pure ReLU MLP | scratch seed0 | raw `R` | 是，800 ep |
| **L-scratch-raw**（conditional，已运行） | `302→64→32→1` pure ReLU MLP | scratch seed0 | raw `R` | 是，800 ep |

关键设计：**S-scratch 与 L-scratch 是同一函数族，只差宽度**，因此
`Δcap = MAE(L-scratch) − MAE(S-scratch)` 干净隔离 capacity。L-warm 用原 head module
（含 LayerNorm/Dropout），回答「继续优化原 head」的问题。未测 FM/CatBoost/XGBoost/
spline/hinge；未做 width sweep；未延长 horizon。

## 8. Parameter accounting

从真实 state dict 精确计算：

| head | params |
|---|---:|
| H0 original head | **21633** = 21505 linear + 128 LayerNorm（Dropout 0） |
| S-scratch | **4135** |
| L-scratch | **21505** |
| L-warm | 21633（同一 module） |

整个 compact-v4 每 fold/seed 的实际总参数不同（fold-specific typed/parent vocab）。
以 fold0/seed0 为例：total **96141**，head **21633**，non-head **74508**。
替换成 small head 后 total = 74508 + 4135 = **78643**（10 个 fold/seed 平均
**78732.6**），相对原 head 节省 **17498 / 模型**（10 个平均 17498）。

> 注：任务 §49 的估算 ~99.6k → ~82k 与本仓库真实 checkpoint 不符；真实是
> ~96.1k → ~78.6k。若做 E2E 替换，参数节省是 **17498**，不是 ~17.4k 的近似值
> 与 ~99.6k 总参数的组合。

## 9. Training protocol

S / L-scratch / L-warm / S-raw / L-scratch-raw 全部使用**完全相同**的：

- loss L1/MAE；optimizer Adam(lr=1e-3, weight_decay=0)；
- deterministic mini-batch 512（seeded batch order，每 head 相同）；
- horizon 800 epochs（复用上一轮已验证的 fixed horizon，无 convergence sweep）；
- best-selection checkpoint（min 800 selection MAE）；
- `head_seed=0`。

trace 只在 epoch 100/200/400/800 记录 fit/selection MAE，且只做 eval-mode forward，
不触碰 optimizer/RNG（Test 11 证明 logging 不改变训练）。

## 10. Main results

Pooled（2 backbone seeds × 5 folds，outer-heldout 2000）：

| head | pooled mean MAE |
|---|---:|
| H0 | 0.175031 |
| **S-scratch** | **0.170280** |
| L-scratch | 0.170534 |
| **L-warm** | **0.169661** |
| S-raw | 0.168045 |
| L-scratch-raw | 0.164781 |

`S-scratch` 与上一轮 H2 pooled `0.170280` 的差 = **4.9e-7**，`H0` 与上一轮差 =
**2.4e-7** —— 协议被逐位复现。

| delta | pooled mean | 95% CI | P(>0) | fold-averaged + |
|---|---:|---|---:|---:|
| ΔS = H0 − S | **+0.004751** | [+0.00150, +0.00952] | 0.9997 | 5/5 |
| ΔL = H0 − L-scratch | **+0.004496** | [+0.00108, +0.00946] | 0.9983 | 5/5 |
| ΔW = H0 − L-warm | **+0.005369** | [+0.00414, +0.00660] | 1.0000 | 5/5 |
| Δcap = L-scratch − S | +0.000255 | [−0.00097, +0.00150] | 0.658 | 2/5 |
| Δinit = L-scratch − L-warm | +0.000873 | [−0.00390, +0.00406] | 0.695 | 3/5 |
| S-raw − S | −0.002234 | [−0.00356, −0.00090] | 0.0004 | 3/5 |

Per-backbone：

| backbone | ΔS | ΔL | ΔW | Δcap |
|---|---:|---:|---:|---:|
| seed 0 | +0.005757 | +0.005461 | +0.006600 | +0.000296 |
| seed 1 | +0.003745 | +0.003532 | +0.004138 | +0.000213 |

## 11. Capacity comparison

**Δcap = +0.000255，95% CI [−0.00097, +0.00150]，P(>0)=0.658，fold-averaged 仅
2/5 为正。** S-scratch 与 L-scratch 在完全相同 frozen-R / standardisation /
7200-800 / optimizer / horizon 下**在统计上不可区分**。

更强的否证来自 scale interaction：在 raw 输入上，large head **反超** small head：

- `Δcap_raw = MAE(L-scratch-raw) − MAE(S-raw) = −0.003264`（1/5 fold 为正）；
- raw small = 0.168045，raw large = 0.164781（large 好 +0.00326）。

即 capacity 的正负号都随输入 scale 反转。**Case A (CAPACITY GO) 明确不成立。**

## 12. Warm-refit comparison

**ΔW = +0.005369，CI [+0.00414, +0.00660]，P=1.0，5/5 folds。** 这是全场最稳、
最强的信号。原 `302→64→32→1` head 在 frozen `R` 上继续训练（step 0 与 H0 等价），
outer-heldout MAE 从 0.175031 降到 0.169661，而且比 scratch 的 S/L 还略好。
`Δinit = +0.000873`（CI 跨 0），说明 checkpoint init 与 scratch 没有稳定差异。

结论：**原 architecture 足够；joint training 结束时 head 对最终 `R` under-adapted。**
这是 Case C (LATE READOUT ADAPTATION) 的直接证据，也同时满足 Case B (POST-HOC
REFIT) 的条件（S ≈ L-scratch 且都 ≫ H0）；两者指向同一个 downstream 动作
（head-only adaptation）。由于 warm continuation 无需换 architecture、无需重新初始化
就达到最好效果，本阶段以 **LATE-ADAPTATION** 作为最精确的 mechanism 命名。

## 13. Raw-vs-standardized control

`S-raw`（0.168045）比 `S-std`（0.170280）**好 0.002234**，CI [−0.00356, −0.00090]，
P(>0)=0.0004。即 **standardisation 不是 gain 的来源，反而轻微有害**。触发条件
（|gap| > 0.002）满足，因此运行了 L-scratch-raw：0.164781，比 L-scratch-std 好
0.00575。结合 §11，raw 与大 head 的交互使 capacity 结论完全不稳定。

因此任务 §35 预期的「S-std ≫ S-raw → standardisation 贡献」**方向相反**。上一轮
H2 的 gain 是在 standardisation 之下拿到的，但不是因为它。

## 14. Learning curves

epoch 100/200/400/800 的 mean fit / selection MAE：

| head | ep100 fit / sel | ep200 | ep400 | ep800 |
|---|---|---|---|---|
| S-scratch | 0.0936 / 0.1797 | 0.0857 / 0.1786 | 0.0799 / 0.1801 | 0.0740 / 0.1817 |
| L-scratch | 0.0729 / 0.1789 | 0.0578 / 0.1792 | 0.0455 / 0.1822 | **0.0371 / 0.1871** |
| L-warm | 0.0930 / 0.1910 | 0.0866 / 0.1916 | 0.0879 / 0.1984 | 0.0779 / 0.2018 |

- **Pattern 1（overfitting）适用于 L-scratch**：fit 一路降到 0.037（远低于 S 的
  0.074），selection 反而从 0.179 升到 0.187。大 head 在 fit/selection 上明显过拟合。
- **Pattern 3（under-adaptation）适用于 L-warm**：从 H0 出发，fit 从 0.093 降到
  0.078，selection 单调变差但 outer-heldout 最好 —— 它学到的是「适应最终 R」，
  不是「在 selection 上更好」。
- S-scratch 的 selection 基本走平（~0.18），没有明显过拟合。

但**关键**：L-scratch 在 selection 上的过拟合**没有**转化为 outer-heldout 上的稳定
劣势（Δcap ≈ 0，raw 下甚至反转）。所以「大 head 过拟合」不足以支撑 capacity GO。
按 §38，不因 epoch 800 仍在变化而延长 horizon（记为 optimisation-speed caveat）。

## 15. Mechanism decomposition

1. **Capacity：否。** Δcap CI 跨 0，standardised 下仅 +0.00025，raw 下 −0.00326。
2. **Frozen refit：是。** ΔS、ΔL 都稳健 > 0（CI 下界 > 0），且与 head size 无关。
3. **Late readout adaptation：是，且最强。** ΔW 最稳最大，L-warm ≈/略优于 scratch。
4. **Initialisation path：弱。** Δinit CI 跨 0，无稳定结论。
5. **Standardisation：非但无助，反而有害。** S-raw 显著优于 S-std。
6. **Random init / capacity / optimisation：** 全部不足以解释；真正 binding 的约束是
   **backbone 训练结束后 head 没有跟上最终 representation**。

## 16. What is and is not proven

**Proven（在本 protocol 范围内）：**

- 上一轮 small-head signal **完全复现**（pooled ΔS = +0.00475，逐位匹配 H2/H0）。
- 该 gain 的**主因不是 head capacity**：在同一 frozen-R refit protocol 下
  `302→13→13→1` 与 `302→64→32→1` 不可区分（Δcap CI 跨 0，raw 下符号反转）。
- gain **主要是 frozen representation 上的 readout refit / late adaptation**：
  ΔS、ΔL、ΔW 都稳健为正；L-warm 用原 architecture、原 weights、原 head module
  即可达到最好水平。
- fit-only standardisation **不是** gain 来源（raw 反而更好）。
- 原 head 在 joint training 结束时确实 under-adapted to final `R`。

**Not proven：**

- 不能断言任何 small architecture 都不行（本轮只比 13→13 vs 64→32）。
- 不能断言 standardisation 永远有害（只在本 frozen `R` + direct refit 上观察到）。
- 不能断言 head-only adaptation 在 official valid 上一定改善（需要下一阶段
  leakage-safe 的 train/valid 协议）。
- 不能断言 pwn capacity 在所有 scale 下都不重要（raw 下 large 更好）。

## 17. Next-step authorization

按预注册：

- **不授权** `compact-v4 + 302→13→13→1` end-to-end（capacity 未 GO）。
- **不**扫 width/depth/activation；**不**重开 FM/CatBoost/XGBoost；**不**延长 horizon。
- 下一步测试：**canonical training → freeze backbone → leakage-safe head-only MAE
  adaptation**，在 official train/valid 下严格分离 selection 与 final evaluation，
  防止用 valid label 拟合 head 再在同一 valid 上宣称改善（§46–47）。
- 若未来真的做 small-head E2E：upstream shared tensors 必须**逐 tensor identical
  init**（`_copy_shared_upstream` primitive 已就绪），不能只设相同 random seed。

## 18. Final verdict

**LATE-READOUT-ADAPTATION GO（Case C，并满足 Case B 的 refit 条件）—— 不是 capacity。**

> 同一个 frozen compact-v4 `R`，small MLP 之所以看起来「更好」，不是因为
> 13→13 比 64→32 更强，而是因为原 jointly-trained head 在训练结束时没有适应最终
> representation。**任何**合理的 MLP head，只要在 final frozen `R` 上重新拟合
> （甚至只是 warm-continue 原 head），都能拿到几乎同样的 gain。因此这个现象应被
> 记为 **frozen-readout adaptation signal**，而不是 architecture novelty。

---

## Q1–Q17

- **Q1** H0 原 head 结构与参数量？`302→64→32→1` + LayerNorm(64) + Dropout(0.05)，
  **21633** params（21505 linear + 128 LayerNorm）。
- **Q2** Small H2 参数量？`302→13→13→1`，**4135**。
- **Q3** 若替换成 small head 整个模型参数量？真实 checkpoint total 96141（fold0/seed0）
  → **78643**；10 个 fold/seed 平均 **78732.6**，每模型节省 **17498**。
- **Q4** standardization-equivalent transform 数值成立？是。float64 identity
  ≤ 6.2e-15；float32 deployment ≤ 4.4e-5 < 1e-4（残差来自 frozen degenerate 约定）。
- **Q5** H0 pooled MAE？**0.175031**（与上一轮差 2.4e-7）。
- **Q6** S-small scratch pooled MAE？**0.170280**（与上一轮 H2 差 4.9e-7）。
- **Q7** L-large scratch pooled MAE？**0.170534**。
- **Q8** L-warm pooled MAE？**0.169661**。
- **Q9** pooled ΔS？**+0.004751**（CI [+0.00150,+0.00952]，P=0.9997，5/5）。
- **Q10** pooled ΔL？**+0.004496**（CI [+0.00108,+0.00946]，P=0.9983，5/5）。
- **Q11** pooled ΔW？**+0.005369**（CI [+0.00414,+0.00660]，P=1.0，5/5）。
- **Q12** 最关键 pooled Δcap？**+0.000255**（CI [−0.00097,+0.00150]，P=0.658，
  fold-averaged 2/5；raw 下 −0.003264，1/5）。
- **Q13** 两个 backbone 方向是否一致？ΔS/ΔL/ΔW/Δcap 两个 backbone mean 都为正
  （Δcap 两 seed 都是 +0.0003/+0.0002，但都很小且 fold 不一致）。refit/warm 方向
  一致，capacity **不**一致（raw 反转）。
- **Q14** learning curves 更支持 overfit 还是 optimization？L-scratch 是 overfit
  （fit↓、selection↑），但外推不掉；L-warm 是 under-adaptation（从 H0 快速改善，
  外推最好）。没有证据支持「大 head 是 optimization 慢」。
- **Q15** standardisation 贡献多少？**负贡献**：S-raw 比 S-std 好 0.002234
  （CI [−0.00356,−0.00090]）；L-scratch-raw 比 L-scratch-std 好 0.00575。
- **Q16** 是否需要 second head init？**不需要**。|Δcap| = 0.00025 不在 [0.001,0.003]
  borderline 区间内，未运行第二 init。
- **Q17** 最终属于哪个？**LATE-ADAPTATION GO**（Case C；同时满足 Case B 的 refit
  条件）。不是 CAPACITY / PATH-DEPENDENCE / MIXED / NO-GO / INCONCLUSIVE。

---

## 复现入口

```
uv run python -m tracks.ksvd.experiments.luyin16.zinc_graph_head_refit_capacity_decomposition <stage>
# stages: inventory parameters spec integrity splits standardization
#         refit summary bootstrap raw decision figures all
uv run pytest tracks/ksvd/tests/test_graph_head_refit_capacity_decomposition.py
```

输出目录：`tracks/ksvd/results/graph_head_refit_capacity_decomposition/`。
