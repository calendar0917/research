# ERRATUM — GOS-v1 route A: what the shipped operator actually computes

**Scope:** `tracks/structural-sufficiency/zinc_global_operator_search_v1`（下文 GOS-v1）。
**Status:** 训练提交 `19652e0` / `b245e88` 中的 `MixA` 实现有广播错误；本文件说明实际运算、
撤回的机制表述、**不受影响**的成绩，以及两个补救实验（L2G / GN）。
**不改动：** `REPORT.md` 原文、`runs/A_s*/best.pt` 等历史 checkpoint、任何历史数字。

---

## 1. 已核实的广播错误

出问题的三行（`code/gos_models.py`，提交 `19652e0`/`b245e88`）：

```python
score = torch.einsum('bhid,bhjd->bhij', q, k)   # [B,H,query,key]
ken   = 0.5 * k.square().sum(-1)                # [B,H,key]
score = (score - ken.unsqueeze(-1)) / sqrt(d_head)
attn  = softmax(score + relation_bias, dim=-1)
```

`ken` 的最后一维是 **key** 轴；`ken.unsqueeze(-1)` 得到 `[B,H,key,1]`，与 `score` 的
`[B,H,query,key]` 对齐后 N 落在 **query** 轴上，1 落在 key 轴上。因此对固定 query 的所有
key 减去的是**同一个常数**，行 softmax 把它完全抵消。正确写法是 `ken.unsqueeze(-2)`
（沿 key 轴变化），那才等价于 `softmax(-||q-k||²/(2√d) + bias)`。

所以历史 route A 的有效运算是：

```
a = softmax(q·k/√d + β(r))              # 普通点积 attention + 关系 bias
g = 1 + tanh(γ(r))                      # 关系条件内容门
m = Σ_c a_bc · g_bc · v_c               # 门控后的**未归一化**加权和
h ← h + O(m) ; h ← h + FFN(AdaRMSNorm(h))   # AdaRMSNorm 保留
```

即 **dot-product attention + relation-conditioned content gate + AdaRMSNorm**。
它没有、也不可能验证“key-energy / L2 幅值校正”。

## 2. 用真实 forward 做的核实（`code/gos_errata_checks.py`，`analysis/errata_checks.json`）

25 项 gate（含后来为 `gnlog` 补的等价/稳定性、以及两个新臂 checkpoint 的独立重算，全部通过；`analysis/errata_checks.json`）；
全部走 `MixA.mix` / `MixA._attend` 这一条生产代码路径，不是公式副本。

| gate | 内容 | 结果 |
|---|---|---|
| E0 | 四种算术模式参数张量/构造顺序相同；拷贝同一 canonical 初态 | 4 模式均 460,017 参数、82 个 state 张量；拷贝后最大差 **0.0**；init sha 全同 |
| E0 | 新臂初态是否就是历史 ho_A_s0 的初态 | **是**，`4b8c59f8…36ffd9` 逐位一致（无需重建历史环境） |
| E1 | legacy attn vs 纯点积 attn（fp64） | 最大差 **2.2e-16** |
| E1 | `l2` attn vs `softmax(-‖q−k‖²/(2√d)+bias)` | 最大差 **4.4e-16**；与 dot 差 **0.76**（不是同一算子） |
| E1 | 给 legacy logits 加任意逐 query 常数 | 权重最大变化 **2.8e-16**（数值上重演“行常数被抵消”） |
| E2 | q=0、两个 key 范数 1 与 3、bias=0 | legacy **0.5/0.5**；`l2` **0.881/0.119**，与闭式 softmax(−‖k‖²/(2√d)) 一致 |
| E3 | legacy = mass × 归一化权重；GN 与 legacy 用同一 attention | attn 差 2.2e-16、ctx 相对差 3.6e-16 |
| E3b | `gnlog` 与 `gn` 定义式等价（fp64） | ctx 相对差 1.4e-16、有效权重差 2.2e-16 |
| E3c | 门被学到极负时（γ bias −25），定义式给 NaN、`gnlog` 给有限值 | 同参数/同输入对照 |
| E4 | 新运算的最终 loss 梯度（模块级 5 个参数 × legacy/l2/gn/gnlog + 全模型 1 个输入，fp64 中心差分） | 最大相对误差 6.9e-8 / 2.9e-8（l2）/ 1.7e-8（gn）/ 4.3e-8（gnlog） |
| E5 | 冻结 ho_A_s0 checkpoint：legacy vs 纯 dot，64 条真实训练结构 | 预测最大差 **2.4e-7**（预测尺度 3.5，≈7e-8 相对），layer0 attn 差 5.1e-7 → **只是浮点噪声** |
| E5 | legacy 重算历史 valid 预测 | 最大差 2.9e-6，MAE 0.08129934127 vs 记录 0.08129934083（差 4.5e-10）→ checkpoint 确实是 legacy 算子训出来的 |
| E6 | 训练/评估同一 forward（新模式 l2 / gn / gnlog） | train/eval 输出差 **0.0**；直接 forward vs `ad_train.evaluate` 差 **0.0** |
| E7 | 9000/1000 内部划分 | 与历史 `holdout_split_1000.json` 的 `39295fd7…` 相同 |
| E8 | 用训练时的算术重读两个新 checkpoint，重算 valid/留出 | valid 差 ≤2e-8、留出差 ≤8e-9（CPU vs A100 噪声）；同一 checkpoint 换回 legacy 算术会变形（0.114650 / 0.120170，仅示意，不是成绩） |

（E0/E1/E5 用的都是真实 `MixA`/`Backbone`，`E5` 的权重来自 `runs/ho_A_s0/best.pt`。
E5 中把同一 checkpoint 的算术换成 `l2`/`gn` 只作**量级参考**（预测最大差 0.046 / 0.203），
不作为任何成绩，也不报告为“修复后的 A”。）

## 3. 撤回 / 保留

**撤回（机制解释）：**

- “`softmax((q·k − ½‖k‖²)/√d + β(r))` 的幅值/key-energy 校正有效”——从未被训练验证。
- REPORT §0/§3/§8 中把 A 的 valid 增益归因于“幅值敏感 logit + 关系门控**打包**”的表述：
  该“包”里实际只有一个有效成分（门控），外加普通点积打分。因此
  “组件包值 ~0.005，但未拆分”这句话应改为：**未拆分的成分只有门控一项**。
- §8 提出的“下一步把 −½‖k‖² 与 1+tanh(γ(r)) 拆开”的前提（幅值项可能提供尾部修正）不再成立：
  这个实验从没跑过 L2 版本。本轮用一个正确的 L2 臂（L2G）直接替代该拆分。

**保留（成绩与对照，均不受影响）：**

- A 的 valid 0.075919 / 0.074577、家族均值 0.075248，test 0.067623 / 0.067963 —— 数字有效，
  只是它们对应“dot + 关系门控”这个算子。
- A 优于同载体 C0（−0.0052 valid，两 seed、4 个窗口）、Asm/K/N0 全部回到 C0 带内 —— 对照仍然有效，
  因为 Asm 关掉的是“门控 + （无效的）幅值项”，实际等于“关掉门控”。
  **注意 Asm 与 A 的差别因此不再是“幅值项 + 门控”两件东西，而是“有没有内容门”（外加 legacy 里那条
  被抵消的常数项）。**
- 9,000 行重训 + 内部 1,000 行留出（A 0.114666 vs C0 0.124116，差 −0.00945）—— 有效。
- A 家族与 DCC-L 在 test 上无法区分（+0.00047，CI 跨 0）—— 有效。
- A160 加宽失败、reg_A 正则失败 —— 有效。

**同时被这次核实顺带确立的一件事**（不属于撤回，而是新增的事实）：
冻结的 `ho_A_s0` 在 official valid 上，门控总增益 `mass = Σ_c a_bc g_bc` 的分布是
逐层 mean 0.483 / 0.469 / 0.379、max 2.0，97.6–98.7% 的 (query,head) 对落在 [0.9,1.1] 之外
（`analysis/errata_checks.json` → `E5b`）。也就是说这个“可变总增益”不是理论隐患，
而是训练真正用上的东西——它把对混合结果的写入整体压到 ~0.5 倍，并在不同 query 间大幅变化。
这条事实是 GN 臂（去掉可变总增益）的购买理由。

## 4. 两个补救实验

- **L2G** = 原关系门 + 真正的 L2 打分（`ken.unsqueeze(-2)`）：`score=(q·k − ½‖k‖²)/√d + β(r)`，
  `m = Σ a·g·v`。
- **GN** = 原 dot 打分 + 原门，但门控权重重新归一化：`w = a·g`，`m = Σ (w/Σw)·v`。
  按定义式（除法）实现的第一次运行 `gos-gn0` 在 **ep284 因 `non-finite gradient` 退出**：
  `1/Σ(a·g)` 在该 query 的门把全部 key 压到接近 0 时把反向放大到 1e11 量级（冻结 legacy
  checkpoint 上实测 mass 最小 1.6e-11）。这是实现形式的数值病态，不是算法结论，按规则停下修错；
  重跑用同一算子的无除法形式 `gnlog = softmax(score + log(1+tanh γ))`，
  `log(1+tanh x) = log2 − softplus(−2x)`，**没有新增参数、没有温度**，fp64 下与定义式等价
  （E3b：ctx 相对差 1.4e-16、有效权重差 2.2e-16），同参数下定义式给 NaN、`gnlog` 给有限值（E3c）。
  失败运行的 checkpoint/逐 epoch 日志留在 `runs/gn_A_s0_failed_ep284/`，成本计入预算。

两者与历史 A **参数张量、构造顺序、初态完全相同**（E0 逐位一致），只差计算开关；
配方、数据划分（9000/1000）、SSL 初始化、batch 顺序、600 epoch、checkpoint 选择规则全部复用。

## 5. 结果

同一 9,000 行、同一初态、同一配方；选点只看 official valid，内部 1,000 行只在选定后过一遍。

| arm | 算术 | 参数 | best-valid@ep | W20 | final | train@best | **内部留出 1000** | GPU·h |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A**（历史 `ho_A_s0`） | dot + gate | 460,017 | **0.081299**@364 | 0.083016 | 0.083032 | 0.015609 | **0.114666** | 0.752 |
| **L2G**（本次） | 真 L2 + gate | 460,017 | 0.080472@427 | 0.081500 | 0.081532 | 0.013085 | 0.115199 | 0.756 |
| **GN**（本次，`gnlog`） | dot + 归一化 gate | 460,017 | 0.081915@405 | 0.082787 | 0.082824 | 0.013330 | 0.120440 | 0.846 |

**持续窗口**（10-epoch 均值，valid MAE；单 epoch 噪声大，这里给窗口）

| 窗口终点 ep | A | L2G | ΔL2G−A | GN | ΔGN−A |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.112197 | 0.120607 | +0.008410 | 0.115957 | +0.003760 |
| 200 | 0.094519 | 0.097744 | +0.003225 | 0.095383 | +0.000864 |
| 300 | 0.087487 | 0.085935 | −0.001552 | 0.087570 | +0.000083 |
| 400 | 0.084171 | 0.082936 | −0.001235 | 0.084347 | +0.000175 |
| 500 | 0.083186 | 0.081548 | −0.001639 | 0.082992 | −0.000194 |
| 600 | 0.083024 | 0.081488 | −0.001537 | 0.082795 | −0.000229 |

两臂都没有触发预注册的停止规则（ep300 与 ep400 未同时落后 ≥0.004），因此都跑满 600 epoch。

**内部 1,000 行留出**（逐分子配对；bootstrap 95% CI）

| arm | MAE | 中位 AE | Δ vs A | CI | A 更优比例 | 去掉 A 最差 10 行 |
|---|---:|---:|---:|---|---:|---:|
| A | 0.114666 | 0.043296 | — | — | — | — |
| L2G | 0.115199 | 0.039715 | **+0.000533** | [−0.002949, +0.004023] | 51.5% | +0.001565 |
| GN | 0.120440 | 0.043168 | **+0.005774** | [+0.001534, +0.010244] | 48.0% | +0.005109 |

**按旧 A 的固定逐分子误差分组**（留出集上 A 的 AE 三档 + 最差 10%；只作描述）

| arm | q0（A 最准 333） | q1（333） | q2（A 最差 334） | A 最差 10%（100） | 去掉 A 最差 10 行 |
|---|---:|---:|---:|---:|---:|
| A 组内均值 | 0.012576 | 0.044450 | 0.286456 | — | — |
| ΔL2G | +0.015863 | −0.003661 | **−0.010569** | **−0.008751** | +0.001565 |
| ΔGN | +0.018935 | −0.000450 | −0.001140 | **+0.011761** | +0.005109 |

读法（分组的条件化偏差必须说清）：q0 是“A 恰好误差极小”的 333 行，按定义对 A 有利，
不能读成“L2G 在简单分子上更差”；有信息量的是 q2 与 A 最差 10%：**L2G 的改善几乎全部集中在
旧 A 的高误差分子上（−0.011 / −0.009），GN 恰好相反（尾部 +0.012）**。

**实际成本**：L2G 0.756 + GN（`gnlog`）0.846 + 失败的 GN 定义式 0.406 = **2.008 A100·h**，
上限 2.0。超 0.008 GPU·h（0.4%），是最后约 1 分钟的余弦尾部：预算守卫算到按当时速率只能跑到
ep596，停在那里会丢掉 `curve.csv`/`train.json` 和那一次一次性留出评估，因此选择跑完并在报告里
明写超支，而不是悄悄停在 596 或改口径。CPU 上的 25 项 gate、拉取、分析不占 GPU。

### 结论（限定在本配置、单监督 seed、9,000 行训练）

1. **L2G 没有得到留出复现的改善**。official valid 略好（best −0.00083；ep500/ep600 的窗口
   −0.0015），但内部留出点上方向相反（+0.00053）、CI 跨零（[−0.00295, +0.00402]），
   且训练行拟合更好（train@best 0.0131 vs 0.0156）。按预注册规则：**不进入第二 seed / 全 10k 验证**，
   保留旧的 dot+gate。没有证据表明“正确公式必定更好”，也没有证据表明它更差——它是**无差别**。
   唯一可说的分布效应是：L2G 把误差从旧 A 的尾部往中段搬（q2 −0.011、A 最差 10% −0.009，
   代价是其余行 +0.0016）。
2. **GN 在留出上退化**（+0.0058，CI 不含 0），而 official valid 基本持平（窗口 −0.0002，
   best +0.0006）。训练正常（无 NaN、train@best 0.0133、best-valid 平台与 A 同形），因此这不是
   训练失败：**去掉可变总增益后在选点集上看不出来，在新分子上看得出来**。这与“门控的总增益
   mass 携带跨分子可迁移的幅度信息”一致——冻结 A 上实测 mass 逐层均值 0.483/0.469/0.379、
   max 2.0、97.6–98.7% 的 (query,head) 落在 [0.9,1.1] 之外（E5b），即它是一个被训练实际使用、
   且逐 query 大幅变化的乘性幅度。GN 的留出值（0.1204）落在 A（0.1147）与 C0（0.1241）之间：
   混合本身仍比无混合好，但**去掉总增益后，A 相对 C0 的增益少了一半以上（−0.0095 → −0.0037）**。
3. 两条轨迹都**没有**赢过 DCC-L 的锚（valid 0.07599 / test 0.0673 为全 train 口径，不能与
   9,000 行训练的数字直接比）；本轮也没有读 official test。

## 6. 下一决定

**停止这条“打分几何 / 权重归一化”线**，不为 L2G 或 GN 买第二个 seed、不做组合臂、不做全 10k。

唯一有跨分子证据支撑的下一步（如果将来只买一条轨迹）：**把门控的质量 `mass_b = Σ_c a_bc g_bc`
显式作为每层一个标量通道送进 reader**（一个 `Linear(3, k)` 级的新增，或 3 个标量特征），
因为本轮唯一在留出上可测的机制就是这个 mass：去掉它掉 0.0058，而改打分（L2）无差别。
这不改变关键结论所需的东西，也不在本轮执行（不在 2.0 GPU·h 内、且不属于“纠正语义”）。

## 7. 复现

- 代码：`code/gos_models.py`（`MixA._attend` 的 5 种 mode）、`code/gos_runner.py`（`--mix`）、
  `code/gos_errata_checks.py`（25 项 gate → `analysis/errata_checks.json`）、
  `code/gos_errata_analyze.py`（→ `analysis/errata_results.json`）。
- 命令：
  `python code/gos_runner.py train --route A --mix l2  --seed 0 --tag l2g --holdout 1000 --device cuda:0`
  `python code/gos_runner.py train --route A --mix gnlog --seed 0 --tag gn --holdout 1000 --device cuda:0`
- 曲线：`runs/l2g_A_s0/curve.csv`、`runs/gn_A_s0/curve.csv`、`runs/ho_A_s0/curve.csv`。
- checkpoint：`runs/l2g_A_s0/best.pt`（sha `97818a8a…82caa`，ep427）、
  `runs/gn_A_s0/best.pt`（sha `a96f7807…f681c`，ep405）、历史 `runs/ho_A_s0/best.pt`（未改动，
  sha `6b40dc12…2faa6`）。
- 账本：`LEDGER.md`（预注册、停止规则、成本、超支说明）。

