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

17 项 gate 全过；全部走 `MixA.mix` / `MixA._attend` 这一条生产代码路径，不是公式副本。

| gate | 内容 | 结果 |
|---|---|---|
| E0 | 四种算术模式参数张量/构造顺序相同；拷贝同一 canonical 初态 | 4 模式均 460,017 参数、82 个 state 张量；拷贝后最大差 **0.0**；init sha 全同 |
| E0 | 新臂初态是否就是历史 ho_A_s0 的初态 | **是**，`4b8c59f8…36ffd9` 逐位一致（无需重建历史环境） |
| E1 | legacy attn vs 纯点积 attn（fp64） | 最大差 **2.2e-16** |
| E1 | `l2` attn vs `softmax(-‖q−k‖²/(2√d)+bias)` | 最大差 **4.4e-16**；与 dot 差 **0.76**（不是同一算子） |
| E1 | 给 legacy logits 加任意逐 query 常数 | 权重最大变化 **2.8e-16**（数值上重演“行常数被抵消”） |
| E2 | q=0、两个 key 范数 1 与 3、bias=0 | legacy **0.5/0.5**；`l2` **0.881/0.119**，与闭式 softmax(−‖k‖²/(2√d)) 一致 |
| E3 | legacy = mass × 归一化权重；GN 与 legacy 用同一 attention | attn 差 2.2e-16、ctx 相对差 3.6e-16 |
| E4 | 新运算的最终 loss 梯度（模块级 5 个参数 + 全模型 1 个输入，fp64 中心差分） | 最大相对误差 6.9e-8 / 2.9e-8（l2）/ 1.7e-8（gn） |
| E5 | 冻结 ho_A_s0 checkpoint：legacy vs 纯 dot，64 条真实训练结构 | 预测最大差 **2.4e-7**（预测尺度 3.5，≈7e-8 相对），layer0 attn 差 5.1e-7 → **只是浮点噪声** |
| E5 | legacy 重算历史 valid 预测 | 最大差 2.9e-6，MAE 0.08129934127 vs 记录 0.08129934083（差 4.5e-10）→ checkpoint 确实是 legacy 算子训出来的 |
| E6 | 训练/评估同一 forward（新模式） | train/eval 输出差 **0.0**；直接 forward vs `ad_train.evaluate` 差 **0.0** |
| E7 | 9000/1000 内部划分 | 与历史 `holdout_split_1000.json` 的 `39295fd7…` 相同 |

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
逐层 mean 0.483 / 0.469 / 0.379、max 2.0，98.5–98.7% 的 (query,head) 对落在 [0.9,1.1] 之外
（`analysis/errata_checks.json` → `E5b`）。也就是说这个“可变总增益”不是理论隐患，
而是训练真正用上的东西——它把对混合结果的写入整体压到 ~0.5 倍，并在不同 query 间大幅变化。
这条事实是 GN 臂（去掉可变总增益）的购买理由。

## 4. 两个补救实验（见 §5 结果）

- **L2G** = 原关系门 + 真正的 L2 打分（`ken.unsqueeze(-2)`）：`score=(q·k − ½‖k‖²)/√d + β(r)`，
  `m = Σ a·g·v`。
- **GN** = 原 dot 打分 + 原门，但门控权重重新归一化：`w = a·g`，`m = Σ (w/Σw)·v`。

两者与历史 A **参数张量、构造顺序、初态完全相同**（E0 逐位一致），只差计算开关；
配方、数据划分（9000/1000）、SSL 初始化、batch 顺序、600 epoch、checkpoint 选择规则全部复用。

## 5. 结果

（待两个臂完成后填入：best-valid / 持续窗口 / internal holdout / train MAE / 相对旧 A 差值 / 成本。）
