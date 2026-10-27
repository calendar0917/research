# zinc_support_weighted_objective_v1 — 计划（训练前冻结）

**状态**：**规则**在任何训练之前冻结于 commit `c4e8e26`（2026-10-27T07:47+08:00）的
`configs/swo_v1.json` + `code/swo_weights.py`；**本文件的正文是 09:40 训练结束后的
转写**（唯一权威是上述 commit 里的 config 与代码，不是这份 markdown）。
后续 commit 只修了权重加载/校验路径，训练核心 `swo_train_core.py` 逐位未变
（sha256 `c0e9e4cd…`，`analysis/cpu_checks.json` 有记录）。

## 0. 这一轮是什么 / 不是什么

* **是**：对**历史最强主体 DCC-L（B-R0）** 做一次**目标函数层面**的对照实验，
  检验"训练标签的**利用方式**（而不是结构表示）是否是 SSRC 当前的瓶颈"。
* **不是**：新架构、新机制、新模块、新参数量；**不**改动模型前向、参数量、
  初始化、优化器、学习率、schedule、batch 顺序、更新次数（精确 47,400 updates）。
* **不**触碰 official test；权重函数**只用 train-10k 的结构**，不用 y、不用 valid、
  不用 test、不用任何模型误差、不用 row 172 信息。

## 1. 两个臂

| 臂 | 说明 |
|---|---|
| **E**（对照） | 原始配方：raw-y L1，全样本均匀权重 |
| **W**（实验） | **完全相同**的 L1，唯一差别是**每个分子一个固定标量权重** `w_i` |

`w_i` 由 train-10k 的 **radius-2 精确类型签名支持度**导出，函数在训练前冻结：

```
low4_i = 分子 i 中满足 1 <= count_train(sig(atom)) <= 4 的原子比例
z_i    = clip((low4_i - mean)/std, -1, +1)      # mean=0.06396, std=0.08814
w_i    = 1 + 0.5 * z_i                          # 斜率保守固定
w_i   /= mean(w)                                # 归一化到均值 1
```

理由：`count >= 5` 的环境在 10k 分子里"至少出现过 5 次"，`1..4` 是**已经在训练集
出现过、但出现得极少**的环境（真正的 unseen 无法靠加权帮到）。这个量**只看结构**，
不看标签，所以不构成对 y 的信息注入。

**冻结统计**（`analysis/weights_train10k.json`，向量 sha256
`a932a05638a0aff912b50df07eb20b08f4cd4cbf6cea936823d63b263bbe62ae`）：

| 量 | 值 |
|---|---|
| n | 10,000 |
| 签名词表大小 | 12,108 |
| low4 均值 / 标准差 | 0.06396 / 0.08814 |
| clip 命中比例 | 13.2%（z 触到 ±1） |
| w 范围 | [0.68396, 1.61022]，均值恰为 1 |
| **有效样本量 ESS** = (Σw)²/Σw² | **8,941.2（89.4%）** |
| w>1 的比例 | 38.9% |
| corr(w, n_atoms) | −0.2498 |

训练时在**远端重新计算**该函数并与 config 里的 sha256 逐位比对（先比对再训练，
不符即拒绝启动）——冻结的是**函数**，不是一个二进制文件。

## 2. 配方（R0，一字不改）

AdamW lr 1e-3 → 1e-5 cosine，warmup 500，wd 1e-5，dropout 0，clip 5，batch 128，
600 epoch = 47,400 updates，raw-y L1，1000 行 valid 上取严格最小作为 best。
参数 494,193（≤500k）。权重只改 loss 的**标量系数**：
`loss = Σ w_i |ŷ_i − y_i| / Σ w_i`（E 臂即 w≡1，见 `cpu_checks` 的 E 等价性检查）。

## 3. 种子与配对

* 主对比：**同一 seed、同一 SSL 编码器、同一初始状态**的 E vs W（seed 0 先跑）。
* seed 1 配对：**只有在 seed 0 给出可解释结果后才跑**（本计划允许跑，属预注册范围）。
* seed 2/3：**只有 W 显示出稳定且足够大的增益才考虑**（当前规则下不跑）。
* 已知限制：seed ≥ 2 复用 `pretrain_encoder_s1.pt`，初始化多样性被截断（见
  `zinc_adaptive_arch_search_v1/ERRATA.md` E1）。

## 4. 预注册判定（`configs/swo_v1.json`）

* `PROCEED_TO_SEED1`：`W − E ≤ −0.003` **且** `radius2_low_support` 改善
  **且** ordinary（y>−3）回归 ≤ +0.002。
* `CONFIRM_EXTENSION`：seed 0 / seed 1 同号，配对均值 ≤ −0.003。
* `STOP_NO_EFFECT`：`−0.003 < W − E < +0.0025`，或跨 seed 符号翻转。
* `STOP_NEGATIVE`：`W − E ≥ +0.0025`，或 ordinary 回归 > 0.002。
* **NO_RESCUE**：第一对跑完后**不**改权重函数（斜率、clip、band、归一化），
  不做超参扫描，不换 band 定义。

必报清单：overall 1000、row 172 保留、ordinary、tail、radius-2 三档、rings≥4、
worst-10 vs 其余 990、W20、T100、train@best/final、配对 bootstrap CI、
E 重训 vs 历史锚点的漂移。

## 5. 预算

软 14 GPU·h / 硬 16 GPU·h（全局 track 预算沿用）；本轮 4 个 run ≈ 1.45 GPU·h，
墙钟停止线 T+6.5h。
