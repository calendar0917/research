# SCSB v1 — 支持度条件的结构回退算子：Phase 1 最终报告

轮次：`zinc_support_conditioned_backoff_v1`
协议：`ssrc-zinc-support-conditioned-backoff-v1`（`configs/scb_v1.json`，预注册见 `PROTOCOL.md`）
状态：**Phase 1 停止；机制关闭；Phase 2 未解锁、未运行。**
official test：**全程未构造、未读取、未评分。** Phase 1 也未读取 official valid。

---

## 0. 结论（TL;DR）

预注册判定：**`STOP_UNSAFE`**（`ordinary_seen` 与 `tail` 安全门失败）。但这不是一个“只差安全门”的结果：

- **S−L primary = +0.003364**（两个 seed 都为正；目标 ≤ −0.015）；95% CI `[−0.004159, +0.010948]`。
- **S−L natural overall = +0.002805**（两个 seed 都为正；目标 ≤ −0.003）；95% CI `[−0.000143, +0.005729]`。
- **S−P primary = +0.001502**（seed 0 `+0.003502`，seed 1 `−0.000498`，符号分裂）；natural `−0.000227`，CI 跨 0。
- **L−P primary = −0.001862**（L 略好，CI 跨 0）；L−P natural `−0.003032`，CI 上界 `−0.000175`。

即：**支持度条件的局部分统计收缩没有带来任何实用收益；它在 primary、整体、普通分子和长环尾部全部变差；真实支持度位置也不比同分子内置换更好。** 这不是 near miss，也不是“只修了人工环尾”的失败模式——它连人工环尾也没有修好。

GPU 预算：**1.5585 A100·h**（S/P 四 run + smoke；L 为冻结 checkpoint，0 新 GPU 时），低于 3 h 软上限。
提交：`b9670777043b7ad5fd855ed16dc8c2cc3df2ccb4`（Phase 1 runs）；Phase 0 证据绑定同一提交。

---

## 1. 本轮测试的数学对象

### 1.1 干预

DCC-L 的局部 composer 输入为

```
Wl_b = sum_{j touches b} w_j,    Vl_b = sum_{j touches b} w_j^2
```

本轮只把它替换为

```
Wbar_b = alpha_b * Wl_b + (1 - alpha_b) * Wg_b
Vbar_b = alpha_b * Vl_b + (1 - alpha_b) * Vg_b
```

其中 `Wg_b, Vg_b` 是模型**已有的** `group_to_bonds` 图级均场统计，`alpha_b = s_b/(s_b+4)`，
`s_b = min(s_u, s_v)`，`s` 是键端点原子的精确 typed radius-2 签名在 **Fit** 中的 document frequency。
训练时每分子排除自身（`df−1`），推理只用 Fit 表；计数与标签无关；Monitor/Locked/valid/test
结构不进入支持表。无新参数、无新模块、无新特征、无标签加权、无蒸馏/集成。

### 1.2 三个臂

| 臂 | alpha | 说明 |
|---|---|---|
| **L** | 不传 alpha（等价 1） | 原始 DCC-L；冻结 DCB Fit8k seed0/1 best-monitor |
| **S** | 真实训练支持度 | 本轮候选 |
| **P** | 同分子内固定 salt 置换 | 保留每分子 alpha 多重集，破坏“支持度↔键位置”对应 |

L 臂没有重训：它直接使用 `zinc_dccl_bottleneck_v1/runs/fit8k_s{0,1}` 的冻结 checkpoint。
该 checkpoint 与本轮 S/P 使用**完全相同的 Fit8k、初始化、批序、47,400 updates 和 recipe**；
Phase 0 已验证其在 Locked 上的预测与保存预测最大差 `1.9e-6`，且 `alpha=None`/`alpha=1`
前向逐位相同。

### 1.3 冻结划分与分组

复用 label-blind 的 `cscl-predictive-bottleneck-v1-A`：Fit 8,000 / Monitor 1,000 / Locked 1,000；
三组互斥、覆盖 10k、sha256 与冻结 JSON 一致。所有臂同一 Fit；Monitor 只用于每 epoch 选点；
Locked 只在四 run 全部结束后由 `scb_score.py` 读一次。

| 组（Locked） | n | 定义 |
|---|---:|---|
| `primary` | 278 | `label_excess == 0` 且至少一个原子的 k2 在 Fit 中 df=0 |
| `unseen_all` | 287 | 至少一个 k2 在 Fit 中 df=0 |
| `low4_nopen` | 388 | 无 unseen，含 df 1–4 原子，无人工环惩罚 |
| `full_nopen` | 303 | 所有原子 df≥5，无人工环惩罚 |
| `ordinary_seen` | 691 | `low4_nopen + full_nopen` |
| `ordinary_label` | 932 | `y > −3` |
| `tail` | 68 | `y ≤ −3` |
| `penalty_all` | 31 | `label_excess > 0` |

`label_excess` 来自冻结的 long-cycle audit，只用于评估分组，不进入训练/支持/alpha/选点。
primary 组平均原子数 22.02，exact-longest≥7 比例 41.4%；Locked 其余 722 行为 23.32 / 32.5%。

### 1.4 alpha 的真实分布（Phase 0 冻结）

| 集合 | mean | median | alpha=0 | alpha≤0.5 | s1–4 均值 | s5–19 均值 | s≥20 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit（LOO） | 0.7859 | 0.9216 | 5.79% | — | 0.3445 | 0.7067 | 0.9537 |
| Locked 全部键 | 0.7840 | 0.9200 | 5.37% | 17.17% | 0.3419 | 0.7059 | 0.9538 |
| Locked primary | 0.6419 | — | 19.15% | 33.02% | 0.3256 | 0.7015 | 0.9539 |

关键事实：**65% 的 Fit 键 s≥20，alpha≈0.954**；因此 S 不是“只对稀有键回退”，而是对大多数键
也注入约 4.6% 的均场分量。primary 中 19.15% 的键 s=0，alpha=0，完全走均场。
P 臂逐分子多重集逐位相同，87.4% 的 primary 键 alpha 改变，逐分子 corr(alpha_S, alpha_P) 均值
0.00073（池化 0.1186 是分子间均值差造成的 Simpson 假象）。

### 1.5 Phase 0 门控（CPU，13/13 通过）

`analysis/cpu_checks.json` 记录：

- 划分隔离、支持表 Fit-only、标签置换/删除 held-out 结构后支持表 hash 不变；
- 25 个 Fit + 25 个 Locked 分子的 LOO 暴力重算最大 alpha 误差 `0.0`；加查询后再 LOO 不变；
- 50 个分子原子重编号/键端点交换/键序随机化后签名与 alpha 多重集不变；
- `SCBDualChannel` 参数 494,193；初始 state hash 与 DCB 记录一致；`alpha=None` 与原始
  `DualChannel` 逐位相同；`alpha=1` 与 `alpha=None` 最大差 `0.0`；加载 DCB 冻结 L 后
  Locked 预测与保存预测最大差 `1.9e-6`；
- 梯度：所有参数有限；local composer 第 2 步梯度非零（第 1 步因 reader 新列零初始化而
  故意为零，reader 新列第一步后唤醒）；无 NaN；优化器覆盖全部参数；
- 合成图：`alpha=0` 时 `Wmix=Wg, Vmix=Vg`，`alpha=1` 时 `Wmix=Wl, Vmix=Vl`，且两者确实不同；
- 训练循环：L 臂一个完整 horizon 的 epoch 1 与冻结 DCB L 的 `curve[0]` 对齐，
  `online_train_mae` 差 `1.9e-9`、`monitor_mae` 差 `1.2e-8`、lr/step 相同；
- cache/checkpoint hash、无 test 访问扫描全部通过。

---

## 2. 主结果（best-monitor，预注册主 checkpoint）

### 2.1 各臂 raw-y MAE

| 臂 | seed | primary | natural | all | ordinary | low4 | full | unseen_all | tail | penalty | train@best | W20 | T100 | best ep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L | 0 | **0.116376** | **0.084755** | **0.083573** | **0.050528** | 0.060510 | 0.037746 | **0.155348** | **0.293532** | **0.525995** | 0.015415 | **0.123819** | **0.123834** | 437 |
| L | 1 | **0.121635** | **0.084853** | **0.083423** | **0.047367** | **0.054426** | **0.038328** | 0.162758 | **0.304300** | **0.544443** | 0.007030 | 0.128421 | 0.128481 | 552 |
| S | 0 | 0.122987 | 0.086567 | 0.085259 | 0.050212 | 0.061264 | **0.036059** | 0.163986 | 0.303434 | 0.528138 | 0.010420 | 0.127915 | 0.127937 | 509 |
| S | 1 | 0.121753 | 0.088651 | 0.087271 | 0.051559 | 0.058481 | 0.042695 | 0.163787 | 0.327526 | 0.574077 | 0.004329 | 0.128507 | 0.128528 | 607 |
| P | 0 | 0.119485 | 0.086577 | 0.085357 | 0.051058 | 0.061376 | 0.037846 | 0.159913 | **0.293473** | 0.543824 | 0.004012 | 0.127738 | 0.127798 | 609 |
| P | 1 | 0.122251 | 0.089096 | 0.087770 | 0.051023 | 0.059951 | 0.039592 | 0.165616 | 0.344264 | 0.597653 | 0.007921 | 0.124743 | 0.124765 | 543 |

粗体为同列最好。L 在 primary、natural、all、ordinary、unseen_all、penalty 上最好；
tail 上 L/P 基本持平（P s0 仅低 0.00006）。S 只在 `full_nopen` seed0、`ordinary_seen` seed0
（0.050212 vs 0.050528）和 `train@best` 上更低；这些单 seed 的微小优势没有在 seed 均值上保持。`train@best` 更低但 Locked 更差，是本轮最重要的
诊断之一（见 §4.3）。

### 2.2 配对对比

Δ = 前者 MAE − 后者 MAE；负值 = 前者更好。所有区间为 10,000 次分子级 bootstrap
（natural 为 tier 分层 bootstrap；primary key-cluster 为 274 cluster / 270 singleton）。

| 对比 | primary seed Δ | seed mean | 95% CI | drop1 | drop10 | cluster CI | natural Δ (CI) |
|---|---|---:|---|---:|---:|---|---|
| S−L | +0.006611 / +0.000118 | **+0.003364** | [−0.004159, +0.010948] | +0.004553 | +0.001278 | [−0.004120, +0.011064] | **+0.002805** ([−0.000143, +0.005729]) |
| S−P | +0.003502 / −0.000498 | **+0.001502** | [−0.004262, +0.007498] | +0.000140 | −0.000409 | [−0.004119, +0.007460] | −0.000227 ([−0.002407, +0.002027]) |
| L−P | −0.003109 / −0.000616 | **−0.001862** | [−0.008734, +0.004930] | −0.002582 | −0.001028 | [−0.008796, +0.004978] | −0.003032 ([−0.005818, −0.000175]) |

安全与稳定性：

| 对比 | ordinary max seed Δ | tail max seed Δ | ordinary_label max | seed spread | 符号 |
|---|---:|---:|---:|---:|---|
| S−L | **+0.004192**（> +0.002 cap） | **+0.023226**（> +0.010 cap） | +0.002434 | 0.006493 | 同为正 |
| S−P | +0.000536 | +0.009961 | +0.000686 | 0.003999 | **分裂** |
| L−P | −0.000530 | +0.000060 | −0.001749 | 0.002493 | 同为负 |

**S−L 的失败是弥散的，不是少数行驱动**：primary 上 seed0 有 119 个分子 S 更好 / 159 个更差，
seed1 为 142 / 136（但损失幅度更大）；去掉 |ΔAE| 最大的 1 行和 10 行后仍为 `+0.004553` 和
`+0.001278`（仍为正）。key-cluster CI 与分子 bootstrap 几乎相同（primary 几乎全是 singleton cluster）。

### 2.3 final checkpoint 稳健性

预注册主指标用 best-monitor；final 只作稳健性读数。结论不变：

| 臂 | seed | primary | natural | all | ordinary | tail |
|---|---:|---:|---:|---:|---:|---:|
| L | 0 | 0.115919 | 0.083781 | 0.082548 | 0.048575 | 0.294433 |
| L | 1 | 0.121212 | 0.085110 | 0.083694 | 0.047474 | 0.309046 |
| S | 0 | 0.123188 | 0.086379 | 0.085054 | 0.050186 | 0.300442 |
| S | 1 | 0.122319 | 0.088618 | 0.087219 | 0.051275 | 0.327110 |
| P | 0 | 0.119519 | 0.086445 | 0.085228 | 0.050820 | 0.293626 |
| P | 1 | 0.121963 | 0.088452 | 0.087098 | 0.050377 | 0.342487 |

final 的 S−L：primary `+0.004188`，natural `+0.003053`，all `+0.003016`，ordinary `+0.002706`，
tail `+0.012037`；与 best-monitor 同向，且 primary 更差。不是 checkpoint 抽签。

### 2.4 训练曲线（Monitor）也同向

| 臂 | seed | best monitor | W20 | T100 |
|---|---:|---:|---:|---:|
| L | 0 | 0.123151 | 0.123819 | 0.123834 |
| L | 1 | 0.127772 | 0.128421 | 0.128481 |
| S | 0 | 0.127512 | 0.127915 | 0.127937 |
| S | 1 | 0.128066 | 0.128507 | 0.128528 |
| P | 0 | 0.127475 | 0.127738 | 0.127798 |
| P | 1 | 0.124311 | 0.124743 | 0.124765 |

Monitor 上 S 也比 L 差（seed mean `+0.00233`），P 与 S 接近（S−P `+0.00190`）。
W20/T100 与 best-monitor 一致，排除“选点抽签”。P seed1 的 Monitor 较好（0.1243）但 Locked
primary 仍为 0.12225，说明 Monitor 的优势没有转化为未见结构优势。

---

## 3. 预注册门控

### 3.1 S−L

| 门 | 结果 | 数值 |
|---|---|---|
| primary ≤ −0.015，两 seed 同向，CI 上界 <0 | **FAIL** | +0.003364；两 seed 同为正；CI 上界 +0.010948 |
| natural overall ≤ −0.003，两 seed 同向，CI <0 | **FAIL** | +0.002805；两 seed 同为正；CI 上界 +0.005729 |
| ordinary_seen max ≤ +0.002 | **FAIL** | +0.004192 |
| seed stability（同号 + spread ≤0.015） | PASS | 同号；spread 0.006493 |
| drop-top-1/10 ≤ −0.010 且保号 | **FAIL** | +0.004553 / +0.001278 |
| key-cluster CI 上界 <0 且保号 | **FAIL** | [−0.004120, +0.011064] |
| tail max ≤ +0.010 | **FAIL** | +0.023226 |
| ordinary_label max ≤ +0.005 | PASS | +0.002434 |

### 3.2 S−P

| 门 | 结果 | 数值 |
|---|---|---|
| primary ≤ −0.015，两 seed 同向，CI <0 | **FAIL** | +0.001502；seed 符号分裂；CI 上界 +0.007498 |
| natural overall ≤ −0.003，两 seed 同向，CI <0 | **FAIL** | −0.000227；均值未达 −0.003；CI 上界 +0.002027 |
| ordinary_seen max ≤ +0.002 | PASS | +0.000536 |
| seed stability | **FAIL** | 符号分裂（+0.003502 / −0.000498） |
| drop-top-1/10 | **FAIL** | +0.000140 / −0.000409 |
| key-cluster CI 上界 <0 | **FAIL** | [−0.004119, +0.007460] |
| tail max ≤ +0.010 | PASS（贴边） | +0.009961 |
| ordinary_label max ≤ +0.005 | PASS | +0.000686 |

### 3.3 判定树

预注册规则先检查安全门：S−L 的 `ordinary_seen` 和 `tail` 安全门失败 → **`STOP_UNSAFE`**。
但即使忽略安全门，S−L 的 primary 与 overall 也都是**正号**（变差），S−P 是 wash + 符号分裂；
因此不存在任何“改一个门就能通过”的解释。`near_miss` 为 false（primary 不在 (−0.015,−0.010]）。
不触发 Phase 2；不调 alpha 分母、不换支持度定义、不加层/宽/参数、不换优化器、不加 seed。

---

## 4. 解释：为什么失败

### 4.1 不是“位置对应”问题

S−P 的 primary 是 `+0.001502`，seed 0 为 `+0.003502`、seed 1 为 `−0.000498`，符号分裂；
natural 是 `−0.000227`，CI 跨 0。真实支持度与键位置的对应**没有**比同分子内置换更好。
P 臂保留了每分子 alpha 多重集，但 P 也没有超过 L（L−P primary `−0.001862`，natural
`−0.003032`）。所以“支持度分布本身”也不足以带来收益：无论真实还是置换，收缩都略有害。

### 4.2 收缩本身有害：4.6% 的均场分量已经足够

alpha 在 65% 的键上约为 0.954，即 S 对大多数高支持键也注入了约 4.6% 的均场分量。
DCC 的 C 臂（把 local composer 输入**全部**换成同一均场统计）在历史 valid 上比 L 差
`0.00383`；本轮 S 在 primary 上比 L 差 `0.00336`，量级几乎相同。这说明：

- DCC-L 的局部 incidence 统计即使在低支持/未见环境也有价值；把它部分替换成图级均场，
  即使只有 5%，也足以损失这份价值；
- 在 s=0 的键上（primary 的 19.15%）alpha=0，完全走均场——这正是最需要“局部是否可靠”
  的地方，但结果没有变好，反而更差；
- 因此“训练支持度低 → 局部统计不可靠 → 应回退到均场”这个前提，在本模型/本数据上不成立。

这不是说支持度没有信息；而是说**用均场作为回退目标**是错误的方向。DCC-L 的 local 统计
在未见环境仍然优于 group mean，这与 DCC C 臂、以及 bottleneck audit 的“方差/异方差而非
均值漂移”诊断一致。

### 4.3 训练更好、Locked 更差：容量不是瓶颈

`train@best`：L `0.015415 / 0.007030`，S `0.010420 / 0.004329`，P `0.004012 / 0.007921`。
S 在训练集上拟合得**更好**，但 Locked primary/overall/ordinary/tail 全部更差。final checkpoint
的 train MAE 都接近 0（L 0.00044/0.00050，S 0.00047/0.00053），说明模型有足够容量记住训练集；
差异不在“拟合能力”，而在输入统计量把函数推向了不泛化的方向。这与
`zinc_support_weighted_objective_v1` 的结论一致：低支持环境的问题是 y 的不可辨识性/方差，
不是梯度注意力或拟合容量；但本轮进一步说明“把局部统计往均场收缩”不是正确的方差对策。

### 4.4 没有修好未见结构，也没有只修人工环尾

- primary（无人工环惩罚、含 Fit 未见 k2）：`+0.003364`（变差）；
- unseen_all：`+0.004834`（变差）；
- tail：`+0.016564`（变差）；penalty_all：`+0.015889`（变差）。

三种可能的“局部成功”都不存在：不是“整体没变但 primary 变好”，不是“只修了环尾”，也不是
“primary 变好但被 tail 抵消”。S 在 primary、unseen、tail、penalty 上全部变差。

### 4.5 与先例的关系

- **DCC C 臂**：全量均场替换比 L 差；本轮逐键、部分、连续替换仍然差。这不是 C 臂的重复，
  而是对“均场回退目标”的更强否定：即使只替换低支持键、且保留高支持键的局部统计，净效应
  仍是负的。
- **SWO**：标签加权让 train 更好、valid 更差；本轮输入收缩也让 train 更好、Locked 更差。
  两者共同指向“低支持环境是方差/不可辨识性问题”，但本轮证明“往均场收缩”不是解。
- **frozen patch sharing / donor**：稀有内容的 donor/均值替换没有可用信号；本轮的支持度
  条件均值回退是同一类失败的另一个实现。
- **RBO v0**：新局部构件没有修复未见环境；本轮说明“保留原构件、只改混合系数”也没有。
- **hierarchical backoff**：那是离散词表级 0/1 回退 + 线性模型，且在弱模型上为正；
  本轮是连续逐键收缩 + 端到端 DCC-L，数学对象不同，不能把它的正结果当作本轮的先验。

---

## 5. 最终报告必须回答的 5 个问题

### 1) 支持度条件混合是否比相同参数的原始 local composer 有实用收益？

**没有。** S−L primary `+0.003364`（两 seed 同为正），natural overall `+0.002805`（两 seed 同为正），
all `+0.002767`，ordinary `+0.001938`，tail `+0.016564`，penalty `+0.015889`。没有一个目标组
变好；预注册目标 `≤ −0.015` 不仅未达到，方向相反。95% CI 跨 0 只说明“没有显著变差”，
不构成“没有显著变好”的借口——点估计和两个 seed 都指向变差。

### 2) 收益是否来自支持度与具体局部结构位置的匹配，而不是支持度分布本身？

**没有收益可归因。** S−P primary `+0.001502`（符号分裂），natural `−0.000227`（CI 跨 0）。
真实支持度位置没有超过同分子内置换；而 P 本身也没有超过 L（L−P primary `−0.001862`）。
因此既没有“位置匹配”收益，也没有“分布本身”收益。

### 3) 是否真的修复未见结构，而不只修复人工长环尾部？

**两者都没有修复。** primary（无惩罚、含未见 k2）`+0.003364`；unseen_all `+0.004834`；
tail `+0.016564`。未见结构变差，人工环尾也变差。不存在“只修尾部”的局部成功。

### 4) 能否达到或稳定超过当前 DCC-L？离单模型 test 0.05x 目标还有多远？

**不能。** L 在 primary（0.119006 vs S 0.122370）、natural（0.084804 vs 0.087609）、all
（0.083498 vs 0.086265）、ordinary、unseen、tail、penalty 上都优于 S；S 没有在任何目标组
稳定超过 L。internal L 的 Locked MAE 为 `0.0835`；历史 DCC-L 单模型 test 为 `0.066267`
（外部参照，本轮未读 test，不可直接比较）。S 没有缩短这个距离，Phase 2 未解锁。
距 0.05x 目标：**没有可报告的进展**；本轮不读 test，因此不能给出新的 test 数字。

### 5) 若失败，该机制是否值得彻底关闭？

**是。** 预注册停止规则触发（`STOP_UNSAFE`，且 primary/overall 同向失败）；不调 alpha 分母 4、
不换 k2 定义/count unit/min-of-endpoints、不加层/宽/参数、不换优化器/调度、不加 seed、不跑
Phase 2、不写 BOSI。该机制应加入停止清单：DCC C（均场替换）、SWO（标签加权）、patch sharing
（donor/均值替换）、RBO v0（新局部构件）之后，SCSB 是“连续支持度条件均值回退”的否定结果。
未来若继续，必须是**不同的机制**（例如改变监督/可辨识性，而不是把局部统计往均场拉），
而不是本机制的参数救援。

---

## 6. 预算、复现与负结果保留

### 6.1 GPU

| 项 | run id | pool | GPU·h |
|---|---|---|---:|
| smoke S s0（64 updates） | `scb-smoke-S-s0-20261027-153609-e08ec7ef` | res-gpu0 | 0.0020 |
| S s0 | `scb-S-s0-20261027-153742-3327b9c3` | res-gpu0 | 0.3734 |
| P s0 | `scb-P-s0-20261027-153751-0c8e476a` | res-gpu1 | 0.4118 |
| S s1 | `scb-S-s1-20261027-160618-2b38e805` | res-gpu0 | 0.3606 |
| P s1 | `scb-P-s1-20261027-160625-3e21c3e2` | res-gpu1 | 0.4108 |
| L s0/s1 | 冻结 DCB checkpoint | — | 0 |
| **合计** | | | **1.5585** |

软预算 3 h，硬上限 4 h；剩余 1.4415 h。没有 Phase 2，没有额外 seed/sweep。
运行 regime：`res` / `a100-2` / driver 550.163.01 / torch 2.5.1+cu124 / 2×A100-40GB；
四个 run 同 host/node/driver/torch。

### 6.2 读取纪律

- official test：未构造、未读取、未评分；所有 JSON 均写 `official_test_loaded: false`。
- official valid：Phase 1 未读取；所有 Phase 1 产物写 `official_valid_loaded: false`。
- Locked：四 run 全部完成后，`scb_score.py` 本地运行一次；sentinel
  `analysis/locked_read.json` 记录 `locked_rows_sha256` 与 6 个 score。
  第一次评分尝试在读取 L final checkpoint 时因 `KeyError: monitor_mae` 失败；它没有写出任何
  score、没有创建 sentinel，也没有用于任何选择；第二次运行成功。`run_records.json` 如实记录。
- 所有比较臂同一 Fit8k；L 为同 Fit/同初始化/同批序/同 47,400 updates 的冻结 checkpoint；
  Phase 0 已用训练循环 curve[0] 与预测复现验证。

### 6.3 保留的负结果

- `analysis/cpu_checks.json` — Phase 0 13/13。
- `analysis/phase1_scores.json`、`analysis/score_{L,S,P}_s{0,1}.json` — 六次评分与逐组 MAE。
- `analysis/phase1_decision.json` — 门控、bootstrap、drop-top-k、cluster、safety。
- `analysis/locked_read.json` — 一次性读取 sentinel。
- `analysis/run_records.json` — 本文件 §6.1 的机器可读版本。
- `analysis/predictions/*.json` + `manifest.json` — 12 个逐行预测（best/final，L/S/P × 2 seed），
  可直接复算 bootstrap；checkpoint hash 见 `phase1_scores.json`。
- `runs/{L,S,P}_fit8k_s{0,1}/` — checkpoints、predictions、train.json（gitignored，保留在本地/远端）。
- `REPORT.md` — 本报告；`README.md` 状态；`PROTOCOL.md` 顶部结果横幅（预注册正文不改）。

### 6.4 复现命令

```bash
# Phase 0 (CPU)
uv run python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_checks.py

# Phase 1 (after rr deploy res)
rr run res scb-S-s0 --pool res-gpu0 --gpus 1 --time 2:00:00 --result tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/runs -- \
  python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_train.py --arm S --fit fit8k --seed 0 --device cuda:0
# ... S-s1 / P-s0 / P-s1 likewise (see analysis/run_records.json for run ids)

# one frozen Locked read
uv run python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_score.py --device cpu

# analysis + pre-registered decision
uv run python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_analyze.py
```

---

## 7. 一句话总结

> 在 DCC-L 上，用训练支持度把每条键的局部 incidence 统计连续收缩到同一模型的图级均场统计，
> 不但没有修复 Fit 未见的 radius-2 环境，反而让 primary（+0.00336）、整体（+0.00281）、
> 普通分子（+0.00194）和人工长环尾部（+0.01656）全部变差；真实支持度位置也不比同分子内
> 置换更好（S−P +0.00150，符号分裂）。训练集拟合更好而 Locked 更差，说明问题不是容量/注意力，
> 而是“往均场回退”这个方向本身。机制关闭，Phase 2 不启动，official test 未读。
