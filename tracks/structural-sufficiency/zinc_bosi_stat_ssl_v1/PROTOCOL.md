# BOSI Statistics-Only SSL v1 — 预注册协议

轮次 `zinc_bosi_stat_ssl_v1`。配置 `configs/bosi_stat_ssl_v1.json`。
**本文件在第一次 GPU 作业之前冻结。** 之后只允许追加"偏差"记录，不允许修改判决规则、
评估集、预算或对照。

设计基线：`calendar0917/research@21e01a7`，即

* `zinc_bosi_architecture_v0/REPORT.md`（关系分配 `NO_GO`；BOSI-B Locked s0 `0.082449` /
  s1 `0.081491`，Monitor s0 `0.124505` / s1 `0.122400`）；
* `zinc_bosi_partner_content_v1/REPORT.md`（`PARTNER_CONTENT_SUPPORTED`：去掉其他邻键的
  learned 状态使 Locked 变差 `+0.01239`，两 seed 同向、CI 不跨零、非尾部）；
* `zinc_bosi_partner_content_v1/NEXT_SSL_HYPOTHESIS.md`（只写设计，未启动）；
* `structure-contribution/results/bjoc_masked_pretraining_v1/REPORT.md`
  （`SELF_SUPERVISED_PRETRAINING_USEFUL`，`P−C = −0.0077`）；
* `structure-contribution/results/bjoc_task_adaptive_ssl_v1/REPORT.md`
  （`TASSL_NO_GAIN`：训练期持续辅助 loss 无效）。

---

## 1. 要回答的唯一问题

BOSI-B 的**邻键 learned 内容**已被证明有显著边际预测价值（`+0.0124`，两 seed 稳定），
而生产这条内容的 `half_encoder` → `M` → `side` → `compose` 链条**没有任何预训练信号**
（BOSI 只把 atom/bond encoder 用 train-only masked-SSL 初始化过）。

> **只预训练 `half_encoder` / `side` / `compose` 这三个模块，能否在不增加分子标签、
> 不改变推理架构、不改变监督 recipe 的前提下提高泛化？**

不是 attention 实验；不新增 junction；不做架构搜索；不做训练期持续 SSL
（`TASSL_NO_GAIN` 已否定该形态）；不搜索 mask 比例 / λ / 辅助头 / seed / 轮数。

## 2. 预训练设计（本轮唯一的"新东西"）

### 2.1 数据与遮蔽

* 只用 **Fit8k 的 8,000 个分子**的原始结构（`atom_type`、`bond_type`、`bond_index`）。
  **不读取任何性质标签**：预训练进程拿到的是 `s_common.structural_cache(cache)`，
  一个只含结构键的字典；`cache["labels"]` 在该进程里不可达（G1 用守卫字典 + 投毒
  标签位同实验证明）。
* 遮蔽比例固定 **15% 原子 / 15% 物理键**，每个分子每类至少 1 个，`max(1, round(0.15n))`，
  无放回抽样；RNG 为 `numpy.random.SeedSequence([seed, molecule_id, visit_index])`，
  `visit_index` = 预训练 epoch 计数（1..120），与数据 shuffle 的
  `torch.Generator(seed+202)` 完全独立。这与 `bjoc_masked_pretraining_v1` 的冻结
  生成器逐项同构。
* mask token：原子 **27**（Fit8k 原始 atom code 取值范围 0..20，槽位 21..27 未被使用）、
  键 **0**（原始 bond code 取值 1..3，槽位 0 未被使用）。
* **只改属性码，不删边**：`bond_index` 与邻接关系完全不动，因此 T/Q/R/N 关系、biconnected
  block、closure 计数、opening 结构统计都是 v0 冻结对象的原样（G4 逐字段验证）。
  这一选择事前写死，不做事后挑选（`NEXT_SSL_HYPOTHESIS.md` §3.1 已列出两种方案，
  本轮采用"只改键型、不删边"）。

### 2.2 完整重建（不允许增量补丁）

从遮蔽后的原始图**从头重建所有依赖被遮蔽属性的特征**：

| 张量 | 依赖类型的块 | 重建方式 |
|---|---|---|
| `atom_feat` (108) | `0:28` 自身类型 one-hot | 由遮蔽类型 one-hot |
| | `42:46` incident bond-type 直方图 | 由遮蔽键类型重新累加 |
| | `46:74` radius-1 邻居类型直方图 | 由遮蔽原子类型重新累加 |
| | `74:102` radius-2 原子类型直方图 | 由遮蔽原子类型重新累加 |
| `bond_feat` (7) | `0:4` 自身键类型 one-hot | 由遮蔽键类型 one-hot |
| `global_feat` (33) | `19:23` 键类型占比、`31` distinct_bond_types、`32` distinct_atom_types | 由遮蔽类型重新计算 |

其余 20 个 atom 列（degree / in_ring / min_ring_size / ring bond 数 / radius2 标量统计 /
roles）、3 个 bond 列（is_ring / exactly_one_in_ring / both_in_ring）与 27 个 global 列
是**纯拓扑函数**，逐位复制，并在 G8 里用 5 组任意随机类型重排验证其位不变性。

**等价性证据**（不是"我们觉得很像"）：

* G2：对**全部 8,000 个 Fit8k 分子**，未遮蔽时本轮的快速重建 == 冻结 cache 的三个张量
  （max|Δ| = 0.0），且与上游 `structural_relation_objects_v1.build_mol_objects` 在
  100 个原始图上逐位相同；
* G3：对 900 个 (分子, visit) 遮蔽样本，快速重建 == 遮蔽原始图上的
  `build_mol_objects`（三个张量 max|Δ| = 0.0）；`masked_batch` == 用遮蔽 cache 走
  冻结 `bosi_data.collate` 的参考实现（17 个张量全部 0.0）。

### 2.3 两个 pretext 头（固定，不搜索）

| 头 | 输入 | 目标 | 输出 |
|---|---|---|---|
| **H_atom** | 对被遮蔽原子 `v`：`Σ_{e∈Inc(v)} c_e`（96 维） | `v` 的原始 atom type | 28 类 CE |
| **H_bond** | 对被遮蔽物理键 `e`：`c_e`（96 维） | `e` 的原始 bond type | 4 类 CE（观测支撑 1..3） |

* 两个头都是 `96 → 160 → n_class` 的 SiLU MLP（dropout 0），预训练后**丢弃**。
* `L_SSL = CE_atom + CE_bond`，权重 1.0/1.0，**不做 λ 搜索**。
* **禁止**：用聚合伙伴矩直接预测无法定位的任意伙伴身份（H_atom 预测的是**中心原子**
  `v`，`v` 是完全可定位的对象）；用 `global33` 或图级 reader 作为捷径（G15 把
  `global` 换成 N(0,100²) 随机值、删掉 `global` 键、甚至 `del model.reader` 之后
  pretext logits **逐位不变**）。
* 头的输入只有 `c_e`（G15 静态断言 96/96 维、28/4 类）。

### 2.4 冻结与可训练参数

* `atom_encoder` / `bond_encoder`：加载与历史 BOSI-B 同 seed **相同**的 train-only
  masked-SSL checkpoint（`bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s{seed}.pt`），
  并 `requires_grad_(False)`。
* `reader`：**不参与预训练**（pretext 前向在 `c` 处结束，不计算 `z`/`y`），
  `requires_grad_(False)`；监督训练时保留与历史 BOSI-B 完全相同的初始 reader。
* 只更新 `half_encoder`（42,016）+ `side`（91,488）+ `compose`（78,176）+ 临时头
  （36,192）= **247,872** 参数。G10 验证两个头的梯度都真实到达三个模块，
  atom/bond encoder 与 reader 的梯度恒为 0。
* 预训练起点 = `build_model(norm, "B", seed)`，其 `state_hash` 必须等于历史
  BOSI-B 同 seed 的 `ceeb598f…`（s0）/ `c93ea85b…`（s1）。

### 2.5 预训练配方（固定）

`Adam lr 1e-3 / wd 1e-5 / clip 5 / batch 128 / fp32`，**7,560 updates**
（= 120 epochs × 63 step，与 BJOC masked pretraining 相同），常数学习率，
**last-step state only**（绝不用重建精度选 checkpoint）。pilot 用同一 stream 的前
**1,890 updates**（25%）。

## 3. 预训练 pilot（seed 0，先决条件）

pilot 只做一次，固定 1,890 updates，probe 集是固定的 1,024 个 Fit8k 分子
（`rng(20261027)` 抽样，mask `visit_index = 1000000`，与训练 visit 1..120 不相交）。

记录：masked atom/bond CE（对比 Fit8k 属性边际基线）、目标类别分布与非多数类表现、
pretext loss 到三个目标模块的梯度范数、**保持其他输入不变时替换/置乱伙伴统计后的
pretext 响应**（`M` 置零、`partner_half` 全局置乱）、实际吞吐与 projected total cost。

| `PRETEXT_NO_GO`（任一成立即停） | 阈值 |
|---|---|
| 只学到类别先验（atom） | probe atom CE > 边际 CE − 0.10 nats |
| 只学到类别先验（bond） | probe bond CE > 边际 CE − 0.02 nats |
| 统计通路不参与（atom） | `CE(partner-shuffled) − CE(clean) < 0.02` nats |
| 统计通路不参与（bond） | `CE(partner-shuffled) − CE(clean) < 0.02` nats |

pilot 不通过 → 不启动任何完整预训练或监督训练，判决 `PRETEXT_NO_GO`。

## 4. CPU 泄漏审计（GPU 之前，失败即停）

`code/s_checks.py` → `analysis/cpu_checks.json`，16 门，全过为 `CPU_GO`：

| 门 | 判据 |
|---|---|
| G1 | 数据边界：Fit/Monitor/Locked 互不相交；预训练只看结构键；守卫字典 + 标签投毒后参数逐位相同；三个预训练模块里没有 `["labels"]` / `.labels` |
| G2 | 未遮蔽快速重建 == 冻结特征（全部 8,000 分子，max\|Δ\|=0）+ 上游 builder 交叉验证 |
| G3 | 遮蔽快速重建 == 遮蔽图上 `build_mol_objects`（900 例，逐位）；`masked_batch` == 参考 collate（17 张量） |
| G4 | 类型遮蔽不改变 T/Q/R/N、closure、half_deg、partner 指针、block（900 例 × 9 字段），且与冻结关系缓存一致 |
| G5 | mask token 不在 Fit8k 原始词表内；全 cache 中 0 次出现 |
| G6 | **≥200 次 masked hidden-truth swap**（实做 240）：所有模型输入逐位不变、目标确实改变；batch 级 12 个张量 Δ=0.0 |
| G7 | unmasked 正控制：改一个可见原子类型 → 特征变 8 格、`c` 移动、pretext logit 移动 |
| G8 | 半径统计不可泄漏：被遮蔽原子自身 108 维只含 token；自身行对 10 组不同隐藏真值位同；20/3/27 个纯拓扑列对任意类型重排位同；邻居与 distance-2 原子看到的是 token（588/588、698/698） |
| G9 | 目标可定位：每个 atom target ↔ 一个原子行、每个 bond target ↔ 一个**物理键**行；被遮物理键的两条有向半键同步；端点 bond-type 直方图已重算；batch 里没有 target 键 |
| G10 | 两个头分别反传时 `half_encoder`/`side`/`compose` 梯度都 > 0；atom/bond encoder 与 reader 恒为 0 |
| G11 | 推理模型 == 历史 BOSI-B：473,889 参数、六个模块、28 个 state key、无 head、初始 hash 等于历史 |
| G12 | 重编号 / 键反转 / 键序 / batch 重组不变（pretext 状态逐位，图级输出相对误差 < 1e-5） |
| G13 | mask 流可复现、不同 visit 不同、每类至少 1 个、与 shuffle 流独立 |
| G14 | pretext 前向 == 冻结模型前向（`a,b,x,M,n,s,c` 逐位相同，无 reader） |
| G15 | 无图级捷径：`global` 投毒 / 删除 `global` / 删除 `reader` 后 logits 逐位不变 |
| G16 | 接口/预算：六模块参数、247,872 可训练参数、SIDE_IN 617、READER 417、arm=B |

## 5. 监督训练（条件性）

只训练一个新的模型主体 **`BOSI-SSL` = BOSI-B + 三个模块的 SSL 初始化**。
**不重训** BOSI-B / BOSI-R / BOSI-P / DCC-L。

逐 seed 必须完全相同（G11/S3 校验）：Fit8k / Monitor / Locked 划分、atom/bond 初始
checkpoint、reader 初始状态、`Generator(seed+101)` batch 顺序、**47,400 optimizer updates**、
AdamW 1e-3→1e-5 per-step cosine、warm-up 500、wd 1e-5、clip 5、raw-y L1、
Monitor 严格最小选点。唯一差异是 `half_encoder/side/compose` 的初值。

| 步骤 | 门槛 |
|---|---|
| 先训 `S_s0` | — |
| 买不买 seed 1 | `Monitor(S_s0) ≤ Monitor(B_s0) − 0.004 = 0.1205047393636778`；否则停止，不买 seed 1、不买对照 |
| 买不买算力对照 C | 两 seed 都有 `Monitor(S_s) < Monitor(B_s)`，且 seed-mean 改善 ≥ 0.004 |

### 算力对照 C（只有两 seed 都过 Monitor 门槛才启动）

从**历史 BOSI-B 同 seed 初始化**重新训练，**不用新 SSL**；额外监督计算量按 SSL 实测
消耗**事前冻结**：

```
N_s = round( 预训练(seed s) 的实测 wall seconds / 该 seed 监督训练的实测 seconds/update )
```

前 47,400 updates 使用与 BOSI-B 完全相同的 schedule 与 batch 顺序（因此前 47,400 步
与不加载 SSL 的历史轨迹一致），之后 N_s 步**继续同一 batch-order 流**、学习率固定为
schedule 的终值 1e-5。报告新增更新次数、LR 规则与真实 GPU-hours。
**不把"更多训练"当作天然公平**：C 的作用是把"预训练带来的初始化效应"与"等价算力带来
的优化效应"分开。

## 6. 评分流程（顺序不可逆）

1. `s_freeze.py`：冻结所有新 run 的 checkpoint / Monitor / **Fit** 读数与哈希；
   逐文件 sha256 校验复用对象（v0 freeze、B_s0/B_s1 checkpoint 与 Locked npz、
   official-train cache、关系缓存、预训练 state）；校验 S 与历史 B 的初始状态与首 batch
   行号一致；写 **GPU 账本**并检查 4.0 A100·h 上限。`locked_read.json` 已存在则拒绝。
2. `s_locked.py`：**一次性**读取 Locked，对所有候选的 best_monitor 与 final 打分，
   并在同一次读取内做冻结权重干预（`M` 置零、`partner_half` 置乱）。
   二次调用只校验哈希、不重算。BOSI-B 的 Locked 预测直接复用 v0 冻结的 npz，不重跑。
3. `s_analyze.py`：只用冻结文件产出全部表与判决。

Locked 是**历史已暴露**的内部划分（DCC-L 4 seeds、ESOC、additive-reader、BOSI v0/v1
都读过），本轮读取**不是新的盲测确认**。

## 7. 指标与判决（预注册）

`Δ = MAE(BOSI-SSL) − MAE(BOSI-B)`，**负 = SSL 更好**。n=1,000 Locked 分子，
配对 bootstrap 10,000 次（固定分子重采样，只描述分子抽样、不含训练随机性）。

分组（全部是事前冻结的 label-free 定义）：overall、ordinary（`y>−3`）、tail（`y≤−3`）、
low_support、any_unseen、all_supported、has_T_or_Q、alloc_defined_top_quartile、
rings_ge4；另有去配对最差 10 个分子、RMSE、final checkpoint。

| 判决 | 条件 |
|---|---|
| `SSL_PROMISING` | 两 seed `Δ<0`；seed-mean `Δ≤−0.005`；ordinary 两 seed 皆 <0；去 top-10 两 seed 皆 <0；any-unseen 未恶化（seed-mean ≤0 且单 seed ≤+0.002）；**且**相对 C 有独立收益（seed-mean `MAE(SSL)−MAE(C) ≤ −0.003`、两 seed 皆 <0） |
| `GAIN_FROM_EXTRA_COMPUTE_ONLY` | 上述 SSL 条件全成立，但对 C 无独立收益 |
| `NO_GAIN` | 两 seed 条件不成立；或只跑了 seed 0（seed-0 Monitor 门槛没过） |
| `INCONCLUSIVE` | seed 方向相反但均值为负；或缺少算力对照 C |
| `PRETEXT_NO_GO` | pilot 条件不成立 |

报告必须明确区分：**pretext 本身不可识别** / **泄漏或计算路径错误** /
**pretext 成功但未改善性质泛化** / **改善只来自额外计算** / **真实可重复的 SSL 增益**。

## 8. 预算

`res`（a100-2，A100-SXM4-40GB，与历史 BOSI v0/v1 同一执行 regime），
**上限 4.0 A100·h**，按 1 GPU / job × wall-clock 小时计，**失败作业计入**。
预计：pilot ≈0.05 + 预训练 2×≈0.2 + 监督 2×≈0.45 + 对照 2×≈0.5 ≈ 2.4 h。
若实测将超限 → 停止并记录，**不缩短正式更新次数**。

## 9. 禁止清单

official valid/test 任何形式的访问；新分子 / 外部数据 / 新标签分量 / teacher；
attention、learned junction、递归 MP、架构搜索、新查表；重训 B/R/P/DCC-L；
mask 比例 / λ / 辅助头 / 额外 seed / 额外 epoch 的搜索；用重建精度选 SSL checkpoint；
在冻结之前读 Locked；事后修改判决规则、评估集或预算。
