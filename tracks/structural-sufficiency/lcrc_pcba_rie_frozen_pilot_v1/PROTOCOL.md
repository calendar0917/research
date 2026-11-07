# PROTOCOL — LCRC-RIE 冻结结构关系证据快速可行性实验（`lcrc-pcba-rie-frozen-pilot-v1`）

Round: `lcrc-pcba-rie-frozen-pilot-v1`
类型：一次性 **frozen probe**（冻结 backbone + 极小 task-specific residual head 的短程配对比较），
不是完整新模型训练，不是架构搜索。
数据集：OGB `ogbg-molpcba`，官方 Train / Validation；**官方 Test 从未读取、从未评分、从未索引**。
主指标：官方 Validation macro AP（OGB `Evaluator("ogbg-molpcba")`），固定 **epoch 8**。

本文件在第一次正式 GPU 运行前冻结。运行期间不修改 motif 家族、距离分桶、特征宽度、
head 宽度、训练长度、学习率、稀疏系数、评价点、判定阈值或资源上限。任何修改都必须先
停轮、显式重写本协议并重新提交。

---

## 0. 唯一研究假设与本轮问题

已有事实（不重新争论）：

* 原始 LCRC-A，seed 0，epoch 40，Validation macro AP **0.2463560956**；参数 **287,644**。
* 后续 attention / 描述符 / branch token / masked atom-bond SSL 均未带来明确的广泛提升。
* MolPCBA 局部化学变化路线显示部分 assay 有可预测局部变化，但无监督 KSVD 压缩与跨任务迁移效果不佳。

本轮只回答一个问题：

> 在保留原始 S1/S2 局部表示与原 A 全部预测能力的前提下，**显式 motif–motif 拓扑关系**
> 是否能提供超出“相同模型容量 + 相同 motif 组成信息”的 MolPCBA Validation AP 增量？

判定标准不是“新模块有没有用”，而是：

1. REAL 是否优于原 A（`D_base`）；
2. REAL 是否优于等容量、同初始化、同监督配方的 SHUFFLE 对照（`D_ctrl`）；
3. 该优势是否在正例充分的 assay 上同样存在（`D_ge20` + 分层改善任务数）。

三项方向一致才构成继续研究 RIE 的理由。本轮**无论结果多好都不启动**完整 LCRC-RIE 联合训练。

---

## 1. 冻结的源 checkpoint（必须是真正的原 A）

| 项 | 值 |
|---|---|
| 路径 | `tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A/checkpoint_epoch40.pt` |
| SHA256 | `f110935a6055f3c087f9d0b40580506a512cdeb09f6af67a56bde787cc93a437` |
| 来源 round / arm | `lcrc-pcba-no-mp-sprint-v1` / **A**（`C1 + attention_pool`） |
| seed / epoch / global_step | 0 / 40 / 109520 |
| 参数 | 287,644（base 245,184 + new 42,460） |
| state_dict 张量数 | 57 |
| 存储的 Validation AP | `0.24635609609821457` |
| 存储的 Validation BCE | `0.04577261283603607` |
| 训练时 git commit | `b1b917dc4f72d9815b8f00813a11d95db3868298` |
| 运行登记表 | `runs/phaseC_seed0_A/run.json`（`fixed_epoch_reports["40"]["ap"]`） |

不作为候选的模型（本轮明确禁止）：SSL、A0 重训、A1/A2/A3、G、V、任何“最接近的 checkpoint”。

加载与复现门槛（任一失败 → `SOURCE_CHECKPOINT_UNAVAILABLE_OR_MISMATCH`，停止本轮，
不得改用其他 checkpoint，不得给科学阴性结论）：

| 门槛 | 判据 |
|---|---|
| 文件身份 | SHA256 等于上表冻结值 |
| payload 身份 | schema / round_id / arm=A / variant=`C1 + attention_pool` / seed=0 / epoch=40 / global_step=109520 / n_outputs=128 |
| 参数计数 | 总参数 287,644；`model_audit` base 245,184 / new 42,460 |
| 权重加载 | `load_state_dict(..., strict=True)`，与 checkpoint 逐 tensor **bitwise** 相等 |
| 冻结 | `model.eval()`，全部参数 `requires_grad=False`，无 `.grad` |
| Validation AP | 复现值与 `AP_A = 0.24635609609821457` 的绝对差 `<= 1e-4` |
| 说明 | 冻结前向使用 CUDA `index_add_`（atomic）归约，跨调用 logits 不可 bitwise 复现；因此 AP 判据是容差判据，实测值一律记录 |

---

## 2. Motif 定义（固定 8 类化学家族，不得新增）

| id | family | 定义来源 | 说明 |
|---:|---|---|---|
| 0 | Donor | RDKit `ChemicalFeatures` + `BaseFeatures.fdef` | `GetFamily() == "Donor"` |
| 1 | Acceptor | 同上 | |
| 2 | Aromatic | 同上 | |
| 3 | PosIonizable | 同上 | |
| 4 | NegIonizable | 同上 | |
| 5 | Hydrophobe | 同上 | `LumpedHydrophobe`、`ZnBinder` **不属于**任何家族，直接忽略 |
| 6 | Carbonyl | SMARTS `[CX3]=[OX1]` | 完整匹配的 C 与 O 两个原子 |
| 7 | Halogen | SMARTS `[F,Cl,Br,I]` | 命中的单原子 |

* 记录 `BaseFeatures.fdef` 的 SHA256（`766f9790513f8f94c67ec6d1c8b6d5bede59b81f6da8c2081f57e49fb28d9784`）
  与 RDKit 版本。
* 每个匹配产生一个 motif：`motif = {family_id, sorted_unique_atom_indices}`。
  同一分子中相同 `(family_id, atom_indices)` 的重复匹配只保留一次；不同 family 可以共享原子。
* 使用真实 atom indices，不挑选代表原子；motif 列表按 `(family_id, atom_indices)` 规范排序，
  使聚合顺序与输入顺序无关。
* **不使用**历史 8192 桶 hash 词表。
* 若某分子 RDKit 解析失败：保留该官方行，REAL 与 SHUFFLE 关系特征都严格为零，
  原 A logits 保持正常；记录官方 row id 与失败原因；不删除任何监督样本、不跳过该行。
* 不因覆盖率不足临时增加/删除 motif 类型。

---

## 3. Motif 表示

```
z_i = mean(h_v for v in M_i)            # 64 维，来自冻结原 A 的原子状态 h
```

`h` 是原 A 前向 `aux["h"]`（当前已训练得到的 64 维原子局部上下文）。
不对 `z_i` 追加任何 GNN / Transformer / 上下文传播 / 消息传递。

---

## 4. 关系定义（固定）

对每对无序 motif `(M_i, M_j), i < j`：

```
d(M_i, M_j) = min shortest_path_distance(u, v),  u in M_i, v in M_j
```

* 距离定义在**真实物理键图**上（与模型输入的 `bond_index` 同一张图，且与 RDKit 图逐键一致）。
* 只保留同一连通分量内有限的最短路径距离；不同连通分量之间**直接排除**，不视为“超长距离”。
* `d < 3` 的 pair 丢弃（不重复改造原 S1/S2 的局部职能）。
* 三个固定区间：bin0 = `3–4`，bin1 = `5–6`，bin2 = `>= 7`。

pair 特征（对 motif 次序交换必须不变）：

```
pair_feature(i,j) = concat(z_i * z_j,            # 64
                           abs(z_i - z_j),       # 64
                           family_pair_onehot,   # 36  (8*9/2 无序配对)
                           1.0)                  # 1
=> 165 维
```

图级聚合（每图、每 bin 独立）：

```
S_b(G) = sum_{pair in bin b} pair_feature(i,j) / sqrt(1 + n_pairs_in_bin_b)
```

* 某 bin 无合法 pair → 该 bin 严格为全零向量。
* 最终 `S(G) = [S_0, S_1, S_2]`，`shape = [3, 165]`。
* 一次性、显式、无方向的关系聚合；没有任何 learned state propagation。

---

## 5. SHUFFLE 对照（只破坏“哪一对对应哪个距离”）

对每个分子：

1. 提取与 REAL 完全相同的 motif 集合；
2. 提取与 REAL 完全相同的合法 motif pairs；
3. 每个 pair 的 `pair_feature(i,j)` 保持不变；
4. 收集 REAL 中每个 pair 的距离 bin 列表；
5. 用固定 RNG 对该 bin 列表做**分子内排列**；
6. 把排列后的 bin 重新分配给同一批 pairs；
7. 使用与 REAL 完全相同的聚合公式。

RNG：

```
tag    = "rie-distance-shuffle-v1"
entropy= int(SHA256("rie-distance-shuffle-v1|<official_row_id>"), 16)
seed   = 0   (作为 numpy SeedSequence 的 spawn_key)
rng    = numpy.random.Generator(PCG64(SeedSequence(entropy=entropy, spawn_key=(0,))))
```

排列只依赖官方 row id，**不使用** assay 标签、Validation AP 或任何模型预测。

SHUFFLE 保留：同一分子 motif 家族组成、相同原 A 原子表示、完全相同的 motif pairs、
相同 pair feature、每个 bin 内 pair 数量、相同输入宽度与归一化公式。
它只破坏具体 pair 与真实距离之间的对应关系。

若某分子所有合法 pair 落在同一个 bin，SHUFFLE 与 REAL 相同；保留该行，不强行改变其特征。

必须报告（官方 Validation，且与 Train 一并给出）：

```
fraction_with_eligible_pairs
fraction_with_multiple_distance_bins
fraction_REAL_ne_SHUFFLE
per-bin pair count distributions
```

若 Validation 中 `REAL != SHUFFLE` 的分子比例 `< 15%` → 标记
`INSUFFICIENT_RELATION_INTERVENTION`，**在正式训练前停止**，不允许临时修改区间或类别。

---

## 6. 特征提取与缓存

对官方 Train / Validation：

| split | 分子数 |
|---|---:|
| Train | 350,343 |
| Valid | 43,793 |

* 只运行一次原 A forward（`model.eval()`、`requires_grad_(False)`、`torch.inference_mode()`），
  同时得到 `logits` 与 `aux["h"]`。
* 保存物：原始 128 维 logits、图级 REAL/SHUFFLE 关系特征 `[N,3,165]`、官方 row id、标签副本。
  64 维原子状态**不落盘**：按 batch 提取后立即聚合成图级关系特征。
* 所有标签严格沿用原 OGB 监督数据与 NaN mask；不重采样、不加权、不改 loss。
* 官方 Test 不进入特征构造、不进入预测、不进入评分。

缓存文件（`runs/features/`，git-ignored）：

```
train_base_logits.npy        [350343,128]
valid_base_logits.npy        [43793,128]
train_relation.npy           [350343,3,165]      # 归一化后（模型输入）
valid_relation.npy           [43793,3,165]
train_relation_shuffle.npy   [350343,3,165]
valid_relation_shuffle.npy   [43793,3,165]
train_relation_raw.npy       [350343,3,165]      # 归一化前（审计用）
train_relation_shuffle_raw.npy
valid_relation_raw.npy
valid_relation_shuffle_raw.npy
train_labels.npy / valid_labels.npy
train_rows.npy / valid_rows.npy
relation_rms_scale.npy       [495]
```

### RMS 归一化

仅由 **REAL Train** 特征计算，按展平后的 `(bin, feature)` 列（495 列）：

```
rms_k = sqrt(mean_{train REAL rows} S_k^2)
if rms_k < 1e-4: rms_k = 1.0
```

* REAL 与 SHUFFLE 共用同一 scale；Validation 使用 Train scale。
* 不做均值中心化，保证“没有关系 pair”仍对应严格零向量。
* 记录全部零方差 / 被 floor 的列、非零列、NaN/Inf 检查、Train/Valid 行指纹与结构特征哈希。

---

## 7. 两组完全相同的预测模型（8,784 可训练参数）

```
Input: S(G) shape [3,165]

shared_relation_projection : Linear(165, 16, bias=False)
relation_evidence          : [3,16] -> flatten -> [48]
task_weight                : Linear(48, 128, bias=False)
final_logits               : frozen_A_logits + task_weight(relation_evidence)
```

即 `logits_t = base_logits_t + sum_b sum_k W[t,b,k] * R[b,k]`，其中 `R[b] = W_shared @ S[b]`。

新增参数：`165*16 + 48*128 = 8,784`（原 A 的 3.05%）。

初始化：

* `shared_relation_projection`：PyTorch `nn.Linear` 默认初始化
  （`kaiming_uniform_(a=sqrt(5))` ≡ `U(-1/sqrt(165), 1/sqrt(165))`），
  由显式 `torch.Generator().manual_seed(0)` 生成，不推进全局 RNG；
* `task_weight.weight`：精确全零；
* REAL 与 SHUFFLE 两组的初始 state **bitwise 相同**（以 state SHA256 记录）；
* 两组共享完全相同的原 A logits；
* Epoch 0 两组 logits **严格等于**原 A logits（零残差，逐位）。

训练时只有这 8,784 个参数 `requires_grad=True`；冻结 A 完全不参与计算图（只以常量 logits 出现）。
本轮不联合训练 A、不修改原模型、不迁移任何预训练头权重。

---

## 8. 监督训练协议（两组完全相同）

| 项 | 值 |
|---|---|
| seed | 0 |
| epochs | 8 |
| batch size | 1024（343 步/epoch，2,744 步） |
| optimizer | AdamW |
| learning rate | `3e-4` constant |
| weight decay | `1e-3` |
| betas | `(0.9, 0.999)` |
| eps | `1e-8` |
| gradient clip | `5.0` |
| precision | FP32 |
| dropout | 0 |
| scheduler | none |
| early stopping | **false** |
| loss | `masked BCEWithLogitsLoss(observed entries)` + `1e-5 * mean(abs(task_weight.weight))` |
| L1 范围 | 只约束 128×48 的 task weight；不约束 shared projection，不约束冻结 A |
| 行顺序 | `torch.Generator().manual_seed(0)` + 每 epoch `torch.randperm(350343)`；两组共享同一 plan（记录 sha256） |
| 评价点 | epoch `0, 2, 4, 8` |
| 主比较 | **仅 epoch 8**；即使 epoch 4 更好也不提前停止、不替换 |

两组均从同一份新初始化开始，不让一组接着另一组继续训练。不按 Validation 结果修改任何系数。

---

## 9. 预检正确性门槛（正式训练前必须全部通过）

1. 历史 A checkpoint 正确，Validation AP 复现（§1）。
2. 原 A 全部权重冻结，提取前后参数指纹相等，且无 `.grad`。
3. 真实物理图距离与独立 BFS 参考一致（`scipy.sparse.csgraph.shortest_path`）。
4. Motif 顺序变化不改变图级关系。
5. 原子重编号不改变图级关系。
6. 交换 motif pair 两端不改变 pair feature。
7. 相同 motif 重复匹配正确去重。
8. 不同连通分量之间不产生虚假长距离关系。
9. 无可用 motif pair 的图得到全零关系特征。
10. SHUFFLE 对每个分子保留 motif 集合、pair 集合、每 bin pair 数量。
11. 对受干预分子，REAL 与 SHUFFLE 关系张量确实不同。
12. REAL / SHUFFLE 缓存与官方 row id 严格对应。
13. 两组 head 参数量都为 8,784。
14. Epoch 0 两组 logits 等于原 A logits。
15. 两组初始 head state、optimizer 配方、RNG、训练样本顺序一致。
16. 原 A 无梯度；新增模块梯度有限且非零。
17. Test 从不加载、预测、评分或用于调参。
18. 官方 Valid AP 与原 `pc_metrics` / OGB Evaluator 一致。
19. CPU 与 GPU smoke 无非有限值。
20. 保存并重载 head 后预测等价。

实现层面的容差（事前声明，理由是 float32 求和顺序）：

* 门槛 4/7（motif 顺序、去重）要求 **逐位相等**（聚合前已规范排序）。
* 门槛 5（原子重编号）允许 `<= 1e-5` 绝对差（实测约 `1e-6`）。
* 门槛 6 要求 **逐位相等**。
* 门槛 12 的端到端抽查（单 batch 重算）允许 `<= 1e-5`（关系特征）/`<= 1e-4`（logits），
  原因是冻结前向的 CUDA atomic 归约。

---

## 10. 固定判定规则

```
AP_A       = frozen original A Validation AP (replayed)
AP_REAL    = REAL epoch8 Validation macro AP
AP_SHUFFLE = SHUFFLE epoch8 Validation macro AP

D_base = AP_REAL - AP_A
D_ctrl = AP_REAL - AP_SHUFFLE
D_ge20 = mean over tasks with >=20 Validation positives of (AP_REAL - AP_A)
```

### PROMISING_RELATION_SIGNAL（需同时满足）

```
D_base >= +0.005
D_ctrl >= +0.003
D_ge20 >= +0.003
improved_ge20_tasks >= 60
```

该标签只表示值得进行完整联合训练，不表示研究假设已确认。

### WEAK_OR_MIXED_SIGNAL

`D_base >= +0.002` 但未满足全部 PROMISING 条件。此时不自行启动完整实验，并报告缺少哪一项证据。

### NO_RELATION_SIGNAL

`D_base < +0.002`，或 REAL 对 SHUFFLE 没有明确优势。

### 无效状态（不得强行给科学结论）

* `SOURCE_CHECKPOINT_UNAVAILABLE_OR_MISMATCH`
* `INSUFFICIENT_RELATION_INTERVENTION`（Validation `REAL != SHUFFLE` 比例 `< 15%`）
* `INCOMPLETE_RESOURCE`（硬预算内无法完成）
* `INCOMPLETE`（任一预检/完整性门槛失败）

对 `D_base` 很大但 `D_ctrl` 很小的结果，报告必须明确写出：

> 新增的小型监督 readout 可能有用，但没有证据表明收益来自真实 motif 距离关系。

不得用“模型整体提升”替代“结构关系机制成立”的判断。

---

## 11. 必报结果（逐任务与分层）

* 官方 Validation macro AP、Train macro AP、Train–Valid gap、Valid BCE（epoch 0/2/4/8，两组）。
* 127 个可评分任务的逐任务 AP、正例数。
* `REAL−A`、`SHUFFLE−A`、`REAL−SHUFFLE` 的逐任务 AP 差。
* 按 Validation 正例数分层：`>=20`、`5–19`、`<5`。
* REAL 比 A 改善的任务数；REAL 比 SHUFFLE 改善的任务数。
* 高支持度任务的平均 / 中位 AP 差值；正向增量的 top-5 / top-10 贡献占比。
* Motif 与距离覆盖率；训练关系权重的 L1 范数、非零比例、各 assay 权重规模。
* 每组 GPU 时间与峰值显存；额外推理 FLOPs 或每分子计算成本；Test isolation audit。

**不得只报告最大改善的几个 assay。**

---

## 12. 资源预算与停止条件

| 项 | 值 |
|---|---|
| Target GPU budget | `<= 0.5` A100 GPU-hour |
| Hard GPU cap | `<= 1.0` A100 GPU-hour |
| 执行主机 | `res`（`a100-2`），pool `res-gpu1`（`CUDA_VISIBLE_DEVICES=1`，不抢占 GPU0） |
| 工作流 | `rr doctor res` → `rr deploy res --pool res-gpu1` → `rr run …` → `rr logs/status` → `rr pull` |
| CPU 计时 | 1024 分子真实计时 + 峰值内存，外推全量 Train+Valid 构建成本 |
| 超预算 | 优雅停止，保留已完成结果，记录 `INCOMPLETE_RESOURCE`，不追加预算 |

允许使用完全等价的分块流式实现（memmap / 分块写入）来降低内存与磁盘压力；
不得通过减少训练样本、随机采样 motif pairs 或截断距离改变实验数学定义。

---

## 13. 交付物

```
PROTOCOL.md
README.md
REPORT.md
code/rie_common.py   rie_motifs.py   rie_features.py   rie_extract.py
code/rie_model.py    rie_train.py    rie_compare.py    rie_audit.py
tests/               （CPU-only 单元测试，28 项）
analysis/checkpoint_replay.json
analysis/motif_coverage.json
analysis/relation_invariance_audit.json
analysis/shuffle_control_audit.json
analysis/feature_manifest.json
analysis/paired_initialization.json
analysis/epoch8_comparison.json
analysis/per_task_deltas.csv
analysis/gpu_ledger.json
analysis/decision.json
runs/                （git-ignored：structures / features / heads / smoke）
```

大数组与 checkpoint 不入 Git，但其 SHA256 与恢复路径写入 `analysis/*.json`。

---

## 14. 解释边界（事前声明）

本轮不得声称：

* 阴性结果证明“显式结构关系在 MolPCBA 上无效”（只是单 subset、单 seed、8 epoch、单宽度）；
* `D_base > 0` 证明 S1/S2 表征不足或原 A 有特定机制缺陷；
* 单 seed 单 epoch 差异具有统计显著性；
* 已经用于历史模型选择的 Validation 可充当独立最终确认集；
* 任何结果自动授权完整 LCRC-RIE 联合训练、超参数搜索、更多 seed 或更多 motif 家族。
