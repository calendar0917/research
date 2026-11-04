# Peptides-func 显式 S2 计数机制实验 v1 — PROTOCOL

- **Round ID**：`peptides-count-s2-mechanism-v1`
- **结果标签**：`EXPLORATORY_VALIDATION_ONLY_MECHANISM`
- **目录**：`tracks/structural-sufficiency/peptides_count_s2_mechanism_v1/`
- **数据集**：LRGB Peptides-func，官方固定划分
- **目标**：只检验“显式二跳（S2）计数特征是否携带 C1 之外的任务信息”，以及 FULL 与结构匹配的 NULL 的差异；不做文献调研、不做超参数搜索、不替换模型架构、不提出下一轮方案。

> 本文件在正式 P1 GPU 训练前冻结。任何实现选择都写在这里；运行中只允许修复实现 bug，不允许修改数据划分、学习率、正则、模型宽度、路径半径、哈希维度、损失或停止规则。

---

## 1. 数据与评测边界

| split | 行数 | 用途 |
|---|---:|---|
| Train | 10,873 | 唯一训练数据；所有标准化、模型拟合、计数特征构建 |
| Validation | 2,331 | 仅评估与 checkpoint 选择 |
| Test | 2,331 | **本轮完全不读取、不构建、不评估** |

- 不重新划分数据，不创建 Fit/Monitor/Locked。
- 目标标签不参与任何输入特征、计数、哈希或 NULL 置换的构建。
- 官方 Train/Validation 行身份必须与历史 Fast v1 Peptides 缓存逐行、逐原子、逐键一致；门禁见 §6。
- Test 保护：所有正式入口在导入数据加载器前安装 `sys.addaudithook`。禁止打开的子串包括：
  `peptides_func_test`、`peptides_func_y_test`、`test_predictions.npz`、
  `peptides-func/processed/test.pt`、`peptides-func/raw/test.pt`、
  `LRGB_Peptides-func`、`lrgb_splits_peptides_func.json`。
  任何尝试会立即抛出 `TEST_ACCESS_BLOCKED`，并在 `run.json` / manifest 的
  `guard.attempts` 中记录。正式运行允许的尝试次数为 **0**。
- 唯一例外（已在运行前记录）：P0 ECFP 复现必须使用历史同源的官方 CSV **SMILES 字段**。
  该 CSV 是 Train/Validation/Test 混合文件；`pc_p0_ecfp.py` 使用逐行流式读取器，
  **只解析 Train/Validation 行的 SMILES 子串**（SMILES 是第一列且不含逗号），
  从不解析任何标签列，从不保留或评估 Test 行。manifest 记录
  `test_rows_parsed=0`、`test_labels_read=false`。不使用该 CSV 做任何 P1 特征。

### 1.1 为什么 ECFP 用 CSV SMILES 而不是 9 列 PyG 图

官方 PyG 的 9 列离散原子特征不含同位素/显式氢表示；CSV 的 SMILES 含 `[2H]` 等
同位素信息。用 PyG 图重建 RDKit 分子后，只有 2 个 Train 行和 2 个 Validation 行的
ECFP 计数不同，但足以使 macro AP 从历史 72.98% 变为 72.70%。因此 ECFP **按历史
实现使用 CSV SMILES**；这是 ECFP 基线的输入口径，不是 P1 的图特征口径。该诊断
事实记录在最终 REPORT。

---

## 2. P0：标签无关图级计数特征

### 2.1 二跳路径定义

对每个图，枚举所有**有向非回退二跳路径** `(u, v, w)`：

- `u, v, w` 是原子索引；
- `(u, v)` 与 `(v, w)` 是原始无向键（键存储一次；内部按 `(min, max)` 规范化）；
- `u != w`（排除立即回退和三原子闭环的“回退”路径）；
- 每条路径只按有向三元组计一次；同一图的多重键不允许出现（缓存已验证无重复键）。

路径按原子索引升序、邻接表按邻居索引升序枚举。枚举总数必须与历史缓存的
`path_ptr` 每根计数一致（门禁）。

### 2.2 TYPE-S2

每条路径的 key 为 5 个原始离散字段，顺序固定：

```
(root atom column 0,
 first bond column 0,
 middle atom column 0,
 second bond column 0,
 endpoint atom column 0)
```

`column 0` 是原始 `x[:, 0]` / `edge_attr[:, 0]`（原子 0..118，键 0..4），不是
one-hot 列偏移，也不使用 `bond_type=column0+1`。key 为 5 字节，按内容哈希到
2048 桶，累加非负整数计数。

### 2.3 FULL-S2

每条路径的 key 为 33 个原始离散字段，顺序固定：

```
root atom 9 columns
first bond 3 columns
middle atom 9 columns
second bond 3 columns
endpoint atom 9 columns
```

所有字段都是原始整数离散属性（原子列范围 119/4/12/12/6/6/6/2/2；键列范围
5/6/2）。key 为 33 字节，使用与 TYPE 相同的哈希，累加非负计数。

### 2.4 FULL-S2-BINARY

由 FULL-S2 计数逐元素定义：`binary[i, b] = 1 if full[i, b] > 0 else 0`。
不重新哈希、不改变桶。

### 2.5 固定哈希

- 实现：`hashlib.blake2b(key, digest_size=8, person=b"pcs2v1")`。
- 桶：`int.from_bytes(digest, "little") % 2048`。
- 不使用 Python 内置随机化 `hash()`；不使用运行期随机盐。
- Train/Validation 使用完全相同的实现与语义。
- 记录每个 split、每种特征的：
  - 唯一内容模式数 `n_unique_patterns`；
  - 实际非零桶数 `n_nonzero_buckets`；
  - 64 位 digest 唯一数 / 64 位 digest 碰撞数；
  - 桶碰撞率 `1 - n_nonzero_buckets / n_unique_patterns`（分母为 0 时记 0）；
  - 最大桶计数；
  - 每图路径数分布；
  - `FULL-S2-BINARY` 每图非零桶数分布、全局非零桶数。

### 2.6 NULL-S2（A3 专用）

只用于 A3 的 FULL 分支，不改变 TYPE/FULL 的定义。

1. 对每个图，先按 33 字节 FULL key 的**字典序**对路径排序，得到与节点编号/
   键存储顺序无关的 canonical path order。
2. 路径分成前半 `(root 9, bond1 3, middle 9)` 和后半 `(bond2 3, endpoint 9)`。
3. 图的 seed = `sha256(b"pc-canonical-full-v1" + sorted_full_keys_bytes)` 的前 8 字节，
   小端解释为整数。该 seed 与标签无关，且对节点重编号/键顺序不变。
4. 用 `numpy.random.Generator(numpy.random.PCG64(seed))` 生成路径数 `P` 的置换；
   若 `P>1`，重新采样直到无固定点（derangement）；最多 1000 次，退化时使用固定
   循环移位 `1 + seed % (P-1)`。`P<=1` 时只能是恒等置换。
5. 前半保持原位，后半按置换重新配对，拼成新的 33 字节 key，哈希与计数同 FULL。

因此 FULL 与 NULL 的路径总数、前半字段多重集合、后半字段多重集合完全相同；
联合 key 在 `P>1` 的图上不同。逐图记录 `marginal_first_equal`、
`marginal_second_equal`、`derangement`、`joint_equal_rows`、
`full_vs_null_bucket_vectors_equal`。

### 2.7 P0 LightGBM

每种特征（TYPE、FULL、FULL-S2-BINARY）单独训练 10 个二分类器，严格使用：

```python
LGBMClassifier(
    n_estimators=500,
    is_unbalance=True,
    importance_type="gain",
    random_state=0,
    n_jobs=8,
    verbose=-1,
)
```

其余参数保持 LightGBM 默认；无调参、无早停、无树数搜索。训练只用官方 Train，
Validation 只做评估。主指标为 10 个标签 `average_precision_score` 的非加权平均
（百分数），同时报告逐标签 AP。运行环境为历史隔离 CPU venv：
`/home/calendar/.cache/research-venvs/peptides-ecfp/bin/python`。

### 2.8 ECFP 复现门禁

`ECFPFingerprint(count=True, radius=2, fp_size=2048)`，只覆盖这三个参数；原子不变量
使用 skfp/RDKit 默认。输入为官方 CSV 的 Train/Validation SMILES 行。训练/评估协议
与 LightGBM 相同。历史值 **72.98%**（Train-only → Validation）。接受区间
**±0.01 pp**；否则停止，不进入 GPU 阶段。必须同时核对特征矩阵的
`nnz_train=1,105,722`、`nnz_valid=235,742`、最大计数 114/93（历史记录）。

### 2.9 P0 MLP 读出（仅诊断）

在**同一个 FULL-S2 计数矩阵**上，固定：

```
input = log1p(FULL-S2 counts)          # 2048
Linear(2048, 64) -> SiLU -> Linear(64, 10)
AdamW(peak_lr=1e-3, weight_decay=1e-4, betas=(0.9,0.999), eps=1e-8)
batch_size=256, epochs=200, per-step cosine to 1e-5, grad clip 5
unweighted BCEWithLogitsLoss, seed=0
```

无 dropout、无归一化、无调参、无早停。按 Validation macro AP 选最佳 checkpoint；
报告 Train/Validation macro AP、逐标签 AP、最佳 epoch、末 epoch。该结果**不用于**
修改 P1 模型。

---

## 3. P1a：原始 C1 的 480-epoch 训练长度对照

保持 LCRC-C1 结构与输入不变（`bp_model.build_c1`，464,739 参数）：

| 项 | 固定值 |
|---|---|
| seed | 0 |
| max epochs | 480 |
| cosine horizon | 480 epochs（per-step） |
| min epochs / patience | 480 / 禁用（跑满 480） |
| optimizer | AdamW |
| peak LR | 3e-4 |
| weight decay | 1e-3 |
| betas / eps | (0.9, 0.999) / 1e-8 |
| dropout | 0.1 |
| batch size | 64 |
| gradient clip | 5.0 |
| loss | unweighted BCEWithLogitsLoss（10 个独立标签） |
| selection | 最高 Validation macro AP；并列取更早 epoch |

每个 epoch 记录 Train loss、Validation macro AP、逐标签 AP、LR、耗时；保留
Validation AP 最佳 checkpoint。与历史 120-epoch / 240-epoch 结果对比时，必须注明
cosine 调度周期不同；不得声称这是“纯 epoch 数量”的因果效应。

---

## 4. P1b：四个指定模型（seed 0，240 epoch）

### 4.1 共同配方

所有 A0–A3 使用与历史 C1 240-epoch 方案相同的结构/优化器：

| 项 | 固定值 |
|---|---|
| seed | 0 |
| max epochs | 240 |
| cosine horizon | 240 epochs（per-step） |
| min epochs / patience | 240 / 禁用（四臂都跑满 240） |
| optimizer | AdamW |
| peak LR | 3e-4 |
| weight decay | 1e-3 |
| betas / eps | (0.9, 0.999) / 1e-8 |
| dropout | 0.1 |
| batch size | 64 |
| gradient clip | 5.0 |
| loss | unweighted BCEWithLogitsLoss |
| selection | 最高 Validation macro AP；并列取更早 epoch |
| batch order | `torch.Generator().manual_seed(0)` 的同一 `randperm` 序列 |

历史 240-epoch run 实际跑满 240/240，因此“跑满 240、禁用早停”与历史停止规则在
本轮等价；四臂长度严格一致，避免早停差异混淆配对比较。A0 是本轮重新训练的同轮
匹配基线；与历史 P2 的差异包含 CUDA 归约非确定性，不声称逐位复现。

### 4.2 A0：BASE

原始 LCRC-C1，完全没有计数分支，logits = C1(batch)。参数 **464,739**。

### 4.3 A1 / A2 / A3：计数分支

在原始 C1 的 10 维输出 logits 上增加一个分支：

```python
logits = C1(batch) + Linear(2048, 10, bias=False)(log1p(counts))
```

- 新增线性层 **全零初始化**；bias=False；参数量 `2048*10 = 20,480`。
- A1 使用 TYPE-S2 计数；A2 使用 FULL-S2 计数；A3 使用 NULL-S2 计数。
- A1/A2/A3 架构、维度、参数量、初始化方式、公共 C1 参数、batch 顺序、训练配方
  完全相同；只有 `counts` 矩阵不同。
- 新分支不改变 C1 基础编码器；不增加预训练、attention、消息传递、额外特征或损失。
- 总参数 **464,739 + 20,480 = 485,219**；必须同时记录 C1 参数与新增分支参数。
- 计数按图对齐；不做跨图聚合、不做图间归一化、不做标签相关变换。

### 4.4 共享初始化与 RNG

- 每个 arm 构建前 `torch.manual_seed(0)`；`bp_model.build_c1` 内部的 seed-matched
  FULL→C1 映射产生与历史 seed0 checkpoint 相同的 `init_state_hash` 门禁。
- A1/A2/A3 的 C1 子模块 `state_dict` 必须与 A0 逐位相同（门禁）。
- 零初始化分支不消耗 RNG。
- 模型构建完成后重新 `torch.manual_seed(0)`，再开始训练；四臂 dropout 流相同。
- 首 epoch 行序 sha256 四臂必须相同（门禁）。

---

## 5. 指标与报告口径

- 主指标：10 个标签的 macro Average Precision（百分数），Validation。
- 同时报告：逐标签 Validation AP、Train AP（选中 checkpoint）、Train loss 曲线、
  最佳 epoch、末 epoch、seed 差异。
- 单位：AP 百分点（pp）。
- Gain：
  - `Gain_FULL_BASE = AP(A2) - AP(A0)`
  - `Gain_FULL_NULL = AP(A2) - AP(A3)`
  - `Gain_FULL_TYPE = AP(A2) - AP(A1)`
- 所有比较必须同协议、同数据、同 split、同训练长度；P1a 与 P1b 不混入同一主表。
- 报告中必须区分“观察到的数值”与“因果解释”。本轮只做机制检验，不做论文定位。

---

## 6. 实现正确性门禁（正式 GPU 前必须全部 PASS）

写入 `analysis/correctness_gates.json`：

1. **行身份**：官方 Train/Validation 每行与历史 Fast v1 Peptides 缓存的原子 9 列、
   键 3 列、无向连接、标签逐行一致；行数 10,873/2,331；标签哈希匹配历史
   `peptides_func_y_{train,valid}.npy`。
2. **手工小图 S2**：四原子链的 4 条非回退二跳路径、三原子环的 6 条路径
   （端点不等于根）、FULL key 33 字节布局正确。
3. **重编号/键序不变性**：随机图节点重编号、无向键存储顺序打乱后，
   TYPE/FULL/NULL 图级计数向量逐元素相同。
4. **单图 vs batch 前向**：A0 与 A2 的单图 logits 与 batch logits 最大绝对差 ≤1e-5。
5. **FULL/NULL**：路径总数相同；前半/后半字段多重集合相同；derangement 成立；
   多路径图的联合 key 与桶向量确实不同；全 split 的 marginal failure = 0。
6. **哈希固定**：固定 key 列表在两个独立子进程与本进程的 digest/bucket 完全一致；
   记录 TYPE/FULL/NULL 的碰撞审计。
7. **分支初始化**：新增分支权重全零；分支增量 `count_linear(log1p(counts))` 最大绝对值恰为 0。GPU 上独立重复调用同一 C1 编码器可能有 float32 舍入差，因此另行记录 `max_abs_output_delta_vs_base` 与 `max_abs_base_forward_repeat_diff`，不把该舍入差误判为分支非零；CPU gate 要求加分支前后 logits 最大差为 0。
8. **第一步梯度**：第一个 batch 反传后，分支权重梯度有限且非零。
9. **训练后分支**：权重不再恒零，Validation 分支输出不是恒零。
10. **checkpoint 重载**：保存的最佳 checkpoint 重新加载后复算 Validation macro AP，
    与选择值差 ≤1e-4；四臂 smoke 都必须通过。
11. **一致性**：四臂使用相同原始标签、官方 split、优化器设置、checkpoint 选择规则；
    首 epoch 行序哈希相同。
12. **无泄漏/无 NaN**：无跨图聚合、无标签进入特征、无 Test 访问、无非有限
    loss/梯度/概率。
13. **ECFP 复现**：`reproduction_ok=true`，72.98% ±0.01 pp；特征 nnz/max 与历史一致。
14. **P0 产物**：TYPE/FULL/FULL-S2-BINARY 的 LightGBM 与 MLP 结果文件存在且
    `test_evaluated=false`。

任一失败：先修实现错误并记录；不得修改科学设计、不得绕过执行系统。

---

## 7. 执行顺序、停止规则与确认

1. 构建 P0 特征 → ECFP 复现门禁 → P0 LightGBM / MLP → CPU correctness gates。
2. 只有 §6 全部 PASS 才部署并运行 P1a/P1b。
3. P1a：seed 0，480 epoch。
4. P1b：seed 0 的 A0/A1/A2/A3，各 240 epoch。
5. seed 0 完成后计算三个 Gain 与逐标签 FULL≥NULL 计数。
6. **停止判定**：
   - 若 `Gain_FULL_BASE >= 2.0` 且 `Gain_FULL_NULL >= 1.5` 且 FULL 在至少
     6/10 个标签上 `AP(A2) >= AP(A3)`，则按已冻结方案追加 seed 1、2 的
     **仅 A0/A2/A3** 三臂确认（各 240 epoch）。
   - 否则停止：不追加 seed，不进行任何新模型、参数、特征、损失或读出搜索。
7. 确认阶段报告：
   - 三个 seed 的配对差（A2−A0、A2−A3）；
   - 三 seed 均值；
   - 每标签差异（逐 seed 与均值）；
   - Validation 样本级 paired bootstrap 区间（10,000 次重采样，同一重采样索引
     同时用于两个模型，95% percentile）；
   - Train AP、收敛曲线；
   - 不得只选最好 seed 汇报。
8. 失败只允许修复实际 bug；不允许调整 LR、正则、宽度、路径半径、哈希维度、损失
   作为补救。

### GPU 预算

- 硬上限：**10 A100 GPU-hours**（所有实验、所有重试合计）。
- 预计：P1a ~1.3 h；P1b 4×~0.65 h；确认 6×~0.65 h；合计 ~7.8 h。
- 两个 pool：`res-gpu0`、`res-gpu1`；每张 GPU 同时最多一个训练任务。
- 每个 run 记录 `seconds` / `gpu_hours`；失败重试也计入。若累计接近 10 h，
  停止并记录，不以任何方式绕过上限。
- 统一通过 `rr`：`rr doctor res`、`rr deploy res`、`rr run res ... --pool res-gpu0/1`、
  `rr status/logs/pull`。不手工 `ssh`/`nohup`。

---

## 8. 代码与运行入口

| 文件 | 作用 |
|---|---|
| `code/pc_common.py` | 路径、常量、Test guard、稳定哈希、IO |
| `code/pc_data.py` | 缓存/PyG Train/Validation 读取、行身份、C1 task |
| `code/pc_s2.py` | 路径枚举、TYPE/FULL/NULL 计数、碰撞审计、特征构建 |
| `code/pc_p0_features.py` | 构建 `analysis/features_{train,valid}.npz` + JSON |
| `code/pc_p0_ecfp.py` | 历史 ECFP4 + LightGBM 复现（隔离 CPU venv） |
| `code/pc_p0_lgbm.py` | TYPE/FULL/FULL-BINARY 的固定 LightGBM |
| `code/pc_p0_mlp.py` | FULL-S2 的固定 MLP 读出诊断 |
| `code/pc_gates.py` | §6 CPU correctness gates |
| `code/pc_p1_model.py` | C1 + 零初始化计数分支、参数/初始化门禁 |
| `code/pc_p1_train.py` | P1a / A0–A3 训练、逐 epoch 曲线、checkpoint、重载复核 |
| `code/pc_aggregate.py` | P0/P1 汇总、Gain、停止判定、bootstrap、机器可读结果 |
| `tests/test_pc_core.py` | 小图枚举、不变性、哈希稳定性单元测试 |

### 复现命令

```bash
cd tracks/structural-sufficiency/peptides_count_s2_mechanism_v1

# P0 特征（CPU，约 1 min）
PYTHONPATH=code uv run python code/pc_p0_features.py

# ECFP 复现（隔离 CPU venv）
PYTHONPATH=code /home/calendar/.cache/research-venvs/peptides-ecfp/bin/python code/pc_p0_ecfp.py

# P0 固定 LightGBM
PYTHONPATH=code /home/calendar/.cache/research-venvs/peptides-ecfp/bin/python code/pc_p0_lgbm.py

# P0 MLP 读出
PYTHONPATH=code uv run python code/pc_p0_mlp.py

# CPU correctness gates
PYTHONPATH=code uv run python code/pc_gates.py

# 单元测试
PYTHONPATH=code uv run pytest -q tests

# P1（远端 A100，经 rr；示例）
rr doctor res
rr deploy res
rr run res pcs2-p1a-480 --pool res-gpu0 --gpus 1 --cpus 4 --mem 32G --time 02:30:00 \
  --result tracks/structural-sufficiency/peptides_count_s2_mechanism_v1/runs/p1a_480 \
  -- uv run python tracks/structural-sufficiency/peptides_count_s2_mechanism_v1/code/pc_p1_train.py \
     --arm P1A --seed 0 --device cuda --out-dir tracks/structural-sufficiency/peptides_count_s2_mechanism_v1/runs/p1a_480 \
     --max-seconds 8400
rr run res pcs2-a0-s0 --pool res-gpu1 --gpus 1 --cpus 4 --mem 32G --time 01:15:00 \
  --result tracks/structural-sufficiency/peptides_count_s2_mechanism_v1/runs/a0_s0 \
  -- uv run python tracks/structural-sufficiency/peptides_count_s2_mechanism_v1/code/pc_p1_train.py \
     --arm A0 --seed 0 --device cuda --out-dir tracks/structural-sufficiency/peptides_count_s2_mechanism_v1/runs/a0_s0 \
     --max-seconds 4200
# A1/A2/A3 同理，--arm A1/A2/A3，out-dir 对应
rr status pcs2-p1a-480 && rr pull pcs2-p1a-480
```

---

## 9. 禁止事项（本轮）

- 不读取/构建/评估 Test 数据或标签；不打开 Test 预测。
- 不重新划分数据；不创建 Fit/Monitor/Locked。
- 不进行文献搜索、SOTA 调研、新研究问题设计。
- 不替换模型架构、不增加 attention/消息传递/预训练/额外特征。
- 不调学习率、正则、dropout、batch size、模型宽度、路径半径、哈希维度、损失。
- 不根据结果提出补救方案或新增实验；确认阶段只按 §7 的冻结规则追加 seed 1/2
  的 A0/A2/A3。
- 不修改历史 `lcrc_benchmark_positioning_fast_v1/`、`lcrc_peptides_diagnostic_v1/`
  的报告、checkpoint、结果文件。

---

## 10. 环境与溯源

- Python / torch / numpy / scikit-learn / torch-geometric 以 `uv.lock` 和运行
  `environment` 字段为准。
- ECFP/LightGBM 隔离环境：`/home/calendar/.cache/research-venvs/peptides-ecfp`，
  版本记录在 `analysis/ecfp_validation.json` 与 `analysis/p0_lgbm.json`。
- 历史缓存（只读）：
  - `tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1/cache/peptides_func_train.pt`
  - `.../peptides_func_valid.pt`
  - 特征 spec：`.../analysis/peptides_func_spec.json`
  - 标签：`.../analysis/peptides_func_y_{train,valid}.npy`
- 历史 seed0 240-epoch checkpoint 只用于初始化哈希核对：
  `tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/runs/peptide_c1_seed0_240ep/checkpoint.pt`。
  不修改、不覆盖。
- 每个 run 的 `run.json` 记录 git commit、环境、配置、参数、曲线/ checkpoint/
  predictions 的 SHA-256、GPU 秒数、guard 尝试、`test_evaluated=false`。
