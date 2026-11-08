# PROTOCOL — LCRC-PW v1（W2 原子状态上的一次性 pair 交互读出）

Round id: `lcrc-pcba-pw-v1`
Track: `structural-sufficiency`
状态：**在第一次正式训练步骤之前冻结**（本文件在 `runs/PW` 产生之前提交；之后只追加
`REPORT.md` 的数字，不回改定义）

---

## 1. 问题

历史 **LCRC-W2**（994,324 参数）与 **LCRC-RC v1**（REAL/BLIND，1,051,022 参数）都不含
「原子对（pair）级」的显式交互项：W2 的原子表示只由半径 2 的局部内容决定，RC 的分支只做
**图级**的角色×距离桶共现统计。本轮的**唯一**问题是：

> 在完全相同的 W2 主干、初始化、数据、batch 顺序与优化配方下，把 W2 一次性生成的原子
> 表示 `h ∈ R^{n_atoms×128}` 当作**只读**输入，加一层**一次性**的、按最短路距离分桶的
> pair-token 注意力读出，能否改善官方 Validation / Test 的 macro AP？

本轮只训练**一个**模型（seed 0, 40 epochs）。不做多 seed、不做超参数搜索、不做结构消融、
不因为看到 Test 结果而新增训练。如果 PW 不能明显超过已有最强 LCRC，就客观报告，**不自动
启动 PW-v2**。

---

## 2. 基线（只读引用，本轮不重训）

| 模型 | 参数 | best-Valid epoch | Valid AP (best) | Valid AP@40 | 官方 Test AP (best) | GPU-hour |
|---|---:|---:|---:|---:|---:|---:|
| LCRC-W2 | 994,324 | 24 | 0.26527331610028143 | 0.26186691813350904 | 0.2536164787414065 | 1.3949 |
| LCRC-RC-REAL | 1,051,022 | 33 | 0.2677177024413342 | 0.26695028589805453 | 0.25593967068841966 | 1.9186 |
| LCRC-RC-BLIND | 1,051,022 | 见 `runs/BLIND/run.json` | 见同处 | 见同处 | 见 `analysis/test_scores_peek_*.json` / 最终 `analysis/test_scores.json` | 见同处 |

W2 与 RC 都不在本轮重训。W2 的两个 checkpoint 会被**重新推理一次**（只读，只为把
same-pipeline 的 Test AP 放在同一张表上）。RC-REAL / RC-BLIND 的 Test AP 直接读
`lcrc_pcba_rc_v1/analysis/`（只读），不重算、不改写。

---

## 3. 模型定义（逐字实现）

### 3.1 只读原子状态

```
h = W2.drop(W2.local(batch, a, b, scalar))        # [N_atoms, 128]，每次 forward 只算一次
```

`h` **不允许**被再次更新：没有消息传递、没有邻接传播、没有原子状态写回、没有第二次
atom encode、没有多层 pair attention、没有 pair token 迭代。`h` 与 W2 readout 使用的
`h` 必须是**同一个 tensor**（forward hook 计数 + 身份检查）。

### 3.2 关系索引（离线、只依赖结构）

对每个分子，只考虑**同一连通分量内**的无序原子对 `i < j`，用**原始化学键图的
shortest-path 距离**（无权 BFS）分箱：

| 桶 | 距离 |
|---|---|
| 0 | 1 |
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5–6 |
| 5 | ≥ 7 |

不同连通分量的原子对**不进入索引**。索引在训练前一次性构建
（`runs/pair_cache/`：`pair_i int16`、`pair_j int16`、`pair_bucket uint8`、
`pair_ptr int64`），构建只打开 `atom_ptr` / `bond_ptr` / `bond_index`，
`meta.json` 记录 `labels_read: false`；训练/推理时**不做 BFS**，也不调用
RDKit / NetworkX / SciPy 图算法，per-batch 只做一次向量化 gather。

### 3.3 对称 48 维 pair token

```
p_ij = LayerNorm( SiLU( W_sum(h_i + h_j) + W_diff|h_i - h_j| + W_prod(h_i * h_j)
                        + E[bucket_ij] ) )                    # [P, 48]
```

* `W_sum, W_diff, W_prod: Linear(128, 48, bias=False)`；
* `E: [6, 48]` 距离嵌入；
* `LayerNorm(48)`。

`W_sum(h_i+h_j)` 用 `W_sum(h_i) + W_sum(h_j)` 实现（同一个线性映射，恒等变换，省一次
per-pair matmul）。三项在 `i <-> j` 交换下都**逐位**不变（IEEE 加法/乘法可交换、
`abs` 对称、同一 embedding 相加）。

### 3.4 16 个共享 query 的分子内 softmax

```
k_ij      = Linear(48,48)(p_ij)
v_ij      = Linear(48,48)(p_ij)
s_kij     = <q_k, k_ij> / sqrt(48)                 # q_k 为 16×48 的参数
a_kij     = softmax_{ij ∈ molecule}(s_kij)         # 每个 (分子, query) 一次 softmax
z_k       = sum_ij a_kij v_ij                      # [G, 16, 48]
```

softmax **只在该分子自己的合法 pair 上**归一（segment softmax），绝不跨分子、绝不跨
连通分量，也不做任何采样/截断：一个分子的全部合法 pair 都参与。

### 3.5 残差读出

```
z          = flatten(z)                                        # [G, 768]
residual   = Linear(128,128)( Dropout(0.10)( SiLU(
                 Linear(768,128)( LayerNorm(768)(z) ) ) ) )     # [G, 128]
residual   = residual * 1[n_pairs(molecule) > 0]                # 显式 mask
logits     = W2_logits + residual
```

最后的 `Linear(128,128)` **权重与偏置全零初始化**，因此初始 `residual` 严格为 0、初始
`logits` 与 W2 逐位相同。没有合法 pair 的分子（单原子图）由显式 mask 保持残差严格为 0，
而不是依赖「LayerNorm(0)=0」这类数值巧合。

### 3.6 新增参数（实测，`pw_model.build_model` 断言）

| 模块 | 张量 | 参数 |
|---|---|---:|
| `branch.w_sum` | Linear(128,48,无 bias) | 6,144 |
| `branch.w_diff` | Linear(128,48,无 bias) | 6,144 |
| `branch.w_prod` | Linear(128,48,无 bias) | 6,144 |
| `branch.dist_embedding` | [6,48] | 288 |
| `branch.norm` | LayerNorm(48) | 96 |
| `branch.query` | [16,48] | 768 |
| `branch.key` | Linear(48,48) | 2,352 |
| `branch.value` | Linear(48,48) | 2,352 |
| `branch.combiner.norm` | LayerNorm(768) | 1,536 |
| `branch.combiner.fc1` | Linear(768,128) | 98,432 |
| `branch.combiner.fc2` | Linear(128,128) | 16,512 |
| **合计新增** | | **140,768** |

总参数 **1,135,092**（`PARAMETER_CEILING = 1,200,000`，本轮的工程上限；RC 轮的
1,100,000 是那条分支自己的上限，不适用于这个定义）。

初始化规则：`torch.manual_seed(0)` → 历史 W2 seed-0 构造 → 把 W2 state **逐位**拷入
PW 模型 → 新张量全部来自**私有 CPU generator**（`seed*1000003+1401`）：
Linear 用 `kaiming_uniform(a=sqrt(5))` + `fan_in` 均匀 bias（三个 pair 投影无 bias），
距离嵌入与 16 个 query 用 `normal(0, 1/sqrt(48))`，LayerNorm 用默认 (1,0)，
最后 `Linear(128,128)` 权重与 bias 置零。**不推进全局 torch RNG**，也不改变 W2 的种子状态。

### 3.7 明确禁止

多层 pair attention、pair/atom 消息传递、原子状态写回、pair token 迭代更新、额外预训练、
目标函数变化、pair 采样、pair 截断、按距离桶加权（softmax 已隐含权重）。

---

## 4. 数据与训练配方（完全沿用 W2）

* 官方 OGB MolPCBA 划分：Train 350,343 / Valid 43,793 / Test 43,793，128 个任务；
* 训练只用官方 Train 标签；Validation 只用于 checkpoint 选择；
* Train 标签在训练产物中是唯一可见标签（Test 行全 NaN，`rd_dist.load_task_gated` 门禁）；
* seed 0，40 epochs，batch 128，AdamW(β=(0.9,0.999), eps=1e-8, wd=1e-3)，
  LR 1 epoch 线性 warmup 3e-5→3e-4 后 per-step cosine→1e-5（109,520 步），
  dropout 0.10（主干）+ 0.10（combiner），grad clip 5.0，masked BCE（NaN 只做 mask）；
* Train 顺序：继承的 `make_plan40(seed=0)`，`global_plan_sha256 = 06f437...`；
* W2 与 pair 分支**从头联合训练**，不加载已训练 W2，不 freeze；
* 单次训练，不做第二次以搜索 LR / query 数 / pair 维数。

资源：`res` 的 `res-gpu1`（BLIND 结束并释放后）。GPU0 只有在确认没有既有进程时才使用，
绝不抢占。训练前用**真实 batch** 做 GPU profiling（PW 与 W2 同一批数据同一个 step），
目标完整训练 ≤ 3 A100 GPU-hour；超预算时只允许优化张量布局/实现，不允许裁剪 pair、
随机采样关系或改变数学定义。

保存：每 epoch Validation AP / Valid BCE / Train loss / 耗时 / 诊断量，best-Validation
checkpoint、epoch-40 checkpoint、可续训 last checkpoint、optimizer 与 RNG 状态、参数量、
运行环境、GPU-hour、pair 索引哈希。

---

## 5. 训练前必须通过的正确性门槛

`code/pw_audit.py checks` 在目标设备上运行，结果写 `analysis/preflight.json`
（任一项失败退出码 4）；`tests/` 用 pytest 覆盖同一批断言。

1. PW 的 W2 部分与历史 seed-0 构造**逐位**一致（`base_state_sha256`、trunk fingerprint），
   新增参数名字/形状/数量/总量与协议表一致，最后 Linear 严格为零；
2. zero-init 时 `logits` 与原始 W2 forward **逐位**相同（CPU），CUDA 上残差严格为 0、
   logits 相对误差 ≤ 1e-6；`residual_zero` 路径同样逐位相同；
3. 任意合法原子重编号（含用置换后的 `bond_index` 重建 pair 索引）不改变预测（rel ≤ 1e-4）；
4. pair token 在 `i <-> j` 交换下**逐位**不变（所有设备）；把尾部替换为非零固定随机
   权重后，交换两个端点不改变 logits——CPU 上要求逐位相等，CUDA 上允许 ≤1e-4，
   因为 `index_add`/`scatter_reduce` 是原子累加，两次相同调用的求和顺序可以不同，
   会改变 `z` 的末位；token 逐位相等是不放松的那一条；
5. 不存在跨图 pair：`pair_counts`/`pair_graph` 自洽、`i<j`、两个端点在各自分子内、
   pair 不重复；把每个分子单独 collate 后重新打分，与该分子在 batch 中的 logits 一致；
6. 不存在跨连通分量 pair：索引只在同分量内建；用独立 BFS 复算抽样行，
   「存下的 (i,j,bucket) 集合 == 同分量全部无序对的 BFS 分桶」；
7. 距离桶与原始化学键一致：抽样行逐对 BFS 距离 → 桶，与索引逐条相等；桶互斥且覆盖；
8. segment softmax：每个 (分子, query) 在含合法 pair 的分子上归一（max|Σa-1| ≤ 1e-5），
   无 pair 分子恒为 0，权重非负有限；
9. 若干 optimizer step 后所有新增参数都获得非零梯度（`key.bias` 除外：分子内常数
   偏置被 softmax 精确消去，属定义自带的退化，报告而非隐藏）；
10. 小图 brute-force（float64，逐 pair 逐分子 Python 循环）与批量实现（CPU 与 CUDA）
    在 token / attention / z / residual 上相对误差 ≤ 1e-5；
11. 无合法 pair 的极小图：输出有限，pair residual **严格为 0**，logits 等于 W2；
12. 不存在 Test 标签参与训练或 checkpoint 选择：Train 产物 Test 行全 NaN、
    Train/Valid/Test 互不相交、order plan SHA 等于 W2、checkpoint 选择只用 Validation AP。

---

## 6. 官方 Test 评分

流程与 RC/SP 完全一致（`code/pw_test.py freeze` → `score`）：

1. `freeze` 只写 Test logits 与全部哈希（含 Validation AP 复算），**不打开任何 Test 标签**；
   对 PW best-Valid / PW epoch-40 / W2 best-Valid / W2 epoch-40 各写一份；
2. seal 验证通过后，`score` 才通过继承的 `cc_data.read_official_test_labels`
   读取 `raw/graph-label.csv.gz`；
3. 用官方 `ogb.graphproppred.Evaluator("ogbg-molpcba")` 打分，并用 sklearn
   `average_precision_score` 逐任务独立重算，二者差 < 1e-9；
4. 预测顺序与官方 Test row id 严格对应（`test_row_ids.npy` + SHA256）；
5. RC-REAL / RC-BLIND 的逐任务 Test AP 从 RC 轮已冻结的 `per_task_valid_test.csv`
   **只读 join**，用于同任务轴对比。

Test 在本 track 已被暴露，本轮明确把 Test 当作**已暴露的工程比较集**，不声称盲测；
不因为看到 Test 结果而新增训练轮次或新配置；单 seed 差异不描述为统计显著。

---

## 7. 报告与决策规则

`REPORT.md` 给出固定对比表（参数、Best Valid AP、Valid AP@40、Test AP(best)、
Test AP@40、GPU-hour），以及

* PW − W2 / PW − RC-REAL / PW − RC-BLIND 的 Test AP 差（绝对值、相对值）、
  改善/退化任务数、逐任务正负质量与 top-5 集中度；
* 正例 ≥20 的高支持子集（mean gain、改善比例、median）；
* Train–Validation 差距（Train AP@40、Valid AP@40、Valid−Test）；
* 参数量与运行成本（GPU-hour、step 时间、相对 W2 的推理开销）；
* 16 个 query 的平均注意力熵：原始熵、`log n_pairs` 参考、归一化熵
  `H/log n`、有效 pair 数 `exp(H)`、平均最大注意力权重——判断是否**全部接近均匀**。
  这一步只做推理分析，不重新训练。

决策标签（工程判据，与 RC 同一把尺子，只看 best-Validation 的 Test AP 差）：

| 条件 | 标签 |
|---|---|
| Δ ≥ +0.005 | `PROMISING_PW_SIGNAL` |
| +0.002 ≤ Δ < +0.005 | `WEAK_PW_SIGNAL` |
| −0.002 < Δ < +0.002 | `NO_CLEAR_PW_GAIN` |
| Δ ≤ −0.002 | `PW_REGRESSION` |

Δ 的定义：PW best-Validation Test AP 减去 {W2, RC-REAL, RC-BLIND} 中最强者的
best-Validation Test AP。

不追加消融、不搜索超参数、不跑多 seed、不换数据集、不启动 PW-v2。
