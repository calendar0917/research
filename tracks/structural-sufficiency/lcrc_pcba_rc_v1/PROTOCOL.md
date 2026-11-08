# PROTOCOL — LCRC-RC v1（PCBA 无迭代消息传递的关系组合实验）

Round id: `lcrc-pcba-rc-v1`
Track: `structural-sufficiency`
状态：**在第一次正式训练步骤之前冻结**（本文件在 `runs/REAL`、`runs/BLIND` 产生之前提交）

---

## 1. 问题

历史 **LCRC-W2**（994,324 参数）在 MolPCBA 上不使用任何迭代消息传递：一次原子/键编码、
一次 S1/S2 局部统计、per-atom `f`、global `rho`、四头 attention pooling。它的原子表示
`h_i` 只由半径 2 的局部内容决定，模型没有任何显式的「谁与谁在什么距离上」的
**角色（role）共现**统计。

本轮只问一个问题：

> 在完全相同的 W2 主干、初始化、数据、batch 顺序与优化配方下，把一个**一次性**的、
> **图级**的角色—距离共现统计分支加到 `h` 上，能否改善官方 Validation / Test 的
> macro AP？改善（若有）是来自真实的「角色 × 距离桶」绑定，还是仅仅来自额外的容量与
> 全图角色统计？

对照关系由 **REAL / BLIND** 两臂给出：两臂的原子角色输入、距离桶比例、MLP、任务参数、
输出维度、训练配方完全一致，**唯一**差异是 `F_d` 是否保留「角色 ↔ 距离桶」的真实绑定。

本轮只训练两条轨迹（seed 0），不搜索学习率、不搜索角色数/距离桶、不跑多 seed、不做消融。

---

## 2. 基线

| 项 | 值 |
|---|---|
| 基线模型 | LCRC-W2（`lcrc_pcba_backbone_scaling_v1/runs/W2`，read-only） |
| 参数 | 994,324 |
| seed | 0 |
| best-Valid epoch | 24 |
| Valid AP (best) | 0.26527331610028143 |
| Valid AP @40 | 0.26186691813350904 |
| 官方 Test AP (best-Valid) | 0.2536164787414065 |
| GPU-hour | 1.3949 |

W2 本轮**不重训**。为了验证评分管线本身，本轮用同一套 freeze/score 代码重算
W2 best-Valid 与 W2 epoch-40 两个历史的官方 Test AP（只做推理，不训练）。

---

## 3. 共同计算（两臂完全相同）

### 3.1 原子状态

从头构建 W2（`bs_model.build_model("W2", ...)`，seed 0，历史初始化规则），取**经过局部
编码与原有 dropout 之后**的原子状态

```
h = W2.drop(W2.local(batch, a, b, scalar))        # [N_atoms, 128]
```

`h` 只计算一次，之后不再被更新。新增分支只读取 `h` 与静态距离桶，输出**每分子一个**
128 维 logits 向量。

### 3.2 角色

```
R = softmax(Linear(128, 12)(h), dim=-1)           # [N_atoms, 12]，非负、每行和 = 1
```

### 3.3 关系桶（离线、只依赖结构）

对每个分子，仅考虑**同一连通分量内、不同原子**的有序原子对，用**原始化学键图的
shortest-path 距离**（无权重 BFS）确定六个桶：

| 桶 | 距离 |
|---|---|
| 0 | 1 |
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5–6 |
| 5 | ≥ 7 |

不同连通分量的原子对不进入任何桶；自己与自己的原子对不计入。由此得到的
`A_d`（桶 d 的配对邻接掩码）对称、对角线为零、桶间互斥、覆盖同分量内全部原子对。

符号（全部**逐分子**）：

```
N_d   = sum(A_d)                     # 有序原子对数
A_all = sum_d A_d ,  N_all = sum_d N_d
w_d   = N_d / max(N_all, 1)
T_d   = 144 * (R^T A_d R) / max(N_d, 1)          # N_d = 0 时严格为 0
T_all = 144 * (R^T A_all R) / max(N_all, 1)
```

144 = 12²，使每个 `T_d` 的元素和恒为 144。

### 3.4 关系输入（REAL）

```
F_d = sqrt(w_d) * T_d
```

### 3.5 关系输入（BLIND）

不保留角色与距离的真实绑定，仅保留全图角色共现与每桶数量比例：

```
F_d = sqrt(w_d) * T_all      if N_d > 0
F_d = 0                      if N_d = 0
```

BLIND 与 REAL 共享：角色输入 `R`、距离桶比例 `w`、上三角提取、MLP、任务参数、输出维度、
初始化规则、训练设置。除 `F_d` 的构造方式外没有任何差异。

### 3.6 输入向量（487 维）

`T_d`、`T_all` 均对称，只保留含对角线的上三角 78 个元素：

```
x = concat( triu_78(F_0), ..., triu_78(F_5),   # 6 × 78 = 468
            w_0..w_5,                          #      6
            mean_atoms(R),                     #     12
            log1p(n_atoms) )                   #      1
```

### 3.7 共享任务组合读出（两臂完全相同）

```
z              = Linear(96,32)( Dropout(0.10)( SiLU( Linear(487,96)( LayerNorm(x) ) ) ) )
residual_logits = z @ TaskEmbedding^T + task_bias            # [n_graphs, 128]
logits          = W2.post_local(h, batch) + residual_logits
```

`TaskEmbedding: [128, 32]`，`task_bias: [128]`。`Linear(96,32)` 的权重与偏置初始化为 0，
因此初始 residual 严格为 0、初始 logits 与 W2 完全一致。其余新增参数由私有随机生成器
初始化，不推进全局 torch RNG，也不改变历史 W2 的初始参数。

### 3.8 硬约束（代码审计，不只是声明）

* 每个 forward 中 `atom_encoder` / `bond_encoder` / `local` / `drop` 各只调用一次
  （forward hook 计数，见 `rc_model.count_module_calls_typed`）；
* 分支读取的 `h` 与 W2 readout 使用的 `h` 是**同一个 tensor**（pre-hook 身份检查）；
* **禁止**：迭代消息传递、邻接更新、跨原子状态反馈、多层原子间 attention。
  分支只做图级聚合，绝不把信息写回原子表示；
* REAL 与 BLIND 的可训练参数数量完全相同。

新增参数（实测）：

| 模块 | 张量 | 参数 |
|---|---|---:|
| `branch.role_proj` | Linear(128,12) | 1,548 |
| `branch.combiner.norm` | LayerNorm(487) | 974 |
| `branch.combiner.fc1` | Linear(487,96) | 46,848 |
| `branch.combiner.fc2` | Linear(96,32) | 3,104 |
| `branch.task_embedding` | [128,32] | 4,096 |
| `branch.task_bias` | [128] | 128 |
| **合计新增** | | **56,698** |

总参数 **1,051,022**（< 1,100,000 上限）。

---

## 4. 结构数据与计算效率

* 官方 OGB MolPCBA 划分：Train 350,343 / Valid 43,793 / Test 43,793，128 个任务；
* 训练只用官方 Train 标签；Validation 只用于 checkpoint 选择；
* 距离缓存：一次性离线构建，使用图数据的真实 `bond_index`
  （`scipy.sparse.csgraph.shortest_path(..., unweighted=True, directed=False)`，每分子
  一次全源 BFS）。**训练/推理时不调用 RDKit / NetworkX / SciPy**；
* 缓存格式：每个分子的严格上三角桶码（`uint8`，152,706,480 字节）+ 行指针（`int64`）。
  每个无序原子对恰好一个字节，因此表示是**精确**的（无采样、无截断）；
* **Test 图距离缓存不读取 Test 标签**：构建只打开 `atom_ptr` / `bond_ptr` /
  `bond_index`，`meta.json` 显式记录 `labels_read: false`（自检抽样 876 个分子与独立
  纯 Python BFS 完全一致）；
* 训练时用 GPU 上的批量矩阵运算：每个 batch 一次
  `[n_graphs, N_atoms] @ [N_atoms, 7*144 + 12]` 的 matmul 完成所有 per-molecule 归约，
  没有任何 Python 循环遍历原子对；单原子对的收集用成对 `index_add_`。

---

## 5. 训练前必过的正确性检查

`code/rc_audit.py checks` 在 GPU 上运行并把结果写入 `analysis/preflight.json`；
`tests/` 用 pytest 覆盖同一批断言。检查项：

1. W2 初始化与历史 seed-0 构造逐位一致（`base_state_sha256`、trunk fingerprint）；
   REAL、BLIND 的 W2 部分完全相同；
2. REAL 与 BLIND 的新增参数名、形状、数量、总量完全一致；
3. 新分支 zero-init 时，输出与原始 W2 forward 在数值上完全一致（max|Δ| = 0）；
4. 任意合法原子重编号后模型预测不变（容差 1e-4 相对，考虑 FP32 reduction）；
5. 同一 batch 中两个分子不产生跨分子的角色共现；
6. 距离桶互斥且完整覆盖同一连通分量内的不同原子对；
7. 两个分量不连通时不出现错误的有限距离（255 哨兵）；
8. 所有原子角色分布相同时 REAL 与 BLIND 的关系输入相同；
9. 两个分子角色多重集合与距离桶总数相同但角色—距离对应不同时，REAL 产生不同统计、
   BLIND 保持不变；
10. 若干 optimizer step 后角色投影与组合读出均获得非零梯度；
11. CPU 逐对枚举的结果与实际 batched 实现一致（CPU 与 CUDA 两条路径）；
12. masked BCE、官方 AP、Train-only 数据隔离与原有 W2 实现一致（同一函数对象、
    同一 order plan SHA）。

任何一项失败 → 先修实现，再开始正式训练。

---

## 6. 训练配方（完全沿用 W2）

| 项 | 值 |
|---|---|
| seed | 0 |
| epochs | 40 |
| batch size | 128 |
| 优化器 | AdamW，β=(0.9, 0.999)，eps=1e-8，weight decay 1e-3 |
| LR | 1 epoch 线性 warmup 3e-5 → 3e-4，随后 per-step cosine → 1e-5（109,520 步） |
| dropout | 0.10（主干）+ 0.10（分支 combiner） |
| gradient clip | 5.0 |
| loss | 原有 masked BCE（NaN 目标只做掩码，不填 0） |
| Train 顺序 | 继承的 `make_plan40(seed=0)`，`global_plan_sha256 = 06f437...` |
| 训练方式 | W2 与新增分支**从头联合训练**，不加载已训练 W2，不 freeze |

固定不变：所有数据集 / loss / optimizer / 模型超参数。不做第二次训练以搜索学习率、
角色数量或距离桶。

资源：REAL → `res-gpu1`（主候选）；BLIND → `res-gpu0`（资源允许时并行，绝不抢占/影响
他人进程）。目标每臂 2–2.5 A100 GPU-hour；超过 3 GPU-hour 时只优化缓存与张量计算，
不改算法定义。

保存：每 epoch 的 Validation AP / BCE / 训练 loss / 耗时 / 诊断量、best-Validation
checkpoint、epoch-40 checkpoint、optimizer 与 RNG 状态、参数量、运行环境、
GPU-hour、数据缓存信息。

---

## 7. 官方 Test 评分

两臂训练结束后（各自 40 epoch），按**最高 Validation AP** 冻结 best-Validation
checkpoint（同分取最早 epoch）。对每臂评估四个数：

* best-Validation checkpoint 的 Validation AP；
* best-Validation checkpoint 的官方 Test AP；
* epoch-40 checkpoint 的 Validation AP；
* epoch-40 checkpoint 的官方 Test AP。

同一管线对 W2 best-Valid / W2 epoch-40 也做一次推理，用于交叉核对评分实现。

流程（`code/rc_test.py freeze` → `score`）：

1. `freeze` 只写 Test logits 与全部哈希（含 Validation AP 复算），**不打开任何 Test 标签**；
2. 封条（seal）验证通过后，`score` 才通过继承的
   `cc_data.read_official_test_labels` 读取 `raw/graph-label.csv.gz`；
3. 打分用官方 `ogb.graphproppred.Evaluator("ogbg-molpcba")`，并用
   sklearn `average_precision_score` 逐任务独立重算，二者差值必须 < 1e-9；
4. 预测顺序与官方 Test row id 严格对应（`test_row_ids.npy` + `sha256`）。

Test 已在本 track 之前的轮次暴露过，本轮把 Test AP 作为**工程决策的主要比较指标**，
但：不因为看到 Test 结果而新增训练轮次或新配置；单 seed 差异不描述为统计显著。

---

## 8. 报告与决策规则

`REPORT.md` 给出固定对比表（参数、Best Valid AP、Valid AP@40、Test AP(best)、
Test AP@40、GPU-hour），以及 REAL−W2、BLIND−W2、REAL−BLIND 的 Test AP 差值、可评分任务数、
改善/退化任务数、正例 ≥20 的高支持任务子集统计、稀疏任务是否主导差异、每模型
Train/Valid/Test AP、分支是否真正发挥作用（residual RMS、角色分布统计）、
推理时间与相对 W2 的开销。

决策标签（工程判据，不是统计检验，只看 best-Validation 的 Test AP 差）：

| 条件 | 标签 |
|---|---|
| Δ ≥ +0.005 | `PROMISING_RC_SIGNAL` |
| +0.002 ≤ Δ < +0.005 | `WEAK_RC_SIGNAL` |
| −0.002 < Δ < +0.002 | `NO_CLEAR_RC_GAIN` |
| Δ ≤ −0.002 | `RC_REGRESSION` |

不追加消融、不搜索超参数、不跑多 seed、不换数据集。
