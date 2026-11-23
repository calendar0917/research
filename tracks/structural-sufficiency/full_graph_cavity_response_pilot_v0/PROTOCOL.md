# FGCR-v0 冻结协议 — Full-Graph Cavity Response pilot

Round id: `fgcr-v0-full-graph-cavity-response-pilot`
Directory: `tracks/structural-sufficiency/full_graph_cavity_response_pilot_v0/`

本文件是**运行前冻结**的实验定义。模型数学、势网络、句子/读出、优化配方、
评估、资源门槛和停止规则均不得在观察到结果后修改。Agent 只负责实现、验证
工程正确性、执行和报告事实；研究命题与边界由任务书给定。

---

## 0. 任务与边界

从原始分子图直接学习整分子性质，**非迭代**：

1. 由 9 个 OGB 原子类别字段和 3 个 OGB 键类别字段产生共享的二状态能量势；
2. 在完整分子图上执行**精确** junction-tree sum-product 推断；
3. 由同一次推断得到全图交互响应 `H(G)` 和每个原子的根条件环境响应 `eta(v)`；
4. 用这些响应直接预测整分子标签。

只使用整分子性质标签训练。禁止：R2/LCRC 隐藏向量、教师 checkpoint、蒸馏、
预训练、辅助任务、迭代 GNN、Transformer、额外手工官能团词表、旧 PRCG 规则。
本轮不做图词解释性分析或机制消融。

---

## 1. 数据与划分

复用（只读）：

```
tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz
```

数据入口：`sq_data.load_cache()`（从 `tracks/structural-query-learning/code/`
以只读方式导入）。缓存 SHA-256：

```
fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c
```

官方 split（运行前断言）：

| split | rows |
|---|---:|
| Train | 3360 |
| Valid | 420 |
| Test | 420 |
| total | 4200 |

检查三组 row id 两两无交集、并集恰为 `0..4199`。标签必须全部有限。

输入：全部 9 个 OGB 原子类别字段、3 个 OGB 键类别字段。每条物理无向键只保留
一次（缓存已按 `u < v` 合并，并验证 OGB 有向重复记录在所有键字段上一致）；
不删除独立分量或孤立原子。

类别 embedding 表大小由 **Train** 中出现的最大类别值 + 1 确定；运行前断言
Valid/Test 的所有类别值都落在该范围内（否则记为数据阻塞，不静默截断）。

标准化只使用 Train 标签的均值与**总体**标准差（`ddof=0`）。所有报告的 RMSE/MAE
还原到原始标签单位：

```
y_pred_raw = y_pred_std * std_train + mean_train
```

---

## 2. 能量与精确推断（冻结）

每个分子图 `G=(V,E)`，16 个独立通道 `r=1..16`，每原子隐状态
`sigma_v in {0,1}`：

```
E_r(G, sigma) = sum_{v in V} u_{v r}(sigma_v)
              + sum_{(v,w) in E} J_{e w r}(sigma_v, sigma_w)
log Z_r(G)    = log sum_{sigma in {0,1}^|V|} exp(E_r(G, sigma))
```

代码中键势为 `J[e,r,s,t]`，并强制端点交换对称：

```
J[e,r,0,1] = J[e,r,1,0]
```

每条边只预测 `J00, J01, J11`。

### 2.1 全图交互响应

```
H_r(G) = log Z_r(G) - sum_v logsumexp(u_{v r}(0), u_{v r}(1))
```

每张图输出 16 维 `H`。

### 2.2 根条件环境响应

从**同一次**精确推断的 calibrated clique log-beliefs 中，取每个原子的两种状态
未归一化对数权重 `logw[v,r,0]`、`logw[v,r,1]`。二者之差就是精确后验对数几率。
定义：

```
eta_{v r} = logw[v,r,1] - logw[v,r,0] - (u_{v r}(1) - u_{v r}(0))
```

必须直接使用对数权重之差，不得先求概率再 `logit`。每原子输出 16 维 `eta`。

边势全为零时必须（浮点容差内）：

```
H = 0,   eta = 0
```

### 2.3 推断实现

复用已验证的 BAMF-v1 junction-tree 算法，逐字节复制到本目录：

```
code/bamf_inference.py
source: tracks/structural-sufficiency/zinc_bond_anchored_marginal_field_v1/code/bamf_inference.py
sha256: 34024597b290451d931abe9d806d3497de1585722731bb09f21b1057ec39f2a7
```

不修改历史模块。新代码只新增一个包装函数：调用一次 `_exact_beliefs`，从同一组
calibrated clique beliefs 同时导出 `logZ` 与每个节点的两种状态 `logw`。不得分别
调用两次完整推断。允许 min-fill、clique plan、upward/downward logsumexp 和
scatter/gather；节点/边势必须可微，最终预测梯度必须回传到两种势网络。

这是精确概率图推断，不是学习式消息传递。

---

## 3. 冻结的势网络

| 项 | 定义 |
|---|---|
| channels | 16 |
| states | 2 |
| 原子字段 | 9 个，各自 `Embedding(d, 16)` |
| 原子输入 | concat 9 个 embedding = 144 |
| 原子势 MLP | `144 -> 64 -> 32`，隐藏层 SiLU |
| 原子势输出 | `2 * tanh(raw)`，reshape `[N,16,2]` |
| 键字段 | 3 个，各自 `Embedding(d, 12)` |
| 键输入 | concat 3 个 embedding = 36 |
| 键势 MLP | `36 -> 64 -> 48`，隐藏层 SiLU |
| 键势输出 | `2 * tanh(raw)`，reshape `[M,16,3]` = `J00,J01,J11` |
| 对称性 | `J10 = J01` 由构造强制 |
| 词类型参数 | 无 |

全部参数从 seed 0 随机初始化（PyTorch 默认初始化，局部 RNG 隔离），从头训练。

---

## 4. 冻结的句子与整分子读出

对每个原子：

```
a_v       = concat(9 atom embeddings)              # 144
eta_norm  = LayerNorm(eta_v)                       # 16
h_v       = MLP([a_v, eta_norm])                   # 160 -> 128 -> 96
```

`h_v` MLP：`Linear(160,128) -> SiLU -> Dropout(0.1) -> Linear(128,96) -> SiLU`。

每张图对 96 维 `h_v` 取 sum、mean、population std（`ddof=0`），得到 288 维。
再拼接：

| 项 | 维 |
|---|---:|
| `h_sum, h_mean, h_std` | 288 |
| `H(G)` | 16 |
| `H(G)/(1+n_physical_bonds)` | 16 |
| `log1p(n_atoms)` | 1 |
| `log1p(n_physical_bonds)` | 1 |
| **合计** | **322** |

读出：

```
322 -> 128 -> 64 -> 1
```

`Linear(322,128) -> SiLU -> Dropout(0.1) -> Linear(128,64) -> SiLU -> Linear(64,1)`，
再加一个独立标量 bias。输出是标准化标签的预测值。不增加其他全图统计、环特征、
路径编码、额外池化方式或自由模型版本。

---

## 5. 训练配方（冻结）

| 项 | 值 |
|---|---|
| seed | 0 |
| optimizer | AdamW |
| peak lr | `3e-4` |
| min lr | `1e-5` |
| weight decay | `1e-4` |
| batch | 64 molecules |
| warmup | 5 epochs（按 step 线性） |
| schedule | cosine |
| max epochs | 200 |
| min epochs | 80 |
| early stopping | patience 40，按 Valid RMSE |
| gradient clip | 5.0 |
| loss | Train 标准化标签 MSE |
| precision | float32，不启用 AMP |
| best checkpoint | 严格按 Valid RMSE 最小值（并列取更早 epoch） |

每个 epoch 在 `model.eval()` 下完整评估 Train、Valid、Test，保存三组 RMSE/MAE。
Train 指标必须来自完整 Train 的评估模式前向，不能用训练中 dropout 激活的 batch
loss 代替。

本次明确允许每个 epoch 查看并保存 Test 指标；Test 不参与梯度、标准化、参数选择、
early stopping 或成功判定。不得因为 Test 结果重新训练或改架构。

---

## 6. 运行前工程验证

全部通过才允许正式训练：

1. 1–7 节点的树、链、环、分支、孤立节点、多连通分量：精确 `logZ`、节点后验
   对数几率对 brute-force 枚举，float64 误差目标 `1e-8`。
2. 全部边势为零：`H`、`eta` 数值接近零。
3. 原子重编号、键方向反转、键存储顺序变化：预测不变（浮点容差内）。
4. 训练与评估使用同一个数学算子；无 train/eval 特征定义差异。
5. 一个真实 Train batch：原子势、键势、句子 MLP、读出网络都有有限且非零梯度；
   optimizer step 正常执行。
6. Train/Valid/Test 尺寸、标签有限性、归一化来源、指标还原检查。

另外检查：单次 `exact_responses` 只调用一次完整 belief 计算。

不开展超出这些项目的大型审计。

---

## 7. 全量拓扑与算力门槛

对全部 4200 张图建立 min-fill 消元计划，记录诱导宽度 median、p95、p99、max，
以及最大 clique 状态表规模。

**若存在诱导宽度 > 8 的图：停止本轮，报告 `BLOCKED`，不得忽略该图，不得改用
近似计算。**

在 `res` 上进行不超过 2 epoch 的真实 GPU 预检（真实训练反向 + Train/Valid/Test
完整评估），按实测时间预估 200 epoch 成本。若预估超过 **2 GPU·h**，停止正式训练
并报告成本；不得自行缩小模型、改批大小或改配方来凑预算。预检使用 seed 0，
不使用额外 seed，不是超参数搜索。

---

## 8. 判定与停止

以 best-Valid checkpoint 的官方 Valid RMSE 为主：

| Valid RMSE | 标签 |
|---|---|
| `<= 0.65` | `PREDICTIVE_CANDIDATE` |
| `(0.65, 0.70]` | `WEAK_PREDICTIVE_SIGNAL` |
| `> 0.70` | `NO_COMPETITIVE_SIGNAL` |
| 数据/精确计算/梯度/资源检查失败 | `BLOCKED` |

这些只是本次单 seed 的工程决策标签，不是统计显著性结论。描述性参考：仓库官方
划分历史 GINE Valid RMSE 约 0.61–0.64，Test RMSE 约 0.66–0.67；训练配方不同，
不能宣称匹配对照。D0 RICE/R2 数值不得与本轮官方 Valid/Test 直接比较。

不自动重复实验、不调学习率、不加宽通道、不执行第二 seed、不做模型级消融、额外
诊断臂、bootstrap、图词解释、D0/D1 或继续搜索。

---

## 9. 交付物

```
PROTOCOL.md                 本文件
configs/fgcr.json           全部固定超参数
code/                       实现、数据、检查、训练、拓扑预检
analysis/                   正确性、拓扑宽度、GPU 预检成本、运行结果
results/                    best-Valid checkpoint、每 epoch 指标、完整预测
REPORT.md                   简洁结果报告
```

报告必须记录：最佳 Valid epoch 与三组 RMSE/MAE；每 epoch 三组曲线数据；总参数
数、GPU 时间、峰值显存、精确推断成本；全图覆盖；`H`/`eta` 非退化与梯度；
与历史官方强基线的距离；模型证明了/没有证明什么；`CONTINUE_DISCUSSION` 或
`STOP_THIS_IMPLEMENTATION`。不得宣称已实现自主图词发现、因果解释或已确认的
原创方法。

完成一轮后立即停止，不产生下一版模型。