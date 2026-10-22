# zinc_shared_factor_v1 — 共享原子隐状态 + 小片段精确收缩（T）vs DensePhi（D）

协议 `zinc-shared-factor-v1`，配置 `configs/sf_atomic_v1.json`
（sha256 `d1f6b67fcd3339bedf0e278fbe09a5e2d57ac842d9b969465574a65123cdabc8`），
预注册见 `DESIGN.md`。本轮执行到**阶段 A 的门槛判定**并按协议停止，没有进入 B/C。

## 0. 结论

**属于第 (2) 类：方向支持 T 更好，但没到预注册的有效性门槛，按协议停在这里，不追加投入。**

- 唯一未过的门槛是绝对门槛：`best_valid(T,s0) = 0.077045 > 0.075`（差 0.0020）。
- 相对门槛过了：`0.077045 ≤ 0.081252 − 0.003 = 0.078252`；20-epoch 滑动中位数
  （0.078646 vs 0.082610）与 tail-100 中位数（0.078800 vs 0.082645）都支持 T。
- 但「相对门槛过」这件事本身不足以支撑结论：**这套 recipe 的 run-to-run 数值离散度下界
  就有 0.0026**（§5），而本轮 T−D 只有 0.0042。也就是说 T 的收益只比噪声大 1.6 倍，
  且在同一轮里 T 的绝对水平仍落在上一轮 DensePhi 的 0.0786 附近（T 只低 0.0016）。
- 因此：**共享原子状态 + 小片段精确收缩确实能降低 valid MAE，但幅度（≈0.002–0.004）
  小于预注册的「值得继续」标准，也小于或接近配方自身的噪声量级。** T vs N 的结构对照
  （收益是否依赖共享变量的连接关系）按协议没有执行，因此仍然没有答案。

| 可能的结论 | 本轮判定 | 需要的证据 |
| --- | --- | --- |
| (1) T 更好，值得继续 | ✗ | 过 A（含 ≤0.075）+ B 两 seed 方向重复 + T−N 对照 |
| **(2) T 更好，但不值得继续** | **✓ 本轮** | A 的相对门槛过、绝对门槛不过；收益 ≤ 噪声下界 ×1.6 |
| (3) T 不更好 | ✗ | 相对门槛也失败（本轮过了） |
| (4) 证据不足（预算/实现缺陷） | ✗ | 预算内 600 epochs 跑完、7/7 检查过、无实现缺口 |

## 1. 冻结项（未改动）

| 项 | 值 |
| --- | --- |
| 载体 / 接口 | P3 载体，198→128 zero-init `W_s`（`w_s_zero=True`，gate 已验证） |
| 数据 | `bf_data` ZINC subset train 10,000 / valid 1,000（S3/S4/F4 模式），未读 official test |
| 初始化 | 只加载 `pretrain_encoder_s0.pt`，sha256 `006fabc1…bfa97`（gate 已验证） |
| D init / content | `466d9481…` / `5c137f9f…`（与 `zinc_local_factorization_v1` 逐字节一致） |
| T=N init s0 / content | `703f7545…` / `db6b8d86…`（T 与 N 同参数，gate 逐一核对） |
| public state s0 | `21963b43…`（三臂相同） |
| 参数 | D 449,601 / T 451,777 / N 451,777（内容模块 12,928；T 比 D 多 0.48%） |
| recipe | raw-y L1，batch 128，600 epochs，79 steps/epoch → 47,400 updates；AdamW 1e-3→1e-5，wd 1e-5，warmup 500，per-step cosine，clip 5；全参数训练；无 EMA/soup/早停；按严格 `<` 取单一 best |
| 门槛 | A：`T ≤ D−0.003`、`T ≤ 0.075`、best20 中位数与 tail100 中位数均支持 T |

## 2. 实现

- **D**：直接继承 `lf_models.ContentBranch("D")`，`SFModel.forward` 在 D 分支上原样调用它；
  `sf_core.train_epoch` 是 `lf_core.train_epoch` 的数学等价副本（仅日志探针按 arm 分派）。
- **T**：`q_c(s|a)=softmax(A[28,C,R])`（C=16, R=4），`K_{b,c}(s,t)=exp(tanh S)`（对称上三角，
  3·C·R(R+1)/2=480 参数，全家族/位置共享）；在 log 空间精确枚举 `R^(n−1)`（4 节点 64、
  5 节点 256）并 logsumexp；无边/padding 用乘法单位 1（mask 乘 0 项），padding 不进 `m`。
- **N**：同参数/同输入/同 MLP/同初值，非根边每条独立复制状态
  `Π_root [Σ_t q(t|a_j)K(s,t)] · Π_nr [Σ_{u,v} q(u|a_i)q(v|a_j)K(u,v)]`；不从 T 推理时切开。
- **内容输入 133 维**：`q_c(s|a_root)` 64 + `log(T/N)/max(m,1)` 64 + family one-hot 3 +
  `n/5, m/10` 2 → `Linear(133,64)→SiLU→Linear(64,32)`（有 bias，无 dropout/LN/skip/辅助损失）。
  T/N 天然重排不变，不枚举置换；D 保留原精确置换平均。

为进预算做了**不改变数学定义**的实现优化（提交 `b145428`、`606ec03`、`2e4c5db`）：
按长度缓存的静态索引计划（只缓存结构/索引，不缓存任何学习表示）、通道前置布局、
`index_select` 取代 expanded`gather`、只计算被请求的 mode、把 `Σ_t log q` 折叠成一次 matmul。
等价性证据：优化前后 `cpu_checks.json` 的**检查数值除浮点末位外完全不变**（17 个单测 + 7/7
检查通过），2-epoch smoke 的 valid 曲线在第 1 个 epoch 完全相同（0.686550），第 2 个 epoch
只在第 5 位有效数字之后不同（0.714362 → 0.714414），差异来自 matmul 取代逐项加法后的求和顺序。
效果：T 从 5.84 s/epoch 降到 3.6 s/epoch（投影 4152 s → 2699 s，`plan.json` 记录
`feasible=True`），600 epochs 未被缩短（实际训练 2208 s）。

## 3. 训练前检查（`analysis/cpu_checks.json`，7/7 PASS，绑定同一 config sha）

| 检查 | 关键数字 |
| --- | --- |
| 精确枚举 vs 暴力（float64） | 4 节点/5 节点、T/N 全部 `max_abs_log_err ≤ 1.5e−15`（q 块 0.0） |
| 根保持重排 / occurrence 顺序 | 全部 0.0（T 与 N） |
| H1/H2 结构见证（预固定种子 20261022，不筛种子） | T：`log_block_max_abs_diff=0.0132`、`phi_rel_l2=7.6e−3`；N：全 0.0 |
| 纯星形 T==N | 2.2e−16（含孤立节点变体） |
| 类型路径 / padding 静默 | 改键类型只改对应 `K_b`；padding 逐位不变 |
| optimizer 覆盖 + `W_s` 打开后 q/K/phi 梯度 | 全覆盖；`w_s_norm` 0.23→6.54（D）/0.23→5.95（T） |
| checkpoint 重载一致 | 本机重算 valid：D 0.081251953 / T 0.077045402（与冻结值差 ≤5e−9） |
| content digest / provenance | D 的 init、forward、256 行 partial-epoch（online MAE 1.596254050731659）与 LF 完全一致；cache digest 与上一轮一致 |

## 4. 结果（阶段 A，seed 0）

| arm | params | best valid | best ep | final valid | best20 窗口中位数（窗口终点） | tail-100 中位数 | s/epoch | GPU-h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SF-Dense (D) | 449,601 | 0.081252 | 339 | 0.082611 | 0.082610 (587) | 0.082645 | 2.37 | 0.411 |
| SF-SharedState (T) | 451,777 | **0.077045** | 384 | 0.078846 | **0.078646** (528) | **0.078800** | 3.58 | 0.620 |

门槛：`margin` ✓（0.077045 ≤ 0.078252）、`absolute` ✗（0.077045 > 0.075）、
`window_median` ✓、`tail100_median` ✓ → **阶段 A 不通过，不进入 B/C**（`analysis/gates.json`）。
方向稳定性：ep ≥ 100 之后 500 个 epoch 中 426 个 T < D（85%），best20 中位数在 528 号窗口最小，
不是单个孤立最低点。

分子级（best checkpoint，1000 个分子，配对 bootstrap 10,000 次）：

| 对比 | ΔMAE | 95% CI（分子抽样） | 改善分子占比 | median ΔAE |
| --- | --- | --- | --- | --- |
| T − D | **−0.004207** | [−0.008219, −0.000284] | 0.526 | −0.002131 |

分组（radius-2 seen/unseen × ordinary/tail；分组来自 `zinc_restricted_adaptation_v1` 冻结参考）：

| 单元 | n | D MAE | T MAE | 对总 MAE 的贡献 D → T |
| --- | --- | --- | --- | --- |
| seen, ordinary | 710 | 0.0510 | 0.0487 | 0.03623 → 0.03459 |
| seen, tail (y ≤ −3) | 39 | 0.1605 | 0.1472 | 0.00626 → 0.00574 |
| unseen, ordinary | 225 | 0.1359 | 0.1339 | 0.03058 → 0.03012 |
| unseen, tail | 26 | 0.3149 | 0.2537 | 0.00819 → 0.00660 |

改善分布在全部分组，最大相对降幅在 unseen+tail（−19%），但 n 只有 26；seen+ordinary
（71% 的分子）只有 −4.5%。**不集中于单个极端样本**：贡献最大的单分子只占 Σ|ΔAE| 的 2.3%，
top-1% 分子占 10.4%；最大的几个改善确实落在极端负标签（y=−20.34：AE 2.596→1.760）。
排掉 y ≤ −3 的 65 个分子后，剩余 935 个分子的 MAE 为 D 0.07145 / T 0.06921（Δ=−0.00224）。

## 5. 关键 caveat：0.003 的 margin ≈ 这套 recipe 自身的噪声下界

`zinc_local_factorization_v1` 的 LF-D（best 0.078613@455）与本轮 SF-D（0.081252@339）
是**同一数学**的两次独立运行：init hash 相同（`466d9481…`）、600/600 epoch 的 batch order
hash 完全相同（`order_hashes.json`，generator seed 101，n=10000）、600/600 epoch 的 lr 完全相同、
epoch-1 的 online MAE 只差 1e-7。它们的 best-valid 却差 0.0026 —— 只能来自 GPU 浮点非确定性
（归约/原子累加顺序）在 47,400 步里的放大。

推论（写进结论而不是脚注）：

1. 本轮 T−D = 0.0042 只有一个 seed、一个 D draw，**只比该噪声下界大 1.6 倍**；
2. 若把 D 换成上一轮的 draw（0.078613），T−D 只有 0.0016，**连 0.003 的 margin 都过不了**；
3. 所以「T 更好」这个方向在本轮数据上成立（4 个 epoch 级指标 + 分子级 CI 都同向），
   但「好到值得继续」不能由单 seed 支撑。协议要求 0.003 margin 与两 seed 重复正是为此；
   本轮连 A 的绝对门槛都没过，因此没有花 B/C 的预算。

## 6. 训练后诊断（辅助证据）

- 学到的原子状态接近均匀：`q` 熵 1.3058 / ln4 = 1.386（94%），`q_max_prob_mean` 0.383，
  argmax 在 4 个状态上分布 [103,138,107,100]（28×16=448 组）。**不对称性主要落在 K 上**：
  `K_offdiag = 1.026 ± 0.378`（范围 0.441–2.558），`max|S| = 1.73`，`tanh` 未饱和。
  即 T 的收益更像是「边势提供了 q 之外的成对耦合」，而不是「每个原子分到了不同状态」。
- **ZINC subset 的 `bf_data` 模式里只出现 2 种键类型**（`pat_pair ∈ {1,2}`），
  `K` 的第三个 type slot（80 个参数/臂）在整个训练中拿不到梯度。这是本数据集的属性，
  检查里用合成模式的 type 3 仍然通过，但真实数据上它是死的。
- 成本：T 3.58 s/epoch vs D 2.37（1.51×），显存 308 vs 315 MiB；MAE 相对降 5.2%。
  以「每 GPU-秒的 MAE 降幅」算，本轮 0.0042/0.21 GPU-h ≈ 0.02 MAE/GPU-h，边际效率很低。

## 7. 与既有工作的关系（核对，不声称首创）

- **GCKN / graph convolutional kernel networks**（Chen et al., 2020）：以 walk 为单位的
  kernel 聚合，把路径核作为聚合权重。这里不是 walk kernel，而是 4–5 节点**诱导子图**上
  log 空间精确枚举的配分函数，节点状态共享（T）或被复制（N）。
- **Deep Homomorphism Networks**（Nguyen et al., 2021）：用（子）图同态计数当作特征来提升
  表达力。本轮的 T 正是「带权同态配分函数」的一个可微、可按根分解精确计算的实例
  （R=4 状态、树上精确消除），动机与 DHN 一致；但没有做 hom 计数向量化，也没有在 MP 层里
  做子图匹配，N 只是同一配分函数丢掉共享一致性的对照，不是近似推理方法。
- **加权图同态配分函数**（Lovász 框架）：`Z(θ)=Σ_φ Π_edges θ`；T 是「根分解 + 精确消除」
  的特例，成本 = `R^(n−1)`（本片段的 n ≤ 5），因此可以在每个片段上精确算；N 去掉共享后
  退化成逐边因子的乘积，这解释了为什么 N 是「共享是否重要」的正确对照组。
- **不声称**：不是首创因子化/局部配分函数（LF、P3 已在此仓库内做过），不是新的结构组合原理，
  也不是「无消息传递」（P3 载体本身就是一个 MP 编码器，只是 `W_s` 从 0 打开）。

## 8. 判断变化、是否值得继续、还缺什么

**改变了什么判断**：从「显式共享原子状态可能带来 >0.003 的可测收益，并值得用两 seed 去确认」
变成「收益方向真实但幅度 0.002–0.004，落在配方噪声量级内，且绝对水平（0.077）没有跨过
上一轮 DensePhi 所在的 0.0786–0.0813 区间下沿之外」。也就是说：**T 没有把这条 P3/LF 线索
推到新的水平，只是把 DensePhi 的 draw 分布往下挪了一点点。**

**是否值得继续**：按预注册门槛，不值得追加 B/C。若之后仍想推进，只有三条更省的路，
且都不在本轮冻结范围内：(i) 先量化噪声（同配置 2–3 次重跑）再重设 margin，否则任何
<0.005 的对比都不可判；(ii) 直接补 N（结构消融）而不重训 D，因为 D 已有两个独立 draw；
(iii) 提高信号强度（更多状态/通道，或在 5 节点上把共享状态与更细的因子分解结合），
但那需要新的预注册与新预算。

**仍缺的证据**（按重要性）：
1. **seed 方差**：只有 seed 0，无法区分「T 更好」与「这次 draw 更好」；
2. **T vs N 的结构对照**：收益是否依赖共享变量的连接关系，本轮没跑（A 未过）；
3. **0.075 以下的绝对水平**：T 最好 0.077045，从未在任何 epoch 低于 0.075；
4. **D 的可重复性**：只有两个 draw（0.078613 / 0.081252），噪声分布未知。

## 9. 预算账目（硬上限 6 GPU-h）

| 项 | GPU-h | 说明 |
| --- | --- | --- |
| 训练前 smoke / profiling / 失败重试 | ≈0.05 | D smoke 2 ep、T smoke 4 次（其中 3 次失败：2 次预算不可行、1 次设备不匹配，均计入）、3 次 profiling |
| SF-D_s0（600 ep，含 4 ep timing + 打分） | 0.411 | wall 1478 s，正式训练 1463 s |
| SF-T_s0（600 ep，含 4 ep timing + 打分） | 0.620 | wall 2231 s，正式训练 2208 s |
| **合计** | **≈1.08 / 6** | 每臂均 ≤1 GPU-h，600 epochs 未缩短 |

## 10. 复现

```bash
# 训练前检查（CPU，7/7）与单元测试（17 个）
uv run python tracks/structural-sufficiency/zinc_shared_factor_v1/code/sf_checks.py
uv run python -m pytest -q tracks/structural-sufficiency/zinc_shared_factor_v1/code/tests

# 训练（远端 res 的两张卡；本仓库统一用 rr 控制面）
rr deploy res
rr run res sf-A-D-s0 --pool res-gpu0 -- uv run python \
  tracks/structural-sufficiency/zinc_shared_factor_v1/code/sf_runner.py run --arm D --seed 0 --device cuda:0
rr run res sf-A-T-s0 --pool res-gpu1 -- uv run python \
  tracks/structural-sufficiency/zinc_shared_factor_v1/code/sf_runner.py run --arm T --seed 0 --device cuda:0

# 拉回并分析（生成 summary.json / gates.json / main_table.md / preds_valid.csv）
rr pull sf-A-D-s0 --path tracks/structural-sufficiency/zinc_shared_factor_v1/runs/SF_D_s0
rr pull sf-A-T-s0 --path tracks/structural-sufficiency/zinc_shared_factor_v1/runs/SF_T_s0
uv run python tracks/structural-sufficiency/zinc_shared_factor_v1/code/sf_analysis.py
```

产物：`analysis/{summary.json,gates.json,main_table.md,preds_valid.csv,cpu_checks.json}`、
`analysis/SF_{D,T}_s0/{freeze.json,init.json,plan.json,timing.json,preds_valid_{best,final}.csv}`。
checkpoint 与逐 epoch 曲线在远端 `runs/SF_{D,T}_s0/`（本地 `runs/` 被 `.gitignore` 排除，
`freeze.json` 里记有 `best/final/last` 的 sha256，可用上面的命令重算：
`best.pt` 重载后 D 0.081251953、T 0.077045402）。**未读 official test，也没有 test 预测。**
远端 `res` 当前部署在本报告的提交上；远端 runner 生成的 `analysis/SF_*_s0/` 副本与仓库内版本
相同，已被移到远端 `.remote-untracked-backup-20261022c/` 以免挡住下一次 `rr deploy`。
