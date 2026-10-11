# cscl-strong-nongnn-predictor-v1 — 静态三体是否突破 Full-Y 内部 dev 0.11641

Track: `structure-contribution` · 日期: 2026-10-11 · 协议: `configs/strong_nongnn_predictor_v1.yaml`（冻结）
代码: `code/run_strong_predictor_v1.py`（prep/cpu-gates/replay/train/analyze）、`code/static_triplet_operator.py`（唯一新模型）
测试: `tests/test_strong_predictor_v1.py` · Phase A CPU 审计: `code/phase_a_strong_predictor_v1.py` → `results/strong_nongnn_predictor_v1/phase_a_s0.json`

> 本轮只回答一个问题：在**不做** α/δ/γ 分解、不引入 GNN/Transformer/消息传递的前提下，
> 给历史强非 GNN 骨架 Full-Y 加入一个**一次性的、非迭代的三体结构组合**算子后，
> 内部 dev MAE 能否跨过预冻结判断带（≤0.095 / ≤0.106 / >0.106），
> 即三体联合信息是否是 Full-Y 现有 forward 不可表达的新增能力。

---

## 1. Phase A：CPU 瓶颈诊断与历史失败审计（无 GPU、无训练）

来源：`results/strong_nongnn_predictor_v1/phase_a_s0.json`（只读历史导出，
official valid/test 未加载）。

### 1.1 Full-Y 内部 dev 误差结构（soup, seed 0）

| 臂 | fit | monitor | dev |
|---|---:|---:|---:|
| Full-Y soup | 0.04220 | 0.11826 | **0.11641** |
| O-bucket bucket 对照 | 0.03641 | 0.11794 | 0.11610 |

- 训练曲线：fit 持续下降到 0.0652@240，monitor 在 0.123–0.155 平台震荡
  （best monitor 0.12263@238）——拟合-泛化 gap ≈ 0.074，瓶颈是
  **表示/泛化能力**，不是优化不足。
- dev 误差按目标分层：y≤−3（6.4% 分子）占约 17.3% 的误差质量；
  corr(|e|, y) = −0.30；低 y 左尾仍是最大误差源（与 v2 §1.2 一致）。

### 1.2 三体定义的数据验证（全 10k，标签盲）

- `pair_bucket == 0`（图距离 1）与 env 键列表去重复核：**0 失配、0 重复、
  0 自环**；最大度 4（ZINC 子集）。
- 三体数 = Σ_j deg(j)(deg(j)−1)/2：fit_inner 249,105 / monitor 27,800 /
  dev 69,453（dev 平均 34.7 个/分子，0 个零三体分子，所有分子三体 ≥1）。
- 三体是现有 forward 的**不可表达**新增量：Full-Y 的 `pool_moments` /
  `pool_pair_moments` 只做 [Σx, Σx², log count]，没有任何跨原子对乘积项；
  三体 (i,j,k) 的联合统计必须由新算子产生。

### 1.3 历史高阶层路线审计（tracks/ksvd，只读）

| 路线 | 形态 | 结果 | 对本轮的启示 |
|---|---|---|---|
| PSD-v0 | pair-state 编码器 | 失败 | 不用对配对状态做迭代编码 |
| GSCN-v0 | 字典核替换 | 被拒绝 | 不改字典核 |
| BAT-v1 | bond-anchored triples + reader | M_soup 0.12300 vs 门槛 0.120，无信号 | 仅"三体+reader"不够；本轮用 live E + bond-angle 语义 + 端到端联合 + 零初始化新列 |
| BAT-joint-tune | train 降 18%，valid 无改善 | 泛化不动 | 端到端优化的收益必须落到 dev，否则停止 |
| JointBond | denormal 塌缩、分支惰性 | 失败 | 新分支必须通过梯度活性 + 敏感性门 |
| RoleCorr / typed-cycle / TCCD | 机制有效但无绝对收益 | 停 | 机制证据 ≠ 精度收益；本轮以 dev MAE 为唯一判据 |

**本轮与 BAT 的关键差异**：三体读取的是训练中实时更新的 per-atom 环境向量 E
（不是冻结描述子），边类型由真实键给出（含闭合位），算子与 reader 端到端联合
训练，且新 reader 列严格零初始化保证初始函数与历史 Full-Y 逐位一致。

## 2. 模型（唯一新组件）：FullY-StaticTriplet

- 数据：每个分子取**真实化学键**集合（`pair_bucket==0` ≡ env 键去重），
  按 (lo,hi) 排序；对每个中心 j 枚举邻居的无序端点对 {i,k}，
  三元组 (j; i, k) 唯一、无自环、无双向重复。
- 特征（177 维）：`[h_i+h_k, |h_i−h_k|, h_i⊙h_k]`（h = 共享 MLP 投影
  E→32）+ `h_j` + 键型对称 one-hot 和 (4) + 键型对称外积 (16) +
  中心原子 one-hot (28) + 闭合位 (1)。
- 算子：共享 MLP 177→64→32（SiLU），对每个三元组输出 q∈R³²。
- 汇总：每分子 `[Σq ; Σq²]`（64 维）拼到原 reader 输入（814→878）。
- 端点为**无序**：交换 (i,k) 与两条键类型，q 严格不变（测试钉住）。
- 参数：三体算子 18,112 + reader 新列 2,496；全模型 429,259 =
  408,651（历史 Full-Y 参数逐位复制）+ 新增。
- 初始恒等：新 reader 列全零 + 新算子不动旧路径 ⇒ 初始预测与历史
  Full-Y 的差 ≤5e-5（实测 1.8e-7）。

三体算子内不含任何消息传递/attention：E 在每个 tripet 内只做**逐元素**
共享投影，汇总只做图级矩池化；无邻居状态迭代、无跨层传播。

## 3. 训练协议（与 Full-Y 完全一致，唯一一次正式运行）

- 数据划分：fit_inner 7200 / monitor 800 / dev 2000（冻结 v0 split）。
- 从头训练（非续训）；旧参数按历史 seed-0 构造逐位复制，新 reader 列零初始化，
  三体算子用独立私有 RNG（seed 20261011），不扰动父模型 RNG 流。
- 超参：raw y L1 + 披露的 label-free 重建辅助（λ=33.95873017865987），
  Adam lr 1e-3 / wd 1e-5 / clip 5 / bs 128 / 240 epochs / shuffle seed 101。
- 选择：top-5 monitor-epoch soup 为主指标；secondary best-monitor；
  tertiary last；auxiliary +fit-median-b 校准。dev 不参与任何选择。
- 计算：`res` GPU1（pool `res-gpu1`，进程内 cuda:0），经 `rr`，commit 冻结后部署。

## 4. 预冻结判断带（运行前写入，运行后不得改）

| 带 | dev MAE | 动作 |
|---|---|---|
| A | ≤ 0.095 | 强突破：报告并停止本线（不追 seed 1） |
| B | 0.095–0.106 | 相对 0.11641 有方向性收益：报告，不调参 |
| C | > 0.106 | 无信号：停止该分支，不做复现 |

## 5. 纪律与边界

- official valid/test 从不加载；seed 0 唯一；不训练第二模型；不 full10k 重拟合；
  不超参搜索；不重复正式运行。
- 三体分支若无法实现或与现有 forward 数学等价 ⇒ 停止 GPU 并报告（Phase A
  已证明"不可表达"；CPU gates 实测梯度活性/敏感性）。
- 解释研究用的计算溯源（原子/E/键/三体/汇总 npz）在本地 analyze 阶段导出，
  不进入正式选择。

## 6. 结果（正式运行后填写）

_待运行：唯一一次 seed-0 GPU1 训练（`FullY-StaticTriplet-seed0`）。_

## 7. 工程门（CPU，正式运行前）

- `cpu-gates`：初始恒等（1.79e-7 ≤ 5e-5）、共享参数逐位一致、新列全零、
  loss→新 reader 列梯度、第二步→三体模块梯度（env_proj/mlp 均 >0）、
  zero_summary 改变预测（|Δ| 0.012）、零列接线对照（1.8e-7）、随机列对照
  （0.123）、批组成不变性（6.0e-7）、y 独立（0.0）、save/load 逐位一致、
  非 GNN 静态扫描（无禁用类）、三体计数（16 分子 555 个）。
- `replay`：历史 soup 载入增强模型后与参考模型的 dev 预测差 ≤5e-5，
  且与历史保存的 dev preds 逐分子对比。
- 测试：`pytest tests/test_strong_predictor_v1.py`（快 14 + 慢 7）。

## 8. 变更记录

- 2026-10-11 初版：Phase A 审计完成；模型/驱动/测试写入；协议冻结。
