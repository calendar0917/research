# cscl-prediction-feasibility-v2 — 预测可行性验证（Full-Y 强预测骨架，seed 0）

Track: `structure-contribution` · 日期: 2026-10-10 · 协议: `configs/prediction_feasibility_v2.yaml`（冻结）
代码: `code/run_prediction_feasibility_v2.py`（prep/train/compare）、`code/analyze_feasibility_v2.py`（analyze/topo-audit）
测试: `tests/test_prediction_feasibility_v2.py`

> 本轮只回答一个问题：在当前 CSCL 固定数据划分下，已有强非 GNN 结构预测骨架
> （历史 Full-Y）以原始 y 为唯一任务监督，能达到什么预测水平——
> 以判断下一步是研究结构贡献分解，还是继续解决预测表示问题。

---

## 1. Phase-1 CPU 审计：0.332 的误差结构（`results/feasibility_v2_cpu/v1_error_structure.json`）

来源：cscl-v1 修正后的原始单位逐分子导出 `cscl_v1_gpu/dev_preds_raw_units.npz`
（**注意**：per-arm `dev_preds_{arm}_s0.npz` 是已废弃的标准化单位导出，不可用；
REPORT §6 已记录）。行序/标签与数据集逐位核对一致。

### 1.1 误差分布（dev 2000，raw-y 单位）

| 臂 | MAE | RMSE | 中位数 | p95 | max |
|---|---:|---:|---:|---:|---:|
| O-rich | 0.33206 | 0.50738 | 0.24577 | 0.88041 | 6.5212 |
| O-unit | 0.35708 | 0.60472 | 0.23883 | 1.02310 | 6.1613 |

重尾结构：top-1/5/10/20/50 误差分子占总 |e| 质量
O-rich 0.98% / 3.53% / 5.82% / 8.60% / 14.35%（O-unit 0.86/3.89/6.66/10.76/18.34%）。
两臂逐分子误差相关仅 **ρ=0.215**——它们在明显不同的分子上失败。

### 1.2 目标分布与低 y 尾部是主要误差源

dev y：mean 0.031 / std 1.872 / p1 −5.93 / min −9.12（全 10k min −42.04）。
corr(|e|, y) = −0.298（O-rich）/ −0.333（O-unit）：**误差集中在目标左尾**。

| 子群 | n | y 均值 | MAE O-rich | MAE O-unit |
|---|---:|---:|---:|---:|
| y ≤ q25（≤−0.99） | 500 | −2.58 | 0.4508 | 0.5760 |
| q25–q50 | 500 | −0.19 | 0.3190 | 0.3336 |
| q50–q75 | 500 | 0.92 | 0.3028 | 0.2823 |
| q75–q90 | 300 | 1.68 | 0.2456 | 0.2381 |
| q90–q99 | 180 | 2.34 | 0.2754 | 0.2285 |
| y > q99 | 20 | 3.11 | 0.2262 | 0.2840 |
| **y ≤ −3** | **128 (6.4%)** | −4.56 | **0.7169** | **0.8631** |

y ≤ −3 的 6.4% 分子贡献约 **14%** 的误差质量；剔除后 O-rich MAE 0.306。
top-20 误差分子画像（O-rich）：y 均值 −3.16、40% 有 y≤−3、85% ring_units≥2
（即大分子重尾目标，不是"小分子/少环"群）。

### 1.3 拓扑分组与混杂控制

cyclomatic（总环数）：0–1 环分子 n=12/156 上 O-rich 明显更好（0.395/0.321 vs
O-unit 0.628/0.427）；≥3 环后 O-unit 反超（3 环 −0.004，4+ 环 −0.002）。
ring_units：0–1 环系分子（n=245）差距 +0.23，≥2 环系（n=1755）基本持平。

**混杂控制（ring_units × y-band 交互）**：同 y 带内 O-unit 仍系统性更差
（y>−1 带内 ru01: 0.329 vs 0.445；y≤−3 带内 ru01: 0.525 vs 1.168）——
v1 的"0–1 环系缺口"**不是** y 分布混杂的假象；但同时 ru01 子群的绝对误差
水平确实因其 y 分布偏负而偏高（ru01 ∩ y≤−3 n=30，MAE 0.53）。

原子/键类型与 |e| 相关全部很弱（|ρ|≤0.10，Cl 最大 +0.099）。

### 1.4 结论（对应最终问题 1）

0.332 的误差不是均匀的"表示噪声"：(a) **低 y 重尾**（6% 分子贡献 14% 误差）
是首要结构；(b) 剩余部分与拓扑复杂度弱相关；(c) 两臂失败模式相关性低，
说明 0.332 附近存在多条可行的表示路径，瓶颈不在"单元 vs 静态特征"的选择。

## 2. Phase-1 topology25 审计（`results/feasibility_v2_cpu/topology25_audit.json`）

1. **O-rich 的 base573 第 412:437 块（topology25）确认全零**：160 行抽样
   max|·| = 0。`run_cscl_v1.py::rich_features_573` 调用
   `typed_cycle_probe_v1.graph_feature_row(data)` 未传 `topology_row`，
   默认零向量。
2. **存在已核验、无标签的真实 topology25**：
   `tracks/ksvd/results/zinc_topology_cache/train_topology_features.csv`
   （10000 行，"hinge" 模式 25 维 = 环谱 + 最长简单环 L + L² + ReLU(L−3..10)
   + mcb 统计 + cycle_rank，纯图不变量；缓存构建时已做置换稳定性核验
   mcb_perm_consistent=1.0）。与 O-rich 缺失的 25 维**同一定义**。
3. **与历史 Full-Y 输入同源**：encoded cache 的 `topology_features` =
   10k-fit standardize(hinge25)；反演后与 raw hinge25 最大差 6.1e-5
   （float32 舍入量级）。即历史 Full-Y 消费的 topology25 就是该定义。
4. **fit-only 纪律可满足**：hinge25 本身无样本集拟合；只需在 fit_inner 上
   重新拟合 standardizer（本轮 prep 已实现）。

**结论（对应最终问题 2）**：O-rich 的拓扑信息确实完全缺失，且补齐它不需要
任何标签——但本轮不训练 O-rich+topo 变体（协议禁止追加候选）。

## 3. Phase-2 门槛审计：历史 Full-Y 的合法性与污染风险

### 3.1 非 GNN 计算图审计（读真实 forward，非类名）

`LatentScaleSEM108`（`e2e_dictenv_scale_v1.py`，MRO 含 AuditModel；
masked forward = `AuditModel.forward(mask=C6_MASK)` 逐行核实）：

1. `coord = code(dict_phi)`：CSSD——`[c/rms; IHT10(D⊥, r)]`，per-patch 行内
   10 步 IHT，无邻居交互；
2. node/edge slots：**单轮** incidence 池化（`index_add_` 一次）把静态编码的
   shell 出现聚合为每原子角色槽（输入是静态结构编码，不是传播后的可训练
   节点状态；无任何 write-back）；
3. fusion MLP → 每行 16 步 unrolled ISTA bridge（`LatentDictionaryBridge`，
   不接触 edge_index/graph_id，行内求解）；
4. graph 级 moment 池化（均值/方差/计数），**静态预计算** pair index +
   结构关系描述子（`pair_relation` 为原始结构量）+ 距离桶门控 → pair MLP
   → 桶内 moment 池化；
5. 全局 62 维静态 context 编码（C6_MASK 置零 atom/bond histogram 组）；
   topology25 编码（**未置零**）；
6. reader MLP → 标量。

**判定：不违反非 GNN 约束**。无消息传递、无邻居隐藏状态迭代、无注意力；
所有迭代（IHT/ISTA）都是单对象稀疏求解器。属于允许清单中的
"显式图结构统计 + 固定/训练结构字典 + 静态局部特征共享编码 + 非迭代
角色绑定 + 显式节点对结构关系 + 汇总后非线性 reader"。

### 3.2 数据污染审计与 fit7200 重建

历史缓存 `encoded_train.pt`/`env_train.pt` 的 patch/ctx/topo/anchor 已按
**全 10k 官方 train** 统计标准化（来源核实：`sdp.prepare_encoded` 的
fit_records=官方 train 10k；anchor_stats fit_split="official train"）——
在当前内部划分下 monitor 800 + dev 2000 包含其中，**直接复用即泄漏**。

本轮处理（`build_prep_fit7200`）：

1. 用 git-tracked 历史 `all_train_prep.npz` 的 `*_all_*` standardizer 作
   **反演基**恢复 raw（任务书明确允许"来源和可逆性验证后恢复"）；恒等性
   检查：inverse → 全 10k 行 refit → 与存储统计比较，非退化列 gap
   实测 ≤1.7e-8（容差 1e-5；anchor 退化列按历史记录单独放行）。
2. **全部**新拟合对象只在 fit_inner 7200 上拟合：patch/ctx/topo/anchor
   standardizer（Std.fit floor 1e-6）、K-SVD `D_fit`（atoms 32 / s 8 /
   epochs 10 / seed 20260924，仅 fit_inner phi 行）、公共子空间 U（q=1，
   仅 fit_inner phi 行）。phi/环境 incidence/pair_relation 均为纯结构量。
3. typed/parent 词表（缓存中存在、10k 拟合）**不进入模型输入**；
   测试断言模型模块图中无任何 embedding/消息传递类消费它们。
4. 无 `target_decomposition.npz`/`g`/`ell`/`s`/`c` 加载（AST 级测试钉住）；
   监督 = raw y L1 + 无标签结构重建辅助（λ=33.9587 冻结常数，目标仅为
   phi 残差——测试证明该损失与 y 无关）。
5. 官方 valid/test 全程未加载；行对齐核查（cscl row i ≡ encoded row i，
   y 与原子数全 10000 行逐位一致）。

## 4. Phase-4 正确性测试（`tests/test_prediction_feasibility_v2.py`，13 项）

seed-0-only；decomp/heldout 引用的 AST 级扫描；协议 yaml 冻结内容；
划分不变性（7200/800/2000、互斥、SMILES 组不跨集）；prep 恒等检查 +
fit7200 ≠ 10k 统计；模型 408,651 参数审计 + 无消息传递类；forward 形状/
有限性 + **y 独立性**（改 y 预测不变）；batch 组成/顺序不变性；save/load
逐位一致；训练步损失有限 + 参数更新；C6_MASK 不置零 topology；
重建损失标签无关。

本地 CPU smoke（512 fit 行 ×3 epochs）：prep 恒等检查通过、K-SVD 12s、
训练下降（mon 1.31→1.10）、三种选择规则导出正常。

## 5. Phase-5 正式 GPU 实验（res / GPU1 / pool res-gpu1）

（正式结果见 §6，运行记录 `results/feasibility_v2_gpu/`。）

## 6. 结果

（待正式运行后填写。）
