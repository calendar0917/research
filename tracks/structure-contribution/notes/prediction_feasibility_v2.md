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

- run: `feasibility-v2-full-y-s0-20261010-225208-9006f42b`，commit `a0f0f7ed7cfb`，
  pool `res-gpu1`（CUDA_VISIBLE_DEVICES=1，进程内 cuda:0），A100-SXM4-40GB，
  driver 550.163.01，torch 2.5.1+cu124，exit 0。
- GPU smoke（先行，declared smoke）：同 commit，512 行 ×3 epochs，恒等检查与
  D_fit 哈希与本地逐位一致；不作为任何调参依据。
- 正式运行前无任何 dev 评估；训练配方 = 协议 yaml 冻结常数。

## 6. 结果（`results/feasibility_v2_gpu/`）

### 6.1 主结果（seed 0，内部 dev raw-y MAE）

| 选择规则（冻结） | monitor MAE | **dev MAE** | fit MAE |
|---|---:|---:|---:|
| **主：monitor top-5 soup（epochs 205–239）** | 0.11826 | **0.11641** | 0.04220 |
| 副：best-monitor 单检查点（ep 205） | 0.12263 | 0.12102 | 0.05497 |
| 末态（ep 240） | 0.13134 | 0.12112 | 0.05866 |
| 辅：soup + median-fit 校准 b（b=−0.0232，fit-only） | — | 0.11452 | — |

**对照（冻结参照，未重训）：O-rich seed 0 = 0.33206。**
配对差 Full-Y − O-rich = **−0.21565**，分子级 bootstrap 95% CI
**[−0.26895, −0.24133]**（2000 次，seed 0；同一 seed 内 dev 重采样，非跨 seed
训练不确定性证据）。81.0% 的 dev 分子上 Full-Y 更好，两臂误差相关 ρ=0.44。

- 训练：240 epochs 固定（无 early stop），5.1 s/epoch，wall 1231 s，
  408,651 参数（审计精确匹配）。
- 监督：raw y L1 + 无标签结构重建辅助（λ=33.9587 冻结；rec 项从 4.3e-4
  降至 2.7e-5，重建接近饱和）。
- 训练曲线（monitor MAE）：ep1 0.571 → ep20 0.218 → ep60 0.181 →
  ep120 0.138 → ep205 0.126（best）→ ep240 0.131。train task MAE 单调降至
  0.065（fit MAE 0.042），无发散、无非有限值。
- `full_y_seed0_dev_preds.npz`（row/pred/pred_cal/y）、monitor preds、
  soup state（1.6 MB）已随结果目录保存。

### 6.2 子群分解（Full-Y vs O-rich，dev raw-y MAE）

| 子群 | n | Full-Y | O-rich | 差 |
|---|---:|---:|---:|---:|
| 全部 | 2000 | 0.1164 | 0.3321 | −0.2156 |
| cyc [0,1) / [1,2) / [2,3) / [3,4) / [4,∞) | 12/156/664/759/409 | 0.115/0.123/0.103/0.098/0.170 | 0.395/0.321/0.295/0.310/0.435 | −0.279/−0.198/−0.192/−0.213/−0.265 |
| ring_units [0,1) / [1,2) / [2,3) / [3,∞) | 12/233/973/782 | 0.115/0.176/0.111/0.106 | 0.395/0.364/0.323/0.333 | −0.279/−0.188/−0.213/−0.227 |
| n_atoms [0,15) / [15,22) / [22,30) / [30,∞) | 44/659/1129/168 | 0.213/0.114/0.105/0.180 | 0.387/0.322/0.324/0.413 | −0.175/−0.208/−0.219/−0.232 |
| y≤−3 / y(−3,−1] / y(−1,1] / y>1 | 128/371/802/699 | 0.315/0.132/0.110/0.080 | 0.717/0.360/0.317/0.264 | **−0.402**/−0.229/−0.207/−0.185 |

读法：**每个子群都一致改善**；最大改善恰好在 O-rich 最弱的两处——低 y 重尾
（y≤−3：−0.40）与高环复杂度（cyc≥4：−0.27）。v1 中 O-unit 对 O-rich 的
0–1 环系缺口（+0.23）在 Full-Y 上完全消失（0.115 vs 0.395）。
Full-Y 的残余误差仍集中于 y≤−3 尾部（top-20 误差分子 y 均值 −3.94，
45% 有 y≤−3；这些分子上 O-rich 同样失败，MAE 1.67）。

### 6.3 fit-only 纪律证据（`results/feasibility_v2_gpu/prep_fit7200_meta.json`）

- 划分 SHA256（与本地逐位一致）：fit_inner `ff1fb939…` / monitor `83501240…`
  / dev `53fe48f8…`（完整哈希见 meta）。
- 反演恒等检查（全 10k 行 refit vs 存储统计，非退化列）：patch 2.2e-8 /
  ctx 9.3e-9 / topo 7.5e-8 / anchor 1.7e-8（容差 1e-5；anchor 退化列单列）。
- D_fit（K-SVD，166,555 个 fit_inner phi 行，atoms 32 / s 8 / epochs 10 /
  seed 20260924）sha256 `e8b6c8a1…`；U（q=1）同源拟合。
- 监督=raw y；`g/c/ell/s` 与 decomp 文件零引用（AST 测试）；重建辅助目标
  仅依赖 phi（测试证明与 y 无关）；official valid/test 未加载。
- monitor/dev 仅用于选择/评估，不进入任何拟合对象；校准 b 仅用 fit y
  （辅助披露，非主指标）。

## 7. 科研判断（任务书 §8.3 规则）

**情况 A 成立**：Full-Y 在相同划分、fit 仅 7200 行、y-only 监督下
0.11641，远优于冻结 O-rich 参照 0.33206（配对 CI 远离 0），且与历史
official-valid 口径的 ~0.117 量级一致——说明历史强预测能力并非依赖旧协议
的辅助监督或更大拟合集，而是来自骨架本身的计算结构（结构字典 + 单轮静态
聚合 + 稀疏编码 bridge + 静态关系路径 + 非线性 reader）。

因此：**我们已拥有一个足够强的非 GNN 预测基础**。下一轮最值得研究的问题
是任务书§8.3-A 所指的方向：如何把这种强计算中的局部结构、结构属性和关系
预测信息，以跨分子共享且可追溯的方式归属到结构单元（贡献分解），
而不是继续解决预测表示问题。

补充证据（本轮产出，服务下一轮设计）：
1. 0.332 水平的误差主要是**低 y 重尾 + 目标分布形状**问题（§1.2），
   不是拓扑表示缺口；Full-Y 把尾部误差从 0.72 压到 0.32，说明其优势
   恰好在最难预测的目标区间。
2. O-rich 的 topology25 确实完全缺失（§2），但静态特征路线的差距不止
   topology：Full-Y 的 win 覆盖所有子群，包括静态特征强的多环分子。
3. 两臂（O-unit/O-rich）与 Full-Y 的失败模式相关性低（ρ≈0.22/0.44），
   归因机制设计时应预期贡献分解与聚合表示有实质不同的误差面。

## 8. 声明与边界

- 单 seed（0）；bootstrap CI 是分子重采样区间，不替代跨 seed 训练稳定性。
- Full-Y 与 O-rich 的参数量（408,651 vs 29,953）、输入、损失、训练时长均
  不同——这是容量/表示可行性参照，不是模块增量对照（任务书§5.3 声明）。
- 本轮未训练第二个候选；未触碰 official valid/test；GPU smoke 仅工程验证。
- 重建辅助损失使该模型是 "raw-y task supervision + label-free
  reconstruction auxiliary"，非纯单项目标（历史冻结机制，已披露）。
