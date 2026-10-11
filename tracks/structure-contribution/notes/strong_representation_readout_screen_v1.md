# cscl-strong-representation-readout-screen-v1 — Typed-WL 结构片段 vs 冻结 Full-Y 读出的 CPU 判别屏

Track: `structure-contribution` · 日期: 2026-10-11 · 协议: `configs/strong_representation_readout_screen_v1.yaml`（冻结于 commit `7c0d84c`，在两臂正式拟合之前）
代码: `code/run_representation_readout_screen_v1.py`（audit / train / analyze）、`code/typed_wl_features.py`（Arm A）
测试: `tests/test_representation_readout_screen_v1.py`（13 fast + 1 slow，全过）
产物: `results/strong_representation_readout_screen_v1/`（git-ignored，本文件与 `STATE.yaml` 为 durable 记录）

> 本轮只回答一个问题：**Full-Y 停留在内部 dev MAE ≈0.116，究竟更值得从新的化学结构表示突破（Arm A：Typed-WL / Morgan-style 计数），还是从现有 814 维表示换一种读出方式突破（Arm B：冻结 Full-Y reader 输入 + XGBoost）？**
>
> 两个候选都是**预先购买**的判别实验：固定特征定义、固定 XGBoost 配方、seed 0、每臂一次正式拟合、只看 fit/monitor 后锁模型、统一评价 dev 一次。无 GPU、无 seed 1、无调参/搜索、无第三个模型、无特征融合、official valid/test 从未加载、旧任务分量目标从未加载。

---

## 1. Phase A 协议审计（`phase_a_protocol_audit.json`）

| 检查 | 结果 |
|---|---|
| fit_inner / monitor / dev 行号 | 7200 / 800 / 2000；SHA256 与冻结 cscl-v0 split 及历史 `prep_fit7200_meta.json` 逐位一致 |
| canonical SMILES 分组 | fit_inner∩monitor = 0、fit_inner∩dev = 0、monitor∩dev = 0（重复 SMILES 4 对，全部同侧） |
| raw-y 对齐 | 与历史 Full-Y dev/monitor 导出逐行逐 y 完全一致（max diff 0.0） |
| 数据集 ↔ encoded cache 行对齐 | y max diff 0.0 / atom_count_mismatches 0 |
| A/B 训练与评估 | 同一批 7200/800/2000 行、同一 y（`arm_y_identity_max_abs_diff = 0.0`） |
| 官方数据 | official valid/test 未加载；driver 源码静态检查通过 |
| 旧目标分量 | `g/ell/s/c`、SA、logP、`target_decomposition` 未加载 |
| 拟合型转换 | 本轮仅有 XGB 模型（fit_inner 7200）与 has_UNK 诊断词表（fit_inner）；特征本身无拟合统计 |
| dev 使用 | audit/train 模式不计算 dev 任何指标；dev 只在 analyze（锁后）统一评价一次 |

**一个必须记录的工程事实（未静默处理）**：第一次 audit 运行时冻结 Full-Y 回放门失败（dev max|dpred| = 0.168）。原因是新 driver 少走了历史训练路径的 `apply_prep_fit7200`（模型消费的是 fit7200 标准化特征，而不是 cache 里的 all-10k 标准化值）。修正为逐字复用历史预处理后，回放门通过。协议、特征与候选方案均未因该失败而改变。

---

## 2. Arm A：Typed-WL / Morgan-style 结构计数（`typed_wl_feature_audit.json`）

**定义**（只读原始分子图：`x` 原子类别、去重无向键 `edge_index/edge_attr`）：

`color_0(v)=atom_type(v)`；`color_{r+1}(v)=sha256(color_r(v), sorted_multiset{(bond_type(v,u),color_r(u))})`；半径 r=0,1,2,3；每个半径独立 1024 桶（半径进 hash payload）；每个 rooted fragment 出现一次计数 +1；`X_WL ∈ R^(N×4096)`。接着拼接 raw 结构 30 维（`structure_short`+`structure_long`）与已核验 topology25（hinge），得到 `X_A ∈ R^(N×4151)`。

| 项目 | 结果 |
|---|---|
| 特征矩阵 | `(10000, 4151)`，SHA256 `085d0256…`，无 NaN/Inf、非负 |
| 计数守恒 | 每半径每分子 Σcounts = n_atoms，max abs dev = **0.0** |
| 置换不变 | 6 分子随机重标号逐位一致 |
| 双向边去重 | PyG 双存储 vs 唯一无向键逐位一致 |
| 键型交换 | 改变 bond type 即改变描述子 |
| 同构分子 | VF2 判定的 3 对同构重复 SMILES 逐位一致（0 mismatch） |
| 半径 hash 独立 | 500 个抽样签名在 4 个半径上无一次同桶 |
| global30 交叉核验 | raw 重算 vs cache 反演 max diff **9.54e-07** |
| topology25 交叉核验 | CSV raw vs cache 反演 max diff **6.1035e-05**（与历史 `topology25_audit.json` 记录值逐位相同，float32 round-trip 量级） |

**Hash collision（仅诊断，未触发任何维度调整）**：

| 半径 | 全库不同签名数 | 占用桶数 | collision ratio |
|---|---:|---:|---:|
| r0 | 21 | 18 | 14.3% |
| r1 | 476 | 388 | 18.5% |
| r2 | 12,108 | 1,024 | 91.5% |
| r3 | 63,032 | 1,024 | 98.4% |

r2/r3 的签名近乎逐原子唯一，1024 桶全部占满——描述子本质是“哈希后的一袋环境”，高冲突率是预期行为，不因冲突率调整宽度（协议禁止）。

**数据事实**：4 对同 canonical SMILES 行中，3 对带属性图同构（描述子一致）；1 对（行 941/2264）raw ZINC `atom_type` 编码不同（bin 6/9 差异），带属性图不同构——因此“同 SMILES”不等于“同带属性图”。该对分子同属同一划分，不构成跨集泄漏（split 按 canonical SMILES 分组使其必然同侧）。

---

## 3. Arm B：冻结 Full-Y 814 维 reader 输入（`fully_reader_input_replay.json`）

按协议用 **reader pre-forward hook** 截取 `AuditModel.forward(data, mask=C6_MASK)` 中 `self.reader` 真正消费的 `unified` 张量：`[unary 289 | relation_readout 5×97=485 | graph_hidden 32 | topology 8] = 814`。模型 `eval()` + `torch.no_grad()`，backbone 参数冻结、不训练、不微调。

| 证明项 | 结果 |
|---|---|
| `model.reader(captured)` vs 原 forward 输出 | max abs diff **0.0** |
| 同一批分子 64 vs 32+32 分组 | max abs diff **1.91e-06**（≤1e-5 容差） |
| y 独立性（batch.y 改变） | **0.0** |
| 行对齐（unified[806:814] vs topology_encoder 输出） | **0.0** |
| C6_MASK 生效（unary/pair count 块为 0） | true |
| 维度/dtype/有限性 | (10000, 814) / float32 / finite |
| soup checkpoint SHA 前后 | `75d1737c…` 不变 |
| 历史回放 monitor | MAE 0.1182581 vs saved 0.1182581，max|dpred| 1.67e-06 |
| 历史回放 dev | MAE 0.1164130 vs saved 0.1164130，max|dpred| 1.91e-06 |

结论：**B 臂确实复用了 Full-Y 真实 reader 输入**，且与历史 Full-Y 的工程回放一致。

---

## 4. 冻结 XGBoost 配方（两臂逐字相同，`lock.json`）

```text
n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=3,
subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
objective=reg:squarederror, eval_metric=mae, tree_method=hist,
random_state=0, n_jobs=8
```

与历史 `zinc_long_range_proxy.DEFAULT_XGB_PARAMS` 在运行时逐项断言一致。**披露**：训练目标是平方误差，评价指标是 MAE；历史固定配方如此，本轮不换成更有利的 L1 objective。500 棵树全部训练，无 early stopping，monitor 不参与树数选择。锁文件在 dev 评价前写入（`dev_evaluated: false`），analyze 前逐项校验 config/feature/model/prediction 哈希。

| | 模型 SHA256 | 拟合时长 | fit 预测 SHA256 | monitor 预测 SHA256 |
|---|---|---|---:|---|
| XGB-A | `bedb8873…` | 4.5 s | `76ae673e…` | `91dd3445…` |
| XGB-B | `17983713…` | 12.5 s | `a275e89d…` | `bb3e34d2…` |

环境：Python 3.12.14 / numpy 2.1.3 / xgboost 2.1.3 / Linux x86_64，CPU 8 线程。

---

## 5. 主结果（内部 dev 2000，raw-y MAE，越低越好）

| 模型 | fit MAE | monitor MAE | dev MAE |
|---|---:|---:|---:|
| Full-Y（历史） | 0.04220 | 0.11826 | 0.11641 |
| O-bucket（历史） | 0.03641 | 0.11794 | 0.11610 |
| **XGB-A：Typed-WL** | **0.24654** | **0.34104** | **0.36880** |
| **XGB-B：Full-Y pooled** | **0.06367** | **0.20334** | **0.19727** |

- `ΔA = MAE(XGB-A) − MAE(Full-Y) = +0.25239`
- `ΔB = MAE(XGB-B) − MAE(Full-Y) = +0.08086`
- `MAE(XGB-A) − MAE(XGB-B) = +0.17154`
- 配对 bootstrap（2000 次，seed 0，有符号每分子 |e| 差）：
  - A−FullY `+0.25239 [+0.23707, +0.26828]`，A 更好的分子占 18.35%
  - B−FullY `+0.08085 [+0.07116, +0.09121]`，B 更好的分子占 32.25%
  - A−B `+0.17154 [+0.15515, +0.18790]`，A 更好的分子占 26.9%
  - 去掉各臂 top-10 大误差分子与 y≤−3 尾部后区间仍完全在正侧（不依赖极端分子）

| 指标 | Full-Y | XGB-A | XGB-B |
|---|---:|---:|---:|
| fit–monitor gap | 0.07606（历史） | **0.09450** | **0.13967** |
| fit–dev gap | 0.07421（历史） | 0.12227 | 0.13359 |
| median | 0.07243 | 0.27703 | 0.12889 |
| p90 | 0.24569 | 0.74651 | 0.40124 |
| p95 | 0.34408 | 0.92808 | 0.52380 |
| RMSE | 0.23882 | 0.56115 | 0.38080 |
| pred–y corr (dev) | 0.9920 | 0.9549 | 0.9791 |

**子群（解释用，不用于改配方）**：

| 子群 | n | Full-Y | XGB-A | XGB-B |
|---|---:|---:|---:|---:|
| y ≤ −3 | 128 | 0.3153 | 0.8364 | 0.5775 |
| y > −3 | 1872 | 0.1028 | 0.3368 | 0.1713 |
| 原子数 30+ | 168 | 0.1804 | 0.4553 | 0.3517 |
| 环数 4+ | 409 | 0.1703 | 0.4966 | 0.3001 |
| has_UNK | 874 | 0.1490 | 0.4134 | 0.2493 |
| no_UNK | 1126 | 0.0911 | 0.3342 | 0.1569 |

**任何子群中两臂都没有优于 Full-Y**；B 相对 A 在几乎所有子群都更好，但差距仍为 +0.06–0.26。B 在 44.35% 的分子上误差超过 Full-Y 的 2 倍。

**fit→monitor 特征分布漂移（fit std 为单位）**：A 中位数 0.0091 / p90 0.0519 / max 0.1636；B 中位数 0.0152 / p90 0.0534 / max 0.1222；两臂 `frac(shift>1) = 0`。即 B 的泛化劣势**不是**由 fit 与 monitor 的特征边缘分布漂移解释的。

**历史对照（披露 recipe 差异，不作排名）**：cscl-v1 CPU screen 中，固定特征 XGB 的最佳 dev 为 rich573 = 0.3717（600 树 × lr 0.05），v1-RINGCHAIN 923 维 = 0.4532；本轮 XGB-A 在 4151 维 Typed-WL 上 dev 0.3688，处于同一平台。即**把静态片段表示从 573 维扩到 4151 维、换成 typed-WL，没有改变该路线的 dev 水平**。

---

## 6. 预冻结判断（`decision.json`）

两臂都未达到第一道信号（monitor ≤ 0.10826）：A 0.34104、B 0.20334；dev 配对区间均完全在正侧（显著更差），且不依赖极端分子。

**判定：情况 D — 本轮停止，不追加模型。** 不做 seed 1、不做 full 8000/10000 重拟合、不调树深/学习率/树数/radius/hash 宽度、不做 official valid/test 评价、不做第三回归器或特征融合、不自动继续三体网络。

---

## 7. 核心科学问题（§9 逐条）

1. **Typed-WL 计数能否提供比 Full-Y 更强的泛化信号？** 不能。在该固定配置下 XGB-A dev 0.36880，远差于 Full-Y 0.11641（+0.2524），且低于/同量级于历史静态特征 XGB 平台。它确实比 v0/v1 RINGCHAIN XGB 略好（0.3688 vs 0.4532）并略优于 rich573 历史 XGB（0.3717，recipe 略不同），但这只是回到静态特征平台，不构成新的泛化信号。
2. **冻结 814 维表示换 XGBoost 后是否明显改善？** 否。XGB-B dev 0.19727，比 Full-Y 差 +0.08085（CI 不含 0），也比 O-bucket 差。
3. **哪一臂 fit–monitor gap 更小？** XGB-A 的 gap 更小（0.09450 vs XGB-B 0.13967）；但 A 的拟合本身就差（fit 0.2465，树模型连训练集都拟合不动），而 B 的 fit 0.0637 更接近 Full-Y 的 0.0422，所以更准确的描述是：**B 是“拟合尚可但外推更差”，A 是“拟合与外推都差”**；两臂 gap 都大于 Full-Y 的历史 0.07606。
4. **哪一臂在普通分子更好？** B 在全部子群（含 y>−3、无 UNK、中等原子数）都优于 A；但两臂在普通分子上也都明显差于 Full-Y（y>−3：B 0.1713 / A 0.3368 vs Full-Y 0.1028）。
5. **A 的 hash 冲突与覆盖？** r0 14.3%（21→18 桶）、r1 18.5%（476→388）、r2 91.5%（12,108→1,024）、r3 98.4%（63,032→1,024）；r2/r3 桶全满；每行仅约 103/4151 个非零。高半径冲突是预期诊断值，本轮不据此改宽度。
6. **B 是否准确复用 Full-Y 真实 reader 输入？** 是：hook 精确捕获；`reader(captured)` 与原 forward 差 0.0；batch 分组差 ≤1.9e-6；y 无关；topology 行对齐 0.0；C6 mask 计数块为 0；checkpoint SHA 不变；历史回放 dev/monitor 与保存预测差 ≤1.9e-6。
7. **B 的二阶段监督表示带来哪些比较偏差？** (a) 814 维表示本身是 fit_inner 标签训练的，XGB-B 又在**同一批** fit 标签上做第二阶段监督学习；B 与原 Full-Y 不是计算预算匹配对照。(b) B 的有效函数类是浅层树（≤500×depth6），与 408k 参数 MLP reader 容量/优化目标不同。(c) 因此 B 失败只能说明“这一具体二阶段树读出不如原 reader”，**不能**证明 814 维表示未被充分利用。
8. **A/B 与历史 Full-Y 的不可忽略差异？** 特征（手工片段/冻结表示 vs 端到端学习的骨架）、模型类（树 vs MLP）、目标（L2 vs L1+label-free 重建辅助）、训练过程（一次性浅层拟合 vs 240 epoch 端到端）、容量（~几千叶子 vs 408,651 参数）、是否使用任务梯度。任何比较都是“表示+读出整体”的比较。
9. **对继续增强 Full-Y、转向 Typed-WL、还是停止当前小数据静态预测路线，哪种判断获得最强证据？** 最强证据是**“换读出方向也不成立”**：在完全相同的 814 维信息上，固定浅层树读出比原 MLP reader 差 +0.081；同时静态 Typed-WL 路线停留在 ~0.37 平台。因此 (a) 优先增强 Full-Y 内部结构表示（在保留其 reader 的前提下）而不是替换 reader；(b) 静态片段表示单独不足以支撑该目标；(c) 当前 7200-fit、单 seed、内部静态表示路线仍未出现可购买信号。
10. **是否出现值得下一次正式 GPU 训练的信号？** 没有。两臂均未跨过预冻结 monitor 阈值（0.10826），dev 方向相反且显著；按规则停止，不购买 GPU。

---

## 8. 结论层级

### A. 可复现的特征构造与工程事实
- Arm A 是确定性、标签盲、可复现的 Typed-WL 计数；守恒精确为 0，置换/去重/键型/同构/半径独立性检查全过；global30 与 topology25 与历史缓存逐位对齐（9.5e-7 / 6.1e-5，后者正是历史记录值）。
- Arm B 的 814 维 reader 输入提取有 6 项独立证明并可回放历史 Full-Y（dev 0.1164130 / monitor 0.1182581）。
- 一个历史预处理遗漏被回放门捕获并修复，本轮未静默替换任何方案。

### B. CPU 预测结果
- XGB-A：fit 0.24654 / monitor 0.34104 / dev 0.36880。
- XGB-B：fit 0.06367 / monitor 0.20334 / dev 0.19727。
- 两臂相对 Full-Y 的历史 dev 0.11641 均显著更差（配对 CI 完全在正侧，trim 后不变）；A−B 也显著（B 更好 +0.1715）。

### C. 对竞争假设的支持和反对证据
- **反对“换读出足以突破”**：相同信息、固定树读出显著更差且 fit–monitor gap 更大；B 在 67.75% 的分子上比 Full-Y 差。
- **反对“Typed-WL 片段计数是小样本泛化的更优归纳偏置”**：A 甚至无法在同配方下拟合训练集（fit 0.2465 vs Full-Y 0.0422），dev 停留在静态特征平台。
- **支持“Full-Y 现有 reader 已经把其输入用得相当好”**（弱证据，非定论）：任何简单替代读出都显著更差；但受限于容量/目标/两阶段监督混杂，不能上升为“reader 最优”。
- **不支持任何关于全部树模型/全部片段表示/非 GNN 上限的全局结论。**

### D. 无法隔离的混杂因素
- 固定浅层 XGB（500×depth6、平方误差、无 early stopping）与 408,651 参数 MLP reader 不匹配容量和优化目标；
- B 的特征来自与拟合同一批标签训练出的表示，存在二阶段监督重叠；
- XGB-A 的稀疏哈希计数与树模型的交互（每行仅 ~103 非零）可能与密集特征不可比；
- 单 seed 0、dev 已被 CSCL 多轮复用。

### E. 下一阶段是否值得投入 GPU 的明确判断
**本轮不值得购买 GPU。** 下一步若继续预测路线，应另行预注册并优先研究：
1. 完整训练数据预算（8000/10000）与更合适的监督/正则设计；
2. **保留 Full-Y 表示 + 学习式读者**的归纳偏置/正则（而非替换为固定浅层树）；
3. 新的静态结构先验（但需先经过标签盲的可压缩性/可预测性筛选，避免直接上 GPU）；
4. 其他非 GNN 预测架构。
以上任一都需要**新的预注册 + 用户授权**，本轮不执行。

---

## 9. 边界与纪律声明

- 单 seed（0）；每臂一次正式拟合；无 early stopping、无超参搜索、无 Optuna/CV、无自动特征选择。
- 固定 Typed-WL 配置（radii 0–3、1024 桶/半径）与固定 XGB 配方；不因 dev 结果调整任何常数。
- B 臂比较不是计算预算匹配；A/B 与 Full-Y 的差异混合了表示、模型类、目标与训练过程。
- 内部 dev 已多轮复用：即使出现改善也只能作为内部方向性证据；本轮没有改善。
- official valid/test 从未加载；官方历史数字（F 0.13295/0.11410 等）与本轮无比较关系。
- 本轮停止：不加 seed、不加模型、不调参、不重训、不做 fusion、不继续三体分支。

## 10. 变更记录

- 2026-10-11 初版：协议冻结（commit `7c0d84c`）→ Phase A/特征/回放审计（`2a7e5c86`）→ 两臂 CPU 拟合 + 锁 → 统一 dev 评价 → 情况 D，停止。
- 2026-10-11 audit 工程记录：首次回放门失败（缺少 `apply_prep_fit7200`），修复后回放通过；详见 §1。
