# COC-v1 — 反事实归属对比组合器（预注册）

轮次：`zinc_counterfactual_ownership_composer_v1`
协议：`ssrc-zinc-counterfactual-ownership-composer-v1`
冻结配置：`configs/coc_v1.json`（机器可读的唯一权威）
冻结时间（UTC）：2026-10-27
状态：**预注册。任何 GPU 作业之前，本文件、配置与实现必须先提交。**

---

## 0. 唯一研究命题

在 DCC-L 全部已有局部结构信息保持不变时，显式学习“真实 ownership 相对分子均场的
非线性差异”，能否提高有限监督标签下的跨分子泛化？

这里检验的是**归纳偏置**，不是增加结构表达能力：

```
c_glob = group_composer(x_null)                 # DCC-L 原有全局通路，不变
c_loc  = local_composer(x_real) - local_composer(x_null)
z      = [Σc_glob, Σa, Σa², global33, Σc_loc, Σc_loc²]
pred   = reader(z)
```

`local_composer` 的两次调用**必须共享同一组参数**；不得替换成两个独立网络，不得
`detach()` 第二个调用。不得新增可训练参数、可调缩放系数、归一化层、额外 loss、
attention、支持度门控或新结构特征。

---

## 1. 先例核对与不可声称的新颖性

本轮开始前已核对：

- `zinc_support_conditioned_backoff_v1`（SCSB）：把局部输入按标签无关支持度
  `alpha_b` **凸混合**到均场 `alpha*Wl+(1-alpha)*Wg`，再送入一个普通 composer。
  它不是 `f(real)-f(null)`；没有共享参数的反事实减法，没有学习非线性差异。
  SCSB 的负结果关闭的是“支持度条件收缩”这一具体算子，不关闭本轮的对比算子。
- `zinc_dual_channel_composer_v1` 的 C 臂：把所有键的局部输入**全部替换**为均场
  （`Wl=Wg,Vl=Vg`）。它测量“有没有真实局部 ownership”，不是“相对均场的非线性
  差异”。L−C 的历史正信号说明真实 ownership 有内容，但没有检验本轮的减法。
- `zinc_dcc_information_audit_v1`：检验 reader 的池化 `z` 对 correspondence-aware
  联合统计是否信息充分，以及在冻结表示上追加联合统计是否有预测增量。它没有改变
  DCC-L 的局部 composer，也没有做共享参数的反事实差分。
- `zinc_relational_contrast_contraction_v0`（RCC-v0）：检验图级二次关系核
  `Zᵀ(K_R−K_D)Z` 相对行和 Null 的独立信息；在 CPU 冗余门禁即停止。它是图级二次型，
  不是逐键共享 composer 的输入差分。
- `zinc_block_ownership_v1/v2`：改变的是 biconnected block 的归属/内容聚合，
  使用新的块级通道与参数；不是 DCC-L 均场参照上的共享非线性差分。
- `zinc_dccl_frozen_transfer_audit_v1`：冻结表示上的固定 reader 迁移审计，说明
  4k→8k 的可提取增益集中在 ownership 条件化的 composer 矩；它没有训练新架构。

因此：**没有任何已有实验实质测试过本轮的 `f(x_real)-f(x_null)` 共享参数算子。**
本轮不声称该算子已证明原创；只检验它是否在冻结的有限标签协议下产生可复现的
泛化增益。

---

## 2. 数据、划分与访问屏障

- 数据：PyG ZINC `subset=True`，official train 10,000 / official valid 1,000 /
  official test 1,000。
- 唯一可用的监督划分来自既有冻结文件
  `tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1/split_a.json`
  （sha256 `53cd2cec29fb658bdbc789845df36b2194a8c8635140d61b4173ce064b48678e`）：
  - Fit8k：8,000 行，唯一监督训练数据；
  - Monitor1k：1,000 行，checkpoint 选择与预注册筛选；
  - Locked1k：1,000 行，只在 Monitor 通过且结果冻结后一次性评分；
  - 三者互斥、并集为 official train 10,000。
- **official valid 与 official test 在本轮任何阶段都不加载、不预测、不评分。**
- 训练 cache：`zinc_arch_discovery_v1/cache/ad_train.pt`。本地与远端存在两个
  torch 序列化字节哈希（`d85d9875…` / `e3a5e239…`），差异只在 `meta` 的
  `prep_seconds`/`created`；对全部 47 个非 meta tensor 按 key/dtype/shape/bytes
  计算的 canonical tensor sha256 为 `ce872e9b…`，两端一致。代码同时断言字节
  哈希属于已冻结集合且 tensor 哈希精确匹配，避免把续跑机器的 zip 元数据差异
  误判为数据变化。
- 训练标准化只用 Fit8k 行（`fit_only_norm`，与 `zinc_dccl_bottleneck_v1` 的
  Fit8k 配方一致）；Monitor 标签只用于每 epoch 评估和选点；Locked 标签只在
  `coc_locked.py` 的一次性评分中读取。
- 历史 L 参照：`zinc_dccl_bottleneck_v1/runs/fit8k_s{0,1}/best_monitor.pt` 与
  `final.pt`。它们使用同一 Fit8k、同一 seed-matched SSL、同一初始化规则、同一
  首批顺序、同一 47,400 updates。本轮**不重新训练、不重新收费**历史 L。

---

## 3. 架构冻结

共同基础是生产版 `DualChannel`（`zinc_dual_channel_composer_v1/code/dcc_models.py`）：

```python
a, b, w = encoders(batch)
Wg, Vg, n = group_to_bonds(w, batch)
Wl, Vl, _ = local_grouping(w, batch, organized=True)

c_glob = group_composer([b, Wg, Vg, log1p(n), bond_feat])

# 唯一架构改动：
c_loc = local_composer([b, Wl, Vl, log1p(n), bond_feat]) \
      - local_composer([b, Wg, Vg, log1p(n), bond_feat])

z = [Σc_glob, Σa, Σa², global33, Σc_loc, Σc_loc²]
pred = reader(z)
```

硬约束：

1. `local_composer` 的两次调用是**同一个模块、同一组参数**；不是两个网络。
2. `local_composer(x_null)` 保留完整自动微分路径；不得 `detach()`。
3. `group_composer`、原子通路、`reader`、输入布局、宽度/深度、激活函数、
   dropout=0、初始化顺序全部不变。
4. 新增可训练参数 = 0；部署参数必须为 **494,193**，用 `named_parameters()` 重新核验。
5. 输入 `Wl/Vl/Wg/Vg` 全部保留；reader 输入维度保持 513。
6. 不新增 loss、无缩放系数、无归一化层、无 attention、无 support gate、无新结构特征。
7. `local_composer(x_null)` 与 `group_composer` 使用相同的 `x_null` 张量布局：
   `[b, Wg, Vg, log1p(n), bond_feat]`。

初始化：完全复制 DCC-L 的 RNG 顺序——先 `torch.manual_seed(seed)` 构造 P3 并加载
seed-matched SSL encoder，再 `torch.manual_seed(seed+90731)` 构造 DualChannel 并
`load_p3_state`。reader 的 321 维旧列复制 P3，后 192 维新列零初始化。因此 COC 与
DCC-L 的全部共享初始张量逐位相同。

---

## 4. CPU 门禁（任何 GPU 之前）

实现：`code/coc_checks.py` → `analysis/cpu_checks.json`。以下 C1–C9 必须全部 PASS；
任何一项 FAIL 都不得启动 GPU。

### C1 初始共享参数逐位相同
对 seed 0 与 seed 1：COC 与 DCC-L 的 `state_dict` key 集合相同、每个张量
`torch.equal` 为真、最大绝对差 0.0；参数数均为 494,193。

### C2 初始预测等价
同一组真实分子（至少 8 行、至少一个 batch）：COC 与 DCC-L 初始 forward 最大绝对差
≤ 1e-6；COC reader 第 0 层 321:513 列严格为零；DCC-L 同样为零。

### C3 强制均场局部输入时差异严格为零，global 路径不变
把 `local_grouping` 临时替换为 `group_to_bonds`（`Wl=Wg,Vl=Vg`）后：
`c_loc` 的 max abs 恰为 0.0；`c_glob`、`Σc_glob`、原子矩、global33 与基线逐位相同。

### C4 真实 junction 对应关系会改变差异表示
固定 `w`，仅在一个真实分子内置换 `junc_bond` 的行（保持每行两个键标签不变），
使每键 `n_b` 的多重集不变；`c_glob` 不变，`c_loc` 的最大绝对变化必须
`≥ 1e-5`。这证明 `c_loc` 对 ownership 对应关系敏感，而不是只对 `w` 的全局统计敏感。

### C5 一阶/二阶局部矩与 x_null 梯度路径
- 初始 step：reader 新局部列有非零梯度；local composer 的梯度因 reader 新列为零而
  为零（预期冷启动）。
- 一次 AdamW step 后：`Σc_loc` 与 `Σc_loc²` 对 local composer 参数的梯度均非零、
  有限；`local_composer(x_null)` 对 local composer 参数、对 `Wg/Vg`、对 `w` 的梯度
  均存在且有限；完整 loss 的 local composer 梯度非零。
- 源码扫描：`local_composer(x_null)` 附近没有 `detach()`。

### C6 结构不变性
对至少 4 个真实分子：
- 原子重编号：prediction max abs diff ≤ 1e-6；
- 无向键端点交换：≤ 1e-6；
- junction 行排序：≤ 1e-6；
- batch 组合（与另一分子同 batch，顺序交换）：各自 prediction 与单独 forward
  max abs diff ≤ 1e-6。

### C7 无跨分子泄漏与空键前向
- 两个分子同 batch 的 prediction 与各自单独 forward 一致（≤1e-6）；每个分子的
  `c_loc` 与另一个分子无关（≤1e-6）。
- 一个“有原子但无键、无 junction”的分子单独 forward 输出有限（无 NaN/Inf）。

### C8 Fit8k 差异表示诊断（不是性能证据）
在固定的 2,000 个 Fit8k 行（`numpy.default_rng(20261029)` 无放回抽取并排序）上，
记录 `c_loc = local_composer(x_real)-local_composer(x_null)` 与
`c_abs = local_composer(x_null)`：
- 每键 RMS 的 median/mean（绝对）；
- 每键 RMS 相对 `c_abs` 的 median/mean；
- 非零比例（`|c_loc| > 1e-8`）；
- 分子内方差（每个分子内跨键方差，再对分子平均）；
- pooled within-molecule covariance 的 participation ratio；
- 与 `c_abs`、`c_real` 的 RMS 比。

**退化门禁**：若 median 每键 RMS(`c_loc`) < 1e-4，或 nonzero fraction < 0.05，
或 median RMS 比(`c_loc`/`c_abs`) < 1e-3，则判定 `STOP_MECHANISM`，不启动 GPU。
更高阈值只用于 Locked 的机制活性陈述（见 §7.6），不在 CPU 阶段额外拦截。

### C9 参数、损失、划分、步数与访问屏障
- `named_parameters()` 参数数 = 494,193；optimizer 覆盖全部参数。
- 训练 loss 是 raw-y L1（`F.l1_loss(pred, y)`；y 为 cache 原始标签，不标准化）。
- Fit8k / Monitor / Locked 行数 8000/1000/1000，sha256 与冻结配置一致，互斥且并集
  为 10,000；不得出现 fit4k 行。
- `steps_per_epoch = ceil(8000/128) = 63`；`total_updates = 47,400`；
  `epochs_run = ceil(47400/63) = 753`；warmup=500；dropout=0；batch=128。
- 静态扫描与 `torch.load` 运行审计：本轮代码不出现 official valid/test cache
  路径；`official_valid_loaded=false`、`official_test_loaded=false`、
  `locked_labels_read` 在 GPU 训练阶段必须为 false。

---

## 5. GPU 训练

- 两个正式 run：`COC_fit8k_s0`（GPU0）与 `COC_fit8k_s1`（GPU1），并行。
- 每个 run 完整 **47,400 updates**，batch 128，AdamW，warmup 500，
  cosine `1e-3 → 1e-5`，weight decay `1e-5`，grad clip 5，dropout 0，raw-y L1。
- 训练行顺序：`torch.Generator().manual_seed(seed+101)` + 每 epoch `randperm`，
  与 `zinc_dccl_bottleneck_v1/code/dcb_train.py` 逐行一致；记录首批行号 sha256。
- 每 epoch 在 Monitor1k 上评估一次；严格最小 Monitor MAE 保存 `best_monitor.pt`
  （ties → 更早 epoch）；同时保存 `final.pt`（第 47,400 update）。
- 标准化：只用 Fit8k 行，函数与 `zinc_dccl_bottleneck_v1` 相同；与历史
  `fit8k_s0/s1/train.json` 的 `norm` 数值逐项一致。
- 初始态与 SSL：COC seed s 的初始 state hash 必须等于历史
  `fit8k_s{s}/train.json` 的 `model_info.initial_state_sha256`；SSL sha256 必须等于
  历史记录。
- 不用 official valid 选点，不读 official test。
- **Profile 先行**：用同一代码路径跑 3 个完整 epoch（189 updates），按
  `753 × mean_epoch_seconds × 1.15 + startup + final_eval + 30s` 保守投影。
  若单个正式 run 投影 > 1.0 A100·h，先停止并报告，不启动正式 run。
- **全轮硬上限 3.0 A100·h**，包含失败作业、profile、评分。超过即停止并记录。

---

## 6. Monitor 阶段判定（唯一一次，冻结后执行）

主对比：`Δ = MAE(COC) − MAE(L)`（raw-y MAE，Monitor1k，负值 = COC 更好）。
必须分别保存并同语义比较：

- `best_monitor`：各自严格最小 Monitor MAE checkpoint；
- `W20`：每 20 个连续 epoch 的窗口 median 的最小值；
- `T100`：最后 100 个 epoch 的 Monitor MAE median；
- `final`：第 47,400 update checkpoint。

**禁止**用 COC 的 best 去比 L 的 final，反之亦然。每个对比都使用同一 checkpoint 语义。

Monitor PASS 必须同时满足：

1. **两 seed 总体 Δ 均为负**（`Δ_s0 < 0` 且 `Δ_s1 < 0`）。
2. **seed-mean 总体改善至少 0.005**：`(Δ_s0+Δ_s1)/2 ≤ −0.005`。
3. **无人工环惩罚 any-unseen 组**（Fit8k exact typed radius-2 支撑 df=0，
   且 `label_excess == 0`；定义为 Monitor `primary` 组）：
   - 两 seed Δ 均为负；
   - seed-mean Δ ≤ −0.008。
4. **无人工环惩罚总体有改善**（Monitor `nopen` 组）：
   - 两 seed Δ 均为负；
   - seed-mean Δ < 0。
5. **W20/T100 方向支持 best 结论**：
   - `Δ_W20` 与 `Δ_T100` 各自 seed-mean < 0；
   - 且各自两个 seed 的 Δ 均 ≤ 0。
6. **无显著训练拟合或数值稳定性故障**：
   - 所有 loss/预测/gradient 有限，无 NaN/Inf；
   - COC 在 Fit8k 上的 `train_mae_at_best` 不超过 L 同一指标的 2.0 倍；
   - COC `final` 的 online train MAE 不超过 L `final` online 的 2.0 倍；
   - COC Monitor 曲线最后 100 epoch median 不超过其 best 的 2.0 倍。

若任一条件不成立：**`STOP_NO_MATERIAL_GAIN`**。不读 Locked，不追加 seed，
不调整架构/阈值/lr/宽度/层数，不把 near-miss 改写成 PASS。

---

## 7. Locked 阶段（仅 Monitor PASS 后）

Monitor 通过后，先把代码、配置、两个 COC checkpoint、Monitor 预测、L 冻结预测、
所有哈希和 Monitor 判定写入 `analysis/monitor_freeze.json`；然后由
`code/coc_locked.py` 一次性读取 Locked1k 标签并评分。脚本必须：
- 验证 `monitor_decision.json` 为 PASS；
- 验证配置/代码/checkpoint/prediction 哈希未变；
- 写 `analysis/locked_read.json` sentinel；第二次读取默认拒绝。

Locked 主对比同样是 `Δ = MAE(COC) − MAE(L)`，使用 `best_monitor` 作为主
checkpoint；`final`、`W20`、`T100` 作为稳健性。进入 full10k 后续研究建议的门槛：

1. **总体**：seed-mean `Δ ≤ −0.006`，且两 seed 同号（均为负）。
2. **无惩罚 any-unseen**：seed-mean `Δ ≤ −0.010`，两 seed 同号（均为负）。
3. **无惩罚 ordinary**（`ordinary_seen = low4_nopen + full_nopen`）不恶化：
   seed-mean `Δ ≤ +0.001`，且两 seed `Δ ≤ +0.002`。
4. **移除历史 L 预先确定的 worst-10**：按冻结 L best-monitor 在 Locked 上的
   **seed-mean 绝对误差**（s0/s1 平均）降序取前 10 行并删除；剩余行
   seed-mean `Δ ≤ −0.003`，且两 seed drop-10 Δ 均为负。该排序不使用 COC 预测
   或 Δ。
5. **不依赖某一个人工环惩罚极端样本**：
   - 删除全部 penalty 行后，seed-mean `Δ ≤ −0.005` 且两 seed 均为负；
   - 删除单个影响最大的 penalty 行后，seed-mean `Δ ≤ −0.005`。
6. **机制共同支持**：
   - `T100` seed-mean `Δ < 0` 且两 seed `Δ ≤ 0`；
   - `final` seed-mean `Δ < 0` 且两 seed `Δ ≤ 0`；
   - COC 与 L 的 Locked 预测变化比例 ≥ 0.50（`|Δpred| > 1e-6`）；
   - 训练后 COC 的 `c_loc` 在 Locked（或固定 Fit8k 诊断行）上满足：
     median 每键 RMS ≥ 1e-4、nonzero fraction ≥ 0.50、
     median RMS 比(`c_loc`/`c_abs`) ≥ 0.01。

同时报告：
- paired molecule bootstrap 10,000 次的 95% CI（seed-mean Δ 与每 seed Δ）；
- 跨 seed 差异 `Δ_s1 − Δ_s0`（描述性；不是独立 seed 方差）；
- 每个结构分组的 n / MAE / Δ（overall、nopen、primary、ordinary_seen、
  low4_nopen、full_nopen、penalty_all、ordinary_label、tail）；
- W20/T100/final 表；
- 预测变化幅度与差异表示诊断。

bootstrap 不能替代独立 seed 复现。

若 Locked 未通过：记录 **`STOP_LOCKED_NOT_CONFIRMED`**，关闭具体假设
“后非线性均场参照差分能够提高 DCC-L 样本效率”；不得扩大为“所有中心化方法无效”。
不追加 seed，不重训，不调架构，不读 official valid/test。

若 Locked 通过：只形成后续 full10k 实验建议；**不得自动启动 full10k、读取
official valid 或 official test**。

---

## 8. 停止规则与预算

- CPU 门禁 C1–C9 任一 FAIL → `STOP_MECHANISM`，0 GPU。
- 差异表示退化 → `STOP_MECHANISM`，0 GPU。
- 单 run 保守投影 > 1.0 A100·h → 停止并报告，不买正式 GPU。
- Monitor 未过 → `STOP_NO_MATERIAL_GAIN`，不读 Locked。
- Locked 未过 → `STOP_LOCKED_NOT_CONFIRMED`，关闭具体假设。
- 全轮硬上限 3.0 A100·h（含失败/profile/评分）。
- official valid/test 全程不访问。

---

## 9. 必答问题（报告必须回答）

A. 该架构是否真正把局部 composer 从绝对表示变成对 ownership 差异敏感的函数？
B. 此计算约束是否在固定监督数据下改善了未见结构，而不只是改变训练动态？
C. 相比历史最强 DCC-L，收益是否至少达到 0.006 左右的实用规模？
D. 若失败，是因为差异表示退化、训练无法学习，还是学到了有效表示却没有泛化收益？
   只报告已有诊断足以区分的原因，不猜测。
