# cscl-bjoc-task-adaptive-ssl-v1 — preregistration note

Track: `tracks/structure-contribution` · 建立: 2026-10-13 · 状态: 预注册（GPU 之前冻结）

## 1. 单一命题

**Masked SSL 是不是不仅能提供好的初始化，还能在监督训练期间持续带来泛化收益？**

在相同预训练初始化（同一 seed 的 split-D Fit-only masked-SSL encoder）和完全相同的
监督训练步数（22,680 updates，R3 per-step cosine）、相同 batch 顺序、相同 last-5 soup
规则下，每 4 个监督 minibatch（0-based 全局 update index `t % 4 == 0`）追加一个
atom-only masked reconstruction 辅助目标（λ = 0.1）：

- `L_t = L_property + 0.1 * L_atom` （aux 步）；其余步骤 `L_t = L_property`。
- `L_property` 是原始 BJOC-Null raw-y L1（完整原始图）。
- `L_atom = CE(h_atom(E_theta(masked view)), hidden atom type)`；masked view 从
  **同一 batch 的同一批 Fit 分子** 用冻结的 `mp.build_masked_sample` 全量重建
  （atom 15% / bond 15%，mask token atom 27 / bond 0）。
- atom context 使用与冻结 `mp.PretextModel` 完全相同的静态 incidence 公式；辅助头
  是同一 seed 的 Fit-only SSL 最后一步 atom head。
- 每 minibatch 只有一次联合 optimizer step；Adam 覆盖 BJOC-Null 参数 + 辅助头；
  clip 5 施加在联合参数集上；学习率曲线、batch 顺序、soup epoch 与 P3 完全一致。
- 推理时辅助头完全删除：最终导出 413,505 参数的 BJOC-Null；soup 只平均推理参数。

禁止因 Monitor 分数修改 λ、aux 频率、mask ratio、目标混合或头宽度。

## 2. 数据与标签边界

- **Split D**（预注册于 `configs/bjoc_architecture_breakthrough_v1.yaml`，salt
  `cscl-bjoc-arch-breakthrough-v1-repl`；Fit 8000 / Monitor 1000 / Locked 1000）。
  该划分在上一轮已预注册但**从未购买**训练 / Monitor 评分 / Locked 评分；
  本轮 prep 重新验证 row fingerprint、duplicate-group 不跨集合、并与上一轮的
  `split_D.json` payload 逐 hash 对齐，同时扫描历史 artifacts 证明 D 从未被训练。
- D 仍是 official train10k 的内部重划分：历史 full-10k 训练与已读内部块与 D 有交集
  （prep 时逐项披露）。因此**不能称 D 为独立盲测集**，只解读 within-split 配对对照。
- SSL：只读 Fit 图结构，不读任何标签。监督：只读 Fit raw-y。Monitor / Locked 的
  对象构建在 `LabelGuard` 下进行（`.y` 不可达）。Fit-only 标准化。
- Monitor：预测冻结并 hash 后一次性读取。Locked：只在晋级门槛通过、compute control
  完成且全部选择冻结后一次性读取；若门槛未通过则**不读取 Locked**。
- Official valid 本轮不加载；**official test 绝不加载**（无代码路径）。

## 3. 冻结的门槛（资源晋级，不是普适声明）

Monitor 配对对照 `Δ = MAE(TA) − MAE(P3)`（两 seed 平均，bootstrap 单位是分子）：

1. 两 seed 平均 Δ ≤ −0.004；
2. 每个 seed 的 Δ < 0；
3. paired molecule-bootstrap 95% CI 上界 < 0；
4. ordinary（y > −3）不恶化（Δ ≤ 0）；
5. 去掉最差 10 个误差分子后仍同向改善（Δ < 0）；
6. RMSE 不恶化超过 +0.005；
7. 单分子增益占比 ≤ 25%；
8. 正确性 / 标签隔离 / 来源审计全部通过。

决策词：`TASSL_GAIN_PROMISING` / `TASSL_WEAK_OR_SPLIT_SENSITIVE` /
`TASSL_NO_GAIN` / `TASSL_INVALID`。小幅改善（如 −0.001 ~ −0.002）忠实归类为弱证据。

## 4. Stage A（CPU，GPU 之前）

- **A1 frozen-loop replay**：同一初始化 / batch 顺序 / schedule 下，
  `tat.train_task_adaptive(aux_every=None)` 与 `bp.train_supervised` 的 soup、
  batch-order hash、lr 曲线逐位一致。
- **A2 SSL wrapper 等价**：`tat.pretrain_masked_with_head`（对冻结
  `bp.pretrain_masked` 的子类捕获包装，不改任何循环代码）与原函数短程 smoke 的
  encoder final hash、loss 曲线逐位一致；atom head 被捕获且来源为同一 run。
- **A2 masked-view 审计**：真实 Fit-D 分子上的 hidden-truth swap 不变性
  （200 trials）与 rebuild 泄漏审计（50 cases）。
- **A3 梯度诊断**：真实 Fit minibatch 上测量 g_y / g_a 对三类 encoder 参数的范数、
  比例、cosine、加权范数、辅助头梯度活性；初始与 2-epoch 短训后两个位置；另在
  真实 SSL 训练完成后复测一次。梯度冲突记录但不改 λ。
- **A4 不变性**：atom relabel / bond endpoint swap / batch 组成 / save-load replay
  ≤ 1e-4；TA 推理路径无辅助头、参数恰为 413,505。

## 5. Stage B（GPU 匹配训练）

| Arm | Seed | SSL 初始化 | 监督 | aux |
|---|---|---|---|---|
| P3 | 0/1 | 同一 Fit-only SSL{seed} | 360ep R3 | 无 |
| TA | 0/1 | 同一 Fit-only SSL{seed} | 360ep R3 | 每 4 update 一次 |

每个模型记录 Fit MAE、property 曲线、atom CE/acc 曲线、encoder 参数变化、梯度
norm/clip、aux batch 数与分子数、真实监督 update 数、训练耗时与 GPU-hours、
soup/checkpoint hash。禁止用 train MAE 最低 checkpoint 替代冻结 soup。

## 6. Stage C / D

- Monitor 评分：一次声明的读取；TASSL 决策词。
- **条件 compute control `P3-EXTRA`**（仅当晋级且预算允许）：同一 SSL encoder、同一
  scratch init，R3 cosine 拉伸到 T_max = 28,350 updates（22,680 × 1.25，对应 5,670 次
  aux forward/backward 的近似等价），last-5 soup。这是近似 FLOP 对照而非精确匹配
  （aux 分支只跑三个 encoder；P3-EXTRA 多见 5,670 次标签），如实披露。
- Locked：先 compute control，再冻结全部选择，然后一次读取。矛盾则披露并停止。

## 7. 预算

硬上限 **3.0 GPU-hours**（按 rr job wall-clock 累加，不做并行折算）。优先级：
CPU 审计 → SSL × 2 → P3 × 2 → TA × 2 → 条件 control。GPU0 被外部任务占用，
全部训练在 `res` GPU1 串行。预期：SSL ≈ 0.47 h/seed、P3 ≈ 0.16 h/seed、
TA ≈ 0.42 h/seed（辅助 masked rebuild 实测 ~0.1–0.2 s/128 分子 batch），合计 ≈ 2.1 h。

## 8. 停止纪律

- Monitor 未晋级：不购买更多 GPU，不读 Locked，完成负结果报告。
- 结果接近门槛：不加 seed、不搜权重、不改 λ / 频率 / mask ratio。
- 若无法正确复现原始预训练过程：停止正式训练并报告技术阻塞。
- 本轮不做 full10k refit，不访问 official valid/test，不产生新的 official-test MAE。
