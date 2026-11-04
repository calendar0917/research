# LCRC Peptides-func Weighted BCE Diagnostic v1 — 冻结实验协议

- Round ID：`lcrc-peptides-weighted-bce-v1`
- 结果标签：`EXPLORATORY_VALIDATION_ONLY_DIAGNOSTIC`
- 角色：工程执行。实现、正确性测试、执行、事实汇总。
- 唯一研究问题：在保持 LCRC-C1 架构、输入特征、数据划分、初始化、优化器和训练预算不变时，把普通 BCE 改为基于官方 Train 标签频率的 `pos_weight` 加权 BCE，能否改善 Peptides-func 的 Validation mean AP？
- 本轮只做一个正式 GPU 训练：**Peptides-func / LCRC-C1 / Weighted BCE / seed 0 / max 240 epochs**。

## 0. 只读边界

- 历史目录只读，不修改、不覆盖：
  - `tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1/`
  - `tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/`
- 所有新产物写入 `tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/`。
- 历史 240-epoch 普通 BCE seed0 结果为固定对照：best epoch 231，Validation macro AP **60.62 %**，best-checkpoint Train macro AP **67.94 %**，参数量 464,739，checkpoint SHA-256 `22643f093cc96dd3d8a81f916d7a6cc6e4bb38a91faf405eaf222fb7bf57634b`。
- ECFP4 + LightGBM 的 72.98 % 只作为不同方法的背景参照，不重训，不作为单变量对照。
- 本轮训练阶段不读取任何 Test 数据、标签或预测，也不读取历史 Test 预测产物；训练结束后用户明确授权的一次 post-hoc 评估见第 7 节。

## 1. 唯一干预：Weighted BCE

### 1.1 权重

对 10 个标签，只用官方 Train split：

```
pos_weight_j = train_negative_j / train_positive_j
```

- 10 个标签各自独立；
- 不截断、不平滑、不开方、不归一化、不设上限；
- 允许 `pos_weight < 1`，不得强制改为 1；
- 任一标签 Train 正例数为 0 时停止并记录错误，不修改公式；
- 完整保存标签顺序、正负计数和实际权重：`analysis/pos_weight.json`。

预期仅用于执行前核对：`antiparasitic` 约为 49，`antimicrobial` 小于 1；实际值必须从 Train 标签重新计算，不硬编码。

### 1.2 Loss

```python
torch.nn.functional.binary_cross_entropy_with_logits(
    logits,
    targets,
    pos_weight=pos_weight,
    reduction="mean",
)
```

- `logits`、`targets` 形状为 `[batch_size, 10]`，`pos_weight` 形状为 `[10]`；
- 数据和标签顺序与历史 Peptides-func 完全相同；
- 保持 10 个独立 sigmoid 输出；
- 正例权重不做任何额外缩放；
- reduction 仍为每个有效元素均值；
- 若支持缺失标签，保留有限标签 mask，先逐元素计算 loss，再只在有效标签上求均值；不得先压平 `[batch, labels]` 再错误广播 `pos_weight`；
- Peptides-func 没有缺失标签，masked 与 unmasked 数值应一致；
- 不修改分类头、激活函数或 Validation 指标。

### 1.3 训练前正确性测试

1. `pos_weight` 与 Train 各标签正负计数严格对应；
2. 标签顺序、维度、dtype、device 正确；
3. `pos_weight = ones(10)` 时，新 loss 与原普通 BCE 的数值和梯度一致；
4. 人工极端不平衡小 batch 上，加权 BCE 与手工公式一致；
5. 带缺失标签 mask 的人工测试无形状或广播错误；
6. `pos_weight` 不含任何 Validation/Test 统计；
7. 梯度有限，AdamW 可以更新参数；
8. Test 访问保护正常。

全部通过后才启动正式 GPU 训练。

## 2. 冻结训练配方

除 loss 外，与 Diagnostic v1 的 P2 完全相同：

| 参数 | 值 |
|---|---|
| 数据集 | Peptides-func 官方 LRGB 固定划分 |
| 模型 | LCRC-C1 / NO_BOND_MEAN |
| Seed | 0 |
| 初始化 | From scratch；与历史 seed0 init hash 一致 |
| 参数量 | 464,739 |
| Optimizer | AdamW |
| Peak LR | 3e-4 |
| Weight decay | 1e-3 |
| Betas | (0.9, 0.999) |
| Epsilon | 1e-8 |
| Dropout | 0.1 |
| Batch size | 64 |
| Gradient clipping | 5.0 |
| Scheduler | Per-step cosine |
| Min LR | 1e-5 |
| Cosine horizon | 240 epochs |
| Max epochs | 240 |
| Min epochs | 80 |
| Patience | 40 |
| Loss | Weighted BCEWithLogitsLoss，Train-only `pos_weight` |
| Checkpoint selection | 最高 Validation mean AP |
| Test | 训练阶段禁止访问；授权后仅允许第 7 节的 post-hoc terminal evaluation |

- 从头训练，不加载历史训练好的 checkpoint。
- 与历史一致的 train/validation cache、feature spec、normalization。
- 不因加权 loss 造成不稳定而降低 LR、调整 gradient clipping 或更改权重。
- 非有限 loss/梯度或 OOM：保留诊断日志并报告；OOM 仅按固定策略尝试 batch 32、16，记录实际 batch size，并标记不再为严格单变量对照。
- 按 Validation macro AP 选最佳 checkpoint，不按 Validation BCE 选模型。

## 3. Test 隔离

- 训练与分析入口导入时安装 `sys.addaudithook`；历史 Test artifact 名称一出现即抛 `TEST_ACCESS_BLOCKED`。
- 正式训练与原始 Validation 分析只报告 Train 与 Validation 指标。
- 必须记录：`test_evaluated = false`，`test_access_attempts = []`。
- 出现 Test 访问立即停止。
- 以上限制适用于正式训练与原始 Validation 分析；训练完成且 checkpoint 冻结后，用户授权的一次 post-hoc Test 评估见第 7 节。

## 4. 记录

每个 epoch：Train weighted BCE loss、Validation mean AP、10 个标签 Validation AP、learning rate、gradient norm（mean/max）、是否最佳、耗时。

最佳 checkpoint 额外记录：Train macro AP、Validation macro AP、10 标签 Train/Validation AP、逐标签 gap、与历史普通 BCE 的逐标签 Validation AP 差值。

**普通 BCE 与加权 BCE 的训练 loss 数值不能直接比较大小，因为优化目标的权重不同。**

## 5. 资源预算

- 机器 `res`，1 张 A100；GPU 总量 ≤ 1.25 A100 GPU-hours。
- 单次训练执行时限 4200 秒。
- 正式训练仅 1 个 seed；不做额外训练或自动重试。
- 只允许必要的短时 CPU 正确性检查和最小 CUDA smoke test。
- 达到预算或无法排除的工程阻塞：记录状态并结束，不越界。

## 6. 交付

`PROTOCOL.md`、`configs/protocol.json`、`code/`、`tests/`、`analysis/pos_weight.json`、`analysis/gates.json`、`runs/weighted_bce_seed0/{curve.csv,run.json,selection.json}`、`results/validation_comparison.csv`、`results/per_label_comparison.csv`、`REPORT.md`。

`REPORT.md` 只给可核验事实、差值与实验局限；不得解释为“架构问题已解决”或“架构问题已证明”。

## 7. 授权后的 Test 修订（post-hoc）

- 正式训练完成、checkpoint 已按 Validation 选择并冻结后，用户于 `2026-11-04T03:36:43Z` 明确授权访问官方 Test split。
- 该授权只允许一次 **post-hoc terminal Test evaluation**：对已冻结 checkpoint 做前向评估，用于报告和比较。
- 不做训练、不调参、不重新选择 checkpoint/seed/配置，不重训 ECFP。
- 正式训练 `runs/weighted_bce_seed0/run.json` 仍记录 `test_evaluated = false`；Test 访问只发生在独立的 `code/wb_test_eval.py`。
- 授权和评估记录：`analysis/test_access_authorization.json`、`configs/test_access_amendment.json`、`analysis/test_eval.json`、`analysis/test_gates.json`。
- Test 结果：`TEST_REPORT.md`、`results/test_comparison.csv`、`results/test_per_label_comparison.csv`。
- 这些 Test 数值未用于任何选择；若未来用其选模型，性质将变为 Test-based selection，必须另行声明。
