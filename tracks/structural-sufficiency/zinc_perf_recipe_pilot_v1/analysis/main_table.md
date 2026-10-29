# ZPR-v1 主表 — SmoothL1(β=0.05) vs L1，seed 0，`VALID_AS_DEV_DESCRIPTIVE`

两臂共享初始化、SSL 权重、缓存、数据顺序、优化器与日程（47,400 updates / 600 ep / batch 128 / lr 1e-3→1e-5 cosine），唯一差异是训练损失。EMA（decay 0.999，第 11,850 step 起被动更新）在两臂内独立评估，不与另一臂的在线 checkpoint 混比。

## 1. 冻结决策规则（PROTOCOL.md §2.4）

| 判据 | 要求 | 实测 Δ (G1−G0) | 通过 |
|---|---|---:|---|
| full valid | Δ ≤ −0.003 | +0.00298 | no |
| ordinary | Δ < 0 | +0.00407 | no |
| minus row172 | Δ < 0 | +0.00537 | no |

**verdict = `NO_COMPELLING_RECIPE_GAIN`**

## 2. 全部预注册指标

| 指标 | G0 online | G1 online | Δ online | G0 EMA | G1 EMA | Δ EMA |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.07923 | 0.08221 | +0.00298 | 0.07966 | 0.08247 | +0.00280 |
| ordinary | 0.06462 | 0.06869 | +0.00407 | 0.06496 | 0.06840 | +0.00343 |
| r2_unseen | 0.15406 | 0.16112 | +0.00706 | 0.15483 | 0.16380 | +0.00897 |
| r2_unseen_ordinary | 0.13873 | 0.14106 | +0.00233 | 0.13950 | 0.14253 | +0.00303 |
| seen_ordinary | 0.04113 | 0.04575 | +0.00462 | 0.04134 | 0.04490 | +0.00356 |
| minus_row172 | 0.07110 | 0.07647 | +0.00537 | 0.07137 | 0.07638 | +0.00500 |
| minus_fixed_worst10 | 0.06117 | 0.06916 | +0.00798 | 0.06138 | 0.06863 | +0.00725 |
| tail | 0.28948 | 0.27683 | -0.01265 | 0.29115 | 0.28486 | -0.00629 |
| W20 | 0.07973 | 0.08443 | +0.00470 | 0.07969 | 0.08258 | +0.00289 |
| T100 | 0.07976 | 0.08450 | +0.00474 | 0.07974 | 0.08448 | +0.00474 |

`W20` = 最后 20 个 epoch 的 valid MAE 均值；`T100` = 最后 100 个 epoch 均值。两者都不依赖 checkpoint 选择，用来判断 full 的差异是不是选择噪声。

## 3. 配对逐分子比较（online best，1000 个 valid 分子）

| 量 | online best | EMA best |
|---|---:|---:|
| Δ MAE (G1−G0) | +0.00298 | +0.00280 |
| 配对 bootstrap 95% CI | [-0.00575, +0.00979] | [-0.00521, +0.00931] |
| G1 更好的分子数 | 454 | 477 |
| G0 更好的分子数 | 546 | 523 |
| 最差 10 个分子的平均 Δ | +0.37759 | +0.39048 |

## 4. EMA 在各自轨迹内部是否有收益

| 臂 | online best | EMA best | Δ (EMA−online) |
|---|---:|---:|---:|
| G0 | 0.07923 | 0.07966 | +0.00043 |
| G1 | 0.08221 | 0.08247 | +0.00025 |

## 5. 执行噪声带（同实验重跑的差异尺度）

- 冻结 trainer 两次同机重跑：最大 valid 差异 `0.42714053087879444`
- 冻结 trainer 重跑 vs 2026-10-28 历史曲线：最大 valid 差异 `0.39727437102154367`
- L1 控制臂 G0 vs 历史曲线：最大 valid 差异 `0.35853669730306137`
- 结论：a valid-MAE difference below the replay band cannot be attributed to the loss change; a difference at or above it can only be interpreted in direction, not in size

## 6. 口径

- valid drove checkpoint selection for both arms, so every number here is selection-contaminated (VALID_AS_DEV_DESCRIPTIVE).
- the two arms share init, SSL weights, cache, data order, optimizer and schedule; they differ only in the training loss, so the online comparison is paired.
- the comparison is one seed. seed-to-seed spread of this recipe is ~0.004 MAE, which is larger than any effect size this pilot can resolve.
- EMA vs online is a within-arm comparison only; an EMA checkpoint is never compared against another arm's online checkpoint.
- 本表不含 official test。test 只在决策冻结后由 `zpr_test_read.py` 读一次，并标记 `TEST_AS_DEV_DESCRIPTIVE`。
