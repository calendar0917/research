# MolHIV `S+cross_cov`：PCA-16 完整调参与冻结 valid/test 结果

日期：2026-09-03  
状态：**结果整理快照，不重新运行**

## 1. 一句话结论

将 `cross_cov` 从 PCA-8 改为 PCA-16 后，`S+cross_cov` 在只用
`official-train` scaffold CV 调参的完整协议下得到：

- official-valid：**0.837160**（5-seed mean），ensemble **0.838666**；
- strict train-only official-test：**0.803619**（5-seed mean），ensemble **0.806223**；
- train+valid classifier refit、但 PCA 也重拟合：**0.797590**，ensemble **0.799998**。

本轮没有用 test 选择视图、参数或 rank；但由于仓库更早路线已经查看过 MolHIV
official-test，不宣称 untouched test。

## 2. 协议边界

| 项 | 内容 |
|---|---|
| dataset | `ogbg-molhiv` |
| split | train/valid/test = 32901/4113/4113 |
| test positive | 130 |
| representation | radius-2 all-centre rooted-WL topology + strict chemistry |
| `cross_cov` raw | 5088D（96D topology × 53D attribute） |
| projection | train-only PCA-16；test 只 transform |
| classifier | XGBoost `binary:logistic` |
| tuning | 16-trial Optuna TPE，只在 3 个 official-train scaffold folds 上进行 |
| model seeds | 0, 1, 2, 3, 4 |
| test selection | 否 |

`S+cross_cov` 的最终维度是：

```text
S                         205D
local marginal + context 303D
cross_cov PCA              16D
--------------------------------
S+cross_cov               524D
```

这里的 `local marginal + context` 已经包含在该 view 中；不能把该结果解释成
`205+16=221D` 的纯 `S+cross_cov`。

## 3. 与 PCA-8 对照

| 指标 | PCA-8 | PCA-16 | 变化 |
|---|---:|---:|---:|
| cross_cov train scaffold CV | 0.789985 | 0.790890 | +0.000905 |
| cross_cov official-valid mean | 0.839257 | 0.837160 | −0.002097 |
| cross_cov official-valid ensemble | 0.842530 | 0.838666 | −0.003864 |
| cross_cov strict test mean | 0.800937 | 0.803619 | +0.002682 |
| cross_cov strict test ensemble | 0.805587 | 0.806223 | +0.000636 |
| cross_cov train+valid refit test mean | 0.799824 | 0.797590 | −0.002234 |
| cross_cov train+valid refit ensemble | 0.803652 | 0.799998 | −0.003654 |

PCA-16 在小规模 train-only sensitivity screen 中相对 PCA-8 的 `cross_cov`
提升较明显，但在完整 official-valid 上没有复现为 valid 增益；在 strict test
mean 上只有小幅增加。因此目前不能据此宣称 PCA-16 全面优于 PCA-8。

## 4. Frozen XGBoost 参数

这些参数由 official-train scaffold CV 选择，未使用 official-valid/test：

```json
{
  "n_estimators": 457,
  "max_depth": 7,
  "learning_rate": 0.015322072275263636,
  "min_child_weight": 24.319244078501743,
  "subsample": 0.7409059602685463,
  "colsample_bytree": 0.7659381803440465,
  "reg_lambda": 12.474093954716258,
  "reg_alpha": 13.58065075070035,
  "gamma": 1.372632419210349,
  "max_bin": 256
}
```

最佳 train scaffold CV 为 **0.790890**，三折为：

```text
0.804340 / 0.771360 / 0.794256
```

## 5. Official-valid

| view | valid mean | valid ensemble |
|---|---:|---:|
| `S` | 0.791586 | 0.793498 |
| `S+marginal`（同一次 PCA-16 run 的冻结搜索参考） | 0.841285 | 0.845047 |
| `S+cross_cov`（PCA-16） | **0.837160** | **0.838666** |
| `S+binding`（同一次 PCA-16 run 的冻结搜索参考） | 0.838859 | 0.841449 |
| `S+both`（同一次 PCA-16 run 的冻结搜索参考） | 0.838222 | 0.841259 |

为了防止误读，上表中其余 view 使用的是同一次 PCA-16 run 中的 frozen-parameter
参考，并不是本轮对它们重新做的 PCA-16 调参。PCA-16 本身的完整五-view valid 快照见：

- [`CROSS_CENTER_INTERACTION_PCA16_OFFICIAL_VALID_20260903.md`](CROSS_CENTER_INTERACTION_PCA16_OFFICIAL_VALID_20260903.md)

在 valid 上，PCA-16 `S+cross_cov` 低于冻结的 PCA-8 `S+marginal`：

```text
0.837160 - 0.841285 = -0.004125
```

因此 valid 仍不能支持把 PCA-16 `cross_cov` 作为性能主线替代 `marginal`。

## 6. Official-test 冻结检查

### Strict train-only

PCA 只在 official-train 拟合，分类器也只在 official-train 训练：

| view | test mean | test ensemble |
|---|---:|---:|
| `S` | 0.743173 | 0.746054 |
| `S+marginal`（同一次 PCA-16 run 的冻结搜索参考） | 0.788872 | 0.791753 |
| `S+cross_cov`（PCA-16） | **0.803619** | **0.806223** |
| `S+binding`（同一次 PCA-16 run 的冻结搜索参考） | 0.789697 | 0.792804 |
| `S+both`（同一次 PCA-16 run 的冻结搜索参考） | 0.808749 | 0.813525 |

相对同一次 PCA-16 run 中冻结的 `S+marginal` reference：

```text
S+cross_cov - S+marginal
= 0.803619 - 0.788872
= +0.014747
```

但这个差值只能说明该冻结 test 检查中 `cross_cov` 的迁移表现较好，不能替代
新的 outer scaffold 验证，也不能用 test 反向选择模型。

### Train+valid refit

PCA 和分类器都在 train+valid 上重拟合：

| view | test mean | test ensemble |
|---|---:|---:|
| `S` | 0.753994 | 0.756810 |
| `S+marginal` | 0.785279 | 0.788797 |
| `S+cross_cov`（PCA-16） | **0.797590** | **0.799998** |
| `S+binding` | 0.793064 | 0.795979 |
| `S+both` | 0.801946 | 0.804531 |

PCA-16 `cross_cov` 的 strict train-only 与 refit test mean 相差：

```text
0.803619 - 0.797590 = +0.006029
```

这与此前观察一致：交互块的 PCA 坐标系变化会影响树模型的迁移。

## 7. 如何解释

当前最稳妥的结论是：

1. **PCA-16 是有意义的敏感性结果，不是无效设置。** 它把 `cross_cov` 的
   train-only CV 提升到 0.790890，并在 strict test mean 达到 0.803619。
2. **但 PCA-16 没有在 official-valid 上超过 marginal。** valid mean 为 0.837160，
   低于 `S+marginal` 的 0.841285。
3. **不能把 test 上的相对优势直接写成已确认的普适增益。** test 只有 130 个
   positive，且历史路线已经查看过该 test；本轮 test 是冻结后的 controlled check。
4. `cross_cov` 目前仍然是最值得继续分析的交互块，但更适合进入新的 outer
   scaffold split / bootstrap 稳定性实验，而不是马上替代 `S+marginal`。

## 8. 结果来源

- PCA-16 valid：[`CROSS_CENTER_INTERACTION_PCA16_OFFICIAL_VALID_20260903.md`](CROSS_CENTER_INTERACTION_PCA16_OFFICIAL_VALID_20260903.md)
- PCA-16 test：[`CROSS_CENTER_INTERACTION_PCA16_OFFICIAL_TEST_20260903.md`](CROSS_CENTER_INTERACTION_PCA16_OFFICIAL_TEST_20260903.md)
- PCA-16 test 原始 ledger：`cross_center_interaction_terminal_pca16/test_summary.json`
- PCA-16 valid 原始 ledger：`cross_center_interaction_terminal_pca16/summary.json`
- 配置：[`cross_center_interaction_terminal_pca16.yaml`](../../configs/luyin16/cross_center_interaction_terminal_pca16.yaml)
- valid runner：[`cross_center_interaction_terminal.py`](../../experiments/luyin16/cross_center_interaction_terminal.py)
- test runner：[`cross_center_interaction_terminal_test.py`](../../experiments/luyin16/cross_center_interaction_terminal_test.py)
