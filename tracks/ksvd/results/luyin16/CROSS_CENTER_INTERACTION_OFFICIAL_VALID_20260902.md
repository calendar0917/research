# MolHIV 中心级交互路线：official-valid 终端开发结果

> **后续状态（2026-09-02）**：本文是 official-valid 冻结搜索的结果快照。随后已
> 按冻结协议完成 controlled official-test，数字和 PCA-scope 敏感性见
> [`CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md`](CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md)
> 与 [`CROSS_CENTER_INTERACTION_TEST_PCA_CONTROL_20260902.md`](CROSS_CENTER_INTERACTION_TEST_PCA_CONTROL_20260902.md)。
> 本文的 valid 排名仍然有效，但不能单独代表 test 泛化排名。

日期：2026-09-02  
协议：`luyin16-molhiv-cross-center-interaction-terminal-valid-v1`

## 结论

这条显式统计路线已经达到强模型范围，但 official-valid 上最强的是简单统计，
不是额外的交互块：

> `S + marginal topology/attribute statistics → tuned XGBoost`

五个 model seeds 的 official-valid 平均 ROC-AUC 为 **0.841285**，预测均值集成为
**0.845047**。因此“统计模型不强，分析没有意义”的担忧在当前结果上不成立；
但交互实验的真正价值，是进一步说明 0.84 的主要来源仍是稳定的边际统计，而非
尚不稳定的 cross/binding 深融合。

## 协议

- official-train：32901 图，1232 positive；
- official-valid：4113 图，81 positive；
- 每个视图独立进行 16-trial Optuna TPE；
- 超参数目标只使用三折 official-train scaffold CV；
- 每个 scaffold fold 的 `cross_cov`/`binding` PCA-8 只在该 fold train 上拟合；
- official-valid 投影只在完整 official-train 上拟合；
- official-valid 用 seeds `0..4` 冻结评估；
- official-test 未编码、未评分。

## 结果

| view | train scaffold CV | fixed valid mean | tuned valid mean | tuned seed ensemble |
|---|---:|---:|---:|---:|
| `S` | 0.761323 | 0.779013 | 0.791586 | 0.793498 |
| `S+marginal` | 0.787106 | 0.818908 | **0.841285** | **0.845047** |
| `S+cross_cov` | 0.789985 | 0.818362 | 0.839257 | 0.842530 |
| `S+binding` | 0.788789 | 0.817119 | 0.838859 | 0.841449 |
| `S+cross_cov+binding` | **0.790662** | **0.824530** | 0.838222 | 0.841259 |

超参搜索是必要的：相对固定 XGBoost，official-valid 平均增益为：

- `S`：+0.012574；
- `S+marginal`：**+0.022377**；
- `S+cross_cov`：+0.020895；
- `S+binding`：+0.021740；
- `S+both`：+0.013692。

`S+marginal` 五个 seed 的 valid AUC 为：

`0.844542 / 0.842221 / 0.844327 / 0.836421 / 0.838917`。

## 如何理解交互结果

### 1. train-only CV 仍偏向交互

`S+both` 相对 `S+marginal` 的 scaffold-CV 增量为 `+0.003556`，2/3 folds；
因此按照预注册的 train-only CV 选择规则，会选择 `S+both`，其 official-valid
均值为 0.838222。

### 2. official-valid 上排序反转

当每个视图获得独立、等预算调参后，`S+marginal` 相对：

- `S+cross_cov`：+0.002028，4/5 seeds 更高；
- `S+binding`：+0.002426，3/5 seeds 更高；
- `S+both`：**+0.003063，4/5 seeds 更高**；
- `S+both` seed ensemble：+0.003788。

这说明 cross/binding 在内部 scaffold slices 中具有真实信号，但增量没有在
official-valid scaffold shift 上稳定转移。

### 3. 为什么固定参数时 `both` 又更好

使用同一套固定 XGBoost 时，`S+both − S+marginal = +0.005622`，4/5 seeds；
但各视图分别调参后，marginal 反超。这更像是：交互块提供了一些容易被树利用的
额外容量，但 marginal 经过更合适的强正则化和采样参数后已经吸收了主要任务信号，
额外交互维度变成冗余或 scaffold-sensitive 噪声。

## 与旧路线的参考比较

同一 official-valid 上，旧的 tuned `S+r_raw` 五种子均值为 0.830670；当前
`S+marginal` 为 0.841285，参考增量约 **+0.010616**。两者搜索预算分别为 12 和
16 trials，因此该数字是路线参考，而不是严格等预算的统计检验。

## 路线决策

1. **在 official-valid 开发阶段**，性能主线冻结为 `S+marginal`；它简单、稳定并达到 0.84 valid。
2. `cross_cov` 与 `binding` 保留为机制诊断，不再作为默认生产特征。
3. 不引入 GINE、attention、结构独立预训练或 K-SVD task update 来“挽救”交互。
4. 随后已按冻结参数和五个 seeds 完成一次 controlled official-test；结果见
   [`CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md`](CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md)。
   test 未用于回改视图或参数。

## 冻结的 `S+marginal` 参数

```json
{
  "n_estimators": 582,
  "max_depth": 7,
  "learning_rate": 0.02845665332405855,
  "min_child_weight": 2.0552437989268113,
  "subsample": 0.7195099542843616,
  "colsample_bytree": 0.44325241391504167,
  "reg_lambda": 32.87731853406676,
  "reg_alpha": 13.160423776210738,
  "gamma": 1.5959505815360724,
  "max_bin": 256
}
```

原始结果：

- [summary.md](cross_center_interaction_terminal/summary.md)
- [summary.json](cross_center_interaction_terminal/summary.json)
- [runner](../../experiments/luyin16/cross_center_interaction_terminal.py)
- [config](../../configs/luyin16/cross_center_interaction_terminal.yaml)
