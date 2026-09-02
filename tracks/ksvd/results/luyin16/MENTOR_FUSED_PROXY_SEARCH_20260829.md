# 导师路线：S + K-SVD 融合与下游搜索

日期：2026-08-29  
范围：OGB MolHIV official train/validation；official test 未编码、未评估。

## 协议

固定 K32 typed-slot proxy 特征，只在 official train 内做 3-fold stratified inner-CV，
使用 12 组 XGBoost 参数的随机搜索（Optuna 不可用时的等价 fallback），然后将每个 view
的最优参数冻结，在完整 official train 上训练，并在 official validation 上评估 5 个模型种子。

当前 proxy 的特征视图为：

```text
S       = 205D explicit topology + atom/bond composition
R_raw   = 624D slot-aligned typed patch object
R_final = 624D K-SVD reconstructed slot object
S+R_*   = 829D concatenation
```

## 结果

### 此前离线 fallback（ParameterSampler）

| view | dim | fixed-XGB valid | inner-search valid | 备注 |
|---|---:|---:|---:|---|
| S | 205 | 0.7817 ± 0.0066 | 0.7746 ± 0.0124 | 搜索结果不保证高于固定参数 |
| S + R_raw | 829 | 0.7763 ± 0.0102 | **0.7875 ± 0.0074** | 相对 tuned S 约 +0.0129；相对 fixed S 约 +0.0058 |
| S + R_final | 829 | 0.7429 ± 0.0072 | 0.7734 ± 0.0066 | K-SVD reconstruction 仍未提供稳定增量 |

### Optuna TPE（正式运行）

正式 Optuna TPE 复核（12 trials，3-fold inner-CV，5 个最终模型种子）：

| view | dim | valid ROC-AUC |
|---|---:|---:|
| S | 205 | 0.7916 ± 0.0063 |
| S + R_raw | 829 | **0.7976 ± 0.0053** |
| S + R_final | 829 | 0.7761 ± 0.0052 |

### Sparse-code rich 融合补充

K64/T8 的已有 sparse-code rich readout 也与同一 `S` 做了融合，并使用 8-trial
Optuna TPE、3-fold train-inner CV：

| view | dim | valid ROC-AUC |
|---|---:|---:|
| S | 205 | 0.7848 ± 0.0091 |
| S + sparse-rich-no-recon | 845 | 0.7771 ± 0.0062 |
| S + sparse-rich | 853 | 0.7765 ± 0.0020 |

该结果说明字典原子级统计单独有信号，但当前定义下直接与 `S` 拼接没有带来增益；
相比之下，slot-aligned `R_raw` 的融合仍是当前更有希望的代理。

正式搜索输出：[`xgb_optuna_search.json`](./mentor_typed_slot_proxy_v2_frozen_features/xgb_optuna_search.json)。

此前离线 fallback 输出：[`xgb_random_search.json`](./mentor_typed_slot_proxy_v2_frozen_features/xgb_random_search.json)。

## 判定

1. 不能再说 K-SVD 与 S 完全割裂：在当前 typed-slot proxy 中，`S+R_raw` 经过 train-inner 下游搜索后首次超过 `S`。
2. 这个增益仍然很小，且只针对代理特征；不能声称已复现导师的约 0.80 结果。
3. 增益主要来自 raw typed object，而不是 K-SVD final reconstruction。当前 K-SVD 的无监督重建目标仍可能损失任务相关信息。
4. 搜索本身影响明显；正式 Optuna TPE 已在 `optuna==4.2.1` 环境中完成，但当前脚本保存的是 JSON trial 摘要，还没有持久化 Optuna SQLite study 数据库。
5. `S` 的搜索结果低于此前固定参数结果，说明 inner-CV 与 official validation 存在分布差异；不能只报告搜索后最高的单个结果。

## 下一步

- 如需扩大搜索，再增加 trials 或持久化 SQLite study；当前 12-trial Optuna 结果已经完成；
- 增加 K24/T3 作为容量对照，但不同时改变统计对象；
- 优先研究 `R_raw` 中的 sparse-code/atom/context 条件统计，而不是继续增加 reconstruction error；
- 暂不查看 official test。
