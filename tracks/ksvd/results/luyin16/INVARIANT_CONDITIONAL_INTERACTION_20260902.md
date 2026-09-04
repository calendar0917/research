# 不变谱交互与条件 binding 快筛

日期：2026-09-02  
协议：`luyin16-molhiv-invariant-conditional-interaction-screen-v1`

## 问题与边界

当前 `cross_cov`/`binding` 依赖每折 train-only PCA-8。PCA 坐标会随拟合范围改变，
因此本轮测试两个不依赖 PCA 坐标的压缩方向：

1. 将四个 topology×attribute covariance block（node/edge role × node/edge
   attribute）压成奇异值谱、能量、谱熵、有效秩和集中度摘要；
2. 将 binding 的 node/edge mean/std role×attribute 表压成条件集中度摘要，保留
   `attribute | rooted-WL role` 的质量、熵和最强 role 比例，而不是直接拼接稠密表。

所有特征均从已冻结的 radius-2 all-centre cache 计算。XGBoost 参数在三折
`official-train` scaffold CV 内各视图独立进行 12-trial Optuna；official-valid 和
official-test 没有编码、没有评估，也没有参与选择。

## 结果

| view | 特征维度 | tuned scaffold CV | fixed scaffold CV |
|---|---:|---:|---:|
| `S+marginal` | 508 | 0.787106 | 0.768341 |
| `S+invariant_cross` | 624 | 0.785707 | 0.768942 |
| `S+conditional_binding` | 624 | 0.786035 | 0.769374 |
| `S+invariant_both` | 740 | 0.783748 | 0.768482 |

按 train-only scaffold CV，选择的是简单的 `S+marginal`；两个新摘要均未超过该
基线，联合摘要反而下降。相对固定 XGBoost，调参仍然必要，但调参没有改变视图
排序：本轮不是“模型容量不足”的证据。

## 机制判断

本轮给出的是一个清晰的负结果：去除 PCA 坐标漂移可以得到低维、可解释的交互
描述，但当前的谱/集中度压缩会丢掉足够的任务信息，尚不能替代原始
`cross_cov`/`binding` PCA 块，也不能形成 valid/test 候选。此前 frozen test 中
`S+both` 的迁移信号仍应按原协议解释，不能用本轮 CV 结果改写。

因此当前路线保持：

```text
S + marginal                 （性能主线）
cross-centre covariance     （机制核心，PCA 坐标敏感）
conditional binding         （条件辅助，暂不独立晋级）
```

下一步若继续投入，应在不看 test 的前提下尝试“少量不变标量摘要 + 冻结 PCA 块”
的补充式组合，或先做 bootstrap/校准/错误分层；不建议据此引入 GINE、attention、
独立结构预训练或 task-aware K-SVD。

原始结果：[summary.md](invariant_conditional_interaction_screen/summary.md)；
JSON（被 `.gitignore` 忽略，仅作本地复现缓存）：
`invariant_conditional_interaction_screen/summary.json`。
