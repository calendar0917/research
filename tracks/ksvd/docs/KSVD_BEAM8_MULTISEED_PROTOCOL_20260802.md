# Beam8 continuous cover：3-seed robustness protocol

> 日期：2026-08-02  
> 状态：Beam8 sensitivity/compression/completion 可见后注册；multiseed 结果不可见前冻结。

## 1. Frozen setup

```text
graph bank seed = 810001, 72 graphs
cover seeds = 900101 / 900102 / 900103
s=10,o=3,m=1.5
target-edge vs Beam8_R1
RAW coverage/stitching only
```

## 2. Gate

每个 seed 必须同时满足：

- exact overlap、connected、single-chain invariants；
- Beam8 edge coverage gain >= `0.08`；
- Beam8 RAW full RMSE relative reduction >= `0.20`。

## 3. 判定

- `PASS_BEAM8_THREE_SEED_ROBUSTNESS`：3/3 通过；
- `BEAM8_SEED_SENSITIVE`：1–2/3 通过；
- `REJECT_BEAM8_ROBUSTNESS`：0/3 通过；
- `FAIL_BEAM8_MULTISEED_INVARIANTS`：cover contract 失败。
