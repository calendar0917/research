# Beam8 cover：train-only completion follow-up protocol

> 日期：2026-08-02  
> 状态：beam sensitivity 与 Beam8 compression 结果可见后注册；completion 结果不可见前冻结。

## 1. Frozen setup

```text
sampler = marginal candidate beam
s=10,o=3,m=1.5
beam=8,restarts=1
cover_seed=890101, seed tag=2（复现 Beam8 compression branch）
K=24,T=3,updates=25
same slot reliability and structural ridge protocol
3-fold graph isolation
```

## 2. Questions

1. slot weighting 在 edge-focused beam patches 上是否达到 1% observed RMSE gain？
2. structural completion 是否仍优于 density 和 graph-internal shuffled predictions？
3. Beam8 KSVD + completion 的最终 full RMSE 是否低于 target-chain KSVD + completion `0.3845`？

## 3. Gate

复用原 stitch/completion gates。额外要求 Beam8 weighted structural full RMSE 相对 target-chain weighted structural 至少降低 `0.05` relative。

## 4. 判定

- `BEAM8_COMPLETION_COMPOUNDS_GAIN`：completion gate 与额外 full-RMSE gate 同时通过；
- `BEAM8_GAIN_BUT_COMPLETION_NO_ADDED_VALUE`：beam end-to-end 更好，但 completion gate 不通过；
- `FAIL_BEAM8_COMPLETION_INVARIANTS`：train/test 或 feature contract 失败。
