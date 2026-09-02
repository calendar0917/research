# Marginal beam cover：matched KSVD compression follow-up

> 日期：2026-08-02  
> 状态：RAW beam 结果可见后注册；compression 结果不可见前冻结。

## 1. 触发原因

50-node RAW audit 中，beam `s10/o3` 相对 target `s10/o3`：

- edge coverage `+0.1190`；
- RAW full RMSE relative reduction `31.1%`；
- pair coverage `-0.0287`。

本轮检查 edge-focused cover 是否在相同 sparse capacity 下仍改善完整图重构。

## 2. Frozen branches

```text
target_o5 = target edge, s10/o5/m1.5
target_o3 = target edge, s10/o3/m1.5
beam_o3   = marginal beam, s10/o3/m1.5
graph bank seed 810001
cover seed 890101
beam=32, restarts=2
K=24,T=3,T_min=1,updates=25,PCA3
3-fold graph isolation
```

三者 patch dimension、dictionary size 相同；target_o3 与 beam_o3 patch count/code cost 完全匹配。

## 3. Registered beam gate

beam_o3 相对 target_o3 必须：

1. FINAL full adjacency RMSE 至少降低 `0.05` relative；
2. FINAL full edge recall 和 F1 均提高；
3. FINAL observed RMSE不恶化超过 `0.05` relative；
4. RAW invariants 和 3/3 fold FINAL patch error < INIT；
5. FINAL 优于自己的 PCA3 full RMSE。

pair coverage下降作为明确代价单独保留，不因 full RMSE 改善而隐藏。

## 4. 判定

- `ADOPT_MARGINAL_BEAM_COVER_V2`：beam gate 通过；
- `BEAM_RAW_GAIN_SURVIVES_PARTIALLY`：full graph改善但 gate 不完整；
- `BEAM_RAW_GAIN_LOST_AFTER_COMPRESSION`：FINAL full graph不改善；
- `FAIL_BEAM_COMPRESSION_INVARIANTS`：RAW/fold gate 失败。
