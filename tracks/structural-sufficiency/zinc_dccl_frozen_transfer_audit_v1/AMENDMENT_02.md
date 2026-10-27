# AMENDMENT 02 — `any_unseen` 分组按协议与 `ordinary` 相交（评分后代码修正）

**时间:** 2026-10-28，在一次性 Monitor 评分**之后**、在最终报告与判定冻结之前。
**性质:** **代码与冻结定义的实现不一致**的修正。阈值、reader、cell、步数、
分组规则、判定顺序**一律未改**。

## 发现

`PROTOCOL.md` §5 把两个 any-unseen 组定义为

```
any_unseen_fit4k = (any-unseen w.r.t. Fit4k 支撑表)  AND  ordinary_nopenalty
any_unseen_fit8k = (any-unseen w.r.t. Fit8k 支撑表)  AND  ordinary_nopenalty
```

`fta_score.py` 的 `monitor_groups.json` 正确实现了该交集（423 / 301 行），但
`fta_analyze.py` 的 `group_masks()` 读了 CSV 里的**原始**布尔列
（`any_unseen_fit4k` / `any_unseen_fit8k`，440 / 317 行），漏掉了
`& ordinary`。因此分析首跑的分组表用了不含交集的集合。

## 修正

`fta_analyze.py::group_masks` 现在显式做 `& ordinary`，与协议逐字一致；随后
重跑 `fta_analyze.py` 与 `fta_checks.py`（91/91 仍全过）。唯一变化的数字：

| 组 | 首跑（440 / 317） | 修正后（423 / 301） |
|---|---|---|
| any_unseen_fit4k Δ (s0 / s1) | +0.03423 / +0.03371 | **+0.03039 / +0.03557** |
| any_unseen_fit8k Δ (s0 / s1) | +0.03484 / +0.03415 | **+0.02950 / +0.03638** |

T2 的门（两 seed 均 > 0）在两种口径下都通过；`Δ_all = +0.0297`、
`raw_gain = +0.0387`、其余分组、四个 U 门与最终判定
（`TRUNK_TRANSFER_SIGNAL`）**均不变**。两个口径都写在报告里以保持透明。

## 未做的事

没有因为这次修正改动任何阈值、端点、cell、预算或选择规则；没有重新训练任何
head（修正只影响分组掩码，probe 预测与 Monitor 评分未变）；没有读取新的标签。
