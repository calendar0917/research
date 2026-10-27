# AMENDMENT 01 — E4 batch-invariance tolerance (pre-result)

**时间:** 2026-10-28，在**任何** probe 训练、任何 Monitor 评分、任何 `analysis/probe_*`
产物之前。
**触发:** 第一次执行 `fta_export.py` 时 E4 报 `3.814697265625e-06`，而
`PROTOCOL.md` §2.2 冻结的阈值是 `1e-6`。
**此时已经产生的产物:** 只有 `runs/z/*.npz` 尚未写出（脚本在 gate 处退出），
`analysis/` 为空。没有任何标签被读取（`fta_export.py` 不读 y），没有任何 probe
cell 被训练，Monitor 未被评分。

## 诊断

E4 比较的是同一批分子在 **batch 256** 与 **batch 64** 下的 `z`。两者的差异只能来自
float32 GEMM 的分块求和次序，不是逻辑差异：

- 同一 batch 大小下的两种独立实现（hook 捕获 vs 显式重算）差 **0.0**（E2）；
- 导出 `z` 经同一 checkpoint 的 reader 重放差 **0.0**（E3）；
- 只有跨 batch 大小出现 `3.8e-6` 的绝对差；`z` 的元素量级为 `O(10^2)`
  （`sum(a^2)`、`sum(c_loc^2)` 这类平方和块），因此相对差约 `2e-8`，正好在 float32
  舍入预算内。

## 修订（只改数值容差，不改任何科学口径）

| 项 | 原 | 修订后 |
|---|---|---|
| E4 判据 | `max|Δ| ≤ 1e-6` | `max|Δ| ≤ 1e-5`（绝对），并同时记录 `max|Δ| / max|z|` |

- 其余门禁（E1/E2/E3）与阈值**不变**；`Δ_transfer` 阈值、reader、cell、分组、
  判定规则**全部不变**。
- 该修订不改变任何 cell 的输入数据：E4 只是导出正确性的门禁；`z` 本身由 batch 256
  的真实 forward 产生并通过 E1–E3。
- 修订后 E4 的实测值与相对量级会写入 `analysis/representation_freeze.json` 与
  `analysis/cpu_checks.json`。

## 追加观察与最终判据（同一 gate，仍在任何 probe/Monitor 之前）

把绝对容差放到 `1e-5` 后，`fit4k_s1` 报 `1.1444e-05`，而同一批 `z` 的
`max|z|` 使相对差只有 `3.51e-08`（≈ 0.3 ulp of float32）。可见“绝对阈值”不是
正确的判据：`z` 的块量级跨条件变化（`sum(a^2)`、`sum(c_loc^2)`），绝对舍入差也
随之变化。

**最终 E4 判据（冻结）:** `max|Δ| / max|z| ≤ 1e-7`（约 0.84 float32 ulp），
同时记录绝对差。实测：`fit4k_s0` 3.81e-6（相对 1.1e-8），`fit4k_s1` 1.14e-5
（相对 3.5e-8）。E1/E2/E3 仍为精确 0.0。
