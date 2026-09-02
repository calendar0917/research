# Attributed Beam8/Mutagenicity typed-anchor protocol

> 日期：2026-08-13  
> 状态：attributed geometry audit 后冻结；不使用 graph labels。

沿用旧 typed-anchor 的最小语义修复，但所有 BASE 与 anchor patch 都使用已通过 invariance
gate 的 attributed coordinates 和 typed canonical slots：

- 固定 attributed `s8/o2 Beam8 BASE`；
- 每个图、每种实际存在但 BASE 完全未观察的 bond type，最多加入一个 anchor；
- anchor 是该 bond 的两端点语义 token；在所有候选边和两个 root 方向中选择最小 canonical
  attributed token，避免把原始 node ID 当 tie-break；
- anchor 是独立且 relation-isolated 的 segment；它不伪造连续 chain/overlap relation，
  BASE token、slot、chain 完全不变。

Gate：各 type graph coverage=1、稀有 type aggregate recall≥0.8、mean anchor≤0.25、p95≤1、
BASE ordered tokens 完全不变。通过后才能运行 attributed typed classification。
