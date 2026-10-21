# Follow-up declaration: `A_attn` (the single follow-up of this round)

Written **before** the run started, on the seed-0 evidence only (official valid).
No test number was used to choose this follow-up.

## Why (the observation that motivates it)

Arm A converged to best-valid **0.082189 @ epoch 632** of 1200 — 0.0060 above
P3B (0.076224) and 0.0037 above L (0.078537) — while its online train MAE fell
to **0.00028** by epoch 1200, so the trunk is not under-fit and the deficit is
not "not enough training". Under this round's constraints (one explicit
association contraction, no recursive local-neighbour hidden-state passing) the
readout is the one component that has not been varied: it collapses each graph's
1,369-slot pair set into seven *uniform* summaries (count-scaled sum, mean,
second moment, diagonal mean, and three unary means), which cannot emphasise a
rare or informative pair.

## What (one change, everything else fixed)

`A_attn` = arm A with the three off-diagonal pair blocks replaced by learned
content-based pooling: three zero-initialised queries score each graph's
off-diagonal pairs (`softmax` over the graph's pairs) and produce a learned
sum, a learned mean and a learned second moment (**+288 parameters**, 488,889
total, still far below the 500,000 cap). Trunk, pair descriptors, contraction,
data, seed 0 and the declared recipe are unchanged; zero queries reproduce arm
A to 1.2e-6 (checked in `zae_checks.py`), so any difference is caused by the
pooling weights, not by extra capacity.

## Decision criterion

| A_attn best valid | reading |
|---|---|
| **≤ 0.079** | pooling was a real part of the gap: the pooled-pair route keeps upside and a future round should buy it with a matched P3B control |
| 0.079 – 0.0822 | pooling is second order; the 0.05x gap has to come from a different mechanism |
| ≥ 0.0822 | uniform pooling is not the bottleneck; the one-contraction pair route is closed on this dataset |

## Budget note (declared schedule)

Both remaining trajectories (`A_SSL`, `A_attn`) declare **900 epochs × 157
steps = 141,300 updates**, not 1200. Evidence: A's best-valid arrived at epoch
632 (53 % of its horizon) and the following 568 epochs changed it by 0.0016
while consuming 47 % of the compute; B's best arrived at epoch 332 and never
improved through epoch 817. 900 epochs keeps the whole productive learning-rate
band (`lr` 3e-4 → 1e-5) inside the horizon with margin, and every comparison
against the 1200-epoch arm A is additionally reported in a matched learning-rate
band (`lr` ∈ (1e-4, 3e-4]) so the horizon difference cannot manufacture a
difference.
