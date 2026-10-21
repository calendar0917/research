# zinc-arch-exploration-v3 — REPORT (in progress)

Round: `zinc-arch-exploration-v3`. Track: `structural-sufficiency`.
Status at the time of writing: arm A completed, arm B stopped on budget grounds,
`A_SSL` and `A_attn` still training; official test not yet read for any arm of
this round. Sections marked **(pending)** are filled after those two runs finish
and after the frozen best-valid checkpoints are scored once on test.

All valid numbers are the official ZINC (subset) valid split, raw-`y` MAE,
best epoch chosen by strict `<` with ties going to the earlier epoch. The test
split is an already-exposed benchmark in this repository; historical test numbers
are quoted from `zinc_dual_channel_composer_v1/analysis/test_reporting.json` and
were not recomputed.

## 1. What this round asked, and what was new

Question: can a *static*, explicitly-structured trunk reach raw-`y` test MAE in
`[0.050, 0.060)` on the official ZINC subset, under 500k trainable parameters,
with **no recursive local message passing**? Two new trunks were bought, plus a
native-SSL variant of the first:

* **Arm A — full-graph pair composition with intermediate identity.** Every atom
  pair gets a label-free descriptor object (hop bucket, bond type, common
  neighbours, ring-bond flag, same-block flag, adjacency) and one shared
  one-shot atom state; pair states are formed by a shared MLP, then **one**
  explicit association contraction
  `U_ij = (1/n) Σ_k φ(X_ik) · ψ(X_kj) · χ(a_k)` runs over the intermediate atom
  `k`, and a per-pair nonlinear `H([X_ij, U_ij])` feeds a count-aware reader with
  diagonal and off-diagonal blocks. This is new relative to P3 (local rounds on
  atom/bond/junction tokens) and to the earlier L/C composers (channel split,
  no intermediate-atom identity, no per-pair nonlinear before pooling).
* **Arm B — joint invariants of static whole-graph operators.** Symmetric
  normalised Laplacian heat kernels `K_t = Q e^{-tΛ} Qᵀ` at `t ∈ {0.5, 1, 2}`
  plus three typed static operators; pair invariants `tr(D_u K_t D_v K_s)` and
  three-point traces `tr(D_u K_t D_v K_s D_w K_q)` over 84 declared triples
  (HHH, THH, TTH, TTT). Checks show 8.8 % residual variance beyond
  pair + first-order invariants, so the triple block is not degenerate.
* **Arm A-SSL — native fresh-mask pretraining of the new trunk.** 15 % atom/bond
  masking, new mask every epoch, full feature rebuild from the masked raw graph
  via the frozen BJOC sampler; only train structures, no labels, no valid/test.
  6,176 temporary heads reported separately; the trunk (488,601) is unchanged.

Both trunks use the frozen P3 seed-0 SSL encoders as their initialisation, so
they are "new trunk on the same feature family", not new features.

## 2. Valid main table (seed 0)

| model | params | best valid | best ep | eval-train@best | final valid | updates | SSL | status | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| P3B_s0 (reused) | 413,505 | 0.076223754 | 377 | 0.014680 | 0.078178808 | 47,400 | P3 masked SSL encoders | completed | 0.30 |
| L_s0 (reused) | 494,193 | 0.078537152 | 489 | 0.005510 | 0.079076417 | 47,400 | P3 masked SSL encoders | completed | 0.35 |
| C_s0 (reused) | 494,193 | 0.082367432 | 482 | 0.004906 | 0.082965206 | 47,400 | P3 masked SSL encoders | completed | 0.40 |
| **A_s0** | 488,601 | 0.082189201 | 632 | 0.011543 | 0.083806124 | 188,400 | P3 masked SSL encoders | completed | 2.72 |
| **B_s0** | 448,623 | 0.101720000 | 332 | n/a | n/a (truncated) | 128,269 | P3 masked SSL encoders | CANCELLED_BY_BUDGET @817 | 2.30 |
| **A_SSL_s0** | (pending) | (pending) | | | | 141,300 | A-SSL pretrain → supervised | (pending) | |
| **A_attn_s0** | 488,889 | (pending) | | | | 141,300 | P3 masked SSL encoders | (pending) | |

Horizon-robust check (best valid inside the learning-rate band
`lr ∈ (1e-4, 3e-4]`, i.e. the band in which every reference model sets its best,
so a longer declared horizon cannot flatter a run):

| model | best valid in band | epoch | band epochs | best valid overall |
|---|---:|---:|---:|---:|
| P3B_s0 | 0.076818 | 441 | 102 | 0.076224 |
| L_s0 | 0.078682 | 480 | 102 | 0.078537 |
| C_s0 | 0.082367 | 482 | 102 | 0.082367 |
| A_s0 | 0.082908 | 893 | 203 | 0.082189 |
| B_s0 | 0.107775 | 792 | 53 | 0.101720 |

Paired bootstrap on the 1,000 valid molecules (10,000 resamples, same resample
for every contrast; `delta = AE(first) − AE(second)`, negative = first better):

| contrast | mean delta | 95 % CI |
|---|---:|---|
| A_s0 − P3B_s0 | +0.005965 | [−0.000980, +0.013425] |
| A_s0 − L_s0 | +0.003652 | [−0.004676, +0.011145] |

Reading: A is *behind* on the point estimate but the valid interval still
contains 0 — this is not a demonstrated loss, and the subgroup structure below
shows the deficit is not uniform.

## 3. Arm A: where the error actually is

Fixed subgroups (frozen masks from `zinc_generalization_audit_v1` /
`zinc_dual_channel_composer_v1`, so the groups are comparable across rounds):

| subgroup | n | P3B | L | C | A | A − P3B |
|---|---:|---:|---:|---:|---:|---:|
| signature-unseen | 251 | 0.1513 | 0.1510 | 0.1561 | 0.1529 | +0.0016 |
| signature-covered | 749 | 0.0511 | 0.0543 | 0.0577 | 0.0585 | +0.0074 |
| ordinary (`y > −3`) | 935 | 0.0683 | 0.0655 | 0.0695 | 0.0730 | +0.0047 |
| tail (`y ≤ −3`) | 65 | 0.1905 | 0.2655 | 0.2674 | **0.2147** | +0.0241 |
| rings ≥ 4 | 213 | 0.1130 | 0.1117 | 0.1151 | **0.1119** | −0.0011 |
| rings ≥ 4 & unseen | 47 | 0.3141 | 0.3176 | 0.3216 | **0.2811** | −0.0330 |
| P3B worst-100 | 100 | 0.3590 | 0.3654 | 0.3782 | **0.3053** | −0.0537 |

Two structure facts:

1. **A's deficit is concentrated on the easy/covered molecules** (+0.0074 on 749
   covered), where its error is 0.0585 against a 0.0511 floor. On the hard tail
   it is the best model of the four (0.2147 vs P3B 0.1905 → still worse than
   P3B, but far better than L 0.2655 / C 0.2674), and on P3B's own worst-100 it
   is 0.0537 *better*.
2. **A is better where the graph is bigger.** Valid MAE by size quartile:

| size quartile (atoms) | P3B | L | A |
|---|---:|---:|---:|
| 10–19 | **0.0919** | 0.0884 | 0.1086 |
| 20–22 | 0.0683 | 0.0644 | **0.0637** |
| 23–25 | **0.0683** | 0.0823 | 0.0746 |
| 26–36 | 0.0790 | **0.0785** | 0.0851 |

Small graphs are where the pair composition loses badly (+0.017 on the smallest
quartile); mid-size graphs are where it wins. With `n = 10..19` atoms the
contraction has few triadic terms and the uniform pair moments have little to
average, which is the mechanistic reading of that pattern (not a proven cause).

Error structure: corr(signed error A, P3B) = 0.745, corr(|AE|) = 0.751; A beats
P3B on 46.2 % of molecules, and when it does it wins by 0.052 while losing by
0.056 on the rest. A is a *higher-variance* predictor, not a strictly worse one.

Descriptive diagnostic (valid only, **not** a candidate of this round and not
scored on test): the equal-weight average of the frozen A and P3B valid
predictions is **0.072465** (vs P3B 0.076224, A 0.082189; worst-100 0.3435 vs
0.3590). No weight was fitted; the number only says the two trunks carry
different information. `0.5·(A + L)` gives 0.074002.

## 4. Arm B: negative, and why

B's best valid is 0.101720 at epoch 332, i.e. **+0.0255 versus P3B** and +0.0231
versus A — the multi-point spectral invariants are the weakest readout tried in
this line. It was stopped at epoch 817 of 1,200 (`status:
CANCELLED_BY_BUDGET`): its best had not moved for 485 epochs even after entering
the `lr` band where every reference model sets its best, so no remaining epoch
could change its rank. The curve and record were reconstructed from the epoch
log by `code/zae_recover.py`, which refuses to run unless `best.pt` matches the
best epoch line; nothing was deleted.

Diagnostic reading: the invariants are *sufficient in principle* (they are exact
matrix-function traces, and the triple block does add 8.8 % variance beyond
pair + first-order) but they are **averaged over the whole graph** — a single
vector per (u, v) pair of atom slots. ZINC's target depends on local
composition, and a global trace cannot localise. This mirrors the earlier
finding for junction-level patterns: whole-graph summaries are not a substitute
for structure-resolved aggregation.

## 5. Budget decisions taken mid-round

| run | best valid | best epoch | declared | share of horizon after the best | action |
|---|---:|---:|---:|---:|---|
| A | 0.082189 | 632 | 1,200 | 47 % | completed |
| B | 0.101720 | 332 | 1,200 | 72 % | cancelled at 817 |

A's last 568 epochs bought 0.0016 valid at 47 % of the compute, and B's best was
set at 28 % of its horizon. The two trajectories that had not started were
therefore **declared** at 900 epochs (141,300 updates) instead of 1,200 — a
horizon chosen before launch, not a mid-run reset — and every A-versus-`A_attn`
comparison is additionally reported in the matched `lr` band above.

## 6. A-SSL and A_attn **(pending)**

* A-SSL pretraining: 120 epochs × 149 steps = 17,880 updates, 9500 fit / 500
  holdout molecules, best holdout reconstruction objective 0.1479 at epoch 99,
  0.37 GPU-h, 6,176 temporary heads (reported separately, discarded afterwards).
* (pending) A_SSL supervised best valid, and the pretrain-versus-P3-init delta.
* (pending) A_attn best valid and the pooling verdict against the criterion in
  `analysis/followup_A_attn.md` (≤0.079 / 0.079–0.0822 / ≥0.0822).

## 7. Official test **(pending)**

Frozen before reading test: the four best-valid checkpoints of this round
(`A_s0` 0.082189 @632, `B_s0` 0.101720 @332 truncated, `A_SSL_s0`, `A_attn_s0`)
in `analysis/test_freeze.json` with sha256 per checkpoint. Reference test numbers
(P3B 0.069864841, L 0.066266555, C 0.070014820) are quoted from the existing
dual-channel report.

## 8. What to keep, is there 0.05x evidence, is the next experiment worth buying

**(pending — written after the test read.)**
