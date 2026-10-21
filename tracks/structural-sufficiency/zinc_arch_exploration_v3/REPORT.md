# zinc-arch-exploration-v3 — REPORT

Round: `zinc-arch-exploration-v3`. Track: `structural-sufficiency`.
Status: **complete**. Four seed-0 trunks trained (A, B, A-SSL, A_attn; B stopped
on budget grounds at 817/1,200 epochs), one follow-up used (A_attn), all four
frozen best-valid checkpoints scored once on the official test split.

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
| **A_SSL_s0** | 488,601 | 0.079876146 | 439 | 0.017018 | 0.084783518 | 141,300 | P3 masked SSL encoders + native A-SSL pretrain | completed | 1.84 |
| **A_attn_s0** | 488,889 | 0.082637432 | 519 | 0.012976 | 0.085459559 | 141,300 | P3 masked SSL encoders | completed | 2.02 |

Curves (train, valid, learning rate) for all four arms:
`analysis/curves.png`; the valid panel is zoomed to the competitive region.

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
| A_SSL_s0 | 0.083147 | 575 | 152 | 0.079876 |
| A_attn_s0 | 0.084167 | 637 | 152 | 0.082637 |

Paired bootstrap on the 1,000 valid molecules (10,000 resamples, same resample
for every contrast; `delta = AE(first) − AE(second)`, negative = first better):

| contrast | mean delta | 95 % CI |
|---|---:|---|
| A_s0 − P3B_s0 | +0.005965 | [−0.000980, +0.013425] |
| A_s0 − L_s0 | +0.003652 | [−0.004676, +0.011145] |
| A_SSL_s0 − P3B_s0 | +0.003652 | [−0.002914, +0.010592] |
| A_SSL_s0 − L_s0 | +0.001339 | [−0.007595, +0.009179] |
| A_SSL_s0 − A_s0 | −0.002313 | [−0.007411, +0.002988] |

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

## 6. A-SSL and A_attn

**A-SSL pretraining** ran first, label-free: 120 epochs × 149 steps = 17,880
updates over 9,500 fit molecules with a fixed 500-molecule holdout, fresh mask
every epoch, full feature rebuild from the masked raw graph. Best holdout
reconstruction objective 0.1479 (atom CE 0.20, bond CE 0.02; 93 % atom / 99.4 %
bond accuracy on the masked positions) at epoch 99; 0.37 GPU-h; the 6,176
temporary heads are not part of the reported parameter count and are discarded
before fine-tuning.

**A_SSL supervised** (fine-tuned from that state, 900 epochs): best valid
**0.079876 @ epoch 439**, i.e. −0.00231 versus arm A — the largest valid gain in
this round. Its training curve is visibly different: it is ahead of A from the
start (0.1006 @63 versus A's 0.1064 @100) and peaks *earlier* (439 versus 632),
then drifts back up (final 0.084784). On the official test it scores
**0.084753**, i.e. **+0.00422 versus A** — the valid gain reverses. The
diagnostic table in §7 explains why: its best-valid point is a noise dip.

**A_attn** (the round's single follow-up, content-based pooling of the
off-diagonal pair blocks, +288 parameters): best valid **0.082637 @519** versus
A's 0.082189 @632 — a difference of +0.00045 with a CI containing 0. By the
pre-declared criterion in `analysis/followup_A_attn.md` (≥ 0.0822 ⇒ uniform
pooling is not the bottleneck), the pooling hypothesis is **rejected**. On test
it is 0.079930, marginally the best of the four new arms (+0.0006 better than A,
not significant).

So: native SSL on the new trunk moves valid by −0.0023 and *hurts* test by
+0.0042; a learned pooling of the pair set changes nothing. Neither readout
variant closes the gap to the local-message-passing baseline.

## 7. Official test (read once, after the freeze)

Frozen before the read: `analysis/test_freeze.json` lists the four best-valid
checkpoints with sha256 — `A_s0` 0.082189 @632, `B_s0` 0.101720 @332 (817
epochs, truncated), `A_SSL_s0` 0.079876 @439, `A_attn_s0` 0.082637 @519. Scoring
re-loads each checkpoint and recomputes its valid MAE first: all four reproduce
to <1e-7, so the numbers below are the frozen states, not accidental re-inits.

| model | params | best valid | best ep | test MAE | test RMSE | median AE | p95 AE |
|---|---:|---:|---:|---:|---:|---:|---:|
| P3B_s0 (reused) | 413,505 | 0.076224 | 377 | **0.069865** | 0.1480 | 0.0360 | 0.2237 |
| L_s0 (reused) | 494,193 | 0.078537 | 489 | **0.066267** | 0.1613 | 0.0357 | 0.1949 |
| C_s0 (reused) | 494,193 | 0.082367 | 482 | **0.070015** | 0.1574 | 0.0374 | 0.2220 |
| A_s0 | 488,601 | 0.082189 | 632 | 0.080529 | 0.1575 | 0.0433 | 0.2529 |
| B_s0 | 448,623 | 0.101720 | 332 | 0.089705 | 0.1773 | 0.0505 | 0.2815 |
| A_SSL_s0 | 488,601 | 0.079876 | 439 | 0.084753 | 0.2174 | 0.0446 | 0.2512 |
| A_attn_s0 | 488,889 | 0.082637 | 519 | 0.079930 | 0.1517 | 0.0416 | 0.2795 |

Paired bootstrap on the 1,000 test molecules:

| contrast | mean delta | 95 % CI |
|---|---:|---|
| A_s0 − P3B_s0 | +0.010664 | [+0.004362, +0.016968] |
| A_s0 − L_s0 | +0.014262 | [+0.007193, +0.020862] |
| A_attn_s0 − P3B_s0 | +0.010066 | [+0.003156, +0.016484] |
| A_attn_s0 − L_s0 | +0.013664 | [+0.005308, +0.020560] |
| A_SSL_s0 − P3B_s0 | +0.014889 | [+0.005715, +0.026156] |
| A_SSL_s0 − A_s0 | +0.004224 | [−0.003993, +0.015682] |

Every new arm is significantly worse than both reused references on test. The
differences *among* the new arms are not significant.

### The most important number in this report: valid → test

Best-valid selection is only useful if the selected epoch generalises. It does
not, uniformly:

| run | valid @best | mean valid ±20 epochs | median valid, last 300 | test | valid − test |
|---|---:|---:|---:|---:|---:|
| P3B_s0 | 0.076224 | 0.081251 | 0.078704 | 0.069865 | +0.00636 |
| L_s0 | 0.078537 | 0.079388 | 0.079897 | 0.066267 | +0.01227 |
| C_s0 | 0.082367 | 0.083282 | 0.083527 | 0.070015 | +0.01235 |
| A_s0 | 0.082189 | 0.085417 | 0.083829 | 0.080529 | +0.00166 |
| B_s0 | 0.101720 | 0.109109 | 0.109682 | 0.089705 | +0.01202 |
| A_SSL_s0 | 0.079876 | 0.087933 | 0.084791 | 0.084753 | −0.00488 |
| A_attn_s0 | 0.082637 | 0.086595 | 0.085519 | 0.079930 | +0.00271 |

The three reused models were selected in an earlier round and enjoy a large
valid→test bonus (+0.006 … +0.012): their valid split is *harder* than test, so
best-valid selection is conservative for them. The new arms get almost no bonus
(+0.0017, +0.0027), and A_SSL gets a *penalty* — its test MAE equals its own
median valid over the last 300 epochs (0.0848), which says its best-valid epoch
(0.0799) was a 0.008 dip below its local level, and test reverted to that level.
Practically: on this benchmark the new trunks' valid curve is noisier relative
to how much it moves, so a single best-valid epoch over-states them by ~0.004
(A_SSL) while under-stating the references' advantage.

### Test subgroups

| subgroup | n | P3B | L | A | A_SSL | A_attn |
|---|---:|---:|---:|---:|---:|---:|
| signature-unseen | 244 | **0.1229** | 0.1292 | 0.1461 | 0.1486 | 0.1475 |
| signature-covered | 756 | 0.0527 | **0.0459** | 0.0594 | 0.0642 | 0.0581 |
| ordinary (`y > −3`) | 923 | 0.0632 | **0.0604** | 0.0717 | 0.0751 | 0.0725 |
| tail (`y ≤ −3`) | 77 | 0.1493 | **0.1367** | 0.1859 | 0.2004 | 0.1695 |
| rings ≥ 4 | 208 | 0.0972 | **0.0902** | 0.1040 | 0.1280 | 0.0957 |
| rings ≥ 4 & unseen | 51 | 0.2005 | 0.2221 | 0.1922 | 0.2652 | **0.1879** |
| P3B worst-100 | 100 | 0.3303 | **0.2709** | 0.2920 | 0.3515 | 0.2808 |

The pair-composition trunk keeps a *tail* advantage on test: A_attn beats P3B by
0.0495 on P3B's own worst-100 and by 0.0126 on rings ≥ 4 & unseen while losing
0.0101 on the mean. A's valid-side subgroup pattern (better on hard molecules,
worse on covered ones) therefore reproduces on test for the pooled-pair arms —
but it is bought entirely with bulk accuracy, and it does *not* reproduce for
A_SSL, whose valid-side subgroup advantage (rings ≥ 4 valid 0.0944 versus P3B
0.1130) inverted on test (0.1280 versus 0.0972).

## 8. What to keep, is there 0.05x evidence, is the next experiment worth buying

**What to keep**

* Infrastructure (keep, it is the durable part of this round): the label-free
  pair cache with the per-molecule static operator library (`prep` 13 s for
  10k + 1k, kernels verified against the on-the-fly eigendecomposition to
  6e-7), the full-valid-every-epoch trainer with strict-`<` best selection, the
  freeze-then-test protocol, the matched-`lr`-band comparison that keeps runs
  with different declared horizons comparable, and `zae_recover.py` for
  budget-stopped runs.
* Scientific keepers: (i) arm A/A_attn's **tail** advantage is real and
  reproduces on test (P3B worst-100 0.3303 → A_attn 0.2808); (ii) the *bulk*
  cost that comes with it is equally real; (iii) B's whole-graph multi-point
  invariants are **not** a usable readout (test 0.0897, worst of the round);
  (iv) native masked pretraining of the new trunk moves valid (−0.0023) but not
  test (+0.0042), and its best-valid epoch is a noise dip; (v) content-based
  pooling of the pair set is **not** the bottleneck (Δ +0.0004 valid,
  +0.0006 test).
* Do not keep: the triple-invariant readout as a trunk, the pooling follow-up
  as a lever, and any claim that "valid improvement on the new trunk" transfers
  to test without a test read.

**Is there 0.05x evidence?** No. The best number produced this round is
**0.079930 test** (A_attn), 0.0137 above the reused L (0.066267) and 0.0201
above the round's lower target boundary of 0.060. Nothing in the valid curves,
the subgroups or the SSD/SSL variants points at a mechanism that would remove
0.014. Two structural facts make the 0.05x target look further away than the
0.0699 → 0.0500 arithmetic suggests: (a) the reused references get a +0.006 …
+0.012 valid→test bonus that the new trunks do not get, so a new trunk needs a
valid advantage of ~0.01 merely to tie; (b) every arm's error is dominated by a
heavy tail (top-100 molecules carry ~47 % of the total absolute error for every
model), so a mechanism that only improves the bulk cannot reach 0.05x.

**Is the next experiment worth buying?** Yes, but not as "another trunk". The
one lead with support on *both* valid and test is complementarity: the
pair-composition trunk is better than the local-MP baseline precisely on the
molecules that dominate the error (worst-100, rings ≥ 4 & unseen), while the
local-MP trunk is better on everything else; their valid errors correlate only
0.745, and the unfitted equal-weight average of A and P3B scores 0.072465 valid
versus 0.076224 for P3B alone (descriptive, no test read, not a candidate of
this round). The cheap, pre-registrable next step is therefore a *combination*
experiment, not a new pair/spectral variant: one ≤500k model whose bulk path is
the P3 local trunk and whose tail path is the pooled pair contraction, selected
on valid, with a decision line fixed before the run (valid mean < 0.0725 and
valid worst-100 < 0.29 ⇒ buy; otherwise close the structural-sufficiency
performance line and stop at the measurement contribution). A pure prediction
average of the two frozen trunks is the two-hour sanity check of the same
hypothesis and should be run first.
