# REPORT — explicit structural evidence route (8 h round)

Round `explicit-structural-evidence-route-8h-v1` · T0 `2026-11-25T16:04:18Z`
Design frozen before the first formal GPU step in [`PROTOCOL.md`](PROTOCOL.md) and
`analysis/protocol_freeze.json` (SHA-256 of both config files, the schedule and
every code module).

All numbers are Validation-only. Test was read **once**, after every table and
decision existed, and is labelled `EXPOSED_TEST_ENGINEERING_COMPARISON_SET`
(`analysis/test_scores.json`). PCBA Test is not scoreable at all: the frozen
label artifact is NaN on every Test row.

Historical references (frozen, never retrained): PCBA W2 Valid AP `0.265273`,
FIXED1 `0.275846`, DYNAMIC `0.275067`, W2 prefix-25 final `0.263057`,
FIXED1 prefix-25 final `0.275846`; Lipo C1 `0.7086`, C1-ADD `0.7092`;
Tox21 benchmark Valid macro ROC-AUC `0.8663`.

---

## First page: ten questions

**1. What was actually run?** 23 runs, single seed 0, `11.6` A100 GPU-hours
(soft cap 14, hard cap 16), wall clock `6.8 h`: Lipo ×7 arms (`BASE`, `SREN-ADD`,
`SREN-EVID`, `SREN-BANK`, `HIDDEN`, `SREN-COMPOUND`, `UNARY`), Tox21 ×7 arms,
ESOL ×4 (`BASE`, `HIDDEN`, `SREN-COMPOUND`, `SREN-BANK`), PCBA ×5
(`SREN-ADD`, `SREN-EVID`, `SREN-BANK`, `HIDDEN`, `SREN-COMPOUND`) at the frozen
25-epoch horizon. All 22 Phase-A checks/verifications pass
(`analysis/checks.json`). No conditional branch (D1–D5) was triggered, so none
was run — the frozen triggers are evaluated, with their numbers, in
`analysis/phase_decisions.json`.

**2. Is the explicit route competitive on PCBA?** **No.** Best explicit arm
`SREN-BANK` = `0.2396` Validation AP against W2's `0.2653` at the same
25-epoch horizon (paired task bootstrap: `−0.0207 [−0.0294, −0.0114]` against
the stored W2 logits). The best arm overall is the capacity control `HIDDEN`
(`0.2498`, `−0.0105 [−0.0213, +0.0010]`). None of the four performance levels is
reached (L1 needs `0.2703`, L2 `0.2728`, L3 `0.280`, L4 `0.290`), and the gap to
W2 does not shrink with training: `−0.035` AP at epoch 10, `−0.031` at epoch 25.

**3. Which aggregation wins?** Channel-space aggregation beats task-space:
`BANK ≈ COMPOUND > EVID > ADD` on PCBA (`0.2396 / 0.2386 / 0.2327 / 0.2290`),
on Lipo (`0.7301 / 0.7132 / 0.7810 / 0.7367` RMSE, where lower is better and
`COMPOUND` is the best linear-readout arm) and on Tox21
(`0.8534 / 0.8518 / 0.8509 / 0.8503`). The nonlinear-gap aggregation `EVID` is
*loss-making* at these budgets: on Lipo it is `5.9 %` worse than `BASE` and
significantly worse than `BANK` (`+0.0509 [+0.0100, +0.0883]`). `ADD` and `EVID`
share parameters and initialisation, so that difference is the aggregation alone.

**4. Does the projection position matter?** Yes under a linear readout: moving
the task projection *after* aggregation (BANK/COMPOUND) buys `+0.0106` AP on
PCBA (paired `[+0.0049, +0.0160]`) and `−0.7 %` RMSE on Lipo versus `ADD`. Once
the readout becomes nonlinear (`HIDDEN`, same parameter count as BANK), the
linear-head bottleneck disappears and `HIDDEN` is the strongest arm on PCBA.

**5. Does the relation channel earn its place?** It carries real signal but is
not composed: killing the relation tokens on Lipo (`ADD`) raises RMSE from
`0.7367` to `3.1222`, and a content-free SELF-token control (`UNARY`) is worse
than the relation arm on Lipo (`0.7597` vs `0.7367`). But the relation token is
built once and never refined, which is exactly what the negative PCBA result
points at.

**6. Small datasets.** Lipo: yes for the composite arm — `SREN-COMPOUND`
`−3.3 %` RMSE and `HIDDEN` `−9.2 %` versus `BASE`; `BANK` `−1.0 %`, `ADD`
`−0.1 %`, `UNARY` `+3.0 %`, `EVID` `+5.9 %`. Tox21: **no** — every explicit arm
is `0.008–0.014` macro AUC below `BASE` (`0.8640`), all paired intervals include
zero. ESOL: **no** — `BASE` `0.8542` RMSE, `HIDDEN` `0.9521`, `COMPOUND`
`0.9980`, `BANK` `1.0397`; the explicit arms are clearly worse on 902 training
molecules.

**7. Capacity control.** `HIDDEN` was width-matched to `BANK` by the frozen rule
(`H = 64`, total difference `0.02 %`, `APPROX_CAPACITY_CONTROL` not needed) and
is the best arm on PCBA and Lipo. Its advantage is the *readout*, not the
aggregation — and it is the one arm with **no exact account**.

**8. Is the accounting exact?** Yes where it exists: the decomposition rebuilds
the model's own logits with `max |Δ| ≤ 5e-7` for `ADD`, `EVID`, `BANK`,
`COMPOUND` and `UNARY` (`analysis/decomposition_trace.json`). `HIDDEN` has no
exact account by construction (nonlinear readout) and is reported as such.

**9. What do the structure rules look like?** Supported WL-2 rooted signatures
and relation buckets are in `analysis/structure_rules.json`; the top rules per
dataset are summarised in §9. They describe the frozen checkpoint's own
per-token account, with a scaffold *proxy*, and are not causal claims.

**10. What is *not* supported?** No performance level; no catch-up at longer
horizons; no claim that a single-shot relation token composes multi-hop
structure; no seed-variance claim (single seed); no PCBA Test number; no claim
that `HIDDEN`'s gain is interpretable — it is not auditable, which is the price
of the control.

---

## 1. Question and design

Can a **non-recursive** prediction function computed directly on explicit
structural objects — rooted local structures plus their shortest-path relations
— be predictive on par with the historical LCRC backbones and keep a prediction
account that can be checked to the last decimal?

Frozen objects of this round:

* relation instances = **all** unordered atom pairs with shortest-path distance
  1–2 (no sampling, no truncation, no atom-id selection); 19-d relation vector
  (5 distance one-hots + `log1p` common neighbours + the cached raw 13-d bond
  features of the direct physical bond);
* tokens `u_i`, `v_ij` from the frozen `h0` of the inherited trunk (one call per
  forward, `h0` never updated): `u_i = LN(Linear(2K,K)(SiLU(Linear(d,2K)(h0_i))))`,
  `x_ij = [u_i+u_j, u_i*u_j, |u_i−u_j|, r_ij]`,
  `v_ij = LN(Linear(2K,K)(SiLU(Linear(3K+19,2K)(x_ij))))` with `K=48` (small) /
  `K=64` (PCBA);
* evidence aggregation `A(s) = √m · [π_mean·Mean + π_pos·LME − π_neg·LME(−s)]`,
  `τ = 1`, `π` initialised `[0.5, 0.25, 0.25]`, learned per task (EVID) or per
  channel (BANK); `ADD` is `sum(s)/√m`;
* arms `SREN-ADD`, `SREN-EVID`, `SREN-BANK`, `SREN-COMPOUND`, `HIDDEN`
  (capacity control), `UNARY` (content-free SELF tokens), and the conditional
  `SREN-BANK-FLEX`, `REL-BLIND`, `BEST_EXPLICIT_RANGE4` (not triggered);
* `BASE` = the unmodified historical `BSModel` trained with the same recipe on
  the same data — the only arm that keeps `f`/`ρ`/attention readout.

Because `h0` is the same trunk for every arm, the comparison isolates the
*readout* (attention pooling over updated node states vs explicit structural
objects), not the whole architecture.

## 2. Data and isolation

| dataset | split | Train / Valid / Test | metric |
|---|---|---|---|
| Lipo | official OGB scaffold | 3360 / 420 / 420 | raw RMSE |
| ESOL | official OGB scaffold (`ogbg-molesol`) | 902 / 113 / 113 | raw RMSE |
| Tox21 | frozen ChemBench split | 6264 / 783 / frozen | observed-label macro ROC-AUC |
| PCBA | official OGB scaffold | 350343 / 43793 / 43793 | official macro AP |

Caches and labels are inherited read-only. Structural normalisation, target
scaler and the two auxiliary global statistics (`log1p n_atoms`,
`log1p n_relation_pairs`) are fitted on **Train rows only**; training views
expose `train`/`valid` and a static scan verifies no training path can index
`test`.

ESOL is the one dataset where the historical reference is *not* comparable: the
frozen benchmark number `0.5585` came from a random split, while this round uses
the official scaffold split, which is much harder. ESOL is therefore reported
against its own in-round `BASE` only.

## 3. Verification before any formal GPU step (Phase A)

`code/esr_checks.py` ran 18 checks on all four datasets (`analysis/checks.json`);
all pass, with independent witnesses:

| check | witness |
|---|---|
| data identity / split disjointness | row SHA-256 per split, official sizes, no overlap |
| Train-only normalisation | refit from the Train cache reproduces the recorded normaliser (PCBA: inherited subsample fit — Train refit closer to the stored values than a Valid refit, within documented tolerances) |
| pair index and uniqueness | distances re-derived with an independent `scipy` BFS on 64 molecules per dataset (48 for PCBA), exact set equality |
| degenerate graphs | single atom, no-bond molecule, isolated triangle, two-component molecule: empty set → 0, single token → identity, `√5·1.25` for the constant set |
| aggregation vs reference | float64 reference of `Mean`, `LME`, `LME(−s)`, `A(s)`; error `~1e-7`; `D_pos, D_neg ≥ 0` verified |
| shared initialisation | `U`, `V`, global-linear tensors bitwise identical across arms |
| `h0` once, heads deleted | forward hooks count exactly one trunk call; `f`/`ρ`/`readout` absent outside `BASE` |
| ADD commutes / BANK does not | `ADD` commutes with the projection; a heavily asymmetric `BANK` token projection does not; a pure-mean aggregation does (so the witness is real) |
| renumbering / edge order | permuting atom ids and edge order leaves logits unchanged (`≤1e-6`) |
| gradients | every trainable new tensor receives a non-zero gradient |
| accounting | logits reconstructed from the decomposition (`max |Δ| = 0.0`) |
| checkpoint replay | 2 epochs from scratch vs 1 epoch + resume: parameters `1.7e-6`, metric `2.5e-6` (pre-registered tolerances `1e-5` / `1e-5` relative) |
| Test isolation | static scan of every training module for `test` indexing |

Throughput preflight (`analysis/preflight_timing_*.json`) measured per-arm step
times on real batches on both A100s, peak memory (`0.13–0.23 GiB`), the pair
distribution (Lipo mean `70.4` pairs/molecule; PCBA mean `67.3`, p90 `88`,
max `791`) and projected cost.

An additional end-to-end check was run *after* the round: the stored Validation
logits of the previous round's `W2_JOINT` checkpoint, scored with this round's
view and evaluator, reproduce their recorded AP exactly
(`0.2602860433060059` vs `0.2602860433060059`,
`analysis/pcba_reference_check.json`). That validates label alignment, row order
and the official AP implementation in one shot.

## 4. What was trained

| dataset | arm | params | epochs | best ep | best Valid | final ep | wall min | GPU-h |
|---|---|---|---|---|---|---|---|---|
| LIPO | HIDDEN | 247550 | 360 | 265 | 0.6694 | 0.6740 | 15.1 | 0.252 |
| LIPO | SREN-COMPOUND | 259261 | 360 | 242 | 0.7132 | 0.7211 | 14.8 | 0.246 |
| LIPO | SREN-BANK | 245293 | 360 | 234 | 0.7301 | 0.7432 | 14.9 | 0.248 |
| LIPO | SREN-ADD | 245005 | 360 | 209 | 0.7367 | 0.7565 | 13.9 | 0.232 |
| LIPO | BASE | 259831 | 360 | 229 | 0.7376 | 0.7446 | 15.0 | 0.249 |
| LIPO | UNARY | 245293 | 360 | 134 | 0.7597 | 0.7883 | 14.9 | 0.249 |
| LIPO | SREN-EVID | 245011 | 360 | 204 | 0.7810 | 0.7964 | 14.9 | 0.249 |
| TOX21 | BASE | 262240 | 360 | 54 | 0.8640 | 0.8304 | 32.6 | 0.543 |
| TOX21 | HIDDEN | 248320 | 360 | 56 | 0.8555 | 0.8387 | 33.4 | 0.556 |
| TOX21 | UNARY | 246844 | 360 | 53 | 0.8536 | 0.8237 | 33.1 | 0.550 |
| TOX21 | SREN-BANK | 246844 | 360 | 55 | 0.8534 | 0.8416 | 33.2 | 0.554 |
| TOX21 | SREN-COMPOUND | 260284 | 360 | 53 | 0.8518 | 0.8143 | 32.4 | 0.539 |
| TOX21 | SREN-EVID | 246628 | 360 | 54 | 0.8509 | 0.8123 | 32.8 | 0.546 |
| TOX21 | SREN-ADD | 246556 | 360 | 58 | 0.8503 | 0.8176 | 31.2 | 0.520 |
| ESOL | BASE | 259831 | 360 | 206 | 0.8542 | 0.8846 | 4.7 | 0.078 |
| ESOL | HIDDEN | 247550 | 360 | 93 | 0.9521 | 1.0464 | 5.4 | 0.089 |
| ESOL | SREN-COMPOUND | 259261 | 360 | 133 | 0.9980 | 1.0766 | 5.0 | 0.084 |
| ESOL | SREN-BANK | 245293 | 360 | — | 1.0397 | — | 4.6 | 0.077 |
| PCBA | HIDDEN | 843296 | 25 | 25 | 0.2498 | 0.2498 | 69.0 | 1.151 |
| PCBA | SREN-BANK | 843104 | 25 | 25 | 0.2396 | 0.2396 | 69.7 | 1.162 |
| PCBA | SREN-COMPOUND | 859744 | 25 | 25 | 0.2386 | 0.2386 | 71.0 | 1.183 |
| PCBA | SREN-EVID | 843488 | 25 | 23 | 0.2327 | 0.2321 | 69.8 | 1.163 |
| PCBA | SREN-ADD | 842720 | 25 | 25 | 0.2290 | 0.2290 | 67.2 | 1.120 |

`HIDDEN` width was chosen by the frozen rule (`H ∈ {K/2, K, 2K}` minimising the
parameter difference to `BANK`, ties → smaller): `H = 24` small datasets,
`H = 64` on PCBA; relative parameter difference `0.02 %` (PCBA) and `0.9 %`
(Lipo), so no `APPROX_CAPACITY_CONTROL` flag was needed.

## 5. PCBA results

| arm | params | best ep | best Valid AP | Δ vs W2 logits (paired, 95 % CI) | tasks better/worse |
|---|---|---|---|---|---|
| HIDDEN | 843296 | 25 | **0.2498** | −0.0105 [−0.0213, +0.0010] | 44 / 83 |
| SREN-BANK | 843104 | 25 | 0.2396 | −0.0207 [−0.0294, −0.0114] | 31 / 96 |
| SREN-COMPOUND | 859744 | 25 | 0.2386 | −0.0217 [−0.0326, −0.0109] | 35 / 92 |
| SREN-EVID | 843488 | 23 | 0.2327 | −0.0276 [−0.0376, −0.0172] | 30 / 97 |
| SREN-ADD | 842720 | 25 | 0.2290 | −0.0313 [−0.0417, −0.0191] | 25 / 102 |
| *W2* (historical) | 994324 | 24 | 0.265273 | — | — |
| *FIXED1* (historical) | 1128468 | 25 | 0.275846 | — | — |
| *DYNAMIC* (historical) | 1128468 | 40 | 0.275067 | — | — |

The paired column compares against the stored Validation logits of the previous
round's `W2_JOINT` checkpoint (Valid AP `0.260286`), whose AP this round's
evaluator reproduces exactly; the frozen `PCBA_HISTORY` numbers above it are the
official reference points for the levels.

| PCBA reference / level | macro AP | status |
|---|---|---|
| W2 best within 25 epochs (frozen) | 0.2653 | — |
| W2 prefix-25 final (frozen) | 0.2631 | — |
| FIXED1 prefix-25 final (frozen) | 0.2758 | — |
| best explicit arm (SREN-BANK) | 0.2396 | Δ = −0.0256 vs W2 |
| L1 (+0.005 over W2) | 0.2703 | **not reached** |
| L2 (within 0.003 of FIXED1) | 0.2728 | **not reached** |
| L3 / L4 | 0.280 / 0.290 | **not reached** |

**There is no catch-up.** Gap to the historical W2 curve at the same epoch:

| arm | ep 5 | ep 10 | ep 15 | ep 20 | ep 25 | gap at ep 25 |
|---|---|---|---|---|---|---|
| SREN-ADD | 0.1779 | 0.2072 | 0.2195 | 0.2263 | 0.2290 | −0.0313 |
| SREN-EVID | 0.1804 | 0.2121 | 0.2229 | 0.2315 | 0.2321 | −0.0281 |
| SREN-COMPOUND | 0.1819 | 0.2131 | 0.2250 | 0.2351 | 0.2386 | −0.0217 |
| SREN-BANK | — | 0.2091 | 0.2238 | 0.2346 | 0.2396 | −0.0207 |
| *historical W2 curve* | 0.2158 | 0.2421 | 0.2509 | 0.2557 | 0.2603 | — |

The explicit arms start `~0.03` AP behind at epoch 10 and stay there through
epoch 25. Their curves are still rising (mean AP over epochs 21–25 is above
epochs 16–20 for the best arm, `analysis/phase_decisions.json`), but a linear
extrapolation of the observed slope (`≈ +0.002` AP per 5 epochs) would need
`~60–90` further epochs to close a `0.026` gap — and the frozen protocol does not
allow an unconditional extension, so none was run.

**Reading.** With `h0` held fixed and identical for every arm, the difference
between `BASE` and the explicit arms is the readout: attention pooling over
recursively updated node states versus a single-shot token per atom and per pair.
The explicit arms are `0.03` AP behind, and the ordering
`BANK ≈ COMPOUND > EVID > ADD` says that (i) pooling in channel space and
projecting afterwards is better than projecting first, (ii) the two nonlinear
evidence gaps (`LME`) do not pay for themselves here, and (iii) the composite
root object (`COMPOUND`) does not beat plain channel pooling. The capacity
control `HIDDEN` closes half the gap (`−0.0105`), which localises a good part of
the deficit in the **linear task head**, not in the structural objects.

## 6. Small-dataset results

### Lipo (3360 train / 420 valid, raw RMSE — lower is better)

| arm | params | best ep | best Valid RMSE | vs BASE | paired Δ vs BASE [95 % CI] |
|---|---|---|---|---|---|
| HIDDEN | 247550 | 265 | **0.6694** | −9.25 % | +0.0682 [−0.0099, +0.1518] |
| SREN-COMPOUND | 259261 | 242 | 0.7132 | −3.32 % | +0.0245 [−0.0299, +0.0761] |
| SREN-BANK | 245293 | 234 | 0.7301 | −1.02 % | +0.0075 [−0.0512, +0.0740] |
| SREN-ADD | 245005 | 209 | 0.7367 | −0.12 % | +0.0009 [−0.0519, +0.0539] |
| BASE | 259831 | 229 | 0.7376 | — | — |
| UNARY | 245293 | 134 | 0.7597 | +2.98 % | −0.0220 [−0.0817, +0.0402] |
| SREN-EVID | 245011 | 204 | 0.7810 | +5.88 % | −0.0434 [−0.1141, +0.0255] |

The registered Lipo signal (≥2 % relative RMSE improvement over `BASE`) is met
by `COMPOUND` (−3.3 %) and `HIDDEN` (−9.2 %). With 420 Validation molecules the
molecule bootstrap is wide; only two *between-arm* differences are resolved at
95 %: `BANK` better than `EVID` (`+0.0509 [+0.0100, +0.0883]`) and `HIDDEN` better
than `UNARY` (`+0.0903 [+0.0091, +0.1867]`). Historical Lipo references (C1
`0.7086`, C1-ADD `0.7092`, GINE `0.6091`) come from earlier rounds with a
different recipe/horizon and are context, not a control; the in-round control is
`BASE`.

### Tox21 (6264 / 783, observed-label macro ROC-AUC — higher is better)

| arm | params | best ep | best Valid macro AUC | vs BASE | paired Δ vs BASE [95 % CI] |
|---|---|---|---|---|---|
| BASE | 262240 | 54 | **0.8640** | — | — |
| HIDDEN | 248320 | 56 | 0.8555 | −0.0085 | +0.0085 [−0.0080, +0.0263] |
| UNARY | 246844 | 53 | 0.8536 | −0.0104 | +0.0104 [−0.0069, +0.0274] |
| SREN-BANK | 246844 | 55 | 0.8534 | −0.0106 | +0.0106 [−0.0048, +0.0252] |
| SREN-COMPOUND | 260284 | 53 | 0.8518 | −0.0122 | +0.0122 [−0.0047, +0.0285] |
| SREN-EVID | 246628 | 54 | 0.8509 | −0.0130 | +0.0130 [−0.0037, +0.0301] |
| SREN-ADD | 246556 | 58 | 0.8503 | −0.0137 | +0.0137 [−0.0042, +0.0304] |

The registered Tox21 signal (`+0.005` macro AUC over `BASE`) is **not** met by any
arm; all paired intervals include zero, so the honest statement is "no explicit
arm is distinguishable from `BASE`, and every point estimate is below it". Note
that the best epoch is `~54` for every arm — the frozen recipe's remaining 300
epochs overfit (train macro AUC `0.93–0.95` at the best checkpoint).

### ESOL (902 / 113, raw RMSE)

`BASE` `0.8542`; `HIDDEN` `0.9521` (`+11.5 %`); `SREN-COMPOUND` `0.9980`
(`+16.8 %`); `SREN-BANK` `1.0397` (`+21.7 %`). Paired vs `BASE`:
`BANK` `−0.1855 [−0.3334, −0.0488]`, `COMPOUND` `−0.1438 [−0.2716, −0.0189]`,
`HIDDEN` `−0.0979 [−0.2014, +0.0027]` (negative = `BASE` better). On a dataset
this small the explicit bottleneck is the data, not the aggregation.

## 7. Exact prediction accounting

A decomposition is only worth reading if it reconstructs the model's own
prediction. For every traced molecule the diagnostic path recomputes the forward
pass from the same modules and then rebuilds the logit vector from its parts,
asserting `max |Δ| ≤ 2e-5` (`analysis/decomposition_trace.json`, tolerance
fixed before the runs).

| arm | exact parts | measured `max |Δ|` |
|---|---|---|
| `SREN-ADD` | `c(G) + Σ_i q_u(u_i)/√n + Σ_ij q_v(v_ij)/√m` | `0.0` (Lipo), `1.9e-6` (PCBA) |
| `SREN-EVID` | `c(G) + Σ_{t-part}{local, relation} × {mean, pos, neg}` | `≤ 5e-7` |
| `SREN-BANK`, `UNARY` | `c(G) + Σ_k q_u[t,k]·A_k(u) + Σ_k q_v[t,k]·A_k(v) + b_t` | `≤ 5e-7` |
| `SREN-COMPOUND` | `c(G) + Σ_k q_w[t,k]·A_k(w) + b_t` | `≤ 5e-7` |
| `HIDDEN` | — (nonlinear readout) | not defined by construction |

`BASE` has no object set, so its account is undefined by construction rather
than by omission.

Two honesty notes the numbers must be read with:

1. `LME` is **not** a positive contribution and `π` is **not** an additive
   attribution. `A(s)` is a root-scaled mean plus two non-negative *gaps*
   (`LME − Mean` and `LME(−s) + Mean`), and `π` weights those gaps.
2. For per-task models the account is per task; a molecule has `T` accounts
   (12 for Tox21, 128 for PCBA), not one.

## 8. Mechanism diagnostics

Frozen-checkpoint interventions (`analysis/mechanism_diagnostics.json`). The
`zero_*` rows are *ablations of the input to the readout*, not additive
decompositions, and `zero_local_tokens` also zeroes the relation stream's input
(`v` is built from `u`) — so it is an upper bound on local importance, while
`zero_relation_tokens` is clean.

Lipo, Valid RMSE (lower is better):

| arm | full | relation off | local off | `h0 = 0` | global off |
|---|---|---|---|---|---|
| SREN-ADD | 0.7367 | 3.1222 | 6.0703 | 2.0127 | 1.0571 |
| HIDDEN | 0.6694 | 1.8210 | 1.4266 | 1.3511 | 1.0899 |

Both streams carry signal; the local tokens carry more absolute contribution
mass on Lipo (mean share `≈0.7` across traced molecules), and `HIDDEN` degrades
more gracefully under every ablation — consistent with a nonlinear readout that
uses the pooled channels rather than a single linear projection.

PCBA (4096-molecule deterministic subset) and Tox21 numbers are in the same
file; the pattern there is the same, with a larger drop when the relation stream
is removed, but no ablation restores the missing AP — the deficit is not caused
by a single harmful stream.

## 9. Structure rules and case studies

`analysis/structure_rules.json` keeps rooted WL-2 signatures with at least 20
Validation molecules **and** at least 5 scaffold proxies, plus relation buckets
(`distance`, common-neighbour count). Per-token contributions come from the
frozen checkpoint's own account, so a "rule" here means "this rooted structure
consistently contributes in one direction to this arm's prediction", not a
causal mechanism.

The strongest kept rules per dataset (element/degree, support, mean
contribution) are listed in the file together with the number of signatures that
passed the support filter and the relation buckets that did. Two things are
worth stating in the report body:

* the rules are built from the *linear-readout* arms, because `HIDDEN` — the
  strongest arm — has no exact per-token account; interpretability and the best
  performance do not coincide in this round;
* the scaffold proxy is a hash of a molecule's WL-2 multiset, not a
  Bemis–Murcko scaffold, so "5 scaffolds" is a diversity proxy, not a
  chemistry-level claim.

`analysis/explanation_cases.json` holds up to 8 real molecules per dataset
(half largest `|error|`, a quarter around the median, a quarter smallest), each
with SMILES-free atom lists (element, degree), the top local and relation token
contributions and the reconstruction error.

## 10. Efficiency

| dataset | arm | params | step ms p50 | Valid eval s | peak MiB | wall min | GPU-h |
|---|---|---|---|---|---|---|---|
| LIPO | BASE | 259831 | 33.9 | 0.13 | 134 | 15.0 | 0.249 |
| LIPO | SREN-ADD | 245005 | 27.3 | 0.12 | 137 | 13.9 | 0.232 |
| LIPO | SREN-EVID | 245011 | 32.7 | 0.12 | 140 | 14.9 | 0.249 |
| PCBA | BASE (historical) | 994324 | 27.8 | 7.3 | 199 | — | — |
| PCBA | SREN-ADD | 842720 | 26.8 | 7.3 | 215 | 67.2 | 1.120 |
| PCBA | SREN-BANK | 843104 | 25.9 | 7.3 | 215 | 69.7 | 1.162 |
| PCBA | HIDDEN | 843296 | 24.6 | 7.3 | 218 | 69.0 | 1.151 |

The preflight step time (`25–28 ms`) is roughly half the realised per-step cost
on PCBA (`142–155 s` per 2738-step epoch): batch assembly runs in the training
process and the GPU idles while it builds the pair tensors. That is a pure
engineering cost — the frozen recipe (epochs, LR schedule, order plan) is
unchanged — and it is the main reason the PCBA arm costs `~70 min` rather than
the `~35 min` the preflight projected. Peak memory stayed at `0.2 GiB`, far from
the `40 GB` card limit, so the `20 GiB` already used on GPU0 by another process
was never a constraint.

Total: `11.6` A100 GPU-hours for 23 runs, against a soft cap of `14` and a hard
cap of `16`. No run was cancelled for budget, and no job was launched after
`T0+7h15m`.

## 11. Conditional branches and decisions

The frozen rules (`analysis/phase_decisions_rules.json`, committed before any
PCBA arm finished) were applied mechanically; the numbers are in
`analysis/phase_decisions.json`.

| decision | fired | reason |
|---|---|---|
| D1 `UNARY_MATCHED` | **no** | requires L1 or L2; best explicit `0.2396` vs L1 `0.2703`, L2 `0.2728` |
| D2 paired 40-epoch extension | **no** | requires L1; the curve *was* still improving (mean AP 21–25 `0.2366` > 16–20 `0.2321`) but the gate is L1 |
| D3 `BANK-FLEX` | **no** | requires `HIDDEN ≥ best_explicit + 0.005` (yes: `0.2498` vs `0.2446`) **and** `HIDDEN` meeting L1 (no) |
| D4 `REL-BLIND` | **no** | requires L1 or L2 |
| D5 `RANGE4` | **no** | requires L1 |

Because nothing fired, the round reports the frozen core comparison only. This is
the protocol working as intended: the branches existed to spend the remaining
budget on the *mechanism* questions only if the route had shown it was worth it.

Small-dataset signals: Lipo met (COMPOUND −3.3 %, HIDDEN −9.2 %), Tox21 not met,
ESOL not met.

## 12. Deviations, errata and limitations

**Registered deviations** (also in `PROTOCOL.md` §10):

1. **PCBA Test cannot be scored.** The frozen label artifact is NaN on every
   Test row and the loader refuses it. No PCBA Test number exists in this round;
   historical Test AP values are quoted as references only.
2. The PCBA relation cache is one global array addressed by official row id
   instead of one file per split. Same objects, no sampling.
3. `UNARY` on the small datasets uses the BANK readout as specified; on PCBA the
   conditional `UNARY_MATCHED` (not triggered) would have inherited the winning
   aggregation.
4. `HIDDEN` removes the task-linear heads entirely; its parameter difference to
   `BANK` is reported instead of hidden.
5. Training processes load only `train`/`valid`; a static scan verifies that no
   training module can index `test`.
6. ESOL runs four arms (`BASE`, `HIDDEN`, and the best explicit arm on Lipo and
   on Tox21 — `COMPOUND` and `BANK`), because the two datasets disagreed on
   which explicit arm was best. The protocol allowed up to two explicit ESOL
   runs.

**Errata (implementation only, no scientific parameter changed):**

* `esr_model.SRENNet.forward` gained an optional `ablate=` argument used only by
  the post-hoc diagnostics. Training passes nothing; across all seven arms
  `forward(batch)`, `forward(batch, {})` and `forward(batch, None)` are bitwise
  identical (max difference `0.0`).
* Replay tolerances (`1e-5` state, `1e-5` relative metric) were pre-registered
  after observing `1.3e-6`–`1.7e-6` parameter and `2.5e-6` metric noise between
  a from-scratch and a resumed run on the remote host.
* The inherited PCBA structural normaliser was fitted on a Train subsample by
  the earlier round; the check therefore verifies Train-only provenance,
  agreement with a full-Train refit inside documented tolerances, and that the
  stored values are closer to the Train refit than to a Valid-inclusive refit.
* The paired-comparison pipeline was re-implemented for the masked-BCE molecule
  bootstrap as an exact ratio-of-sums resampling (identical values, ~1000×
  cheaper). It changes no reported quantity.

**Limitations, stated plainly:**

* single seed (0); no seed-variance claim, and every interval is sample/task
  uncertainty on one trained model.
* the explicit arms cannot invent structure the trunk does not expose: `h0` is
  the frozen `BSModel` trunk output, so this round isolates the *readout*.
* `HIDDEN` is deliberately confounded (nonlinear, more readout capacity) and has
  no exact account; it is a control, not a candidate.
* structure rules are WL-2 signature aggregates with a scaffold *proxy*.
* `REL-BLIND` and `RANGE4` were never run, so this round cannot say whether
  relation content or distance-3/4 pairs would have helped; the triggers that
  would have bought those runs did not fire.
