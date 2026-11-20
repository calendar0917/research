# PROTOCOL — Frozen response dictionary: numerical check and explanation acceptance (Lipo, seed 0)

Status: frozen **before** the formal analysis. Base commit
`4fc17e33b6cc7fd2f301e05d705183787162e233`. Both source rounds are read-only:

* `response_dictionary_small_s0_v1/` (frozen Lipo RESP best-Val, epoch 63);
* `response_dictionary_lipo_followup_s0_v1/` (K=16 dictionaries, edit replay).

No new training. No new seed. No new data. No K sweep, no selector sweep, no
change to prototypes, panel, `D0`, split, model, optimiser or cache. Test is
evaluated only under the existing `DEVELOPMENT_TEST` authorization and never
selects a prototype, a solver setting or an explanation.

## Questions (only these)

1. Does the K=16 `D_RESPONSE` full-`F` fidelity advantage survive a
   high-precision projection, or is it an artefact of the frozen 64-step
   Frank–Wolfe budget?
2. For a real structural replacement, can the "word response" explain the full
   model prediction change, and how large is the unexplained remainder?
3. Which statements are model-internal, and which have real held-out label
   support?

This is not a chemical causal validation.

## Frozen reuse

* `fu_dict_k16.py` `k_map` / `build_dictionaries` with the same rule: per `tau`
  class `K = 16` if Train candidates > 64 else `min(64, candidates)`; three
  compressed classes (442/81/91 candidates), 13 classes exact storage; 136
  words / 702 candidates.
* `rd_dictionary.py` inventory, exact fragment identity, `D0`, 16-environment
  Train panel, response matrix, frozen `z/a/q/k/b/w`, one-hot prototypes.
* Frozen Lipo RESP state `c42fc25f…` (epoch 63), fragment cache `ba6e9a41…`,
  official cache `fe002287…`, official split.
* The frozen 64-step Frank–Wolfe solution is replayed first; its three dictionary
  predictions must match the follow-up `stress_predictions.npz` before any new
  number is used. Native RESP `F(G)` must match the saved predictions to float
  round-off.
* Response matrices and all solves stay in the model's original (standardised)
  output units; `mean/std` are applied exactly once, after the final `F`
  aggregation.

## Projection certificate

For every non-prototype fragment of a compressed class and each dictionary, the
frozen problem is

```
P: K x m        rows = the K prototype responses y_k on the fixed m=16 panel
y: m            target response of the fragment
f(alpha) = mean((P.T @ alpha - y)**2),  alpha >= 0, sum(alpha) = 1
g(alpha) = 2 * P @ (P.T @ alpha - y) / m
gap(alpha) = g(alpha).dot(alpha) - min(g(alpha))
```

`gap` is the first-order optimality certificate of this convex problem:
`f(alpha) - f(optimum) <= gap` (floating-point tolerance allowed).

Frozen pass rule for one row (numerical precision, not a scientific effect
threshold):

```
gap          <= 1e-8 * max(1, mean(y**2))
min(alpha)   >= -1e-10
|sum(alpha)-1| <= 1e-9
```

Reported for the old 64-step solution per row: `f`, `gap`, simplex violation;
per class x selector: median / p95 / max and pass fraction. Step count alone is
never used as proof of convergence.

## High-precision projection (three compressed classes only)

* Float64; `scipy.optimize.minimize(method="SLSQP")` with the exact analytic
  gradient above, `bounds=[(0,1)]*K`, one equality constraint `sum(alpha)=1`,
  `ftol=1e-12`, `maxiter=2000`.
* Two deterministic starts per row: the old FW64 `alpha` and the uniform
  `alpha = 1/K`. These are numerical starts, not new training seeds.
* Prototypes remain one-hot. The other 13 classes keep their frozen exact
  storage and their existing FW64 `alpha` for any non-Train target fragments.
* No ridge, entropy, sparsity or other regulariser; the objective is unchanged.
* A solution is "qualified" only if it meets the pass rule above. If both starts
  fail, the row is `NUMERICALLY_UNRESOLVED` and keeps the FW64 `alpha` for every
  downstream full-`F` number (stated explicitly).
* Main solution per row: the qualified solution with smaller `f`; if the
  `f` difference is below the row's numerical tolerance
  `1e-8 * max(1, mean(y**2))`, prefer the old-FW start.
* The primary full-`F` "high precision" version (`HP_CERTIFIED`) uses the main
  solution only where at least one start qualified; it keeps FW64 on
  `NUMERICALLY_UNRESOLVED` rows. A second, explicitly **uncertified** sensitivity
  version (`HP_SLSQP_ALL`) replaces every `NUMERICALLY_UNRESOLVED` row by the
  smaller-`f` feasible SLSQP start (FW start on ties within the row tolerance).
  It is reported only to check whether the selector comparison is driven by the
  subset that happened to certify; no precision claim is attached to it.
* For every ambiguous row (both starts qualified, panel `f` difference below
  tolerance, different `alpha`) report the actual Val/Test readout difference:
  per-view `s_D(C,S)` difference, affected views/molecules, and the induced
  full-`F` difference. If material, the label is
  `PANEL_UNDERDETERMINES_PREDICTION`; the panel is not enlarged and no
  tie-breaker is invented.

Full `F` is recomputed for FW64 and high precision for all three dictionaries:
Train/Val/Test RMSE, MAE, fidelity `sqrt(mean((F_D-F_RESP)^2))`, in raw units.
The same table carries both versions.

## Explanation object: the full prediction change

Pairs: read-only `lcrc_lipo_edit_supervision_public122_s0_v1/analysis/pairs.csv`
(823 pairs), mapped by RDKit canonical SMILES to official rows; labels and split
re-derived from the official row id. Orientation follows the catalogue
`row_a -> row_b` (`A = row_a`, `B = row_b`); `delta_y = y_B - y_A`. No pair is
re-mined and none is selected by model behaviour. Report 791 / 5 / 27 / 0 cells
and the 7 Test molecules / 7 components if reproduced.

### Common context

For each pair, search the existing fixed (<=4) views of both endpoints for the
same canonical `C` identity **and** the same cut-bond attributes:

```
G_A = C (+) S_A,   G_B = C (+) S_B
```

If several, take the `C` with the largest atom count; tie: smallest canonical
fragment key. The choice never uses labels or errors. Both `S_A` and `S_B` must
have a supported dictionary encoding. Otherwise record
`NO_SHARED_SELECTED_CONTEXT` (no common selected view) or
`SHARED_CONTEXT_UNSUPPORTED` (encoding missing). No new split and no approximate
core is invented. Coverage over all 823 pairs is reported by split and by
independent endpoint component.

### Local vs full change

For each dictionary and each version (FW64 / high precision):

```
delta_full = F_D(G_B) - F_D(G_A)
delta_local = s_D(C,S_B) - s_D(C,S_A)
remainder   = delta_full - delta_local
```

`F_D` is the actual fixed-view mean with the existing fallback rule; a single
`s_D` never substitutes for it. Report remainder RMSE / MAE / median / p90 |·|,
next to the Train label standard deviation and `|delta_full|`; no relative ratio
is formed from values near zero. Also report the true `delta_y`, the error of
`delta_full` and of `delta_local` against `delta_y`, and the zero-change error
`|delta_y|` as reference. This separates "explains the model" from "predicts the
property".

### Word account

With `S_A` as the explicit reference and `alpha(S_B)` on the actual dictionary:

```
contribution_k = alpha_k(S_B) * [ s_D(C,D_k) - s_D(C,S_A) ]
sum_k contribution_k = delta_local
delta_full = sum_k contribution_k + remainder
```

`D_k` are the actual stored prototypes, one-hot as defined; `alpha >= 0`,
`sum alpha = 1`. Closure is verified numerically but closure is not correctness.
`C (+) D_k` is a model readout probe, not a validated chemical replacement.

## Cases

At most 12, ordered by `pair_id`: the first 6 interpretable pairs with at least
one non-Train endpoint, then the first 6 interpretable Train–Train pairs; if
fewer, fill with Train examples under the same ordering. No selection by effect
size. Each case lists: both actual molecules, the shared `C` and the edited
fragment locations, the main prototype `D_k` structures and ports, `alpha` and
the main word-response terms, `delta_local`, `remainder`, `delta_full`,
`delta_y`, the FW64 vs high-precision change, split, whether the fragment is a
Train word, and the independent component id. RDKit is used only for local
display of existing structures; if rendering is not available, exact fragment
keys, atom/bond tables and the numeric table replace the figure. Train cases are
never counted as independent task validation.

## Compression statement

Report the actual inference path: whether `alpha` still needs the fragment
encoder and the original `a/q` computation, which parts reduce to a Train word
lookup, how unknown fragments are handled, and the stored weight/shape counts.
`702 -> 136` words is not by itself "a smaller/faster network".

## Labels (fixed before the run)

* projection: `RESPONSE_ADVANTAGE_SURVIVES_HIGH_PRECISION` /
  `RESPONSE_ADVANTAGE_STEP_LIMITED` / `RESPONSE_ADVANTAGE_REVERSED`;
* per-row: `PROJECTION_CERTIFIED`, `NUMERICALLY_UNRESOLVED`,
  `PANEL_UNDERDETERMINES_PREDICTION`;
* explanation: `WORD_ACCOUNT_CLOSES` (numeric closure only), plus counts and
  residuals; no causal label;
* model-internal vs held-out is reported separately; held-out edit support is
  still the 27 Train–Test / 5 Train–Valid pairs of the follow-up.

No new training, seed, selector scan, data expansion or automatic edit
augmentation. Any ambiguity is reported, not tuned away. The research decision
returns to the researcher.

## Budget and stop rules

CPU only; no GPU is needed (frozen feature cache and linear algebra). Target
wall <= 30 min, hard stop 1 h; no GPU unit is authorised in this round. Old
result directories are read-only. If the replay identity check fails, the run
stops instead of forcing numbers to match. If fewer than 6 interpretable held-out
pairs exist, that is reported as a limitation, not filled with Train cases.
