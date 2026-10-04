# REPORT — `zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1`

**Scientific question.** Under the same compressed skeleton and capacity, does a shared local
dictionary with direct access to the correspondence `(phi_v, a_v, a, t)` through the real incidence
`C_v[t,a]` generalise the chemical target `g` better than combining the same raw attributes by
their marginals `C_ind_v[t,a] = n_t·n_a/d_v`?

**Answer in one line.** `INCONCLUSIVE` / `EXPLORATORY_PERFORMANCE_SIGNAL`: J is directionally better
than I (dev G0 calibrated g-MAE `0.100416` vs `0.103992`; paired raw gain `+0.00578`, 95% CI
`[+0.00190, +0.00968]`) and directionally better than the historical B, but the pre-registered
calibrated gate thresholds are not met (`G_IJ` G0 cal `+0.00358`, CI `[-0.00023, +0.00727]`;
`G_BJ` G0 cal `+0.00146`, CI `[-0.00289, +0.00601]`), so no positive claim is made.

All results below use the frozen fold (fit 8000 / dev 2000), seed 20261004, dev only, reference B
read-only.

## Six closing questions

### Q1. What was frozen, and did every pre-training operator/contract check pass?

Yes. Frozen before any dev score: `PROTOCOL.md`, `METHOD_CONTRACT.md`, `EVIDENCE_SCOPE.md`,
`ERRATA.md`, the committed `local_tuple_index.npz` (sha256 `4facb6ec…`), commit `83d910727fd4`.

* Incidence reference: `C`, `d_v`, `n_t`, `n_a` rebuilt from raw `edge_index` match the frozen env
  cache exactly (37 molecules / 879 roots, 0 mismatches); all 249,279 undirected edges are symmetric
  doubles with consistent attributes.
* Marginal identity `C_ind = n_t·n_a/d_v`: max Δ `0.0` (float64) / `6.0e-8` (float32).
* Production-model equivalence: single-vs-batch, two-graph offset and shuffled-order restore all
  `0.0`; J/I share the same `D_loc` and produce bit-identical codes under the same weights.
* Correspondence witness: synthetic star `e_I` identical across two graphs with different real
  incidence (`0.0`), `e_J` differs; among exact-duplicate raw `phi65` fit roots, 744 duplicate
  groups (565 with different `C`) — the operator has real content beyond the marginals.
* Contrast: 38,722/185,204 fit roots (20.9%) have `C ≠ C_ind`; 42.8% of union-support fit pairs have
  `w_J ≠ w_I`.
* B reproduction: fit-median bias recomputed `−0.0313783`; fit cal `0.029280`, dev overall cal
  `0.103717`, dev G0 cal `0.101875`, max Δ `3.4e-7` vs published.
* GPU smoke `all_ok=true` (init identity eval `6.6e-7`, target-never-read, same dropout stream,
  endpoint offsets `< 5e-7`, `W_loc` grad nonzero, `D_loc` path established after `W_loc`
  perturbation, replay `< 2e-6`).

### Q2. Did either arm beat the historical B (Gate A)?

Dev calibrated g-MAE:

| arm | dev G0 (k=0) | dev overall | G0 raw | overall raw |
|-----|--------------|-------------|--------|-------------|
| B (M_g soup) | 0.101875 | 0.103717 | 0.107251 | 0.108956 |
| J | **0.100416** | **0.101907** | 0.102679 | 0.104099 |
| I | 0.103992 | 0.104682 | 0.108463 | 0.108945 |

Paired gains (positive = arm better than B), `N_BOOT=1000`, shared resamples, seed 20261004:

| gain | G0 cal | G0 raw | overall cal | overall raw |
|------|--------|--------|-------------|-------------|
| `G_BJ` | +0.001459 `[-0.002886, +0.006012]` | +0.004572 `[+0.000205, +0.009014]` | +0.001810 `[-0.002073, +0.005961]` | +0.004857 `[+0.000804, +0.009270]` |
| `G_BI` | −0.002117 `[-0.005766, +0.001768]` | −0.001212 `[-0.005047, +0.002614]` | −0.000965 `[-0.004639, +0.002578]` | +0.000011 `[-0.003654, +0.003569]` |

Gate A requires calibrated G0 and overall gains ≥ +0.003 with CI_low > 0 and raw gains > 0. J's
calibrated points are below the threshold (G0 `+0.001459`, overall `+0.001810`) with CIs crossing
zero, so **`EXPLORATORY_PERFORMANCE_SIGNAL`**: J is directionally and raw-significantly better than
B, but the calibrated performance gate is not passed; I is not better than B (raw CIs cross zero,
calibrated points negative). Calibration bias per arm: B `−0.031378`, J `−0.016432`, I `−0.028504`.

### Q3. Joint vs independent (Gate B)?

J vs I, paired, dev:

* G0 cal: MAE `0.100416` (J) vs `0.103992` (I) → `G_IJ = +0.003575`, CI `[-0.000230, +0.007270]`.
* G0 raw: `0.104099` vs `0.108945` → `+0.005784`, CI `[+0.001904, +0.009678]`.
* overall cal: `0.101907` vs `0.104682` → `+0.002775`, CI `[-0.000947, +0.006911]`.
* overall raw: `+0.004846`, CI `[+0.000905, +0.008889]`.

Bootstrap identity checks pass exactly (point and replicate identity max Δ `0.0`; mirror/constant
shift self-tests pass).

Fired category: **`INCONCLUSIVE`**.
`JOINT_SUPPORT` fails because the G0 calibrated CI includes zero (`CI_low = −0.00023 < 0`) even
though the point estimate `+0.00358` is above the 0.003 threshold and the raw gain excludes zero.
`INDEPENDENT_SUPPORT` fails (wrong direction). `LOCAL_EQUIVALENCE` fails because the G0 cal CI upper
bound `+0.00727 > +0.003` (the difference is not demonstrated to be within the equivalence band).
Interpretation: **J tends to be better, but this single seed does not establish a joint-support
claim; equivalence is also not established.**

### Q4. Did the local tuple path actually do work, and was the operator intervention meaningful?

Mechanism health (soup states, dev/fit, eval mode):

* Zeroing the local path at **init** changes predictions by exactly `0.0` (`W_loc = 0`); after
  training, zeroing it raises dev cal MAE from `0.10191` to `0.54847` (J) and from `0.10468` to
  `0.46546` (I): the local-tuple term carries the large majority of the prediction signal relative
  to the weak other path, and is not a decorative head.
* Operator switch J↔I on the frozen soups: mean |Δprediction| `0.00338` (J→I) / `0.00389` (I→J);
  MAE change `+0.00008` (J→I, worse) and `−0.00013` (I→J, better) on dev, consistent with the small
  J advantage. Both prediction change and error change are reported, so the switch changes the
  function without trivially destroying it.
* Tied IHT dictionary: alpha sparsity exactly `8.0` per tuple (matches `s=8`); 60–61 of 64 atoms
  used; 3–4 dead root-code dimensions of 64; `D_loc` norm `88.8 → 33.3` with relative drift `0.75`
  (the dictionary does move); `W_loc` norm `9.2–9.3`; per-epoch probes show nonzero `W_loc` and
  `D_loc` task gradients from epoch 40 onward.
* Replay: reloading the soup reproduces fit/dev predictions to `≤1.9e-6` (`< 1e-5`), CPU↔GPU same
  state.

### Q5. Sensitivity, contributions, and what remains unattributable?

* Drop-one-row sensitivity (largest combined dev error, stable id 8052): overall cal B `0.101218`,
  J `0.100259`, I `0.102160`; G0 cal B `0.099263`, J `0.098694`, I `0.101357`. The J > I ordering
  in both overall and G0 is stable to the most influential row.
* Per-`k` contributions (dev cal MAE; contribution in parentheses):
  * `k=0` (1915 rows, 97.5% of MAE): B `0.1019` (.0975), **J `0.1004` (.0961)**, I `0.1040` (.0996).
  * `k=-1` (73): B `0.1264` (.0046), J `0.1227` (.0045), **I `0.1082` (.0039)**.
  * `k=-2` (10): B `0.1828` (.0009), J `0.1877` (.0009), **I `0.1759` (.0009)**.
  * `k<=-3` (2): B `0.6447` (.0006), J `0.3431` (.0003), **I `0.2819` (.0003)**.
  The aggregate J > I advantage comes entirely from G0; on the rare cycle groups I is directionally
  better (2–10 dev rows — descriptive only, no ring-relief claim per `EVIDENCE_SCOPE.md`).
* Fit side: J fit overall `0.028155`, I `0.031167`, B `0.029280` — the J advantage is present in
  sample and large relative to dev (`3.6×` fit→dev gap for J), so it is not a fit-only artifact.
* Unattributable with one seed, one fold, 2–10 rare dev rows and one soup: optimisation/dropout
  noise vs a genuine G0-specific effect of direct correspondence access; whether the marginal arm
  would catch up with more capacity or a different `s`; the exact role of the historical B's
  calibration bias difference (J's fit-median bias is `0.0149` above B's); and whether the
  G0-driven advantage extends to rare chemistry (it does not in these 12 rare dev rows).

### Q6. Single next design (design only, not executed here)

**Incidence-permuted marginal control, multi-seed.** Keep exactly J's architecture, capacity and
training recipe, but replace `w_J` by a per-root random permutation of the incidence entries that
preserves each root's marginal counts `(n_t, n_a, d_v)` while breaking the `(t,a)` pairing
(`w_perm`). Compare three arms — J, I, perm — over ≥3 seeds on the same fold, with the same paired
bootstrap and the same `0.003` calibrated G0 gate, and include the pre-registered equivalence band
`±0.003`. This separates "joint correspondence content" from "any nonzero tuple weights consistent
with the marginals", which the present two-arm round cannot distinguish, and would turn the current
`INCONCLUSIVE` into either `JOINT_SUPPORT` (J > I and J > perm), `MARGINAL_ONLY` (perm ≈ J > I) or
`LOCAL_EQUIVALENCE`. Not executed in this round (no new seeds/config search permitted).
