# TCCD-v6 — POST Gain Decomposition: Normalization vs Marginal Moments vs Nonlinearity

Preregistration: `notes/tccd_v6_post_gain_decomposition_preregistration.md`.
Preregistration commit: `822a235`.
Formal implementation commit: `e016b4d8031994ca0d33f15242b444488fbe8713`.
Device-fix commit: `ac94daea613d8d19bf5ef978756c82568918a157`.
Formal result: `results/tccd_v6/stageA_seed0.json`.

## Verdict

**Case R1 — NONLINEARITY NOT MATERIAL / RECON SUFFICIENT.**

The TCCD-v5 POST gain is essentially fully explained by the 69 POST
coordinates that are exactly reconstructible from the frozen TCCD-v2
`h_base`.  A small decoder over those coordinates recovers `99.51%` of
`G_FULL` (`99.25%` with the ReLU removed).  ReLU contributes only
`0.000196` MAE.  Therefore the v5 positive result is primarily a
**readout / coordinate-geometry effect**, not evidence that TCCD-v2 lacks
structural information or occurrence placement.

## Execution integrity

* Formal compute: remote A100 **GPU1** only (`physical_gpu_requested: 1`,
  logical `cuda`; no GPU0 use).
* Seed 0; exact fixed internal split `20260922`, 8000 train / 2000 dev.
* Frozen TCCD-v2 PrototypeREL checkpoint SHA-256:
  `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42`
  (verified on the remote before the run).
* Exact TCCD-v5 pair cache reused; metadata certifies
  `official_test_loaded: false`; the audit path never reads `pair_all_y.npy`.
* v5 Stage-A JSON SHA-256:
  `2ab232c74219529ef3ddbdf1c6175d664756eb0dc7c0a0a30325d62a94b8aefe`.
* v5 POST soup SHA-256:
  `5bf555be35d7db8358fa78a5359330a4af2cb82d22d528168d992be307c10866`;
  best SHA-256:
  `c705176388bbb455c9e1261eab7a387690b637a55768872b9219628c6e753390`.
* FULL-NL reuse verified by re-evaluation: soup `0.2522333264350891`
  (delta `0.0`), best `0.2608269155025482` (delta `0.0`).
* Official ZINC test: never loaded (`official_test_loaded: false`).
* Total Stage-A wall `466.0 s`; peak GPU memory `109.2 MB`.

## Gate 0

**PASS** on GPU1 (`results/tccd_v6/gate0.json`, commit `ac94dae`).

* Data-free checks all pass: mask partition exact, FULL = RECON + NOVEL,
  `197 -> 64 -> 16` shapes, parameter equality (`13,712 + 10,481 = 24,193`),
  bit-identical initialization across all four arms (max diff `0.0`,
  checksum `9fbedb92...e8487f`, equal to the recorded v5 POST init checksum),
  v5 descriptor equivalence `2.98e-08`, batching / pair-order / relabel
  invariance, empty-pair branch exactly zero.
* Real-cache checks all pass: A reconstructibility, per-relation D
  reconstructibility, `FULL = RECON + NOVEL`, and descriptor equivalence with
  the exact v5 tensor path (`2.98e-08`).

## Label-free algebra audit (never reads y)

`results/tccd_v6/label_free_audit.json`, 10,000 graphs, tolerance `1e-6`.

| check | result |
|---|---:|
| A: `A_G = 2*BAG_G/n_G`, max abs diff | `4.62e-07` PASS |
| D: all 5 relations reconstructible, max per-relation diff | `<= 1.53e-07` PASS |
| `2*C(n,2)*D_pair + n*delta_geo == S_all`, relative diff | `1.08e-07` PASS |
| RECON nonzero coordinates | `69` (A 0..63, D 192..196) |
| NOVEL nonzero coordinates | `128` (B 64..127, C_prod 128..191) |
| frozen mask == audited mask | `true` |

Derivation used:

```text
A_G  = 2 * BAG_G / n_G,  n_G = sum_k BAG_G[k]

S_all[r] = sum_{k,l}(C^T R_r C)[k,l]
         = 2 * S_up[r] - diag_in_base[r]      (symmetric R_r, stored upper triangle)
         = sum_{i,j} R_r[i,j]                (softmax rows sum to 1)

sum_{i<j} R_r[i,j] = (S_all[r] - n_G * delta_{r,R_geo}) / 2
D_G[r]             = sum_{i<j} R_r[i,j] / C(n_G, 2)
```

The D marginals depend only on the graph's frozen relation operators and the
occurrence count, not on prototype assignments; they are nevertheless exactly
carried by the base relation blocks.

## Stage A — frozen decomposition

All four arms share identical trainable parameters (`24,193`), identical
initialization (`9fbedb92...e8487f`), and the exact TCCD-v5 optimizer /
stopping / Top-5 soup protocol.

| arm | input coordinates | branch | best dev MAE | Top-5 soup MAE |
|---|---|---:|---:|---:|
| BASE (reused v5) | `h_base` | linear | `0.287915766` | `0.278363913` |
| FULL-NL (reused v5 POST) | 197 | Linear-ReLU-Linear | `0.260826916` | `0.252233326` |
| RECON-NL | 69 (A + D) | Linear-ReLU-Linear | `0.257836252` | `0.252360463` |
| RECON-LINFACT | 69 (A + D) | Linear-Linear (no ReLU) | `0.262409121` | `0.252556831` |
| NOVEL-NL | 128 (B + C_prod) | Linear-ReLU-Linear | `0.263861537` | `0.254013419` |

Derived quantities:

```text
G_FULL            = 0.026130587
G_RECON           = 0.026003450   rho_RECON = 0.995135
G_NOVEL           = 0.024350494   rho_NOVEL = 0.931877
gap (LINFACT-NL)  = 0.000196368
```

## Decision

Registered precedence R > M > J > N:

* **Case R holds:** `G_RECON = 0.026003 >= 0.015` and RECON-NL
  `0.252360 <= MAE_FULL + 0.005 = 0.257233`.
* `gap = 0.000196 < 0.005` -> **R1, NONLINEARITY NOT MATERIAL**.
* The ambiguity interval `0.005 <= gap < 0.010` never triggers, so **no
  paired seed 1 is authorized or run**.
* **Case M also holds** (`G_NOVEL = 0.024350 >= 0.015`, NOVEL-NL
  `0.254013 <= 0.257233`), but R takes precedence.  The M result is reported
  as an additional observation: genuinely new prototype-distribution moments
  are *not required* to recover most of `G_FULL`.
* Cases J and N are false.

## Scientific interpretation

1. **How much of the v5 POST gain is base-reconstructible normalized
   coordinates?** Essentially all of it.  RECON-NL recovers `99.51%`
   (`99.25%` without ReLU).  The 69 RECON coordinates are deterministic
   functions of `h_base`; no new graph information enters the branch.  The
   gain therefore comes from *exposing better-conditioned global coordinates*
   (the size-normalized prototype mean `2*BAG/n` and the normalized relation
   density) to a linear reader, not from new structural signal.
2. **Does nonlinear decoding matter?** No.  Removing the ReLU changes the soup
   MAE by `0.000196`, two orders of magnitude below the registered `0.005`
   threshold.  The best-checkpoint metric shows the same (RECON-NL best
   `0.257836` is in fact *better* than FULL best `0.260827`).  The result is
   not driven by learned nonlinearity.
3. **How much is explained by genuinely additional prototype-distribution
   moments?** NOVEL-NL alone recovers `93.19%` of `G_FULL`.  So the new
   moments (absolute dispersion and pair product) also carry most of the
   signal, but they are not necessary: the parsimonious explanation is the
   RECON coordinate exposure, and the two blocks are substantially redundant
   because they are all functions of the same frozen assignments.
4. **Does the evidence still support occurrence-preserving /
   center-preserving composition as the next priority?** No.  There is no
   evidence that the missing ingredient is occurrence placement or structural
   identity; the frozen decomposition shows the v5 gain is recoverable without
   any new information.
5. **Primary TCCD bottleneck:** within the frozen TCCD-v2 representation, the
   bottleneck is now **readout / coordinate geometry** (conditioning and
   normalization of the global summary), not structural information shortage.

Caveats, stated honestly:

* One seed for each new arm.  The registered seed-1 route applies only to the
  nonlinearity ambiguity interval and did not trigger.
* The three new arms are close in soup MAE (`0.252360` / `0.252557` /
  `0.254013`).  No ordering claim beyond the registered sufficiency and gap
  gates is made; in particular RECON > NOVEL is not claimed.
* All evidence is internal-dev only.  No official-valid training and no
  official-test evaluation occurred, and none is authorized by this result.

## Not run (frozen stop)

No PRE / PRE-SHUFFLE, no occurrence-preserving or center-preserving model, no
end-to-end prototype retraining, no vocabulary change, no deeper MLP, no
nonlinear 10,464-D reader, no moment-complete architecture, no official-valid,
no official test.  The round ends at the frozen decomposition.

## Next hypothesis (requires a new preregistration; not auto-authorized)

For Case R1 the registered priority is an **explicit
normalization/moment-complete frozen representation with a simple reader**:
keep the frozen TCCD-v2 assignments and expose the size-normalized prototype
mean and prototype-distribution moments directly, with a small
(linear / shallow) reader, and test whether the POST gain can be captured
without enumerating pairs.  Do not deepen the MLP and do not re-open
center-preserving composition in this round.

Evidence:

* `tracks/ksvd/notes/tccd_v6_post_gain_decomposition_preregistration.md`
* `tracks/ksvd/results/tccd_v6/label_free_audit.json`
* `tracks/ksvd/results/tccd_v6/gate0.json`
* `tracks/ksvd/results/tccd_v6/stageA_seed0.json`
* `tracks/ksvd/results/tccd_v6/stageA_decision.json`
* `tracks/ksvd/code/tccd_v6.py`
* `tracks/ksvd/code/run_tccd_v6.py`
* `tracks/ksvd/tests/test_tccd_v6.py`
