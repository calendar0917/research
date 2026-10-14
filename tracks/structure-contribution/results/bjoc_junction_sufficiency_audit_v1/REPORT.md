# BJOC Junction Sufficiency & Structural Information Audit v1 — final report

* Protocol: `configs/bjoc_junction_sufficiency_audit_v1.yaml` (`cscl-bjoc-junction-sufficiency-audit-v1`), frozen 2026-10-14 at commit `82f76cf` (remote `res` freeze), training and Monitor score at the same commit.
* Track: `tracks/structure-contribution`.
* Status: **complete** (one seed, one predeclared Monitor read, Locked never read, official valid/test never loaded).
* Frozen decision words: **`J_MOMENTS_NO_CLEAR_GAIN` · `COUNT_CONDITIONING_NO_CLEAR_GAIN` · `NULL_OWNERSHIP_BLINDNESS_VERIFIED` · `REAL_COLLISION_NOT_FOUND_WITHIN_BUDGET`**.
* GPU spend: **0.342 of 1.5 GPU-hours** (`gpu_ledger.json`), two parallel A100 jobs ~10 min each; every audit below is CPU-only.
* Purpose: mechanism evidence for the advisor discussion, not a SOTA search and not a new architecture round.

## 1. Question

**(A)** How much of the current BJOC-Null competitive accuracy needs the learned Junction channel at all, and how much needs per-bond Junction-count conditioning? Three functional arms, only the way Junction information enters the Bond Composer changes:

| arm | W_e | V_e | q_e | what it isolates |
|---|---|---|---|---|
| `A0 NO_LEARNED_J_CONTENT` | 0 | 0 | 0 | Atom/Bond encoders + fixed global topology only |
| `A1 GLOBAL_J_MOMENTS` | μ_G | ν_G | 0 | + full-molecule Junction content, broadcast uniformly |
| `A2 P3_NULL_REFERENCE` | n_e·μ_G | n_e·ν_G | log1p(n_e) | + per-bond Junction count conditioning (historical Null) |

No new learnable modules; all arms are 413,505 parameters with the same module shapes. A2 is read-only reuse of the historical P3-D-s0 soup and its frozen Monitor predictions.

**(B)** Which structural-organization information is provably lost? Part B separates three evidence classes that must not be conflated: (1) information that is algebraically invisible to the Null grouping; (2) synthetic counterexamples with known truth; (3) real-molecule collisions in the official train 10k structure fields.

## 2. Reuse, integrity and correctness gates

* Reused read-only (all hashes re-verified at run time): split-D objects (`objects_fit_D.pt` `e55cf639…`, `objects_monitor_D.pt` `fd4cb2caad…`, `objects_locked_D.pt`), seed-0 Fit-only masked-SSL encoder `pretrain_D_s0.pt` `8da281fa…` (`encoder_used 2c2bbd3f…`), P3 soup `13a78a63…`, frozen P3 Monitor predictions, frozen split-D rows (Fit `9468656a…`, Monitor `d196bc75…`, Locked `f3525ae8…`).
* Both new trajectories started from the same scratch init (full `d06d5e16…`, group `d9ffbd15…`, encoder `91af980a…`), the same SSL encoder, and the same 22,680-step batch order (`d142e307…`, independently recomputed equal).
* 17 CPU gates (`stage_a.json`, all pass): A2 bitwise-equal to the parent forward and to the frozen P3 Monitor predictions (max |Δ| 1.9e-6, cross-machine float32 reduction noise); A0 W/V/q exactly 0 and junction-insensitive; A1 uniform within each molecule; permutation/relabel/batch/save-load invariances ≤ 4e-7; A1/A2 junction gradients non-zero; A0 junction branch exactly zero; no official valid/test path; no forbidden modules.
* `verify-inputs` and `freeze` re-checked every input hash; `subspace` row alignment of the 10,000 objects against official train row ids is asserted before the B3 scan.
* **Post-freeze amendment (analysis driver only, before the Monitor read):** `post_freeze_amendments.json` records five CPU-side corrections (B3 row alignment + NetworkX re-verification, near-neighbour accounting, B5 split, detach fix, finalize aggregation). Module/composer/objects/config hashes are byte-identical before and after; arm definitions, thresholds, budget and seed unchanged. Training ran at `82f76cf`; the Monitor score was computed after the amendment.

## 3. Part A — Monitor results (split D Monitor, n=1000, seed 0, historically exposed)

### 3.1 The three numbers

| | A0 | A1 | A2 |
|---|---|---|---|
| **Monitor MAE** | **0.121514** | **0.106239** | **0.103960** |
| RMSE | 0.326862 | 0.517173 | 0.493170 |
| median abs | 0.069321 | 0.047912 | 0.048793 |
| P90 abs | 0.248948 | 0.205615 | 0.203204 |
| ordinary MAE (y > −3, n=929) | 0.102599 | 0.082460 | 0.081137 |
| tail MAE (y ≤ −3, n=71) | 0.369005 | 0.417385 | 0.402585 |
| top-10-error-removed MAE | 0.100499 | 0.079531 | 0.078675 |
| Fit MAE | 0.002741 | 0.001434 | 0.001082 (P3) |
| updates / epochs | 22,680 / 360 | 22,680 / 360 | 22,680 / 360 (historical) |
| training wall | 648.8 s | 555.1 s | 538.7 s (historical) |

### 3.2 Paired molecule-bootstrap contrasts (5,000 resamples, seed 0)

| contrast | Δ MAE | 95% CI | frac better | ordinary Δ | tail Δ | RMSE new vs ref | label |
|---|---|---|---|---|---|---|---|
| Δ10 = A1−A0 | −0.015275 | [−0.028561, +0.003351] | 0.605 | −0.020140 | **+0.048379** | 0.517 vs 0.327 | `J_MOMENTS_NO_CLEAR_GAIN` |
| Δ21 = A2−A1 | −0.002280 | [−0.006969, +0.002233] | 0.517 | −0.001323 | −0.014800 | 0.493 vs 0.517 | `COUNT_CONDITIONING_NO_CLEAR_GAIN` |
| Δ20 = A2−A0 | −0.017555 | [−0.030225, −0.000658] | 0.615 | −0.021463 | +0.033580 | 0.493 vs 0.327 | (descriptive: CI excludes 0) |

Neither predeclared "useful" label was earned: Δ10 has a clear negative point estimate but its CI upper bound is positive, and Δ21 sits at the −0.002 threshold with a CI crossing zero. No single molecule dominates (max single-gain fraction 0.037 / 0.044 / 0.047).

**Reading of the numbers.** Junction moments change *which* molecules are predicted well, not uniformly for the better: both A1 and A2 are much better than A0 on ordinary molecules (−0.020, −0.021) and much better on median/P90 errors, but A1 is *worse* than A0 on the 71 tail molecules (+0.048) and its RMSE is 0.52 vs 0.33. The architecture's error distribution under Junction moments is heavier-tailed: better typical accuracy, worse extremes. This is the opposite of the historical full-10k No-Junction comparison (which found Junction content worth ≈ +0.0198 Locked MAE) and it is a single seed on an exposed internal split — consistent with "Junction content has predictive value here, but this round does not certify it as a stable gain."

### 3.3 Effective training capacity (functional ablation, not capacity-matched)

| | allocated | structurally inactive entries | one-batch nonzero-grad entries | junction-encoder grad norm | composer W/V/q column grad |
|---|---|---|---|---|---|
| A0 | 413,505 | 143,776 (junction encoder 112,896 + composer W/V/q columns 30,880) → 65.2 % designed-active | 263,393 (0.637) | none (branch discarded) | W=V=q=0 exactly |
| A1 | 413,505 | 160 (composer q column) → 99.96 % | 405,089 (0.980) | 0.437 (init) / 0.519 (trained) | W,V > 0, q = 0 |
| A2 | 413,505 | 0 | 405,249 (0.980) | 1.669 (init) / 0.550 (trained) | all > 0 |

Zero dead hidden units (mean |activation| ≥ 0.064) in every encoder and composer for every arm; forward time 0.016–0.019 s per 256-molecule batch. A0 loses the entire junction encoder plus the composer's W/V/q input columns before training even starts — the official A0-versus-A2 contrast is therefore a *functional* ablation with a real capacity difference (65 % vs 100 % designed-active), and is reported as such.

### 3.4 Interpretation against the predeclared matrix

The observed pattern is closest to the predeclared **case 5 (weak/edge evidence, not decision-grade)**: Δ10 is large but CI-crossing; Δ21 is at the threshold. It is not case 1 (A1 clearly better and A2 ≈ A1) because neither contrast reaches the frozen useful-label bar, and not case 3 either, because the point estimates are not "all the same". Officially recorded: `J_MOMENTS_NO_CLEAR_GAIN`, `COUNT_CONDITIONING_NO_CLEAR_GAIN`, single seed, exposed Monitor — nothing more.

## 4. Part B — summary (full detail in `STRUCTURAL_INFORMATION_AUDIT.md`)

* **B1 algebraic blindness (conditional):** for fixed per-bond bond representations, fixed per-bond junction counts and a fixed graph Junction multiset, the Null grouping `W_e = n_e·μ_G, V_e = n_e·ν_G, q_e = log1p(n_e)` cannot change under Junction-to-Bond reassignment, while the Organized grouping `W_e = Σ_{j∋e} w_j` does. Worked example: 4 bonds / 4 junctions, Null `[3,3,3,3]` in both assignments, Organized `[2,4,3,3]` vs `[3,3,3,3]`. This is representation-level and conditional — not a claim that arbitrary reassignments are realisable molecules. A scaled family (m = 2…5) reproduces it with `max|ΔW_perbond| = 0` for Null and ≥ 0.0586 for Organized, with the ownership-only target differing by 8–36 while the marginal-only target stays equal.
* **B5b nuance (new):** ownership is *not* fully erased at the raw-token level. Because the junction encoder consumes the two incident bond embeddings, two structures with identical raw token multisets and counts can still produce different `w` multisets when the pairing differs (38 within-group pairs, max Null summary change 785). Exact algebraic blindness holds only conditional on the encoded `w` multiset.
* **B2 layer-wise:** at Level 3 the reader input is 321-dim as declared. The `global` block is nearly linearly redundant given the other blocks (ridge R² 0.938), `atom_sum` 0.968, `atom_sq_sum` 0.927, while the composer sum is the least redundant (0.536 from the other three). Fit-standardised distinct-molecule distances: composer-sum median 11.3 (p01 7.05), global median 7.1 (p01 2.71). Junction moments are rank ~0.61 of their dimension; the composer summary under A2 has effective rank 27.9 (d90 = 4).
* **B3 real-molecule scan (10,000 official-train structures, label-blind):** 3 exact structural-statistics collision groups were found, and **all 3 are duplicate canonical SMILES** (same molecule appearing twice), 0 cross-molecule exact collisions. The verdict is therefore **`REAL_COLLISION_NOT_FOUND_WITHIN_BUDGET`**. The *nearest distinct-molecule* pairs are much closer than typical: minimum reader-input standardised L2 distance 3.35 versus a Fit distinct-pair 1st percentile of 14.26 and median 22.59 — the representation is strongly compressive for some genuinely different structures, but this round found no exact collision.
* **B4 interventions:** junction-row permutation is a no-op at float noise (≤ 2.4e-7) for A0/A1/A2; post-encoding assignment switch leaves Null unchanged and changes Organized (W up to 0.117, V up to 0.098); block neutralisation on the frozen P3 shows the reader is sensitive to all four blocks (mean |Δpred|: global 0.174, composer 0.466, atom-sum 0.833, atom-sq 0.892) — an out-of-distribution sensitivity diagnostic, not an attribution.
* **B5 synthetic known-truth:** reused historical object-level counterexample (Null cannot distinguish, Organized can, on the ownership-only target) and extended it deterministically; no new training.

## 5. Joint interpretation (predeclared table)

| question | A evidence | B evidence | conclusion strength |
|---|---|---|---|
| Do learned Junction contents have predictive value? | Δ10 point estimate −0.015, CI [−0.029, +0.003] | Junction moments carry the full graph Junction multiset but not ownership | suggestive single-seed signal, not certified |
| Does per-bond count conditioning add value? | Δ21 −0.0023, CI [−0.007, +0.002] | `n_e` is a per-bond scaling of the same moments; correlated with endpoint degree | no clear gain; single seed |
| Is true Junction ownership preserved? | not trained this round | Null algebraic blindness verified; ownership partially leaks into `w` at the raw-token level | confirmed as a representation property |
| Does the graph representation lose structure? | A0/A1/A2 differ mainly in ordinary-subgroup placement | real near-collisions at 3.35 standardised units, no exact collision in budget | loss is real but compressive, not demonstrated as the ZINC error bottleneck |
| Does information loss cause ZINC error? | one internal split, one seed | real-sample diagnostics only | still undetermined |
| Can the architecture be faithfully attributed? | prediction quality cannot show it | non-linear reader over Null moments | not supported by this round |

## 6. Conclusions explicitly NOT drawn

* Single-seed A1 < A0 on MAE is not "Junction content is useless", and not "Junction content is proven useful".
* A2 − A1 < 0 does not prove that *true* Junction-to-Bond ownership is useful — `n_e` is a count, not an ownership relation, and is correlated with endpoint degree.
* A synthetic collision does not localise the real ZINC generalisation bottleneck.
* Non-zero Junction-encoder gradients do not mean chemically interpretable contributions were learned.
* Finding no exact real-molecule collision does not mean the representation has no limits (the exact key is finite and the near-search budget was 200 candidate pairs).
* A0/A1/A2 allocated parameter counts are equal, but A0's effective training capacity is lower; this round is a functional ablation, not strict active-capacity matching.

## 7. Artifacts

All under `tracks/structure-contribution/results/bjoc_junction_sufficiency_audit_v1/` (binary/large artifacts are intentionally not committed; sha256 below is the durable fingerprint):

| artifact | sha256 (prefix) | role |
|---|---|---|
| `protocol_freeze.json` | — | frozen rules, input hashes, commit `82f76cf` |
| `post_freeze_amendments.json` | — | analysis-driver amendment record |
| `input_audit.json` | — | reused-file hash verification |
| `stage_a.json` | `e52921a3` | 17 CPU correctness gates + init gradient audit |
| `parameter_gradient_audit.json` | `be147113` | per-module grad norms, active entries, dead channels, forward timing (init/trained) |
| `b1_algebraic.json` | `53cc22bc` | B1 worked example + B4-Test2 + B5 |
| `b5_synthetic.json` | `d60b81ff` | synthetic known-truth counterexamples |
| `b2_layerwise.json` | `9a2211cf` | Level 0–3 representation audit |
| `b2_level_arrays.npz` | `65e4ab79` | per-molecule level arrays (3000 × blocks) |
| `b3_collisions.json` | `58d037a2` | exact/near structural collision search summary |
| `prediction_manifest.json` | `43980a52` | frozen A0/A1/A2 Monitor predictions + hashes (A2 vs frozen P3 max |Δ| 1.9e-6) |
| `predictions/monitor_{A0,A1,A2}_D_s0.npz` | `4ef6777a` / `5964b5c9` / `152791ee` | frozen predictions |
| `train_A0_D_s0.json` | `3d20b867` | A0 run record (init/batch-order hashes, 22,680 updates, Fit MAE) |
| `train_A1_D_s0.json` | `40ca41de` | A1 run record |
| `checkpoint_manifest.json` | `5b2f71d0` | reused + new checkpoint hashes and roles |
| `scoring_manifest.json` | `0f0d082f` | prediction/label/lock/score hashes, single-read disclosure |
| `soup_A0_D_s0.pt` | `849244a0` | A0 checkpoint |
| `soup_A1_D_s0.pt` | `ee6c9729` | A1 checkpoint |
| `monitor_scores.json` | `07d65ada` | single Monitor score (tables, contrasts, tokens) |
| `monitor_read.lock` | — | one-read lock; reuses the historical hash-verified label file |
| `gpu_ledger.json` | `c15a7c67` | GPU 0.342 h / 1.5 h |
| `decision.json` | `fac73da1` | merged A+B decision record and tokens |

Reproduction on a CPU machine with the reused artifacts present:
`uv run python tracks/structure-contribution/code/run_bjoc_junction_sufficiency_audit_v1.py {verify-inputs,freeze,stage-a,grad-audit --phase init,grad-audit --phase trained,b-algebraic,b-layerwise,b-collisions,predict,score-monitor,finalize}`;
GPU trajectories: `rr run res <exp> --pool res-gpu{0,1} -- python tracks/structure-contribution/code/run_bjoc_junction_sufficiency_audit_v1.py --device cuda train --arm {A0,A1}` (records in `train_A*_D_s0.json`, commit `82f76cf`).

## 8. What remains uncertain

1. Whether the Δ10 signal (Junction content helping ordinary molecules) exists at another seed, and why RMSE/tail errors get worse.
2. Whether the near-collisions found by B3 (reader distance 3.35 vs typical 14–23) correspond to any measurable property error.
3. Whether an ownership-aware grouping (the historically studied Organized route) would change the tail behaviour that A1/A2 make worse — not re-run here by design.
4. Whether the raw-level leakage of ownership into `w` is large enough to matter after the non-linear reader.
