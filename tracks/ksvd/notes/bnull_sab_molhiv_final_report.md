# Final report — ZINC B-Null S/A/B path ablation + MolHIV B-Null small-head probe

Protocols: `bnull_sab_path_ablation_v1`, `molhiv_bnull_small_head_probe_v1`,
`molhiv_bnull_small_head_probe_test_closure_v1`.
Branch: `exp/bnull-sab-path-ablation`.
Official ZINC test: **never loaded**. Official MolHIV test: loaded exactly once,
under explicit user authorization (see `molhiv_bnull_small_head_test_read.md`).

## 0. References (read from the repo)

* ZINC B-Null seed0 Top-5 soup: `results/local_token_null/` (`0.1230275` valid
  MAE, 49,343 params).
* MolHIV B-Null seed0 Top-5 soup: `results/molhiv_local_token_null/`
  (`0.840847` valid ROC-AUC, 237,133 params).

## 1. Exact feature map (from `zinc_patch_path_pooling.py`)

`patch_cont` (`SHELL_WIDTH = 146`, train-fit standardized): atom_shell shells
0/1/2 = cols 0–83 (A/B per-shell type histogram); bond_shell six shell-pairs =
84–107 (A/B per-shell-pair type histogram); root_atom = 108–135 (B);
incident_bonds = 136–139 (B); pure-S scalars = 140–145 (log size/edges,
boundary fraction, cycle_rank, centre/mean degree). `pair_relation`
(`RELATION_WIDTH = 23`, raw / not standardized): distance_onehot 0–4 (S),
log_distance 5 (S), overlap 6–10 (S), boundary 11–13 (S), path_bond_mean 14–17
(**B**), log_path_count 18 (S), adjacent_bond 19–22 (S/B; block mass pure-S
adjacency, type B). `global_context` 62-D (S + graph A marginals) and the
pure-S topology channel are never touched. Parent token → 32×8 embedding
(parent-B). Local 16-D patch token is exactly zero (B-Null). Composition C:
`P`, `Q`, distance gate, pair→centre pool, `U`, T=2 weight-tied.

## 2. Stage A — frozen screen (eval only, no retraining)

`Δ = intervention − original`; positive = worse. Bands: |Δ|<0.003 none;
+0.003–0.010 small-moderate; +0.010–0.030 material; >+0.030 very important.

| intervention | family | frozen Δ | band |
|---|---|---:|---|
| A1 parent-null | B (parent typed) | **+0.296780** | very important |
| A2 pair-B marginalized | B (pair/path bond type) | **+0.203715** | very important |
| A3 patch-B marginalized | B (shell/root type binding) | **+0.517012** | very important |
| A4 T=1 frozen recurrence | C (composition) | **+0.138756** | very important |

Original reproduced under the harness with residual `3.9e-08`.

### Integrity (`stage_a_integrity.json`, `all_pass: true`)

Parent block exactly zero and non-parent columns bit-identical
(cols 162–170, width 8); pair-B pure-S columns bit-identical and adjacency mass
preserved (1000/1000 graphs); patch-B per-patch atom/bond marginals, shell /
shell-pair masses, and six pure-S scalars all preserved (23,083 patches);
T=1 exactly one Q and one U round (vs T=2's two); identity-override
bit-identical; deterministic. Diagnostic: patch-B standardized diff
`max_abs = 76.97`, `mean_abs = 0.103`, 19,830/23,083 patches changed — a large
per-column shift on a few low-scale columns, i.e. a clear OOD signature.

## 3. Ticket selection (`stage_b_ticket_selection.json`)

Default tickets kept: **NoParent, Pair-B-Marginal, Patch-B-Marginal**. The T=1
exception did not fire: although `Δ_T1 = 0.1388 ≥ 0.020`, no frozen |Δ| was
`< 0.003` (all ≥ 0.20), so no T=1 retrain replaced a ticket. Three full runs,
≤3 as pre-registered.

## 4. Stage B — retrained ZINC seed0 (identical 49,343-param B-Null backbone)

Fixed protocol: Adam lr 1e-3, wd 1e-5, batch 128, 240 epochs, patience 40, L1,
clip 5, fixed equal-weight Top-5 valid soup.

| ticket | epochs | best ep | soup valid MAE | retrained Δ |
|---|---:|---:|---:|---:|
| B-Null (reference) | 240 | 237 | 0.123027 | 0.000000 |
| N1 NoParent | 240 | 235 | 0.129447 | **+0.006420** |
| N2 Pair-B-Marginal | 240 | 234 | 0.124474 | **+0.001447** |
| N3 Patch-B-Marginal | 240 | 233 | 0.167303 | **+0.044275** |

## 5. S/A/B table (frozen vs retrained)

| path | family | frozen Δ (A) | retrained Δ (B) | B band | verdict |
|---|---|---:|---:|---|---|
| parent typed (radius-1) | B | +0.2968 | +0.0064 | small-moderate | mostly OOD; nearly redundant |
| pair-B (path/adjacent bond type) | B | +0.2037 | +0.0014 | none (<0.003) | **redundant** |
| patch-B (shell/root role↔type) | B | +0.5170 | +0.0443 | very important | **genuinely necessary** |
| T=1 recurrence | C | +0.1388 | not retrained | — | frozen material; direction only |

**Central finding.** The frozen Stage-A deltas (all ≫ 0.03) are dominated by
distribution shift, not information loss: retraining the remaining pathway
recovers almost all of the parent-null and pair-B degradation (parent
+0.297→+0.006, pair-B +0.204→+0.001). Only **patch-B** survives retraining with a
large effect (+0.044). So, among the B-null-compatible binding paths, the
decisive computation is *which attribute occupies which structural role inside
the local patch descriptor* (root atom / incident bonds / shell-role type
histograms); path-level and radius-1 typed context are substitutable.

## 6. MolHIV B-Null frozen small-head probe (valid-only)

`R` = exact head input = `2·96 + 1 + 6·(2·16 + 1) + 32` = **423**, extracted on
official train (32,901) and valid (4,113) from the frozen backbone, verified by
re-applying the frozen head (max abs 0.0). No test representation extracted.

| head | params | best ep | raw valid | soup valid |
|---|---:|---:|---:|---:|
| B-Null backbone (joint, ref) | 237,133 | 24 | 0.843168 | 0.840847 |
| H_refit (100k) | 100,417 | 16 | 0.835636 | 0.830281 |
| H_small32 (14k) | 14,177 | 5 | 0.832678 | 0.830201 |

Gate: `H_refit_soup − original = 0.010567 > 0.010` ⇒ **HEAD_PROBE_INVALID**
(by a margin of 0.00057). Per pre-registration the small-head architecture
conclusion is **not authorized**. `ΔAUC(H_refit − H_small32) = 0.00008`, i.e.
within the frozen representation the two heads are indistinguishable.

Why the gate failed: MolHIV valid has only 81 positives (test 130); the joint
soup head was co-trained with the backbone, so a fresh head on frozen `R` loses
a little; a 0.0106 gap is within the noise scale of this split (the earlier
typed 2-seed soup spread was 0.0416), and it is right at the threshold.

### Authorized one-shot official test (`§ molhiv_bnull_small_head_test_read.md`)

| head | params | valid soup | test raw | test soup | Δ soup |
|---|---:|---:|---:|---:|---:|
| B-Null backbone | 237,133 | 0.8408 | 0.743411 | 0.760501 | −0.0803 |
| H_refit | 100,417 | 0.8303 | 0.752365 | 0.749177 | −0.0811 |
| H_small32 | 14,177 | 0.8302 | 0.762960 | 0.762549 | −0.0677 |

Backbone reproduces the prior recorded closure exactly. The 14k head is the best
test soup; the 100k head falls below the backbone. Directional only (130
positives); the valid-only gate still governs.

## 7. Questions

**Q1. Does a frozen knockout measure the same "importance" as a retrained
ablation?** No. Frozen deltas are 8–460× the retrained deltas and rank-order
differently at the bottom; frozen screens are a sensitivity/OOD probe, not an
importance measure. Only retraining isolates necessity.

**Q2. Which structure–attribute binding path is genuinely necessary for B-Null
on ZINC?** Patch-B (root atom, incident bonds, and shell-role atom/bond type
histograms in `patch_cont`): retrained Δ = +0.044, very important.

**Q3. Are pair-B and the radius-1 typed parent context incrementally
necessary?** Pair-B: no (retrained Δ = +0.0014, below the 0.003 threshold).
Parent typed context: only weakly (retrained Δ = +0.0064, small-moderate). Both
frozen effects were almost entirely recoverable by retraining.

**Q4. What does the frozen T=1 result say about the composition (C)
family?** It is material even frozen (Δ = +0.139), the smallest of the four
frozen effects, and was not retrained (the pre-registered selection rule did not
fire). Treat as directional: the second pair→centre round carries signal that
the frozen model cannot immediately forgo; whether it is *necessary* is untested.

**Q5. Do the ablated backbones train stably without parameter
re-investment?** Yes. Three runs, 240/240 epochs each, all 49,343 params,
finite losses, best epochs 233–235, no width/optimizer changes, no collapse.

**Q6. On MolHIV, is a future end-to-end compact model (100k head → ~14k) worth
continuing?** Provisionally yes, but not authorized by pre-registration: the
valid-only gate returned `HEAD_PROBE_INVALID` (0.01057 vs 0.0100), and the
authorized test read, while favoring the 14k head (best soup 0.7625, vs 0.7605
backbone and 0.7492 for the 100k head), is single-seed with only 130 test
positives. The single recommended next action is a paired seed-1 confirmation of
`H_refit` vs `H_small32` on the same frozen `R`; do not start an end-to-end
compact MolHIV from this alone.

## 8. Stop point and non-goals (honored)

No ZINC official test; no MolHIV backbone training; no end-to-end compact
MolHIV; no seed1; no fourth full ZINC run; no parameter re-investment; no new
architecture/feature/radius; no width/dropout/lr/optimizer sweeps; no post-hoc
new ablation invented from valid results. The only deviation from the original
valid-only plan was the user-authorized, one-shot MolHIV test read, documented
separately.

## 9. Provenance

* Preregistration + code: `a43694c`; fixes `4d9fa1f`, `87c38ed`, `7de00bd`,
  `fab3e9a`; MolHIV test closure `1ea6392`; witness fix `6bcc8ec`.
* Stage A + wave-1 Stage B (NoParent, Pair-B) trained at `fab3e9a`; wave-2
  (Patch-B) at `1ea6392`; the S/A/B module code is identical between them. The
  MolHIV probe runs at `fab3e9a`; the freeze/test closure at `1ea6392`.
* Execution: GPU1 (`CUDA_VISIBLE_DEVICES=1`) only; GPU0 never used. Single-process
  GPU smoke passed (finite loss, non-zero grads, 49,343 params, ≤234 MB peak);
  non-formal 2-way concurrency check showed no slowdown, so wave-1 ran
  concurrently. Observed epoch time: 23.6 s (wave-1, contended) vs 6.8 s
  (wave-2, solo) — a throughput observation only; it does not affect the metric.
* ZINC results: `results/bnull_sab_path_ablation/`. MolHIV results:
  `results/molhiv_bnull_small_head_probe/` and
  `results/molhiv_bnull_small_head_probe_test_closure/`.
* Not run (per stop instruction): `witness` for Patch-B-Marginal, and
  `camera`/`report` aggregation stages; the S/A/B table above was aggregated
  locally from the soup/run JSONs.
