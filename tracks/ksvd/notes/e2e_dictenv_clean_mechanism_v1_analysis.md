# e2e_dictenv_clean_mechanism_v1 — analysis note

Round: `e2e_dictenv_clean_mechanism_v1` (tracks/ksvd, ZINC dictionary-environment line).
Preregistration: `notes/e2e_dictenv_clean_mechanism_v1_preregistration.md` (frozen before any run).
Results root: `results/e2e_dictenv_clean_mechanism_v1/` (reports: `REPORT.md`, `DECISION.md`,
`analysis_tables.md`).

## 0. What was asked

Four frozen questions about the C6 "clean" architecture (matched CPU regime, official test
never loaded, no GPU/SSH):

* **Q1** is C6 (remove graph atom/bond histogram + path_count + unary/pair counts) stable
  across seeds relative to the matched BASE;
* **Q2** is the node/edge structure–semantic binding mechanism fine correspondence
  (per-occurrence pairing) or only shell marginals;
* **Q3** what is the minimal relation bundle (is distance-only or distance+boundary enough);
* **Q4** is the sparse dictionary coordinate code specific (Sparse vs matched DenseTied).

## 1. Protocol and provenance

* CPU-only (`CUDA_VISIBLE_DEVICES=""`), 16 cores, 3 worker processes × 4 threads.
* Protocol constants frozen in the preregistration: Adam 1e-3 / wd 1e-5, batch 128,
  clip 5.0, 320 epochs, Top-5 soup, matched init/data order per seed, 10 000 train /
  1 000 official-valid molecules, official test never loaded.
* Seed 0 BASE/C6/C1 are reused from `results/e2e_dictenv_h1_clarity_audit/matched_cpu/`
  (bit-exact re-verified at round start: BASE 0.143297936, C6 0.128498517, abs diff 0.0).
* Revision sequence (all on `main`): `bb004bf` (harness preflight fix) → `21fc45d`
  (seed-suffixed artifact lookup) → `fa5ea5f` (binding-override masked forward) →
  `84a188f` (matched-seed gates C/F) → `38dda2a`/`b21cf91` (stage-B report/enrichment) →
  `f6a8c57` (dictionary diagnostics definitions). Only the first two touch the harness
  before the formal training wave; none changes the training path, the protocol, the mask
  semantics or the gates. Every artifact records its own `git_commit`.
* Peak RSS ≈ 1.93 GiB per process; observed 320-epoch wall ≈ 3 930 s at concurrency 3
  (≈ 12.3 s/epoch), single-process ≈ 7.4 s/epoch.
* Harness defects found and fixed during the round (all discovered by the gates
  disagreeing with hand checks, no silent retraining):
  1. multi-seed artifacts were written as `<tag>_seed<k>_e320.json` but looked up as
     `<tag>_e320.json`, so gate A initially saw only seed 0 and wrongly took the C1
     fallback (the four initial wave-1 runs are valid, their pair-matching was never in
     question; the fallback jobs were killed with no artifacts);
  2. `AuditModel.forward(mask=None)` bypasses `environments_masked`, so the frozen BASE
     control rows of stage C silently reported "no effect"; the variant evaluation now
     forces the semantically identical `AuditMask()` path whenever a non-default binding
     operator is requested (regression test);
  3. gate C differed every seed against the seed-0 clean-base run instead of the matched
     same-seed run; gate F used the seed-0 sparse reference for all dense seeds. Both now
     use matched-seed references. The round's decisions are unchanged (see §3), but the
     per-seed numbers in the first gate-C write-up were not usable;
  4. `stage_b` rows lacked `delta_mae`; the dictionary diagnostics reported
     active-atoms/rows as "sparsity" and max activation frequency as "top1 share". Both
     were corrected and the affected artifacts recomputed.

## 2. Q1 — C6 across seeds (Stage A)

| seed | BASE | C6 | Δ (C6−BASE) | relative |
|---|---|---|---|---|
| 0 (reused) | 0.143298 | 0.128499 | −0.014799 | −10.33 % |
| 1 | 0.130622 | 0.130505 | −0.000116 | −0.09 % |
| 2 | 0.131494 | 0.122265 | −0.009229 | −7.02 % |

Gate (frozen): mean −0.008048, 3/3 negative, max −0.000116, range 0.014683 < 0.015 →
**STRONG_SUPPORT**, C6 adopted as `CLEAN_BASE`. No C1 fallback, no seed-3 extension.

Honest reading: the *mean* gain is dominated by the seed-0 BASE run (0.1433, an outlier
against its own seeds 1/2 at 0.1306/0.1315). C6 is no worse than matched BASE on every seed
(worst case −0.000116) and its across-seed spread is smaller (std 0.0043 vs 0.0071,
range 0.0082 vs 0.0127). The supportable claim is therefore "removing the bundle costs
nothing and the C6 regime is at least as stable", not "C6 is reliably 0.008 better".

Mechanistic confirmation (Stage B, distribution-preserving): GS1 (graph-chemistry row
shuffle) changes C6 valid MAE by **exactly 0.0000** on all three seeds, both with the
training mask active and with the removed channels re-opened (identity view); BASE moves
+0.055/+0.079/+0.078. RS4 (log path count shuffle) is exactly 0.0000 on C6 (removed) and
non-zero on BASE. PS3/PS6 (unary/pair count rows) are ≈ 0 on both regimes. The shortcut is
structurally unused in C6, not merely masked.

## 3. Q2 — binding mechanism (Stage C)

Frozen, distribution-preserving replacement of the paired statistic by the analytic
assignment-independence null `(Σa)(Σc)/(n√D)` (same trained weights, same inputs):

| checkpoint | node Δ | edge Δ | both Δ |
|---|---|---|---|
| BASE s0 / s1 / s2 | +0.0124 / +0.0047 / +0.0020 | +0.0365 / +0.0408 / +0.0263 | +0.0490 / +0.0414 / +0.0281 |
| C6 s0 / s1 / s2 | +0.0068 / +0.0093 / +0.0120 | +0.0571 / +0.0308 / +0.0324 | +0.0649 / +0.0372 / +0.0498 |

* Both regimes read the correspondence; **edge binding is 4–6× more load-bearing than node
  binding** (C6 means: edge +0.0401, node +0.0093).
* C6 does not reduce reliance on either (it slightly increases it): the binding is not a
  redundant fallback for the removed chemistry shortcut, it is what the clean model uses.
* Residual structure (‖R‖/‖U_pair‖): node shell1 mean 0.10–0.13 (≈ 19 % of multi-slots are
  exactly 0 because the group's chemistry is constant, e.g. all-carbon neighbours), shell2
  0.14–0.22; edge shellpair1 (bond roles) 0.10–0.20, **shellpair4 (2-hop) 0.22–0.28 with
  p90 0.52–0.68** — correspondence carries more information at longer range, where
  marginals are least informative.
* n=1 slots (shell0: 23 083 slots; shellpair3: 70) are exactly paired = indep and carry no
  evidence; they are excluded from the verdict.

From-scratch controls (C6 mask with the independence operator, 320 epochs, same protocol)
and 20-epoch warm-start adaptation (control: matched M0 continuation 0.128782):

| role | from-scratch Δ per seed (0,1,2) | mean | max | warm Δ | adoption rule | adopted |
|---|---|---|---|---|---|---|
| node | +0.00008 / +0.00368 / +0.00925 | +0.00434 | +0.00925 | +0.00261 | max ≤ 0.005 and mean ≤ 0.005 | **no** |
| edge | +0.00900 / +0.00471 / +0.00895 | +0.00755 | +0.00901 | +0.00833 | mean ≤ 0.003 | **no** |

Conclusion: "shell marginals suffice" is rejected for edges and not confirmed for nodes.
Node fine correspondence is nearly free on two of three seeds (+0.0001, +0.0037) but costs
+0.0093 on seed 2, the best C6 run; edge fine correspondence is materially needed on every
seed (≈ +0.005…+0.009, i.e. ≈ 4–7 % of MAE). The frozen architecture keeps paired/paired.

## 4. Q3 — relation bundle (Stage D)

* `relation_groups.json` regenerated with the upstream provenance checks: 9/9 checks pass
  on 200 000 pairs (distance one-hot rows, bucket argmax, log-distance match, overlap
  block, binary boundary, log1p integer path count, adjacent bond-type one-hot, path-bond
  mean block, used-slice identity).
* Screening (20-epoch warm adaptation from the C6 seed-0 soup, control 0.128782):
  REL-DIST (distance only) +0.01488, REL-DIST-BOUNDARY (distance + boundary) +0.00518.
  Per the preregistration the simplest survivor (REL-DIST) received the 320-epoch seed-0
  run: soup 0.134159 → **Δ +0.005661**, i.e. `INCONCLUSIVE_KEEP_FULL` (> 0.005, ≤ 0.015).
* Decision: no relation simplification is adopted; the full C6 relation bundle
  (distance + overlap + boundary; log path count stays removed as part of C6) remains the
  minimal supported relation for this round. Distance-only is **not** supported.

## 5. Q4 — dictionary specificity (Stage F)

FINAL-CLEAN = C6 mask, paired/paired bindings, sparse tied-IHT coding; no simplification
was adopted, so the sparse reference is the reused C6 seed-0 soup (0.128499). The matched
DenseTied run differs only in the coding operator (`phi @ Dbar`), same mask, same init
policy, same seed/data order, same optimizer and horizon.

```
G_dict = MAE(DenseTied) − MAE(Sparse) = 0.125563 − 0.128499 = −0.002936
```

Gate: `G_dict >= +0.003` required for a sparse-specificity candidate → **not established**;
single seed, no dense extension per protocol. Dense tied is not worse (it is marginally
better on seed 0), so the round records
`DICTIONARY_COORDINATE_LOAD_BEARING_BUT_SPARSE_DICTIONARY_SPECIFICITY_NOT_ESTABLISHED`.

Dictionary diagnostics (corrected definitions, stage-B summary, 6 checkpoints):

| statistic | BASE (3 seeds) | C6 (3 seeds) |
|---|---|---|
| nnz / row (s = 8) | 8.0 | 8.0 |
| coordinate sparsity fraction | 0.25 | 0.25 |
| active atoms (of 32) | 27 / 28 / 28 | 27 / 27 / 28 |
| effective atoms | 14.1 / 13.7 / 14.7 | 14.5 / 14.8 / 15.3 |
| top-5 atoms' share of activations | 4.7 / 5.0 / 4.5 | 4.8 / 4.6 / 4.4 |
| dictionary movement (Frobenius) | 5.47 / 5.53 / 5.74 | 5.60 / 5.42 / 5.57 |

The sparse code is genuinely sparse (8 of 32 coordinates, 25 % of entries) but strongly
concentrated: the top-5 atoms carry ≈ 4.4–5.0 of the 8 activations per row and one atom is
active in (almost) every row. That concentration is a mechanism note, not a specificity
claim — the matched dense-tied code performs at least as well.

## 6. Stage B core-probe revalidation (frozen registry, distribution-preserving)

Key deltas (own-mask view; the identity view re-opens removed channels and is diagnostic
only). Values are Δ valid MAE; `0.0000` entries are exact zeros.

| probe | BASE s0/s1/s2 | C6 s0/s1/s2 |
|---|---|---|
| GS1 graph-chemistry shuffle | +0.0550 / +0.0788 / +0.0778 | 0.0000 / 0.0000 / 0.0000 |
| GS4 topology shuffle | +0.2583 / +0.2583 / +0.2397 | +0.2674 / +0.2524 / +0.2489 |
| EG2 bond-histogram fill | +0.0023 / +0.0058 / +0.0029 | 0.0000 / 0.0000 / 0.0000 |
| PS1 unary first-moment shuffle | +1.3377 / +1.2549 / +1.2967 | +1.4517 / +1.4316 / +1.3604 |
| PS3 unary count shuffle | ≈ 0 | 0.0000 |
| PS6 pair count shuffle | +0.0023 / +0.0004 / +0.0036 | 0.0000 |
| N3 node assignment shuffle | +0.0273 / +0.0106 / +0.0034 | +0.0154 / +0.0174 / +0.0215 |
| N4 edge assignment shuffle | +0.0867 / +0.0654 / +0.0544 | +0.1052 / +0.0693 / +0.0637 |
| N6 code zero | +0.3238 / +0.2721 / +0.4176 | +0.2678 / +0.2646 / +0.3835 |
| RS3 relation group shuffle | +0.3022 / +0.1444 / +0.0877 | +0.6614 / +0.1118 / +0.0867 |
| RS4 log path count shuffle | +0.0016 / +0.0014 / +0.0236 | 0.0000 / 0.0000 / 0.0000 |
| T1 topology25 zero | +0.1609 / +0.1510 / +0.1257 | +0.1691 / +0.1992 / +0.1338 |
| A2 anchor marginal fill | +0.5808 / +0.6115 / +0.5457 | +0.6440 / +0.7530 / +0.5727 |
| EB3 pair projection zero | +0.7339 / +0.5379 / +1.2450 | +0.4499 / +0.4568 / +1.0979 |

Consistent with the frozen results: the unary first-moment readout dominates; count rows
are dormant everywhere; anchor chemistry remains load-bearing in the clean model (C6 removes
graph chemistry, not anchor chemistry); topology25 is retained and load-bearing; edge
assignment > node assignment in both regimes.

## 7. Adopted architecture

```text
FINAL-CLEAN = C6 mask (no graph atom/bond histogram, no path_count,
              no unary/pair count rows; anchor chemistry retained)
              + paired node binding (per-occurrence product)
              + paired edge binding (per-occurrence product)
              + sparse tied-IHT coding (s = 8, IHT-10)
```

## 8. Claims, falsifications, and what did not replicate

* C6 is a clean, safe simplification of the matched CPU base: **supported** (Q1 gate), with
  the mean-gain caveat driven by the seed-0 BASE outlier.
* The graph-chemistry shortcut is structurally unused in C6: **supported** (GS1 = 0.0000 in
  both views, 3 seeds).
* Fine node correspondence is necessary: **not supported** (nearly free on 2/3 seeds), but
  not removable either (max Δ +0.0093 > gate): **kept**.
* Fine edge correspondence is shell-marginal-replaceable: **rejected** on all seeds (Q2).
* Distance-only relation: **rejected** (Δ +0.005661, inconclusive-keep-full).
* Sparse dictionary specificity: **not established**; dense tied is at least as good
  (G_dict = −0.002936, single seed).
* The earlier H1 finding "graph chemistry marginal is removable" replicates mechanically
  and now multi-seed at the level of the trained model (Q1 + Stage B GS1).

## 9. Limitations

* Three seeds; seed 0 reused (bit-exact verified). BASE seed 0 is an outlier within BASE
  itself, so the C6 mean gain overstates a "C6 is better" reading; the matched-pair deltas
  are the gate input.
* The C6 gate range 0.014683 lies just below the 0.015 UNSTABLE threshold; a fourth seed
  was not required by the gate and was not run.
* DenseTied is seed 0 only (protocol stops on a negative G_dict); the Q4 verdict is
  "not established", not "dense is better".
* Identity-view absolute MAEs are out-of-distribution diagnostics, never verdicts.
* CPU-only matched regime; numbers are not comparable with the historical GPU H1 number
  (0.123549).
* No architecture/hyperparameter search, no PCA32 control, no Stage E (its precondition
  "NODE-INDEP supported" is not met), no shell/radius/shellpair experiment.

## 10. Evidence map

| artifact | content |
|---|---|
| `stage_a_cpu_baseline/gate_a.json` | Q1 gate, per-arm per-seed soups |
| `stage_c_independence/gate_c.json` | Q2 frozen gate, matched-seed deltas, adaptation, adoption |
| `node_edge_independence_diagnostics.json` | per-checkpoint marginals/null norms, per-shell and per-shellpair residual distributions |
| `stage_d_relation/gate_d.json`, `relation_groups.json` | Q3 320-epoch result and provenance checks |
| `stage_f_dictionary_specificity/gate_f.json` | Q4 G_dict and matched sparse reference |
| `stage_b_mechanism/mechanism_summary.json` | frozen probe registry revalidation + dictionary diagnostics |
| `analysis_tables.md` | consolidated read-only tables |
| `final_clean.json` | adopted composition (C6, paired, sparse) |
| `summary.json` | machine-readable roll-up of the whole round |
