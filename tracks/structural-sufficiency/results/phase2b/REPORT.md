# SSRC Phase 2b — standard ZINC-12k performance validation

**Frozen candidates F/M from Phase 2a, trained on the official ZINC-12k train
split, evaluated on official valid and (conditionally) official test against the
historical full-10k P3/R3 checkpoints.**

| field | value |
|---|---|
| protocol_id | `ssrc-phase2b-zinc-v1` |
| status | **COMPLETE** |
| round verdict | **`NO_PRACTICAL_ADVANTAGE`** |
| valid outcome | M promoted (`VALID_GAIN_REPLICATED_ON_OFFICIAL_VALID`); F not promoted |
| mechanism outcome | `MP_INCREMENT_SUPPORTED` on official valid only; seed-1-driven |
| one-shot test outcome | not confirmed (+0.001273 MAE, CI95 [−0.002258, +0.004725]) |
| finalist | **M** (LocalMP-2), seed 0 and seed 1 |
| protocol freeze | `ffa8ee1`; final pre-GPU implementation commit `29c4c2c` |
| run commit | `29c4c2cb653b6a0bf7393c1dcef0bdb635e704a6` |
| GPU budget | **0.947 / 2.5 A100-hours** (37.9%), no failed GPU job |
| test disclosure | official test is **historically exposed**; this was a one-shot re-read, not a first blind test |

---

## 0. TL;DR

1. The historical P3/R3 soups, both seed-matched full-10k masked-SSL encoders,
   the R3 recipe, the train-10k standardiser, and the P3 valid/test predictions
   were reused read-only. All ten reuse conditions pass; the P3 soups replay the
   frozen valid arrays within 2.9e-6 and reproduce the historical test MAE
   exactly (0.069304 / 0.070340).
2. Exactly four new formal models were trained: F and M, seeds 0 and 1, 360
   epochs / 28,440 updates each, with the frozen R3 recipe. Every run matches
   the historical P3 init, seed-matched SSL encoder and batch order, and has the
   expected parameter count (F 463,489; M 464,257). No early stopping, no valid
   selection, no seed-0-based change to seed 1.
3. **Official valid** (development split): P3 0.079844; F 0.079616
   (+0.000228, fails both-seeds-positive); M 0.076788 (**+0.003056**, passes all
   five frozen promotion checks). M is the only finalist.
4. The valid M gain is **threshold-marginal**: +0.003056 against a 0.003 band;
   the molecule-level CI95 is [−1.8e-5, +0.00623] and therefore still straddles
   zero. The frozen rule did not require a valid CI, so promotion proceeded.
5. **One-shot official test**: M 0.068549 vs P3 0.069822, gain **+0.001273**,
   per-seed +0.000669 / +0.001877, CI95 [−0.002258, +0.004725]. Both seeds are
   positive but the pre-registered 0.003 band and the CI>0 condition both fail.
   The round verdict is **`NO_PRACTICAL_ADVANTAGE`**.
6. The valid mechanism label `MP_INCREMENT_SUPPORTED` is technically satisfied
   by the frozen rule (`Gain_MP` +0.002828, both seeds positive, complement
   +0.002108), but seed 0's M−F difference is only +0.000033. The M-vs-F
   separation is essentially a **seed-1** effect and is not confirmed on test.
7. **Do not keep M as the next performance base.** Keep P3 as the base and keep
   M/F as frozen, documented borderline/negative evidence. Do not start Phase 2c.

---

## 1. What was run, and what was not

New GPU work is limited to:

| arm | module | seeds | epochs | rows | params | purpose |
|---|---|---|---:|---:|---:|---|
| F | NodeFFN-2 (per-atom, adjacency-blind) | 0, 1 | 360 | 10,000 | 463,489 | capacity control |
| M | LocalMP-2 (2 GINE rounds on real bonds) | 0, 1 | 360 | 10,000 | 464,257 | MP candidate |

The P3 reference is the **historical** BJOC-Null/R3 checkpoint (413,505 params,
full-10k masked-SSL encoder, 360-epoch R3, last-5 soup), reused without
retraining. No GraphGPS/ESA+/GRIT/GSN or other public model was retrained; no
ZINC-Full, external molecules, new descriptor, new SSL objective, width/depth
search, or hyper-parameter search was run. The Phase 2a model definitions in
`code/phase2a/phase2a_models.py` were not modified.

Because at most one finalist may be promoted, **F has no official-test number**.
This is a protocol consequence, not an omitted run: F's valid gain was null and it
was not the finalist. Its valid result is reported below and is never placed in
the official-test table.

The official test was read exactly once, after the valid decision and after both
M test prediction files were frozen and hashed. No test-driven change, rerun,
seed addition, subgroup search or recipe change followed.

---

## 2. P3 reuse audit (read-only, before any F/M training)

Source: `p3_reuse_audit.json`. All conditions are `true`.

| check | result |
|---|---|
| soup SHA-256 seed 0 / seed 1 | `a021d36e…` / `b622824a…` (match R3 and terminal-test records) |
| parameters / forbidden modules | 413,505 / none |
| rebuilt init matches R3 record (`init_full`, `init_group_state`, encoder) | both seeds exact |
| seed-matched full-10k SSL encoder | `006fabc1…` (s0), `ff8d6474…` (s1); `encoder_used` `7165eb90…` / `54a232ed…` |
| R3 recipe / batch order | 360 epochs, 28,440 updates; order `3c964a71…` / `8710751a…` |
| P3 valid replay vs frozen arrays | max abs Δ **1.907e-6** (s0), **2.861e-6** (s1) on the compute CPU (0.0 locally); < 5e-5 |
| P3 valid MAE reproduction | 0.07910149778419873 / 0.08058639967715135, abs diff 0.0 |
| P3 test MAE reproduction | 0.0693043516818434 / 0.07033976497594267, abs diff 0.0 |
| test structure fingerprint | `processed_pt` `bd7a1ceb…`, `structure_sha256` `75668e78…`; match |
| model/object/SSL code hashes | match the revision used when P3 was evaluated |

The small replay difference on `res` is float32 reduction-order noise on a
different CPU/BLAS; the arrays are hash-verified historical files, not
regenerated predictions. The 1.9e-6 difference is three orders of magnitude
below the 0.003 decision band and does not affect the comparison.

The historical `valid_R3*.npz` / `test_P3*.npz` files are gitignored. They were
staged read-only on the compute host (as the frozen `.pt` soups already are) and
verified by both file SHA-256 and array SHA-256. At the test stage the P3 test
arrays were byte-copied into this results directory and frozen before any test
label read. This is a portability staging step, not a numerical substitution.

### CPU gates and cost audit

`cpu_gates_phase2b.json`: **10/10 gates pass** (`G1`–`G10`), including per-seed
init/SSL parity, exact batch order, Phase 2a frozen-prediction replay (max Δ
1.9e-6–5.2e-6), permutation/edge-direction/batch invariance, F adjacency
blindness (0.0), M adjacency and bond-feature sensitivity, no label input, local
gradients and parameter counts. `G4` (Phase 2a training-loop parity) accepts a
max state Δ ≤ 1e-4 instead of bitwise equality because the remote CPU/BLAS
reproduces the 6-update loop with 1.4e-6 (F) / 6.5e-7 (M) state difference;
batch order and update count are exact, and the soup-fit MAE differs by
5.7e-8 / 8.5e-7. This tolerance was set before the formal runs.

`flops_phase2b.json` (forward GEMM FLOPs, fixed 128-graph batch): P3 1.995 G;
F 1.147×; M 1.151×.

---

## 3. Formal training

All four runs: official train 10,000 rows, raw-y L1, batch 128, Adam, lr 1e-3
cosine → 1e-5 per step, wd 1e-5, clip 5.0, 360 epochs, last-5-epoch parameter
mean, shuffle seed `seed + 101`, no early stopping and no valid-based selection.
The supervised targets match the official PyG train targets exactly.

| run | params | updates | wall (s) | mean step (ms) | init / encoder / order | soup SHA-256 |
|---|---:|---:|---:|---:|---|---|
| F s0 | 463,489 | 28,440 | 733.8 | 25.53 | exact / exact / exact | `2e5a3451…` |
| F s1 | 463,489 | 28,440 | 744.4 | 25.92 | exact / exact / exact | `2f04c09d…` |
| M s0 | 464,257 | 28,440 | 929.6 | 32.05 | exact / exact / exact | `6eb82d94…` |
| M s1 | 464,257 | 28,440 | 938.3 | 32.35 | exact / exact / exact | `f3e1cc71…` |

Every record says `labels_used = official train 10000 raw y only`,
`official_valid_loaded = False`, `official_test_loaded = False`. Valid predictions
for all four runs were frozen before the valid label read.

---

## 4. Official-valid result (development split, single label read)

Primary metric: raw-y MAE on the 1,000 official-valid molecules, seed-mean of two
single-model seeds.

| model | seed 0 | seed 1 | seed mean | gain vs P3 | paired CI95 (seed-mean) | better fraction |
|---|---:|---:|---:|---:|---:|---:|
| P3 (historical, reused) | 0.079101 | 0.080586 | **0.079844** | — | — | — |
| F | 0.076367 | 0.082865 | 0.079616 | +0.000228 | [−0.004850, +0.005419] | 0.486 |
| M | 0.076333 | 0.077242 | **0.076788** | **+0.003056** | [−0.000018, +0.006230] | 0.521 |

Frozen promotion checks:

| check | F | M |
|---|---|---|
| mean gain ≥ 0.003 | false (+0.000228) | **true (+0.003056)** |
| both seeds positive | false (+0.002735 / −0.002279) | **true (+0.002768 / +0.003345)** |
| drop-top-1 favourable ≥ 0.002 | false (−0.001180) | **true (+0.002683)** |
| drop-top-10 P3-error ≥ 0.002 | false (−0.000977) | **true (+0.002004)** |
| penalty-complement gain > 0 | true (+0.000884) | **true (+0.002992)** |
| qualifies | **false** | **true** |

M is the only finalist; `valid_decision.json` SHA-256
`ea8e68d8a6ff3ee05c1397d58d31b006757da33497f92dfa209186718bfe909b`.

The margins matter: M's mean gain exceeds the band by **0.000056**, its
drop-top-10 check by **0.000004**, and its CI95 lower bound is **−0.000018**.
This is a valid-split threshold pass, not a robustly significant effect. The
frozen rule is applied as written; no threshold was changed.

### Descriptive subgroups (valid)

| subgroup | n | P3 | F | M | M − P3 |
|---|---:|---:|---:|---:|---:|
| all | 1000 | 0.079844 | 0.079616 | 0.076788 | +0.003056 |
| ordinary (`y > −3`) | 935 | 0.068090 | 0.068261 | 0.064977 | +0.003113 |
| tail (`y ≤ −3`) | 65 | 0.248923 | 0.242948 | 0.246684 | +0.002239 |
| cycle-penalty proxy | 36 | 0.229828 | 0.247178 | 0.225047 | +0.004781 |
| penalty complement | 964 | 0.074243 | 0.073358 | 0.071251 | +0.002992 |
| without extreme valid row 172 | 999 | 0.074689 | 0.075869 | 0.071523 | +0.003166 |

M's gain is broad across ordinary molecules and the penalty complement; it is not
created by the single extreme valid-172 molecule (removing it leaves +0.00317).
F is slightly better than P3 on the tail but worse on ordinary molecules. The
penalty-proxy group is small (36) and the proxy is cycle-basis-order sensitive for
a handful of rows; it is descriptive only.

### Mechanism (valid only)

`Gain_MP` = MAE(F) − MAE(M), positive means M better:

| seed | Gain_MP |
|---|---:|
| 0 | +0.000033 |
| 1 | +0.005623 |
| seed mean | **+0.002828** |
| paired CI95 | [−0.002204, +0.007517] |
| penalty complement | +0.002108 |

The frozen checks (`mean ≥ 0.002`, both seeds positive, complement positive) all
pass, so the decision record labels the mechanism `MP_INCREMENT_SUPPORTED`. The
honest reading is weaker: **seed 0's F and M are indistinguishable**, and the
entire separation comes from seed 1. With two seeds this is not a stable
mechanism estimate. The test stage did not evaluate F, so there is no
test-level M-vs-F check.

---

## 5. One-shot official test (after promotion only)

Both M test predictions and byte-exact copies of the historical P3 test
predictions were frozen and hashed before `test_y` was read. The structure
fingerprint matched the historical test split. The test labels were read once;
`test_decision.json` SHA-256
`ffb1febebf7cd50928dc51b0afb00b4829730046aede1732b965c0d336acdabb`.

| model | seed 0 | seed 1 | seed mean | gain vs P3 |
|---|---:|---:|---:|---:|
| P3 (historical, reused) | 0.069304 | 0.070340 | **0.069822** | — |
| M (finalist) | 0.068636 | 0.068462 | **0.068549** | **+0.001273** |

Per-seed gain: **+0.000669** (s0), **+0.001877** (s1). Paired molecule-level
bootstrap: mean +0.001273, CI95 **[−0.002258, +0.004725]**, better fraction 0.506.

| test-rule check | result |
|---|---|
| mean gain ≥ 0.003 | **false** (+0.001273) |
| both seeds positive | true |
| CI95 lower > 0 | **false** (−0.002258) |
| confirmed | **false** |

The gain shrinks from +0.003056 on valid to +0.001273 on test; both seeds remain
positive but the effect is below the practical band and statistically
indistinguishable from zero. Test subgroups: ordinary (n=923) P3 0.064300 → M
0.062985 (+0.001315); tail (n=77) P3 0.136012 → M 0.135241 (+0.000772). The
seed-0 ordinary subgroup is actually slightly worse for M (0.063952 vs 0.062860),
another sign of seed fragility. The paired MSE-difference CI also straddles zero
([−0.006788, +0.006664]); the largest single-molecule MSE contribution is 20.8%
(row 402), so the result is not one-molecule-driven but is also not precise.

No test-driven follow-up was performed. M is not retrained, reseeded, or tuned on
the test result.

---

## 6. Official ZINC-12k comparison table

All rows are official ZINC-12k **test** MAE (1,000-molecule test split of
`ZINC(subset=True)`), lower is better. This is a landscape, not a controlled
comparison: parameter counts, seeds, structural priors and pretraining differ.
**ZINC-Full is a different benchmark and is excluded** (ESA+ 0.011 ± 0.000 on
ZINC-Full is not comparable). No internal Phase 2a Monitor number (e.g. 0.085345)
appears here.

| model | ZINC-12k test MAE | params | prior / pretraining | source |
|---|---:|---:|---|---|
| **BJOC-P3/R3** (this repo, 2 seeds) | **0.069822** | 413,505 | masked atom/bond SSL on the same 10k train structures; BJOC encoder; no RWSE/ring features | historical `bjoc_r3_frozen_terminal_test_v1` |
| **Phase 2b M** (this round, 2 seeds) | **0.068549** | 464,257 | P3 + LocalMP-2 on real bonds; same SSL; no external data | this report |
| GraphGPS (GINE + Transformer + RWSE-20) | 0.070 ± 0.004 | 423,717 | RWSE-20; no pretraining | Rampášek et al., arXiv:2205.12454, Table 3 / B.1 |
| GRIT (no MP, RRWP) | 0.059 ± 0.002 | ~500k budget | RRWP; no pretraining | Ma et al., arXiv:2305.17589, Table 1 |
| CSA-rings (no MP) | 0.056 ± 0.002 | ~500k budget | RWSE + explicit ring-membership features | Maziarka et al., arXiv:2304.10933, Table 1 |
| **ESA+** | **0.052 ± 0.001** | not stated in Table 6 | ESA edge-set attention + RWSE + ring-membership; no pretraining | Buterez et al., *Nat Commun* 16 (2025), Table 6, DOI 10.1038/s41467-025-60252-z |
| GSN (substructure counts + MPNN) | 0.101 ± 0.010 | ~500k budget | graphlet/substructure counts; no pretraining | GraphGPS Table 3 |
| CIN (higher-order MPNN) | 0.079 ± 0.006 | ~500k budget | simplicial/cellular complexes; no pretraining | GraphGPS Table 3 |
| GIN (reference MPNN) | 0.526 ± 0.051 | ~500k budget | none | GraphGPS Table 3 |
| GatedGCN (reference MPNN) | 0.282 ± 0.015 | ~500k budget | none | GraphGPS Table 3 |
| PNA (reference MPNN) | 0.188 ± 0.004 | ~500k budget | none | GraphGPS Table 3 |

Reading: P3 is at GraphGPS level; M is nominally 0.0013 below P3 and 0.0015 below
the GraphGPS mean, both well inside GraphGPS's ±0.004 run spread. Neither is
competitive with the RWSE/ring/ESA+ line (GRIT 0.059, CSA 0.056, ESA+ 0.052).
P3/M additionally use masked SSL on the same train structures, unlike the
literature rows, so this is not a like-for-like architectural ranking.

The Phase 2b conclusion does **not** depend on the literature gap: the internal
pre-registered test rule already failed.

---

## 7. Cost

| quantity | P3 | F | M |
|---|---:|---:|---:|
| parameters | 413,505 | 463,489 (+12.09%) | 464,257 (+12.27%) |
| forward FLOPs, fixed 128-graph batch | 1.995 G | 2.288 G (1.147×) | 2.297 G (1.151×) |
| formal training wall, seed 0 / seed 1 | 677.6 / 676.0 s | 733.8 / 744.4 s | 929.6 / 938.3 s |
| mean step time | ~23.8 ms | 25.5 / 25.9 ms | 32.1 / 32.4 ms |
| wall overhead vs P3 | — | +9.2% | +38.0% |
| official-valid gain | — | +0.000228 (null) | +0.003056 (threshold-marginal) |
| official-test gain | — | not tested | +0.001273 (not confirmed) |

M costs +12.3% parameters, +15.1% forward FLOPs and +38% supervised training wall
for a +0.00127 test gain that is not confirmed. F costs less wall (+9.2%) but has
no measurable gain. The frozen recipe was not shortened or retuned to hide this.

### GPU ledger (cap 2.5 A100-h)

| job | GPU | wall | GPU-h |
|---|---:|---:|---:|
| smoke (2 epochs, 1,024 rows) | gpu0 | 14 s | 0.0039 |
| F s0+s1 formal + valid freeze | gpu0 | 1,497 s | 0.4158 |
| M s0+s1 formal + valid freeze | gpu1 | 1,889 s | 0.5247 |
| one-shot M test forward | gpu0 | 9 s | 0.0025 |
| **total** | | **3,409 s** | **0.947** |

Formal training alone is 3,346 s = 0.929 GPU-h. No GPU job failed. The five CPU
prep/audit attempts and the two CPU scoring jobs consumed 0 GPU-hours; they are
listed in `gpu_ledger.json` for completeness. Remaining budget: 1.553 GPU-h.

---

## 8. The seven questions

**Q1. Does the Phase 2a insertion improve official ZINC-12k valid MAE over the
frozen P3 under the standard full-10k protocol?**
For M, yes: 0.076788 vs 0.079844, +0.003056, both seeds positive. For F, no:
+0.000228 with seed 1 negative. The Phase 2a capacity effect survives on valid
only for the MP-shaped arm, and only at the edge of the practical band.

**Q2. Is the valid gain at least the pre-registered 0.003 practical band and
robust to the top favourable molecule, the top-10 P3 errors, and the penalty
complement?**
M passes all five frozen checks: +0.003056, drop-top-1 +0.002683, drop-top-10
+0.002004, complement +0.002992. The margins are 0.000056 / 0.000004, and the
molecule-level CI95 still includes zero. F fails all but the complement check.

**Q3. Is the gain specific to real-bond message passing rather than generic
capacity?**
On valid, the frozen rule says yes (`Gain_MP` +0.002828, both seeds positive,
complement +0.002108), but seed 0's M−F difference is +0.000033 and the
separation is effectively seed-1-only. The test stage only evaluated M (one
finalist), so there is no test-level M−F estimate. The mechanism claim is
therefore weak and not independently replicated.

**Q4. Does the promoted finalist confirm the gain on the one-shot official test
at the 0.003 band with CI > 0?**
No. M's test gain is +0.001273, CI95 [−0.002258, +0.004725], both seeds positive
but below the band and not significant. The valid gain did not replicate at the
pre-registered practical threshold.

**Q5. What does the candidate cost relative to P3?**
+12.1% (F) / +12.3% (M) parameters, +14.7% / +15.1% forward FLOPs, +9.2% / +38.0%
training wall. M's test gain of 0.00127 does not justify the +38% wall or the
extra parameters at the pre-registered standard.

**Q6. Where does it stand against the official ZINC-12k literature?**
P3 0.069822 is GraphGPS-level (0.070 ± 0.004). M 0.068549 is nominally slightly
better than both but inside GraphGPS's run spread, and clearly behind GRIT
(0.059), CSA-rings (0.056) and ESA+ (0.052). This is not a new SOTA result and is
not a controlled comparison (different priors, seeds and pretraining).

**Q7. Is the new model worth keeping as the next performance base?**
**No.** The terminal test does not confirm a practical gain; the valid gain is
threshold-marginal; the M-vs-F separation is seed-1-driven; and the cost is
+38% training wall. Keep P3 as the performance base. Keep the F/M code and the
frozen checkpoints as documented borderline/negative evidence and as a reusable
control pair, not as the next base. Do not start Phase 2c.

**One next scientific question (if the line is revisited):**
With **3–5 seeds and a parameter-matched trunk/readout control** (the same
≈50k parameters added to the reader or trunk instead of a local module), is M's
valid gain and the M−F separation reproducible, or was the Phase 2b signal a
seed-1 fluctuation? This directly tests capacity vs MP specificity and the
seed fragility that the two-seed design cannot resolve. It must be a new
pre-registration, not a Phase 2c continuation.

---

## 9. Verdict

**Round verdict: `NO_PRACTICAL_ADVANTAGE`.**

Why this label rather than the alternatives:

- `STANDARD_ZINC_GAIN_CONFIRMED` is false: the one-shot test gain is +0.001273
  with CI crossing zero and below the 0.003 band.
- `VALID_GAIN_NOT_REPLICATED` would misdescribe the valid result: the valid gain
  *was* observed for M (+0.003056) under the frozen rule. What failed is the
  terminal test confirmation.
- `CAPACITY_GAIN_WITHOUT_MP_SPECIFICITY` is not what the valid decision says:
  the frozen `Gain_MP` checks pass. But the separation is seed-1-only, so this
  label is a live concern rather than the formal verdict.
- `MP_INCREMENT_SUPPORTED` is a valid-only mechanism label, not a practical
  performance verdict; the test did not confirm the overall gain.
- `PROTOCOL_OR_REPRODUCTION_UNRESOLVED` is false: the audit, gates, P3 replay,
  test fingerprint and hashes all pass.

Verdict dimensions:

| dimension | label |
|---|---|
| practical standard-protocol advantage | **`NO_PRACTICAL_ADVANTAGE`** |
| valid replication (M only) | `VALID_GAIN_REPLICATED_ON_OFFICIAL_VALID` (threshold-marginal) |
| mechanism (valid only) | `MP_INCREMENT_SUPPORTED` (seed-1-driven; not test-confirmed) |
| protocol / reproduction | resolved (all audit and gate checks pass) |

---

## 10. Limitations, non-claims and deviations

**Limitations.**
- Two seeds per arm; the seed-0 M−F difference is ~0 and the mechanism estimate
  is dominated by seed 1.
- The valid M gain is threshold-marginal (0.000056 over the band; CI includes 0);
  the test CI is wide (±0.0035).
- Only one finalist could be tested; F has no official-test number.
- The official test is historically exposed and was re-read once; it is not a
  first blind test and cannot be presented as one.
- The official valid split is a development split that was already exposed in the
  historical P3 rounds.
- One dataset (ZINC-12k), one backbone family, one recipe; no width-matched
  control, no third seed, no seed-mean ensembling.
- The cycle-penalty proxy (`max cycle basis length > 6`) is order-sensitive for a
  handful of rows; the subgroup is descriptive, never an input or selection rule.
- FLOP counting covers forward GEMMs; scatter and Python overhead are not counted,
  which is why the M wall overhead (+38%) exceeds its FLOP overhead (+15%).

**Non-claims.**
- M is not a new SOTA model and not a novel architectural claim; it is P3 plus a
  frozen Phase 2a module.
- This is not evidence that message passing is generally useless; it is evidence
  that this local MP insertion does not deliver a confirmed standard-ZINC-12k
  gain under this recipe.
- The 0.085345 Phase 2a internal Monitor number is not an official ZINC-12k
  number and is not compared with the literature table.
- ZINC-Full numbers (e.g. ESA+ 0.011) are a different benchmark and are excluded.
- No test-driven follow-up, no third seed, no width/depth/gating/SSL/recipe
  variant was tried after the verdict.

**Deviations from the frozen protocol (all pre-result or portability).**
1. `G4` loop parity accepts a 1e-4 state tolerance instead of bitwise equality;
   the remote CPU/BLAS gives 1.4e-6, while batch order and updates are exact.
   Set before the formal runs.
2. The historical gitignored P3 prediction arrays were staged read-only on the
   compute host and hash-verified; the test arrays were byte-copied and frozen
   before labels. The protocol's implementation note records this.
3. `score-valid` and `score-test` additionally verify that the frozen P3 arrays
   reproduce the historical MAE (0.0 difference at both endpoints).
4. The `run-arm`/`prep` wrappers only sequence stages; they do not change the
   training loop, data, recipe, metrics or decision rule.
5. No scientific threshold, arm, metric, subgroup or test rule was changed after
   seeing any Phase 2b score.

---

## 11. Reproduction and artefacts

Protocol: `results/phase2b/PROTOCOL.md` (`ssrc-phase2b-zinc-v1`), frozen at
`ffa8ee1`, final pre-GPU implementation at `29c4c2c`. Config and input hashes:
`configs/phase2b.yaml` (SHA-256 `bd0c1441…`). Run commit:
`29c4c2cb653b6a0bf7393c1dcef0bdb635e704a6`.

Commands (on `res`, via `rr`):

```bash
python tracks/structural-sufficiency/code/phase2b/phase2b_runner.py prep --device cpu
python tracks/structural-sufficiency/code/phase2b/phase2b_runner.py smoke --device cuda
python tracks/structural-sufficiency/code/phase2b/phase2b_runner.py run-arm --arm F --device cuda
python tracks/structural-sufficiency/code/phase2b/phase2b_runner.py run-arm --arm M --device cuda
python tracks/structural-sufficiency/code/phase2b/phase2b_runner.py score-valid --device cpu
python tracks/structural-sufficiency/code/phase2b/phase2b_runner.py test-predict --device cuda
python tracks/structural-sufficiency/code/phase2b/phase2b_runner.py score-test --device cpu
```

Key artefacts and SHA-256:

| artefact | SHA-256 |
|---|---|
| `p3_reuse_audit.json` | all 10 conditions true (see file) |
| `valid_decision.json` | `ea8e68d8a6ff3ee05c1397d58d31b006757da33497f92dfa209186718bfe909b` |
| `valid_metrics.json` | `97718bb3f550c2391955e1e032821f5be54f5b6931a0b7658463997b71652d85` |
| `valid_prediction_freeze.json` | `ba60747726c2b872163faa483156b9d0cbb7e31780ea3af784a8af114b96bf3c` |
| `test_decision.json` | `ffb1febebf7cd50928dc51b0afb00b4829730046aede1732b965c0d336acdabb` |
| `test_metrics.json` | `3b25bd4d9bdb29646819f5d15221ae17e96a0136703c3e5c6e95f2e4cbc68bbd` |
| `test_prediction_freeze.json` | `71fa92534315a28e6f38a4f247cec876988c3a4e476df574faefdbfbb4179317` |
| F soup s0 / s1 | `2e5a3451…` / `2f04c09d…` |
| M soup s0 / s1 | `6eb82d94…` / `f3e1cc71…` |
| F valid pred s0 / s1 | `535632af3bdb…` / `f379d13e17fd…` |
| M valid pred s0 / s1 | `0ad1f1b4f780…` / `d1b0fb120731…` |
| M test pred s0 / s1 | `fc94bfe4769c…` / `256dc1e76048…` |

`valid_read.lock` and `test_read.lock` record one read each. The soups and
prediction arrays are kept in this results directory (gitignored) and their
hashes are committed. `run_manifest.tsv` lists the `rr` run IDs, commit, GPU
seconds and key hashes; `gpu_ledger.json` is the full accounting.
