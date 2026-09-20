# PSCD-R-v0 — Frozen-Dictionary Motif Readability Audit (analysis)

**Question.** PSCD-v0's Stage-A object is healthy (100 % exact decode,
permutation-invariant, `mean(M_G/n_G)=0.320`, MDL ratio 0.215, 64/64 reuse,
clean ports, 0 full-PSCD collisions), but its Stage-B composition reader read the
frozen dictionary at valid MAE ~0.54 against a raw atom graph at ~0.31. This round
freezes the entire dictionary and changes **only how a tiny reader represents a
dictionary atom**, to separate H1 (opaque-atom: independent per-motif ID
embeddings destroy atom/bond parameter sharing) from H2 (port-bottleneck: the
reader collapses attachment ports into one occurrence state too early).

**Verdict (first line).**

> **The dominant PSCD-v0 failure was the opaque motif reader; frequency-derived
> motifs remain viable** — and, concretely, a large part of it was a *correctable
> batching defect* in the Stage-B composition collate, not a property of the
> frozen object.

**Next decision.** `develop the structured dictionary reader`. Do **not** start
task-aware motif discovery yet.

---

## 0. Provenance

| item | value |
|---|---|
| round | PSCD-R-v0 (pre-registered in `notes/pscd_reader_diagnostic_preregistration.md`) |
| code | `tracks/ksvd/code/run_pscd_reader_diagnostic.py` |
| tests | `tracks/ksvd/tests/test_pscd_reader.py` (5 data-free tests, pass) |
| frozen artifact | `results/pscd_compositional/stageA.pkl` (PSCD-v0, commit `cdae8b0`; md5 `b180e9695fb4f0a2cb366338bd58f459`, local == remote) |
| phase-1 commit | `782f49e4` (all readers, screening) |
| phase-2 commit | `16e6d6f5` (raw-control extension) |
| execution | remote A100 host `res` (GPU 0 phase 1, GPU 1 phase 2) |
| splits | official PyG ZINC `subset=True`; **train 10 000 / valid 1 000**; official `test` never loaded |
| `y` | used only inside the readers |
| results | `tracks/ksvd/results/pscd_reader_diagnostic/` (git-ignored) |

## 1. Freeze gate

Before any training the run re-decoded every frozen `(D, C_G)` and compared exact
colored canonical keys:

| split | exact | fraction |
|---|---|---|
| train (10 000) | 10 000 | **1.000000** |
| valid (1 000) | 1 000 | **1.000000** |

No vocabulary, merge rule, partition, canonical motif, port or composition edge
was changed; no relearning / sweep / official test. **PASS.**

## 2. Pre-run reader defect (the main finding)

The PSCD-v0 as-shipped composition collate lays the node array out interleaved per
graph (`[occ_g0, conn_g0, occ_g1, conn_g1, …]`, `off += M + Ec`) while the reader
forward assumes all occurrence nodes first:

```
h[:M_total] = E_M(occ_ids)      h[M_total:] = E_B(conn_ids)
```

For any batch with `>1` graph these disagree, so the motif/bond features land on
the wrong nodes. Verified with a fixed random init: batch-of-4 predictions differ
from the per-graph predictions by up to **1.36 MAE** for the composition reader,
while the bag and raw readers reproduce single predictions to `<1.4e-6`. Locked by
`test_shipped_composition_collate_is_not_batch_invariant` and
`test_corrected_composition_collate_is_batch_invariant`.

Two opaque controls were therefore run:

* `R0ship` — the as-shipped reader, unchanged (historical reproduction);
* `R0` — the **same module**, with only the collate node order fixed to
  `[all occurrence nodes | all connection nodes]` and edges remapped.

## 3. Matched learning-curve results

Frozen dictionary; `d = 32`, `L_outer = 2`, seed 0, Adam lr 1e-3 / wd 0 / clip 5,
batch 128; evaluation every 100 steps; train 10 000 / valid 1 000. `best-valid` is
the best valid MAE achieved at or before the step.

### 3.1 Matched step 2000 (all five readers have an evaluation here)

| reader | params | train MAE | valid MAE | best-valid ≤2000 | share of corrected-opaque→raw gap |
|---|---:|---:|---:|---:|---:|
| `R0ship` opaque (as-shipped) | 10 497 | 0.5586 | 0.5779 | 0.5779 | worse than opaque |
| `R0` opaque (corrected) | 10 497 | 0.4183 | 0.4558 | 0.4558 | 0 (reference) |
| `R1` structured-static | 16 737 | 0.3933 | 0.4167 | 0.4167 | **0.93** |
| `R2` port-resolved | 29 281 | 0.3596 | 0.3794 | 0.3794 | **1.82** |
| `R3` raw atom graph | 10 241 | 0.4066 | 0.4139 | 0.4139 | 1.00 (target) |

At the matched budget the port-resolved structured reader is **better than the raw
atom reader** and the structured-static reader is on par with it.

### 3.2 Terminal screening step and the 6000-step ceiling

| reader | steps | params | train MAE | valid MAE | best valid | recovery (valid) | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `R0ship` opaque (as-shipped) | 2600 (early) | 10 497 | 0.5525 | 0.5683 | 0.5683 | — | 71 s |
| `R0` opaque (corrected) | 3000 | 10 497 | 0.3892 | 0.4346* | 0.4346 | 0 (ref) | 74 s |
| `R1` structured-static | 2800 (early) | 16 737 | 0.3785 | 0.4115 | 0.4115 | 0.55 @3000 | 105 s |
| `R2` port-resolved | 6000 (extended) | 29 281 | 0.2740 | **0.3246** | 0.3246 @5900 | 1.77 @3000 | 263 s |
| `R3` raw atom graph | 6000 (extended) | 10 241 | 0.3165 | 0.3363 | 0.3363 @6000 | 1.00 | 172 s |

\* `R0` final-at-3000 is 0.4470; best ≤3000 is 0.4346 @2900. `R3` phase-1
early-stopped at 2100 (best ≤3000 = 0.4139); the row above is the pre-registered
**raw extension** (phase 2, `--no-early-stop`, ≤6000).

Recovery ratio `R_recover = (M_opaque − M_structured) / (M_opaque − M_raw)` at
matched steps, `M_opaque = R0`:

| step | R1 valid | R2 valid | R1 train | R2 train |
|---:|---:|---:|---:|---:|
| 2000 | 0.93 | 1.82 | 0.93 | 1.98 |
| 3000 | 0.55 | 1.77 | 0.35 | 1.21 |

### 3.3 Gap decomposition at matched step 2000 (valid)

| contribution | Δ valid MAE | share of `R0ship→raw` gap (0.164) |
|---|---:|---:|
| as-shipped reader defect (`R0ship` 0.578 → `R0` 0.456) | 0.122 | **74 %** |
| H1 opaque-atom → shared static structure (`R0` 0.456 → `R1` 0.417) | 0.039 | 24 % |
| H2 static → port-resolved (`R1` 0.417 → `R2` 0.379) | 0.038 | 23 % |
| `R2` vs raw (`R2` 0.379 vs `R3` 0.414) | −0.035 | overshoots |

So the **reader**, not the dictionary, dominates the historical gap — and within
the reader the batching defect is the single largest contribution, with the opaque
parameterisation and port-collapse each adding a real but smaller share.

### 3.4 Learning curves

`learning_curve_valid_combined.png` and `learning_curve_train_combined.png`
(valid / train MAE vs optimizer step, with `R3` extended to 6000). All structured
and opaque readers descend monotonically in trend; `R0ship` plateaus at ~0.57; the
raw reader is visibly noisier and still descending at the ceiling.

## 4. Case classification (frozen §29–32)

* `R_recover^valid ≥ 0.50` at every matched step, and ≥ 0.60 at step 2000;
  structured train MAE (`R1` 0.393, `R2` 0.360) is **below** raw train MAE
  (0.407) at step 2000.
* `R2 ≫ R1 ≫ R0ship` and `R2 < R1` at matched steps.

⇒ **Case A** (structured dictionary solves most of the gap) **and** the port
sub-result of Case B (port resolution is specifically necessary). The
frequency-only motif abstraction is **not** implicated: a shared, low-capacity
reader reads the frozen dictionary as well as the raw graph at the tested budget.

## 5. Mechanism audit (trained `R2` dictionary encoder)

**A. Motif-embedding diversity — no collapse.** 64 learned motifs, `h_k`:
mean pairwise cosine **0.360** (min −0.304, max 0.995), fraction cosine > 0.99 =
0.001, mean pairwise Euclidean 17.30, effective rank (participation ratio) 2.58.
The dictionary is not collapsed onto a few directions.

**B. Structural-similarity sanity — weak but correct sign.** Pairs sharing a
connected induced substructure of ≤4 atoms (`n = 1091`) have mean embedding
distance **16.89** vs **17.79** for unrelated pairs (`n = 925`); related motifs are
closer by ~0.9. Descriptive only, not an objective.

**C. Port differentiation.** For every size-≥2 motif the within-motif pairwise
cosine of `r_{k,p}` averages **0.342**; 95.3 % of motifs have mean port cosine
< 0.9. Distinct canonical ports carry distinct structural states.

## 6. Evaluation-only ablation (critical test, no retraining)

Replace every dictionary port vector `r_{k,p}` by the motif mean
`\bar r_k = (1/|V_k|) Σ_p r_{k,p}` at inference, for the best structured reader
(`R2`, valid 0.3396 / train 0.2889):

| | train MAE | valid MAE |
|---|---:|---:|
| intact | 0.2889 | **0.3396** |
| port vectors → motif mean | 0.5575 | **0.5598** |
| Δ | +0.269 | **+0.220** |

The reader relies heavily on port-specific dictionary state; port-resolved
semantics are genuinely used, not decorative.

## 7. Parameter accounting

| reader | dictionary-shared | per-motif lookup | outer reader | head | total |
|---|---:|---:|---:|---:|---:|
| `R0ship` / `R0` | 0 | 2 720 | 6 336 | 1 089 | 10 497 |
| `R1` | **9 216** | **0** | 6 336 | 1 089 | 16 737 |
| `R2` | **9 216** | **0** | 18 880 | 1 089 | 29 281 |
| `R3` | 0 | 0 | 6 336 | 1 089 | 10 241 |

`R1`/`R2` contain no large motif-specific lookup table; all dictionary structure
lives in the shared 9 216-parameter encoder. Parameter efficiency (valid MAE per
unit of parameter increase, matched step 2000):

* `R0 → R1`: −0.039 MAE for +6 240 params (6.2 × 10⁻⁶ MAE/param);
* `R1 → R2`: −0.038 MAE for +12 544 params (3.0 × 10⁻⁶ MAE/param);
* `R0 → R2` vs raw: `R2` (29 281 params, 0.379) is 0.035 better than raw
  (10 241 params, 0.414) at step 2000, but at the 6000 ceiling `R2` (0.340/best
  0.325) only **ties** raw (0.336). Per parameter, the raw reader is the more
  efficient fitter; `R2`'s value is that it *reads the compressed code*, not that
  it beats raw on efficiency.

## 8. Answers to the five questions

* **Q1 — was the PSCD-v0 reader failure mainly the opaque categorical ID?**
  Not mainly, and not only. A correctable batching defect is the single largest
  contribution (0.122 of the 0.164 matched-budget gap, 74 %); the opaque-ID
  parameterisation then contributes a real but smaller share (0.039, 24 %). Both
  are *reader* problems.
* **Q2 — does a shared intra-motif encoder recover atom/bond parameter
  sharing?** Yes. `R1` (shared 9.2 k encoder, zero per-motif lookup) beats
  corrected opaque `R0` by 0.039 at matched step 2000 and matches the raw reader.
* **Q3 — are port-specific dynamic states necessary?** Yes. `R2 < R1` by 0.038
  at matched 2000 (and `R2 < R1` at 3000), and the inference-only port-mean
  ablation degrades `R2` valid MAE by +0.220.
* **Q4 — how far is the composition representation from raw afterwards?** At the
  tested budgets the structured reader is at or below raw (`R2` 0.379 ≤ raw 0.414
  at step 2000; `R2` 0.325 best ≈ raw 0.336 at the 6000 ceiling). The historical
  *converged* raw (≈0.311 at ~13 k steps) was not reached by any reader within the
  registered ceiling, so the converged gap is **not resolved**.
* **Q5 — reader problem, motif-discovery problem, or both?** Predominantly a
  **reader** problem (defect + opaque parameterisation + port collapse). There is
  no evidence of a motif-discovery bottleneck: frequency-derived motifs are
  readable by a shared structured reader.

## 9. Caveats and discipline

* Compute discipline respected: each model ≤ 6000 optimizer steps; only the
  best structured variant (`R2`) and the raw control were extended, per the
  pre-registered extension gate; no seed/LR/size sweeps, no soup, no official
  test.
* The raw reader is noisy and still descending at the 6000 ceiling (train 0.317 /
  valid 0.336, best at step 6000), and two raw runs differ by ~0.07 at step 2000
  (GPU `index_add` non-determinism), so point comparisons against raw carry a
  variance of that order. The matched 2000-step table (all readers) is the most
  robust single comparison; the 6000 rows are indicative.
* `R2` has 2.8× the parameters of `R0`/`R3`; the round does **not** claim `R2`
  beats raw on a parameter-matched basis.

## 10. Final conclusion and next decision

**Final first line (frozen choice):**

> **The dominant PSCD-v0 failure was the opaque motif reader; frequency-derived
> motifs remain viable.**

The PSCD-v0 Stage-B composition number (valid ≈ 0.54) was produced by a reader
that was both opaque *and* batching-defective; with the batching corrected and the
dictionary atom read structurally (shared encoder + port-resolved state), the
frozen frequency dictionary is read at least as well as the raw atom graph at the
tested budget, and port-resolved state is demonstrably necessary.

**Next decision (frozen choice):**

> **`develop the structured dictionary reader`**

Do not change the motif-discovery objective yet (there is no evidence that
frequency-selected primitives are the bottleneck), and do not re-open the object
(it remains exact/compressive/reusable/port-clean).

### Revisit if

A converged (≥ 20 k-step, matched raw) comparison shows the structured reader
stays ≥ 0.08 behind a fully converged raw reader while keeping the shared encoder
and exact decode — then the remaining gap would be a motif-discovery, not a reader,
problem. Also revisit the `R0` (corrected-opaque) claim if a seed-1 run shows the
batching defect's 0.12 contribution is not stable.
