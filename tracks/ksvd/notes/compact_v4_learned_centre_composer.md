# P1 — Compact-v4 Learned Relation-to-Centre Composer

> **Question.** Why must the already-computed learned pair relation `q_ij` be
> fixed-compressed into `[mean(q); std(q); log1p(count)]` before it enters the
> centre update? Can a small end-to-end learned permutation-invariant relation
> composer, operating before centre contextualization, extract task-useful
> composition structure that fixed coordinatewise moments cannot provide?
>
> **Verdict.** **Decision Case A — P1 MINIMAL LEARNED COMPOSER NO-GO** under the
> locked protocol. P1 seed0 reached official-valid MAE **0.147229** vs the
> reused optimized compact-v4 seed0 reference **0.146420**, i.e.
> `Delta_P1,0 = -0.000809` — far below the pre-registered `+0.004` architecture
> gate. The composer branch was genuinely alive and trained (residual-zero
> inference ablation shifts predictions by max|Δ| = 0.531; `phi`/`rho`
> gradients non-zero every epoch; the learned residual grew from 0.078 to 0.81).
> Per the pre-registered budget rule the matched capacity control and seed1 were
> **not purchased**. Official test was **never loaded**.
>
> Code: `tracks/ksvd/experiments/luyin16/zinc_compact_v4_learned_centre_composer.py`.
> Tests: `tracks/ksvd/tests/test_compact_v4_learned_centre_composer.py`
> (22 pass). Results: `tracks/ksvd/results/compact_v4_learned_centre_composer/`
> (+ `answers_q1_q20.json`).

---

## 1. Motivation

compact-v4 spends learned computation to construct the pair relation `q_ij`,
then immediately compresses it with a *hand-fixed* operator:
`[mean; population-std; log-count]`. The whole P1 question is whether that
compression is a bottleneck. This is the one architecture assumption under test
here; nothing else changes.

## 2. Why we moved away from CIN-driven design

The previous single representation-family run (`compact-v4-cell`) tried to
promote cycles to explicit persistent higher-order objects under the CIN-driven
Top-1 hypothesis and was a clean nested NO-GO at seed0
(`notes/compact_v4_cell_minimal_falsification.md`, `Delta_arch0 = -0.001042`).
That direction required a genuinely new object class and a larger budget
(`+13,456` params). P1 deliberately returns to a *small, local* architectural
dissection of compact-v4 itself, with no new object class, no higher-order
objects, no incidence, no message passing.

## 3. Internal compact-v4 bottleneck

The verified compact-v4 centre path is:

```
pairs (complete, unordered) -> q_ij = pair_encoder(...) in R^16   (computed once)
per (centre i, bucket b):
    a^fixed_ib = [ mean(q) 16 ; population_std(q) 16 ; log1p(count) 1 ]  in R^33
A_i = [a_i1 ; ... ; a_i5] in R^165
patch_i <- patch_i + center_update([patch_i ; A_i])      (zero-init residual)
```

Because `mean(phi(q)) != phi(mean(q))` for nonlinear `phi`, a learner inserted
*before* the mean can represent composition structure (which relation-level
nonlinear motifs are worth expanding) that the fixed moments cannot.

## 4. Why fixed statistics are being questioned

The fixed moments are a strong, well-chosen inductive bias, and they were
*deliberately retained* here (see §11). P1 asks the narrower question: holding
everything else fixed, does the pair encoder benefit from jointly adapting
*with* a learned composition operator that replaces the fixed compression? This
is different from past probes that added a statistic on top of a **frozen**
`q`/`h`.

## 5. Relation to historical frozen NO-GOs

Every earlier local route (endpoint association, centre covariance, triadic
binding, function-basis readers, optimized broad-state screen) added a
statistic / reader over already-frozen states and terminated NO-GO or
INCONCLUSIVE. P1 changes the **computation graph itself** and trains from
scratch: `q -> learnable composition -> h' -> whole backbone co-adapts`. The
pair encoder can therefore adapt to the new operator. That is a genuinely
different hypothesis, not another missing-statistic probe.

## 6. Architecture lock

`results/compact_v4_learned_centre_composer/architecture_lock.json` was written
**before** training and not modified afterwards. It fixes: `q_dim=16`, 5
buckets, the fixed-summary definition, `phi` and `rho` shapes, the activation,
the empty-bucket rule, residual-vs-replace, the zero-init policy, the sharing
policy, `R=302`, the parameter count, the training protocol, the architecture
and mechanism gates, and the seed policy. No field was changed after valid was
seen.

## 7. Existing fixed centre aggregation

From the real code (`zinc_patch_path_pooling.PatchPathModel._pool_pairs_to_centres`,
source SHA-256 `7805fba2…`): each unordered pair contributes identically to
**both** endpoint centres; per `(centre, bucket)` the model keeps
`[mean(q) 16 ; population_std(q) 16 ; log1p(count) 1] = 33D`; `std` uses
`sqrt(clamp(mean(q^2) - mean(q)^2, 0) + 1e-8)`; an empty bucket is exactly `0`.
Five buckets concatenate to **165D**, and the centre update consumes
`[patch(48) ; context(165)] = 213D`. P1 reuses this implementation verbatim
(`super()._pool_pairs_to_centres(...)`) and only *adds* a residual.

## 8. Learned relation transform `phi`

Locked, shared across every graph/centre/bucket, exactly
`Linear(16,24) -> ReLU -> Linear(24,24)` (1008 params). No attention, no
normalization, no dropout, no gating, no bucket-specific parameters. `phi` sees
**only** `q`; not `h_i`, not endpoints, not graph embedding, not the raw 23D
relation descriptor. The main activation of the compact-v4 pair/centre MLP is
ReLU; the same activation is inherited mechanically.

## 9. Invariant composition

For centre `i`, bucket `b` with `n_ib` incident pairs and multiset `Q_ib`:

```
z_ib  = (1/n_ib) sum_{q in Q_ib} phi(q)                in R^24      (n_ib > 0)
u_ib  = [ z_ib ; log1p(n_ib) ]                          in R^25
```

The mean is permutation invariant: `index_add_` makes the result independent of
relation ordering (integrity gate G0.10 measures max diff `4.2e-7` for the
active composer, float32 `index_add_` rounding only).

## 10. Residual composer `rho`

Locked, shared, exactly `Linear(25,33) -> ReLU -> Linear(33,33)` (1980 params).
The final `Linear` is **zero-initialised**, so `r_ib = 0` at step 0 and the
model is bit-identical to the baseline at initialisation (G0.1 max diff `0.0`).
No LayerNorm, dropout, residual stack or extra depth.

## 11. Why baseline moments are retained

The candidate is a **residual** path:

```
a^P1_ib = a^fixed_ib + r_ib
```

If we had *replaced* mean/std/count, a failure could be trivially attributed to
deleting a proven-useful stable summary. Keeping the fixed summary as a residual
baseline makes failure informative.

## 12. Parameter budget

From the instantiated model (`parameter_audit.json`, not an estimate):

| component | params |
|---|---:|
| optimized compact-v4 baseline | 99,613 |
| `phi` (`16->24->24`) | 1,008 |
| `rho` (`25->33->33`) | 1,980 |
| **P1 total** | **102,601** |
| **P1 added** | **2,988** |
| capacity control (`psi` `33->44->33`) | 2,981 |
| P1 vs control added mismatch | **7** (0.23%) |

P1 is `102,601 <= 103,000` (hard cap). No width was chosen by performance.

## 13. Initialization matching

`initialization_match.json`. Under a fixed seed 0, building P1 and the capacity
control from scratch reproduces **every** shared baseline tensor exactly:
48 shared tensors, `max_abs_diff = 0.0`, `explicit_copy_applied = false`,
identical `shared_state_sha256 = d2650caf…`. This is from-scratch
initialization matching (the trained V4 checkpoint is **not** loaded as a warm
start). The fitted vocabulary widths (6785 typed / 32 parent) were re-verified
against the constant V4 references.

## 14. Integrity gates

`integrity_gates.json`, all pass (`official_test_loaded=false`):

`G0.1` zero-init P1 == baseline (`0.0`); residual disabled == baseline (`0.0`);
`G0.2` fixed pooling unchanged (source SHA matches baseline inventory);
`G0.3` q dim 16; `G0.4` fixed bucket 33; `G0.5` 5 buckets 165; `G0.6` R 302;
`G0.7` graph head unchanged; `G0.8` topology branch unchanged; `G0.9`
102,601 <= 103,000; `G0.10` relation-order permutation invariance (`4.2e-7`) and
composer demonstrably active; `G0.11` empty-bucket residual exactly `0` (with
active non-empty buckets), control empty-bucket exactly `0`;
`G0.12` shared init matches seed0 baseline; `G0.13` exactly one `q`
computation stage (no relation refresh).

## 15. Stage 1 — P1 seed0

One run, frozen protocol (Adam lr 1e-3, wd 1e-5, batch 128, max 240, patience
40, no scheduler, L1, best official-valid). No search, no HPO.

| run | valid MAE | best epoch | epochs run | params |
|---|---:|---:|---:|---:|
| optimized compact-v4 seed0 (reference, reused) | **0.146420** | 169 | 209 | 99,613 |
| **P1 seed0** | **0.147229** | 151 | 191 | 102,601 |

Curve: `curves/p1_seed0_curve.csv` (train MAE, valid MAE, lr, optimizer steps,
`checkpoint_selected`, `phi_grad_norm`, `rho_grad_norm`, `composer_grad_norm`,
`learned_residual_norm`, `fixed_summary_norm`, `residual_fixed_ratio`,
`nonempty_bucket_rate`). 15,089 optimizer steps, 1,300 s wall.

## 16. Architecture gate

`Delta_P1,0 = 0.146420 - 0.147229 = -0.000809`. Required `>= +0.004`. **FAIL**,
by a wide margin. No optimization pathology: no NaN; best epoch 151/240 (not
boundary-pinned); early stop fired normally; training loss decreasing; `phi`
gradient max `6.0e-3` and `rho` gradient max `0.177` (both non-zero from epoch
1); the composer branch is alive (inference residual-zero ablation shifts
predictions by max|Δ| = 0.531). So this is a clean scientific NO-GO, **not**
optimization ambiguity. The common-input bulk is safe and, if anything, slightly
better under P1 (see §17).

## 17. Matched capacity-control rationale

P1 adds ~3K nonlinear parameters. A raw `Delta_P1,0` could not distinguish
"learned pre-pooling composition" from "~3K extra nonlinear capacity at the
centre-composition site". The pre-registered control (`P1-capacity-control`) is a
shared residual MLP `psi: 33->44->33` applied to the **already fixed** bucket
summary, never seeing individual `q` beyond the fixed moments, with the same
zero-init / empty-bucket / sharing conventions and a matched parameter budget
(2,981 vs 2,988 added; mismatch 0.23%). It tests whether access to the relation
multiset *before* pooling matters beyond extra capacity.

## 18. Stage 2 mechanism result if run

**Not run.** The Stage-1 architecture gate failed (`-0.000809 << +0.004`), so
per the locked budget rule the matched capacity-control run was **not
purchased**. No `stage2_*` artifact was created and no mechanism claim (in
either direction) is made. `Delta_prepool,0` is undefined by construction.

## 19. Seed1 conditional replication

**Not run.** Seed1 is conditional on *both* Stage-1 (`>= +0.004`) and Stage-2
(`>= +0.0025`) gates; neither was met. No `stage3_*` artifacts were created.
Seeds 2/3 were forbidden throughout.

## 20. Compute overhead

`compute_audit.json`: `phi` is applied once per pair relation. On a 128-molecule
valid batch (2,998 centres, 34,876 pairs; mean **272.5 pairs/molecule**), `phi`
is called 34,876 times; forward time rose from 19.79 ms (baseline) to 30.02 ms
(P1), i.e. **+51.7%** forward wall-clock, at only +3.0% parameters. Peak RSS for
the audit process ≈ 1.92 GB (1,965 MB). Training wall-clock 1,300 s; preprocessing
unchanged (reused the frozen extraction pipeline). This is an explicit reminder
that "same ~100K parameters" is **not** "same compute".

## 21. Method-philosophy compatibility

P1 keeps explicit patch objects and explicit relation objects, is permutation
invariant, uses a shared low-parameter operator, adds no generic propagation
rounds, no attention/Transformer, no higher-order objects, and computes `q`
exactly once. It is not a generic GNN message-passing stack, and it is naturally
composable with a future dictionary/reusable-composition-atom direction.

## 22. What is and is not proven

**Proven (seed0, locked protocol):**

* A minimal learned pre-pool relation composer (`phi` `16->24->24`, shared
  invariant mean, `rho` `25->33->33`) is implementable, permutation-invariant,
  deterministic and numerically healthy; at initialisation it is bit-identical
  to baseline; it is genuinely trained and used (`R=302` unchanged).
* Under the frozen optimized protocol it does **not** improve compact-v4 seed0
  valid MAE: `Delta_P1,0 = -0.000809` (gate `+0.004`).

**Not proven:**

* That learned relation composition is useless in general. This is a statement
  about *this minimal form* under *this protocol*, not a mathematical
  impossibility (the fixed mean/std is a strong inductive bias).
* Anything about the pre-pooling mechanism — the matched capacity control was
  not run because the architecture gate failed first.
* Any official-test number — test was never loaded.

**Descriptive diagnostics** (`composition_diagnostics.json`, official-valid,
selected checkpoint; descriptive only — no target-tail/error-quintile mining):
learned residual norm `0.637`, fixed summary norm `1.709`, ratio `0.373`;
non-empty bucket rate `0.9988`; per-bucket residual norms ≈ `0.64` uniformly
across all five buckets; `phi(q)` output variance `0.0028`. The learned residual
grew monotonically across training (0.078 at epoch 1 -> 0.81 at the final
epoch; 0.63 at the selected epoch-151 checkpoint) while valid MAE plateaued and
then overfit, so the branch became a large, well-used but *unhelpful* correction.

## 23. Final verdict

**Decision Case A — P1 MINIMAL LEARNED COMPOSER NO-GO.**
`Delta_P1,0 = -0.000809 < +0.004`, branch alive, no optimization ambiguity. The
matched capacity control and seed1 were **not purchased**; only **one** new full
training run was spent. `results/compact_v4_learned_centre_composer/final_decision.json`.

## 24. Next step: P2 only if justified

The clean NO-GO closes the *minimal* learned centre-composition hypothesis in
the form `q -> mean(phi(q)) -> rho -> 33D residual`. Per the pre-registered
boundaries, this failure must **not** be rescued in this experiment by attention,
larger widths, alternate pooling, bucket embeddings, centre conditioning, or
relation refresh. The next candidate may only be discussed as a *separate*
pre-registered study, e.g. **P2 — one-shot relation refresh**
(`h^(0) -> q^(0) -> h^(1) -> q^(1)`); P1's failure does **not** auto-trigger P2.
Do not stack P1 + refresh + attention + new cells at once.

---

## Q1–Q20

| # | answer |
|---|---|
| Q1 | `a^fixed_ib = [mean(q) 16 ; population_std(q) 16 ; log1p(count) 1] in R^33`; empty bucket exactly 0; 5 buckets concatenated = 165D. |
| Q2 | `q` dim = 16 (`pair_hidden`). |
| Q3 | 5 distance buckets (integer shortest-path 1,2,3,4,5+). |
| Q4 | `phi = Linear(16,24) -> ReLU -> Linear(24,24)`; shared across graphs/centres/buckets; no norm/attention/gating. |
| Q5 | `rho = Linear(25,33) -> ReLU -> Linear(33,33)`; shared across all buckets; final Linear zero-init. |
| Q6 | Residual: `a^P1 = a^fixed + r_ib`, so a failure cannot be blamed on deleting a proven summary. |
| Q7 | P1 total = 102,601 params. |
| Q8 | +2,988 params (`phi` 1,008 + `rho` 1,980); +3.0%. |
| Q9 | Yes — `R` remains 302D (centre-update input 213D unchanged). |
| Q10 | P1 seed0 best valid **0.147229**, best epoch **151** (191 epochs, early stop). |
| Q11 | `Delta_P1,0 = -0.000809`. |
| Q12 | No (gate `+0.004`). |
| Q13 | Yes — residual-zero max prediction shift 0.531, mean 0.037; `phi_grad` max 6.0e-3; `rho_grad` max 0.177; residual norm 0.078 -> 0.81. |
| Q14 | Yes (bulk gate ≤ +0.002): v4 0.09472 vs P1 0.09358, diff **-0.00114** (P1 slightly better on the easy bulk). |
| Q15 | No — the capacity control was not purchased (architecture gate failed). |
| Q16 | N/A (not run). |
| Q17 | Undefined (control not run). |
| Q18 | No — seed1 not authorised. |
| Q19 | N/A (not run). |
| Q20 | **Case A — P1 MINIMAL LEARNED COMPOSER NO-GO.** |
