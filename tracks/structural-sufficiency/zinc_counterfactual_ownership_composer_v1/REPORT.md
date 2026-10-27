# COC-v1 — Counterfactual Ownership Composer

**Status: `STOP_NO_MATERIAL_GAIN`.**

The pre-registered Monitor stage failed. Locked1k was **not** read, no extra seeds
were run, and no architecture change was made. The mechanism itself is real,
trainable and non-degenerate; under the frozen Fit8k recipe it simply does not
improve cross-molecule generalisation.

Round: `zinc_counterfactual_ownership_composer_v1`  
Protocol: `ssrc-zinc-counterfactual-ownership-composer-v1`  
Formal training revision: `331fab569441`  
Monitor analysis revision: `86bec2fad3da`  
Date: 2026-10-27

---

## 1. Question and one-paragraph answer

**Question.** Keeping every existing DCC-L local structural path unchanged, does
explicitly learning the nonlinear difference of true ownership relative to the
molecule mean field,

```text
c_loc = local_composer(x_real) - local_composer(x_null),
```

with the **same shared** `local_composer` module, improve cross-molecule
generalisation under finite supervision labels?

**Answer.** No. The difference channel is sensitive to the real junction
correspondence, it receives gradients, and it remains a structured
within-molecule signal after training. But on the frozen Monitor1k, both seeds
are worse than their frozen DCC-L counterparts at every checkpoint semantics:
best-Monitor Δ = +0.006019 (s0) and +0.008475 (s1), seed mean **+0.007247**
(negative would be improvement), with paired bootstrap 95% CI
**[+0.002829, +0.012760]**. W20, T100 and final all agree. Every COC run is worse
than **both** frozen L runs. The supported failure category is
**learned-but-non-generalising**, not a degenerate representation and not an
unlearnable gradient path.

---

## 2. History check and novelty boundary

The history check was completed before any code was written and before any GPU
was purchased. The relevant prior results are:

| Prior round | What it established | Why it does not already answer COC-v1 |
|---|---|---|
| `zinc_support_conditioned_backoff_v1` (SCSB) | Input-space convex shrinkage `alpha*Wl + (1-alpha)*Wg` into one composer; primary Δ +0.003364; `STOP_UNSAFE`; 1.5585 A100·h | Different operator: it mixes two composers by a support-dependent coefficient. COC-v1 keeps both composers and subtracts the **same nonlinear function** evaluated at real and null ownership. |
| `zinc_dcc_information_audit_v1` | Global and local DCC information are non-redundant but not predictive | Information audit only; no counterfactual centering and no trained difference channel. |
| `zinc_relational_contrast_contraction_v0` (RCC-v0) | Graph-level relational contrast; 99% of the quadratic kernel reconstructible from graph-level statistics; `STOP_BEFORE_GPU_REDUNDANT_SECOND_ORDER_CHANNEL` | Different object and different granularity: graph-level second-order contrast, not a per-bond shared `f(x_real)-f(x_null)` inside the production DualChannel. |
| `zinc_dccl_frozen_transfer_audit_v1` | On frozen representations, a fixed 33k reader on 3,000 labels recovers ~77% of the Fit4k→Fit8k gain; extractable gain is concentrated in ownership-conditioned composer moments (`Σc_loc`, `Σc_loc²`) and global33 | Fixed-reader transfer audit, not a new architecture. It motivates testing ownership conditioning, but does not train the counterfactual difference. |
| `zinc_dual_channel_composer_v1` | Ownership-only DCC status C: no performance claim; single seed; test spent; L−C −0.003830; L worse than P3B +0.002313 | DCC-L is the frozen baseline. It does not test function-space centering of `c_loc`. |
| `zinc_block_ownership_v1` | Block ownership channel; `UNRESOLVED_DUE_TO_VARIANCE` | Different object (block channel), not the per-bond mean-field difference. |
| `zinc_frozen_bond_representation_audit_v1` | Frozen bond representation audit | Representation audit only; no trained counterfactual composer. |

**Novelty boundary.** No prior experiment was found that materially tests the
exact shared-parameter `f(x_real)-f(x_null)` per-bond counterfactual inside the
production DualChannel. This round does **not** claim proven originality, and it
does **not** claim that all centering or support-conditioning methods fail. It
closes only the exact operator and recipe below.

---

## 3. Frozen design

Production `DualChannel` is kept. The only forward change is:

```python
c_loc = local_composer(x_real) - local_composer(x_null)
```

with the **same shared** `local_composer` module and **no detach** on the null
branch. The reader input stays 513-dimensional:

| slice | content |
|---|---|
| `0:96` | `Σc_glob` |
| `96:192` | `Σa` |
| `192:288` | `Σa²` |
| `288:321` | `global33` |
| `321:417` | `Σc_loc` |
| `417:513` | `Σc_loc²` |

`group_composer`, the atom path and the reader architecture are unchanged. There
are no new trainable parameters, no scaling coefficient, no norm layer, no extra
loss, no attention, no support gate, and no new input features.

Parameter count from `named_parameters()`: **494,193** (verified in C1, C5 and C9):

| component | parameters |
|---|---:|
| encoders | 202,048 |
| group composer | 62,976 |
| local composer | 31,536 |
| reader | 197,633 |
| **total** | **494,193** |

Initialisation, SSL initialisation, Fit8k split, batch order, and the 47,400-update
recipe are exactly the frozen DCC-L recipe. At initialisation the new reader
columns `321:513` are zero, so the initial COC prediction is bitwise equivalent to
DCC-L; the local composer is trained through those previously zero columns.

---

## 4. Data, access barrier and compute

- Official PyG ZINC `subset=True`: 10,000 train / 1,000 valid / 1,000 test.
- Frozen internal split `split_a.json`, sha256 `53cd2cec29fb658bdbc789845df36b2194a8c8635140d61b4173ce064b48678e`:
  - Fit8k: 8,000 rows, the only supervised training data;
  - Monitor1k: 1,000 rows, checkpoint selection and pre-registered screening;
  - Locked1k: 1,000 rows, **not read** because Monitor failed.
- Official valid/test caches were never loaded, predicted or scored. The static
  scan and the `torch.load` audit report `official_valid_loaded: false`,
  `official_test_loaded: false`, `locked_labels_read: false`.
- Locked1k had been used in earlier rounds, so it would not have been a pristine
  blind test even if it had been read. It was not read here.
- The two historical frozen DCC-L checkpoints (`fit8k_s0`, `fit8k_s1`) were reused
  read-only. No L retraining and no L GPU cost are included.
- The Fit8k normalisation is Fit-only and matches the frozen L `train.json`
  values exactly.

Training cache provenance: local and remote copies of `ad_train.pt` have two
different torch serialisation byte hashes (`d85d9875…` and `e3a5e239…`), differing
only in `meta.prep_seconds` / `meta.created`. The canonical hash over the 47
non-meta tensors is `ce872e9b…` on both checkouts; the code asserts both the
accepted byte-hash set and the canonical tensor hash.

Execution used only the `rr` CLI on host `res` (`res-gpu0` / `res-gpu1`,
2×A100-SXM4-40GB, torch 2.5.1+cu124, CUDA 12.4). A profile run projected
**0.4801 A100·h** per formal run, below the 1.0 A100·h stop threshold. Actual
job occupancy is **0.7878 A100·h** including the one failed profile, below the
3.0 A100·h round cap.

---

## 5. CPU gates (all passed before any GPU)

`code/coc_checks.py` → `analysis/cpu_checks.json`; every gate C1–C9 passed.

| Gate | Result | Key evidence |
|---|---|---|
| C1 initial tensors bitwise identical to DCC-L | pass | s0 init `da48d59d…`, s1 init `4bb933a9…`; max tensor diff `0.0`; SSL hashes match frozen L |
| C2 initial predictions equivalent | pass | max prediction diff `0.0`; reader columns `321:513` max abs `0.0` |
| C3 forced `Wl=Wg, Vl=Vg` | pass | `c_loc` max abs `0.0`; `c_glob`, `z_glob`, `a_sum`, `a_sq`, `global33`, `Σc_loc`, `Σc_loc²` all diff `0.0` |
| C4 real junction correspondence changes `c_loc` | pass | junction-row permutation with fixed `w`: `c_loc` max diff **5.412943**; `c_glob` diff `0.0`; bond multiset preserved |
| C5 gradient paths | pass | step 1 `local_composer` grad norm `0.0` (cold start through zero reader columns), reader local-column grad norm `53.2437`, 49,152 nonzero entries; after one AdamW step local-composer grad norm **4.7250**; `Σc_loc`, `Σc_loc²`, `c_loc`, `c_abs`, `Wg/Vg`, `w` gradients all nonzero and finite; no `detach` in the `c_loc` source |
| C6 invariance | pass | atom relabel, bond permutation, endpoint swap, junction permutation and batch regrouping all ≤ `7.2e-7` (tolerance `1e-6`) |
| C7 no leakage / empty-bond molecule | pass | merged-batch vs alone prediction diff ≤ `1.2e-7`, `c_loc` diff `0.0`; empty-bond forward finite (`0.256810`) |
| C8 Fit8k difference diagnostics | pass | initial median bond RMS `c_loc` 0.3530 (s0) / 0.2845 (s1); nonzero fraction `1.0`; ratio to `c_abs` 0.766 / 0.808; within-molecule variance 0.547 / 0.438; participation ratio (within) 2.62 / 2.29 — diagnostics only, not performance evidence |
| C9 parameter/data/recipe/barrier | pass | 494,193 named and optimiser parameters; raw-y L1 manual = torch = 4.886339; 63 steps/epoch, 753 epochs, 47,400 updates, warmup 500, batch 128, dropout 0.0; split hashes match; first batch `2d65c92a…`; no official valid/test loads; no Locked labels |

The cold-start gradient at step 1 is expected: the reader's local columns are zero,
so the local composer initially receives no loss gradient. After the first
optimiser step those columns become nonzero and the composer gradient is
immediately nonzero. The mechanism is therefore trainable, not frozen.

---

## 6. GPU runs

Both formal runs used the committed recipe at `331fab569441`, 47,400 updates,
batch 128, AdamW, warmup 500, cosine `1e-3 → 1e-5`, weight decay `1e-5`, clip 5,
dropout 0, raw-y L1.

| run | pool | wall s | GPU h | best Monitor epoch | best Monitor MAE | final MAE |
|---|---|---:|---:|---:|---:|---:|
| `COC_fit8k_s0` | `res-gpu0` | 1302.8 | 0.3625 | 573 | 0.129170 | 0.129659 |
| `COC_fit8k_s1` | `res-gpu1` | 1482.0 | 0.4122 | 455 | 0.136246 | 0.137437 |

Including the successful profile (19 s) and the failed cache-hash profile (8 s,
no optimiser step), total round occupancy is **0.7878 A100·h** (cap 3.0). The
failed job is recorded in `analysis/run_ledger.json`; it is not hidden.

The formal runs consumed **0.7803 A100·h**, so the profile projection (0.4801
per run) was conservative.

---

## 7. Monitor results

Δ = MAE(COC) − MAE(L); **negative would be improvement**. Paired by molecule on
the same 1,000 Monitor rows.

| checkpoint | L s0 | COC s0 | Δ s0 | L s1 | COC s1 | Δ s1 | seed-mean Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| best Monitor | 0.123150880 | 0.129170336 | **+0.006019456** | 0.127771568 | 0.136246216 | **+0.008474647** | **+0.007247052** |
| W20 | 0.123819207 | 0.129665500 | +0.005846293 | 0.128420856 | 0.137375852 | +0.008954996 | +0.007400644 |
| T100 | 0.123834211 | 0.129683430 | +0.005849218 | 0.128481238 | 0.137453948 | +0.008972710 | +0.007410964 |
| final | 0.123826853 | 0.129658531 | +0.005831679 | 0.128513680 | 0.137436862 | +0.008923182 | +0.007377431 |

The best COC run (s0, 0.129170) is worse than the worse frozen L run (s1,
0.127772). No COC seed beats any L seed. W20 and T100 confirm that this is not a
single-checkpoint artefact.

Paired molecule bootstrap (10,000 resamples, seed 20261029):

| comparison | mean Δ | 95% CI |
|---|---:|---:|
| seed 0 best Monitor | +0.006019 | [+0.001171, +0.011493] |
| seed 1 best Monitor | +0.008475 | [+0.002935, +0.015070] |
| seed-mean best Monitor | +0.007247 | [+0.002829, +0.012760] |
| s1 − s0 cross-seed difference | +0.002455 | [−0.002836, +0.007799] |

The seed-mean CI excludes zero on the **worse** side. The cross-seed difference
CI includes zero, so there is no evidence that the two seeds disagree in sign;
both are positive.

### Group table (best-Monitor checkpoints)

| group | n | Δ s0 | Δ s1 | seed-mean Δ |
|---|---:|---:|---:|---:|
| overall | 1000 | +0.006019 | +0.008475 | +0.007247 |
| unseen_all | 317 | +0.006213 | +0.002523 | +0.004368 |
| low4_all | 359 | +0.005245 | +0.009194 | +0.007219 |
| full_all | 324 | +0.006689 | +0.013500 | +0.010094 |
| penalty_all | 36 | +0.074755 | +0.103620 | +0.089188 |
| **no-penalty any-unseen (`primary`)** | **301** | **+0.004586** | **−0.003224** | **+0.000681** |
| ordinary_seen | 663 | +0.002938 | +0.008620 | +0.005779 |
| tail | 74 | +0.029058 | +0.053962 | +0.041510 |
| no-penalty overall (`nopen`) | 964 | +0.003453 | +0.004922 | +0.004187 |

Only the seed-1 no-penalty any-unseen group shows a small negative Δ
(−0.003224), and it is not replicated in seed 0 (+0.004586); the seed mean is
+0.000681 and the no-penalty overall group is worse in both seeds. This is not a
material unseen-structure gain.

### Gate outcome

| gate | pass | reason |
|---|---|---|
| G1 both seeds overall Δ negative | false | both positive |
| G2 seed-mean overall ≤ −0.005 | false | +0.007247 |
| G3 no-penalty any-unseen both seeds negative, mean ≤ −0.008 | false | s0 positive; seed mean +0.000681 |
| G4 no-penalty overall improves | false | +0.003453 / +0.004922 |
| G5 W20/T100 support | false | both seed means +0.0074 |
| G6 no fitting/instability failure | false | see below |

**G6 nuance.** G6 fires because seed 1's selected best-Monitor checkpoint is early
(epoch 455 vs L epoch 552) and underfits Fit8k at that point: Fit MAE 0.015170 vs
L 0.007030, ratio 2.158 > the pre-registered cap 2.0. This is **not** a divergence:
all losses and gradients are finite, the Monitor tail100/best ratio is 1.009
(seed 1) and 1.004 (seed 0), and the final Fit ratios are 1.160 (s0) and 0.813
(s1). It reflects checkpoint-selection/underfit variance, not a crashed or
unstable run. Since every primary and secondary gate is also false, no Monitor
PASS is possible regardless; the negative conclusion does not depend on G6.

---

## 8. Was the difference representation alive after training?

Yes. On the same fixed 2,000-row Fit8k diagnostic subset (labels never used for
this diagnostic), the best-Monitor COC model still produces a structured
within-molecule difference:

| seed | median bond RMS `c_loc` | median bond RMS `c_abs` | ratio | nonzero `1e-8` | within-mol variance | PR (within) | common-energy fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.090432 | 0.237916 | 0.3707 | 1.000000 | 0.014247 | 8.966 | 0.1758 |
| 1 | 0.095737 | 0.186823 | 0.5025 | 1.000000 | 0.017387 | 8.167 | 0.1711 |

At initialisation the same diagnostic was larger (median RMS 0.353 / 0.284), so
training shrinks the difference but does not collapse it to zero or to a constant.
The difference remains molecule-dependent and bond-dependent. This rules out
"degenerate representation" as the cause of the negative result.

Monitor predictions changed for 100% of rows in both seeds (mean absolute
prediction change ≈ 0.050 / 0.051, max ≈ 1.58 / 2.19), so the null difference is
not a no-op at inference. It changes predictions substantially; those changes are
simply not better.

---

## 9. Answers A–D

### A. Is the local composer now sensitive to the ownership difference?

**Yes.** C4 changes the real junction correspondence with fixed `w` and moves
`c_loc` by up to 5.413 while leaving `c_glob` unchanged. C5 shows nonzero,
finite gradients through `Σc_loc`, `Σc_loc²`, `c_loc`, `c_abs`, the shared local
composer, and both `Wg/Vg` and `w` after the first optimiser step. The source line
has no `detach`. Post-training diagnostics remain non-degenerate. The composer is
not merely a mean-field copy.

### B. Does it improve cross-molecule generalisation under fixed data, not just training dynamics?

**No.** The two seeds have positive overall Δ at best-Monitor, W20, T100 and final.
The seed-mean is +0.007247 at best-Monitor, with a 95% CI excluding zero on the
worse side. The no-penalty overall group is worse in both seeds. The only negative
subgroup (seed-1 any-unseen) is not replicated. This is a held-out Monitor result,
not a training-loss statement.

### C. Is the gain at least the required ~0.006 versus the strongest DCC-L?

**No.** The requirement was Δ ≤ −0.006 (improvement of at least 0.006). The
observed seed-mean is **+0.007247**, i.e. worse by 0.007247, not better. Both
individual seeds are also worse (+0.006019, +0.008475), and every COC run is worse
than both L runs. There is no threshold-interpretation ambiguity: the sign is
opposite to the required direction.

### D. If failure, which causally supported category?

**Learned-but-non-generalising.** The evidence supports this category and rules out
the other two:

- **Not degenerate:** C8 and post-training diagnostics show nonzero, structured,
  within-molecule difference with participation ratio ~8–9 and nonzero fraction
  1.0; C3/C4/C5 show the representation is sensitive to correspondence and has a
  live gradient path.
- **Not unlearnable:** the only zero gradient is the expected step-1 cold start
  through zero reader columns; after one optimiser step the shared composer
  gradient is 4.725 (CPU C5) and 0.676 / 0.953 in the two formal runs. All 47,400
  updates completed with finite losses. Fit8k train MAE reaches 0.000506 / 0.000407
  at the final checkpoints.
- **Learned but non-generalising:** both seeds reach near-interpolation on Fit8k
  (final Fit MAE 0.000506 / 0.000407 vs L 0.000436 / 0.000500; seed 0 is 16% worse,
  seed 1 is 19% better) yet are worse on Monitor at every checkpoint semantics.
  The difference channel changes 100% of predictions but does not improve held-out
  error. We do not claim a more specific causal mechanism (e.g. "it fits
  molecule-specific noise") because no intervention isolating that mechanism was
  run; the causally supported statement is the category above.

---

## 10. What this closes, and what it does not

**Closed (for this exact hypothesis).** Under the frozen DCC-L Fit8k recipe,
47,400 updates, raw-y L1, two independent seeds, the shared-parameter nonlinear
difference `f(x_real)-f(x_null)` does not produce a material cross-molecule gain;
it is worse on Monitor1k and fails every Monitor gate. This is not a
representation-degeneracy or gradient-path failure.

**Not closed.** This round does **not** show that all centering, ownership
conditioning, support conditioning, or counterfactual objectives fail. It does not
test:
- input-space shrinkage (SCSB) or other mixing operators;
- explicit regularisation, capacity changes, or architectural changes to the
  difference channel;
- different label budgets, full10k training, or different losses;
- the official valid/test sets (never accessed);
- Locked1k (not read; and it would not have been pristine).

No `LOCKED_PASS_FORM_FULL10K_PROPOSAL` is made. The full10k follow-up thresholds
were not evaluated because the Monitor gate failed.

**Originality.** The history check found no prior exact test of this shared-parameter
per-bond counterfactual. That is a statement about the searched history, not a
proof of originality.

---

## 11. Reproducibility and deliverables

Artifacts under `tracks/structural-sufficiency/zinc_counterfactual_ownership_composer_v1/`:

| path | content |
|---|---|
| `PROTOCOL.md` | pre-registered hypothesis, gates, thresholds, stop rules |
| `DESIGN.md` | mathematical reading, shared-parameter rule, cold-start analysis |
| `configs/coc_v1.json` | frozen machine-readable config (single authority) |
| `code/coc_model.py` | `COCDualChannel`, shared composer, no detach |
| `code/coc_train.py` | frozen Fit8k trainer and profile |
| `code/coc_checks.py` | C1–C9 CPU gates |
| `code/coc_common.py` | data/split/norm/support/diagnostic helpers, load guard |
| `code/coc_analyze.py` | Monitor scoring, bootstrap, freeze |
| `code/coc_locked.py` | one-shot Locked scorer (not run; refuses without Monitor PASS) |
| `code/test_coc.py` | fast data-free unit tests |
| `analysis/cpu_checks.json` | all C1–C9 evidence and hashes |
| `analysis/profile_s0.json` | profile projection |
| `analysis/run_ledger.json` | rr job IDs, states, wall times, GPU hours |
| `analysis/monitor_metrics.json` | full Monitor metrics, curves, groups, diagnostics |
| `analysis/monitor_group_table.csv` | group MAEs and deltas |
| `analysis/monitor_predictions.npz` | frozen Monitor predictions (both seeds, both checkpoints) |
| `analysis/monitor_decision.json` | gate decision |
| `analysis/monitor_freeze.json` | hashes of config/protocol/code/checkpoints/predictions; `monitor_pass: false` |
| `analysis/locked_refusal.txt` | `coc_locked.py` refusal (exit 2) after Monitor stop; no Locked read |
| `analysis/research_verify.txt` | `uv run research verify` output |

Frozen monitor predictions include `pred_coc_s{0,1}_{best_monitor,final}`,
`pred_l_s{0,1}_{best_monitor,final}`, absolute errors, rows, labels for Monitor
only, and group masks. No Locked or official valid/test arrays are present.

Key commands:

```bash
uv run python tracks/structural-sufficiency/zinc_counterfactual_ownership_composer_v1/code/coc_checks.py
uv run python tracks/structural-sufficiency/zinc_counterfactual_ownership_composer_v1/code/coc_train.py --seed 0 --profile --device cuda:0
uv run python tracks/structural-sufficiency/zinc_counterfactual_ownership_composer_v1/code/coc_train.py --seed 0 --device cuda:0
uv run python tracks/structural-sufficiency/zinc_counterfactual_ownership_composer_v1/code/coc_train.py --seed 1 --device cuda:0
uv run python tracks/structural-sufficiency/zinc_counterfactual_ownership_composer_v1/code/coc_analyze.py
uv run research verify
```

`uv run research verify` result: **11 ok / 0 fail** (exit 0), including
`records.run_pointers_resolve`, `records.references_resolve` and
`records.path_references_exist`. The text output is saved at
`analysis/research_verify.txt`.

Run ledger summary: 4 rr jobs, **0.7878 A100·h** total job occupancy, **0.7803
A100·h** formal training, **0.0022 A100·h** failed profile, cap **3.0 A100·h**.
The failed profile is included, not excluded.
