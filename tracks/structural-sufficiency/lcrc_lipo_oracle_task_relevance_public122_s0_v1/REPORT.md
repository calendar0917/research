# REPORT — LCRC/Lipo oracle task relevance: Part A clarification + two computational tasks (public122, seed 0)

Round / protocol ID: `lcrc-lipo-oracle-task-relevance-public122-s0-v1`
Directory: `tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/`
Source round: `lcrc_lipo_multi_motif_mechanism_public122_v1` (protocol `0273cad6`, report `a5febe96`).
Design and all decision boundaries are frozen in `PROTOCOL.md` (written and committed before the first Part-B training).

**Question.** The previous round showed a training-related structural preference **only inside its
dose-matchable population** (amide). This round asks a different, answerable question: does the frozen
LCRC-C1 model's response to the *same* frozen rewire change with the supervised task, and does it track a
task that has a **computational reference value** — including a target that should *not* change? Part A is a
zero-GPU clarification of the old claims; Part B adds exactly two computational supervised tasks on the
same 4200 original molecules.

---

## 1. Answering the five required questions

**1. Whose strongest previous claim is covered.** The old "amide supported" verdict is a claim about the
dose-calipered sub-population — Test: 62 molecules / 110 focal pairs, i.e. 21 % of the 533-pair amide main
queue (`partA_clarifications.json::A1_coverage_and_claim_matrix`). It is a claim about `rC`/`E` in
dimensionless units, not raw-logD `J`. It is not a claim about all amide-bearing Test molecules, and it is
model-dependent evidence (one checkpoint, one initialisation).

**2. Does the full interaction change the path judgement?** Partly. The old text treated the standalone
scalar block as inert. From the frozen eight cells the **standalone** scalar recovery `R3 = value(123)-value(12)`
is indeed small on the calibrated Test population (amide/Test/RETAINED_C, H-space `R3 = +0.029`; |point| ≤ 0.11
for every family), while the **conditional S2 recovery** `R2 = value(123)-value(13)` is large
(amide H `+0.459`, sulfonamide `+0.549`, arylamine `+0.639`, aryl-ether `+0.272`, ester `+0.155`). The correct
phrasing is therefore **"the standalone scalar term is small, with non-zero interactions (S1×S2 etc.)"**, not
"scalar has no effect" (seven-term identity re-verified, max arithmetic residual `4.4e-16`).

**3. Does the model follow a structural change that has a reference value?** **Yes, for WC_LOGP.** On the
main counterfactual scope (`FULL`, 1810 pairs; the whole new molecule is re-encoded) the per-family mean
`|Δpred − Δref|` beats the fixed zero-change baseline `mean|Δref|` for every family with an effect
(Test: amide `gain +0.108 [0.094, 0.124]`, sulfonamide `+0.116 [0.074, 0.158]`, aryl-ether `+0.119
[0.094, 0.144]`, arylamine `+0.314 [0.258, 0.363]`, ester `+0.063 [−0.007, 0.138]`). Four of five are
Bonferroni-clear (6-item, `α=0.05`); ester is not. Sign agreement on non-zero `Δref` is 0.90–0.99, and the
relative-change sign (`q = |Δ_T|−|Δ_K0|`) agrees on 0.66–0.89 of non-zero cases. The model's own
new-graph MAE vs its oracle reference is 0.067–0.119 logP.

**4. Is a target that should not change stable?** **Approximately, but not exactly.** MolWt is exactly
composition-invariant under the frozen rewires (`Δref = 0` for all 1810 pairs, `molwt_invariant_pass = true`,
max |Δ| `0.0`). The model still moves: mean `|Δpred|` is 0.91–1.85 g/mol (0.008–0.017 of the Train target
std, ≤ 0.054·std at p95 for every family). This is small in target-scale terms, but **non-zero**; no
invariance is claimed and no equivalence is read from it. The MolWt "gain" (the zero-change baseline is
already optimal at 0) is negative by construction and is reported continuously instead.

**5. What evidence does experimental logD still lack?** The rewired molecules still have **no experimental
logD labels**; MM pairs remain low-support; the 539 C links are structure links, not molecules. This round
adds task-relevance evidence on two *computational* references and does **not** promote them to a chemical
mechanism for experimental logD.

---

## 2. Part A — zero-GPU clarifications (frozen source bytes only)

Source: committed catalogues and both per-graph prediction files of `lcrc_lipo_multi_motif_mechanism_public122_v1`
(hashes in `partA_clarifications.json`). No stored number of the old round is changed; the old REPORT gained a
non-destructive revision note only.

### A1 coverage and claim matrix
Per family/split: pre- and post-calibration molecules / clusters / instances / T pairs, C link count, C unique
view count, calibration retention and largest scaffold-cluster share. **539 C links across all families are
structure links over 364 unique control views — not 539 independent molecules**; all effects use the original
molecule→instance→control→donor weighting and the original scaffold-cluster bootstrap.

| family/test | pre-cal. pairs / mol | post-cal. pairs / mol | retention (pairs) | C links | C uniq. views |
|---|---|---|---|---|---|
| amide | 533 / 164 | 110 / 62 | 0.206 | 148 | 115 |
| sulfonamide | 112 / 30 | 29 / 13 | 0.259 | 58 | 33 |
| aryl_ether | 112 / 58 | 10 / 8 | 0.089 | 10 | 8 |
| arylamine | 41 / 17 | 4 / 4 | 0.098 | 5 | 5 |
| ester | 46 / 8 | 7 / 4 | 0.152 | 16 | 13 |
| sulfonate | 0 / 0 | 0 / 0 | — | 0 | 0 |

Claim matrix (`rC`/`E` dimensionless; `E = rC_TRAINED − rC_INIT > 0` only means preference *increased relative
to this initialisation*):

| family (Test, caliper) | `rC_TRAINED` | `E` | `rC_TRAINED>0` | Bonferroni-clear | claim |
|---|---|---|---|---|---|
| amide | `+0.164 [0.052, 0.278]` | `+0.321` | yes | `rC`, `E` yes | supported (population-limited) |
| sulfonamide | `+0.327 [−0.008, 0.617]` | `+0.596` | no (interval spans 0) | `E` yes, `rC` no | partial: preference increased, `rC>0` not established |
| aryl_ether | `−0.433 [−0.872, −0.158]` | `−0.235` | no | `rC` yes | weaker response than control, not "ignores structure" |
| arylamine | `+0.127 [0.106, 0.189]` (n=4) | `−0.378` | yes | `rC` yes, `E` no | `rC>0` but preference *fell* vs init; tiny n |
| ester | `−0.187 [−0.408, 0.065]` | `−0.229` | no | no | not established |
| sulfonate | — | — | — | — | untested (no C, no Valid/Test rewire) — never "zero/equivalent" |

### A2 same-population three tables
Same focal set throughout: `ALL_K0` = every original eligible T/K0 pair; `RETAINED_K0` = only focal T that have a
C, still vs the original K0; `RETAINED_C` = the identical focal T vs the matched C. Example (amide/Test, H-space):

| table | n pairs | `B` | `D` | `J` |
|---|---|---|---|---|
| `ALL_K0` | 533 | 0.280 | `D_L` 0.112 | 0.187 |
| `RETAINED_K0` | 110 | 0.492 | `D_L` 0.263 | 0.273 |
| `RETAINED_C` | 110 | 0.492 | `D_L` 0.173 | 0.146 |

`ALL→RETAINED` is a selected-population/edit change; `K0→C` is the same-population control change. They are
never merged into "dose calibration removed S1" or "the effect is necessarily produced by S2". The full
`J/D/B` and seven-term breakdowns for all families/splits are in `partA_clarifications.json::A2_same_population_three_tables`.

### A3 conditional scalar and S2 recovery
For raw prediction `F` and the `J`/`D` endpoints: `R3 = value(123)-value(12)` (conditional scalar),
`R2 = value(123)-value(13)` (conditional S2). Paired means, 95 % intervals and per-view absolute recovery are in
`A3_conditional_recovery`. On amide/Test/RETAINED_C: H-space `R3 = +0.029` (abs-per-view 0.063), `R2 = +0.459`
(abs-per-view 0.499); the interval for `R2_T_H` excludes zero. Per-view absolute recovery is reported to avoid
mean cancellation. Arithmetic residual ≤ `1e-5` (observed `4.4e-16`).

---

## 3. Part B — oracle references and two formal trainings

### 3.1 The computational reference is built and verified first
`code/oracle.py` (label-blind RDKit functions) over all 4200 originals and the frozen Valid/Test O/T/K0/C views:
identity SMILES→raw mapping for all 4200; `ClearComputedProps` + re-sanitise + `force=True` recompute; a fixed
first 20 non-O views independently re-parsed from their new canonical SMILES (`pass`, max W-C diff `8.9e-16`,
max MW diff `2.3e-13`); edit identity 3617 checked / 0 failed; `molwt_invariant_pass = true` (max |Δ| `0.0`);
WC_LOGP changes for 69.6 % of rewires (mean |Δ| 0.191, p95 0.553, max 1.060). No direct Crippen/WC_LOGP leak
into the global/local inputs (`direct_leak = false`). Frozen Train-only target norms: WC_LOGP `mean 3.2733,
scale 1.3238`; MolWt `mean 383.30, scale 108.29`. **7817 label rows, 1810 CF pairs, main_cf = 1810,
supplement = 0** (no pair has an O/T/K0 duplicate in the Train original canonical set). Test/Valid originals
exact Train duplicates = 0.

### 3.2 Two fixed trainings, shared everything but the target
`WC_LOGP_s0` (res-gpu0) and `MOLWT_s0` (res-gpu1), seed 0, same LCRC-C1 / `NO_BOND_MEAN` (228 674 params),
same training-start init (`init_state_hash aac1e061…`, `full_init_state_hash 666b88d7…`), same feature norm
fingerprint, **same batch order** (`b10278f9…`), same recipe (360 epochs × 27 steps, batch 128). The two target
norms differ as expected. No rewire label and no cross-task label entered training or selection
(`analysis_checks.json::shared_recipe`). Ordinary MSE, per-step cosine, chosen by each task's own Valid RMSE.

| task | best epoch | Test R² | Test RMSE (raw) | Train-mean baseline RMSE | gpu·h |
|---|---|---|---|---|---|
| WC_LOGP_s0 | 237 | 0.9964 | 0.0781 | 1.3065 | 0.088 |
| MOLWT_s0 | 43 | 0.9906 | 10.0822 | 103.973 | 0.096 |

Both original-graph models are strong, so the counterfactual behaviour cannot be blamed on basic prediction
insufficiency.

### 3.3 WC_LOGP counterfactual (Test, main CF, `FULL`)
`gain = mean|Δref| − mean|Δpred − Δref|` (positive = better than predicting no change); `q` = relative-change sign.

| family | n pairs / mol | baseline `mean|Δref|` | `delta_mae` | gain [95 %] | Bonf.-clear | sign agree | `q` sign | new-graph MAE |
|---|---|---|---|---|---|---|---|---|
| amide | 533 / 164 | 0.158 | 0.049 | **+0.108** [0.094, 0.124] | yes | 0.958 | 0.863 | 0.068 |
| aryl_ether | 112 / 58 | 0.185 | 0.065 | **+0.119** [0.094, 0.144] | yes | 0.952 | 0.892 | 0.079 |
| arylamine | 41 / 17 | 0.378 | 0.063 | **+0.314** [0.258, 0.363] | yes | 0.990 | 0.819 | 0.083 |
| sulfonamide | 112 / 30 | 0.178 | 0.063 | **+0.116** [0.074, 0.158] | yes | 0.978 | 0.867 | 0.067 |
| ester | 46 / 8 | 0.136 | 0.073 | +0.063 [−0.007, 0.138] | no | 0.896 | 0.656 | 0.119 |
| sulfonate | 0 | — | — | — | — | — | — | — |

The Valid split is the direction/scale check and matches (gains 0.104–0.328, all intervals > 0). Pooled over
Test: mean `|Δpred| = 0.246` vs mean `|Δref| = 0.241`, mean `|Δpred − Δref| = 0.072`. The C-matched secondary
scope (original matching, not re-searched) shows positive gains for all five families (amide +0.094, aryl-ether
+0.145, arylamine +0.240, ester +0.137, sulfonamide +0.089).

### 3.4 MOLWT counterfactual (Test, main CF, `FULL`) — the invariant control
`Δref = 0` for every pair. Continuous movement of the model, in g/mol and in Train-std units:

| family | n pairs | mean `|Δpred|` | median | p95 | max | mean / std |
|---|---|---|---|---|---|---|
| amide | 533 | 1.42 | 1.23 | 3.44 | 6.33 | 0.0131 |
| aryl_ether | 112 | 0.91 | 0.71 | 2.34 | 5.80 | 0.0084 |
| arylamine | 41 | 1.00 | 0.90 | 2.31 | 4.53 | 0.0092 |
| ester | 46 | 0.97 | 0.87 | 2.44 | 2.98 | 0.0090 |
| sulfonamide | 112 | 1.85 | 1.49 | 4.60 | 14.24 | 0.0171 |

Pooled Test mean `|Δpred| = 1.81` g/mol. This is small relative to the Train target std (108.3) and to the
original-graph Test error (≈ 8–10 g/mol) but is **not zero**; the model is *not* exactly invariant to the
composition-preserving rewiring. Sign/`q` agreement are `NA` because every reference change is zero.

---

## 4. Figures
`analysis/fig_partA_mechanism.png` (Part A same-population mechanism contrast), `analysis/fig_wc_logp_delta.png`
(WC_LOGP `Δpred` vs `Δref`, Test/main-CF), `analysis/fig_task_response.png` (both tasks' scale-normalised
response on the identical CF set, with the MolWt reference band). Headline numbers in
`analysis/summary_tables.json`; cross-task consistency in `analysis/analysis_checks.json`.

---

## 5. Provenance, budget, honesty

| stage | where | artefact | revision |
|---|---|---|---|
| A (Part A + oracle) | local CPU | `partA_clarifications.json`, `partA_recovery_pairs.csv`, oracle tables, `oracle_manifest.json`, `cf_isolation.csv`, `freeze_oracle.json` | `7bf4774f` |
| B (WC_LOGP_s0) | `rr` res-gpu0 | `runs/wc_logp/{best,last}.pt`, `train.json`, `curve.csv` | `7bf4774f` |
| B (MOLWT_s0) | `rr` res-gpu1 | `runs/molwt/{best,last}.pt`, `train.json`, `curve.csv` | `7bf4774f` |
| B (evaluate) | local CPU | `partB_{wc_logp,molwt}_{metrics,pairs,runtime}.*` | `7bf4774f` |
| C (stats, figures) | local CPU | `analysis_checks.json`, `summary_tables.json`, 3 figures | `7bf4774f` |

**Actual new training count: 2** (WC_LOGP_s0, MOLWT_s0, 360 epochs each). **GPU·h: 0.185** (planned ≤ 0.2,
soft 0.5, hard 1.0). Failures/retries (all implementation, no scientific re-tuning): (i) the first MOLWT launch
passed `--device cuda:1` but pool `res-gpu1` already narrows `CUDA_VISIBLE_DEVICES`, so it aborted at
`model.to` before any training step — relaunched with `cuda:0`; (ii) the remote `evaluate` stage failed because
the torch env (`res`) has no `rdkit` — evaluation is forward-only, so it was run locally on CPU;
(iii) `run.py`'s figures stage initially resolved to the source round's same-named module (fixed by making
`NEW_CODE` win `sys.path` and by loading the source `analysis.py` under a private name); the stray regeneration
was byte-identical to the committed source figures, so no source file changed. The `molwt`/`wc_logp` checkpoints
and `train.json` are the formal results; the failed first `molwt` launch produced no output.

The old PROTOCOL is unchanged; the old REPORT gained only a non-destructive revision note. Every number here is
reproducible from the committed frozen catalogues, the two checkpoints, and this round's code:
`uv run python …/code/run.py --stages partA,oracle` (Stage A), the two `rr run` trainings, then
`uv run python …/code/run.py --stages evaluate --task {wc_logp,molwt} --device cpu` and
`uv run python …/code/run.py --stages analysis,figures`. Tests: `uv run pytest -q …/tests` (14 passed).

**Interpretation.** The same graph edit's response changes with the supervised task and can be checked for
direction and magnitude against a computational reference with a real scale and interval: WC_LOGP beats the
no-change baseline on 4/5 families with Bonferroni support and correct sign 90–99 % of the time, while the
target that should not change moves only ~1–2 % of the target std. This supports **task relevance**, not a
universal chemistry mechanism for experimental logD, which still lacks rewire labels. Remaining uncertainty:
small-n families, one checkpoint/one initialisation, and the non-zero (if small) MolWt movement.

**Stop.** After this round: no third property, no seventh structure, no extra seed, no PNA, no large dataset.

---

## 6. Revision note — method formalisation and weighting integration (zero training, zero inference)

Appended after the round closed, as a **non-destructive** note. No stored number, protocol,
checkpoint or figure of §1–§5 is changed; nothing here re-trains, re-infers or re-intervenes.

**New material** (all in `analysis/synthesis/`, plus one table and two scripts):

| file | content |
|---|---|
| `analysis/synthesis/METHOD.md` | the C1 computation exactly as executed (incl. target de-standardisation), the verifiable local sufficient signature `S_v` (blocks A–F), the two derived properties with short proofs, the eight-cell intervention, dose control, parameter checks, task references and their differences |
| `analysis/synthesis/CLAIMS_EVIDENCE.md` | the five fixed claim rows (claim / direct evidence / scope / not yet supported) + four generalisation boundaries, each linked to a concrete table, JSON field or code path |
| `analysis/synthesis/WEIGHTING_NOTE.md` | the weighting and denominator note (molecule vs pair/view, pooled split/scope/dedup, 1810 vs 844, 62/110 vs 164/533, the MolWt arithmetic identity) |
| `analysis/synthesis/MENTOR_BRIEF.md` | one-to-two-page Chinese brief: what changed in our understanding, what is closed, what cannot yet be claimed, what a stronger paper claim would need |
| `analysis/synthesis/PAPER_SECTION.md` | Chinese and English results paragraphs with the boundaries and counter-examples inside the text |
| `analysis/weighted_response.csv` + `.json` | the additional molecule-weighted response table (see below) |
| `analysis/fig_method_flow.png` | method flow: real structure → explicit local statistics → frozen C1 forward → task references, with the traceability relation |
| `analysis/fig_evidence_layers.png` | three-layer evidence chain (real structure → real input signature → model computation) with the task references and the three fixed counter-examples |
| `code/weighted_response.py`, `code/method_figures.py` | the two new stages (`run.py --stages weighted,method_figures`) |

### 6.1 Weighting, stated instead of implied

`partB_*_metrics.json` carries two response statistics with **different weights**:
`delta_mae`/`gain`/`baseline_zero_change` use `analysis.molecule_means`
(molecule → instance → control → donor), while `abs_delta_pred.{mean,median,p95,max}`
are computed from a flattened array of the T and K views of every selected pair.
Both are legitimate; §1–§5 quoted the MolWt mean from the flattened array and the
WC_LOGP gain from the hierarchical one without saying so.

`analysis/weighted_response.csv` now states both side by side for every
task × split × family (plus one `POOLED` row per task/split). The original files are
untouched; the flat columns are recomputed and reproduce the committed ones exactly
(`flat_block_reproduced_max_abs_gap = 0.0`), as do `delta_mae`
(`delta_mae_max_abs_gap_vs_committed = 0.0`) and its bootstrap interval
(`delta_mae_ci95_reproduced_max_abs_gap = 0.0`, i.e. the same frozen draws were reused).

The MolWt arithmetic check passes in all 24 cells
(`molwt_identity_all_cells = true`, max gap `0.0`): with `Δref ≡ 0`, the hierarchical
delta-MAE **must** equal the molecule-weighted `mean|Δp|`. WC_LOGP does not satisfy it
(`wc_logp_identity_holds = false`), which is the expected direction.

Consequences for the text of §1–§5 (numbers themselves unchanged):

| §1–§5 statement | status after the check | molecule-weighted value |
|---|---|---|
| §3.3 "Pooled over Test: mean\|Δpred\| = 0.246 vs mean\|Δref\| = 0.241, mean\|Δpred−Δref\| = 0.072" | **T-view only, pair-weighted**, previously unlabelled | 0.1846 / 0.1723 / 0.0557 (T+K pair/view: 0.1914 / 0.1845 / 0.0560) |
| §3.4 "Pooled Test mean \|Δpred\| = 1.81 g/mol" | **T-view only, pair-weighted** | **1.3293 g/mol = 0.01228·σ_train** (σ_train = 108.294) |
| §3.4 per-family mean \|Δpred\| 0.91–1.85 g/mol | pair/view | 0.9045–1.9214 g/mol = 0.0084–0.0177·σ_train |
| §1 Q4 "≤ 0.054·std at p95 for every family" | **does not reproduce**: 0.0536 is aryl-ether's **max**/σ; the largest **p95**/σ is 0.0425 (sulfonamide) | molecule-weighted p95/σ ≤ 0.0269 |

The worst individual case stays in the statistics (not trimmed): `sulfonamide:1275:1:0:d9-14`,
T view, `|Δpred| = 14.2353 g/mol = 0.1315·σ_train`; the same family's molecule-weighted
mean is 1.9214 g/mol.

### 6.2 Run status, kept as it happened

The `rr` job records are **not** rewritten. The honest layering is:

1. **Training completed and checkpoints exist** — `runs/wc_logp/{best,last}.pt`,
   `runs/molwt/{best,last}.pt`, `train.json`, `curve.csv` (revision `7bf4774f`).
   Both `train.json` show the two target norms, 360 epochs, and the shared init/batch order.
   Naming the location precisely: this round's `.gitignore` excludes `runs/`, so the checkpoints
   and `train.json` live in the **local execution tree** and are not in Git. The committed
   evidence for the same training is `analysis/partB_{wc_logp,molwt}_metrics.json`
   (train/valid/test metrics recomputed in `evaluate`, with
   `original_graph_metrics.crosscheck_vs_train_json.pass = true`, i.e. the committed numbers
   reproduce `train.json` to ≤ 3.2e-6) plus `analysis/analysis_checks.json::shared_recipe`.
2. **The remote `evaluate` stage failed** on host `res` because that torch environment has no
   `rdkit`; evaluation is forward-only, so it was re-run locally on CPU (revision `7bf4774f`).
   The failed job status is preserved rather than converted to `success`.
3. **Local CPU evaluation completed** — `partB_{wc_logp,molwt}_{metrics,pairs,runtime}.json/csv`,
   with `device: "cpu"` recorded in `partB_*_runtime.json`.
4. The first `molwt` launch also aborted before any optimiser step (the pool `res-gpu1` already
   narrows `CUDA_VISIBLE_DEVICES`, so `--device cuda:1` was invalid); it produced no output and
   the affected part was re-run with `cuda:0`. This is an implementation error, not a
   scientific re-tuning.

**Corrected reproduction commands** (documentation only; nothing was launched for this note).
Split training and evaluation so the remote host never needs RDKit, and use the pool's logical
`cuda:0` on both cards:

```bash
# 1) remote, train only (one process per card; the pool already pins the card)
rr run res --pool res-gpu0 --experiment lipo-oracle-wclogp-s0 \
  --result tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/analysis \
  --result tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/runs \
  -- python tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/code/run.py \
     --stages train --task wc_logp --device cuda:0
rr run res --pool res-gpu1 --experiment lipo-oracle-molwt-s0 \
  --result tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/analysis \
  --result tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/runs \
  -- python tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/code/run.py \
     --stages train --task molwt --device cuda:0      # logical device inside the pool

# 2) local CPU: evaluate, statistics, figures, and this note's material
uv run python tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/code/run.py \
  --stages evaluate --task wc_logp --device cpu
uv run python tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/code/run.py \
  --stages evaluate --task molwt --device cpu
uv run python tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/code/run.py \
  --stages analysis,figures
uv run python tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/code/run.py \
  --stages weighted,method_figures      # zero training, zero inference
```

The `sys.path` ordering fix that makes this round's `analysis.py`/`figures.py` shadow the
source round's same-named modules is kept as committed in `round_common.py`; this note does not
refactor it.

### 6.3 What this note does not do

No new training, no new model inference, no new intervention, no new seed/property/group/dataset/
channel ablation/model comparison/probe/hyper-parameter search, **GPU = 0**, CPU ≈ 15 s
(two stages). No PR, no unrelated work, and the user's untracked
`lcrc_lipo_route_clarification_v1/code/lr_dispatch.py` was not touched.

After this note the round is still **stopped**: the five claim rows of
`analysis/synthesis/CLAIMS_EVIDENCE.md` are the maximum the current evidence supports, and every
gap listed there needs a new experiment round (not a re-run of this one).
