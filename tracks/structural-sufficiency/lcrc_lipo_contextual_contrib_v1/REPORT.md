# REPORT — `lcrc-lipo-contextual-contrib-v1`

Single-seed exploratory comparison on the frozen random42 split
(Train3360 / Validation420, seed 0, raw logD). Pre-registered decision:
**`NO_CLEAR_GAIN`**, tail label **`NOT_TAIL_CONCENTRATED`**.

## 1. BASE vs CONTEXT (the primary table)

Checkpoint = average of the last 5 epoch-end state dicts; no early stopping,
no Validation-based checkpoint selection. Metrics are raw logD.

| arm | Train RMSE | Train MAE | Valid RMSE | Valid MAE | Valid R² | params | new params | train wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **BASE** | 0.253621 | 0.190634 | **0.580416** | 0.427854 | 0.777097 | 247,106 | 0 | 312.0 s |
| **CONTEXT** | 0.214009 | 0.158566 | **0.568587** | 0.411573 | 0.786090 | 255,458 | +8,352 | 423.8 s |

Paired difference (CONTEXT − BASE), Validation420:

| quantity | value |
| --- | ---: |
| RMSE gain `RMSE(BASE) − RMSE(CONTEXT)` | **+0.011829** |
| MAE change `MAE(CONTEXT) − MAE(BASE)` | **−0.016282** (improvement) |
| R² change | +0.008993 |
| total SSE `BASE` / `CONTEXT` | 141.490874 / 135.782516 |
| SSE difference `BASE − CONTEXT` | **+5.708358** (4.03 % of BASE SSE) |
| molecules with improved absolute error | **213 / 420** (50.71 %) |
| molecules with worse absolute error / ties | 207 / 0 |
| mean / max absolute prediction difference | 0.203474 / 1.190982 |

Historical reference: the random42 v1 LCRC Validation RMSE was `0.589007`.
It is **descriptive only** and not a matched baseline: that historical training
loop configured dropout 0.10 but never executed it, whereas both arms here
executed it. No claim of improvement over that number is made.

## 2. Was the context path actually trained and used?

Yes. This is a mechanical statement, not a generalisation claim.

* **Initialisation.** CONTEXT and BASE share 35 state entries; all are
  bit-identical at seed 0 (gate 4). New parameter counts: 8,352
  (`3×(64×32+32) + 32×64+64`), total 255,458. Forcing `W_out = 0` reproduces
  same-weight BASE at max |Δ| = 0.0 (gate 5), while the untouched `W_out`
  changes predictions by 0.00117 at initialisation.
* **Optimisation.** All eight new tensors received finite, non-zero gradients
  at the first step and at every recorded probe step. Over all 9,720 training
  steps the minimum gradient norm per tensor stayed above zero
  (`context.W_atom.weight`: min 1.7e-4; `W_out.weight`: min 7.4e-3). The
  trained tensors moved from initialisation; max |Δ| ranges from 0.0413
  (`W_out.bias`) to 0.3640 (`W_out.weight`).
* **Forward usage.** On Validation420 the correction is non-zero on all 2,612
  hetero atoms in all 420 molecules. Mean correction magnitude per hetero atom
  is **2.0931 in L2** (max 7.9213), or **0.2357 mean absolute per component**;
  per molecule, the mean of the per-atom mean L2 is 2.1708 (median 1.9125,
  p95 4.0193, max 6.3826). Averaged over *all* atoms (including the zeros on
  carbon/halogen atoms), the mean absolute component is 0.05547.
* **Readout sensitivity.** `mean |f(h_new) − f(h_original)|` over hetero atoms
  is **0.12216** (max 0.78691); over all atoms it is 0.02875, and the mean
  per-molecule absolute sum is 0.75973.
* **Ablation.** Zeroing the trained `W_out` in eval changes Validation
  predictions by 0.79920 on average (max 4.28459); its RMSE moves from 0.568587
  to 1.112137. This is the trained CONTEXT's own no-correction path, not BASE,
  because the shared weights adapted during training.
* **Dropout.** The frozen `v2_models.DropoutAtLocal` (`p = 0.10`) is attached
  to `model.local` in both arms and executed: in train mode it masks atoms
  (max dropout delta 3.87) and two differently-seeded train forwards differ; in
  eval mode its input/output are bitwise identical. The recorded hook count is
  10,462 (BASE) / 10,529 (CONTEXT) forwards.
* **`rho` provenance.** The captured `rho` input equals the reconstruction from
  the original `h` (max |Δ| 4.8e-7 float32 reduction-order noise) and differs
  from the `h_new` reconstruction by 0.32944, so the check is discriminating.

Roles are fixed element classes, as required — hetero `{N,O,S,P}` and
carbon_halogen `{C,F,Cl,Br,I}` — not HBA/HBD or hydrophobicity labels.
Train3360 has 21,656 hetero atoms (23.78 %) and 69,419 carbon/halogen atoms
(76.22 %); 3,359 molecules contain both roles, one contains only carbon/halogen,
none contains neither (the empty hetero-mean path is exercised by that one
molecule). Validation420 has 2,612 hetero atoms (23.53 %), 8,487 carbon/halogen
atoms (76.47 %), and all 420 molecules contain both roles.

## 3. Overall gain and the pre-registered decision

* `gain = 0.011829`, against a fixed threshold of `0.020` → **below threshold**.
* Validation MAE improved by `0.016282`, so the MAE-worsening guard
  (`> +0.010`) passes; the failure is on the primary RMSE threshold.
* 213/420 molecules improve, 207 worsen: a small net shift, not a broad
  rewrite of the error distribution. The mean absolute prediction change is
  0.2035, larger than the RMSE gain, so the path moves many predictions without
  a correspondingly large net error reduction.
* Decision (fixed before training): **`NO_CLEAR_GAIN`**. One seed, one split,
  no follow-up selection.

Training curves are ordinary but not monotone: BASE reaches its best epoch RMSE
0.564350 at epoch 181, CONTEXT 0.560576 at epoch 227; both final 5-epoch soups
are worse than those single-epoch minima (0.580416 / 0.568587). The
pre-registered soup was used regardless; no epoch was selected on Validation.

## 4. Tail audit (post hoc, descriptive only)

The five row IDs were fixed by `lcrc_lipo_random42_gap_audit_v1` before this
round; no new tail search was performed.

| row | y logD | BASE pred | CONTEXT pred | |err| BASE | |err| CONTEXT | |err| improvement (+ = better) | SSE improvement (+ = better) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1630 | 3.90 | 1.33601 | 1.35444 | 2.56399 | 2.54556 | +0.01843 | +0.09419 |
| 2924 | 3.08 | 0.96212 | 0.86180 | 2.11788 | 2.21820 | −0.10031 | −0.43497 |
| 822 | 3.44 | 2.11112 | 1.90887 | 1.32888 | 1.53113 | −0.20225 | −0.57845 |
| 1153 | 3.37 | 1.67161 | 1.88824 | 1.69839 | 1.48176 | +0.21662 | +0.68890 |
| 3634 | 1.81 | 3.38022 | 3.54836 | 1.57022 | 1.73836 | −0.16815 | −0.55633 |

* Net tail SSE improvement (BASE − CONTEXT) = **−0.786659** (CONTEXT is worse on
  the five rows in net; 2 improve, 3 worsen).
* Tail share of the net overall SSE improvement = **−0.1378**.
* RMSE without the five rows: BASE 0.545111, CONTEXT 0.530561,
  gain **+0.014550**.
* Fixed rule: net SSE improvement > 0 **and** (tail share ≥ 0.50 **or**
  gain-without-tail ≤ 0) would trigger `KNOWN_TAIL_CONCENTRATED`. Neither
  condition holds → **`NOT_TAIL_CONCENTRATED`**. The small gain is not carried
  by the five known tail molecules; it is also not large enough to pass the
  primary threshold.

### Fixed subgroups (boundaries copied from the gap audit, not re-optimised)

| family | group | n | RMSE BASE | RMSE CONTEXT | gain |
| --- | --- | ---: | ---: | ---: | ---: |
| scaffold | seen | 209 | 0.440241 | 0.425052 | +0.015189 |
| scaffold | unseen | 211 | 0.691808 | 0.681588 | +0.010219 |
| heavy atoms | Q1 (7–21) | 114 | 0.547705 | 0.542413 | +0.005292 |
| heavy atoms | Q2 | 114 | 0.598152 | 0.547757 | +0.050395 |
| heavy atoms | Q3 | 87 | 0.558781 | 0.544464 | +0.014317 |
| heavy atoms | Q4 (31.25–49) | 105 | 0.612107 | 0.634318 | −0.022211 |
| TPSA | Q1 (3.24–56.85) | 105 | 0.459397 | 0.466561 | −0.007164 |
| TPSA | Q2 | 106 | 0.636490 | 0.558855 | +0.077634 |
| TPSA | Q3 | 105 | 0.592165 | 0.595485 | −0.003321 |
| TPSA | Q4 (99.36–164.55) | 104 | 0.616825 | 0.639915 | −0.023090 |
| MolLogP | Q1 (−1.97–2.37) | 105 | 0.558314 | 0.549869 | +0.008445 |
| MolLogP | Q2 | 105 | 0.487871 | 0.512259 | −0.024387 |
| MolLogP | Q3 | 105 | 0.581717 | 0.552381 | +0.029336 |
| MolLogP | Q4 (4.02–6.94) | 105 | 0.677793 | 0.650597 | +0.027196 |

The subgroup pattern is mixed: the largest positive entries (TPSA Q2, heavy-atom
Q2) sit next to negative entries (TPSA Q4, heavy-atom Q4, MolLogP Q2). With 420
Validation molecules these are descriptive slices, not independent tests.

## 5. What was run, gates, and GPU budget

**Data.** Frozen random42 split and Train/Validation labels, read-only and
hash-verified: split `1834414a…b4`, labels `1da26935…ca`; target scaler fitted
on Train3360 only (mean 2.2081011904761905, scale 1.1906281212426577); LCRC
feature norm fingerprint `4d6e5bf4…e26`. No Test label, prediction, metric or
per-molecule Test artifact was opened; the label accessor raises on `"test"`;
`train.json` records `test_labels_read / test_metrics_read /
test_predictions_read = false` for both arms.

**Recipe (identical arms).** AdamW, peak LR 3e-4, minimum LR 1e-5, per-step
cosine, weight decay 1e-3, batch 128, 360 epochs, 27 steps/epoch, 9,720 steps,
gradient clip 5.0, MSE on standardised Train targets, average of the last five
epoch-end state dicts, no early stopping. Both arms consumed the identical batch
order (`sha256 b10278f9…`). Dropout 0.10 was executed (see §2). fp32 throughout.

**Gates.** All 12 required CPU gates passed at training code `82d553a`
(`analysis/cpu_gates.json`, sha256 `5c1885c4…`; supplementary checks passed),
before any formal run. They cover split/label integrity and isolation, BASE ≡
original `LCRC`, bit-identical shared init, `W_out = 0` equivalence (max diff
0.0), role masks, atom renumbering (max diff 9.5e-7) and batch reordering
(0.0), no cross-molecule mixing (0.0), non-zero gradients on all four new
layers after two optimizer steps, `rho` reading original `h`, a CPU two-batch
optimizer smoke, and denormalisation/metrics (≤1e-12).

**Run records and charge** (full rr process lifetime, parallel pools summed):

| attempt | pool | state | process window (local) | charged | note |
| --- | --- | --- | --- | ---: | --- |
| `x1-cuda-smoke` | `res-gpu0` | failed | 21:40:02–21:40:10 | 8 s (0.0022 h) | audit batch stayed on CPU; fixed before formal runs |
| `x1-cuda-smoke-2` | `res-gpu0` | completed | 21:44:06–21:44:15 | 9 s (0.0025 h) | at `a3975ff`; exposed eval-mode float32 audit assumption |
| `x1-cuda-smoke-3` | `res-gpu0` | completed | 21:52:00–21:52:11 | 11 s (0.0031 h) | at `07f13de`; both audits pass |
| `x1-base-s0` | `res-gpu0` | completed | 21:55:46–22:01:06 | 320 s (0.0889 h) | 9,720 steps, 4,985 clip events |
| `x1-context-s0` | `res-gpu1` | completed | 21:55:54–22:03:08 | 434 s (0.1206 h) | 9,720 steps, 4,478 clip events |

Total **0.2172 / 0.45 A100 GPU·h** (48.3 %), remaining 0.2328 h. Host `a100-2`,
2 × NVIDIA A100-SXM4-40GB (driver 550.163.01), PyTorch 2.5.1+cu124, CUDA 12.4,
fp32; base peak 106.7 MB, context peak 106.9 MB allocated. Formal runs were
launched at commit `b7f520e`, deployed and validated by `rr`; training code is
`82d553a` (the same code hash frozen in the protocol).

One provenance caveat, reported rather than patched: both formal envelopes
record `git_dirty_tracked: true`. The cause is mechanical — the budget guard
calls `build_ledger()`, which rewrites the tracked `analysis/gpu_ledger.json`
in place before the model starts. After both runs, `git status` on the remote
checkout lists exactly that one analysis artifact; no code, config, data,
protocol or test file differs from `b7f520e`. `rr` had validated the deployment
marker for that commit, and the run records contain the frozen config hash,
protocol-freeze hash and data hashes.

## 6. Limitations

1. **Single seed, single split, Validation only.** The 0.0118 RMSE gain is
   below the pre-registered 0.020 and has no seed-variance estimate. This is an
   exploratory result, not a generalisation claim.
2. **Mixed subgroup signs.** The gain is positive for scaffold-seen and
   scaffold-unseen, but negative for the largest-molecule quartile and TPSA Q4,
   and mixed on MolLogP. It is not a uniform improvement.
3. **The correction is large and entangled.** The trained path changes `h` by
   L2 ≈ 2.09 on hetero atoms and zeroing `W_out` collapses Validation RMSE to
   1.112. The small net gain therefore reflects a re-parameterised shared
   network, not an isolated, interpretable chemical effect. The ablation does
   not identify which role (hetero mean vs carbon/halogen mean) caused it.
4. **No early stopping / fixed soup.** Validation minima occur at epochs 181
   (BASE) and 227 (CONTEXT), while the required last-5 soup is worse
   (0.580416 / 0.568587). A best-epoch report would give a much smaller gain
   (~0.0038); that number is not used for the decision.
5. **Frequent clipping.** 4,985 / 9,720 BASE steps and 4,478 / 9,720 CONTEXT
   steps hit the fixed gradient-clip threshold; the largest pre-clip norm was
   ~2,189. The recipe was not tuned or changed in response.
6. **Role definitions.** `hetero` and `carbon_halogen` are fixed element
   classes, not HBA/HBD or physical hydrophobicity; only hetero atoms receive
   the additive correction, and carbon/halogen atoms act only through the gate.
7. **Historical reference is unmatched.** The v1 random42 LCRC number 0.589007
   came from a loop that never executed its configured dropout. No matched
   historical baseline exists for this exact recipe.
8. **Post-hoc diagnostic repair.** The training-time helper stored signed
   `f(h_new) − f(h_original)` under `*_abs_*` field names. After the frozen runs
   it was corrected and the descriptive block was recomputed from the frozen
   CONTEXT checkpoint (`38bf6926…`) on CPU
   (`analysis/context_diagnostics_recomputed.json`). Original values were
   `f_abs_mean_hetero = −0.102978` and `f_abs_max_hetero = 0.386627`; corrected
   values are 0.122162 and 0.786909. Weights, predictions, metrics and the
   decision were not touched. The original `train.json` is retained unchanged.
9. **Pre-run implementation fixes.** The first CUDA smoke failed because audit
   batches were not moved to the model device; a later smoke exposed an
   over-strict eval-mode equality assumption in the dropout audit. Both were
   fixed and the protocol was re-frozen at `82d553a` before any formal training.
   These fixes changed only audit helpers, not the model definition or recipe.

## 7. Artefacts and commits

| item | path |
| --- | --- |
| protocol / freeze | `PROTOCOL.md`, `analysis/protocol_freeze.json`, `analysis/protocol_sha256.json` |
| data audit / 12 gates | `analysis/data_audit.json`, `analysis/cpu_gates.json` |
| CPU smoke | `analysis/smoke_cpu/` |
| CUDA smokes | `analysis/smoke_cuda/` |
| formal runs (checkpoints, curves, predictions, diagnostics) | `analysis/runs/base_s0/`, `analysis/runs/context_s0/` |
| rr status + GPU ledger | `analysis/rr_status/*.json`, `analysis/gpu_ledger.json` |
| validation metrics / comparison | `analysis/metrics_validation.json`, `analysis/comparison.json` |
| per-molecule validation predictions | `analysis/per_molecule_valid.csv` |
| tail / subgroups / context usage | `analysis/tail_audit.json`, `analysis/subgroups.json`, `analysis/context_usage.json` |
| post-hoc diagnostic repair | `analysis/context_diagnostics_recomputed.json`, `code/x1_posthoc_diagnostics.py` |
| decision | `DECISION.md`, `analysis/decision.json` |

Commits: protocol/model/gates frozen at `82d553a`; formal deployment and runs at
`b7f520e`; validation analysis and the post-hoc diagnostic repair at `5ec3260`.
This report and `DECISION.md` are committed together; see
`git log -1 -- REPORT.md` for the exact report revision.

No follow-up experiment, no structure/LR change, no validation-based checkpoint
selection and no new model proposal were made after the formal runs.
