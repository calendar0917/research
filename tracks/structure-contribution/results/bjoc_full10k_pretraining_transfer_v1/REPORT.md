# cscl-bjoc-full10k-pretraining-transfer-v1 — report

**Decision: `FULL10K_PRETRAINING_GAIN_CONFIRMED_ON_DEV`** (frozen rule
`configs/bjoc_full10k_pretraining_transfer_v1.yaml`; branch written by the driver, not by hand).

Flags: **`BULK_GAIN_WITH_TAIL_COST`**, **`ENCODER_LR_ADAPTATION_NOT_SUPPORTED`**.

One-sentence conclusion: *on the official 10,000-molecule BJOC train split, with no internal
split and a single official-valid read, masked atom/bond pretraining of the three existing
encoders (7,560 updates, structure only) followed by the identical full supervised recipe
improves official-valid raw-y single-model MAE by `P − B = −0.015653` (seed-mean, CI95
`[-0.021051, -0.009904]`, both seeds negative, ordinary molecules `-0.017391`, top-10-error-removed
`-0.015980`, `valid:0172`-removed `-0.016400`) — a 15.8 % relative reduction over the in-round,
same-GPU scratch baseline `B = 0.099202` → `P = 0.083549` — while reduced-LR encoder preservation
(Q) is decisively harmful (`Q − P = +0.019484`, tail blow-up) and the remaining error is
concentrated in a persistent extreme-tail subgroup (`tail y≤-3` MAE `0.2465` vs ordinary `0.0722`).*

Historical context (mandatory, not a comparison): the earlier BJOC-Null full-10k **official-test**
seed-mean raw-y MAE is `0.092233` (`0.089903` for seed 1) against the user target band `0.06–0.07`.
This round ran on the official **valid** split; its MAE levels are a different split and a different
exposure and are **not comparable** to official-test numbers. No extrapolation is claimed to the
official test, which was never loaded.

## 1. Protocol, provenance, budget, discipline

Frozen before any GPU in commit `39f06e7` (`configs/bjoc_full10k_pretraining_transfer_v1.yaml`,
preregistration note `notes/bjoc_full10k_pretraining_transfer_v1.md`), amended once in commit
`9ef5588` **before any GPU training** (see deviation below); all training and prediction artifacts
carry `git_commit = 9ef55888499e`.

* Data: official train 10,000 (raw y) only for supervised training; official valid 1,000 is the
  frozen dev read; official test never loaded anywhere (`official_valid_loaded: true` only inside
  the scoring payload; `official_test_loaded: false` in every artifact). No internal split: the
  same 10,000 structures are used for SSL (structures only) and for supervised training.
* Frozen structure caches (`objects` / `prep` command, Phase A, re-run identically on the remote
  A100 node): `objects_train10k.pt = 14326a7a…`, `objects_valid.pt = 61fca36a…`,
  `prep_train10k.json = d0fb86a3…`; all 10,000 remote-rebuilt files are byte-identical to the
  local freeze.
* Phase A audits all pass: consistency (WL fingerprint shared between train and valid:
  train row 337 vs valid row 376 — disclosed, no row deleted), mask-token slots free on all
  10,000 molecules, `MASK_SWAP_INVARIANT` (200/200 atom + 200/200 bond swaps),
  `MASKED_REBUILD_LEAK_FREE` (100/100), correctness gates 10/10.
* Pretext: 7,560 updates (96 visits à 79 steps), batch 128, 15 % atoms + 15 % bonds, mask tokens
  atom `27` / bond `0`, `L = CE_atoms + CE_bonds`, Adam 1e-3 / wd 1e-5 / clip 5, last-step encoder
  state; 96.768 mean exposures per molecule (vs ≈121 in the 8k round — recorded in the note).
* Downstream: 240 epochs × 79 = 18,960 updates, batch 128, Adam 1e-3 / wd 1e-5 / clip 5, last-5
  epoch parameter-mean soup; B and P share the seed's full-model init and batch order for the
  whole run (verified by `initialization_audit.json`: `same_init_full_model`,
  `same_composer_reader_init`, `same_batch_order` all true for both seeds).
* **Deviation (recorded, post-audit, pre-GPU):** the pre-registered B-reuse audit requires replaying
  the historical full-10k soup against the historical terminal-round valid predictions. That
  reference file is not part of the deployed revision on the compute checkout, so the audit
  conservatively returned `RETRAIN_B` on the cluster (the local checkout had verified replay
  bit-identically, `max|dpred| = 0.0`; supporting evidence only). B seed 0/1 were therefore trained
  **in-round, on the same GPU, with the identical frozen recipe** — a stronger matched baseline
  than a historical replay (it also removes any CPU/GPU regime concern). Cost: ≈19 GPU-min.
* Discipline: six prediction files (B0/B1/P0/P1/Q0/Q1) were written label-blind on CPU, then frozen
  before any label was touched (`prediction_freeze.json`, combined `8bd6d464f7c874fe`); the single
  official-valid y read followed (`valid_read.lock`, read `2026-10-12T18:49:23Z`, y hash
  `8f6ee636…`, raw-vs-encoded `max_abs = 0.0`). No training, selection or metric-favouring choice
  happened after the read.
* Budget: 8 remote GPU stages (B0 647 s, B1 467 s, SSL0 1779 s, SSL1 1666 s, P0 620 s, P1 499 s,
  Q0 661 s, Q1 470 s) = **6,809 s ≈ 1.89 GPU-h** (limit 3 GPU-h); total wall ≈ 2 h 20 min
  (limit 6 h 45 min); two GPU jobs at a time on 2×A100-40GB.

## 2. Primary result — B vs P vs Q on the one valid read

Raw-y official-valid metrics, single models, no ensembling (the headline is the seed-mean of two
independent single-model MAEs):

| arm | seed 0 MAE | seed 1 MAE | **seed-mean MAE** | median | p90 | RMSE | fit MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| B scratch | 0.100172 | 0.098232 | **0.099202** | 0.059102 | 0.195309 | 0.214761 | 0.045902 / 0.040928 |
| **P pretrained** | 0.084786 | 0.082313 | **0.083549** | 0.044493 | 0.166433 | 0.230007 | 0.025333 / 0.028498 |
| Q reduced encoder LR (3e-4) | 0.096611 | 0.109456 | **0.103034** | 0.051632 | 0.182879 | 0.540596 | 0.024353 / 0.025187 |

Paired molecule-level bootstrap (5,000 resamples; negative = candidate better):

| contrast | seed-mean Δ | CI95 | per-seed Δ | ordinary y>-3 (n=935) | tail y≤-3 (n=65) | top-10 removed | better-mol frac |
|---|---:|---|---:|---:|---:|---:|---:|
| **P − B** | **-0.015653** | [-0.021051, -0.009904] | -0.015387 / -0.015919 | -0.017391 | +0.009341 | -0.015980 | 0.621 |
| Q − P | +0.019484 | [+0.005177, +0.044905] | +0.011826 / +0.027143 | +0.008175 | +0.182161 | +0.008578 | 0.416 |
| Q − B | +0.003831 | [-0.012508, +0.032007] | -0.003561 / +0.011224 | -0.009215 | +0.191502 | -0.008652 | 0.571 |

Every frozen condition of `FULL10K_PRETRAINING_GAIN_CONFIRMED_ON_DEV` holds: seed-mean
`-0.015653 ≤ -0.005`; both seed deltas `< 0`; ordinary `-0.017391 < 0`; top-10-removed
`-0.015980 < 0`; `valid:0172`-removed `-0.016400 < 0`; bootstrap `ci_hi = -0.009904 < 0`; no
leak/consistency failure. The gain is **not** carried by the extreme molecule: P is *worse* on
`valid:0172` by `+0.681` and removing the row makes the improvement slightly larger
(`-0.016400`); that row's share of the total delta is `-4.4 %`.

The effect is not a fit artifact: P fits the train set *more tightly* than B (`fit_minus_valid`
B `-0.0543 / -0.0573` vs P `-0.0595 / -0.0538`) and still generalizes better; the P seed spread is
also smaller (0.0848/0.0823 vs 0.1002/0.0982).

## 3. Mechanism — Phase D diagnostics and the conditional Q purchase

All Phase D numbers below were computed **label-blind** (2,000 fixed train structures) before the
valid read; CKA is linear CKA between the SSL and P-fine-tuned encoder representations.

| encoder | param move init→SSL | param move SSL→P | rank init→SSL→P | scale init→SSL→P | CKA SSL↔P (s0 / s1) |
|---|---:|---:|---|---|---:|
| atom | 1.844 / 1.840 | 0.732 / 0.762 | 9.33→12.85→13.21 | 0.081→0.319→0.270 | 0.812 / 0.807 |
| bond | 2.193 / 2.121 | 0.885 / 0.908 | 6.29→18.59→28.51 | 0.044→0.486→0.508 | 0.558 / 0.577 |
| junction | 2.683 / 2.797 | 0.944 / 0.926 | 5.70→14.41→17.38 | 0.045→0.658→0.197 | **0.101 / 0.127** |

(Q values use seed 0 for rank/scale; seed 1 numbers are in `representation_diagnostics.json`.)

The frozen purchase gate fired on the CKA condition: junction linear CKA `0.1015 / 0.1268 < 0.75`
for both seeds. (The rank-drop condition did **not** fire: junction effective rank *rose*
14.4→17.4 and 15.6→14.4; 0/5 replication draws above the +25 % drop threshold.) Q was purchased
and trained with encoder LR `3e-4` vs composer/reader `1e-3`, same init/recipe otherwise.

Q is **decisively not supported**: it is worse than P on every aggregate (MAE +0.0195, RMSE
0.541 vs 0.230, tail +0.182), it is worse per seed, and its damage is concentrated in exactly the
extreme region the gate was meant to protect (`valid:0172` predicted at `-7.95 / -0.27` vs
`-13.33 / -17.89` for P and truth `-20.34`; Q's absolute errors `12.39 / 20.07`). Interpretation:
fine-tuning must substantially rewrite the encoders — especially the junction encoder — to fit the
raw-y scale structure; suppressing that adaptation (reduced LR) leaves the composer unable to
express large magnitudes, producing tail blow-ups. The pretrained *initialization* helps (P−B
confirmed); the pretrained *representation preservation* hurts. This is the frozen
`ENCODER_LR_ADAPTATION_NOT_SUPPORTED`.

## 4. Error decomposition — bulk vs tail (Phase G/H)

Paired subgroup decomposition of `P − B` (seed-mean MAE deltas):

| subgroup | n | P − B | Q − P |
|---|---:|---:|---:|
| ordinary y > -3 | 935 | **-0.017391** | +0.008175 |
| tail y ≤ -3 | 65 | **+0.009341** | +0.182161 |
| junction-rich | 488 | -0.012370 | +0.029512 |
| atom-rich | 469 | -0.015460 | +0.029810 |
| large-diameter | 343 | -0.018421 | +0.005706 |
| ring-containing | 999 | -0.015649 | +0.019325 |

Median/p90 improve in the bulk while RMSE and signed tail error worsen: P's tail MAE is `0.2465`
vs B's `0.2371`, and the mean *signed* tail error grows from `+0.1172` to `+0.1684` (predictions
remain systematically less negative than the truth on extreme molecules). Phase H common-error
analysis: of B's 100 worst rows, **39 are fixed by pretraining and 61 persist** (61 %). The
persistent rows are the large-magnitude ones (e.g. `valid:0172`, `917`, `753`, `214`, `29`, …;
full list in `error_analysis.json`).

Quantified headroom: the tail is 6.5 % of molecules but contributes `0.065 × 0.2465 ≈ 0.0160` of
P's `0.0835` MAE. Bringing the tail to the ordinary rate (`0.0722`) would give
`0.0722 × (0.935 + 0.065) ≈ 0.0722` — i.e. the diagnosed tail alone spans most of the distance to
the `0.06–0.07` band. So the binding constraint is a *composition-scale* failure on extremes, not
bulk representation quality, and it is systematic across seeds rather than noise.

## 5. Recommendation — exactly one next route, non-claims, artifacts

**Recommended next route (one): a new full-molecule composition mechanism — count-conditioned
whole-molecule scale calibration (`bjoc_scale_composition_v1`), tested against the frozen P
recipe.** Rationale: (i) the sample-efficiency route is already saturated at this data scale
(96.8 exposures/molecule, bond CE `0.016–0.019`, atom CE `0.213`, plateau by ~visit 80; the 8k→10k
change already bought the gain by data diversity, not exposure), and more of the same objective
trades bulk gain for tail cost (`+0.0093`) instead of closing the gap; (ii) the residual errors
are systematically signed magnitude under-predictions on a *small, persistent, scale-defined*
subgroup (61/100 common worst rows), where the idealized fix is worth ≈0.011–0.016 MAE — enough,
with the confirmed bulk gains, to put `0.06–0.07` in range; (iii) the Q experiment shows the fix
cannot come from regularizing/shielding the encoders but from how the molecule is composed. The
concrete proposal must be non-iterative, no GNN/attention/message passing, no new
junction/3-/4-body modules, no new local featurization and no "just more parameters": keep the
existing α/δ/γ object contributions and the P initialization (full-LR fine-tune), and add a tiny
(≤ ~2 k params) whole-molecule term `ŷ = a(c)·ŷ_obj + b(c)` driven only by label-free structural
counts `c` (atom-type counts, physical-bond counts, junction counts, ring counts, n_atoms), with
the calibration term reported separately so contribution attribution stays exact and auditable.
Next-round pre-registered gates should include: tail MAE and RMSE improve, ordinary MAE
non-worse, per-seed consistency, paired bootstrap CI, and an out-of-range count extrapolation
sanity check. (Route A is explicitly rejected as the next step for the reasons above; external
data is outside this protocol.)

Non-claims / limitations: this is a single valid read of one architecture family on one split
(n=1,000, seed-mean of two models); no official-test claim; no statement about unpublished target
band achievement; Q's failure is specific to encoder-LR 3e-4 with otherwise unchanged recipe, not
evidence against all encoder regularisation; the tail headroom estimate is an idealized
decomposition, not a promise.

Artifact index (`tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/`):
`configs/bjoc_full10k_pretraining_transfer_v1.yaml` + note (preregistration);
Phase A: `objects_manifest.json`, `consistency_audit.json`, `correctness_gates.json`,
`masked_feature_dependency_audit.json`, `mask_target_swap_audit.json`,
`masked_object_rebuild_audit.json`, `reuse_audit.json`, `training_manifest.json`;
Phase B/C: `train_B_seed*.json`, `pretraining_seed*.json`, `pretrain_encoder_s*.pt`,
`train_P_seed*.json`, `soup_{B,P,Q}_seed*.pt`, `initialization_audit.json`;
Phase D/E: `representation_diagnostics.json`, `q_purchase_gate.json`, `train_Q_seed*.json`;
Phase F: `predictions/valid_*.npz`, `prediction_freeze.json`, `prediction_access_log.json`,
`valid_read.lock`, `valid_metrics.json`, `paired_bootstrap.json`;
Phase G/H: `error_analysis.json`, `scientific_decision.json`.
