# BJOC STRC predictive feasibility v1 — REPORT

protocol `cscl-bjoc-strc-predictive-feasibility-v1` — pre-registration committed
`22d18d8` (config + CPU gates) before any GPU run; three pre-GPU protocol
amendments (`e996cc0`, `7e93013`, `41a49c0`, none after any training);
training on the deployed revision `41a49c0`; generated 2026-10-13.

**Final decision: `STRC_NO_PREDICTIVE_GAIN`.**
The 157-parameter shared additive residual channel (28 `alpha` + 129 `beta`)
is live, cross-seed stable and non-trivial in magnitude, but it makes official-valid
seed-mean raw-y MAE **worse** than the matched masked-pretrained P
(`+0.004543`, CI95 `[+0.000927, +0.009536]`, both seeds positive). The full
`STRC_DEV_GAIN_SUPPORTED` gate failed 7/8 conditions, so the shared additive
residual route stops here; official test was never loaded.

## 1. Question and intervention

On the already-successful masked-pretrained BJOC-Null P (official-valid seed-mean
0.083549, exposed-test 0.073957 as background), can a **tiny**,
cross-molecule-shared, explicitly additive structural residual channel improve
held-out single-model raw-y MAE when trained jointly under the identical recipe?

```
y_hat(G) = f_theta(G) + s_a * sum_i alpha[type_i] + s_j * sum_j beta[context_j]
```

- `f_theta` = the complete pretrained BJOC-Null forward (413,505 params),
  unchanged; `alpha` = 28 shared atom-type scalars, `beta` = 129 shared
  junction-context scalars, both zero-init; 157 new params, total 413,662.
- context = `(endpoint_type_low, centre_type, endpoint_type_high)` with sorted
  endpoints; vocab from official train10k structures only (count once, sort by
  (-count, lexicographic), keep top 128, rest OTHER; 201 distinct contexts,
  kept 128 cover 99.95% of junctions; `vocab_sha256 18303ee1...`).
- `s_a = 1/mean_train(n_atoms) = 0.0431659645`, `s_j = 1/max(mean_train(n_junctions),1)
  = 0.0288718609`, frozen, non-learnable.
- Matching: S starts from the same post-SSL / pre-supervision state as P
  (`f10k._make_downstream_model(seed)` replayed init hash + frozen
  `pretrain_encoder_s{seed}.pt`), same recipe (Adam 1e-3 / wd 1e-5 / batch 128 /
  clip 5 / fp32 / 240 epochs / 18,960 updates / last-5 soup / raw-y L1 only),
  same batch order. Verified in `train_S_seed{0,1}.json`: `same_init_as_P`,
  `same_composer_reader_init_as_P`, `same_encoder_as_P`, `same_batch_order_as_P`
  all true; batch-order hash `94a8ff7f...` equals P's recorded value.

## 2. Official-valid results (n = 1000, single label read after freeze)

| model | seed 0 | seed 1 | seed-mean | seed-mean RMSE |
|---|---:|---:|---:|---:|
| P (frozen, same recipe) | 0.084786 | 0.082313 | **0.083549** | 0.230007 |
| S (STRC) | 0.092619 | 0.083566 | **0.088092** | 0.265855 |
| Δ = S − P | **+0.007833** | **+0.001253** | **+0.004543** | **+0.035848** |

Paired molecule bootstrap (5000 resamples, seed 0): Δ mean `+0.004543`,
CI95 `[+0.000927, +0.009536]`, fraction of molecules better `0.454`.
Predictions were written label-blind and frozen (`prediction_freeze.json`,
combined `7ecdb46a...`) before the single valid read (`valid_read.lock`
2026-10-13T02:27:24Z, y sha `8f6ee636...`).

Frozen gate (`STRC_DEV_GAIN_SUPPORTED`), 7 of 8 conditions fail:

| condition | result |
|---|---|
| CPU correctness gates | True (passed before GPU) |
| seed-mean gain ≤ −0.004 | **False** (+0.004543) |
| both seed deltas < 0 | **False** (+0.007833 / +0.001253) |
| ordinary (y>−3, n=935) not worse | **False** (+0.002437) |
| tail (y≤−3, n=65) strictly better | **False** (+0.034843) |
| seed-mean RMSE ≤ P + 0.005 | **False** (+0.035848) |
| top-10-removed Δ < 0 | **False** (+0.003033) |
| not driven by valid:0172 | **False** (row share 39.6%) |

## 3. Is this a real improvement? No — it is a small, reliable degradation

- **Sign and reliability.** Both seeds are worse and the paired CI lies entirely
  above zero: this is a reproducible ~0.0045 MAE degradation (5.4% relative),
  not a null. Seed 0 degrades more (+0.0078) than seed 1 (+0.0013).
- **Bulk.** Ordinary molecules (935/1000) are worse by +0.002437; on seed 1 the
  ordinary delta is slightly negative (−0.00059), but the seed-mean is positive
  and the pre-registered requirement is not met.
- **Tail.** Tail MAE rises +0.034843 (0.2813 vs 0.2465). Excluding the extreme
  row `valid:0172` (y = −20.34) the tail delta is still +0.007258 and the
  overall delta +0.002746; the row carries 1.800 of the tail delta but the
  degradation is not single-row driven. The signed tail error grows
  (+0.2153 vs +0.1684), i.e. the magnitude shrinkage of extremes is **worse**.
- **RMSE.** +0.035848 (0.2659 vs 0.2300), seven times outside the 0.005
  tolerance; the pre-registered robustness and RMSE checks all fail.
- **Scale strata.** The damage concentrates in the large/complex molecules:
  high n_junctions +0.007332 / high n_atoms +0.008515 vs low +0.001885 /
  +0.001035 — exactly the regime where a count-scaled additive shift and a
  size-dependent composition error have the most leverage.
- **Fit vs valid.** S train fit is seed-mixed but slightly worse on average
  (S0 0.037733 vs P0 0.025333; S1 0.025487 vs P1 0.028498). Train subgroup
  deltas point the other way from valid (train ordinary +0.005911 worse,
  train tail −0.011070 *better*), so the only positive signal the channel
  produced (a small train-tail gain) did not transfer; on held-out tail it
  bought a 4× larger loss. Fit−valid gaps are similar across arms
  (S −0.055/−0.058 vs P −0.059/−0.054), so this is not a generic overfit
  explosion: the jointly trained solution simply generalizes worse.

Answer to the round question: the shared additive residual channel does not
improve held-out single-model raw-y MAE; under the frozen gate it is rejected.

## 4. Mechanism evidence (`parameter_analysis.json`, `error_analysis.json`, `post_hoc_mechanism_analysis.json`)

The channel is *used*, *stable*, and *non-trivial* — the failure is not a dead
parameter block:

- Soup tables: 21/28 `alpha` and 128–129/129 `beta` entries non-zero
  (|·| > 1e-3); mean |alpha| ≈ 0.153/0.155, mean |beta| ≈ 0.168/0.162;
  residual mean |·| ≈ 0.089/0.098 ≈ 6–7% of mean |base output| ≈ 1.4–1.5.
- Cross-seed agreement: Spearman alpha 0.9907, beta 0.8921 (Pearson
  0.9930 / 0.9303). The two seeds learn nearly the same tables, so unstable
  fitting is not the explanation.
- **The channel acted mostly as an optimization reparameterization, not as an
  additive correction.** Seed-mean |S−P| prediction difference is 0.049 while
  |residual| is 0.094 and |base change| is 0.081–0.120; the S−P prediction
  difference is explained by the *base change* (R² 0.426 seed 0 / 0.731 seed 1)
  and essentially not by the additive residual (R² 0.0025 / 0.000005). The
  trained base solution moved far from P (relative L2: atom 0.20/0.22, bond
  0.52/0.57, junction 0.58/0.59, composer 0.91/0.96, reader 0.88/0.88); the
  base and the residual compensate, and the reparameterized optimum is worse
  on held-out data.
- Rare-context coefficients are a secondary contributor: the largest |beta|
  entries include contexts seen in only 30–50 train molecules
  (`[2,4,5]` 41, `[2,4,2]` 53, `[0,12,0]` 85), and 7/21 (seed 0) and 5/18
  (seed 1) of |beta| > 0.3 entries have < 100 supporting molecules; the
  aggregate |beta|–support Spearman is only +0.13/+0.14, so this is a
  contributor, not the dominant story.
- **Identifiability caveats (explicit).** The train design matrix has
  rank 149/157, null dimension 8; 7 atom types have zero support in train10k
  (classes 21–27; 27 is the mask token) and their `alpha` never moves; one
  context pair (slots 96/107) is exactly collinear; the non-zero-spectrum
  condition number is ~3.1e3. Individual coefficients are therefore only
  partially identified, and the OTHER slot carries beta −0.218/−0.198.
  `alpha`/`beta` are **explicit additive residual contributions inside this
  model — they are not verified chemical contributions**, and the cross-seed
  agreement is a descriptive diagnostic, not a causal claim.

## 5. Route decision and next research proposition

**Decision: `STRC_NO_PREDICTIVE_GAIN`.** The seed-mean delta is positive and
7/8 gate conditions fail. Per the frozen stop rule, the shared additive
residual route stops permanently: no vocabulary/scale/tail-weighting/third-seed
follow-up. P remains the reference model; no new training is authorized by this
round.

### Candidate next mechanisms (3 ideas, 1 recommendation)

- **A (recommended) — Scale-separated whole-molecule composition.** Replace the
  extensive-only readout `[Σc, Σa, Σa², g]` by a capacity-matched
  scale-separated readout `[Σc, mean(c), std(c), Σa, mean(a), std(a), n, g']`
  computed from the *same* object tensors (c = per-bond composer outputs,
  a = per-atom encodings), with the reader width reduced so the total stays at
  413,505 parameters. This changes the composition algebra itself (extensive →
  extensive + intensive), not the output and not the local features.
- **B (rejected) — Count-conditioned whole-molecule scale calibration**
  (`a(counts)·f + b(counts)` / Route B). Rejected as the next step: it is
  another small whole-molecule correction from the same family that has now
  failed twice (STRC here; the pre-registered reasoning that the tail failure
  is statistical), and the round brief asks for a genuinely different
  composition bias, not another correction term.
- **C (rejected) — Third-order local interaction channel** (junction-triplet /
  path-context products). The Cross-Bond round already showed that a
  functionally novel two-endpoint interaction channel gives
  information-without-gain, and TypedPathRing-style local typed features
  overfit; a third-order variant of the same local-interaction family is
  excluded by the stop rule.

### The proposition, answering the 10 questions

Q1 *Which whole-molecule relation class might BJOC lack?* BJOC composes the
molecule only through extensive sums of object tensors (plus global fraction
features). It lacks a scale-free description — how the per-object contribution
is distributed across the molecule (mean, dispersion, extremes) and how that
distribution interacts with size. The evidence points there: the diagnosed
failure is tail magnitude shrinkage at low-density extremes, and the STRC
degradation is concentrated in the large-molecule strata.

Q2 *Which existing structural information is hard to combine after pooling?*
The shape of the already-computed object tensors: two molecules with the same
`Σc`, `Σa`, `Σa²` and the same global fractions but different dispersion of
per-object contributions are indistinguishable to the current reader.

Q3 *Why is it not just adding parameters?* It is a re-parameterization of the
readout at matched total capacity (413,505), replacing extensive columns with
intensive ones; no new local features and no output residual. Any gain must be
earned through the same nonlinear reader.

Q4 *How is it fundamentally different from Cross-Bond / TypedPathRing / STRC?*
It adds no new local structure, no pair/three-body interaction and no
output-side table: only the sufficient statistics of the existing object set
change (extensive → extensive + intensive). Cross-Bond changed within-bond
endpoint pairing; TypedPathRing added local typed counts; STRC added a global
output residual. This changes the composition operator itself.

Q5 *Why might it improve small-sample generalization?* The diagnosed failure is
statistical extrapolation/shrinkage (train extremes are fit; held-out extremes
shrink to the bulk). Intensive statistics let the reader model per-object level
separately from size instead of extrapolating an extensive sum; it uses only
deterministic functions of the same tensors, so it adds no sparse one-hot
inputs (avoiding the TypedPathRing overfit mechanism).

Q6 *Does it stay non-iterative (no GNN/attention/Transformer)?* Yes: one static
pooling pass plus the existing reader MLP; no message passing, no iteration, no
attention, no extra hidden stack; encoders/composer untouched.

Q7 *What synthetic task separates old and new readouts?* Construct object
multisets with identical old-reader inputs (same `Σc`, `Σa`, `Σa²`, same global
features) but different intensive statistics, with a target that depends on the
intensive statistic; the old reader is provably constant across the collision
pair while the target differs, and the new readout separates them and recovers
the planted slope. Add a real-molecule audit of how often near-identical
extensive summaries co-occur with different targets.

Q8 *How is alpha/gamma decomposition preserved?* The new statistics are
explicit deterministic functions of the object tensors and per-object
contributions remain available; the prediction can still be written as
reader(deterministic statistics) with a per-object ledger, with no opaque
bypass.

Q9 *Minimal GPU contrast?* One arm V, two seeds, same SSL checkpoints, recipe
and batch order, against the frozen P valid predictions; ~0.35 GPU-h; same
official-valid gate structure (seed-mean, both seeds, ordinary, tail, RMSE,
top-10-removed, extreme row).

Q10 *What should stop the new mechanism?* If V fails the same valid gate
(seed-mean delta ≥ 0, or ordinary/tail conditions fail), stop the
readout-composition route and do not iterate pooling bases; the next escalation
must change the object definition or the data/pretraining route, not another
readout tweak.

## 6. Disclosures

1. Pre-GPU protocol amendments (all before training, recorded in commits):
   (a) base-gradient-control tolerance 1e-12 → 1e-6 because remote float32
   reduction order caused a 3.7e-9 discrepancy (still ≪ gradient scale);
   (b) P reference prediction hashes frozen as constants and `_verify_inputs`
   split into training keys (blocking) and reference keys (recorded), after the
   remote revision was found to lack the F10K round's local-only prediction
   freeze; (c) `stage-cpu-prep` always re-runs the CPU gates so a blocked
   artifact cannot be silently reused. Remote prep runs 1–3 failed for these
   reasons and were re-run; prep-4 succeeded (driver freeze `b9f88a80...`).
2. `post_hoc_mechanism_analysis.json` was computed after the single valid read;
   it is descriptive only and changed no model, vocabulary, scale, gate or
   decision (the frozen decision JSON was already written).
3. Training ran sequentially on `res` pool `res-gpu0` (one trajectory at a
   time): `strc-s0` (547 s), `strc-s1` (668 s), torch 2.5.1+cu124, exit 0 both;
   predictions/scoring on local CPU, matching the P prediction path.
4. `valid:0172` (y = −20.3405): S seed-mean prediction −13.81 vs P −15.61;
   the row is disclosed in every robustness number and never removed from the
   headline metric.

## 7. Discipline

- Official test never loaded in any form. Official valid read exactly once,
  after all four prediction files were hashed; no training, selection or
  retuning after the read; `valid_read.lock` records the y hash.
- No ensemble, calibration, target transform/clipping, tail-weighted loss,
  teacher, extra labels, GNN/attention/iteration, extra MLP, learnable scale,
  count-conditioned scaling, second vocabulary, third seed, or seed/epoch
  selection on valid.
- Capacity: 413,662 = 413,505 + 157 exactly, audited in both train JSONs.
- Budget: 2 trainings (seeds 0/1), ~1,215 s ≈ **0.34 GPU-h** training of the
  0.8 cap; wall from authorization to report ≈ 1 h 40 min of the 6 h 30 min.

## 8. Artifacts

Committed: `configs/bjoc_strc_predictive_feasibility_v1.yaml`,
`code/bjoc_strc_v1.py`, `code/run_bjoc_strc_v1.py`,
`tests/test_bjoc_strc_v1.py`, `notes/bjoc_strc_predictive_feasibility_v1.md`,
this report, `scientific_decision.json`, STATE.yaml.

Git-ignored under `results/bjoc_strc_predictive_feasibility_v1/`:
`context_vocab.json`, `design_matrix_audit.json`, `cpu_precheck.json`,
`synthetic_task.json`, `protocol_freeze.json`, `count_matrix_train10k.npz`,
`train_S_seed{0,1}.json`, `soup_S_seed{0,1}.pt`,
`predictions/valid_S{0,1}.npz`, `predictions/valid_S{0,1}_parts.npz`,
`prediction_freeze.json`, `valid_read.lock`, `valid_y_read_once.npy`,
`valid_metrics.json`, `paired_bootstrap.json`, `train_metrics.json`,
`parameter_analysis.json`, `error_analysis.json`,
`post_hoc_mechanism_analysis.json`, `scientific_decision.json`.
