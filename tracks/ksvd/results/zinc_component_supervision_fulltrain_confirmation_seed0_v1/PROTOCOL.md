# PROTOCOL — `zinc_component_supervision_fulltrain_confirmation_seed0_v1`

Frozen **before** either formal trajectory ran and before the first official-valid read.
One question, two matched body arms, one seed, the full 10,000-row official-train range,
one frozen official-valid confirmation. No search.

## 1. Question

The seed-0 candidate round (`zinc_chemistry_component_supervision_seed0_v1`) measured, on the
internal fresh fold (8000 fit / 2000 reused diagnostic), a calibrated `g`-MAE gain of
`+0.009974` (G0, CI `[+0.006459, +0.013493]`) and `+0.009873` (overall, CI `[+0.006440, +0.013551]`)
for true chemical-component supervision (`COMP = L_g + 0.5·(L_ell + L_s)`) over the matched
total-only arm (`SUM = L_g`).

This round asks exactly two things:

1. **CHEM.** Does the ~0.01 chemistry `g` gain survive when the body is trained on the
   **full 10,000 official-train rows** and confirmed once on the frozen **official-valid** split?
2. **DEPLOY.** With both chemistry predictors connected to the *same* train-only fixed cycle
   head `Q`, what is the gain in the complete deployable `y`? Is the valid `y`-MAE actually
   below `0.09`?

The only scientific intervention is unchanged: same M skeleton, same input, same total target
`g`, compare total-only supervision against total + component supervision. Only the training
data range changes.

This is a single-seed full-train baseline confirmation. It is **not** a training-randomness
replication (no seed 1), and **not** an independent-data validation: official-valid has been
used in earlier research rounds and is explicitly disclosed as reused. The old internal dev2000
is now part of the training set and is no longer held out. official-test is outside this round's
authorisation: it is never read, cached, predicted or scored.

## 2. Fixed sources (read-only)

* Structure / incidence cache (structure only, byte-checked):
  `tracks/ksvd/results/zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1/local_tuple_index.npz`
  (`root_base`, `node_sizes`, `pair_ptr`, `pair_t`, `pair_a`, `pair_wJ`, `pair_wI`, `root_atom`).
* Train-only raw labels: `zinc_long_cycle_audit/train_cycle_audit_label.csv` (10,000 positional
  train rows) + `zinc_long_cycle_audit/cache/gvae_full_properties.npz` (audit logP/SA/cycle) +
  `zinc_dictionary_real_data_handoff/train.npz` (canonical `y`, `canonical_group_id`).
* Inversion constants for the encoded cache: `zinc_joint_dictionary_decision_v1/prep/fold_objects.npz`
  (`*_all_mean` / `*_all_scale` only; the stored `*_fit_*` 8k scalers are never used as this
  round's train-only objects).
* Body skeleton: `mlpmod.build_arm_mj` (297,499 params) with the same `ComponentReader`
  (39→2 half-split of the original 39→1 output head, 297,539 params).
* Cycle head recipe: `zinc_frozen_chemistry_learned_cycle_v1` small head structure / optimiser
  recipe only. That round's trained Q model is never loaded, and its old decomposition labels
  are never reused.
* Official valid/test are never loaded in any train / calibrate / manifest phase.

## 3. Full-train data, preprocessing and labels (10k train-only fit)

Training set = all 10,000 stable official-train IDs in fixed canonical (positional) order.
The old internal dev2000 is inside the training set; it is no longer held out.

Every fit-dependent object is rebuilt / refit on the **full 10,000 train rows**, keeping the
existing algorithm, initialisation, sampling seed and epsilon/scale floors:

* body patch / context / anchor / topology standardizers (invert the all-train cache constants,
  refit on all 10,000 root rows with `CANON_STD_FLOOR`);
* local-tuple `phi` scaler (mean/std with `PHI_STD_FLOOR`, L2-RMS `phi_scale`) and the
  `kappa_M` sample (`KAPPA_SEED = 20261004`, up to 8192 uniformly sampled full-train roots);
* target-decomposition constants `sigma_logP`, `sigma_SA`, `mu_SA`, `sigma_cycle`, `mu_cycle`
  (cycle-free OLS bulk + pre-existing Nelder–Mead snap loss, all 10,000 rows).

`MU_LOGP = 2.4570953396190123` stays a fixed label-generation constant and is never re-estimated.
Row-wise:

```
c = (k − mu_cycle) / sigma_cycle
g = y − c
ell = (logP − MU_LOGP) / sigma_logP
s = g − ell                       # "residual chemistry component"
```

The source SA sign convention is preserved. `s` is never called pure SA.

The frozen `full_train_targets.npz` cache records stable ID / `y` / `g` / `c` / `k` / `ell` /
`s` / `s_SA` / `epsilon`, the constants, source paths + SHA-256, and the training row set.
Float64 identities `y = g + c` and `g = ell + s` are checked; float32 training allows normal
rounding. Labels live only at the trainer / scorer boundary: `model.forward` and `Q.forward`
never read `y/g/c/k/ell/s` or any label-derived snap value, and the raw `logP`/`SA` attribute
table is never fed to the model as a new input.

The old 8k scalers, old kappa, old 8k schedule and the mixed 12k (train+valid) constants are
never reused as this round's train-only objects.

## 4. Exactly two body trajectories

Batch-mean L1:

```
L_g   = MAE(hat_ell + hat_s, g)
L_ell = MAE(hat_ell, ell)
L_s   = MAE(hat_s, s)

SUM:  L = L_g
COMP: L = L_g + 0.5 × (L_ell + L_s)
```

The `0.5` coefficient is fixed: no scan, no gradient normalisation, no dynamic balancing.
SUM's component losses are logged detached only.

Shared recipe: seed 0; 240 epochs; batch 128; Adam `lr 1e-3`, coupled `weight_decay 1e-5`;
grad clip 5; FP32; no AMP / DDP; fixed parameter soup epochs 236–240 (parameter mean, never
selected by any held-out score).

At 10,000 rows each epoch is `ceil(10000/128) = 79` steps, so 18,960 steps per arm. A new
schedule is built with the existing `build_schedule` at `seed0 + TRAIN_SHUFFLE_OFFSET (=101)`
and its hash is frozen; it is **not** required to equal the old 8000-row schedule hash. The
global graph-ID stream is also frozen and asserted identical across the two arms.

During training only the train diagnostics are inspected: no official-valid evaluation, no old
internal dev evaluation, no change to the stopping epoch. `init` / `last` / `raw_soup` states,
training curves, the actual recipe and metadata are saved. No diagnostic/data iteration consumes
the formal dropout/schedule RNG.

## 5. Shared cycle head Q (fixed deploy accessory, not a new exploration axis)

Exactly one `Q_seed0` is trained. Both arms use the same Q weights, the same input prep and the
same `q_raw`; there is no per-arm Q and no head selection.

Q input is the actual `topology25` model input after this round's full-train prep; the output is
in `c` (y-contribution) units. Structure `25→64→32→1`, SiLU, 3,777 params. The existing recipe of
`zinc_frozen_chemistry_learned_cycle_v1` is reused strictly:

* seed 0; last layer `weight = 0`, `bias = median(c_train)`;
* batch 128, 300 epochs, 23,700 steps; Adam `lr 1e-3`, coupled `wd 1e-5`, clip 5; uniform sampling;
* the existing independent `torch.Generator` seed `20261003`;
* fixed soup epochs 296–300; `L1(q, c)`; local CPU, FP32, at most 8 threads.

Q only receives the full-train T25 and `c_train`. No severity weighting, no size-input repair,
no capacity probe, no extra topology features, no resampling, no `q` bias search. Q's independent
RNG never touches the body arms. The small head is retrained (not reused) so that it is aligned
with this round's 10k train-only constants and T25 prep.

Q's unresolved extreme-cycle errors are a disclosed boundary of this round, not a purchase
condition for a follow-up experiment.

## 6. Calibration and deployable wrappers (one bias per endpoint, per arm)

With `h_a = hat_ell_a + hat_s_a`, `a ∈ {SUM, COMP}`:

```
chemistry diagnostic:
g_raw_a = h_a
b_g_a   = median(g_train − h_a_train)
g_cal_a = h_a + b_g_a

complete task:
y_raw_a = h_a + q_raw
b_y_a   = median(y_train − h_a_train − q_raw_train)
y_cal_a = h_a + q_raw + b_y_a
```

`b_g` and `b_y` are two different train-only endpoint biases. The complete-`y` wrapper does not
fold `b_g` first and does not add a second Q bias (`Q`'s own learned last-layer bias is already
part of `q_raw`). No stacked `b_g + b_q + b_y`, no valid median, no raw/cal selection by score.

Both wrapper inferences read only the graph and its fixed features: one shared Q, the
corresponding body, one `b_y`. True `c` is never supplied, and oracle `g + c` is never presented
as deployable performance.

Per-arm `g`/`y` raw/cal, per-arm `ell`/`s` raw and the shared `q_raw` are saved. The float64
row-wise identity

```
y_cal_a − y = (hat_ell_a − ell) + (hat_s_a − s) + (q_raw − c) + b_y_a
```

is checked. Component absolute errors are never added as a `g`/`y` budget. A shared Q does not
imply the complete-`y` MAE gain equals the `g` gain: cancellation and `b_y` change the final
error.

## 7. Two freezes and one official-valid read

**First freeze** (before formal training): PROTOCOL, METHOD_CONTRACT, runner, target/input
manifest, shared init, recipe, schedule, gate and evaluation code.

**Second freeze**: after both body arms and Q complete, the train-only `b_g`/`b_y` are computed
and the train replay passes; the `frozen_eval_manifest` lists

* the three soups, prep, constants, targets, shared Q and the four biases with SHA-256;
* the inference wrapper and evaluation code source version / hash;
* the two-arm prediction composition, calibration paths, group definitions and the
  bootstrap/gate.

Only then is official-valid instantiated / read for the first time in this round. The read time,
source, stable-ID / label alignment and frozen commit are recorded in `heldout_access`.

official-valid is evaluated unconditionally after the two arms and Q are frozen; train scores
never decide whether to evaluate. Exactly one complete frozen evaluation is performed;
engineering replays may replay the same predictor but never become selection / tuning.
official-test is never instantiated, read, predicted or scored. No ensemble is formed and the
old internal dev is not evaluated again.

## 8. Two separately pre-registered judgements

All gains are `MAE(SUM) − MAE(COMP)`; positive means COMP improves. All primary results use the
fixed soup; `last` is saved but never ranks or changes the choice.

**`CHEM_CONFIRMED`** — all four:
1. valid G0 calibrated `g` gain ≥ 0.003;
2. valid overall calibrated `g` gain ≥ 0.003;
3. G0 calibrated `g` paired 95% CI lower bound > 0;
4. G0 and overall raw `g` gains both > 0.

**`DEPLOY_CONFIRMED`** — all four:
1. valid overall calibrated `y` gain ≥ 0.003;
2. overall calibrated `y` paired 95% CI lower bound > 0;
3. overall raw `y` gain > 0;
4. COMP vs SUM G0 calibrated `y` worsening ≤ 0.001.

`valid y_cal < 0.09` is a separate absolute score marker `BENCHMARK_TARGET_OBSERVED`; it never
replaces the incremental gate and no rows may be deleted to claim it. `g < 0.09` is listed
separately and is never mixed with `y < 0.09`. A single-seed near-threshold value is this
round's observation only, not a robust SOTA.

| CHEM | DEPLOY | this round's conclusion and next responsibility |
|---|---|---|
| pass | pass | component supervision is fully confirmed; promote to the new chemistry/deployment reference; design the next targeted `s` question |
| pass | fail | chemistry gain retained; complete task not confirmed. Use the row-wise identity to judge cycle-error cancellation / calibration / component effects; do not buy a cycle-rescue training run |
| fail | pass | only a deploy signal under the complete combination; do not claim chemistry-generalisation confirmation; keep the candidate and report the inconsistency |
| fail | fail | this full-train confirmation failed; keep the internal-dev positive result scoped as such, close this fixed-recipe confirmation chain; no coefficient/seed/model-selection rescue |

Below-gate but positive-direction results are recorded as `DIRECTIONAL_NOT_CONFIRMED`; a CI
crossing 0 is not "equivalence". Incomplete engineering or failed identity is recorded
separately as `EXECUTION/IDENTITY` and never disguised as a scientific negative.

Paired bootstrap: 1000 draws, seed `20261009`, shared row indices across arms; G0 / overall use
their own row sets. Witnesses: identical predictions → 0, swapped arms → mirror, constant shift
→ inside the bound. The CI covers only this model on these rows; it does not cover training-seed
or repeated-selection uncertainty.

The main table must contain train/valid × G0/overall × g/y × raw/cal, the biases and the gaps;
no endpoint is omitted. G0 uses the frozen `k = 0`; the other groups are `k = −1`, `k = −2`,
`k ≤ −3`.

## 9. Analysis restricted to this round's saved predictions (no new fitting probe)

Reported:

* per-group `n`, MAE, contribution `Σ|err|/N`, paired gain; group contributions and gains add
  back to the overall exactly.
* COMP `ell`/`s` train/valid, G0/overall raw-MAE, signed error, fit-constant-median contrast
  and gap.
* Q's train/valid per-group error against `c`; extreme-cycle rows with row-wise
  `h`, `g`, `q`, `c`, `y` prediction, bias and the error identity.
* COMP `e_ell`/`e_s` opposite-sign rate, `triangle_gap`, the component sum vs the true `g`
  error — never added as a budget.
* complete-`y` chemistry/cycle signed error, cancellation and bias; why the `g` gain and the
  `y` gain can differ.
* fixed sensitivities: excluding valid row 0172 (if the ID mapping exists) and excluding the
  largest SUM calibrated `y` error row; both the unfiltered main result and the filtered result
  are reported; the gate is unaffected and no training sample is deleted.
* the full-train prep / target constant changes versus the old 8k source, as deterministic
  differences only; no experiment is reopened because of them.

If chemistry confirmation fails, a single large `s`-MAE is not enough to buy an `s` head /
feature. If it succeeds, any later `s` design must keep total `g`/`y` as the final judgement to
avoid sacrificing favourable cancellation. This round writes the next question only; it does
not execute it.

## 10. Server, time and required engineering checks

Target wall clock ≤ 120 min, hard stop 180 min; no new training starts after minute 120.
GPU allocation budget ≤ 1.2 GPU-hours including smoke, failed fragments and recovery; at most
two concurrent GPUs. The body arms run first (parallel when legal); Q runs on the local CPU in
parallel; total local threads ≤ 8.

Remote execution uses the `remote-research-runner` skill (`rr doctor/deploy/run/status/logs/pull`)
with `host=res-2`, `pool=res2-cu124`. Both arms run in the same GPU model / driver / torch / CUDA /
Python regime and on the same deployment commit. The actual GPU UUID / PCI, Slurm allocation and
node are captured inside each task process; `CUDA_VISIBLE_DEVICES=0` is a local index, not shared
physical-card evidence. The two overlapping tasks must show different physical UUIDs; if only one
card is available they may run sequentially but must still share the regime and stay within
budget. If no GPU is available, the round reports the resource block and does not switch the body
training to CPU.

Pre-flight checks are limited to what this round needs: label/index/shape contracts, initial
function identity, paired init/RNG/schedule, fit-only smoke, label-independence of the forward,
save/reload, train replay. Floating-point forward tolerance is 1e-5; exact comparison is only for
state/hash identity. Real batches assert target `(n,)` and components `(n,2)` to rule out
broadcasting. Smoke states are discarded and use an independent RNG.

The final wrapper vs native raw prediction replay tolerance is 1e-5; cumulative float64 statistic
identities are 1e-9. A valid training run is never cancelled just to print a field or because of
a small CPU/GPU rounding difference.

Recovery is frozen before formal training: model/optimizer/epoch/step, schedule position, tail
soup and Python/NumPy/Torch CPU/CUDA RNG are saved; RNG byte tensors are moved to CPU first.
Preemption may resume the same logical trajectory. If a resume is impossible, an engineering
restart is allowed at most once only when that trajectory's held-out score has not been seen, the
recipe is unchanged and the budget allows; discarded fragments are charged to the budget and
never used for model selection. Any score-driven restart / continuation is forbidden.

If the budget or resources cannot complete the body arms / Q and the frozen confirmation, the
existing results are saved and marked `INCOMPLETE`; epochs are not lowered, the recipe is not
swapped, no old Q is substituted and no unmatched historical control is stitched together. All
self-launched jobs reach a terminal state before delivery; no running/pending job is left.

Only the relevant fast checks and the repository-mandated `research verify` are run; no unrelated
large test suite and no repeated historical audit. The body training should be submitted within
the first 60 minutes; if pre-flight blocks, the concrete blocker is delivered instead of burning
the budget rebuilding the whole research platform.

## 11. Delivery and stop

Isolated branch `task/zinc-component-supervision-fulltrain-confirmation-seed0-v1`; directory
`tracks/ksvd/results/zinc_component_supervision_fulltrain_confirmation_seed0_v1/`.

Register and commit this round's protocol, implementation and results per repository rules; do
not push / merge, do not move `main`, do not modify historical results or other people's
uncommitted files.

Delivered together:

* PROTOCOL / METHOD_CONTRACT / REPORT / DECISION / EXECUTION / EVIDENCE_SCOPE / ERRATA as needed;
* train-only prep / targets / provenance, shared init / schedule / stream, both arms'
  init/last/raw_soup, Q init/soup and curves;
* frozen_eval_manifest, heldout_access, GPU UUID/allocation, recipe/runtime, budget and manifest;
* the g/y main table, group contributions, COMP component/cancellation, Q group table, row-wise
  train/valid predictions, paired gain/CI and both gates;
* two deployable wrappers plus a single-file replay and the analysis re-run command;
* one appended conclusion entry in the existing RESEARCH_STATE / evidence ledger; the historical
  paragraphs are not rewritten.

The REPORT opens by directly answering:

1. On the full-train valid, is COMP's `g` gain over SUM confirmed? Do raw/cal agree in sign?
   How much of the internal `+0.00987` is retained?
2. What are the two arms' raw/cal complete deployable `y`? Is it actually `< 0.09`? Does the
   DEPLOY gate pass?
3. Which rows, chemistry/cycle signed errors and calibration produce the `g` vs `y` difference?
   How much of the gain lands in G0?
4. Is `s` still the main gap object inside COMP? How large is the cancellation? What can and
   cannot be concluded from these facts?
5. Which conclusion is kept/closed this round, and where does the next research responsibility
   lie? How are the single-seed, extra-supervision-label, historical-valid-reuse and
   dictionary-claim boundaries disclosed?

Stopping point: this confirmation round. No seed 1, no third body arm, no new fold, no
loss/optimizer/capacity scan, no tail weighting, no dictionary revival, no oracle deployment, no
test evaluation and no post-hoc rescue. Whether there is a material gain and whether it is
explained are answered separately; work is not extended just to obtain a positive result.
