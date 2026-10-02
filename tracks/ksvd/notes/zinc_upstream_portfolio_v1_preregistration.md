# Upstream-Portfolio-v1 pre-registration (Workstream Z, ZINC)

Round: `upstream_portfolio_v1`. Core module:
`tracks/ksvd/experiments/luyin16/e2e_dictenv_upstream_portfolio_v1.py`.
Orchestration: `tracks/ksvd/experiments/luyin16/zinc_upstream_portfolio_v1.py`.
Runner: `zinc_upstream_portfolio_v1`.

## 1. Fixed starting point

* Parent model: `E2E-DictEnv-Scale-v1` Full (`m = 3`), class `LatentScaleSEM108`,
  408,651 trainable parameters; valid Top-5 soup 0.1191540920 (historical CPU
  reference). No message passing, no Transformer/attention, no node/edge hidden
  state, no write-back.
* Parent checkpoint (reused, never retrained here):
  `tracks/ksvd/results/e2e_dictenv_scale_v1/checkpoints/SCALE-FULL-seed0_soup_state.pt`,
  sha256 `17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb`.
* Split fingerprint `58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a`.
* Structural dictionary `D` K32/s8, bridge `D_L [144, 288] / V_L [288, 144]`,
  16-step ISTA, `lambda1 = 0.05`, `lambda2 = 0.01`: untouched, never swept.
* The official ZINC **test** split is never read, instantiated or cached.

## 2. Arms (exactly three candidates + one shared control)

Ordered priority: `code`, `drop`, `r3`. No fourth candidate may be added.

| arm | unique change | Full params | Small params |
|---|---|---:|---:|
| `control` | none | 408,651 | 106,925 |
| `code` | reader input `814 + 576 -> 1390`, old columns/bias/later layers reused, new columns zero | 431,115 | 109,421 |
| `drop` | training-only shared atom mask, `E = (c * mask/(1-p)) @ V_L`, `p = 0.10`, no new params | 408,651 | 106,925 |
| `r3` | fusion input `446 + 36 -> 482`, old columns reused, new columns zero | 420,963 | 111,029 |

Closed-form counts are asserted by `arm_parameter_audit` and the Small audit.
Small is constructed for the parameter audit only and **never trained**.

* `code`: `c_i = rho_i * alpha_i` from the *same single* bridge call that
  produces `E`; `z = [sum_i c_i, sum_i c_i^2]` gets two train-only RMS scalars
  (one per moment block), never per-column standardisation. No new dictionary,
  no second ISTA call, no raw bypass, no count slot, no write-back.
* `drop`: mask length `K`, shared across all object rows, independent of node
  order, drawn from an arm-private generator; no solver-internal noise; eval
  closes the mask and is exactly the parent path.
* `r3`: exact BFS shell-3 `[28 atom counts; 4 (2,3) bond counts; 4 (3,3) bond
  counts]` on the physical de-duplicated undirected graph; three train-only
  block RMS scalars (`1.0` for an all-zero block); appended at the *end* of the
  fusion input.

## 3. CODE stage-0 conditional (before any CODE training)

Frozen-parent (`control`, parent soup) forward over official train/valid
exports the frozen 39-D *second* reader hidden (the completed `H39_PHYSICAL`
design; the first hidden was the discarded RMS-scaled variant) and `z`. Fixed
convex fit
`[1, H39, z/scale]` with MAE + `1e-5/2 ||w||^2`, certified primal-dual gap
`<= 1e-6`, solver `upstream_portfolio_v1_reference/prototype_dictionary.py`.
Baseline `H39_PHYSICAL.npz` valid MAE must reproduce `0.1150237843`.

CODE proceeds to the 80-epoch pilot only if **all** hold:

* valid `<= 0.112`;
* gain vs the H39 baseline `>= 0.003`;
* at least 4 of the 5 fixed `id % 5` valid groups improve.

Otherwise record `CODE_CONDITIONAL_STOP` (a stop of this fixed proxy only, not a
claim about the theoretical ceiling of native dictionary codes). An
unconverged/over-budget fit is `INCONCLUSIVE`, never a negative.

## 4. Shared 80-epoch warm pilot

All arms warm-start the identical frozen Full soup (not the fitted H39 head, not
a pre-set parent bias). Control is bought once and shared.

* 80 epochs, train 10,000, batch 128, task loss `MAE + lambda_rec * rec`
  (the parent training loss; `lambda_rec = H1_LAMBDA`), Adam `lr = 1e-4`,
  `weight_decay = 1e-5`, `clip = 5`, parent dropout settings.
* Identical sample shuffle `seed = 0 + 91011`; identical shared parameter
  initial values; branch randomness uses a private generator.
* Pilot soup is the **fixed** parameter average of the last five epochs
  (76..80); no valid-based epoch selection.
* Each arm folds its own `delta = median(y_train - p_train)` into the existing
  reader output bias; raw and train-median-calibrated full-1000 valid MAE are
  both reported, and comparisons use calibrated-vs-calibrated.

Automatic purchase gate for the fresh 320-epoch screen (all required):

* candidate calibrated valid `<= 0.111`;
* gain vs the same-round CONTROL calibrated valid `>= 0.004`;
* at least 4 of the 5 fixed `id % 5` valid groups improve;
* descriptive non-`id172` mean gain `>= 0.002`;
* finite values, live training channels, and an estimated fit inside the budget.

No seed / K / s / lambda / dropout-p / field / LR / horizon rescue of a failed
candidate. A negative 80-epoch pilot closes the fixed candidate only.

## 5. Fresh 320-epoch screen (only if it passes)

At most two candidates, ranked by same-round calibrated valid; ties
(`<= 0.0005`) prefer `drop`, then `code`, then `r3`. Fresh seed-0
`ScaleSpec` Full initialisation (not a warm continuation), 320 epochs, batch
128, Adam `lr = 1e-3`, `weight_decay = 1e-5`, `clip = 5`, parent Top-5 soup
protocol. Main result: raw and train-median-calibrated full-soup valid MAE.
`<= 0.110` is worth follow-up confirmation; `<= 0.105` is a strong signal;
`> 0.110` closes the purchase. Single-seed exploratory absolute performance; no
causal claim, no second seed, no Small training, no ensemble/combination.

## 6. Budget and regimes

Total wall clock `<= 4 h`. GPU host `res-2` (Slurm, physical GPU1 regime), at
most two concurrent GPU jobs (Slurm quota `gres/gpu=2`); CPU work may overlap.
Only new code is committed to the research branch; remote checkouts are compute
copies.

## 7. Delivered evidence

`stage0_code.json`, `pilot_<arm>.json`, `screen_<arm>.json`, per-arm prediction
vectors, calibration records, wall clock, parameters, and the runner manifests
with `runtime.device`, GPU model/UUID, torch/CUDA, parent sha, seed and config
hashes. The historical CPU 0.119/0.115 numbers are background only and are
never a matched control for the GPU pilots.