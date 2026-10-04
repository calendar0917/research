# REPORT — zinc-task-dictionary-and-cycle-witness-seed0-v1

Single seed (seed 0), fresh-only protocol, new model-internal fold, two matched
GPU arms (`D` = current shared task dictionary, `M` = parameter-matched local
MLP) plus a train-only CPU witness task on the old CPU head's `topology25`
input.  No official-valid / official-test access, no 10k confirmation, no
write-back.  Branch `task/zinc-task-dictionary-and-cycle-witness-seed0-v1`,
training revision `77c40dd`.

## 0. Conclusion first

**Main (GPU): the pre-registered gate is `INCONCLUSIVE`; the compressed-body
dictionary bridge shows no generalisation advantage over the matched MLP
bridge, and is clearly worse than the MLP at fitting the severe-ring tail.**

* New-dev overall calibrated MAE: `D 0.122960`, `M 0.117032`;
  gain `MAE(M) - MAE(D) = -0.005928` (negative = `M` better),
  paired-bootstrap CI `[-0.014148, +0.001230]` — crosses zero.
* Primary endpoint new-dev G0 (k=0) calibrated MAE: `D 0.104300`,
  `M 0.103305`; gain `-0.000995`, CI `[-0.006240, +0.003491]` — a tie.
* Raw endpoint: overall gain `-0.003994` CI `[-0.012246, +0.003241]`,
  G0 raw gain `+0.001025` CI `[-0.004064, +0.005583]`.  The overall point
  improvement >= 0.003 favours `M` and has the same raw direction, but it is
  not CI-separated and it is concentrated on 12 severe dev rows, not on G0.
* Fit error says the same thing more strongly: `M` fits the whole fit set
  better (`0.032990` vs `0.039078` calibrated) and the tail much better
  (`k<=-3`: `1.150` vs `3.962`; `k=-2`: `0.229` vs `0.317`).  The current
  16-step/288-atom ISTA dictionary code underfits the frequent-chemistry-to-
  extreme-penalty mapping even in sample.
* Row-level movement is symmetric in count (dev cal: `1009` rows better,
  `991` worse, `max |delta| 4.61`); the mean gain comes from the two dev rows
  at `k<=-3` (`train:0593`, `train:1760`), not from a broad G0 advantage.

**CPU: the "same topology25, different ring targets" samples are real input
aliasing, not a statistics artefact.**

* Old fit has 327 exact `T25` classes; 190 are repeated (the old
  `0.982875` = repeated-row fraction), and exactly **4 classes / 10 rows
  (0.125%) are conflicting** (same exact `T25`, different trusted
  `c`/severity `k`).
* All 4 conflicts are **across severity groups**; `group_only_min_l1 = 0.0`
  for every group — there is no within-severity input conflict on the old fit.
* Witnesses: class 86 (`train:3741` k=-1 vs `train:7491` k=0), class 232
  (`train:0593` k=-5 + 3x k=0), class 250 (`train:2232` k=-2 vs `train:3626`),
  class 259 (`train:1270` k=0 vs `train:1424` k=-6).  Every witness pair is
  non-isomorphic (topology-only and typed).
* Flags: `INPUT_ALIASING_WITNESS` + `INPUT_EQUIVALENT_TARGET_CONFLICT` for all
  four; class 232 additionally `ORDER_DEPENDENT_HELPER` (`train:2447` scores
  cycle excess -5 under the repo stored-order helper, 0 under canonical/GVAE
  order); `UNRESOLVED_PROVENANCE` is empty.
* Node renumbering (16 perms, seed 20261004) changes the stored-order cycle
  helper on 7/8 witness molecules (score moves up to 6, e.g. -6 <-> 0).  This
  localises a class of helper-order sensitivity; it does not prove the
  official labels wrong.

**Corrections to old numbers** (details in `ERRATA.md`): `0.982875` was the
repeated-input row fraction; `5.54946` was the severity-blind class-median
predictor evaluated on the old `k<=-3` rows (`global_opt_group_cost`), not a
group-specific floor (`group_only_min_l1 = 0`); `global_min_l1 = 0.0060697`
reproduced exactly; the old J/M "no reaction" statement is softened to "no net
MAE benefit, per-graph deltas ~0.018 mean / 0.176 max".

## 1. Protocol actually executed

* New fold generated once from `np.random.default_rng(20261004).permutation(10000)`,
  `fit = sort(perm[:8000])`, `dev = sort(perm[8000:])`.
  SHA-256 fit `7bf1cfb856a79617baf83c4ba95290d08248b4633609fa24725a5c028c9096d9`,
  dev `a61c801073fbda237a46e4f8fe6c8e4f8f24e867e7a8b452f7aaffc86a9cceff`.
  Overlap empty, union 10000, stored in `fold_indices.npz`.
* Severity mix (new fold): fit `k=0: 7713, k=-1: 252, k=-2: 30, k<=-3: 5`;
  dev `k=0: 1915, k=-1: 73, k=-2: 10, k<=-3: 2`.
* All input standardizers were refit on the new fit roots only after
  inverting the frozen all-train cache standardizers back to raw values
  (`new_fit_prep.npz`, floor `1e-6`).  No old fitted scaler, `dict_phi`,
  `x175` or KSVD object is used.
* Both arms built by `build_deploy_model(state=None)` and contain no learned
  tensor: no `S_M/A0/C/T/O/H/Y`, no old bridge, no distilled fold bias.
* `D`: current `LatentDictionaryBridge` (`rho`, `x=h/rho`, column-normalized
  `Dbar`, 16 ISTA steps, `lambda1=0.05`, `lambda2=0.01`, `E = rho * alpha @ V_L`).
  `M`: `E = rho * W2 SiLU(W1 x)`, no bias, `W1 = D_L_init.T`,
  `W2 = V_L_init.T`.  Bridge 82,944 parameters each; total 267,611.
* seed 0, 240 epochs, batch 128, Adam `lr=1e-3 wd=1e-5`, clip 5.0, loss exactly
  mean `L1(y)`, soup epochs 236-240, calibration `b = median(y - p_raw)` on fit.
  Shared schedule from `torch.Generator(seed=101)`.
* Paired bootstrap 1000 resamples, seed 20261004, gain `MAE(M) - MAE(D)`.
  Gate thresholds as pre-registered in `PROTOCOL.md` (Delta = 0.003).

## 2. Contract and anchors (all verified before any score was read)

`smoke_checks.json`: `all_ok = true`.  Parameter audit `body = 184,667`,
`bridge = 82,944`, `total = 267,611` for both arms; dictionary and MLP bridge
outputs equal the aux `E` consumed downstream; `y` randomisation changes no
prediction; grouped and per-molecule endpoints agree; one real optimizer step
changes both arms; perturbing the bridge moves the full prediction; tiny arm
training uses an identical schedule.  No official split loaded anywhere
(`official_valid_loaded = false`, `official_test_loaded = false`).

Trained-state checks (`analysis.json`): the saved `D_init_state.pt` and
`M_init_state.pt` share 34 non-bridge tensors with **zero mismatch**
(`shared_init_sha256 = 5a8d8ff042fff496802a64a3266a1c18caff4a132331b154bd6087e5200ba2a6`,
identical to the smoke hash).  Both runs completed `15,120/15,120` steps,
`stopped_reason = completed`, with the same schedule/data-stream hash
`7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65`;
replay max |diff| `1.67e-06` for both.  `contract_errors = []`.

Group contributions are disjoint (`k=0`, `k=-1`, `k=-2`, `k<=-3`) and sum to
the overall MAE; the `k<=-2` row in `group_table.csv` is the cumulative
`k=-2 + k<=-3` convenience view and is not part of the sum.

## 3. Main results (dev, soup state)

| endpoint | D | M | gain (M-D) | 95% CI |
|---|---:|---:|---:|---|
| overall cal | 0.122960 | 0.117032 | -0.005928 | [-0.014148, +0.001230] |
| overall raw | 0.124138 | 0.120144 | -0.003994 | [-0.012246, +0.003241] |
| G0 cal (primary) | 0.104300 | 0.103305 | -0.000995 | [-0.006240, +0.003491] |
| G0 raw | 0.105450 | 0.106475 | +0.001025 | [-0.004064, +0.005583] |
| k=-1 cal | 0.188039 | 0.182923 | -0.005116 | (group n=73) |
| k=-2 cal | 0.527302 | 0.420991 | -0.106311 | (group n=10) |
| k<=-3 cal | 13.593456 | 9.336111 | -4.257345 | (group n=2) |

Contributions to the overall cal MAE (share of the total):

| group | D | M |
|---|---:|---:|
| k=0 | 0.099867 (81.2%) | 0.098914 (84.5%) |
| k=-1 | 0.006863 | 0.006677 |
| k=-2 | 0.002637 | 0.002105 |
| k<=-3 | 0.013593 (11.1%) | 0.009336 (8.0%) |

The overall difference is therefore mostly the two `k<=-3` dev rows plus the
ten `k=-2` rows; G0 itself is a 0.001-scale tie.  Fit shows the same direction
much more strongly:

| endpoint (fit) | D | M |
|---|---:|---:|
| overall cal | 0.039078 | 0.032990 |
| k=-1 cal | 0.059682 | 0.057425 |
| k=-2 cal | 0.317181 | 0.228567 |
| k<=-3 cal | 3.961965 | 1.150320 |

Row movement (soup, cal): fit 4155 rows better / 3845 worse for `M`
(mean delta -0.006088); dev 1009 / 991 (mean delta -0.005928, max |delta|
4.61).  `M` is not broadly better per row; it wins the few large-error rows.
Raw dev has 951 better / 1049 worse.  Pre-registered sensitivity: dropping the
single dev row with the largest combined error (`train:0593`, combined 23.303)
leaves overall cal gain `-0.003626`, overall raw gain `-0.001695`, G0 cal gain
`-0.000995` — conclusion unchanged.

State trajectory (dev cal):

| state | D overall | D G0 | M overall | M G0 |
|---|---:|---:|---:|---:|
| init | 1.632411 | 1.515064 | 1.691666 | 1.582400 |
| last (ep 240) | 0.133442 | 0.112357 | 0.132490 | 0.114989 |
| soup (ep 236-240) | 0.122960 | 0.104300 | 0.117032 | 0.103305 |

Initial function inequality is explicit: at init `D` is better than `M` on
both endpoints and is not claimed away.  Soup helps both arms and helps `M`
more (last -> soup: `D -0.010482` overall / `-0.008057` G0; `M -0.015458`
overall / `-0.011684` G0).

## 4. Mechanism health and initial probe

| quantity | D | M |
|---|---:|---:|
| bridge gradients at init (fixed 128-row fit batch) | D_L 0.511, V_L 0.532 | fc1 0.697, fc2 0.883 |
| initial output RMS | 0.1367 | 0.1585 |
| code non-zero fraction (dev) | 0.911 (262.5/288 atoms) | 1.000 (dense hidden) |
| ISTA reconstruction rel. abs. error | 0.1731 (fit) / 0.1788 (dev) | n/a |
| trainable-change rel. norm | D_L 0.7727, V_L 0.6078 | fc1 0.7975, fc2 0.6876 |
| clip fraction (mean / max) | 0.482 / 0.873 | 0.592 / 0.952 |
| wall clock (15120 steps) | 903.2 s | 600.7 s |

The dictionary is not collapsed (91% of atoms active) and its code is trained;
the MLP's hidden layer is fully dense.  Both bridges receive healthy gradients
and both trained the full budget.  The only qualitative difference is the
`k<=-3`/`k=-2` fit underfit of the ISTA path above.

## 5. CPU witness task (old CPU head, old 8000/2000 internal split)

`cycle_bounds.json` (reproduced exactly): `global_min_l1_per_row =
0.0060697235` (total 48.5578), `global_opt_group_cost[k<=-3] = 5.5494615`,
`group_only_min_l1 = 0.0` for every group, conflict rows 10/8000 = 0.125%,
repeated-row fraction 0.982875, no negative zeros, byte-level classes = exact
classes = 327.

`cycle_class_witnesses.json` enumerates all four conflicting classes with
members, old fit positions, stable ids, `k`, `c`, `T25` hashes, old-fit
audit cycle scores and q_U/q_B traces.  Two of the five old fit `k<=-3` rows
(`train:0593`, `train:1424`) are conflict members; `train:1760`, `train:2347`,
`train:3776` are uniquely determined on the old fit.  The two old dev tail
rows (`train:2210` k=-12, `train:5050` k=-5) have no exact fit match; q_U vs
q_B dev errors are 2.30 -> 28.42 and 12.20 -> 13.60 under the old heads.

`renumber_check.json` (16 permutations, seed 20261004): 7 of 8 witness
molecules change stored-order cycle score under renumbering
(`train:0593` -5 <-> 0, `train:2447` -5 <-> 0, etc.).  The repo helper is
node-insertion-order dependent; the upstream official label pipeline
canonicalizes order, so this is a localisation of helper semantics, not label
error evidence.

`jm_flip_replay.json`: old `R_SJ` dev cal 0.117110 -> 0.117802, `R_DJ`
0.117561 -> 0.117891 with the trained weights fixed and the block flipped;
per-row `|delta_pred|` mean 0.0182 / p95 0.0485 / max 0.176 (R_SJ) and
0.0164 / 0.0451 / 0.116 (R_DJ); ~1000 rows improve and ~1000 worsen.  The
old MAE-level "no reaction" wording is corrected in `ERRATA.md`.

## 6. Deviations, failures and recovery

* Two engineering-only defects in the local analysis path were found and
  fixed before any conclusion was drawn: a bootstrap stratification index
  mismatch (`IndexError`, fixed by bootstrapping the already-subset arrays),
  and a mislabelled `exists` field for the "overall improvement >= 0.003"
  label (now `abs(point) >= Delta` with an explicit `favored` direction).
  Neither touched training code, the deployed revision `77c40dd`, or any
  trained state.
* No GPU job failed; both jobs ran to completion on `c05` (pool
  `res2-cu124`).  The D arm is ~50% slower than M (903 s vs 601 s) because of
  the 16-step ISTA unroll; both were launched in parallel, within the
  2-GPU / 1.2 GPU-hour budget.
* No third arm, no second seed, no hyper-parameter search, no extension of
  the better arm.  The new fold was generated once and never redrawn.

## 7. What is reproducible vs exploratory

Reproducible from committed artifacts: fold indices and hashes; both full
training runs (schedule/data-stream hashes, parameter audit, replay
`1.67e-06`, `{D,M}_*.pt` and `*_raw_predictions.npz`); the analysis
tables/CIs (`analysis.json`, `main_table.csv`, `group_table.csv`,
`gain_table.csv`, `per_graph_fit/dev.csv`, bootstrap self-tests); the CPU
witness numbers; the J/M flip replay.

Exploratory / single-run: every performance delta in section 3 — one seed,
one new internal fold, and effects of the order 0.001-0.006 that sit inside
the fold's bootstrap interval.  The `k<=-3` fit gap is a robust in-sample
observation, but its dev counterpart is only 2 rows.  No official-valid
confirmation exists in this round (by design), so nothing here should be
promoted to the main line on the strength of these numbers alone.

## 8. Next experiments worth buying (and what not to do)

Worth buying, in order: (1) a fresh fold / second seed for the `D` vs `M`
contrast to see whether the `k<=-3` fit gap translates into a CI-separated dev
effect; (2) an ISTA-capacity control (more steps or atoms, parameter-matched
by shrinking `V_L`) to test whether the tail gap is the 16-step/288-atom
coding bottleneck rather than "dictionary vs MLP" as such; (3) only if (1)
and (2) are stable, a 10k confirmation.

Do not: reweight the tail head on the same rows, tune `lambda1/lambda2` or
width on dev, open official-valid to rescue this gate, or average the two
arms' soup states (not pre-registered).

## 9. Budget and data boundaries

Kept: seed 0 only; new 8000/2000 fold fixed before training; exactly two
formal arms; <= 2 parallel A100 GPUs; total trained wall `903.2 + 600.7 =
1503.9 s ≈ 0.42 GPU-hours` (limit 1.2); CPU <= 8 threads for analysis; all
self-created `rr` jobs completed and none left running; official-test never
loaded; official-valid loaded 0 times; old result directories byte-preserved.

Artifacts: `PROTOCOL.md`, `METHOD_CONTRACT.md`, `protocol.json`,
`fold_indices.npz`, `new_fit_prep.npz`, `batch_schedule.npz`,
`smoke_checks.json`, `{D,M}_meta.json`, `{D,M}_*_state.pt`,
`{D,M}_raw_predictions.npz`, `analysis.json`, `gate.json`, `gains.json`,
the CSV tables, `cycle_bounds.{json,csv}`, `cycle_class_witnesses.{json,csv}`,
`witness_graphs.json`, `witness_adjs.csv`, `renumber_check.{json,csv}`,
`jm_flip_replay.{json,py}`, `figures/*.png`, `ERRATA.md`, `EXECUTION.md`,
`manifest.json`, `budget.json`.
