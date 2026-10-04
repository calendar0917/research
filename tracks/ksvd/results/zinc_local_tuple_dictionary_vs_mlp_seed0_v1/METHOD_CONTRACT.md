# METHOD_CONTRACT — `zinc_local_tuple_dictionary_vs_mlp_seed0_v1`

What is actually computed, where, and what is held fixed. Frozen before the M_J formal run.

## 1. Data and frames

* Official-train only: `zftd.load_train_only()` (encoded train cache + env train cache). The
  official-valid loader is never called; `official_valid_loaded=false`,
  `official_test_loaded=false` in every artifact.
* Fold: `zw.build_fold()` → fit 8000 (`7bf1cfb8…`), dev 2000 (`a61c8010…`).
* Target: `g = y − c`, `fit_only_targets.npz` sha256 `e2adf5f2…` (fit-only constants; not refit).
* `phi` scaler, `pair_ptr`, `pair_t`, `pair_a`, `pair_wJ`, `pair_wI`, `root_base`, `root_atom`,
  `kappa` and `kappa_sample` come unchanged from the committed `local_tuple_index.npz`
  (sha256 `4facb6ec…`). No tuple cache or scaler is regenerated.

## 2. The one new block: local encoder M

* New parameter `A_raw[64,125]` = 8,000 floats. `A_bar = row_l2_normalise(A_raw, eps=1e-12)`.
* Per tuple: `f_M(x) = SiLU(x @ A_bar.T)`, `x` the same 125-D vector
  `[phi65_normalised; onehot28(a_v); onehot28(a); onehot4(t)]`.
* Aggregation: `e_M(v) = sum_{t,a} pair_wJ(v,t,a) · f_M(x(v,a,t))` — per-tuple nonlinearity first,
  then real correspondence weights via `index_add_` on the root rows of the concatenated batch.
* Injection: `preact = fusion0(Sem110) + W_loc.(kappa_M · e_M)`; `W_loc[342,64]` remains
  bias-free and exactly zero-initialised; the posterior 144→288→144 MLP bridge, body semantics
  (`Sem110`), static relations, mask (C6), topology and reader are unchanged.
* Initialisation: `A_raw_init = D_loc_raw_init.T` (element-wise copy of the canonical untrained
  frame `prev.init_d_loc()`); no trained state enters. `kappa_M` is a fixed scalar attribute,
  computed once (below). No bias, gain, BatchNorm/LayerNorm, dropout, second layer,
  reconstruction loss or separate value matrix.
* Parameter audit (hard-asserted by name and count, not just total):
  body 184,667 + bridge 82,944 + `A_raw` 8,000 + `W_loc` 21,888 = **297,499**; no idle
  `D_loc_raw` parameter is kept.

## 3. One-shot label-free scale match

On the frozen `kappa_sample` (≤8192 fit roots, seed 20261004; membership in fit verified):

* `r_D = RMS(kappa_D · e_D_init(sample))`, `r_M = RMS(e_M_init(sample))`,
  `kappa_M = r_D / r_M`.
* Only this one global scalar is fitted; no labels, no dev, no per-dimension/dynamic normalisation,
  no multiplier grid. Recorded: sample hash, `r_D`, `r_M`, `kappa_D`, `kappa_M`. A non-finite or
  near-zero value would raise `MECHANISM_INIT_BLOCKED` rather than be rescued.

## 4. Pairing and RNG

* All shared initial tensors (body, bridge, `W_loc`) are byte-identical to the stored
  `J_init_state.pt` (`185db5ed…`); `A_raw_init` equals `D_loc_init.T` byte-exactly; `W_loc = 0`.
* The M builder calls the canonical previous-round constructor (`prev.build_arm("J")`) and only
  swaps the encoder; the replacement consumes no global RNG draws. Post-construction CPU RNG state
  equals the historical one (`rng_state_equal_after_construction=true`), and the schedule/data
  stream hash equals the frozen `7b11a529…`, so the dropout stream is the historical stream.
* `kappa`/sample estimation, health probes, eval loaders and smoke use separate generators or
  store/restore RNG; they never consume the formal training stream. The M encoder itself contains
  no dropout and no runtime randomness.

## 5. Training (copied from the actual previous run)

Adam lr `1e-3`, coupled `weight_decay=1e-5`, global grad clip `5.0`, batch 128, 240 epochs,
15,120 optimizer steps, L1 loss on the same `g`, same frozen per-epoch schedule
(`SEED + 101`, seed 0), soup = mean of epoch-end states 236–240 (raw soup used for comparison;
last state described only). FP32, no AMP/DDP. Fit-only curve and probes (epochs 1/40/120/240:
`A_raw` and `W_loc` norms/task grads, code std/dead dims/RMS, injection RMS, clip fraction) are
taken from the real production functions, not a probe re-implementation.

## 6. Evaluation, statistics, gates

* Each arm's raw soup fit prediction gets exactly one fit-median bias (`b = median(g_fit − p_raw)`),
  then `p_cal = p_raw + b`; raw/cal are both reported. Old D_J/B biases are recomputed from their
  stored raw fit predictions and must reproduce the published values to ≤1e-5.
* `gain_MJ = MAE(D_J) − MAE(M_J)`; 1000 paired bootstrap draws, seed 20261004, one shared index
  set per draw per metric; G0 resampled inside the 1915 G0 rows, overall inside all 2000 rows.
* Classification and PERFORMANCE_SIGNAL exactly as frozen in `PROTOCOL.md`.
* One pre-fixed drop-one-row sensitivity (max mean calibrated error across the two arms);
  group-contribution sum and group-gain sum identities are checked.

## 7. Mechanism (fit-only, no retraining)

1. Operator switch J→I for M_J raw soup (fit/dev: mean|Δpred|, signed mean, raw/cal MAE and G0
   changes), compared side by side with the already-stored D_J switch.
2. Fit-mean replacement for D_J and M_J: μ = mean over fit roots of `kappa·e(v)`; each root with
   at least one neighbour receives μ, roots with d=0 stay zero, bias untouched; fit/dev prediction
   and error changes recorded.
3. Zero-ablation for M_J at soup (interface check), the frozen J/I four-grid decomposition re-read
   and verified as a descriptive `G = O + W` split only.

## 8. Outputs

`PROTOCOL.md`, `METHOD_CONTRACT.md`, `EVIDENCE_SCOPE.md`, `ERRATA.md`, `REPORT.md`, `DECISION.md`,
`EXECUTION.md`, `input_manifest.json`, `historical_anchor_checks.json`, `init_identity.json`,
`operator_path_checks.json`, `kappa_M.json`, `tuple_environment_checks.json`, `smoke_checks.json`,
`M_{init,last,raw_soup}_state.pt`, `M_meta.json`, `M_curve.json`, `M_raw_predictions.npz`,
`analysis.json`, `bootstrap.json`, `gains.json`, `gate.json`, `main_table.csv`, `group_table.csv`,
`per_graph_{fit,dev}.csv`, `mechanism_health.json`, `replay_checks.json`, `budget.json`,
`manifest.json`.
