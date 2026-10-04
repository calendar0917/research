# METHOD_CONTRACT — `zinc_local_tuple_fresh_fold_replication_seed0_v1`

What is actually computed, where, and what is held fixed. Frozen before the two formal runs.

## 1. Data and frames

* Official-train only: `zftd.load_train_only()` (encoded train cache + env train cache). The
  official-valid loader is never called; `official_valid_loaded=false`,
  `official_test_loaded=false` in every artifact.
* Fold: `build_fresh_fold()` — old fold seed `20261004`, then new dev seed `20261005` sampled
  inside old fit; `new_fit = setdiff1d(arange(10000), new_dev)`. Arrays loaded from the frozen
  `fresh_fold.npz` and re-checked against the expected SHA256.
* Target `g = y − c`, constants fitted on new fit rows only. Saved in `fresh_targets.npz`
  (`y, c, g, k, gid, constants, constant_names`). `y` is the train-only handoff `y`.
* Raw train-only label fields come from `train_cycle_audit_label.csv` × `gvae_full_properties.npz`
  × `train.npz`; the mixed 12k verification CSV is never opened. Lineage anchor: refitting on the
  old fit mask reproduces the historical saved constants (`sigma_logP` Δ 6.7e-16, others 0.0) and
  the old `k`/`c` exactly.
* Body prep: `zw.apply_new_fit_prep(train_data, fold_objects_blob, new_fit)`. The blob's
  all-train constants are **inversion only**; the four fit standardizers (`floor=1e-6`) are refit
  on new fit roots and saved in `fresh_prep.npz`. At load time the recomputation must equal the
  frozen arrays element-wise.
* Tuple payload `fresh_tuple_payload.npz`: structure arrays copied from the committed
  `local_tuple_index.npz` (`root_base, node_sizes, pair_ptr, pair_t, pair_a, pair_wJ, pair_wI,
  root_atom`), plus the new-fit `phi_mean/phi_std/phi_scale`, `kappa`, `kappa_sample`. The old
  payload's scalers are never used.
* Phi scaler: fit-realised (`pair_wJ>0`) incident tuples on new-fit roots, `std floor 1e-3`,
  `phi_scale = RMS(z)`. Kappa sample: seed `20261004`, up to 8192 new-fit roots, sorted.
* `kappa_D` from the new payload; `kappa_M` one-shot label-free scale match (see PROTOCOL §3).
  No D_J/I training.

## 2. Models

* `B`: `zw.build_arm("M")` — the original compressed M_g skeleton (body 184,667 + posterior MLP
  bridge 82,944 = 267,611). No local-tuple parameters.
* `M`: `mlpmod.build_arm_mj(fresh_payload, kappa_M=...)` — the original M_J constructor:
  `prev.build_arm("J", payload)` then the encoder is replaced by `LocalTupleEncoderM`
  (`A_raw[64,125]` = 8,000, `W_loc[342,64]` = 21,888, zero-init). Total 297,499.
* Both use `A_raw_init = prev.init_d_loc().T`; shared body/bridge tensors byte-identical;
  post-construction CPU RNG states equal. The M builder consumes no extra global RNG draws.

## 3. Training

Exactly the frozen recipe (PROTOCOL §5): seed 0, Adam lr 1e-3, coupled WD 1e-5, clip 5.0,
batch 128, 240 epochs, 15,120 steps, L1 on `g`, soup = mean of epoch-end states 236–240, C6 mask,
FP32. The training loop starts from `seed_everything(0)` after construction for both arms, so the
dropout stream is shared. The per-epoch position schedule comes from `zw.build_schedule(8000,240,
101)`; the actual global graph-ID sequence is hashed as a separate witness. Probes (epochs
1/40/120/240) record the first-step gradient norms and M health stats; they consume no RNG.
Saved per arm: `{B,M}_init_state.pt`, `{B,M}_last_state.pt`, `{B,M}_raw_soup_state.pt`,
`{B,M}_raw_predictions.npz` (init/last/raw_soup × fit/dev), `{B,M}_meta.json`, `{B,M}_curve.json`.
A CPU/GPU soup replay check must be ≤ 1e-5.

## 4. Evaluation

* One fit-median bias per arm, from its own raw fit predictions; `p_cal = p_raw + b`.
* Main table: fit overall cal, dev overall raw/cal, dev G0 raw/cal, bias.
* Per-k group table with `n`, MAE, contribution `Σ|err|/N_dev`; group-contribution and
  group-gain identities checked.
* `gain = MAE(B) − MAE(M)` on G0 cal (primary), overall cal, G0 raw, overall raw; 1000 paired
  bootstrap draws, seed `20261005`, shared endpoint indices.
* Frozen gate conditions and classification exactly as PROTOCOL §7.
* Pre-fixed drop-worst-row sensitivity; no other deletion rule.
* Mechanism: probe stats + dev zero-ablation of the M local path at the soup; class
  HEALTHY/CHANNEL_COLLAPSED/IMPLEMENTATION_FAILED.
* Replay: independently reload both soup states and compare 128 fit + 128 dev predictions to
  stored values, tolerance 1e-5.

## 5. Boundaries / outputs

No new loss, head, ridge/tree, coverage match, kappa scan, post-hoc weighting or capacity probe.
No seed/fold/config search; no third trajectory. Outputs under
`tracks/ksvd/results/zinc_local_tuple_fresh_fold_replication_seed0_v1/`: PROTOCOL, METHOD_CONTRACT,
EVIDENCE_SCOPE, ERRATA, REPORT, DECISION, EXECUTION; `fresh_*` fold/targets/payload/prep artifacts
and their meta; `fold_meta.json`, `targets_meta.json`, `tuple_payload_meta.json`, `prep_meta.json`;
`fresh_manifest.json`, `operator_path_checks.json`, `smoke_checks.json`, `init_identity.json`
(from `--phase-a` witness), per-arm states/curve/meta/predictions; `analysis.json`,
`bootstrap.json`, `gains.json`, `gate.json`, `main_table.csv`, `group_table.csv`,
`per_graph_dev.csv`, `mechanism_health.json`, `replay_checks.json`, `budget.json`,
`manifest.json`, `input_manifest.json`.
