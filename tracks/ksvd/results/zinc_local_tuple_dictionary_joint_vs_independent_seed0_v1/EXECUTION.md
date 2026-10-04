# EXECUTION — `zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1`

Round start (first tool call): **2026-10-04 12:54:24 CST**.
Budget: ≤180 min wall, compute stop at 150 min, ≤1.2 GPU-h, ≤2 concurrent GPUs, local CPU ≤8 threads.
Branch `task/zinc-local-tuple-dictionary-joint-vs-independent-seed0-v1`, revision **`83d910727fd4`**
(deployed to `res-2`; all remote runs used this exact revision, `git_dirty=false`, `git_diff_hash` clean).
The final commit adds one **analysis-only** patch on top of `83d910727fd4`
(`_root_health`/`_tuple_health` pass full-length probe targets to `make_batch`); it changes no
training code and was applied before any health/gate artifact was produced.

## 1. Local pre-work (CPU, ≤8 threads)

| step | command | result |
|------|---------|--------|
| tuple index | `--build-tuple` | 508,032 union pairs; marginal identity float64 max Δ `0.0`, float32 `6.0e-8`; `local_tuple_index.npz` sha256 `4facb6ec…`; 7.5 s |
| operator checks | `--operator-checks` | `all_ok=true`; raw-vs-env-cache incidence exact (37 molecules / 879 roots); batch/offset/shuffle equivalence 0.0; J/I same-code exact; synthetic star witness `e_I` identical / `e_J` differs; 744 exact-duplicate-`phi` groups, 565 with `C` contrast; fit roots with `C≠C_ind` 38,722/185,204 |
| phase A | `--phase-a` | B reproduction: fit cal 0.029280, dev overall 0.103717, G0 0.101875, max Δ `3.4e-7`; init identity J=I=fresh-M exact (eval), old M init shared keys exact; parameter audit 297,499 |
| CPU smoke | `--smoke` | `all_ok=true` (before GPU smoke) |

Frozen docs `PROTOCOL.md`, `METHOD_CONTRACT.md`, `EVIDENCE_SCOPE.md`, `ERRATA.md` were written and
committed **before** any new dev prediction was produced.

## 2. Remote execution (`res-2`, Slurm, pool `res2-cu124`, 1 GPU / 4 CPU / 32 G per job)

| experiment | run_id | Slurm job | node | started | completed | wall | exit |
|------------|--------|-----------|------|---------|-----------|------|------|
| `tup-smoke` (GPU) | `tup-smoke-20261004-142409-a09a4505` | 55924 | c05 | 14:22:09 | 14:22:58 | 49 s | 0 |
| `tup-J` | `tup-J-20261004-142616-c27d31cb` | 55925 | c05 | 14:24:17 | 14:38:36 | 826.6 s | 0 |
| `tup-I` | `tup-I-20261004-142628-145a0ed0` | 55926 | c05 | 14:24:28 | 14:38:37 | 817.8 s | 0 |

Regime recorded by `rr`: `NVIDIA A100-PCIE-40GB`, driver `525.85.12`, torch `2.5.1+cu124`, CUDA
12.4, python 3.12.14, commit `83d910727fd4`.
**Honest caveat:** both arm jobs were scheduled on node `c05` and both report
`cuda_visible_devices=0` with `slurm_gpus=0` in the job record (this partition exposes GPUs
unmanaged); effective execution may therefore have shared physical GPU 0 rather than using two
distinct GPUs. The two arms ran under identical conditions and in parallel, so the paired
comparison is unaffected; GPU-hours are reported both as two allocations (0.457 h) and as one
shared device (0.229 h) in `budget.json`'s per-arm fields. No other user jobs were touched.

Both arms: `steps_done = 15,120 = steps_expected`; `stopped_reason=completed`; identical
schedule hash `7b11a529…` (frozen) and identical data-stream hash; identical initial `D_loc`
hash; CPU/GPU replay max Δ `1.9e-6` < `1e-5`; J/I `b_raw_soup = −0.016432 / −0.028504`.

## 3. Local analysis (CPU)

`rr pull` of all three run_ids; then `--analyze`, `--replay`, `--mechanism`, `--figures`,
`--budget`, `--manifest`. Analysis used only dev (plus fit for calibration/contributions);
`official_valid_loaded=false`, `official_test_loaded=false` in every artifact.
`--replay` `all_ok=true` (fit/dev ~1e-6). `--mechanism` completed with the interventions and health
tables. Budget: wall at `--budget` call **119 min 5 s** (12:54:24 → 14:53:29), training GPU-hours
0.457 (two allocations), smoke +0.014.

## 4. Incidents found and fixed during pre-work (all before the dev run)

* `root_codes` initially mapped batch rows via molecule count instead of node rows; caught by the
  single-vs-batch operator check, fixed (root-row mapping now uses `data.batch` + `ptr`).
* Target indexing in `make_batch` requires a full per-dataset target array; smoke and mechanism
  initially passed sliced targets, causing out-of-range errors; fixed before any formal run.
* The identity comparison initially compared train-mode forward passes; the encoders contain
  dropout, so two arms can differ by dropout masks even with identical weights. Identity checks are
  now eval-mode; additionally the smoke verifies that both arms consume the **same dropout mask
  stream** from the same seed (true).
* `same_phi_different_C` witness rewritten to a deterministic byte-grouping (no Python `hash()`).
* The committed `local_tuple_index.npz` was force-added (`*.npz` is gitignored) so the deployed
  revision carries the frozen index.

## 5. Artifacts

All under `tracks/ksvd/results/zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1/`
(`PROTOCOL.md`, `METHOD_CONTRACT.md`, `EVIDENCE_SCOPE.md`, `ERRATA.md`, this file, `REPORT.md`,
`DECISION.md`; `local_tuple_index.npz`, `tuple_index_meta.json`, `operator_checks.json`,
`phase_a.json`, `historical_anchor_checks.json`, `init_identity.json`,
`tuple_environment_checks.json`, `input_manifest.json`, `smoke_checks.json`;
`J_*`/`I_*` state/meta/curve/predictions; `analysis.json`, `bootstrap.json`, `gains.json`,
`gate.json`, `paired_gains.*`, `main_table.csv`, `group_table.csv`, `per_graph_*.csv`,
`contributions` tables in `analysis.json`, `mechanism_health.json`, `replay_checks.json`,
`figures/*`, `budget.json`, `manifest.json`).

## Addendum — post-round merge/push (operator instruction)

The round was executed and committed on its isolated branch under the
original "no push / no merge" constraint.  After the round closed, the
operator instructed "merge to main, then push".  On 2026-10-04 (CST)
`task/zinc-local-tuple-dictionary-joint-vs-independent-seed0-v1` (`8995109`)
was merged into `main` with `--no-ff` (merge commit `ac4dfea`) and pushed to
`origin/main`.  No result file, model state, prediction, metric or gate
changed; the research decision is unaffected (no write-back into the model
pipeline) — this is a code/record merge only.  `manifest.json` was refreshed
after this addendum.
