# EXECUTION — `zinc_local_tuple_dictionary_vs_mlp_seed0_v1`

Round start (first tool call): **2026-10-04 19:20:38 CST**. Budget: ≤120 min wall, compute stop at
110 min, ≤0.6 GPU-h total, ≤1 concurrent GPU, local CPU ≤8 threads per child process.
Branch `task/zinc-local-tuple-dictionary-vs-mlp-seed0-v1`; the formal training ran on the deployed
revision **`9af8e92713b9`** (`git_dirty=false`). One **analysis-only** follow-up commit
`691e3d095272` (uniform fit-mean replacement patch + full-length probe targets) was made after the
formal run had started and touches **no** training-path code (`build_arm_mj`, `train_arm_mj`,
`phase_a`, `run_smoke` unchanged; `git diff 9af8e927 691e3d0` confined to mechanism helpers).

## 1. Pre-work (CPU, ≤8 threads)

| step | command | result |
|------|---------|--------|
| phase A | `--phase-a --device cpu` | 8.0 s; all anchors/identity checks pass (below) |
| smoke | `--smoke --device cpu` | `all_ok=true`; one fit-only smoke, ≤10 steps (3), model/optimizer/RNG discarded |

Key pre-run evidence (`phase_a.json`, `init_identity.json`, `historical_anchor_checks.json`,
`operator_path_checks.json`, `kappa_M.json`):

* B recomputed: dev overall cal `0.103717336`, G0 `0.101874977` (Δ ≤ 3.9e-9).
* D_J recomputed: fit overall cal `0.028155213`, dev overall `0.101907112`, G0 `0.100416242`
  (Δ ≤ 2.6e-9); bias recomputed `−0.016431744` identical to meta.
* Init pairing: stored `J_init_state.pt` reproduced (`hash 185db5ed…`), shared tensors byte-equal
  (max Δ `0.0`), `A_raw_init = D_loc_raw_init.T` (max Δ `0.0`), `W_loc` exactly zero,
  parameter audit `184,667 + 82,944 + 8,000 + 21,888 = 297,499`, initial function equal
  (Δ `0.0` vs J and fresh reference), post-construction CPU RNG state equal to the J builder.
* `kappa_M`: `kappa_D=2.077404`, `r_D=1.000210`, `r_M=0.495900`, `kappa_M=2.016961`; 8192/8192
  sample roots in fit, sample sha256 `a2c36229…`.
* Operator path: model `root_codes` vs `compose_root_codes_m` max Δ `0.0` (170 roots); naive
  per-pair Python reference max Δ `1.9e-6`; single-vs-batch offsets `2.5e-7`.
* Smoke witnesses: dropout RNG stream identical to the J frame, label permutation Δ `0.0`,
  `W_loc` grad nonzero (2.11) at step 1, `A_raw` grad zero at step 1 then `2.0e-3` after a
  non-zero `W_loc`, tuple code equals `SiLU(x@A_bar.T)` exactly.
* Frozen schedule/data-stream sha256 `7b11a529…` verified before launch.

## 2. Formal trajectory (single arm, `res-2`, Slurm, pool `res2-cu124`)

| field | value |
|---|---|
| experiment / run_id | `mj-M` / `mj-M-20261004-193505-0f27d47f` |
| Slurm job / node | `55937` / `c05` |
| allocation (probe) | `TRES=cpu=4,node=1,billing=4,gres/gpu=1`, `TresPerNode=gres:gpu:1` |
| GPU | A100-PCIE-40GB, UUID `GPU-c7067be6-0979-516e-45c4-27f69628e1df`, PCI `00000000:1D:00.0` |
| driver / torch / CUDA / python | `525.85.12` / `2.5.1+cu124` / 12.4 / 3.12.14 |
| commit | `9af8e92713b9` (clean) |
| start / end | 19:33:06 / 19:44:26 CST (RunTime 00:11:20, exit 0) |
| steps | 15,120 / 15,120, `completed` |
| schedule / data stream sha256 | `7b11a529…` / `7b11a529…` (frozen) |
| calibration bias (fit median) | `−0.015250` |
| replay (CPU/GPU, 128 fit + 128 dev) | `max 1.19e-6` ≤ 1e-5 |

GPU allocation evidence: `gpu_allocation_probe.txt` (from `rr` probe job
`mj-gpu-probe-20261004-195026-ac0d0335` on c05 after the training job; `scontrol show job 55937`
reports `JobState=COMPLETED`, `gres/gpu=1`). The earlier runner record `slurm_gpus=0` is an
environment-variable artifact of this partition; the Slurm TRES record and the probe job
demonstrate a real single-GPU allocation. Peak GPU concurrency was 1 (probe ran after training).

## 3. Local analysis (CPU)

`--analyze`, `--mechanism`, `--replay`, `--budget`, `--manifest`. Analysis used only dev plus fit
for calibration/contributions; `official_valid_loaded=false`, `official_test_loaded=false` in all
artifacts. Bootstrap witnesses pass exactly (same predictions → 0; swap mirror; constant shift).

* Frozen classification: **INCONCLUSIVE**. `MLP_LOCAL_SUPPORT` fails on both the `≥ +0.003` point
  threshold (G0 cal `+0.002200`) and the CI-lower condition `[−0.001564, +0.006029]`;
  `DICT_LOCAL_SUPPORT` fails (wrong direction); `LOCAL_EQUIVALENCE` fails (CIs wider than
  ±0.003); `MLP_MECHANISM_FAILED` does not apply (A drift 0.72, `W_loc` norm 7.62, zeroing the
  local path changes dev cal MAE 0.0994 → 0.6229).
* PERFORMANCE_SIGNAL vs B: 4 of 5 conditions pass (G0 cal `+0.003659`, overall cal `+0.004336`,
  G0 raw `+0.006817`, overall raw `+0.007542`, all positive); only `G0 cal CI_lower > 0` fails
  (`−0.000438`). Not a frozen performance candidate.
* Sensitivity (pre-fixed max mean-error row, dev pos 1632, gid 8049, in G0): G0 cal gain
  `+0.001915`, overall cal gain `+0.002253`, both still positive.

## 4. Incidents

* One bug found and fixed locally **before** the formal run: the naive reference used
  `phi[root].view(1)` for a 65-vector (fixed to `.view(1,-1)`); no training effect.
* One bug found and fixed **after** the formal run and before any mechanism artifact:
  D_J's previous-round encoder has no `mean_replace` attribute, so the mean-replacement
  intervention would have silently been a no-op for D_J; replaced by a shared per-row
  `root_codes` patch (`_patch_mean_replace`), tested on D_J against stored predictions
  (reload Δ 1.4e-6, restore Δ 0.0) before producing `mechanism_health.json`.
* One **analysis-only enrichment** after the first mechanism pass: raw (uncalibrated) MAE
  changes were added to the switch/mean-replacement fields and the D_J operator switch was
  recomputed on CPU with the identical procedure (it reproduces the previous round to 5.1e-9).
  No training-path code changed (`build_arm_mj`, `train_arm_mj`, `phase_a`, `run_smoke`
  untouched across all three commits).
* No formal-run failure, no recovery, no failed GPU attempt.

## 5. Budget

* Wall clock at `--budget`: ~50 min (19:20:38 → ~20:11 CST); compute stopped well before 110 min.
* GPU: training 0.181 GPU-h (651.7 s) + probe ~0.008 GPU-h ≈ **0.19 GPU-h** (limit 0.6).
* CPU: all local children `torch.set_num_threads(4 or 8)`; caches/tuple index reused read-only.
* Smoke: 1 fit-only local CPU smoke (3 steps); models/optimizers/RNG discarded; not used in the
  formal init or stream.
* Remote jobs at round end: `mj-M` completed, `mj-gpu-probe` completed; nothing running/pending.

## 6. Artifacts

All under `tracks/ksvd/results/zinc_local_tuple_dictionary_vs_mlp_seed0_v1/`; hashes in
`manifest.json`, inputs in `input_manifest.json`, remote allocation in
`gpu_allocation_probe.txt`. Old D_J/B results were read-only by hash/stable id; no old report or
state was edited.

## Addendum — post-round merge/push (operator instruction)

The round was executed and committed on its isolated branch under the original
"no push / no merge" constraint.  After the round closed, the operator
instructed "merge to main, then push".  On 2026-10-04 (CST)
`task/zinc-local-tuple-dictionary-vs-mlp-seed0-v1` (`8119059`) was merged into
`main` with `--no-ff` (merge commit `e40836f`) and pushed to `origin/main`.
No result file, model state, prediction, metric or gate changed; the research
decision is unaffected (no write-back into the model pipeline) — this is a
code/record merge only.  `manifest.json` was refreshed after this addendum.
