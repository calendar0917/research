# E2E-DictEnv-T1 — implementation note

Round **E2E-DictEnv-T1** · protocol `e2e_dictenv_t1` · study `zinc-context-gap`.
Companion documents: [preregistration](e2e_dictenv_t1_preregistration.md),
[prior-artifact audit](e2e_dictenv_t1_prior_artifact_audit.md),
[analysis](e2e_dictenv_t1_analysis.md).

This note records **how** the pre-registered round was built and executed.  No
specification content is re-opened here; every numeric claim is backed by a
committed result JSON.

---

## 1. Artifacts

```text
module   tracks/ksvd/experiments/luyin16/e2e_dictenv_t1.py        (architecture, PRNG, parameter algebra)
runner   tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_t1.py   (stages, gates, health, freeze, test)
tests    tracks/ksvd/tests/test_e2e_dictenv_t1.py                 (15 CPU tests, no ZINC/GPU)
results  tracks/ksvd/results/e2e_dictenv_t1/
```

Module source sha256 (frozen):
`2915d9657ac6d371cc37c024e93e0e0a171999a41df14b064cd0e7f3ae6e92f6`.

Formal-run commit: `3c13c816cede53e5e03d5933fdada5e908561201`
(implementation commit `6cc5ab19ebd6a9ebde347ae3c2ef36895e595857`, then a
runner-only orchestration change, see §5).

---

## 2. What T1 changes relative to v0

T1 keeps the entire v0 backend (exact `phi65` FSAR-R2-AR0 coordinate, `K = 32`,
`s = 8`, 10 unrolled tied-IHT steps, SDB-v0 K-SVD `D` init, tied encode /
reconstruct, task-coupled `D`, no message passing / recurrence / attention /
pair→centre / relation refresh / typed lookup) and changes exactly three
pre-registered axes:

```text
A  dictionary -> environment interface   (A1 pooled64 / A2 slot48 / A3 slot64)
B  reconstruction/task balance           (lambda_0, 0.5 lambda_0, 0.25 lambda_0)
C  training horizon                      (240, conditional 320)
```

The environment decoder input becomes the spec's exact

```text
z_i = [ x_i^146 ; m_i^D ]
```

where `x_i^146` is the FEC-S0 factorized, train-fit-standardized 146-D coarse
chemistry descriptor read from `data.patch_cont`, and `m_i^D` is the
dictionary→environment code (`dictionary_environment`).  This deliberately
**removes v0's separate 16-D bond marginal `m_E`** (bond chemistry is already in
`x_i^146`'s `bond_shell` (24) + `incident_bonds` (4) blocks), matching the
pre-registered parameter algebra exactly.

Parameter audit (gate G10, `sparse == dense == accounted`):

| cand | mode   | r  | decoder in | decoder hidden | local | whole | vs FEC-S1 |
|------|--------|----|-----------|----------------|-------|-------|-----------|
| A1   | pooled | 64 | 210       | 172            | 50708 | 66067 | −103      |
| A2   | slot   | 48 | 290       | 135            | 50773 | 66132 | −38       |
| A3   | slot   | 64 | 338       | 116            | 50860 | 66219 | +49       |

All three match the pre-registration digit-for-digit.

`x_i^146` identity is re-checked by gate **G0b**
(`fec_s0_factorization.FactorizedFeatureTransform`, train-fit scaler,
`max_abs_diff = 0.0` over 4 molecules / 100 patches); `phi65` identity by **G0a**
(`max_abs_diff = 0.0`).  `D` init identity by **G11** (`D_bit_identical = true`,
`D_init_sha256 = f31350865f3f27064f3490263dd7a59e5b8cb26aef6610ef32bd08e14fe42f09`;
the on-disk SDB-v0 artifact `results/sdb_v0/dictionary.pt` remains
`sha256 = 925d573a58083c3c…`, the value quoted in the pre-registration — the two
hashes differ only in serialization).

---

## 3. Correctness gates

`correctness_stage(device="cpu")` → `all_passed = true`, 32 sub-gates:

```text
G0a phi65 identity                 PASS  (max_abs 0.0)
G0b patch_cont146 identity         PASS  (max_abs 0.0, width 146)
G1  chemistry purity               PASS
G2  exact top-8                    PASS  (max l0 = 8, exact fraction 1.0)
G3  task gradient -> D             PASS  (nonzero, all candidates)
G4  tied reconstruction            PASS
G5  no dense fine-topology bypass  PASS
G6  environment freeze (pair mut.) PASS
G7  no pair -> centre              PASS
G8  once-only composition          PASS
G9  relabel invariance             PASS
G10 parameter identity (a1/a2/a3)  PASS
G11 initialization matching        PASS  (D bit-identical, 0 mismatched tensors)
G12 official-test blocker          PASS  (test access raises; encoded cache 10000/1000)
```

The local + remote CPU suites are `15/15` pass (the 12 original plus three new
tests covering the per-candidate tuning orchestration, §5).

---

## 4. Execution protocol

Local-first: architecture + tests on CPU, then the pre-registered remote A100
protocol (`remote-research-runner` skill), deployed at
`3c13c81`.  Remote host `hxy@a100-2`, repo `/home/hxy/cy/research`.

Frozen training setup (every tuning run): official train 10 000 / official valid
1 000, seed 0, Adam lr 1e-3 / wd 1e-5 / batch 128 / clip 5, no scheduler, L1 +
`lambda_rec · normalized reconstruction`, fixed equal-weight Top-5 soup by
official-valid MAE.  `lambda_0 = 135.83492071463948` reused from v0 (no
recalibration).  Official valid is the development/tuning set; official test is
terminal reporting only.

Budget spent — **6 new full Sparse tuning runs**:

```text
Stage A  a1 (pooled64)  a2 (slot48)  a3 (slot64)        3 runs
Stage B  lambda050      lambda025                       2 runs
Stage C  horizon320 (triggered)                         1 run
-------------------------------------------------------------
plus seed-1 confirmation: final Sparse + final DenseTied  (2 runs, confirmation layer)
```

No K / s / IHT / attention / MP / recurrence / reader / optimizer search.

---

## 5. Parallel scheduling and its one incident (recorded honestly)

The round was launched once as a monolithic `all` job, then stopped before any
training artifact was written in order to split the independent tuning runs
across the two A100s.  The runner was extended (commit `3c13c81`) with
`--only {a1,a2,a3,b1,b2}` and `resolve_stage_a` / `resolve_stage_b` now read the
per-run JSONs directly rather than a shared summary file, so parallel jobs cannot
race.  Paired seed-1 confirmation stays sequential on one GPU.

Cluster state during tuning (observed, not assumed):

```text
GPU0  foreign 37 GB job at ~95-100% utilization   (our a1 shared it)
GPU1  foreign 35 GB job at ~0% utilization         (memory held, compute idle)
```

Two operational facts are recorded so the wall-clock numbers are interpretable:

1. The first `all` launch left an **orphaned `uv run` child** after the session
   leader was killed (the pid file tracked the `setsid` bash leader, not the
   python process).  It was detected via `nvidia-smi` and killed before it could
   write a competing Stage-A file.
2. A3 was first started on the contended GPU0, then killed at ~epoch 60 and
   restarted **from scratch on the idle GPU1** (no partial checkpoint existed;
   only the final per-run JSON is durable).  Its recorded artifact is the GPU1
   run.

Because of (1)/(2), one Stage-A candidate (`a1`) ran on a heavily contended GPU
while `a2`/`a3` ran on the idle GPU.  Wall-clock is therefore **not** a
cross-candidate comparator this round; only the valid-MAE and health contrasts
are used.  All Stage-B/C/confirmation/health/mechanism runs were sequential on
GPU1.

Wall clocks (informational): a1 3140 s, a2 1453 s, a3 2164 s,
lambda050 1609 s, lambda025 1617 s, seed1 sparse 1563 s, seed1 dense 1305 s.

---

## 6. Stages as run

```bash
# correctness (CPU) at the formal commit
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 correct
# Stage A, three independent jobs (a1 GPU0 / a2 GPU1 / a3 GPU1)
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 stage_a --only a1
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 stage_a --only a2
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 stage_a --only a3
# selection (health-gated) then Stage B (both arms, GPU1)
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 select
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 stage_b
# Stage C (triggered), freeze, confirmation, mechanism, health, analyze
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 stage_c
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 finalize
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 freeze
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 confirm
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 mechanism
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 health
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 analyze
# terminal test (reporting only) after the freeze is committed
python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_t1 unlock
```

Stage A selection, Stage B selection and the freeze all read only official-valid
MAE and dictionary health; the official test is loaded exactly once, last, by
`unlock`.

---

## 7. Freeze record (committed before the test read)

```text
config_id                  e2e_dictenv_t1_a2_lam0.25_h240
candidate                  a2 / A2_COARSE146_SLOT48
lambda (effective)         33.95873017865987   (lambda_scale 0.25)
horizon                    240
parameter_count            66132
K / s / IHT                32 / 8 / 10
D_init_sha256              f31350865f3f27064f3490263dd7a59e5b8cb26aef6610ef32bd08e14fe42f09
architecture_source_sha256 2915d9657ac6d371cc37c024e93e0e0a171999a41df14b064cd0e7f3ae6e92f6
seed0_soup_state_sha256    952337b71d5d747e4a5693dadd529ecd6d254bdca72a0c0f9ca489421dba7637
seed0 winner tag           stage_b_lambda025
soup member epochs         [170, 212, 217, 234, 240]
health_all_passed          true
official_test_loaded_at_freeze_time  false (this round)
```

Seed-1 confirmation state hashes are recorded in the same
`architecture_freeze.json`.  The complete JSON (including per-tensor soup
hashes) is in `results/e2e_dictenv_t1/architecture_freeze.json`.
