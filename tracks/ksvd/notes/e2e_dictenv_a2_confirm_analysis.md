# E2E-DictEnv-A2-Confirm — analysis (frozen 320-epoch OMP pairing confirmation)

Round `E2E-DictEnv-A2-Confirm`, protocol `e2e_dictenv_a2_confirm`.
Preregistration `tracks/ksvd/notes/e2e_dictenv_a2_confirm_preregistration.md`
(rules commit `4b9e206`), implementation `45d53df`, single-arm stage `a9d3e08`,
stage-record fix `76f65af`.  All CUDA on physical **GPU1** only
(`CUDA_VISIBLE_DEVICES=1`), seed 0, official ZINC **test never loaded**.

Frozen verdict: **`ATTRIBUTED_PAIRING_SUPPORTED_BUT_NOT_COMPETITIVE` (case C)**

```
G_pair_320    = MAE(INDEP-OMP-320) - MAE(REAL-OMP-320) = +0.005566724489443009  >= 0.003  (material)
Delta_vs_TOPO = MAE(REAL-OMP-320)  - MAE(TOPO-OMP-320) = +0.010168869751389115  >  0.003  (not competitive)
```

---

## 1. Q1 — exact resume was impossible, so both arms ran fresh matched 320

`continuation_mode.json` (written before any training) records the check, not an
assumption.  The completed 160-epoch lite segments persist only the best model
state and the Top-5 soup **average**:

| required for exact continuation | present in the lite 160-epoch segments |
|---|---|
| model state | yes (best state + soup average, 51 tensors each) |
| optimizer state | **no** |
| scheduler state | **no** (the trainer builds no scheduler) |
| current epoch | yes (curve reaches 160) |
| RNG states (python/torch/CUDA) | **no** |
| loader / batch-order state | **no** (rebuilt from `SEED + offset` per call) |
| soup-member states for a 1–320 Top-5 soup | **no** (only their average is stored; members cannot be recovered) |

The frozen trainer confirms this statically: `accepts_resume_argument = false`,
`writes_optimizer_state = false`, loop starts at epoch 1.  Exact continuation
therefore could not be proven for either arm, and the preregistration forbids
pseudo-resume, so **both arms restarted as fresh matched 320-epoch runs** — the
same regime as the parent `TOPO-OMP` arm, with no mixing (the runner refuses the
resume mode outright).  Cost: 41.4 min (REAL) + 40.9 min (INDEP) of training; the
two arms ran under the **user-authorised parallel GPU1 schedule** (see §6).

## 2. Frozen-arm results

| arm | soup valid MAE | best valid MAE (epoch) | soup members | wall |
|---|---:|---|---|---:|
| `ATTR-REAL-OMP-320` | **0.1369818167760386** | 0.14135768095491222 (290) | [266, 287, 288, 290, 303] | 2481.7 s |
| `ATTR-INDEP-OMP-320` | **0.1425485412654816** | 0.14588125765224685 (319) | [304, 311, 315, 317, 319] | 2453.4 s |
| `ATTR-TOPO-OMP-320` (completed parent arm, never retrained) | 0.12681294702464949 | 0.13136472144449363 (314) | — | 2255.3 s |

Every arm artifact records `horizon = 320`, `frozen_dictionary = true`, its
frozen dictionary sha (`REAL c1cafb08…`, `INDEP 400821ee…`, verified live by the
verify stage), `official_test_loaded = false`, and the continuation mode.

## 3. Primary result: the pairing signal is real and material at the parent horizon

```
G_pair_320 = +0.005566724489443009   (frozen material bar 0.003)   -> PASS
```

The lite screen's positive direction **survived the full horizon**: the gap moved
by only `+5.56e-6` between 160 and 320 epochs while both arms improved by almost
identical amounts (REAL soup `−0.017389`, INDEP soup `−0.017384`), i.e. the
paired gap was **KEPT**, not a short-horizon artefact, and did not reverse.

Within the *same* 32-D frozen OMP bottleneck and the same matched protocol, the
real structure↔attribute assignment therefore carries material predictive value
over the assignment-independent marginal control.

## 4. But it is not competitive with the completed topology-only route

```
Delta_vs_TOPO = +0.010168869751389115  >  practical tolerance 0.003   -> NOT competitive
```

The preregistered Case-C reading applies: the pairing carries task information,
yet placing it inside the current attributed dictionary / code formation does not
form a better overall representation than the topology-only dictionary.  Per the
frozen preregistration this round **does not** proceed to IHT qualification,
task-coupled E2E, mechanism or specificity work.

Structural context (already established by A1/A2, not re-derived here): the frozen
exact-OMP code reconstructs INDEP's 433-D object almost exactly
(`train_rec 2.119e-02`) and TOPO's essentially exactly (`1.41e-05`), while the
REAL object carries a large residual (`1.965e-01`).  The comparator therefore
enjoys a near-lossless code while REAL pays a ~20 % reconstruction residual — the
mechanism-level reason a real attributed dictionary can still lose to
`topology sparse code → post-code chemistry binding` even when its *assignment*
demonstrably contains more task information.  This is a mechanism finding about
where the chemistry should enter, not a refutation of the pairing signal.

## 5. Diagnostics (reported, never gates)

Paired late window 241–320 of `delta(e) = valid_MAE_INDEP(e) − valid_MAE_REAL(e)`:

| statistic | value |
|---|---:|
| mean delta | +0.006538768048032944 |
| median delta | +0.006299895591568197 |
| positive fraction | 0.7625 |
| first (epoch 241) | +0.0018247949490323712 |
| last (epoch 320) | +0.021618235073052328 |
| min / max | −0.04480428454838695 / +0.04751586554106327 |

160 → 320:

| quantity | at 160 (lite) | at 320 (confirm) | change |
|---|---:|---:|---:|
| `REAL` soup | 0.15437116196932038 | 0.1369818167760386 | −0.01738934519328178 |
| `INDEP` soup | 0.15993232336913935 | 0.1425485412654816 | −0.017383782103657736 |
| `G_pair` | +0.005561161399818965 | +0.005566724489443009 | +5.563089624044393e-06 (KEPT) |

The direction is also *more* consistent late in training than it was at the 160
screen (76.25 % of the last 80 epochs favour REAL vs 65 % of epochs 121–160 in the
lite run), and the gap widens at the very end (last-epoch delta +0.0216) — but the
per-epoch spread is large (min −0.0448), so the honest statement is "a stable,
material soup gap carried by the later epochs", not "REAL dominates every
checkpoint".

## 6. Schedule honesty (what actually ran, and what it does not change)

The user authorised running the two arms concurrently on GPU1.  What happened:

* `12:07:19` sequential job (`a2confirm-all`, commit `45d53df`) → verify,
  continuation, then `ATTR-REAL-OMP-320` (finished `12:49:17`, soup 0.136982);
* `12:11:30` single-arm job (`a2confirm-indep`, commit `a9d3e08`) →
  `ATTR-INDEP-OMP-320` (finished `12:52:40`, soup 0.142549), running concurrently
  with REAL for ~38 of its 41 minutes;
* the sequential job was stopped at the REAL→INDEP boundary (`12:49:20`) so it
  could not duplicate the INDEP arm; `decision` and `report` then ran in the
  foreground (CPU-only).  No partial checkpoints exist, and no CUDA process was
  ever started on GPU0 (foreign-occupied, 37.2 GiB).

Why this is a scheduling-only deviation from the preregistration's letter
("one CUDA process at a time"): each arm is an independent single-process job with
the identical frozen protocol (seed 0, matched init, batch order
`SEED+91011`/`SEED+91012`, frozen dictionary and exact-OMP codes, horizon 320,
deterministic algorithms).  The two jobs share only read-only frozen artifacts and
touch disjoint output tags; there is no DDP, no gradient sharing, no cross-arm
communication and no time-dependent logic.  GPU cost of sharing: 844 MiB + 842 MiB
on a 40 GB device.  The parallel schedule is recorded in `stage_status.json`
(`arm_schedules`, `schedule_note`) and in each arm artifact where the field existed
at run time (`INDEP`); `REAL`'s artifact was written by the sequential job at
commit `45d53df`, before the schedule field existed, and is recorded as such.

Wall clock: ~42 min of GPU1 time for both arms in parallel (~82 min sequential).

## 7. Independent local recomputation

`uv run python tracks/ksvd/code/analyze_e2e_dictenv_a2_confirm.py --write`
recomputes every gate, sign convention, diagnostic and provenance flag from the
pulled artifacts alone (own re-implementation of the decision matrix, late-window
statistics, 160→320 change, dictionary/horizon/test-blocker checks):
**43 checks, 0 errors**, verdict reproduced exactly.

## 8. What is *not* claimed

* No claim that REAL beats TOPO, or that the attributed route should replace the
  topology-only route — it does not, on this evidence.
* No mechanism attribution: no code-pairing-removal, node-only or edge-only
  intervention and no liveness audit was run, so *which part* of the REAL object
  carries the +0.00557 is untested.
* No second seed: the reported gap has no seed-level error bar (the frozen seed
  set for this line is seed 0).
* No claim about rank/capacity trade-offs (no PCA32, K64 or s12 work here) and no
  official-test number of any kind.

## 9. Deferred (never auto-started)

seed 1; PCA32 dense control; continuity-v2; IHT-10/30/100/200 qualification;
task-coupled E2E sparse dictionary (E0/E1/E2); REAL→INDEP code-pairing-removal
mechanism; node-only / edge-only interventions; DenseTied specificity; TOPO
retraining; K64; s12; official test; any architecture or hyper-parameter sweep —
all `DEFERRED_PENDING_USER_AUTHORIZATION`.

## 10. Reproduction

```bash
# remote (GPU1), commit 45d53df / a9d3e08 / 76f65af
bash scripts/launch_remote.sh 1 a2confirm-all python -m \
  tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_a2_confirm all --device cuda
bash scripts/launch_remote.sh 1 a2confirm-indep python -m \
  tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_a2_confirm arm --arm INDEP --device cuda
bash scripts/run_remote.sh 1 python -m \
  tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_a2_confirm decision --device cpu
bash scripts/run_remote.sh 1 python -m \
  tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_a2_confirm report --device cpu

# local
bash scripts/pull_results.sh tracks/ksvd/results/e2e_dictenv_a2_confirm
uv run python tracks/ksvd/code/analyze_e2e_dictenv_a2_confirm.py --write
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_a2_confirm.py
```
