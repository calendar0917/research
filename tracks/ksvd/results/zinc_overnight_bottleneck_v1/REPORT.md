# zinc_overnight_bottleneck_v1 — overnight run report

**Protocol** `zinc-overnight-bottleneck-v1` (`test_policy: terminal`).
**Branch** `task/zinc-overnight-bottleneck-v1` @ `8657343edc6218087286b55ee6cd51dabcafd78a`
(no push, no merge). **Runner** `zinc_overnight_bottleneck_v1` (registered; launched only through
`uv run research run` on `res-2`, pool `res2-cu124`, Slurm, 1×A100-PCIE-40GB, 8 CPU per run).
**Authorised compute window** start `2026-10-02T16:43:29Z`; new compute stopped `2026-10-02T23:16:27Z` (6h33m);
hard deadline `2026-10-03T00:43:29Z`. **GPU budget used ≈ 4.9 GPU-h** (limit 8 GPU-h, ≤2 GPUs).

> **Official test was never instantiated, loaded, read or evaluated.** Every run record in this report
> carries `official_test_loaded: false` and `test_access=blocked` (protocol `test_policy: terminal`).
> The official **valid** split is used only as the exploration/evaluation set, as pre-registered.

---

## 0. Headline

| question | answer |
|---|---|
| Did any arm reach the ≥0.09 calibrated-valid target? | **No.** Best arm is the control itself (0.111206). |
| Did any pre-registered arm pass its frozen performance gate? | **No.** All five intervention arms have a *negative* calibrated gain. |
| Was the node-channel collapse located and explained? | **Yes** — and it was shown to be *causally irrelevant* to valid MAE. |
| Outcome class | **FAIL** on the 0.09 target; **DIAGNOSIS DELIVERED** on the bottleneck question. |

The overnight run therefore *falsifies* the working hypothesis it was launched to test: the ZINC
node-channel collapse is **not** the bottleneck that holds the model at 0.109–0.111 calibrated MAE.
Rescuing the channel is possible (two independent single-factor rescues were demonstrated) but it does
not buy accuracy — it slightly *costs* accuracy, because the recovered node channel is used mainly to
fit training rows that the 965-row bulk group G0 already fits well.

---

## 1. Setup, provenance and guardrails

| item | value |
|---|---|
| dataset | ZINC control (shared structure dictionary + structure–semantics binding + local-environment statistics) |
| split fingerprint | `58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a` (train 10000 / valid 1000 / test 1000) |
| metric | `b = median_train(y − pred_raw)`, `pred_cal = pred_raw + b`, MAE on official valid |
| groups (frozen) | G0 = 965 rows (cycle penalty 0), G1 = 34 rows (penalty −1 = 30, −2 = 4), G172 = row `valid:0172` id 172 (penalty −6) |
| severity strata | `penalty_0` n=965, `penalty_-1` n=30, `penalty_le_-2` n=5, `penalty_any_neg` n=35 |
| canonical init hash | `93c2f23c39b1654d720d4d7be04a008c25c163a3c21aef682b7250c3be5ac396` (shared by N0/N1/N2/N3 seed 0) |
| frozen node scale | `kappa = 4050.2396551095235` (`frozen_node_scale_seed0.json`); GPU recompute 4050.2394974060226 → rel. diff 3.9e-8, accepted at tol 1e-5 |
| recipe | LR 1e-3, WD 1e-5, batch 128, grad-clip 5, 240 epochs, last-5 epoch soup {236..240}, diag epochs {0,1,10,20,40,80,160,240} |
| checks stage | `all_passed=true` (node arms share canonical init; kappa=1 ≡ product; additive branch ≡ canonical reader on the first 806 columns; grads reach other/tail/topo/topology_encoder; zero-mask zeroes) |
| local resume equivalence | continuous 4-epoch vs 2+2 resumed → identical final state hash `a04f08973416c427a139…` (MATCH) |

Run provenance (all `git_dirty: false`, `backend: slurm`, `node: c05`, driver 525.85.12, torch 2.5.1+cu124):

| arm dir | rr experiment | run id | stage | epochs | wall |
|---|---|---|---|---|---|
| `N0_s0_prefix` | `zinc-ob2-N0-prefix-s0` | `…20261003-044054-eb26958d` | train_prefix | 80 | 12.9 m |
| `N1_s0_prefix` | `zinc-ob2-N1-prefix-s0` | `…20261003-044109-7dcd332a` | train_prefix | 80 | 12.9 m |
| `N2_s0_prefix` | `zinc-ob2-N2-prefix-s0` | `…20261003-044121-05e3452d` | train_prefix | 80 | 12.7 m |
| `N3_s0_prefix` | `zinc-ob2-N3-prefix-s0` | `…20261003-044135-b9518e6a` | train_prefix | 80 | 12.9 m |
| `N0_s0` | `zinc-ob3-N0-full-s0` | `…20261003-051306-0a41c0e9` | train | 240 | 40.2 m |
| `N1_s0` | `zinc-ob3-N1-full-s0` | `…20261003-051320-d858bcb0` | train | 240 | 40.0 m |
| `N2_s0` | `zinc-ob3-N2-full-s0` | `…20261003-063311-09dc4821` | train | 240 | 40.3 m |
| `N3_s0` | `zinc-ob3-N3-full-s0` | `…20261003-063323-f81967f9` | train | 240 | 40.4 m |
| `T0_s0` | `zinc-ob2-T0-full-s0` | `…20261003-044204-7fda9c9c` | train | 240 | 40.7 m |
| `N0_s1` | `zinc-ob3-N0-full-s1` | `…20261003-054200-95586867` | train | 240 | 40.2 m |

The raw `meta.json` for each run is vendored next to the results (`run_meta/`), including
`git_commit`, `git_diff_hash`, `requested` resources, `runtime` (node, driver, GPU names).

Execution regime: driven through the `remote-research-runner` skill
(`~/.pi/agent/skills/remote-research-runner/SKILL.md`, incl. `references/execution-regimes.md` and
`references/gpu-migration.md`) — remote control plane `rr` only (no manual ssh/sbatch), host `res-2`,
pool `res2-cu124`, Slurm backend, offline `uv` on the air-gapped cluster, `rr doctor` 10 ok / 4 warn /
0 fail before launch.

---

## 2. Direction N — fixed 2×2 node-collapse diagnosis

**Factor A (amplitude):** original product binding (node slot RMS ≈ 2.4e-4 at init) vs frozen unit
scale `kappa = 1/r_init` (node slot RMS = 1).
**Factor B (node-specific weight decay):** original 1e-5 on `W_A_S`, `W_A_C`, `node_encoder.*`
vs 0 on exactly those tensors (fusion/reader WD untouched).

Node-health gate (two fixed **train** sentinel batches, eval mode, no dropout → RNG stream untouched):

| arm | amplitude | node WD | slot RMS ep0 → ep80 | healthy probes | `node_out` alive at ep80 |
|---|---|---|---|---|---|
| **N0** | original (2.42e-4) | 1e-5 | 2.418e-4 → **0.0** | 3/6 | **False** |
| **N1** | unit (0.979) | 1e-5 | 0.979 → 1.362 | **6/6** | **True** |
| **N2** | original (2.42e-4) | **0** | 2.418e-4 → 2.045e-3 (≈8× growth) | **6/6** | **True** |
| **N3** | unit (0.979) | **0** | 0.979 → 1.361 | **6/6** | **True** |

**Finding N-1 (mechanism).** Collapse requires the **conjunction** of (i) the tiny initial node-slot
amplitude and (ii) node-specific weight decay. Either factor alone prevents it: the unit scale keeps
the channel alive despite WD 1e-5, and switching node WD off lets the original tiny channel grow ~8×.
This is a clean 2×2 interaction, not a single-factor effect.

**Finding N-2 (frozen selection).** By the pre-registered priority (`N1` if healthy, else `N2`, else
`N3`, else close direction), **N1 is the selected node candidate**. N2/N3 were run to 240 epochs as
the remaining pre-registered arms of the same 2×2, to complete the diagnosis at the reporting horizon.

Full 240-epoch trajectories:

| arm | amplitude | node WD | cal valid MAE | raw valid | fitted bias | Δ vs `N0_s0` | Δ without id172 | G0 contrib. worsening | N-gate |
|---|---|---|---|---|---|---|---|---|---|
| `N0_s0` | original | 1e-5 | **0.111206** | 0.110736 | −0.012043 | — (control) | — | — | — |
| `N1_s0` | unit | 1e-5 | 0.112710 | 0.112947 | −0.016348 | **−0.001504** | −0.000310 | +0.003393 | **FAIL** |
| `N2_s0` | original | 0 | 0.112413 | 0.113879 | −0.019978 | **−0.001207** | −0.001250 | +0.002210 | **FAIL** |
| `N3_s0` | unit | 0 | 0.121682 | 0.121656 | −0.003313 | **−0.010476** | −0.009130 | +0.011310 | **FAIL** |
| `N0_s1` | original | 1e-5 | 0.114080 | 0.115257 | +0.013867 | +0.002874 *(seed spread)* | — | — | — |

Frozen N-gate = `cal gain ≥ +0.003` **and** `gain without id172 > 0` **and** `G0 contribution worsening ≤ 0.001`.
No node arm passes any of the three clauses. **Direction N performance: FAIL.**

**Finding N-3 (the decisive negative result).** Rescuing the collapsed node channel does not improve
calibrated valid MAE; every rescue is slightly *worse* than the collapsing control. The reason is
visible in the train/valid split and in the group decomposition:

| arm | raw train MAE | cal train MAE | eval gap | G0 MAE (965) | G1 MAE (34) | G172 MAE |
|---|---|---|---|---|---|---|
| `N0_s0` | 0.038990 | 0.037360 | 0.073846 | 0.087130 | 0.247161 | 18.7222 |
| `N1_s0` | **0.037299** | **0.034399** | **0.078311** | 0.090646 | **0.156479** | 19.9165 |
| `N2_s0` | 0.039642 | 0.035565 | 0.076848 | 0.089420 | 0.218872 | 18.6802 |
| `N3_s0` | **0.033356** | 0.033240 | **0.088442** | 0.098850 | 0.182762 | 20.0772 |

Rescuing the node channel *helps the hard rows* (G1: 0.247 → 0.156 for N1) but *hurts the bulk rows*
(G0: 0.0871 → 0.0906) by more, and it raises the train/valid gap monotonically with the strength of
the rescue (0.0738 → 0.0783 → 0.0884). In other words the recovered channel is **capacity that overfits**;
the aggregate MAE is dominated by the 965 easy rows (G0 contributes ≈0.084 of the 0.111 MAE), so the
net effect is negative. The control's own seed-to-seed spread is **0.002874** (N0 seed 0 vs seed 1),
which is *larger* than the magnitude of the best node gain (−0.001207): no node arm separates from the
control beyond seed noise.

## 3. Direction T — additive topology readout

`T0` replaces the reader with `pred = GenericReader_other(R[:, :806]) + Linear(8, 1)(R[:, 806:814])`,
so the explicit topology contribution is no longer gated by the other coordinates (408 348 params,
303 fewer than the control; head seed 0). This is a *single* pre-registered arm, as specified.

**Mechanism verification** (`t0_mechanism.json`, `t0_mechanism_check.py`, on the frozen `T0_s0` soup):

| check | value |
|---|---|
| prediction change under a topology swap vs the scalar head delta alone | max abs residual **5.96e-8** (rel. 9.6e-7) — additive identity holds |
| same fixed-T swap under 8 different `other` vectors: delta spread, **additive** reader | **1.19e-7** (identical across `other`) |
| same fixed-T swap under 8 different `other` vectors: delta spread, **shared** reader | **0.05964** (strongly `other`-dependent) |

So the arm does what it claims mechanically: in the control reader the topology response is gated by
`other`, and the additive head removes that gate. (Gradients reach `other`, `tail`, the topology head
and the topology encoder — verified in the `checks` stage.)

**Performance: FAIL.**

| arm | cal valid MAE | raw valid | fitted bias | Δ vs `N0_s0` | Δ without id172 | G0 worsening | T-gate |
|---|---|---|---|---|---|---|---|
| `T0_s0` | 0.115754 | 0.117064 | −0.014949 | **−0.004548** | −0.002942 | +0.001237 | **FAIL** |

Frozen T-gate = `cal gain ≥ +0.003` **and** `G0 worsening ≤ 0.001` **and** `gain without id172 ≥ −0.0005`.
All three clauses fail. The arm also does **not** help the severe ring rows it was aimed at: G1 MAE
goes 0.247161 → 0.297215 (worse) and G172 18.72 → 20.33 (worse).

## 4. Conditional combination and second-seed confirmation — not run

* **Combination (C)** requires *both* directions to pass their gates *and* to pass a complementarity
  check (node G0-gain ≥ 0.001, T-((G1+G172))-gain ≥ 0.003, both totals ≥ 0.003). Neither direction
  passed its own gate, so the pre-registered rule **closes** the combination. Not run.
* **Second-seed confirmation** of a frozen candidate requires the seed-0 gate to pass. No candidate
  passed. Not run — **and running it would have been an off-protocol, post-hoc rescue attempt.**
* `N0_s1` (seed-1 control, 240 epochs) was pre-submitted as the control half of a possible confirmation
  pair. It was allowed to finish and is reported above purely as an estimate of control seed
  variability (0.111206 → 0.114080). No seed-1 candidate was run.

## 5. Error budget / group accounting

Group contributions to the calibrated MAE (`contribution = Σ|err| / 1000`):

| arm | G0 (965) | G1 (34) | G172 (1) | total |
|---|---|---|---|---|
| `N0_s0` | 0.084080 (0.087130) | 0.004644 (0.154796) | 0.022482 (4.496366) | 0.111206 |
| `N0_s1` | 0.085951 (0.089068) | 0.005047 (0.168238) | 0.023082 (4.616495) | 0.114080 |
| `N1_s0` | 0.087473 (0.090646) | 0.003706 (0.123539) | 0.021531 (4.306110) | 0.112710 |
| `N2_s0` | 0.086291 (0.089420) | 0.004790 (0.159663) | 0.021332 (4.266396) | 0.112413 |
| `N3_s0` | 0.095391 (0.098850) | 0.004034 (0.134454) | 0.022257 (4.451494) | 0.121682 |
| `T0_s0` | 0.085318 (0.088412) | 0.003975 (0.132508) | 0.026461 (5.292153) | 0.115754 |

Stratified by snapped cycle penalty (`severity_table.csv`): the 5 rows with penalty ≤ −2 carry
≈0.0213–0.0265 of every arm's total MAE, of which the single penalty −6 row is ≈0.018–0.021. This is
why `gain_without172` is the honest signal, and it is negative for every arm.

The Stage-A audit (`AUDIT.md`, `audit_stageA.json`) localises the same structure from the other side:
replaying the published 240-epoch control soup reproduces raw `0.1102208197` / cal `0.10904212296`
exactly, with an error budget of G0 0.084897 + G1 0.007172 + G172 0.016973, a gap to 0.09 of 0.019042,
and an initialiser in which the node-encoder pre-activation signal RMS (1.365e-4) is ~350× smaller
than its bias RMS (0.0486).

## 6. Deviations, caveats and limitations

1. **Cross-protocol control anchor.** The historical pair-protocol control is cal **0.109042**; the
   control re-run inside this protocol is cal **0.111206** (+0.002164). All gains here are computed
   against the *matched* in-protocol control `N0_s0` (frozen rule), which is the correct comparison;
   the historical anchor is reported separately. Against the historical anchor every arm's gain is
   negative as well.
2. **Prefix runs are diagnostics, not truncations.** Iterating a `shuffle=False` DataLoader draws its
   base seed from the *global* RNG, so the per-epoch official-valid evaluation present in `stage=train`
   shifts the training RNG stream relative to `stage=train_prefix`. The 80-epoch prefixes and the
   240-epoch runs are therefore *not* the same trajectory (verified: epoch-1 train MAE 0.790321 vs
   0.800548). Only the *qualitative* health gate is read from the prefixes; every performance number
   comes from a self-contained 240-epoch `stage=train` run, and all five 240-epoch arms share one code
   path and commit, so they are mutually comparable. Prefixes correctly record
   `official_valid_loaded: false`.
3. **Resume path is broken (not used).** `_restore_cuda_rng` passes `state.to("cuda")` to
   `torch.cuda.set_rng_state_all`, which requires a **CPU** `ByteTensor` → remote resume runs died with
   `TypeError: RNG state must be a torch.ByteTensor` (jobs `55782`, `55783`). Because a fix would have
   required re-deploying the remote checkout while other jobs were running, the 240-epoch arms were
   instead run as **fresh complete trajectories on the single deployed commit `8657343edc62`**
   (the local CPU resume-equivalence test passes, so the resume *mechanism* is exact; only the CUDA
   restore branch is buggy). This keeps every scientific run on one commit and one code path, and no
   result depends on the broken branch. The fix (drop `.to("cuda")`) is recorded for the next revision.
4. **No significance claims.** With one seed per intervention arm and a control seed spread of ±0.0029,
   nothing here is a statistically established difference. The claim is only that the interventions do
   not *help*, and that this is consistent with seed noise around a slightly negative effect.
5. **Budget.** Compute stopped at `23:16:27Z`, 6h33m into the 7h20m authorised compute window; ≈4.9 of
   8 GPU-h used. Remaining time went to analysis, tables and this report.

## 7. Artifacts

| artifact | contents |
|---|---|
| `main_table.csv` | per-arm cal/raw train+valid MAE, bias, eval gap, group MAEs/contributions, epochs, members, kappa, WD, init hash |
| `group_table.csv` | group-accounted MAE + contribution for every arm (incl. prefixes) |
| `severity_table.csv` | MAE by snapped cycle-penalty stratum (`0`, `−1`, `≤−2`, any negative) |
| `pair_valid_predictions.csv` | 1000-row paired official-valid predictions (`y`, `<arm>_raw`, `<arm>_cal`) for every arm |
| `health_2x2.csv` | node-health timeline for all eight arm dirs (slot RMS, node-out std/alive, node grads, product RMS, dependency delta, gate) |
| `analysis.json` | machine-readable version of everything above incl. gates and timelines |
| `audit_stageA.json`, `AUDIT.md`, `audit_stageA.log` | Stage-A replay audit, error budget, competing explanations A–E |
| `t0_mechanism.json`, `t0_mechanism_check.py` | additive-topology mechanism verification |
| `analyze_overnight.py`, `build_report.py` | the two read-only analysis scripts |
| `frozen_node_scale_seed0.json` | frozen `kappa`, `r_init`, graph-id sha, init sha |
| `run_meta/<experiment>/meta.json` | vendored `rr` run metadata (commit, dirty flag, resources, runtime regime) |
| `<arm>/` | `curve.csv`, `health.json`, `summary.json`, `valid_predictions.csv`, `soup.json`, `soup_state.pt`, `checkpoint.pt`, `scale_manifest.json` |
| `TASK_STATE.json` | the running ledger kept during the overnight session |

Minimal re-derivation commands:

```bash
uv run python -m tracks.ksvd.results.zinc_overnight_bottleneck_v1.analyze_overnight
uv run python -m tracks.ksvd.results.zinc_overnight_bottleneck_v1.build_report
uv run python -m tracks.ksvd.results.zinc_overnight_bottleneck_v1.t0_mechanism_check
uv run python -m tracks.ksvd.results.zinc_overnight_bottleneck_v1.audit_stageA
```
