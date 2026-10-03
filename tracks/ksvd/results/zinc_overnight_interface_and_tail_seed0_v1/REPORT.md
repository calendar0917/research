# REPORT — zinc-overnight-interface-and-tail-seed0-v1

Single seed (0), train-only exploration on the frozen 8000/2000 internal fold,
no official-valid read, no official-test read.  All numbers are exploratory
single-split signals on a reused internal dev, not independent confirmation.

**Headline**

* On the frozen compressed reference `B`, adding the direct
  *structure-code x atom/bond-category* residual interface gives a small but
  positive dev gain.  The largest frozen-body arm is **R_DM (dense coding,
  marginal statistics)**: dev overall cal `0.115736` vs `B` `0.120130`
  (**+0.004395**, paired 95% CI `[+0.002915, +0.005798]`), dev G0 cal
  `0.092664` vs `0.097347` (**+0.004683**, CI `[+0.003269, +0.005923]`).
* The gain does **not** come from joint structure-attribute correspondence:
  joint (`J`) is not better than marginal (`M`) in either coding
  (sparse `+0.000365` CI crosses 0; dense `-0.001825` CI excludes 0 in the
  wrong direction), and trained `J` arms barely react when the pairing is
  destroyed at inference.  The one clearly positive structural increment over
  a Sem-only adapter is dense-marginal (`R_DM` vs `R_CS`: `+0.002122` overall,
  CI `[+0.000922, +0.003499]`).
* Once the body is unfrozen (Phase 2, matched `A0` control), neither the
  Sem-only adapter `C` nor the selected interface `T` passes the
  pre-registered `+0.003` gate: `C` `+0.000960` overall (G0 `+0.001256`),
  `T` `+0.000205` overall (G0 `-0.000170`), both CIs crossing 0.
  Unfreezing the body itself (`A0` vs `B`) gives `+0.006011`
  (`[+0.001883, +0.010182]`), i.e. the dominant lever is the frozen-body
  limitation, not the new interface.
* The extreme ring tail does not come along: severity-balanced weighting
  (`q_B`) makes the dev endpoint far worse (`P_B` dev cal `0.152442` vs
  `P_U` `0.113180`; gain `-0.039262`, CI `[-0.074767, -0.010800]`); the five
  fit rows at `k<=-3` are fitted (`14.24 -> 0.097` MAE) but the two dev rows
  get worse (`7.25 -> 21.01`) and `k=0` worsens (`0.1003 -> 0.1307`).
* Consequence of the pre-registered gates: **no Phase-3 10k family was
  bought and the single frozen official-valid read was not performed**
  (`final_decision.json`, `final_valid_summary.json`).  The official-valid
  1000 rows remain unread in this round; official-test was never touched.

## 1. Anchors verified before fitting

| anchor | value |
|---|---|
| `S_M` raw soup sha256 | `2def4cb64c1350a60348fb6fc7ce88a07bffe563367ac7e9666ee6f7dff5c5b8` |
| compressed deploy `B` | 267,611 params; new body 267,611 + `D` 2,080 + adapter 126,072 = 395,763 |
| full dev replay `B` vs `phaseA_sm_replay.npz` | max abs diff `1.9e-06` |
| fold hashes | fit `165e87ef…`, dev `fb8b7806…`; dev strata `k=0/ -1/ <=-2` = `1926/65/9` |
| interface scalars | `tau_A = 0.0252396`, `tau_E = 0.061480`, `kappa = 0.380682` (fit-8k, float64) |
| zero-init identity | adapter output exactly 0; `B` reproduction `9.5e-07` fit / `1.9e-06` dev |
| shared schedule | Phase-1 stream `53b4c70c…` (10080 steps/arm), Phase-2 stream `7b11a529…` (15120 steps/arm) |

## 2. Phase 1 — frozen body, five interface arms (internal dev)

Arms: `R_CS` Sem-only adapter, `R_SM/R_SJ` sparse marginal/joint,
`R_DM/R_DJ` dense marginal/joint.  Body and `D` frozen; only the 126,072
adapter parameters train; mean L1 on `y`; soup 156–160.

| arm | fit raw | dev G0 raw | dev G0 cal | dev raw | dev cal | b |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.037966 | 0.097337 | 0.097347 | 0.120122 | 0.120130 | -0.000440 |
| R_CS | 0.024544 | 0.094743 | 0.094717 | 0.117882 | 0.117858 | +0.000623 |
| R_SM | 0.012868 | 0.094595 | 0.094604 | 0.117468 | 0.117475 | -0.000304 |
| R_SJ | 0.013772 | 0.094279 | 0.094289 | 0.117103 | 0.117110 | -0.001289 |
| R_DM | 0.017571 | 0.092734 | **0.092664** | 0.115800 | **0.115736** | +0.001295 |
| R_DJ | 0.017646 | 0.094621 | 0.094578 | 0.117601 | 0.117561 | +0.001251 |

Paired gains over `B` (g = error(B) - error(arm), positive is better; 1000x
paired bootstrap seed 20261003, shared indices):

| arm | dev overall cal | dev G0 cal |
|---|---:|---:|
| R_CS | +0.002273 [+0.000658, +0.003628] | +0.002630 [+0.001281, +0.003969] |
| R_SM | +0.002655 [+0.000965, +0.004247] | +0.002744 [+0.001139, +0.004272] |
| R_SJ | +0.003020 [+0.001454, +0.004621] | +0.003059 [+0.001421, +0.004557] |
| R_DM | **+0.004395 [+0.002915, +0.005798]** | **+0.004683 [+0.003269, +0.005923]** |
| R_DJ | +0.002569 [+0.000985, +0.004023] | +0.002770 [+0.001241, +0.004057] |

**Where does the gain come from?**  Decomposing against the Sem-only adapter
(`R_CS`), which already captures +0.002273 overall:

| contrast (positive = left better) | dev overall cal | dev G0 cal |
|---|---:|---:|
| R_SM - R_CS (sparse marginal increment) | +0.000382 [-0.000898, +0.001823] | +0.000113 [-0.001173, +0.001401] |
| R_SJ - R_CS (sparse joint increment) | +0.000747 [-0.000690, +0.002296] | +0.000428 [-0.000979, +0.001773] |
| R_DM - R_CS (dense marginal increment) | **+0.002122 [+0.000922, +0.003499]** | **+0.002053 [+0.000842, +0.003226]** |
| R_DJ - R_CS (dense joint increment) | +0.000297 [-0.001090, +0.001637] | +0.000140 [-0.001117, +0.001225] |
| R_SJ - R_SM (J vs M, sparse) | +0.000365 [-0.001026, +0.001740] | +0.000315 [-0.001111, +0.001727] |
| R_DM - R_DJ (M vs J, dense) | **+0.001825 [+0.000632, +0.003025]** | **+0.001913 [+0.000644, +0.003054]** |
| R_DM - R_SM (dense vs sparse, marginal) | **+0.001740 [+0.000506, +0.003070]** | **+0.001939 [+0.000783, +0.003189]** |
| R_SJ - R_DJ (sparse vs dense, joint) | +0.000451 [-0.000833, +0.001784] | +0.000289 [-0.001014, +0.001608] |

Coding x joint interaction (gain(J vs M|sparse) - gain(J vs M|dense)) =
`+0.002190`: joint helps only in the sparse path and actually hurts in the
dense path.  A large part of the frozen-body gain is the adapter capacity
itself (`R_CS`), concentrated on `k=0`; the structural statistics add a small
increment, and only dense-marginal is clearly positive.

### Mechanism health (`mechanism_health.json`)

* Adapter output RMS on dev: `R_CS 0.173`, `R_SM 0.201`, `R_SJ 0.210`,
  `R_DM 0.215`, `R_DJ 0.227` — comparable to the bridge's own code activity
  (abs-mean ~0.146), i.e. the interface is active, not dead.
* Structural code use: sparse `alpha` nonzero fraction `0.25` (s=8/32); dense
  `1.00`.  Bridge activity and the frozen Sem path are unchanged.
* **J/M inference switch with trained weights fixed (dev cal MAE):**

  | arm (trained) | own block | flipped block | delta |
  |---|---:|---:|---:|
  | R_SM (marginal) | 0.117475 | 0.124195 (joint) | +0.006720 |
  | R_SJ (joint) | 0.117110 | 0.117802 (marginal) | +0.000692 |
  | R_DM (marginal) | 0.115736 | 0.124725 (joint) | +0.008989 |
  | R_DJ (joint) | 0.117561 | 0.117891 (marginal) | +0.000330 |

  Marginal-trained arms depend strongly on the exact marginal statistic;
  joint-trained arms are nearly invariant when the within-bucket pairing is
  destroyed at inference.  That is direct evidence the trained joint arms were
  **not using the pairing** that the hypothesis is about.
* Frozen-body limitation: `R_DM` has the *worst* fit error of the four
  structural arms (`0.017571` vs `R_SM 0.012868`) yet the *best* dev error —
  the frozen-body gain is not monotonically "more structure fit = more dev
  gain", and the adapter is partly compensating frozen-body misfit
  (`B` fit raw `0.037966` -> arms `0.0129..0.0245`).

Per-group dev table (`stage1_group_table.csv`, partition
`k=0/-1/-2/<=-3`, contributions sum exactly to the overall):
`R_DM`'s overall gain is almost entirely `k=0` (contribution gain `+0.004510`),
`k=-1` `+0.000154`, while `k=-2` `-0.000176` and `k<=-3` `-0.000093` get
slightly worse.  The two `k<=-3` dev rows sit at MAE ~16.3–16.5 for every
arm; the interface does not touch them.

## 3. Phase 2 — matched unfrozen arms (internal dev)

All three arms start from the same compressed state, share dense coding
(matching Phase-1 winner `R_DM`), share the schedule stream, train
body + `D` + adapter with `L1 + 33.95873 * reconstruction`, soup 236–240.

| arm | fit raw | dev G0 raw | dev G0 cal | dev raw | dev cal | b | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 (adapter x0) | 0.028963 | 0.093855 | 0.093236 | 0.114685 | 0.114119 | -0.008762 | 1029 s |
| C (Sem-only delta) | 0.028895 | 0.092671 | **0.091980** | 0.113842 | **0.113159** | -0.012795 | 1025 s |
| T (dense marginal) | 0.033956 | 0.093618 | 0.093405 | 0.114075 | 0.113915 | -0.005515 | 863 s |

Pre-registered Phase-2 gate (per arm vs matched `A0`; G0 cal gain >= 0.003,
CI lower > 0, overall cal gain >= 0.003, overall raw gain > 0):

| arm | G0 cal gain | overall cal gain | overall raw gain | pass |
|---|---:|---:|---:|---:|
| C | +0.001256 [-0.000501, +0.003061] | +0.000960 [-0.000859, +0.002788] | +0.000843 [-0.000958, +0.002683] | **False** |
| T | -0.000170 [-0.001912, +0.001679] | +0.000205 [-0.001702, +0.002205] | +0.000610 [-0.001280, +0.002562] | **False** |

* `T` is not better than `C` (`-0.000756` cal, CI `[-0.002738, +0.001324]`):
  once the body can adapt, the structural interface contributes nothing
  measurable.
* `A0` vs `B`: `+0.006011` overall cal (`[+0.001883, +0.010182]`) — unfreezing
  is the big lever; the interface on top of it collapses to ~0.
* Mechanism: Phase-2 adapter RMS is `A0 0.000`, `C 0.0153/0.0114`,
  `T 0.0322/0.0306` (fit/dev) — an order of magnitude below Phase-1
  (`~0.2`).  The unfrozen body absorbs the fitting, leaving the interface as a
  small residual correction.  `T`'s J->M flip moves dev cal only
  `0.113915 -> 0.114280` (`+0.000365`).

## 4. Extreme ring tail — severity-balanced frozen head (CPU)

Frozen released `h_raw` (`O_seed0`) plus a 25->64->32->1 head on `T25`,
trained 300 epochs, soup 296–300; `q_U` unweighted reproduces the released
`P_seed0` exactly (max abs diff `0.0`).  `q_B` weights groups
`k=0/-1/-2/<=-3` by `N_fit/(G n_g)` (fit counts `7702/260/33/5`, max weight
400, mean 1, ESS `68.29`).

| arm | fit cal | dev cal | dev k=0 | dev k=-1 | dev k=-2 | dev k<=-3 |
|---|---:|---:|---:|---:|---:|---:|
| P_U (unweighted) | 0.045164 | **0.113180** | 0.100290 | 0.115412 | 1.600034 | 7.249826 |
| P_B (severity-balanced) | 0.083556 | 0.152442 | 0.130666 | 0.120955 | 0.476320 | 21.012514 |

Gate `P_B` vs `P_U`: overall cal gain `-0.039262`
(`[-0.074767, -0.010800]`), overall raw `-0.040082`
(`[-0.075816, -0.011623]`), G0 worsening `+0.030376` — **fail** on every
clause.  The weight mass is carried by five fit rows (`k<=-3`): in-sample
they drop `14.24 -> 0.097` MAE, but the two dev rows go `7.25 -> 21.01` and
the shared head loses `k=0` (`0.1003 -> 0.1307`).  Exact-input conflicts are
small overall (`327` exact `T25` classes, `190` repeated covering `98.3%`
rows, only `4` conflicting classes; irreducible L1/row `0.00607`) but the tail
classes are intrinsically ambiguous (`k<=-3` irreducible `5.55` MAE/row).

## 5. Decision (pre-registered, no post-hoc change)

* Phase-2 gate not passed by `C` or `T`; CPU tail gate not passed by `P_B`
  (closed before Phase 2, per protocol).  **No Phase 3 is bought, and the
  single frozen official-valid read is not performed.**  See
  `final_decision.json`; the validation-entry code (`final_eval.py`) is
  committed and rerunnable but was deliberately not invoked.
* Best frozen-body candidate for any future confirmation: `R_DM` (dense
  marginal).  Best matched unfrozen candidate: `C` (Sem-only adapter), but it
  is not a structural-interface result and it failed the +0.003 gate.
* Nothing is written back to the main line (`task/zinc-overnight-interface-
  and-tail-seed0-v1` is isolated, not pushed/merged).

## 6. Failure list / ineffective attempts

1. `P_B` severity-balanced weighting: the only fit rows that can be weighted
   are 5, ESS 68; it overfits them and destroys the dev endpoint.  Not worth
   another head-only attempt with this support size.
2. First Phase-1 submission failed before training because
   `interface_stats.json` was untracked (`tracks/*/results/**/*.json` is
   git-ignored).  Fixed with `git add -f`; the failed jobs never trained.
3. First CPU-tail implementation sampled a fresh permutation per batch; the
   bug was fixed before any result was used, and `P_U` then reproduced
   `P_seed0` exactly.
4. `T` vs `C`: the selected interface is worse than the Sem-only control on
   every dev endpoint (all CIs crossing 0).  The pre-registered rule was kept
   rather than switching to another arm after seeing Phase 2.

## 7. What is reproducible vs exploratory

* **Reproducible now, with committed artifacts**: anchor hashes and identity
  replays (`replay_check.json`: all arms `<= 4.8e-06`; `cpu_replay_check.json`
  exact `0.0`); every Phase-1/2 table and CI (`analyze.py --stage 1/2`);
  mechanism J/M flips (`analyze.py --mechanism`); CPU conflict analysis.
* **Exploratory single-split signals**: all performance deltas — one seed,
  reused internal dev, adaptive Phase-1 selection of `T` (the CI does not
  correct for selection).  The J-vs-M and sparse-vs-dense contrasts are
  within-split contrasts with CIs but remain single-seed.
* **Not evidence**: the internal-dev `+0.003`-scale effects are close to the
  fold's noise scale; no official-valid confirmation exists in this round.

## 8. Next experiments worth buying (and what not to do)

Do next (if a new round is opened): run `R_DM` vs `B` (and `R_DM` vs `R_CS`)
on a fresh independent split with seed 0 and, only if it passes a pre-set
`+0.003` bar there, one frozen official-valid read.  Also record the
J-vs-M contrast on that fresh split — the current flip evidence predicts it
will stay null.

Do not: add more arms/widths to this reused dev, reweight `q` again on the
same 5 tail rows, tune `H1_LAMBDA`/`kappa`/adapter width on the dev, or read
official-valid to rescue the failed gate.
