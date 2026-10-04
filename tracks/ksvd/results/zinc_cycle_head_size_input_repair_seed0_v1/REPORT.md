# REPORT — zinc-cycle-head-size-input-repair-seed0-v1

Round type: fixed-config, single-change diagnostic on the old 8000/2000
internal fold (reused dev, seed 0, CPU only).  The one change under test: give
the existing independent cycle head the actual graph size (`N = num_nodes`,
`E = number of unique undirected pairs`) in addition to the frozen `T25`.

**Verdict: `INPUT_REPAIRED_NOT_FIT`.**  The pre-registered *structure* gate
passes — adding N/E removes 3 of the 4 full-fit conflict classes and cuts the
fit-wide class-median cost by 64.3 % — but the paired heads show no material
fit or transfer improvement: the original 5 extreme fit rows move by −0.47 %
(not the pre-set −25 %), the dev overall calibrated gain is `+0.000135`
(CI `[-0.000785, +0.001173]`, far below the `0.003` gate), and the result does
not survive as a transfer signal.  The small-head fitting path remains the
limiting layer for the tail; the input-identity layer is mostly repaired.

## 0. Direct answers to the five questions

1. **What conflicts did N/E remove, and what is left?**  On the 8000 fit rows,
   `X25 = T25` has 327 exact classes with 4 conflict classes / 10 rows
   (0.125 %); `X27 = [T25, N, E]` has 1214 classes with 1 conflict class /
   2 rows (0.025 %).  The fit-wide class-median L1 drops `0.0060697235 →
   0.0021677584` (−0.003902 abs, −64.3 % rel).  The surviving conflict is the
   same-T25, same-size pair `train:0593` (`k=-5`, `c=-17.342`) and
   `train:9913` (`k=0`, `c=+0.00046`), both `N=28, E=31`; N/E cannot separate
   them.  **Exact dev coverage gets worse:** 1970/2000 (98.5 %) under X25 vs
   1860/2000 (93.0 %) under X27 — 110 more dev rows have no exact fit match.
   This is the expected conflict-vs-coverage tradeoff and it is real.

2. **Did the same-T25 tail rows fit better, and did the two uncovered dev tail
   rows transfer?**  Fit: the 5 `k<=-3` rows' q-MAE moves `14.2209 → 14.1539`
   (−0.47 %); the two old conflict tail rows move by −0.05 % (`0593`) and
   −0.51 % (`1424`), while `1760` gets worse (`0.0827 → 0.2956`) and
   `2347`/`3776` move ≤1.5 %.  Dev tail: `train:2210` (`k=-12`) is uncovered
   under both inputs and gets worse (q error `1.181 → 1.719`, P error
   `2.305 → 2.843`); `train:5050` (`k=-5`) is uncovered under both and moves
   marginally better (q `11.702 → 11.464`, P `12.195 → 11.956`).  **Neither
   dev tail row gains exact coverage, so this is not a migration
   improvement**; their chemistry-term errors (`|g−h_raw|` 1.118 and 0.487)
   are untouched by the head.

3. **Do the overall gain and G0 protection hold, and is the gain one row?**
   G0 is protected (cal worsening `+4.3e-5 <= 0.001`).  The overall cal gain is
   `+0.000135` with a CI crossing zero and therefore is **not** a usable
   transfer gain.  It is not literally one row, but it is concentrated in the
   `k=-2` dev group (`+4.28e-4` contribution) and inside it mainly in two rows
   (`train:7507 +3.07e-4`, `train:2052 +1.92e-4`), partly offset by
   `train:2210 (-2.69e-4)`; the other three groups are negative.  Dropping
   Q0's single worst row (`train:5050`) leaves `+1.6e-5`; dropping any single
   dev row leaves the gain in `[-1.72e-4, +4.04e-4]`.  No single-row deletion
   reaches the `0.003` bar.

4. **Where is the problem now — input split, fitting path, or
   transfer/extrapolation?**  Mostly the **fitting path**, with the tail also
   hitting **extrapolation**.  The input repair worked structurally (4 → 1
   conflict classes; the k<=-3 fit-wide class-median cost `5.549 → 1.734`), and
   four of the five extreme rows are now uniquely determined, yet the head
   barely moves on them; the size branch's gradient is live (`W_S` norm 0.617,
   step-2 grad 2.4e-4) but its dev effect is a thin reshuffle (`mean|ΔP| =
   0.0020`, `max = 0.614`, 992 better / 1008 worse).  The two dev tail rows have
   no exact fit match under either input, so their behaviour is extrapolation,
   not covered-row fitting.  Remaining evidence is insufficient to attribute
   the extreme-tail failure to capacity vs optimisation vs target rarity: one
   seed, one reused dev, only 5 fit rows and 2 dev rows.

5. **What is the single next thing worth buying?**  One pre-registered,
   fit-only diagnostic of the same frozen `X27` input and the same head family
   that asks whether the 5 extreme rows are fittable *at all* when isolated
   (same data, same recipe, any change fixed before the run).  If that probe
   cannot fit them, the current small-head path is the blocker and no new input
   should be bought on this dev; if it can, the blocker is the joint
   whole-fit optimisation, and a single frozen confirmation round is the
   candidate.  This round does not run it.  No feature/hyper-parameter menu.

## 1. Scope and protocol actually executed

* Old fold reused: fit `165e87ef…`, dev `fb8b7806…`; fit `k=0/-1/-2/<=-3 =
  7702/260/33/5`; dev `1926/65/7/2`.  This dev has been explored before, so all
  head results are exploratory single-split signals.
* Only official-train objects loaded: `T25_all.npz`,
  `O_seed0_raw_soup_state.pt` + `O_seed0_predictions.npz`,
  `target_decomposition.npz`, `fold_objects.npz`, handoff `train.npz`
  (`official_test_loaded=False`).  No `official-valid`/`official-test` object,
  no old row cache of them, no 12k cycle re-audit.
* `c/g/k` are the existing train-only, label-derived cycle decomposition; the
  cycle and `g` losses are extra supervision (disclosed).  `c` is never an
  inference input.
* Two fresh heads, seed 0, same base init, same schedule, 18900 steps each,
  300 epochs, soup 296–300, CPU 8 threads, no GPU, no remote job.
* Wall-clock: 47 s for the full Phase-A + Phase-B runner; all compute stopped
  well inside the 60-minute compute window.

## 2. Identity and graph audits (all pass)

| check | value |
|---|---|
| T25 array hash | `dc2e1516…` (matches frozen) |
| fit / dev idx hash | `165e87ef…` / `fb8b7806…` |
| O soup / O preds file | `61d4aeb…` / `5cee5782…` |
| prep blob | `968e82dd…` |
| re-forward of the real O soup, **all 8000 fit + 2000 dev rows** | max abs `1.907e-6 ≤ 2e-6` |
| fixed 32-row dev batch | `9.537e-7` |
| `prep topology_features` vs `T25` | `0.0` |
| fit / dev y-c decomposition identity | `0.0` |
| schedule hash (both arms) | `d836d9b3…` |

Graph audit from handoff `train.npz` (10000 train graphs, 231 664 nodes,
249 279 raw edges): endpoints always in `[0, N)`; 0 self-loops; 0 duplicate
unordered pairs; 0 reverse edges (the stored list is a single undirected pair
list); independent NetworkX reference reproaches `N`/`E` for **all 10000**
graphs with 0 mismatches; a deterministic 256-graph renumbering check changes
neither N nor E.  No anomaly required an edge-deletion contract: nothing was
deleted.  N/E are graph-only; the recorded T25 relation is high but not
sufficient (max column |corr| with N = 0.779, with E = 0.849; 185 of 190
repeated T25 classes contain ≥ 2 distinct N and E).

## 3. Phase A — full-fit conflict structure

Fit scaler (graph-level, unweighted, float64, floor 1e-6): `N` mean 23.18275
std 4.488552; `E` mean 24.9435 std 5.288602.  Standardising the 113 distinct
integer (N,E) pairs gives 113 distinct float32 pairs — no accidental merging;
no signed zeros.

| statistic | X25 (T25) | X27 (T25+N+E) |
|---|---:|---:|
| exact classes | 327 | 1214 |
| repeated-row fraction | 0.982875 | 0.92025 |
| conflict classes | 4 | **1** |
| conflict rows | 10 (0.125 %) | **2 (0.025 %)** |
| global class-median L1 / row | 0.0060697235 | **0.0021677584** |
| — k=0 cost | 0.002026 | 0.001126 |
| — k=-1 cost | 0.006670 | 0.0 |
| — k=-2 cost | 0.105103 | 0.0 |
| — k<=-3 cost | 5.549461 | 1.734207 |

(the per-group rows are the fit-wide class-median output evaluated on that
group, **not** group-specific irreducible bounds.)

**Full split of the four original X25 conflict classes:**

| old class | members | X27 outcome |
|---|---|---|
| 86 | `3741` (k=-1, N31/E34), `7491` (k=0, N28/E31) | both singletons, span 0 |
| 232 | `0593` (k=-5, N28/E31), `2447` (k=0, N27/E30), `2472` (k=0, N25/E28), `9913` (k=0, N28/E31) | `2447`, `2472` singletons; **`0593+9913` remain one class, span 17.342** |
| 250 | `2232` (k=-2, N26/E29), `3626` (k=0, N28/E31) | both singletons |
| 259 | `1270` (k=0, N34/E38), `1424` (k=-6, N33/E37) | both singletons |

**The 5 fit `k<=-3` rows under X27:**

| row | k | old class | new class | new span | mixed k? |
|---|---:|---|---:|---:|---|
| `train:0593` | -5 | 232 (conflict) | size 2 with `train:9913` | 17.342 | yes (k=-5 and 0) |
| `train:1424` | -6 | 259 (conflict) | singleton | 0.0 | no |
| `train:1760` | -5 | 323 | singleton | 0.0 | no |
| `train:2347` | -4 | 195 | singleton | 0.0 | no |
| `train:3776` | -6 | 247 | singleton | 0.0 | no |

**Structure gate:** identity checks pass; reduction `0.003902 ≥ 0.001`; at
least one old conflict `k<=-3` row (`train:1424`) has new span `0.0 ≤ 1e-10`.
**Gate passed** → Phase B was bought.  Per the frozen rule, dev coverage is
reported only after the gate file was written.

**Dev coverage (post-gate, never used to decide the purchase):**

| input | covered | fraction | covered-in-conflict | uncovered | severe uncovered |
|---|---:|---:|---:|---:|---|
| X25 | 1970 | 0.985 | 1 | 30 | 5 (`2052, 2210, 4344, 5050, 7507`) |
| X27 | 1860 | 0.930 | 0 | 140 | 7 (adds `1238, 7659`; no severe row gains coverage) |

## 4. Phase B — the paired heads

Contract: `Q0` is exactly the published `P_U` protocol: init tensors equal to
`cpu_P_U_head_init_state.pt`, `q` maxdiff `0.0` and `b_P`
`0.005214758217334747` equal to `cpu_P_U_head_soup_state.pt`; `Q0` dev cal
`0.1131797829947609` reproduces the released anchor exactly.  Native-25 replay
of the Q0 soup (W_S dropped) reproduces Q0 predictions with `max|diff| = 0.0`.
`QNE` uses the same init/schedule with `gate=1`; `W_S` starts at zero, has zero
gradient at step 1 (last layer `W=0`), nonzero from step 2 (2.39e-4) and ends
at norm 0.61666.  Each arm: 3905 parameters, 18900 steps, soup 296–300.

| arm | b_P | fit P_cal | dev raw | dev cal | dev G0 cal | dev k=-1 | dev k=-2 | dev k<=-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q0 | 0.005214758 | 0.0451640 | 0.1136121 | **0.1131798** | 0.1002897 | 0.1154121 | 1.6000336 | 7.2498257 |
| QNE | 0.005349785 | 0.0447437 | 0.1134855 | **0.1130446** | 0.1003326 | 0.1185332 | 1.4778688 | 7.3994487 |

Gains `MAE(Q0) − MAE(QNE)` (positive = QNE better):

| endpoint | point | paired 95 % CI |
|---|---:|---|
| dev overall cal | **+0.0001351** | `[-0.0007847, +0.0011728]` |
| dev overall raw | +0.0001266 | `[-0.0008002, +0.0011706]` |
| dev G0 cal (QNE worsening) | −0.0000430 | `[-0.0001021, +0.0000186]` |
| dev severity-stratified cal (secondary) | +0.0001351 | `[-0.0006937, +0.0009899]` |

Bootstrap self-tests pass: identical predictions → 0; arm swap → exact mirror;
`+0.25` constant shift changes the gain by 0.173 ≤ 0.25.

**Group contribution to the dev cal MAE** (`Σ|err|/2000`, groups disjoint and
summing to the overall; the `k<=-2` cumulative view is not added):

| group | n | Q0 | QNE | gain contribution |
|---|---:|---:|---:|---:|
| k=0 | 1926 | 0.0965789 | 0.0966203 | −0.0000414 |
| k=-1 | 65 | 0.0037509 | 0.0038523 | −0.0001014 |
| k=-2 | 7 | 0.0056001 | 0.0051725 | **+0.0004276** |
| k<=-3 | 2 | 0.0072498 | 0.0073994 | −0.0001496 |

Fit shows the same thin picture: overall q-vs-c MAE `0.0137079 → 0.0133399`,
fit P_cal `0.0451640 → 0.0447437`; the fit-wide gain is positive for every
leave-one-fit-row-out (`0.000308…0.000524`) but is not concentrated in the
tail.

**Fit-tail marker (descriptive, pre-registered):** the 5 `k<=-3` rows' q-MAE
`14.2209 → 14.1539` = **−0.47 %**, far from the ≥25 % marker.  Per row:

| fit row | k | q err Q0 | q err QNE | note |
|---|---:|---:|---:|---|
| `train:0593` | -5 | 17.304 | 17.296 | conflict pair; +0.05 % |
| `train:1424` | -6 | 20.795 | 20.690 | now singleton; +0.51 % |
| `train:1760` | -5 | 0.0827 | 0.2956 | **worse** |
| `train:2347` | -4 | 12.107 | 11.975 | +1.1 % |
| `train:3776` | -6 | 20.816 | 20.513 | +1.5 % |

**Dev tail rows (complete budget in `per_graph_dev.csv` / `dev_tail_rows.csv`):**

| dev row | k | X25/X27 coverage | chem err `|g-h|` | q err Q0→QNE | P err Q0→QNE |
|---|---:|---|---:|---:|---:|
| `train:2210` | -12 | none / none | 1.118 | 1.181 → 1.719 | 2.305 → 2.843 |
| `train:5050` | -5 | none / none | 0.487 | 11.702 → 11.464 | 12.195 → 11.956 |

**Size-branch diagnostics.**  QNE with the size gate switched to 0 on dev:
`mean|ΔP| 8.2e-4`, `p95 1.05e-3`, `max 0.273`; group MAE (gate on → off):
`k=0 0.100333→0.100391`, `k=-1 0.118533→0.119589`, `k=-2 1.477869→1.390293`,
`k<=-3 7.399449→7.375704`; on the whole dev, gate-off is marginally *better*
(0.112805 vs 0.113045).  So the trained size branch is active but its dev
effect is a small reshuffle, mainly helping `k=-1` and slightly hurting the
`k=-2` pool; gradient liveness is not a performance gain.  Q0 keeps `W_S`
identically zero.  q output ranges/slopes barely change (dev q std 1.1514 →
1.1418; slope-vs-c 0.897 → 0.892; fit q min −17.259 → −17.637).

**Movement:** fit `mean|ΔP| 0.00187`, `p95 0.00188`, `max 0.895`,
3951 better / 4049 worse; dev `mean|ΔP| 0.00204`, `p95 0.00205`, `max 0.614`,
992 better / 1008 worse.  Small MAE difference is not "predictions unchanged".

**Deploy checks (wrapper):** online N/E recomputed from graph edges equal the
stored X27 rows exactly (`0.0`); wrapper output vs cached `P` `9.54e-7`;
repeat call bit-identical; permuting `y`/`c`/`k` changes prediction by exactly 0;
inference never receives labels; one fit-only `b_P` per arm.

## 5. Performance gate and row-level budget

| condition | value | met |
|---|---|---|
| dev overall cal gain ≥ 0.003 | +0.000135 | no |
| main paired CI lower > 0 | −0.000785 | no |
| dev overall raw gain > 0 | +0.000127 | yes |
| dev G0 cal worsening ≤ 0.001 | +0.000043 | yes |
| count / identity / no-label / single-calibration / replay | all pass | yes |

**Performance gate not met.**  Leave-one-row-out (fixed, no training rows
removed): full-dev gain `+1.35e-4`; without Q0's worst row (`train:5050`)
`+1.58e-5`; range over any single dropped dev row `[-1.72e-4, +4.04e-4]`.
The two positive leaders are `train:7507` (`k=-2`, +3.07e-4) and
`train:2052` (`k=-2`, +1.92e-4); `train:2210` contributes −2.69e-4.

## 6. Decision and interpretation

Per the frozen table the outcome is **`INPUT_REPAIRED_NOT_FIT`**: the structure
gate passed, but neither the original conflict/extreme tail q-fit nor the dev
transfer moved materially; the current small-head training path is still
limited.  This does **not** say "all new representations are useless": the
fit-side conflict structure is genuinely improved, and the remaining
single conflict (`0593`/`9913`) is an input-identity case that N/E cannot fix
by construction (same size and topology, different label-derived `c`).

Neither the fit marker nor the performance gate licenses a transfer claim.
No official-valid/test read is involved; the internal dev is not converted to
an official-valid number.  Even a passing round would only support the
independent-head information repair, not task-dictionary superiority over an
MLP, and not SOTA.

## 7. Boundaries / what remains unknown

* One seed, one fixed 25→64→32→1 architecture, one reused diagnostic dev
  (2000 rows, 9 severe), 5 fit extreme rows.
* The size branch changes predictions by ~2e-3 on average (max 0.61); the small
  positive dev gain is not separable from zero and is concentrated in a few
  `k=-2` rows.
* Input conflicts are reduced but not eliminated; the surviving class cannot
  be split by N/E because its members have identical N and E.
* The extreme-tail failure cannot yet be attributed between head capacity,
  optimisation under unweighted mean L1 with 5 rare rows, and genuine
  extrapolation; the two dev tail rows are uncovered under both inputs.
* `c`'s cycle basis is node-order dependent; N/E are order-invariant counts.

## 8. Budget / artifacts

Full Phase A+B CPU wall time 43–48 s (plus a few seconds of analysis), 8
threads, GPU 0, no remote job, no official-valid/test access.  Artifacts:
`PROTOCOL.md`, `METHOD_CONTRACT.md`, `EXECUTION.md`, `DECISION.md`,
`structure_gate.json`, `conflict_x25.json`, `conflict_x27.json`,
`original_conflict_split.csv`, `tail_rows_structure.json`, `dev_coverage.json`,
`size_features.npz`, `size_scaler.json`, `size_merge_check.json`,
`size_vs_t25_relation.json`, `h_verify.json`, `Q0*`/`QNE*` states and
predictions, `schedule.npz`, `Q0_native25_state.pt`, `deploy_bundle.pt`,
`deploy_wrapper.py`, `main_table*.csv`, `group_table*.csv`, `gain_table.csv`,
`per_graph_fit.csv`, `per_graph_dev.csv`, `dev_tail_rows.csv`,
`bootstrap.json`, `performance_gate.json`, `wrapper_checks.json`,
`stage_b_analysis.json`, `descriptive.json`, `phase_a_summary.json`,
`phase_b_summary.json`, `size_diagnostics.json`, `sensitivity.json`,
`fit_tail_q_errors.json`, `decision.json`, `manifest.json`, `budget.json`,
`analyze.py`, `make_manifest.py`.
