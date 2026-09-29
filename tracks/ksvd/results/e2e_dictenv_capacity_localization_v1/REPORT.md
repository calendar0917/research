# REPORT — `e2e_dictenv_capacity_localization_v1`

Capacity Localization + One-Winner Full Training on the ZINC
dictionary-environment line (Workstream Z).  CPU only (`CUDA_VISIBLE_DEVICES=""`,
`torch.device("cpu")`), no GPU, no SSH, no remote compute; the official ZINC
test was never loaded (`official_test_loaded = false` in every payload).

* Pre-registration:
  `notes/e2e_dictenv_capacity_localization_v1_preregistration.md`
  (frozen before the first screening process; sha256 `f5c4a8d2e489…`,
  implementation freeze commit `becc6f6`).
* Full discussion:
  `notes/e2e_dictenv_capacity_localization_v1_analysis.md`.
* Evidence: `results/e2e_dictenv_capacity_localization_v1/` —
  `preregistration_snapshot.json`, `parameter_budget.json`, `preflight.json`,
  `init_audit.json`, `screening/{m0,fusion,relation,readout}/`
  (`result.json`, `curve.csv`, `soup.json`, `soup_state.pt`),
  `screening/screening_summary.csv`, `screening/decision.json`,
  `full/NOT_RUN.json`, `mechanism/probes.json`, `analysis_tables.md`,
  `summary.json`, `run_chain.sh`.

## A. Bottom line

1. **No capacity direction localises: `NO_CLEAR_CAPACITY_LOCALIZATION`.**
   Under the frozen screening protocol (40 epochs, warm-started from the same
   `CSSD-q1` seed-0 Top-5 soup, fresh Adam 1e-3), the seed-0 Top-5 soup over
   epochs 21–40 is `0.129464` (F), `0.129168` (R), `0.129663` (G) versus
   `0.129662` (M0 control) — `Δsoup = −0.000198 / −0.000494 / +0.000002`,
   i.e. 8–20× short of the required `−0.004`.  Gate not fired; branch 1 of
   §8 applies; `full/NOT_RUN.json` written; **no 320-epoch run was bought**.
2. **The secondary late-window signal is real but not decisive.**  The frozen
   secondary metric (mean valid MAE over epochs 31–40) improves for all three
   candidates: M0 `0.154314` → F `0.150048` (`−0.004266`), R `0.149848`
   (`−0.004466`), G `0.147582` (`−0.006732`).  All three pass
   `Δlast10 ≤ −0.003`; nobody passes both conditions, so no winner exists.
3. **The screen itself was the limiting factor (post-hoc, descriptive).**
   Every arm — including the pure control — degrades after the fresh-Adam
   warm restart: M0 goes from step-0 `0.130028` to a best of `0.135621`
   (epoch 21) and `0.156383` at epoch 40; the same is true for F
   (`0.136613` / `0.156190`), R (`0.138940` / `0.142631`) and G
   (`0.135855` / `0.136734`).  The control's own soup differs from its own
   starting checkpoint by `−0.00037`; the candidates' `Δsoup` values are at
   or below that noise floor.  The round therefore cannot separate
   structure-semantic fusion (F), static pair composition (R) and invariant
   readout (G) capacity.
4. **Step-0 audit clean for all candidates.**  Valid MAE `0.131102` (F,
   +0.001074), `0.130860` (R, +0.000832), `0.129941` (G, −0.000087) vs
   CAP-BASE `0.130028`; every capacity parameter has finite, strictly
   positive gradient; all prediction shifts far below the +0.03 bound.
5. **Parameter discipline held.**  Added parameters 29 568 / 34 400 / 30 336,
   totals 127 295 / 132 127 / 128 063 (preferred 120 k–160 k), ratio
   `1.1634 ≤ 1.5`, hard ceiling 250 k; widths were never tuned.
6. **Budget spent: 160 of the authorised 480 epochs** (4 × 40).  No full run,
   no seed 1/2, no combination (`F+R`, `F+G`, `R+G`), no post-hoc rescue.

## B. Frozen context

```text
CAP-BASE = CSSD-q1   seed-0 soup valid MAE 0.130028 (recomputed in preflight)
  C6 clean mask; q1 train-only common coordinate; q1-orthogonal sparse
  dictionary K=32 / s=8 / IHT-10; paired node + edge structure-semantic
  binding; full relation (distance + overlap + boundary); first + second
  moment readout; 97 727 parameters.
Historical reference only: FINAL-CLEAN sparse seed-0 soup 0.128499
(no rollback was authorised or performed).
```

New capacities (one frozen architecture each, one seed, one budget):

| candidate | change | added params | total |
|---|---|---|---|
| F | multi-rank factorized bilinear residual fusion, `H=4`, `d_h=24` | 29 568 | 127 295 |
| R | two static residual FiLM blocks on the pair state, hidden 256 | 34 400 | 132 127 |
| G | gated DeepSets summaries (dim 192, node + pair) appended to the reader | 30 336 | 128 063 |

## C. Initialization audit (frozen, before training)

| candidate | step-0 valid MAE | Δ vs CAP-BASE | mean shift | max shift | capacity grads |
|---|---|---|---|---|---|
| M0 | 0.130028 | 0.000000 | 0.00000 | 0.00000 | — (control) |
| F | 0.131102 | +0.001074 | 0.01003 | 0.05211 | 18/18 finite > 0 |
| R | 0.130860 | +0.000832 | 0.01741 | 0.12942 | 20/20 finite > 0 |
| G | 0.129941 | −0.000087 | 0.00113 | 0.00300 | 8/8 finite > 0 |

## D. Screening (frozen primary + secondary)

| candidate | params | step-0 | best valid | best epoch | soup (21–40) | last-10 | Δsoup vs M0 | Δlast10 vs M0 | branch grad | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 97 727 | 0.130028 | 0.135621 | 21 | 0.129662 | 0.154314 | 0.000000 | 0.000000 | 0.000 | CONTROL |
| F | 127 295 | 0.131102 | 0.136613 | 37 | 0.129464 | 0.150048 | −0.000198 | −0.004266 | 0.239 | NO_CAPACITY_SIGNAL |
| R | 132 127 | 0.130860 | 0.138940 | 28 | 0.129168 | 0.149848 | −0.000494 | −0.004466 | 2.513 | NO_CAPACITY_SIGNAL |
| G | 128 063 | 0.129941 | 0.135855 | 27 | 0.129663 | 0.147582 | +0.000002 | −0.006732 | 0.001 | NO_CAPACITY_SIGNAL |

Frozen gate: `Δsoup ≤ −0.004` **and** `Δlast10 ≤ −0.003` (STRONG at
`Δsoup ≤ −0.010`).  Passing candidates: `[]` →
`NO_CLEAR_CAPACITY_LOCALIZATION`.  Frozen tie order (F > R > G) was never
needed.

### Why the screen had no power (post-hoc, descriptive)

1. The warm restart is a *degradation* regime for every arm, control included
   (`M0` best `0.135621` vs step-0 `0.130028`; epoch-40 `0.156383`), with
   epoch-to-epoch valid fluctuation over a ~0.02 band.
2. The soup metric mostly cancels that noise: individual window states score
   `0.1356–0.1424`, averages `0.1292–0.1297`.  The control's soup is already
   `−0.00037` from its own start — the protocol noise floor — and the
   candidate deltas sit at/below it.
3. Only the late window (epochs 31–40) separates: all three capacities slow
   the drift (`−0.0043…−0.0067`), largest for G, without improving best
   valid accuracy (all ≥ 0.1356).  This is a stabilisation hypothesis, not a
   bottleneck localisation, and it does not fire the frozen gate.
4. G's added branch is nearly gradient-starved (`max ||grad|| = 0.001` vs
   `0.239` for F, `2.513` for R); its near-zero-init summaries act mainly as
   a small perturbation of the shared reader.

## E. Full winner run

Not run.  `full/NOT_RUN.json`:
`reason = "no candidate passed the preregistered capacity gate"`,
`selection.reason = "NO_CLEAR_CAPACITY_LOCALIZATION"`, `winner = null`.
Consequently no post-full mechanism audit exists
(`mechanism/probes.json` = `{"run": false, ...}`); P1–P5 and the
candidate-specific diagnostics are defined on a frozen winner soup only.

## F. Interpretation

* Direction of the line's bottleneck: **not localised** among
  structure-semantic fusion / static environment composition / invariant
  graph readout under this protocol.
* Money exchange rate of the screen: ≈ 0 (29–34 k added parameters,
  `|Δsoup| ≤ 0.0005`); an honest `Δparams → ΔMAE` claim cannot be made from
  this round.
* CAP-BASE stays `CSSD-q1`; the 0.128–0.130 plateau is untouched.
* The plausible next step is *protocol repair before capacity search*: a new
  preregistration with a discriminative screen (from-scratch epoch-40 gate
  à la CSSD, or a low-rate warm adaptation with an M0-measured noise floor),
  then the same three frozen architectures.  No post-hoc rescue of this round
  is authorised.

## G. Provenance and discipline

* 40/40 focused tests pass
  (`tests/test_e2e_dictenv_capacity_localization_v1.py`), including CAP-BASE
  identity, bit-identical warm load, permutation invariance of F and G, the
  static single-pass contract of R, budget/gate/winner logic, CPU guard and
  official-test blocker.
* Every payload carries `official_test_loaded: false`; the official test
  blocker was invoked in every stage; `docs/luyin/luyin19.txt` untouched.
* Wall clock: preflight + init audit ≈ 20 min; wave 1 (M0/F/R) ≈ 36 min;
  wave 2 (G) ≈ 8 min; total ≈ 65 min of CPU compute for 160 epochs.
