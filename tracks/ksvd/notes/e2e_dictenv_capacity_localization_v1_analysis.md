# Analysis — `e2e_dictenv_capacity_localization_v1`

Capacity Localization + One-Winner Full Training round, ZINC
dictionary-environment line, CPU only, official test never loaded.
Preregistration:
`tracks/ksvd/notes/e2e_dictenv_capacity_localization_v1_preregistration.md`
(sha256 `f5c4a8d2e489…`, implementation frozen at commit `becc6f6`).

## 1. Question

Which expressive capacity does the CAP-BASE model (`CSSD-q1`: C6 clean mask,
q1 common structural coordinate, paired node/edge binding, full relation,
first/second-moment readout; seed-0 soup valid MAE `0.130028`) actually lack?

* **F** — structure-semantic fusion capacity (multi-rank factorized bilinear
  residual, `H = 4`, `d_h = 24`; +29 568 params, total 127 295);
* **R** — static environment-environment composition capacity (two residual
  FiLM blocks on the pair state, hidden 256; +34 400 params, total 132 127);
* **G** — graph-level invariant readout capacity (gated DeepSets summaries,
  dim 192, node + pair; +30 336 params, total 128 063).

Added-parameter ratio max/min `1.1634 ≤ 1.5`; every candidate inside the
preferred 120 k–160 k total band.

## 2. Frozen protocol actually executed

1. `preflight` — CAP-BASE soup MAE recomputed `0.130028` (matches the frozen
   value to `< 5e-7`); preregistration hash + commit recorded.
2. `init-audit` — step-0 valid MAE, prediction shift and single-batch
   capacity-parameter gradients for M0/F/R/G.
3. Wave 1 (`M0`, `F`, `R`; 3 processes × 4 threads) and wave 2 (`G`; 1 × 4):
   each arm fresh `Adam(lr = 1e-3, wd = 1e-5)`, batch 128, clip 5.0,
   `H1_LAMBDA * reconstruction` term, exactly 40 epochs, warm-started from the
   same CAP-BASE seed-0 Top-5 soup, epoch-21–40 window states retained.
4. `summarize` — primary metric = Top-5-by-valid soup over epochs 21–40;
   secondary = mean valid MAE over epochs 31–40; frozen gate `Δsoup ≤ −0.004`
   **and** `Δlast10 ≤ −0.003`; frozen tie order F > R > G within 0.002.
5. Gate fired for nobody → `full/NOT_RUN.json`; no 320-epoch run; no
   post-full mechanism probes (they are defined on a frozen winner soup only).

## 3. Initialization audit (step 0, before training)

| candidate | step-0 valid MAE | Δ vs CAP-BASE | mean shift | max shift | capacity grads |
|---|---|---|---|---|---|
| M0 | 0.130028 | 0.000000 | 0.00000 | 0.00000 | — (control) |
| F | 0.131102 | +0.001074 | 0.01003 | 0.05211 | 18/18 finite > 0 |
| R | 0.130860 | +0.000832 | 0.01741 | 0.12942 | 20/20 finite > 0 |
| G | 0.129941 | −0.000087 | 0.00113 | 0.00300 | 8/8 finite > 0 |

G is the only candidate that is (negligibly) better than CAP-BASE at step 0
because the appended summary columns start near zero; F and R perturb the
frozen optimum slightly. All shifts are far below the +0.03 audit bound, so
the audit passed for all four candidates.

## 4. Screening result (frozen primary and secondary metrics)

| candidate | params | step-0 MAE | best valid | best epoch | soup (21–40) | last-10 mean | Δsoup vs M0 | Δlast10 vs M0 | branch grad | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 97 727 | 0.130028 | 0.135621 | 21 | 0.129662 | 0.154314 | 0.000000 | 0.000000 | 0.000 | CONTROL |
| F | 127 295 | 0.131102 | 0.136613 | 37 | 0.129464 | 0.150048 | −0.000198 | −0.004266 | 0.239 | NO_CAPACITY_SIGNAL |
| R | 132 127 | 0.130860 | 0.138940 | 28 | 0.129168 | 0.149848 | −0.000494 | −0.004466 | 2.513 | NO_CAPACITY_SIGNAL |
| G | 128 063 | 0.129941 | 0.135855 | 27 | 0.129663 | 0.147582 | +0.000002 | −0.006732 | 0.001 | NO_CAPACITY_SIGNAL |

Gate (frozen): `Δsoup ≤ −0.004` **and** `Δlast10 ≤ −0.003`.
All three candidates pass the secondary condition and fail the primary one by
an order of magnitude (`Δsoup` between `+0.000002` and `−0.000494`, i.e. 8–20×
short of the required `−0.004`). Nobody passes both → frozen branch 1:
**`NO_CLEAR_CAPACITY_LOCALIZATION`**, `full/NOT_RUN.json`, stop.

## 5. Why the screen could not discriminate (post-hoc, descriptive)

The frozen decision is the null result; the following facts explain *why* the
screen was uninformative and are recorded for the next preregistration, not as
a rescue of this round.

1. **The warm-start adaptation is a degradation regime.** Every arm, including
   the pure M0 continuation control, becomes worse than the checkpoint it
   started from as soon as training resumes: M0 step-0 `0.130028` → best over
   the 40 epochs `0.135621` (epoch 21) → epoch 40 `0.156383`. The same holds
   for F (`0.136613` / `0.156190`), R (`0.138940` / `0.142631`) and G
   (`0.135855` / `0.136734`). Fresh `Adam(lr = 1e-3)` on a 320-epoch soup
   optimum overfits the 10 000-molecule train split; valid MAE fluctuates
   epoch-to-epoch over a ~0.02 band.
2. **The soup metric is dominated by variance reduction.** Individual window
   states have MAE `0.1356–0.1424` (far above their Top-5 average), so the soup
   mostly cancels the adaptation noise. The control's own soup
   (`0.129662`) is already 0.00037 *below* its starting point — that is the
   protocol's noise floor, and the candidate `Δsoup` values (−0.0002, −0.0005)
   are at or below it. The frozen `−0.004` threshold correctly demanded a
   real effect; none exists.
3. **Only the secondary late-window metric separates the arms.** On epochs
   31–40, M0 degrades to `0.154314` while F `0.150048`, R `0.149848` and
   G `0.147582` degrade less (Δ `−0.0043`, `−0.0045`, `−0.0067`). This is a
   stabilisation/regularisation effect in the overfitting regime — the added
   capacity slows divergence from the soup optimum — and it does not improve
   the best achievable accuracy (best valid still ≥ 0.1356 in every arm).
   Since the frozen gate requires both metrics, this does **not** fire the gate
   and is not used for selection.
4. **Candidate G's added branch is almost gradient-starved.** The maximum
   capacity-parameter gradient norm is `0.001` for G versus `0.239` for F and
   `2.513` for R (step-0 and per-epoch logs agree). The near-zero-init gated
   summaries are nearly inactive; G's trajectory differs from M0 mainly
   through a small perturbation of the shared reader, which nevertheless
   changes the late-window drift.

## 6. What this round establishes and what it does not

Establishes (frozen decision):

* under the preregistered short warm-start adaption, none of the three
  capacity directions moves the retained-window soup by the required `−0.004`;
  the round cannot localise the bottleneck to structure-semantic fusion,
  environment composition or graph readout;
* the CAP-BASE (`CSSD-q1`) checkpoint is adopted unchanged as the line base —
  the round is not authorised to change it, and no full 320-epoch run was
  bought;
* the `Δparams → ΔMAE` exchange rate of this screen is ≈ 0 (29–34 k added
  parameters, `|Δsoup| ≤ 0.0005`).

Does not establish:

* that F/R/G are useless capacities — the screen was run in a degradation
  regime where the control itself loses ~0.0056–0.026 of valid MAE, so a
  capacity could in principle help without crossing the frozen `−0.004` soup
  threshold; this protocol simply had no power to see it;
* any claim about the 320-epoch from-scratch behaviour of the candidates (no
  full run was authorised);
* anything about the mentor's target band (0.06–0.07); the line stays on the
  0.128–0.130 plateau.

## 7. Provenance

* preregistration sha256 `f5c4a8d2e489fbe1…`, snapshot
  `results/e2e_dictenv_capacity_localization_v1/preregistration_snapshot.json`;
* implementation freeze commit `becc6f6` (module, runner, tests, prereg);
* `preflight.json` (CAP-BASE MAE `0.130028` reproduced),
  `parameter_budget.json` (ratio 1.1634), `init_audit.json`,
  `screening/decision.json`, `screening/screening_summary.csv`,
  `screening/*/result.json` + `curve.csv` + `soup.json`,
  `full/NOT_RUN.json`, `mechanism/probes.json` (`run: false`),
  `analysis_tables.md`, `summary.json`;
* 40/40 focused tests
  (`tracks/ksvd/tests/test_e2e_dictenv_capacity_localization_v1.py`);
* `official_test_loaded = false` in every payload; `CUDA_VISIBLE_DEVICES=""`,
  `torch.device("cpu")` everywhere; `docs/luyin/luyin19.txt` untouched.

## 8. Next-round shape (for a *new* preregistration only)

The failed ingredient is the screening adapter, not (demonstrably) the
capacities. A future round should first make the screen discriminative, e.g.:

1. a from-scratch short screen with the epoch-40 checkpoint gate used by the
   CSSD round (live-model dictionary/usage diagnostics + valid MAE), or
2. a low-rate warm adaptation (lr `1e-4`, no fresh-Adam jump) with the same
   frozen F/R/G architectures and a preregistered noise floor measured on M0
   (e.g. require `Δsoup ≤ −4 × control spread`), or
3. a direct from-scratch one-winner design that does not rely on a warm-start
   screen at all.

The secondary stabilisation signal (Δlast10 ≤ −0.004 for all three, largest
for G) is a hypothesis to test in such a round — not evidence for a capacity
bottleneck, and never combined with this round's numbers as a selection.
