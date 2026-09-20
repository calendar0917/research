# PSD-v0 — analysis (Pair-Structural representation + Function-Preserving Dictionary)

Pre-registration: [zinc_pair_dict_v0_preregistration.md](zinc_pair_dict_v0_preregistration.md)
(frozen before any run). Code: `code/run_zinc_pair_dict_v0.py`; tests
`tests/test_zinc_pair_dict_v0.py` (12 pass, fresh-clone-safe, CPU-only). Regime:
deterministic A100, canonical PyG ZINC `subset=True` official **train 10 000 /
valid 1 000**; **official test never loaded** (`official_test_loaded: false`).
Commit `e1bef7c`. Seed 0. Canonical OPTIMIZED_PROTOCOL (Adam 1e-3, wd 1e-5, L1,
clip 5, best-monitor checkpoint, equal-weight Top-5 soup).

**Frozen verdict (first line):**

> **Pair-state structural representation is not viable (dominated by the raw node
> baseline).**

**Next decision (frozen):** `STOP the pair-state structural encoder; do not
start the dictionary (Q2).`

---

## 0. What was asked

After GSCN-v0 rejected the dictionary-core architecture
(`decision-gscn-v0-reject-dictionary-core-20260920`), this round asked, in order:

* **Q1** — can a *relation-preserving pair-state* structural encoder learn a
  task-discriminative `u_G ∈ R^d` from **raw** input, at least matching a raw
  node-MPNN of comparable budget? (The hypothesis was that GSCN lost information
  by pooling neighbours into one node vector *before* coding.)
* **Q2** — only if Q1 passes: does a **function-preserving** sparse dictionary
  inserted on `u_G` beat matched controls (dense-restart, fixed-identity,
  parameter-matched generic Top-k) at equal budget?

## 1. Input discipline / parameter accounting (frozen)

Raw atom category + raw bond category + adjacency only (asserted in the runner).
Real ZINC category counts: 21 atom / 3 bond.
Parameters (canonical, `params.json`):

| arm | params | note |
|---|---:|---|
| `raw` (B-Null-Raw, d=64, L=4) | **55 681** | imported verbatim from `run_gscn_v0.py::RawBaseline` |
| `pair` (d=64, L=3 relation layers) | **76 609** | pair-state structural encoder |
| `raw_wide` (d=76, L=4) | **77 977** | node-MPNN parameter-matched to `pair` (1.75 % mismatch) |

The primary comparison is `pair` vs `raw_wide` (parameter-matched); `raw` is the
canonical reference.

## 2. Stage 0 — data-free correctness (GPU)

All passed (`stage0_tests.json`): layer-state permutation equivariance max
9.5e-7, graph-prediction invariance 3.7e-9, batching invariance 3.7e-9, padding
invariance 0.0, graph-order invariance 3.7e-9, gradients nonzero for every
structural tensor (`E_A, E_R, W_init, φ, ψ, η, W_read, head`) at every layer, no
absolute-position parameter, official test blocked.

Synthetic mechanism sanity (`synth_mechanism.json`), mechanism-only: on C6
(diameter 3) vs 2·C3 (diameter 2), two 2-regular 1-WL-equivalent graphs, the raw
node-state multiset gap is 0.0 while the pair-state projection gap is 0.051 and
the graph-representation gap is 0.043 — the pair update does build relation
structure that a node-MPNN cannot distinguish.

## 3. Stage 1 — 128-graph overfit sanity

First run (canonical batch 128, 60 epochs) is **not** the evidence: with batch
128 over 128 graphs there is exactly **one gradient step per epoch**, so 60
epochs = 60 steps. Both arms still fell (pair 1.36 → 0.61, raw 3.18 → 0.71) and
failed the `≤ 0.15` gate — an **optimization-budget artifact**, not capacity.

**One targeted revision** (batch 16 → 8 steps/epoch, 300 epochs = 2400 steps;
architecture and everything else frozen): both arms **pass** the gate.

| arm | epoch-1 train MAE | best train MAE | gate |
|---|---:|---:|---|
| `pair` | 1.310 | **0.0626** (=0.05×) | GO |
| `raw` | 2.195 | 0.0694 | GO |

Q1 Stage 1 **GO**: the pair encoder has the capacity to fit the task.

## 4. Stage 2 — train-only structural screen (no official valid)

Deterministic official-**train** split: 2048 `train-dev` + 512 `train-monitor`
(seed 20260922), batch 64, seed 0. Official valid is *not* used.

| arm | params | best monitor MAE | Top-5 soup monitor MAE | epochs | wall | train@last |
|---|---:|---:|---:|---:|---:|---:|
| `pair` (80 ep) | 76 609 | 0.5177 | 0.5043 | 80 | 133 s | 0.285 |
| `pair` (200 ep, early-stopped) | 76 609 | **0.5038** | 0.4955 | 143 | 149 s | 0.195 |
| `raw_wide` | 77 977 | **0.4115** | 0.4094 | 80 | 72 s | 0.424 |
| `raw` | 55 681 | **0.4152** | 0.4105 | 80 | 84 s | 0.255 |

`pair − raw_wide` (monitor) = **+0.106** (80 ep) / **+0.088** (best, 200 ep);
`pair − raw` = +0.103 / +0.089. The gap is an order of magnitude above the
pre-registered practical-effect scale (`0.005`), and far above seed noise. The
extended 200-epoch run **plateaued early** (no monitor improvement after epoch
103, early stop at 143) while its train MAE kept falling (0.195) — the pair
encoder **overfits the 2048 training graphs** (train 0.195 vs monitor 0.513),
whereas the raw node baselines do not (train 0.26 / monitor 0.42). The pair
encoder is also ~1.8–2.1× slower per epoch.

**Early-stop rule fired:** the pair encoder is clearly worse than the raw node
baseline, so no full-10000 / longer run is warranted (§24 of the round brief).

## 5. Mechanism diagnosis (why)

* Frozen pair-field effective rank: 8.5–11.7 at layers 1–2, 4.6 at layer 3 —
  the pair states are not numerically degenerate.
* Graph readout effective rank: `u_G` 1.46, `mean_pairs` 2.07, `mean_diag` 1.94.
* **But** the raw baselines' sum-pooled graph vectors are also low rank
  (`raw` 2.46, `raw_wide` 2.18) and still generalise to 0.41. **Low-rank graph
  readout is therefore not the differentiator and not a numerical pathological
  state.** The distinction is inductive bias / effective capacity: the
  higher-order pair contraction fits the 2048 training graphs much better while
  generalising worse.

No clear mechanism *bug* was found (Stage 0 clean, mechanism active, no norm
explosion, no collapse unique to `pair`). Per §13 the failure was therefore **not**
subjected to architecture fishing (no residual/gate/attention/size/λ changes).

## 6. Q2 — dictionary: not started

The pre-registration gates Q2 on Q1. Q1 Stage 2 **fails**, so the
function-preserving dictionary insertion (`preserve`/`dictrun`) was **not run**.
No `z_G`, no matched controls, no representation-budget table — none is reported
or claimed.

## 7. Answers

* **Q1** — can a raw pair-state structural encoder match a node-MPNN of equal
  budget? **No.** It is dominated by ≈0.09 monitor MAE and overfits.
* **Q1 (raw input in general)** — the *raw node-MPNN* itself does learn a usable
  representation at this scale (≈0.41 monitor on 512 held-out train graphs, and
  0.195 official-valid soup at full data in GSCN-v0); the **pair lift is the part
  that does not work**.
* **Q2** — **not asked / not answered.** The dictionary question remains open in
  principle but is not reachable through this encoder.

## 8. Hypothesis mapping

* **H4** — "the pair structural representation itself is the wrong direction" —
  **best supported**.
* **H1** — a raw representation is learnable (the node-MPNN works), but the
  relation-preserving *pair* representation adds negative value, so H1 is not the
  reading.
* H3 (dictionary unique benefit) — untested; cannot be claimed.

## 9. Scope / what cannot be said

* 2048 train-dev / 512 train-monitor, seed 0, ≤200 epochs, batch 64; **official
  valid and test never used**. This is a *screen*, not a full-data result.
* Cannot say "pair-state encoders cannot work on ZINC in general" — only that
  **this** raw-input pair formulation at this capacity/budget is dominated by a
  node-MPNN. A data-scale or capacity explanation (higher expressivity needs more
  data) is plausible but **untested by design** (the brief forbids the full-10k
  run once the 2048 gap is clear).
* Cannot say anything about the dictionary: **no dictionary experiment was run.**

## 10. Decision

**STOP the pair-state structural encoder. Do not start the dictionary round.**
`revisit_if`: a *new* pre-registration proposes a structurally different lift (a
different object class or a differently-justified parameterisation), or supplies
the full-10000 / effective-capacity-matched evidence that the pair lift's 2048
overfitting is purely a data-size effect; or the round is re-scoped to the
dictionary question on the **raw node** representation (`raw_wide`), which is the
one raw encoder that does work.

## Artefacts

`results/zinc_pair_dict_v0/`: `stage0_tests.json`, `synth_mechanism.json`,
`params.json`, `overfit_pair.json`, `overfit_raw.json`, `screen_pair.json`,
`screen_raw.json`, `screen_raw_wide.json`, `curves/`, `states/`,
`RESULTS_SUMMARY.md`.
