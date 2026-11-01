# PROTOCOL — LCRC Performance Long-Run v1 (Sprint 04)

Round: `lcrc-performance-longrun-v1`.
Track: `tracks/structural-sufficiency/lcrc_performance_longrun_v1/`.
Frozen pre-registration: `analysis/protocol_freeze.json`, written **after** the
P0 tests and the committed chem8 artefact and **before** the first GPU job.

This document fixes every quantity that a score could otherwise influence. It
does not contain results; results live in `REPORT.md`, the stage tables under
`analysis/`, and the per-run records under `analysis/runs/`.

---

## 0. Question and scope

LCRC-A is frozen. The round asks which bottleneck limits it:

* **A — local chemistry is incomplete.** Candidate `CHEMLOCAL8`.
* **B — the global statistics compress too much.** Candidate `RHOSHAPE`.
* **C — the single additive contribution readout is too simple.** Candidate
  `MULTICONTRIB8`.

If none moves the anchors, the answer is `NO_LCRC_B_SIGNAL_THIS_ROUND`.

No new message passing, attention, Graph Transformer, pair interaction, SSL,
prototype dictionary, DIST/PATH/G2L, ZINC architecture search, or large
hyper-parameter search. The frozen S1/S2 definitions, path type, path endpoint
content, one-shot local encoding and additive readout are preserved. Every arm
is a **single zero-initialised extension** of the frozen graph, so step 0 is
BASE (≤1e-6).

## 1. Execution regime

| item | value |
| --- | --- |
| host / node | `res` / `a100-2` (process backend) |
| GPU | 2 × NVIDIA A100-SXM4-40GB, pinned by pools `res-gpu0` / `res-gpu1` |
| driver | 550.163.01 |
| torch / CUDA | 2.5.1+cu124 / 12.4, fp32, no AMP |
| python | 3.12.14 |
| commands | all remote execution through `rr` |
| queue | none; two jobs may run in parallel on different GPUs |

Each `train.json` records hostname, GPU name/count, torch/CUDA, driver regime,
commit, dirty flag, code hashes and timings. Numbers from another regime are
not comparable.

## 2. Data and splits

* **Anchors:** Lipophilicity `random122`, ESOL `random122`, BACE `scaffold122`
  — the frozen published repeat 0 of Sprint 02.
* **P4 transfer:** Lipophilicity `random123` / `random124`, FreeSolv
  `random122`, BBBP `scaffold122`, Tox21 `random122`, ogbg-molhiv official OGB
  scaffold split.
* **P5:** up to three of Lipophilicity/ESOL/BACE with seeds 1 and 2.

All structure caches are label-free. The chem8 channels are a committed,
label-free artefact (`analysis/source/chem8/*.npz`) built once from the frozen
source SMILES; their sha256 is checked at load time. Row ids, atom offsets and
SMILES are re-verified against the frozen cache.

Regression uses Train-only target standardisation. The chem8 extension does not
touch scalars or globals; the Train-only standardiser fingerprint must be
identical to the base view.

## 3. Arms and zero-init

| arm | single change | default params |
| --- | --- | --- |
| `base` | frozen LCRC-A | 247,106 |
| `chem8` | first `atom_encoder` linear 51 → 59; new columns zero | 248,130 |
| `rhoshape` | `rho` input + `max_v h_v`, `mean_v |h_v − mean h|` (128 dims); new columns zero | 255,298 |
| `multicontrib` | `Q: 64→64→8`, graph sum; `delta = MLP([Q_graph, log1p(n_atoms)])`; last layer zero | 252,139 |
| `combo_chem_contrib` | `chem8` + `multicontrib` | 253,163 |
| `combo_chem_rho` | `chem8` + `rhoshape` | 256,322 |
| `combo_rho_contrib` | `rhoshape` + `multicontrib` | 260,331 |

Shared tensors are copied bit-for-bit from a same-seed BASE model; the widened
columns and the `delta` final layer are exactly zero. The P0 tests verify the
step-0 prediction equality, effective gradients, atom-permutation invariance,
single-vs-batch consistency, batch isolation of segment statistics, multi-task
shapes, and checkpoint reload.

## 4. Recipe

Unless the protocol is `hiv_short`, the frozen R3 recipe is used unchanged:

* AdamW (0.9, 0.999), ε 1e-8; peak LR 3e-4 → min 1e-5, per-step cosine;
* weight decay 1e-3, gradient clip 5.0, batch 128, dropout 0.10 at the final
  atom-level `h`, 360 epochs, no soup/EMA/SWA/SSL/early stopping;
* regression MSE on Train-standardised targets; binary BCEWithLogits; Tox21
  masked-cell BCE (frozen definition, unchanged).

**HIV short protocol:** 160 epochs, minimum 60, early-stop patience 30 on
Validation ROC-AUC, cosine horizon 160. Both `HIV-BASE-SHORT` and
`HIV-WINNER-SHORT` run at seed 0. The best-Validation checkpoint is saved even
if training stops early. HIV uses unweighted BCE, matching the historical BASE.

**Checkpoint rule:** lowest Validation RMSE / highest Validation ROC-AUC /
highest Validation macro ROC-AUC; ties break to lower Validation BCE then the
earlier epoch. The selected state is restored and re-verified before any Test
read.

## 5. P1 — screening

9 runs: 3 arms × 3 anchors × seed 0. Each run trains fully, selects one
Validation epoch, restores it, then reads Test once. Report Train/Valid/Test
metrics, best epoch, parameters, wall/GPU time, and deltas vs the historical
BASE (Sprint 02 for the anchors; Sprint 03 for Lipophilicity 123/124).

Signals (not a rigid significance threshold):

* `STRONG_SIGNAL` — ≥2 anchors improve both Validation and Test, or one large
  gain with no catastrophic regression elsewhere.
* `WEAK_SIGNAL` — one anchor, or Valid/Test directions disagree, but the arm is
  worth observing.
* `NO_SIGNAL` — broad regression, or Train-only gain with Valid/Test worsening.

Promote at most **two single arms**.

## 6. P2 — combination

Only if at least two directions show positive signal. Build exactly **one**
combo, preferring the strongest representation candidate + the strongest
readout candidate. Do not build the triple unless all three singles are clearly
positive. If only one arm is positive, skip P2 and promote it. At most 3 runs on
the anchors, seed 0.

## 7. P3 — capacity check

If P1/P2 yields a clear winner, train `BASE-SCALE-M` and `WINNER-SCALE-M` on
Lipophilicity and ESOL (4 runs). The start config is:

```yaml
width: 80
stat_rank: 40
local_hidden: 230
atom_hidden: 144
atom_out: 104
bond_hidden: 176
bond_out: 104
rho_hidden: 80
```

`local_hidden` was nudged from 224 to 230 only to bring BASE-SCALE-M into the
pre-registered 320k–450k band (actual 320,230). No width grid search. The
comparison is scaled BASE vs scaled winner, not scaled winner vs unscaled BASE.

## 8. P4 — transfer

Freeze `WINNER_ARCH` after P1–P3 and do not change it. Seed 0 on
Lipophilicity `random123/124`, FreeSolv, BBBP, Tox21, and ogbg-molhiv (with the
matched short BASE). Seven runs.

## 9. P5 — seed confirmation

Only if the winner is worth confirming. Up to three tasks (default
Lipophilicity, ESOL, and one indicated by P4), winner seeds 1/2 plus matched
BASE seeds 1/2 where absent. At most 12 runs; do not fill the budget for its own
sake. Do not report only the best seed.

## 10. Test policy and budget

Test is a **DEVELOPMENT_TEST**. It is read once per run, only after the
best-Validation checkpoint is frozen. It may inform the next stage's go/no-go;
it never selects a checkpoint, epoch, learning rate, dropout, width, or seed.
The append-only ledger is `analysis/test_access_ledger.json`. Any architecture
choice informed by Test is labelled `TEST_USED_FOR_DEVELOPMENT`.

Hard budget: 15.0 A100 GPU-hours, 8 h wall time. Priority: P1 → P2 → P4 →
P3 → P5. Keep time for sync, analysis, report and Git.

## 11. Decision rules

* CHEMLOCAL8 wins → recommend **LCRC-Chem**.
* RHOSHAPE wins → recommend **LCRC-Stats**.
* MULTICONTRIB8 wins → recommend **LCRC-Composition**.
* COMBO clearly best → name **LCRC-B** and decompose the two contributions.
* No stable multi-task gain → `NO_LCRC_B_SIGNAL_THIS_ROUND`.

Single-dataset or single-seed improvements do not establish LCRC-B.
