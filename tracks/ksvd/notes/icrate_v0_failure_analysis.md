# I-CRATE-v0 — seed-0 feasibility result and failure analysis (ZINC)

**Round.** `I-CRATE-v0` (incidence-structured white-box dictionary transformer).
Pre-registration: `notes/icrate_v0_preregistration.md` (frozen before the run).

**Status.** The formal seed-0 **training** completed (206/240 epochs, early stop,
best checkpoint epoch 166). The pre-registered gate classifies the result as
**Case A (clear failure)**: valid is far above the 0.20 threshold. The final
mechanism audit **crashed** after training (a variable-token-count concatenation
bug in the audit), so the Top-5 soup value and the full-data incidence
intervention were never materialised. The bug was fixed and the run relaunched,
but the corrective re-run was **stopped by user request** (the result is clearly
mediocre), so the authoritative full-data mechanism audit is **not available**.
The mechanism evidence below therefore comes from (a) the per-epoch probes in the
formal training log and (b) the full audit of the 64-train-graph overfit run.

**Official ZINC test was never loaded** by any stage (`test_access=blocked`); the
official test read count is unchanged.  This is a `test_policy=terminal`
protocol, mode `screen`.

---

## 1. Provenance

| item | value |
|---|---|
| formal run id (training) | `20260919-190822-dfe9337d` (pulled to `results/icrate/formal_seed0/`) |
| formal code commit | `2479ad2e555774dbf6b2feb0bef79f47250de1af` |
| 64-graph overfit run | `20260919-190242-16c1555e` (pulled to `results/icrate/overfit64/`) |
| execution regime | A100-SXM4-40GB (GPU 1), CUDA 12.4, torch 2.5.1 |
| seed | 0 |
| data | official PyG ZINC `subset=True`; train 10 000 / valid 1 000; test never loaded |
| protocol | `zinc-context-gap` (`test_policy=terminal`), mode `screen` |
| parameters | **61 377** (tokenizer 1 152; 4 structural layers ×11 712; global MSSA 2 496; global dict 4 608; head 6 273) |
| peak GPU memory | 162 MB |
| wall time (training, 206 epochs) | ≈ 4 300 s (72 min); ~26 s/epoch |

**Integrity note.** The first formal run crashed in the evaluation-only audit and
its trained weights were *not* checkpointed before the audit, so they were lost;
the audit was fixed and training relaunched (deterministic — identical log). This
was an avoidable infrastructure defect; the runner now checkpoints
`soup_state.pt` + `training_history.json` immediately after the training loop and
records `audit_error` instead of aborting.

## 2. Direct results (valid only)

| metric | I-CRATE-v0 | WG-ICSC-v0 | A100 reference (B-Full / A0) |
|---|---|---|---|
| best-checkpoint valid MAE | **0.398928** (epoch 166) | 0.505933 | — |
| Top-5 soup valid MAE | not materialised (audit crash) | 0.505752 | 0.119818 / 0.124704 |
| train task MAE (epoch 1 → end) | 1.05952 → **0.35482** (epoch 200) | 1.427 → 0.501 | — |
| valid MAE (epoch 1 → end) | 0.81864 → **0.41754** (epoch 200) | 1.297 → 0.511 | — |
| epochs run | 206 (early stop, best 166) | 240 | — |

The valid best-checkpoint improved by ~0.107 absolute (≈21 % relative) over
WG-ICSC, but remains **≈3.4×** the matched A100 reference and **above the 0.20
Case-A gate**. train (0.355) and valid (0.418) are close and both high — the run
is still a representation/optimisation **underfit**, just less severe than
WG-ICSC.

## 3. 64-train-graph overfit smoke (pre-registration §7)

64 train graphs, early stopping disabled, 400 epochs: train task MAE
1.268 → **0.177** (valid 0.808, i.e. memorisation of a tiny set). No NaN. This
passes the §7 overfit gate: the architecture *can* fit a small set, so the
full-data plateau (train 0.355) is a **capacity/optimisation limit on the task
distribution**, not a broken optimiser. Peak memory 162 MB.

## 4. Mechanism audit

The full-data audit did not run (see §1). Numbers below are from the **64-graph
overfit model** (fully trained) plus the formal run's **per-epoch probes**.

### 4.1 Incidence intervention (60 graphs; within-graph endpoint rewiring)

| quantity | value |
|---|---|
| Δα relative change (mean / median / p10 / p90) | **0.425 / 0.395 / 0.119 / 0.867** |
| Δy (mean abs prediction change) | 0.712 |
| layer-wise ΔZ relative change L0→L4 | 0.00 → 0.377 → 0.694 → 0.815 → 0.891 |

The graph code α_G changes by ~**42 %** under a pure bond↔endpoint reassignment
(vs **1.5 %** for WG-ICSC). The incidence signal now propagates materially into
the graph representation — this is the single clearest architectural win over
WG-ICSC. (L0 = 0 by construction: Z⁰ is the raw token embedding.)

### 4.2 Structural MSSA vitality

Per-layer r_mssa = ‖ΔMSSA‖_F/‖Z‖_F: **0.29–0.44** (64-graph model); the formal
probes show L0 r_mssa rising 0.308 → 0.555 over training. Attention is
non-degenerate (entropy ≈0.34–0.65; max weight ≈0.62–0.84). Structural
communication is active, not switched off.

### 4.3 Token ODL sparsity (per layer, valid tokens)

Nonzero fraction **0.409–0.430** per layer; dead atoms **0/96 every layer**;
mean active atoms/token ≈40.5; top-1 mass ≈0.076, top-5 mass ≈0.303; coefficient
entropy ≈3.40. The formal-run probe shows L0 nonzero fraction falling 0.437 →
**0.205** by epoch 200 (getting sparser with training). Healthy: not >90 % dense,
not <2 % collapsed.

### 4.4 Global dictionary / code

α_G nonzero fraction **0.492**; active atoms/graph 47/96; dead global atoms
**13/96**; amplitude mean 1.23 / max 15.3. The final code is genuinely sparse and
not collapsed to a constant.

### 4.5 Dictionary utilisation / rank / coherence

All nine dictionaries (D_a, D_s per layer, D_G): effective rank **41–42 / 96**
(rank ratio ≈0.43–0.44), max pairwise coherence **0.47–0.63**, near-collinear
fraction **0.000** (vs WG-ICSC's D_V coherence max **1.0** with duplicate atoms).
No dictionary collapse.

### 4.6 Gradient vitality

Every block (tokenizer E_V/E_E, U^ℓ, D_a^ℓ, D_s^ℓ, U^G, D_G, head) has finite,
non-zero gradient after one task backward. No dead block.

## 5. Interpretation

* **Supported:** the architecture is stable, the incidence mechanism is truly
  active (α rewiring sensitivity 42 % vs 1.5 %), the token ODL is genuinely
  sparse and non-collapsed, the dictionaries are full-rank-ish and
  non-degenerate, and all parameters receive gradient. I-CRATE fixes the two
  concrete WG-ICSC defects (assignment-invariant readout; collapsed dictionaries).
* **Supported:** the 64-graph overfit shows the forward pipeline can fit a small
  set (train 0.18), so the full-data plateau is a capacity/distribution limit of
  the **fixed sparse forward operators**, not a broken optimiser or a dead
  mechanism.
* **Not supported:** competitiveness. Best valid 0.399 is ≈3.4× the matched A100
  reference and above the 0.20 gate; train≈valid≈0.40 is still severe underfit.
* **Caveat (honesty):** the full-data audit (soup, incidence intervention,
  dictionary spectra) never executed; the mechanism numbers are from a 64-graph
  model and from per-epoch probes. A full-data mechanism audit remains owed if
  the line is revisited.
* Not tested: controls A/B (gated on a Case-B-or-better main run).

## 6. Answers to the six required questions

1. **Q1 — does it solve WG-ICSC's underfit?** Partially. Best valid 0.399 vs
   0.506, but train 0.355 / valid 0.418 both high ⇒ severe underfit persists
   (just less severe). The representation still cannot fit 10 000 graphs.
2. **Q2 — does incidence survive to α_G?** Yes, materially: within-graph
   rewiring moves α_G by ~42 % (WG-ICSC: 1.5 %), growing monotonically through
   the layers. (Full-data number not measured.)
3. **Q3 — are the ODL coefficients sparse and non-collapsed?** Yes: 0.41–0.43
   nonzero per layer, 0 dead atoms, healthy concentration; on full training L0
   sparsifies to ≈0.20.
4. **Q4 — is the dictionary alive?** Yes: eff. rank ≈42/96, coherence max
   ≈0.5–0.6, no near-collinear pairs, live gradients.
5. **Q5 — continue value?** No, by the pre-registered gates: valid ≫ 0.20 ⇒
   **Case A**. (Even a generous soup≈best ≈0.399 does not approach 0.13/0.15.)
6. **Q6 — controls?** Not run: the pre-registration authorises A/B only after
   the seed-0 main passes Case B. The 64-graph audit already shows *both*
   mechanisms (incidence and dictionary) are active, so the limit is capacity,
   not mechanism absence.

## 7. Next decision

**Stop the I-CRATE-v0 performance line in its current form.** The mechanism is
real and the WG-ICSC defects are repaired (incidence reaches α_G; dictionaries
do not collapse), but the fixed forward pipeline cannot fit the ZINC
distribution — train and valid plateau together at ~0.40, ≈3.4× the reference.
Per the frozen gates this is Case A: no sweeps of d / L / heads / overcomplete
ratio / λ / readout, and no gated controls.

The falsifiable remainder (not authorised here) would be an architecture that
*changes the capacity object* — e.g. letting the sparse structure carry more
than 96 global atoms or letting the graph seed retain more than one pooled
vector — but that is a new mathematical object requiring its own pre-registration.

## 8. Reproduction

```
uv run research run zinc_icrate --mode screen --set model.device=cuda
# 64-graph overfit smoke:
uv run research run zinc_icrate --mode scratch \
  --set data.limit_train=64 --set data.limit_valid=64 \
  --set model.epochs=400 --set model.patience=400 --set model.device=cuda
```
Artifacts: `results/icrate/formal_seed0/` (manifest, config, stdout; `results.json`
absent due to the audit crash), `results/icrate/overfit64/artifacts/`
(`results.json`, `training_curve.png`, `vitality_spectra.png`).
