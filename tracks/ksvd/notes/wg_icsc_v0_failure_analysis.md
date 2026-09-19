# WG-ICSC-v0 — seed-0 feasibility result and failure analysis (ZINC)

**Round.** `WG-ICSC-v0` (whole-graph incidence-coupled sparse coding).
Pre-registration: `notes/wg_icsc_v0_preregistration.md` (frozen before the run).

**Status.** One formal seed-0 run completed; the pre-registered gate classifies
it as **Case A (clear failure)**.  Two mechanism controls (No-incidence /
Dense) were **not** run: the pre-registration only authorises them after the
seed-0 main passes the basic feasibility gate.  The run was stopped after the
main result; no tuning was performed.

**Official ZINC test was never loaded** by any stage.  Test access was
`blocked`, and the official test read count is unchanged.

---

## 1. Provenance

| item | value |
|---|---|
| formal run | `runs/2026/09/19/20260919-170642-bc1d4aef` (pulled to `results/wg_icsc/formal_seed0/`) |
| code commit (formal run) | `6fcb3c4d99b304292d45973111b5321500597596` |
| later fix (intervention scope) | `439ac1b` — see §6 caveat |
| execution regime | A100-SXM4-40GB (GPU 1), CUDA 12.4, torch 2.5.1 |
| seed | 0 |
| data | official PyG ZINC `subset=True`; train 10 000 / valid 1 000; test never loaded |
| split fingerprint | `58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a` |
| protocol | `zinc-context-gap` (`test_policy=terminal`), mode `screen` |
| training protocol | Adam lr 1e-3, wd 1e-5, batch 128, L1, clip 5.0, max 240 epochs, patience 40, best-valid checkpoint, fixed equal-weight Top-5 soup |
| peak GPU memory | 198 MB |
| wall time | 4095 s (68 min) |

Frozen graph schema (confirmed from the loader, `notes/wg_icsc_v0_preregistration.md` §2):
`d_V = 21` atom categories, `d_E = 3` bond categories, each undirected bond
stored as two directed entries with equal `edge_attr`, no self-loops, no
multi-edges.

## 2. Pre-registered sparsity calibration (train-only, unlabelled, at initialisation)

256 official **train** graphs, model at initialisation, no labels.  The x1 grid
was uniformly too dense (whole active-row fraction 0.305–0.836 > 0.30), so the
whole grid was scaled by 3 once, as pre-registered.  Frozen:
**λ₁ = 0.03, λ_g = 0.15** (chosen by the documented rule: discard rows with
dead atoms > 0.60·192, then minimise `dist(whole_active,[0.15,0.30]) +
0.5·dist(element_in_active,[0.20,0.50])`, ties by fewer dead atoms).

| grid | λ₁ | λ_g | whole active | node | edge | elem-in-active | dead/192 |
|---|---|---|---|---|---|---|---|
| x1 | 0.10 | 0.10 | 0.305 | 0.199 | 0.515 | 0.681 | 78 |
| x3 | 0.03 | 0.15 | **0.296** | 0.189 | 0.512 | 0.749 | 83 |
| x3 | 0.09 | 0.15 | 0.232 | 0.112 | 0.471 | 0.739 | 116 |
| x3 | 0.30 | 0.06 | 0.178 | 0.064 | 0.407 | 0.560 | 126 |

The two calibration targets were **not jointly reachable** inside the permitted
grid (a documented calibration finding); the chosen row-support target is met
(0.296), the element-in-active target (20–50 %) is not (0.749).

## 3. Direct results (valid only)

| metric | value |
|---|---|
| best-checkpoint valid MAE | **0.505933** |
| Top-5 soup valid MAE | **0.505752** |
| best epoch | 230 / 240 |
| top-5 epochs | 230, 234, 210, 211, 215 |
| trainable params | **23 489** (D_V 2 688, D_E 192, W 8 192, head 12 417) |
| reference A100 soups (same protocol, different archs) | B-Full 0.118, A0 0.125 (valid Top-5 soup) |

The run is a clear underfit: train task MAE 1.427 → **0.501** and valid MAE
1.297 → **0.511** plateau together; the gap between train and valid stays
≈ 0.01.  Valid soup is ≈ **4×** the matched strong A100 references and far
above the 0.15 Case-A threshold.

## 4. Mechanism audit

### 4.1 Q1 — is the representation actually sparse?

Per-graph over the final Top-5 soup model (train-128 diagnostics + 1000 valid
graphs):

* active row fraction at epoch 1 → 240: whole 0.369 → **0.515**, node → 0.503,
  edge → 0.535 (calibrated 0.296 at init; training pushes codes denser);
* element-in-active fraction ≈ 0.72 (dense end);
* dead atoms over 1000 valid graphs: **node 30/128, edge 24/64**;
* node activation frequency mean 0.506 / median 0.539; edge mean 0.531 /
  **median 0.978** (when an edge atom is active it is active on almost every
  bond of the molecule);
* dictionary coherence max = **1.0** for both D_V and D_E (duplicate /
  collinear columns exist); D_V effective rank 17.3 / 21, D_E effective rank
  2.88 / 3, W effective rank 12.5 / 64.

Verdict: the codes are genuinely sparse-ish (about half the dictionary active
per graph, real dead atoms) but drift toward the dense end during training; the
node dictionary partially collapses (coherence max 1.0, 30 dead atoms).

### 4.2 Q2 — does sparse inference use graph structure?

* **Composition-gradient vitality** (per inference step, final model):
  `‖g_comp‖/‖g_attr‖` mean **0.245**, median **0.123**, p10 0.027, p90 0.484.
  The composition gradient is ~10× smaller than the attribute gradient, but it
  is **not** dead (not the two-orders-of-magnitude failure mode).
* **Assignment intervention**: re-assigning bond objects to different endpoint
  pairs changes the graph code α by only **1.5 %** (norm-relative mean 0.0153,
  median 0.0146, p10 0.0099, p90 0.0216) while changing the prediction by
  **0.448** (≈ 52 % of the valid error scale).

Verdict: the pipeline *is* wired to the true incidence (composition residual is
nonzero and the energy decrease uses it; the head is sensitive to the
assignment), but the **graph code α — a row-L2-norm readout — is nearly
assignment-invariant by construction**, so only a weak incidence signal reaches
the head.  By the pre-registered criterion "assignment shuffle barely changes
the code ⇒ mechanism not really used", the incidence contribution is **weak**.

### 4.3 Q3 — is the solver actually a solver?

Final model, T=8, train-128 subset: the lower-level energy is strictly
non-increasing across every block update (1.0000 → 0.6406), with node
reconstruction 0.500 → 0.363, edge reconstruction 0.500 → 0.043, composition
0.000 → 0.012; active-row fractions rise monotonically 0 → 0.502 / 0.535.

But T=8 is **not converged**: evaluation-only T=16 continues to decrease the
energy (0.6406 → 0.6032) and moves α by **16.6 %** relative.  So the solver
behaves like a (slowly converging) optimisation-derived solver, but the frozen
budget T=8 under-solves the lower-level problem.

### 4.4 Q4 — is the performance in a continue-worthy range?

No.  Valid Top-5 soup **0.5058** (only a small gain over a weak best
checkpoint 0.5059, i.e. the soup adds almost nothing) versus matched A100
references 0.118–0.125.  Train ≈ valid ⇒ severe underfitting, not overfitting
and not a selection problem.

### 4.5 Q5 — Main / No-incidence / Dense

The two pre-registered controls were **not run** because the main failed the
gate.  What can be said from the main-model audit: the failure is **not** a dead
mechanism (composition gradients are finite, the solver descends, sparsity is
non-degenerate) and **not** an optimisation blow-up (energy monotone, no NaN,
stable train/valid).  The blocker is representational: a 192-D,
assignment-invariant, row-norm code plus a 12 417-param head cannot express the
ZINC target, and the dictionaries can only be shaped through the hard bilevel
path of an unrolled, non-converged solver.

## 5. Interpretation

* Supported: the sparse dictionary + incidence composition machinery is
  *implementable and stable*; it produces real sparsity and a real
  optimisation-like solver (Q1/Q3).
* Supported: the incidence mechanism is only **weakly** used — the row-norm
  readout kills almost all endpoint-assignment information (Q2).
* Not supported: competitiveness.  The representation is far from the matched
  A100 references and from the 0.15 gate (Q4).
* Not tested: whether removing incidence (γ=0) or sparsity (λ=0) changes the
  outcome — controls were gated off by the Case-A failure.

## 6. Caveat on the intervention

The assignment-permutation numbers in §4.2 come from the first formal run,
whose permutation was applied across the whole batch (a *stronger* corruption
than a per-graph permutation).  A within-graph fix was written afterwards
(commit `439ac1b`) but the confirmatory rerun was stopped by the user before it
finished, so no corrected number exists.  Because the global permutation is a
strictly stronger corruption, the "α barely changes (≤1.5 %)" reading is an
*upper bound* on the true per-graph change and remains valid as evidence that
the row-norm code is nearly assignment-invariant.

## 7. Next decision

**Revise the mathematical objective** (do not continue this form as-is, do not
abandon the line).  The mechanism is alive but the readout discards the
incidence information and the outer objective cannot shape the dictionaries
through the un-converged solver.  The single highest-leverage change is the
graph code / readout: keep the joint node–bond sparse inference but stop
summarising it with a permutation/assignment-invariant row-L2 norm.

## 8. Reproduction

```
uv run research run zinc_wg_icsc --mode screen --set model.device=cuda
# calibration + 240-epoch training + Top-5 soup + mechanism audit
```
Artifacts: `results/wg_icsc/formal_seed0/artifacts/` (`results.json`,
`calibration.png`, `training_curve.png`, `solver_dynamics.png`).
