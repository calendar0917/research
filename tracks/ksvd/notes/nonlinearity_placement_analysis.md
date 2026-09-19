# NPA — Nonlinearity Placement Audit (ZINC)

**Question.** DOI/AIOM stack propagation steps with no nonlinear discrimination
between them (`H_t = S^t X`). Is adding nonlinearity **between** propagation
steps already enough — or must the nonlinear structural coordinates be
**task-coupled**?

**Answer (one line).** Neither a fixed nonlinear scattering lift nor a ~2.5K
task-coupled incidence lift materially fixes the readout: all three lifts floor
at train ≈ 0.31–0.33 / valid ≈ 0.33–0.39, so **a minimal pre-dictionary
structural lift is insufficient for ZINC**.

**Status.** Representation diagnostic only. No dictionary, ISTA, K-SVD, GNN
backbone, attention, learned pooling, graph token, canonical IDs, shortest paths
or hand-crafted statistics. Official ZINC `test` was **never** loaded; the test
read count is unchanged. `y` is used only by the capacity probes. Pre-registered
in `notes/nonlinearity_placement_preregistration.md` before the run.

---

## 0. Provenance

| item | value |
|---|---|
| git commit | `6585bf60173b15d97b31acb21de24908ce4eeff8` |
| worktree at run | clean |
| splits used | official PyG ZINC `subset=True`; **train 10 000 / valid 1 000** |
| official `test` | **never read / instantiated / referenced** |
| execution | remote A100 host `res`, CPU numpy + CPU torch, **1125 s** |
| code | `tracks/ksvd/code/run_nonlinearity_placement_audit.py` |
| tests | `tracks/ksvd/tests/test_nla.py` (9 data-free tests, pass) |
| results | `tracks/ksvd/results/nonlinearity_placement/` (git-ignored) |
| loader / operator / perturbations / probe | reused from `run_aiom_representation_audit.py` |
| distributional RFF/KME | reused from `run_doi_representation_audit.py` |

Confirmed semantics: `C_V=21`, `C_E=3`, 0 self-loops, 0 multi-edges. The frozen
DOI features were re-materialized from the recorded hyperparameters
(`scales_T4.npz`, `bandwidth_stats.json`, `rff_metadata.json`) and reproduce the
pulled artifact to `8.8e-08` (float32 storage), confirming the geometry baseline
is the same object as `decision-4b8e2d1f`.

---

## 1. Primary performance

| Lift | structural params | RFF dim | feature dim | train MAE | valid MAE |
|---|---:|---:|---:|---:|---:|
| Linear DOI (frozen) | 0 | 1024 | 3074 | 0.3273 | **0.3865** |
| Fixed Nonlinear Scattering DOI | 0 | 2048 | 6146 | 0.3041 | **0.3657** |
| Tiny Task-Coupled Incidence Lift | 2496 | 1024 | 3074 | 0.3138 | **0.3310** |

Head identical for scattering and learned: `Linear(d,64)→SiLU→Linear(64,1)`
(d=6146 / 3074). Prediction-head parameters are not counted as structural
capacity.

* **Fixed scattering**: −0.021 valid vs DOI, and only after the label-free
  dimension rule forced `D_RFF` from 1024 to **2048** (feature dim 6146 vs
  3074). So the fixed nonlinear lift gives almost nothing, *even at 2× the RFF
  dimension*.
* **Learned incidence**: −0.0555 valid at **2496** structural parameters and the
  same 3074 readout as DOI. Train also drops (0.3273 → 0.3138). This is the best
  of the three, but the train/valid gap is small (0.314 / 0.331): a genuine
  pipeline floor, not overfitting.
* All three stay far above any dictionary-ready band; the learned lift is still
  **> 0.25 valid**.

Constant-train-mean reference valid MAE ≈ 1.48.

---

## 2. RFF approximation audits

Label-free thresholds: median < 0.02 and p95 < 0.05 for all three kernels.

| lift | D | worst-kernel median | worst-kernel p95 | Spearman | pass |
|---|---:|---:|---:|---:|---|
| fixed scattering | 2048 | **0.0255 (E)** | 0.0338 | 0.998 | **False** |
| learned (init, seed 0) | 1024 | 0.0301 | 0.0365 | 0.994 | False |
| learned (post-training) | 1024 | 0.0122 (V) | 0.0218 | 0.995 | **True** |

Two honest caveats, reported rather than hidden:

1. **Scattering is dimension-mismatched and RFF-limited.** At the maximum allowed
   `D=2048` the bond-kernel median error is still 0.0255 > 0.02, so its 6146-D
   object is both *larger* than the others and *approximation-limited*. Its
   small +0.021 valid gain is an upper bound on the fixed-lift effect, not a
   lower bound — the conclusion (fixed nonlinearity does not significantly fix
   DOI) is therefore robust.
2. **Learned lift**: after training the RFF approximation actually *improves*
   (p95 0.022 < 0.05, `confounded = False`), so the learned result is not
   RFF-confounded. Bandwidths were frozen at the seed-0 init and never
   re-estimated.

---

## 3. Unified geometry (500 train graphs)

Standardized `Δ_Φ`; `R_topo = mean d(random size-matched) / mean d(degree-preserving 2-switch)`
(same mean convention as the DOI `R_topo ≈ 10.7`).

| lift | dim | 2-switch | random | **R_topo** | relative 2-switch | atom-sub | bond-type swap | permutation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear DOI | 3074 | 7.45 | 72.34 | **9.71** | 0.0100 | 53.36 | 8.80 | 4.2e-14 |
| Fixed scattering | 6146 | 18.53 | 102.54 | **5.53** | 0.0182 | 51.43 | 15.95 | ~0 |
| Learned incidence | 3074 | 19.21 | 73.87 | **3.85** | 0.0303 | 29.81 | 15.67 | ~0 |

Both nonlinear lifts make degree-preserving rewiring **more visible** in `Φ`
(DOI 9.71 → scattering 5.53 → learned 3.85). The learned lift is the strongest:
its 2-switch response (19.21) is 2.6× DOI's and 3× larger relative (0.0303 vs
0.0100), i.e. a small real topology change now moves the whole-graph object
substantially. Pure permutation is a numerical floor for all three.

**WL fidelity** (3-round typed subtree histogram, size-matched random control
0.9058):

| lift | top-1 WL | top-10 WL | Spearman(d_Φ, d_WL) | top-10 overlap |
|---|---:|---:|---:|---:|
| Linear DOI | 0.9331 | 0.9292 | 0.426 | – |
| Fixed scattering | 0.9324 | 0.9282 | 0.390 | – |
| Learned incidence | **0.9356** | **0.9313** | 0.422 | – |

The learned lift gives the best absolute WL neighbour fidelity (top-1
0.9331 → 0.9356) at essentially unchanged distance correlation. The improvement
over DOI is small but in the expected direction; the fixed scattering lift is
not better than DOI on WL fidelity (it is worse on Spearman).

---

## 4. Learned incidence — mechanism audit

Within-graph degree-preserving 2-switch, per-layer mean L2 change:

| Δ_h0 | Δ_e0 | Δ_e1 | Δ_h1 | Δ_e2 | Δ_h2 | **Δ_Φ** |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.0000 | 0.0774 | 0.1772 | 0.3144 | 0.3369 | **0.1306** |

The input coordinates (`h0,e0`) depend only on categories, so rewiring cannot
move them (0). The structural difference is created **inside** the rounds and
**grows monotonically** (e1 → h1 → e2 → h2), and it is **not** washed out by the
KME (`Δ_Φ = 0.131` survives the readout). So the learned lift *does* use the
real atom↔bond incidence, and its small gain is accompanied by real structure
use. The 64-graph smoke confirms gradients flow and loss descends with no NaN
(1.474 → 0.484 over 40 epochs).

Because the learned lift beats DOI by only 0.0555 valid (< the pre-registered
0.10 trigger), the **gated no-message control was not run** — the causal
atom↔bond attribution is therefore *not* isolated (see Q3).

---

## 5. Answers to the four required questions

* **Q1 — Is fixed interleaved nonlinearity enough to fix DOI?** **No.**
  Scattering valid 0.3657 vs DOI 0.3865 is a 0.021 gain, and it required
  `D_RFF` 1024 → 2048 (feature dim 3074 → 6146). Gate A (valid ≤ 0.25) is not
  met. The fixed scattering operator does make rewiring more visible
  (R_topo 9.71 → 5.53) but does not materially improve the readout.
* **Q2 — Does the tiny task-coupled incidence lift significantly fix it?** **No,
  but it helps the most.** Learned valid 0.3310 (−0.0555 vs DOI, −0.021 vs
  scattering) with only 2496 structural parameters, and the strongest geometry
  improvement (R_topo 9.71 → 3.85; best WL top-1). It still sits **> 0.25**, so
  Gate B (≤ 0.20) and Gate C (≤ 0.25) both fail → Gate D.
* **Q3 — Does the learned gain depend on real atom↔bond communication?** **Not
  isolated.** The pre-registered trigger (> 0.10 valid improvement) was not met
  (only 0.0555), so the no-message control was not run. The layer-wise audit is
  consistent with genuine structural use (Δ grows through the atom↔bond rounds
  and survives the KME), but this is mechanism evidence, not the causal control.
* **Q4 — Which hypothesis?** **Hypothesis C: even a minimal task-coupled lift
  lacks enough capacity.** Hypothesis A is rejected (fixed nonlinearity barely
  moves the readout: +0.021 valid at 2× dimension). Hypothesis B is rejected
  (the task-coupled lift is better but still 0.331 > 0.25). The train error also
  stays ≈ 0.31, so this is a pipeline floor of the whole-graph invariant
  distributional object, not an optimisation failure.

---

## 6. Interpretation

**Supported by the data.**
* Interleaving a fixed nonlinearity between propagation steps (scattering) is
  *not* the missing ingredient: +0.021 valid, and only at 2× RFF dimension.
* A ~2.5K task-coupled atom–bond incidence lift is the best of the three and
  genuinely encodes topology (R_topo 9.71 → 3.85, Δ increasing through rounds,
  Δ_Φ surviving the KME), but it still floors at train 0.314 / valid 0.331.
* All three lifts live in the same 0.31–0.39 band; neither placement of
  nonlinearity reaches a dictionary-ready domain.

**My interpretation (not a direct fact).** The bottleneck is not *where*
nonlinearity sits along the propagation. A small learned local lift can improve
the geometry and the readout a little, but the fixed, invariant,
distribution-summary object itself — even with task-coupled local coordinates —
does not expose enough decodable task signal. This is the pattern Gate D
describes: a minimally parameterised pre-dictionary structural lift is
insufficient for ZINC.

---

## 7. Final conclusion

> **Minimal pre-dictionary lifting is insufficient; the current whole-graph
> dictionary-primary route should be reconsidered.**

Neither fixed nonlinear structural lifting (scattering) nor a tiny task-coupled
incidence lift materially fixes the DOI readout; both remain far from a
dictionary-ready band, and the learned lift's small gain is not sufficient to
justify continuing to treat a fixed whole-graph descriptor as the primary
representation.

### Next decision (one)

> **`reconsider whether dictionary can remain the primary graph representation`**

### What this round did **not** do (per pre-registration)

No dictionary, sparse coding, GNN width/depth sweep, residual, LayerNorm,
attention, edge gates, hidden 32/64, learned RFF/bandwidth, graph pooling,
`T > 4`, higher scattering orders or other wavelet families. No official test
was loaded.

---

## 8. Machine-readable outputs (`results/nonlinearity_placement/`, git-ignored)

`SUMMARY.json`, `provenance.json`, `edge_semantics.json`,
`scattering_metadata.json`, `scattering_rff_approximation.json`,
`doi_reproduction_check.json`, `learned_metadata.json`, `learned_smoke.json`,
`learned_rff_approximation.json`, `learned_probe.json`, `learned_mechanism.json`,
`learned_features.npz`, `learned_model.pt`, and `plots/` (`mae_comparison`,
`rewiring_vs_random`, `wl_fidelity`, `learned_layerwise_rewiring`,
`geometry_ecdf`).
