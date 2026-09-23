# FEC-D0 — analysis and frozen verdict

Round: **FEC-D0** (*Within-Coarse-Role Residual Structural Dictionary Audit*),
protocol `fec_d0`, study `zinc-context-gap`.
Pre-registration: [`fec_d0_preregistration.md`](fec_d0_preregistration.md).
Prior-artifact audit: [`fec_d0_prior_artifact_audit.md`](fec_d0_prior_artifact_audit.md).

Lineage HEAD: `ca2e947013263a2590c7ad4edffff03a46e6b830`.
Runner: `tracks/ksvd/experiments/luyin16/zinc_fec_d0.py`.
Module: `tracks/ksvd/experiments/luyin16/fec_d0.py`.
Tests: `tracks/ksvd/tests/test_fec_d0.py` (20 pass).
Artifacts: `results/fec_d0/`.

**Regime: LABEL-FREE.** No property training, `official_valid_loaded=false`,
`official_test_loaded=false`, `targets_loaded=false`, no FEC-S1 retrain, no
task-coupled dictionary. Only official ZINC **train** (10000 molecules) was
read, split FIT 8000 / HOLDOUT 2000 by the canonical `SPLIT_SEED=20260922`
permutation convention.

---

## 0. Frozen verdict

```
FEC_D0_BASIS_HAS_NO_HELDOUT_SUBROLE_SIGNAL
```

First decisive failure: **Gate 2b** — the dense audited rooted structural basis
has no held-out marked-subrole signal beyond the within-patch degree baseline
(aggregate `G_O = 0.0244 < 0.03`). Per the pre-registration this stops the
round; the semantic dictionary gate is reported but is not decisive.

---

## 1. Gate 0 — purity / correctness: ALL PASS (7/7)

| check | result |
|---|---|
| G0 chemistry invariance | pass (basis depends only on adjacency) |
| G1 relabel invariance | residual max abs diff `0.0`; codes exactly equal |
| G2 exact sparsity | max `‖α‖₀ = 4`; exact-top-4 fraction `1.0` |
| G3 FIT-only transforms | `μ_s, σ_s` from FIT indices only; HOLDOUT stats differ from FIT |
| G4 no target | target access raises; `targets_loaded=false` |
| G5 coarse anchor removed | no exact shell coordinate kept; `E[φ^⊥|shell]` max abs `≈1.6e-16` |
| G6 numerical reference | reported == independent float64 reconstruction (`abs diff 0.0`) |

FIT/HOLDOUT occurrence counts: **1,134,082 / 284,418**.
Per-shell FIT counts: shell0 `185,190`, shell1 `398,558`, shell2 `550,334`.

---

## 2. Gate 1 — reconstruction: PASS, but uninformative by construction

```
E_D = 0.000216   E_R = 0.057305   E_P (PCA-4) = 5.1e-31
R_rec = E_D / E_R = 0.0038      (gate ≤ 0.80)  -> PASS
```

This gate passes, but the pass is **structural, not empirical**. The audited
rooted node basis, after deleting the exact shell coordinates and per-shell
standardizing, is **low-rank**:

| shell | rank of `φ^⊥` (FIT) | explained variance |
|---|---:|---|
| 0 | **2** | `[0.751, 0.249, 0, …]` |
| 1 | **4** | `[0.487, 0.398, 0.113, 0.0007, …]` |
| 2 | **4** | `[0.596, 0.401, 0.003, 0, …]` |

Two of the seven kept coordinates are **exactly shell-deterministic** and are
floored out by residualization: coordinate `5` (`neigh_by_shell[0]`, which is
`1[shell=1]` in these patches) and coordinate `8` (`walk1 = A e_root`, which is
exactly `1[shell=1]`). After that, the residual spans at most a rank-4 space.

Consequence: a `K=16` dictionary is **massively overcomplete** for a rank-4
residual. The `s=4` code reconstructs almost exactly, and the sparse code is
(piecewise) a **linear reparametrisation** of the dense residual. This is why
the reconstruction gate is weak — exactly the concern already recorded in
`pec_v0_preregistration.md` Amendment A1 for the same `b^V` basis.

---

## 3. Gate 2 — reuse: PASS (perfect)

```
active FIT = 16/16      active HOLDOUT = 16/16      Jaccard = 1.000
atoms on ≥20 HOLDOUT molecules = 16/16
usage Spearman(p_fit, p_holdout) = 1.000
FIT usage:     entropy 2.376 nats, effective atoms 10.76, top1 0.196, top4 0.623, Gini 0.484
HOLDOUT usage: entropy 2.377 nats, effective atoms 10.77, top1 0.196, top4 0.623, Gini 0.484
```

The learned dictionary is maximally reused: every atom is used across
molecules, roots and held-out data with a perfectly stable usage profile. No
memorisation. But given the rank-4 residual this is again *easy* — the
dictionary has far more atoms than the residual has dimensions.

Noise stability (report-only, `ε~N(0,1e-4)`): support Jaccard `0.9999`,
coefficient cosine `0.99999`, top-1 atom stability `1.0`.

---

## 4. Gate 2b — the decisive failure

Diagnostic label `T_iv` = exact **untyped marked-node rooted topology class**
(root flag, marked flag, root-distance colour; no chemistry), canonicalized
with the repaired `typed_patch_tokenizer.corrected_canonical_key`
(certificate + canonical semantic colour sequence), unit-tested for
relabel-invariance and non-collision.

Eligible classes (FIT ≥ 20 and HOLDOUT ≥ 5): **512** —— shell0 90, shell1 208,
shell2 214; HOLDOUT eligible occurrences 46,274 / 99,443 / 137,286.

Probe: multinomial logistic regression, L2, `C=1`, fixed `max_iter`, FIT-only
fit, HOLDOUT-only eval. Macro-F1 (primary), accuracy.

| shell | n_hold | B (degree) | R (random) | D (learned) | O (dense) | G_O | G_D | G_R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 46,274 | 0.0237 | 0.0413 | 0.0413 | 0.0413 | 0.0177 | 0.0177 | 0.0177 |
| 1 | 99,443 | 0.0086 | 0.0581 | 0.0623 | 0.0623 | **0.0537** | 0.0537 | 0.0495 |
| 2 | 137,286 | 0.0024 | 0.0126 | 0.0079 | 0.0079 | 0.0055 | 0.0055 | 0.0102 |

Occurrence-weighted aggregate (weights `[0.164, 0.351, 0.485]`):

```
F1_B = 0.0081   F1_R = 0.0333   F1_D = 0.0325   F1_O = 0.0325
G_O  = F1_O - F1_B = 0.0244      (gate ≥ 0.03)  -> FAIL
G_D  = F1_D - F1_B = 0.0244
G_R  = F1_R - F1_B = 0.0252
F1_D - F1_R = -0.0008            (gate ≥ 0.03)  -> FAIL
```

**`G_O = 0.0244 < 0.03` → `FEC_D0_BASIS_HAS_NO_HELDOUT_SUBROLE_SIGNAL`.**

### 4.1 Why `F1_D == F1_O` exactly, and why the oracle is weak

`D` and `O` give **identical** macro-F1 and accuracy in every shell, and
`G_D == G_O`. This is not a bug:

* the learned dictionary has `rank(D) = 7` while the holdout residual has
  `rank = 4`, and its holdout relative reconstruction error is `2.2e-4`;
* therefore an `s=4` code reconstructs the residual essentially exactly, and
  the code is a linear image of the residual → a linear probe on the code has
  the same decision function as a linear probe on the residual.

So the sparse dictionary neither loses nor adds subrole information relative to
the dense residual — it is a faithful but *information-neutral* recoding. The
failure is in the **basis**, not the dictionary.

The oracle's weakness is genuine and structural. `φ_{iv}` is a
**root-relative** fingerprint of the occurrence (degree, neighbour-by-shell,
rooted walk counts). A marked-node-rooted topology class `T_iv` encodes far
more: the occurrence's own local rooted structure *and* the shell-2 topology
around it. Within a shell, most of the surviving variation of `φ^⊥` is a
function of the **within-patch degree** (shell 0 is exactly rank-2 with
everything a function of degree). Hence the dense basis moves macro-F1 only
`0.0244` beyond a degree-only baseline, dominated by the large shell-2 stratum
(`G_O = 0.0055`), with only shell 1 showing material degree-beyond signal
(`G_O = 0.0537`).

### 4.2 Convergence caveat (honest limitation)

The frozen protocol fixes `max_iter=200` and forbids tuning; some lbfgs fits
emitted `ConvergenceWarning`. This is recorded as a limitation, not used to
overrule the gate. A cheap bounded check (200 vs 2000 iterations) was
considered and **deliberately not auto-run** to keep the round lean; the
decisive evidence is structural (per-shell rank ≤ 4, `D == O` identity,
degree-dominated residual) and does not depend on optimiser convergence.

---

## 5. Gate 3 — atom semantics (report-only)

`MI(atom → degree) = 0.357`; `MI(atom → marked topology) = 0.989`.
Every atom's support spans only 3–4 degree buckets while seeing hundreds of
distinct marked-topology classes. Combined with §2 (rank-4 residual largely
degree-determined), the honest reading is:

> For this basis the dictionary is **primarily a discretized degree
> representation**; it is not a rich subrole vocabulary.

This must not be described as "rich structural vocabulary".

---

## 6. The six round questions

**Q1 — Is there residual structure?**
Only a low-rank one: after the known shell is removed, the residual is rank 2
(shell 0) to rank 4 (shells 1–2), and is dominated by within-patch degree.
A dictionary can reconstruct it (R_rec 0.0038) but the reconstruction gate is
uninformative.

**Q2 — Is it genuinely sparse?**
`K=16, s=4` represents it essentially exactly, but trivially, because the
residual rank ≤ 4 ≪ K. Sparsity is not doing work here.

**Q3 — Is it reusable?**
Yes, maximally: 16/16 atoms active on both splits, Jaccard 1.0, usage Spearman
1.0, every atom spans ≥20 held-out molecules. No memorisation.

**Q4 — Does it encode real structural subroles?**
**No, not beyond degree.** The dense root-relative basis gives only
`G_O = 0.0244` macro-F1 over a within-shell degree baseline, below the
pre-registered 0.03.

**Q5 — Is it better than a random sparse vocabulary?**
No. `F1_D - F1_R = -0.0008` (gate ≥ 0.03). The matched random dictionary is
statistically indistinguishable, and marginally *better* overall.

**Q6 — Is a task experiment scientifically justified?**
**NO — stop this dictionary placement.**
Verdict is `FEC_D0_BASIS_HAS_NO_HELDOUT_SUBROLE_SIGNAL`, not
`FEC_D0_RESIDUAL_STRUCTURAL_DICTIONARY_QUALIFIED`. FEC-D1 is **not** authorized.

---

## 7. Why this is not a repeat of history (§24)

* **vs TCCD.** TCCD-v0…v7 encode the complete *attributed* patch
  (`x_v ∈ R^714`, chemistry included) and closed at Gate-1 continuity. FEC-D0
  encodes only the **pure-topology residual subrole within a known coarse
  rooted role** — no chemistry, no composition, no reader. New object, new
  failure mode (basis insufficient, not coordinate discontinuity).
* **vs SDB-v0.** SDB codes the **graph-level / patch-aggregated** FSAR-R2-AR0
  coordinate `φ_v ∈ R^65` and its strong-backbone residual route failed.
  FEC-D0 explicitly forbids reusing `C_D` or the aggregated coordinate and
  codes the **occurrence**, residualized within the coarse role.
* **vs SDPK-v0 / SRDA-v0.** Those promote the dictionary into the *pair
  kernel* / occurrence relation algebra. FEC-D0 has **no pair object at all**;
  pair composition is out of scope.
* **vs PEC-CK (c1/i1) and PEC-v0.** PEC lets the frozen sparse dictionary
  *be* the learned role coordinate `[one_hot(shell); α]` and binds it to
  chemistry in an environment. FEC-D0 keeps the coarse exact role as a given,
  never forms an environment, never binds chemistry, never trains a reader.
* **vs DTX-v0.** DTX is a graph-level aligned dictionary × topology cross;
  FEC-D0 is a within-occurrence residual question with a matched-random
  control.

The new decisive content — *residualize the coarse role, then ask whether the
remainder is a reusable sparse vocabulary* — had never been run.

---

## 8. Interpretation discipline (§25)

Even though the dictionary passes reconstruction and reuse, this round does
**not** license any statement about chemical motifs. The dictionary reads **no
chemistry whatsoever**. The only accurate descriptions are:

> a reusable sparse **pure-topology structural** code of a low-rank,
> degree-dominated occurrence residual

or, more bluntly,

> primarily a discretized degree representation.

No chemical-environment claim is supported.

---

## 9. FEC-S1 background (not run this round)

```
FEC-S1 seed0  Top-5 soup official-valid MAE = 0.130422
              best = 0.136783 @238
FEC-S1 seed1  intentionally not run.
```

FEC-S1 is a **strong seed-0 anchor** sufficient to motivate a
baseline-preserving future dictionary experiment. It is **not** a stable
2-seed baseline (seed 1 was authorized but not executed). FEC-S1 was **not**
retrained in this round.

---

## 10. Decision

STOP. The dictionary placement at
`coarse rooted role → sparse residual subrole` is **not qualified** on the
audited rooted node basis `b^V`.

Combined with the historical dictionary closures (TCCD, SDB, SDPK, SRDA,
GSCN, PEC-CK, DTX), the honest cross-round reading is:

> In this ZINC predictive architecture the dictionary is better understood as
> an **interpretation / compression tool** than as the core inductive bias.

No rescue is authorized: no `K=32`, no `s=8`, no more iterations, no nonlinear
dictionary, no new basis invented on the spot, no edge dictionary, no second
seed, no property run.

**FEC-D1 is not authorized.** A future round would need a *new* pre-registration
and, critically, a rooted structural basis that actually carries degree-beyond
held-out subrole information (FEC-D0 localizes the current deficiency to the
basis, not the dictionary).
