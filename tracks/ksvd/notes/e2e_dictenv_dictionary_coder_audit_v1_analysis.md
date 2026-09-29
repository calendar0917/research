# e2e_dictenv_dictionary_coder_audit_v1 — analysis note

Round: **Dictionary–Coder Disentanglement** on the closed ZINC
dictionary-environment line.  CPU-only, local, official ZINC test never loaded.
Preregistration: `notes/e2e_dictenv_dictionary_coder_audit_v1_preregistration.md`
(frozen before the first Stage-A process; commit `b325770`).
Generated evidence: `results/e2e_dictenv_dictionary_coder_audit_v1/`
(`analysis_tables.md`, `summary.json`, `coder_geometry/`, `dominant_atom/`,
`atom_specialization/`, `cross_seed_stability/`, `sparse_dense_recoverability/`,
`gate.json`).

This note records the audit, including two **descriptive** additions made after
the formal gate result (the IHT step-count convergence table and the per-atom
profile export).  Neither feeds a gate, a threshold or any selection.

---

## 0. What was audited, and what was not

Three reused C6 soup dictionaries (seeds 0/1/2, all with the frozen
FINAL-CLEAN recipe) and, contextually, the DenseTied seed-0 control
dictionary.  Every audit coder sees the *same* column-normalised
`Dbar` per seed.  No predictor was trained: the frozen IHT-30 training gate
did not fire.  No architecture, loss, feature, relation, split or threshold
was touched after the freeze.

Input space: the raw float32 phi65 in
`results/e2e_dictenv_p1/cache/env_{train,valid}.pt` (231 664 × 65 train,
23 083 × 65 valid) — the exact tensor the model's dictionary receives.

## 1. Stage A — same-`D` coder audit

Valid-split summary (`coder_geometry.csv`, `analysis_tables.md` §1):

| seed | coder | recon_fro | active | N_eff | top-5 share | max rate | J(OMP) | cos(OMP) |
|---|---|---|---|---|---|---|---|---|
| 0 | IHT-10 | 0.012388 | 27 | 14.52 | 4.796 | 1.000 | 0.208 | 0.609 |
| 0 | IHT-30 | 0.008463 | 30 | 14.84 | 4.733 | 1.000 | 0.204 | 0.609 |
| 0 | IHT-100 | 0.007359 | 31 | 14.86 | 4.733 | 1.000 | 0.204 | 0.608 |
| 0 | OMP | 0.039774 | 32 | 19.96 | 4.025 | 0.976 | — | — |
| 1 | IHT-10 | 0.010885 | 27 | 14.80 | 4.559 | 1.000 | 0.216 | 0.667 |
| 1 | IHT-30 | 0.007968 | 27 | 14.79 | 4.559 | 1.000 | 0.222 | 0.667 |
| 1 | IHT-100 | 0.007238 | 28 | 15.26 | 4.559 | 1.000 | 0.222 | 0.667 |
| 1 | OMP | 0.041820 | 32 | 22.54 | 2.957 | 0.977 | — | — |
| 2 | IHT-10 | 0.011858 | 28 | 15.35 | 4.409 | 1.000 | 0.264 | 0.605 |
| 2 | IHT-30 | 0.008352 | 31 | 15.22 | 4.409 | 1.000 | 0.259 | 0.606 |
| 2 | IHT-100 | 0.007297 | 31 | 15.24 | 4.408 | 1.000 | 0.260 | 0.607 |
| 2 | OMP | 0.065397 | 32 | 21.34 | 3.316 | 0.828 | — | — |

Three facts dominate:

1. **More IHT iterations improve reconstruction but not the code
   structure.**  IHT-30/100 reduce the Frobenius reconstruction error by
   27–40 % (absolute 2.9–3.9e-3) and shrink the per-row residual variance, but
   activation counts, `N_eff`, top-5 share, max activation rate, Gini and the
   similarity to the OMP code are unchanged to the third decimal.
   The descriptive convergence table (written after the gate) confirms
   this directly: IHT-10 vs IHT-30 support Jaccard 0.972/0.980/0.975, exact
   support match 0.916/0.910/0.889, code cosine 0.9999, normalised L2
   difference 0.012/0.010/0.011.  **IHT-10's support is already the IHT fixed
   point; only the coefficient amplitudes move.**
2. **The trained dictionary is intrinsically concentrated.**  No atom leaves
   the support except through disuse: 22–23 atoms are used at >1 %, 5–10 atoms
   are never used at all, `max_activation_rate = 1.000` on every seed (at
   least one atom is in *every* row), top-5 atoms carry 4.4–4.8 of the 8
   activations, `N_eff ≈ 14.5–15.3`.
3. **OMP is not an oracle on this dictionary.**  With the same `Dbar`, exact
   OMP(s = 8) reconstructs 3.2–5.5× worse than IHT-10 (Frobenius
   0.0398/0.0418/0.0654 vs 0.0124/0.0109/0.0119), while spreading usage over
   all 32 atoms (`N_eff` 20.0–22.5) and producing much more top-heavy
   coefficients (mean top-1/L1 0.56–0.67 vs 0.28–0.30; mean max|α|
   4.8–5.0 vs 1.9–2.0).  Verified independently: a direct
   `sklearn.orthogonal_mp(Dbar, X.T, n_nonzero_coefs=8)` reproduces the frozen
   wrapper exactly, and the selected supports are well-conditioned
   (cond 1.8–3.5 on the worst rows) — the loss is greedy support selection, not
   numerical failure.  The previously quoted "exact OMP 1.45e-5" was measured
   on the **K-SVD** dictionary, which was fit *for* OMP; the trained dictionary
   was shaped by IHT gradients for 320 epochs, so coder fidelity is a property
   of the dictionary–coder pair, not of the coder alone.

Coefficient geometry (valid means, `analysis_tables.md` §2): IHT-10
`L1 6.59–6.81 / L2 3.18–3.29 / max|α| 1.88–2.04 / top1-L1 0.28–0.30`; IHT-30
and IHT-100 differ only in the third decimal; OMP `L1 7.15–9.02 / L2
4.94–5.40 / max|α| 4.77–5.02 / top1-L1 0.56–0.67`.

Frozen classification (preregistration §4): all three seeds `NONE`
(A1 support-artifact: no, support Jaccard unchanged; A2 amplitude-bottleneck:
reconstruction improves but not by the frozen 0.5× factor; A3
intrinsic-concentration: `N_eff_OMP > N_eff_IHT10 + 1` because OMP spreads
usage).  The honest reading is a **mixture of A2 and A3**: the *coder* is
amplitude-under-converged (and only that), the *dictionary* is intrinsically
concentrated — the frozen labels understate A3 because the label was defined
against the OMP code, which this dictionary punishes.

## 2. Stage B — dominant atom and the common direction

`dominant_atom/seed{0,1,2}.json`, `analysis_tables.md` §3 and §8.

* The dominant atom is **6 in all three seeds** for IHT-10/IHT-30, active in
  **100 % of valid rows**, with `|cos(atom 6, mean direction μ)| =
  0.965/0.983/0.941`.  OMP also picks atom 6 on seeds 0/1 (activation
  0.975/0.977) and atom 9 on seed 2.
* **Atom 6 is not the variance direction**: `|cos(atom 6, PC1)|` is only
  0.342/0.436/0.322 even though PC1 carries 57.7 % of phi65 variance.  It is
  the **DC / patch-mass direction**.
* Two more atoms, **24 and 27, are also active in 100 % of rows on all three
  seeds**, with `|cos(·, μ)| = 0.90–0.94` (atom 24) and 0.76–0.86 (atom 27).
  Atom 6 and atom 24 are 0.884–0.987 collinear with each other.  Exactly three
  atoms per seed have `|cos(·, μ)| > 0.5`: `{6, 24, 27}`.
* A fourth near-universal atom, **23**, is active in 83.3–97.7 % of rows and
  is the **PC1 atom** (`|cos(atom 23, PC1)| = 0.80/0.71/0.77`); it is nearly
  orthogonal to the DC triplet.
* The atom-6 coefficient is a patch-scale coordinate: Spearman(|α_6|, z) =
  patch_nodes 0.93–0.94, std_log1p_walk3 0.92–0.93, patch_edges 0.80–0.88,
  root_induced_degree 0.77–0.88, shell_pop1 0.78–0.82.

Mechanism reading: the model spends **three of its eight activations on a
near-duplicate DC/scale triplet** (6, 24, 27) plus one PC1 atom (23) in most
rows.  Top-5 ≈ 4.4–4.8 and `N_eff ≈ 14.5–15.3` follow arithmetically.  The
concentration is a dictionary-geometry fact, not a coder artefact — it is
already present at OMP with 97.6 % atom-6 usage on seeds 0/1.

## 3. Stage C — atom structural specialisation

`atom_specialization/seed{0,1,2}.json`, `atom_specialization.csv`,
`analysis_tables.md` §4 and §8.  SMD profiles are fit on train only and frozen
for valid; the top-response population is the top 10 % of each atom's non-zero
|α| (label-free).

| seed | coder | median Spec | usage-weighted Spec | median top-response Spec | median profile cos |
|---|---|---|---|---|---|
| 0 | IHT-10 | 0.564 | 0.207 | 0.837 | 1.000 |
| 0 | IHT-30 | 0.639 | 0.213 | 0.828 | 1.000 |
| 0 | OMP | 0.423 | 0.237 | 0.859 | 0.999 |
| 1 | IHT-10 | 0.349 | 0.191 | 0.953 | 1.000 |
| 1 | IHT-30 | 0.356 | 0.194 | 0.863 | 1.000 |
| 1 | OMP | 0.438 | 0.309 | 0.889 | 0.999 |
| 2 | IHT-10 | 0.619 | 0.237 | 1.067 | 1.000 |
| 2 | IHT-30 | 0.676 | 0.231 | 1.007 | 1.000 |
| 2 | OMP | 0.492 | 0.321 | 0.898 | 0.999 |

* Atoms **do** carry stable, transferable structural roles: every finite
  train→valid SMD profile cosine is ≥ 0.926, median 1.000.
* Specialisation is concentrated in the **rare** atoms: the three to five
  highest-Spec atoms (e.g. seed 0 atoms 12 and 8: `root_walk3`,
  `edge_frac_shellpair_11`, `mean/std edge_log1p_common_neighbours`, Spec
  3.3–3.7) have usage 0.001–0.06, so the usage-weighted Spec falls to
  0.19–0.24.  The always-active DC triplet has Spec 0.00 mechanically (its
  active population is the whole dataset).
* Top-response profiles are *sharper* than active-population profiles
  (median Spec 0.84–1.07 vs 0.35–0.62), i.e. an atom's strongest activations
  are its most structurally distinctive.
* Caveat recorded, not corrected: two of the 32 descriptors
  (`root_neighbour_shell1`, `root_walk1`) are identically zero by the frozen
  phi65 provenance (they are the root's shell-0 neighbours and length-1 walk
  from the root); their Spearman entries are `nan` and they carry no
  information.  This affects no verdict (SMD components are z-scored; a
  constant descriptor is numerically harmless but should be dropped in any
  future descriptor revision).

## 4. Stage D — cross-seed dictionary stability

`cross_seed_stability/`, `dictionary_stability.csv`, `analysis_tables.md` §5.
Hungarian matching on `|cos|` (sign- and permutation-symmetric):

| pair | mean \|cos\| | median | min | p10 | #≥0.90 | #≥0.95 | median profile cos |
|---|---|---|---|---|---|---|---|
| 0↔1 | 0.738 | 0.807 | 0.117 | 0.342 | 13 | 6 | 0.879 |
| 0↔2 | 0.777 | 0.828 | 0.225 | 0.492 | 10 | 7 | 0.756 |
| 1↔2 | 0.809 | 0.855 | 0.178 | 0.635 | 12 | 6 | 0.891 |

Frozen verdict: **PARTIAL_VOCABULARY** (mean 0.775, mean #≥0.90 = 11.7,
median profile cos 0.853).

* The **universal atoms replicate exactly**: 6↔6 `|cos|` 0.989/0.969/0.971,
  24↔24 0.975/0.982/0.961, 27↔27 0.905/0.939/0.939, 23↔23 0.681/0.787/0.949.
  Mid-frequency atoms replicate too (9↔9 0.94–0.97, 30↔30 0.95–0.97).
* Rare atoms are only **partially** stable: 35/96 matched pairs ≥ 0.90, and
  the Hungarian assignment can pair a rare atom with a different rare atom
  whose structural role is not equivalent (e.g. 0↔2 atom 26↔26 has
  `|cos| = 0.838` but profile cosine −0.968).  Spearman(`|cos|`, max usage) =
  0.44: usage predicts stability but does not determine it.
* Reading: there is a small **seed-stable core** (the DC triplet + PC1 atom +
  a handful of mid-use atoms), not a stable 32-atom vocabulary.  Any future
  claim of "learned discrete dictionary vocabulary" must be restricted to the
  stable core.

## 5. Stage E — sparse ↔ dense recoverability

`sparse_dense_recoverability/`, `recoverability.csv`, `analysis_tables.md` §6.
OLS on train only, evaluated on valid; metrics on the representation itself.

| seed | direction | R² | per-dim median R² | normalised error | mean cosine | CKA |
|---|---|---|---|---|---|---|
| 0 | dense_tied → IHT-10 | 0.9708 | 0.9746 | 0.0314 | 0.9996 | 0.9987 |
| 0 | IHT-10 → dense_tied | 0.9988 | 0.9999 | 0.0063 | 1.0000 | 0.9998 |
| 1 | dense_tied → IHT-10 | 0.9763 | 0.9778 | 0.0279 | 0.9997 | 0.9991 |
| 1 | IHT-10 → dense_tied | 0.9989 | 1.0000 | 0.0059 | 1.0000 | 0.9999 |
| 2 | dense_tied → IHT-10 | 0.9770 | 0.9823 | 0.0273 | 0.9997 | 0.9988 |
| 2 | IHT-10 → dense_tied | 0.9989 | 0.9997 | 0.0061 | 1.0000 | 0.9998 |
| 0 | dense_tied_own_D → IHT-10 | 0.9720 | 0.9724 | 0.0308 | 0.9996 | 0.9987 |
| 0 | IHT-10 → dense_tied_own_D | 0.9978 | 0.9998 | 0.0080 | 1.0000 | 0.9999 |

IHT-30 behaves like IHT-10 (0.970–0.975 / 0.9993–0.9994), and the DenseTied
control's own dictionary reproduces the pattern (0.972 / 0.998), so the result
is not specific to one dictionary.

* **Sparse → dense R² = 0.9988–0.9989**: the 8-sparse code contains
  essentially all linearly recoverable content of the dense tied coordinate.
  There is **no evidence that sparsification discards downstream-relevant
  information** — the frozen hypothesis (4) is not supported.
* **Dense → sparse R² = 0.971–0.977**: the sparse code (including its 24/32
  zero pattern) is almost fully predicted by a linear map of the dense
  coordinate.  The 2–3 % residual is the code's unique nonlinear content; it
  is not associated with any downstream advantage, since the dense arm is not
  worse.  Frozen hypothesis (3) — sparse is largely a linear
  re-parameterisation of the dense tied coordinate — is supported.
* OMP's code is less recoverable from dense (R² 0.74–0.87) and its own
  residual is 3–5× larger; the OMP family is a different, weakly-fitting
  parameterisation of the same coordinate, not an alternative signal.

## 6. Frozen gate (Stage F)

`gate.json`, `analysis_tables.md` §7.  **Gate fired: False.**
No new training was performed — *no new training was scientifically
justified by the frozen gate*.

| seed | cond 1 recon: factor (≤0.5) / drop (≥1e-4) | cond 2a support (+0.05) | cond 2b concentration (+1.0 N_eff / −0.10 top5 / −0.10 max rate) | cond 2c geometry (−0.05 top1/L1 / +0.05 cos) |
|---|---|---|---|---|
| 0 | 0.683 / 3.93e-3 → **False** | −0.0041 → False | +0.32 / −0.063 / 0.0 → False | −0.0004 / −0.0004 → False |
| 1 | 0.732 / 2.92e-3 → **False** | +0.0060 → False | −0.01 / 0.0 / 0.0 → False | −0.0004 / −0.0002 → False |
| 2 | 0.704 / 3.51e-3 → **False** | −0.0051 → False | −0.13 / 0.0 / 0.0 → False | −0.0010 / +0.0007 → False |

Condition 1 (reconstruction) missed the frozen 0.5× factor on all seeds
although the absolute improvement is large (2.9–3.9e-3, ~2–3× the 0.003
specificity margin).  This is exactly the amplitude/convergence effect of §1.
**The decision does not hinge on that threshold**: condition 2 fails on all
three axes on all three seeds, by margins far outside noise (support gain
needs +0.05 and gets |gain| ≤ 0.006; concentration needs +1.0 N_eff and gets
+0.32; max activation rate does not move at all).  Under any reasonable
relaxation of condition 1 the gate still does not fire, because "no support /
concentration / geometry change" is the substantive half of the gate.

## 7. What the audit says about the four hypotheses

| hypothesis | verdict | evidence |
|---|---|---|
| (1) IHT-10 is under-converged | **only for coefficient amplitude** | IHT-30/100 cut reconstruction 27–40 % but leave support (J 0.97), usage (`N_eff`, top-5, max rate) and geometry unchanged; IHT-10 is the IHT fixed point |
| (2) the dictionary is intrinsically concentrated | **supported** | atoms 6/24/27 active in 100 % of rows and 0.76–0.98 aligned with μ (6 and 24 0.88–0.99 collinear); atom 23 is 0.71–0.80 aligned with PC1 and 83–98 % active; the dominant coefficient is a patch-scale (DC) coordinate; concentration is already present in OMP |
| (3) sparse is a near-linear re-parameterisation of dense | **supported** | dense→IHT-10 R² 0.971–0.977, code cosine 0.9996, CKA ≈ 0.999; same on the DenseTied control's own dictionary |
| (4) sparsification loses downstream-relevant information | **not supported** | IHT-10→dense R² 0.9988–0.9989; the dense coordinate has no linearly recoverable content beyond the sparse code in this setup |

Synthesis: the dictionary and the coder can now be disentangled.
The coder is amplitude-limited but structurally converged at 10 steps.  The
dictionary is a *partially redundant* overcomplete basis whose top use is a
three-atom DC/scale clone plus one variance atom — which explains both the
observed concentration and why a "sparse" 8-slot code behaves like a slightly
nonlinear copy of the dense 32-D coordinate.  Sparse specificity was not
established previously because there is very little representation for it to
be specific *about*: the two coordinates span nearly the same linear subspace
of phi65, and the sparse arm's 2–3 % unique variation buys no accuracy.  The
audit does **not** identify a coder defect that a longer IHT would fix.

## 8. Descriptive additions, recorded for auditability

* IHT-10 ↔ IHT-30 ↔ IHT-100 support/code agreement on valid
  (`coder_geometry/convergence.{json,csv}`, `analysis_tables.md` §9).  These
  were computed after the gate decision with the frozen coder functions; they
  change no threshold and are not inputs to any decision.
* `coder_geometry/coder_geometry.csv` and
  `atom_specialization/atom_specialization.csv` are machine-readable exports
  of the already-stored Stage-A/C payloads (added after the fact for the
  required diagnostic-table deliverable).
* Independent-OMP verification on 500 valid rows: direct
  `sklearn.linear_model.orthogonal_mp(Dbar, X.T, n_nonzero_coefs=8)` matches
  the frozen wrapper; OMP Frobenius 0.0406 vs IHT-10 0.0157 on that subsample
  (consistent with the full-split 0.0398 vs 0.0124); worst OMP rows have
  well-conditioned supports (cond 1.8–3.5).
* The `root_neighbour_shell1`/`root_walk1` constant-descriptor defect (§3) is
  a property of the frozen phi65 provenance, found while interpreting the
  Spearman tables; recorded for the next descriptor revision.

## 9. Limits

* Three seeds, one architecture family, one dataset, valid split only;
  single-seed results (e.g. the OMP seed-2 atom change) are not replicated
  claims.
* The DC-triplet explanation rests on geometry (cosines, activations) and a
  rank/usage analysis; no causal intervention (e.g. explicitly removing atoms
  6/24/27 or re-orthogonalising them) was authorised in this round.
* The dense→sparse R² of 0.97 is a statement about *linear* recoverability
  on this split; a 2–3 % nonlinear residual is unresolved and is not claimed
  to be meaningless.
* OMP is an approximation, not a globally optimal top-8 solver; the audit
  never claims "the exact top-8 code" for it.  The K-SVD-dictionary OMP
  comparison in the prior round remains valid *for that dictionary*.

## 10. Next round (recorded only, not executed)

The audit points at the **dictionary structure**, not the iteration count.
A justified follow-up preregistration would, for example, (a) test whether
removing the DC-triplet redundancy (orthogonalising 6/24/27, or a
usage-balance / coherence penalty on `D`) changes reconstruction, usage and
the Sparse-vs-DenseTied margin at matched capacity, and (b) evaluate whether
the sparse arm has any advantage in a regime where phi65 is not dominated by a
single DC direction.  Per the round protocol none of this was started; the
existing thresholds, K, s, λ and step count stay frozen.
