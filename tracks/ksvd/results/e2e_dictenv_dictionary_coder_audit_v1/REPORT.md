# REPORT — e2e_dictenv_dictionary_coder_audit_v1

CPU-only representation audit on the closed ZINC dictionary-environment line
(Workstream Z).  No predictor was trained; the frozen IHT-30 training gate did
not fire.  No GPU, no SSH, no architecture / feature / split change, official
ZINC test never loaded (`official_test_loaded = false` in every payload).

* Preregistration: `notes/e2e_dictenv_dictionary_coder_audit_v1_preregistration.md`
  (frozen before Stage A; commit `b325770`).
* Evidence: `results/e2e_dictenv_dictionary_coder_audit_v1/` —
  `preflight.json`, `preregistration_snapshot.json`, `coder_geometry/`,
  `dominant_atom/`, `atom_specialization/`, `cross_seed_stability/`,
  `sparse_dense_recoverability/`, `gate.json`, `analysis_tables.md`,
  `summary.json`, `coder_geometry/coder_geometry.csv`,
  `atom_specialization/atom_specialization.csv`.
* Full discussion: `notes/e2e_dictenv_dictionary_coder_audit_v1_analysis.md`.

## A. Bottom line

1. **The IHT-10 coder is structurally converged and only amplitude-limited**
   (H1: coder under-convergence).  IHT-30/100 reduce reconstruction 27–40 %
   (absolute 2.9–3.9e-3) but change neither the support (IHT-10↔IHT-30
   Jaccard 0.972–0.980, exact support 0.889–0.916), nor usage (`N_eff`
   14.5–15.3, top-5 4.4–4.8/8, max activation rate 1.000), nor coefficient
   geometry.  More IHT iterations buy coefficients, not structure.
2. **The learned dictionary is intrinsically concentrated** (H2).  In every
   seed, atoms **6, 24, 27 are active in 100 % of rows** and are 0.76–0.98
   aligned with the dataset mean direction μ (6 and 24 are 0.884–0.987
   collinear); atom **23** is 0.71–0.80 aligned with PC1 and 83–98 % active.
   Exactly `{6, 24, 27}` have `|cos(·, μ)| > 0.5`.  The dominant atom's
   coefficient is a patch-scale (DC) coordinate (Spearman 0.93–0.94 with
   `patch_nodes`).  The concentration is already present in OMP, so it is a
   dictionary property, not a coder artefact.
3. **Sparse is largely a linear re-parameterisation of the dense tied
   coordinate** (H3).  Train-only OLS on valid: dense→IHT-10 R² 0.971–0.977,
   IHT-10→dense R² 0.9988–0.9989, code cosine 0.9996–1.0000, CKA ≈ 0.999;
   the pattern repeats on the DenseTied control's own dictionary.
4. **Sparsification does not lose downstream-relevant information** (H4) in
   the linear recoverability sense: IHT-10→dense R² 0.9988–0.9989 means the
   8-sparse code retains essentially everything linearly extractable from the
   dense coordinate.  The earlier "specificity not established" result is
   therefore not a lost-information problem — the two coordinates are nearly
   equivalent representations on this object.
5. **OMP is not an oracle coder on this dictionary.**  With the same
   normalised `Dbar`, OMP(s = 8) reconstructs 3.2–5.5× worse than IHT-10
   (0.0398/0.0418/0.0654 vs 0.0124/0.0109/0.0119) and is more top-heavy
   (top-1/L1 0.56–0.67 vs 0.28–0.30).  The "exact coder is 700× better"
   diagnostic from the prior round was measured on the K-SVD dictionary, which
   was fit *for* OMP; the trained dictionary was shaped by IHT gradients.
   Coder fidelity is a dictionary–coder pair property.
6. **Frozen gate did not fire; no training was run**.
   `gate.json: fired = False`, all three seeds `NONE`.  Condition 1
   (reconstruction) missed the frozen 0.5× factor (0.68–0.73) while the
   absolute drop is material (2.9–3.9e-3), and condition 2 (support /
   concentration / geometry) fails on all three axes on all three seeds by
   wide margins.  **NO NEW TRAINING WAS SCIENTIFICALLY JUSTIFIED.**  The
   decision is robust to the condition-1 threshold because condition 2 —
   the substantive half — fails everywhere.
7. **Vocabulary stability is partial**: frozen verdict
   `PARTIAL_VOCABULARY` — mean matched `|cos|` 0.775, #≥0.90 = 11.7/32, median
   profile cosine 0.853.  The universal/mid-frequency atoms replicate; rare
   atoms only partially.

## B. Protocol and provenance

| item | value |
|---|---|
| device | local CPU only (`CUDA_VISIBLE_DEVICES=""`), 3 procs × 4 threads |
| data | official ZINC train 10 000 / valid 1 000 from the frozen P1 phi65 cache; official test never loaded |
| audited dictionaries | C6 soup `D` seeds 0/1/2 (bit-hash recorded in `preflight.json`); DenseTied seed 0 context |
| normalisation | column-normalised `Dbar` (float64 / float32 torch), the operator the model uses |
| coders | frozen `v0.tied_iht_codes` (10/30/100), frozen `tccd_v0.omp_codes` (s = 8), DenseTied `phi @ Dbar` |
| preregistration | `notes/e2e_dictenv_dictionary_coder_audit_v1_preregistration.md`, sha256 `cc8856d96f6a…`, committed `b325770` before analysis |
| focused tests | `tests/test_e2e_dictenv_dictionary_coder_audit_v1.py`, 20/20 pass |
| descriptors | 32 named topology-only descriptors from the verified phi65 provenance (two are identically zero; recorded as a defect) |
| statistics | train-only SMD / PCA / OLS; valid only for evaluation |

## C. Stage A — same-D coder audit (valid)

| seed | coder | recon_fro | recon_row_sq | active | N_eff | top-5 | max rate | Gini | J(OMP) | cos(OMP) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | IHT-10 | 0.012388 | 1.40e-4 | 27 | 14.52 | 4.796 | 1.000 | 0.654 | 0.208 | 0.609 |
| 0 | IHT-30 | 0.008463 | 6.3e-5 | 30 | 14.84 | 4.733 | 1.000 | 0.648 | 0.204 | 0.609 |
| 0 | IHT-100 | 0.007359 | 4.6e-5 | 31 | 14.86 | 4.733 | 1.000 | 0.648 | 0.204 | 0.608 |
| 0 | OMP | 0.039774 | 1.58e-3 | 32 | 19.96 | 4.025 | 0.976 | 0.525 | — | — |
| 1 | IHT-10 | 0.010885 | 1.06e-4 | 27 | 14.80 | 4.559 | 1.000 | 0.647 | 0.216 | 0.667 |
| 1 | IHT-30 | 0.007968 | 5.6e-5 | 27 | 14.79 | 4.559 | 1.000 | 0.648 | 0.222 | 0.667 |
| 1 | IHT-100 | 0.007238 | 4.6e-5 | 28 | 15.26 | 4.559 | 1.000 | 0.637 | 0.222 | 0.667 |
| 1 | OMP | 0.041820 | 1.75e-3 | 32 | 22.54 | 2.957 | 0.977 | 0.440 | — | — |
| 2 | IHT-10 | 0.011858 | 1.28e-4 | 28 | 15.35 | 4.409 | 1.000 | 0.637 | 0.264 | 0.605 |
| 2 | IHT-30 | 0.008352 | 6.1e-5 | 31 | 15.22 | 4.409 | 1.000 | 0.639 | 0.259 | 0.606 |
| 2 | IHT-100 | 0.007297 | 4.6e-5 | 31 | 15.24 | 4.408 | 1.000 | 0.639 | 0.260 | 0.607 |
| 2 | OMP | 0.065397 | 4.25e-3 | 32 | 21.34 | 3.316 | 0.828 | 0.483 | — | — |

Convergence (descriptive, added after the gate; no threshold):
IHT-10↔IHT-30 support Jaccard 0.972/0.980/0.975, code cosine 0.9999,
normalised L2 0.012/0.010/0.011.  Coefficient geometry (valid mean): IHT-10
`top1/L1 0.280–0.301`, OMP `0.561–0.671`.  Frozen labels: `NONE` ×3; the
evidence is A2 (amplitude under-convergence) + intrinsic concentration.

## D. Stage B — dominant atom / common direction

| seed | dominant atom | activation | \|cos(atom, μ)\| | \|cos(atom, PC1)\| | PC1 var |
|---|---|---|---|---|---|
| 0 | 6 | 1.000 | 0.965 | 0.342 | 0.577 |
| 1 | 6 | 1.000 | 0.983 | 0.436 | 0.577 |
| 2 | 6 | 1.000 | 0.941 | 0.322 | 0.577 |

Universal atoms (active in 100 % of rows, all seeds): **6** (`|cos μ|`
0.94–0.98), **24** (0.90–0.94, 0.88–0.99 collinear with 6), **27** (0.76–0.86);
PC1 atom **23** (`|cos PC1|` 0.71–0.80, active 0.83–0.98).  The atom-6
coefficient tracks patch mass/scale (`patch_nodes` 0.93–0.94,
`std_log1p_walk3` 0.92–0.93, `patch_edges` 0.80–0.88).  The model spends three
of eight activations on a near-duplicate DC triplet plus one PC1 atom in most
rows — the arithmetic origin of top-5 ≈ 4.4–4.8 and `N_eff` ≈ 14.5–15.3.

## E. Stage C — atom structural specialisation

Median Spec (IHT-10): 0.564/0.349/0.619; usage-weighted 0.207/0.191/0.237;
median top-response Spec 0.837/0.953/1.067.  Median train→valid SMD profile
cosine 1.000 on every seed/coder (all finite values ≥ 0.926).  The most
specialised atoms are rare (e.g. seed 0 atoms 12/8: `root_walk3`,
`edge_frac_shellpair_11`, `mean/std edge_log1p_common_neighbours`, Spec
3.3–3.7, usage 0.001–0.06); the always-active DC triplet scores 0.00
mechanically.  Atoms carry stable, transferable structural roles, but the
usage-weighted role strength is modest.  Defect recorded: `root_neighbour_shell1`
and `root_walk1` are identically zero in the frozen phi65 layout.

## F. Stage D — cross-seed dictionary stability

| pair | mean \|cos\| | median | min | p10 | #≥0.90 | #≥0.95 | median profile cos |
|---|---|---|---|---|---|---|---|
| 0↔1 | 0.738 | 0.807 | 0.117 | 0.342 | 13 | 6 | 0.879 |
| 0↔2 | 0.777 | 0.828 | 0.225 | 0.492 | 10 | 7 | 0.756 |
| 1↔2 | 0.809 | 0.855 | 0.178 | 0.635 | 12 | 6 | 0.891 |

Frozen verdict **PARTIAL_VOCABULARY**.  The universal atoms replicate exactly
(6↔6 0.97–0.99, 24↔24 0.96–0.98, 27↔27 0.91–0.94, 23↔23 0.68–0.95) as do
mid-frequency atoms (9↔9, 30↔30); rare-atom matching is partial (35/96 ≥ 0.90)
and occasionally semantically inconsistent (0↔2 atom 26↔26 `|cos|` 0.838,
profile cos −0.968).

## G. Stage E — sparse ↔ dense recoverability (valid, train-only fit)

| seed | direction | R² | per-dim median R² | norm. error | mean cos | CKA |
|---|---|---|---|---|---|---|
| 0 | dense → IHT-10 | 0.9708 | 0.9746 | 0.0314 | 0.9996 | 0.9987 |
| 0 | IHT-10 → dense | 0.9988 | 0.9999 | 0.0063 | 1.0000 | 0.9998 |
| 1 | dense → IHT-10 | 0.9763 | 0.9778 | 0.0279 | 0.9997 | 0.9991 |
| 1 | IHT-10 → dense | 0.9989 | 1.0000 | 0.0059 | 1.0000 | 0.9999 |
| 2 | dense → IHT-10 | 0.9770 | 0.9823 | 0.0273 | 0.9997 | 0.9988 |
| 2 | IHT-10 → dense | 0.9989 | 0.9997 | 0.0061 | 1.0000 | 0.9998 |
| 0 | dense_ownD → IHT-10 | 0.9720 | 0.9724 | 0.0308 | 0.9996 | 0.9987 |
| 0 | IHT-10 → dense_ownD | 0.9978 | 0.9998 | 0.0080 | 1.0000 | 0.9999 |

IHT-30 = 0.970–0.975 / 0.9993–0.9994; OMP is a weakly-fitting alternative
(dense→OMP 0.74–0.87).  The 2–3 % dense→sparse residual is the sparse code's
nonlinear content and is not associated with any downstream advantage.

## H. Frozen IHT-30 training gate

`gate.json`: **fired = False**, supported seeds [], per-seed classification
`['NONE']`.

| seed | cond 1 recon factor (≤0.5) | absolute drop (≥1e-4) | cond 2 support | cond 2 concentration | cond 2 geometry |
|---|---|---|---|---|---|
| 0 | 0.683 ✗ | 3.93e-3 ✓ | −0.0041 ✗ | +0.32 / −0.063 / 0.0 ✗ | −0.0004 / −0.0004 ✗ |
| 1 | 0.732 ✗ | 2.92e-3 ✓ | +0.0060 ✗ | −0.01 / 0.0 / 0.0 ✗ | −0.0004 / −0.0002 ✗ |
| 2 | 0.704 ✗ | 3.51e-3 ✓ | −0.0051 ✗ | −0.13 / 0.0 / 0.0 ✗ | −0.0010 / +0.0007 ✗ |

The relative-factor miss of condition 1 is the amplitude effect; the
substantive condition 2 fails on every axis everywhere.  **NO NEW TRAINING WAS
SCIENTIFICALLY JUSTIFIED.**  Single-seed-IHT-30 discipline
(`FINAL-CLEAN + IHT-30, seed 0`, one trajectory) was not entered; no
`iht30_training/` artifact exists.

## I. Synthesis

The dictionary and the coder are now disentangled.  The coder's loss is an
amplitude tail; its support is the IHT fixed point.  The dictionary is a
partially redundant overcomplete basis: a three-atom DC/scale clone plus a PC1
atom dominate usage, and the remaining atoms are rare, structurally
specialised and only partially seed-stable.  The sparse tie code and the dense
tied coordinate span nearly the same linearly recoverable subspace (R² ≥ 0.97
both directions), which explains why specificity was not established: the
sparse arm's 2–3 % unique nonlinear variation buys no downstream accuracy.
No coder defect justifies a longer IHT; the actionable object is the
dictionary's DC redundancy / usage balance.

## J. Limits

Three seeds, one architecture family, one dataset, valid only; OMP is an
approximate solver (never described as the exact top-8 optimum); no causal
intervention on atoms 6/24/27 was authorised; the 2–3 % nonlinear residual is
unresolved; two zero descriptors are recorded, not fixed.

## K. Provenance

Preflight, checkpoint hashes and the preregistration sha256 are in
`preflight.json`; `summary.json` records the git commit of every generated
payload; the runner was not modified in any measurement path after the freeze
(the post-freeze additions are the CSV exports, the convergence stage and the
per-atom markdown section, all reading stored payloads / re-calling frozen coder
functions; no threshold or gate was touched).  `docs/luyin/` is untouched.

## L. Recorded next step (not executed)

A new preregistration targeting the dictionary structure: (a) does removing
the DC-triplet redundancy (orthogonalise 6/24/27, or add a coherence /
usage-balance penalty on `D`) change reconstruction, usage and the
Sparse-vs-DenseTied margin at matched capacity; (b) does the sparse arm show
any advantage in a phi65 variant not dominated by one DC direction.  No such
run was started here.
