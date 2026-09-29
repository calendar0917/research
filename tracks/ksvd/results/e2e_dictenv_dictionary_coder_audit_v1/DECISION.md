# DECISION — e2e_dictenv_dictionary_coder_audit_v1

Date: 2026-09-29 (CPU-only audit; official ZINC test never loaded)
Preregistration: `notes/e2e_dictenv_dictionary_coder_audit_v1_preregistration.md`
(commit `b325770`, sha256 `cc8856d96f6a…`)
Evidence: `REPORT.md`, `notes/e2e_dictenv_dictionary_coder_audit_v1_analysis.md`,
`analysis_tables.md`, `gate.json`, the stage directories under
`results/e2e_dictenv_dictionary_coder_audit_v1/`.

## Decision

```text
1. NO NEW TRAINING.  The frozen IHT-30 gate did not fire:
   gate.json fired = False, supported seeds [], classification ['NONE']; the
   substantive half (support / concentration / geometry) fails on all three
   seeds by wide margins, so the decision is robust to the condition-1
   reconstruction factor.  No IHT-30 seed-0 trajectory, no predictor of any
   kind, is authorised from this audit.

2. SPARSE-SPECIFICITY QUESTION REMAINS OPEN AND IS NOW LOCALISED.
   The audit excludes "IHT-10 is under-converged in its support" and
   "sparsification discards information" as explanations, and identifies two
   structural causes: (a) the trained dictionary spends three of eight
   activations on a near-duplicate DC/scale triplet (atoms 6/24/27, 100 % rows,
   0.76-0.98 aligned with the mean direction) plus a PC1 atom (23), so the code
   is concentrated by construction; (b) the sparse tie code and the dense tied
   coordinate are near-linear transforms of each other (R² 0.971-0.977 and
   0.9988-0.9989), so there is little representation difference for specificity
   to live in.

3. KEEP FINAL-CLEAN UNCHANGED: C6 + paired node binding + paired edge binding
   + sparse tied-IHT (K=32, s=8, IHT-10) + full relation.  No architecture,
   loss, feature, relation, split, K/s/lambda/step count survives this audit as
   a justified change.
```

## Because

* **H1 (coder under-convergence): amplitude only.**  IHT-30/100 cut the
  reconstruction error 27–40 % but the support is already the IHT fixed point
  (IHT-10↔IHT-30 Jaccard 0.972–0.980, exact support match 0.889–0.916, code
  cosine 0.9999) and usage/concentration metrics do not move (`N_eff`
  +0.32/−0.01/−0.13, top-5 drop ≤ 0.063, max activation rate drop 0.000).
  The frozen gate's condition 1 was written as factor ≤ 0.5 and measured
  0.683/0.732/0.704 — a miss that is itself the amplitude-tail effect.
* **H2 (intrinsic concentration): supported.**  Atoms 6/24/27 are active in
  every row of every seed, `|cos(·, μ)| = 0.76–0.98`, and 6/24 are 0.884–0.987
  collinear; atom 23 is the PC1 atom (`|cos(·, PC1)|` 0.71–0.80) and 83–98 %
  active.  Exactly `{6, 24, 27}` exceed `|cos(·, μ)| = 0.5`.  The dominant
  atom's coefficient is a patch-mass coordinate (Spearman 0.93–0.94 with
  `patch_nodes`), not the variance direction (0.32–0.44 with PC1 at 57.7 %
  explained variance).  The concentration is already present in OMP
  (atom 6 active on 97.6/97.7 % of rows), so it is a dictionary property.
* **H3 (near-linear re-parameterisation): supported.**  Train-only OLS on
  valid: dense→IHT-10 R² 0.9708/0.9763/0.9770, IHT-10→dense R²
  0.9988/0.9989/0.9989, code cosine 0.9996–1.0000, CKA 0.9987–0.9999; the
  DenseTied control's own dictionary gives 0.9720/0.9978, so the equivalence
  is not an artefact of one `D`.
* **H4 (sparsification loses information): not supported.**  IHT-10→dense R²
  0.9988–0.9989 means the dense coordinate has essentially no linearly
  recoverable content beyond the sparse code in this setup; the earlier
  DenseTied ≥ sparse ordering cannot be explained by information destroyed by
  hard thresholding.
* **OMP is not an oracle here.**  On the trained dictionaries OMP(s = 8)
  reconstructs 3.2–5.5× worse than IHT-10 (0.0398/0.0418/0.0654 vs
  0.0124/0.0109/0.0119) with top-1/L1 0.56–0.67 vs 0.28–0.30; the frozen
  wrapper matches an independent `sklearn.orthogonal_mp` call and the worst
  supports are well-conditioned, so this is greedy-selection geometry.  The
  prior "exact OMP is 700× better" line was measured on the K-SVD dictionary,
  which was fit for OMP; the trained dictionary was shaped by IHT gradients.
  No future claim may treat OMP as the ground-truth coder on a trained `D`.
* **Vocabulary stability is partial** (`PARTIAL_VOCABULARY`): mean matched
  `|cos|` 0.775, #≥0.90 = 11.7/32, median profile cosine 0.853.  The universal
  atoms replicate exactly; rare atoms only partly.  A "learned vocabulary"
  claim must be restricted to the stable core.
* **Gate discipline.**  Condition 2 needs support +0.05 (measured |≤0.006|),
  `N_eff` +1.0 (measured ≤ +0.32), top-5 −0.10 (≤0.063), max-rate −0.10 (0.0),
  top1/L1 −0.05 (≤0.001) or code cosine +0.05 (≤0.0007); 0/3 seeds satisfy any
  sub-criterion.  Relaxing condition 1 cannot flip the gate, so "no training"
  is a robust preregistered outcome, not a threshold technicality.

## Alternatives rejected

* **Run the IHT-30 seed-0 training anyway ("only amplitude, but cheap").**
  Rejected: the preregistered gate did not fire; the round explicitly forbids
  training without it, and the gate's condition 2 is exactly the test of
  whether the extra iterations change anything a predictor could see.  The
  evidence already shows they do not change support, usage or geometry.
* **Re-interpret the OMP result as an IHT-10 failure and lower the
  condition-1 factor post hoc.**  Rejected: OMP is a worse coder on this
  dictionary, so its code cannot define "what IHT should have found"; and
  post-hoc threshold edits are forbidden by the preregistration.
* **Claim a coder fix (IHT-100) from the reconstruction gain.**  Rejected:
  the gain is real but buys no structural change, and no downstream predictor
  was (or should be) trained to convert it into a task claim.
* **Declare sparse specificity established because the sparse code retains
  all dense information.**  Rejected: the audit shows the opposite reading —
  the two representations are near-equivalent, which explains why specificity
  fails; retention of information is not specificity.
* **Orthogonalise atoms 6/24/27 or add a usage penalty now.**  Rejected: new
  architecture/intervention requires its own preregistration; recorded as the
  next-round shape instead, not executed.

## Revisit if

A new preregistration targets the dictionary structure rather than the coder:
(a) remove the DC-triplet redundancy (orthogonalise 6/24/27, or a coherence /
usage-balance penalty on `D`) and test at matched capacity whether
reconstruction, usage concentration and the Sparse-vs-DenseTied margin move;
(b) test the sparse arm on a phi65 variant not dominated by one DC direction.
Secondary: a 4th dictionary seed would tighten the `PARTIAL_VOCABULARY`
boundary, and dropping the two identically-zero descriptors
(`root_neighbour_shell1`, `root_walk1`) is a mechanical cleanup for any
descriptor revision.  None of these is authorised or started by this audit.
