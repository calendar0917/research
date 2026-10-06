# Preregistration: zinc_cssd_nonlinear_binding_v1 (2026-10-06 basis 50eaf5f7)

Protocol `zinc-cssd-nonlinear-binding-v1`, runner `zinc_cssd_nonlinear_binding_v1`,
track ksvd, study zinc-context-gap.  This round freezes the CSSD structural basis
and tests **pre-pooling, member-level nonlinear joint structure–semantics
correspondence** on nodes and real induced bonds, against a **separated additive
binding** of the same marginals, inside the current strongest task path
(M_COMP + Q lineage).

## Candidate — closest old experiment — substantive change

| Candidate | Closest old experiment | Substantive change in this round |
| --- | --- | --- |
| A (strength reference) | M_COMP + Q (zldc `zinc_local_dictionary_component_supervision_seed0_v1`, `zinc_component_supervision_fulltrain_confirmation_seed0_v1`) | A restores the **original product node/edge binding slots** (W_A_S/W_A_C/W_E_S/W_E_C + slot encoders, 446-D fusion layout) on a **fresh fit-only CSSD basis**; the deploy 110-D fusion of M_COMP is widened back to the canonical Full 446-D input; posterior bridge stays the matched MLP; no trained weights reused |
| N_joint / N_sep (node member binding) | F fusion construction (`zinc_dictionary_fusion_clarity_overnight_seed0_v1`): z(root) ⊙ semantic projection then incidence pooling | F factorises into z(root) × a semantic aggregate; it never maps **each member's own z(v) jointly with its atom type before pooling**.  N_joint is a 61→64→48 MLP over [z_v;q_v] evaluated per node occurrence; N_sep keeps identical marginals (MLP_S(z) + MLP_C(q)) and interface but removes the member-level joint function.  The old node product binding in the canonical Full is **inactive** (bottleneck audit: W_A_S denorm, zero_frac=1); merely reviving it is insufficient, so A is a reference, not the candidate |
| E_joint / E_sep (bond endpoint pairing) | Canonical Full edge binding: [cu+cv; \|cu−cv\|; cu\*cv]@W_E_S ⊙ bond@W_E_C | The old edge binding never binds **each endpoint's own atom type inside the edge branch**, and its sum/diff/product inputs delete endpoint identity.  E_joint is a symmetric ψ(126→64→32) over the **endpoint vectors** [x_v;x_w;b_e], x=[z;q]; E_sep keeps both marginal expressivities (symmetric ψ_S over [z_v;z_w] + symmetric ψ_C over [q_v;q_w;b]) without per-endpoint joining; **no coordinatewise sum/diff substitution** |
| Frozen CSSD basis | `e2e_dictenv_common_subspace_dictionary_v1` (CSSD q1) | Same scheme (U=fit mean direction, D_perp colnorm, top-8 tied-IHT-10), but refit **on the new 8001-row fit split only**: K-SVD (10 epochs, seed 20260924) on fit phi rows, 320-epoch CSSD training on fit, soup = **mean of D over epochs 316..320** (never Top-5 on a holdout; the historical soup was selected on official valid).  U/common RMS/D become **buffers shared and frozen across all five arms and both seeds** (historically D trained per model) |
| Fold 8000/1000/1000 | zldc's 8000/2000 two-way fold (rng 20261006) | Same permutation seed but a **three-way** split; SMILES-duplicate groups straddling a boundary move wholly into fit (1 group, 1 row moved: fit=8001/select=999/confirm=1000; fit sha 429150b7… differs from zldc's, so **all** zldc prep/targets/kappa artifacts are refit, none reused) |
| Q head | fulltrain confirmation round's Q (10k train) | Q trained on the **fit rows only** (300 epochs, last-5 average), frozen hash shared by every arm/seed |
| Mechanism interventions | fusion-clarity bridge interventions (root-level) | Branch-level pairing shuffles: N shuffle permutes q↔z correspondence within (root,shell) groups; E shuffle permutes complete chemistry endpoint tuples vs structure endpoint pairs within (root,shellpair,bondtype).  C00 is **exactly invariant** by construction (pooled sums of additive branches cannot see the pairing); joint arms must respond |

Old negatives that do **not** decide this round: frozen convex head / local covariance /
triad witness (linear, specific parents); SRDA channel-deletion confound (all original
channels are kept here); the inactive old node product binding (kept only as reference A).

## Arms

- **A** — original product binding slots + the M_COMP+Q task path, fresh init (436,499 trainable params incl. 24,048 original slot path).
- **C00** — N_sep + E_sep (429,783), **C10** — N_joint + E_sep (429,779), **C01** — N_sep + E_joint (429,751), **C11** — N_joint + E_joint (429,747).  C-to-C spread = 36 params; no filler balancing.

Shared by all arms: fusion 446→342→144 (canonical Full widths), matched MLP bridge,
ComponentReader (39→2), root-tuple local MLP (W_loc 342×64 zero-init, kappa_M from the
new fit), static pair path, global/topology encoders, reader; CSSD basis frozen
(buffers); one shared Q; one recipe.

## Recipe (identical for the five arms)

240 epochs, batch 128, Adam 1e-3 / wd 1e-5, clip 5, locked schedule (seed+101),
soup = mean of epochs 236..240, COMP loss `L = L1(g) + 0.5*(L1(ell)+L1(s))`,
`y_raw = ell_hat + s_hat + Q_raw`, `y_cal = y_raw + b_y` (single fit-median bias).
Diagnostics only at epochs 1/40/120/240.  select/confirm never read during training.

## Budget (frozen)

- Stage 0: one build (fold/prep/CSSD/Q) + focused checks + GPU smoke.
- Stage A: 5 seed-0 body runs (A, C00, C10, C01, C11) on select; primary ranking =
  full **y_raw** MAE; calibrated reported alongside (calibration-only gains are not
  a positive signal).
- Stage B: one-shot confirm on the frozen roster after both seeds' soups are frozen.
- Seed-1 purchase rules (frozen): all-joint-active-no-gain → STOP; only C00 improves →
  re-check A+C00; one candidate → A+C00+chosen (3 runs); two candidates or C11
  interaction → all five (5 runs).  Max 10 body runs total; clean negative = 5.

## Execution plan

res-2 via the rr CLI only (pool res2-cu124, 1 GPU / 4 CPU per job): stage-0 smoke
job first, then five independent seed-0 train jobs (`cssd-bind-v1-train-<arm>-s0`),
then select-eval / confirm-eval / interventions jobs; results pulled via `rr pull`
and committed.  Official valid and test are never instantiated; `test_access=granted`
is refused by the runner.
