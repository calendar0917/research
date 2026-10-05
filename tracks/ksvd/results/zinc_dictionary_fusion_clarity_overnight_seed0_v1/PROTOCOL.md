# PROTOCOL — zinc_dictionary_fusion_clarity_overnight_seed0_v1

Frozen before any formal training trajectory. Execution module:
`tracks/ksvd/experiments/luyin16/zinc_dictionary_fusion_clarity_overnight_seed0_v1.py`
(reusing the verified production sources of
`zinc_local_dictionary_component_supervision_seed0_v1` (`src`) and its
dependencies `prev`/`mlpmod`/`zfr`/`zcs`/`zftd`/`zcdm`/`zw`/`cm` verbatim).
Deviations are recorded in EXECUTION.md; frozen scientific definitions are never
changed after dev scores are seen.

## Question

Is the joint 125-D tuple coding (structure+semantics fused in one dictionary
input) the wrong information division of labour?  At matched parameters,
matched body/bridge/reader, matched COMP supervision and the identical real J
incidence, compare it against a *separate* structure coding (pure-topology
phi65) explicitly combined with a per-tuple semantic branch.

## Arms (Stage 1, split A = source rng(20261006) 8000/2000, all objects reused verbatim)

| arm | local construction | 8,000 local parameters |
|---|---|---|
| J_D | joint 125-D tuple -> source tied IHT top-8 (source-round D_COMP) | D_raw[125,64] |
| J_M | joint 125-D tuple -> SiLU projection (source-round M_COMP) | A_raw[64,125] |
| F_D | z = source-tied IHT(p65) top-8; u = SiLU(q60 @ B_bar.T); f = z*u; e = sum wJ*f | D_S[65,64] 4,160 + B[64,60] 3,840 |
| F_M | z = SiLU(p65 @ A_S_bar.T); same semantic branch and fusion | A_S[64,65] 4,160 + B[64,60] 3,840 |

q(v,a,t) = [onehot28(root atom); onehot28(neighbour atom); onehot4(bond)] (60-D);
p65 = standardised pure-topology phi (A-fit scaler and phi_scale reused verbatim).
Injection identical to the source round: `fusion0(Sem110) + W_loc @ (kappa*e)`,
`W_loc[342,64]` zero-init; posterior bridge = matched MLP 144->288->144 in every
arm; reader = 39->2 half-split (COMP); C6 mask, sum/squared-sum, static
relations, topology25 and the rest of the body unchanged; no message passing /
Transformer anywhere.

Initialisation from the canonical untrained frame D0[125,64] (seed 20261004):
J_D.D=D0; J_M.A=D0.T; F_D.D_S=D0[:65,:]; F_M.A_S=D0[:65,:].T; both F arms share
B=D0[65:,:].T.  Every other parameter is bit-identical across the four arms
(verified: shared tensor max diff 0.0; identical initial graph function from
W_loc=0).  297,539 registered parameters per arm.

## Scales (Stage 0, fit-only, label-free, frozen)

R* = RMS(kappa_D * e_JD_init) over the source round's frozen <=8192 fit-root
sample (J weights, fresh init) = 1.000237.  kappa_F_D = R*/RMS(e_F_D_init) =
17.911595; kappa_F_M = R*/RMS(e_F_M_init) = 14.057879.  J_M keeps the
source-verified kappa_M = 1.983953.  No per-dim/dynamic/learnable scaling; no
re-matching later.  Split-B F arms re-match with the same recipe on the B
sample at the Stage 3 purchase.

## Recipe (every formal arm)

seed 0; Adam lr 1e-3, coupled WD 1e-5, clip 5.0; batch 128; 240 epochs =
15,120 steps (8,000-fit); soup 236-240; checkpoints 1/40/120/240; schedule
seed 101 + identical global graph-ID stream across arms; COMP loss
`MAE(hat_g,g) + 0.5*(MAE(hat_ell,ell)+MAE(hat_s,s))` with fixed 0.5;
calibration `b_g = median(g_fit - raw_soup_fit)` fit-only; dev predictions only
after the roster freeze.  Construction/diagnostics use isolated RNG.

## Endpoints and statistics

Dev G0(k=0) calibrated g-MAE main; overall cal common gate; raw in parallel;
group contributions with additivity identity; sensitivity = drop the common
reference's worst dev row once.  Paired bootstrap 2000x, seed 20261010, ONE
shared per-graph resample index set per draw across all arms and gains.
Stage-1 contrasts: G_J = E(J_M)-E(J_D); G_F = E(F_M)-E(F_D); C_D = E(J_D)-E(F_D);
C_M = E(J_M)-E(F_M); I = G_F-G_J = C_D-C_M (per-draw identity verified).
"Clear performance signal" (vs the corresponding reference): G0 cal gain
>= .003 AND overall cal gain >= .003 AND G0 cal CI lower > 0 AND both raw
gains > 0.  This is this round's investment threshold, not a general truth.

## Stage 2 (split A, purchase: F interface alive, operators verified, budget)

F_D_I / F_M_I (source marginal-I incidence weights, from scratch);
F_D_RAND (structure dictionary frozen at the random frame; 297,539 registered,
293,379 trainable - disclosed); F_D_REC (1000-step label-free REC pre-training
then frozen) and F_D_REC_TASK (same pre-trained dictionary, trainable).  Single
REC recipe: batch 256, 1000 steps, Adam 1e-3, WD 1e-5, clip 5, sampler
rng(20261010) with replacement over the frozen fit-root sample; loss
`mean(||p - alpha Dbar.T||^2/(||p||^2+1e-12))`; fixed last, no best selection;
kappa_REC = R*/RMS(e_REC_J) re-matched on the same sample.  Non-finite stops
REC.  Matched-control interpretation only (RAND vs F_D: task adaptation from a
random basis; REC_TASK vs REC: task adaptation from the same reconstruction
basis; REC vs RAND: two fixed bases; REC_TASK vs F_D: pre-trained init).

## Stage 3 (split B = rng(20261007) 8000/2000, indices frozen at Stage 0)

Purchase only if some D candidate (F_D/F_D_I/F_D_RAND/F_D_REC/F_D_REC_TASK)
reaches vs J_M: G0 cal gain >= .002, overall cal gain >= .001, both raw > 0,
interface healthy; or no D but an M candidate (F_M/F_M_I) meets the full clear
signal.  Selection: best D by G0 cal gain; ties <= .0003 resolved in the fixed
order F_D, REC_TASK, REC, RAND, I.  M_best = better of F_M/F_M_I by G0 cal
(ties <= .0003 -> J version).  Train on B: D_best + M_best + J_M (3 arms), or
M_best + J_M (2 arms).  All B statistics refit on the B fit rows from fresh
init; B roster frozen before any B model score; no B-driven additions.

## Stage 4 (only on a confirmed clear signal + mechanism + ETA)

Two full-10k fresh arms (candidate + matched reference), same COMP recipe,
18,960 steps, soup 236-240; all constants/scales refit on 10k.  Only then one
frozen explicit official-valid read (g and y reported separately; valid is
disclosed as previously-used exploration, not pristine).  Official test stays
closed.  No rule changes after the valid read.

## Forbidden

New message passing/Transformer; new handcrafted features; node revival; ring
tail weighting; bridge/reader variants; width/WD/loss/K/s/step sweeps; extra
seeds; epoch selection by dev; post-hoc regrouping; test access of any kind;
cross-split/cross-regime gain splicing; re-running trajectories for a bad
loss; parametric rescue of the closed joint-IHT recipe; pushing/merging.
