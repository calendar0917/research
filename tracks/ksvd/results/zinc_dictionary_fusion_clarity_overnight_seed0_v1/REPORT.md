# REPORT — `zinc_dictionary_fusion_clarity_overnight_seed0_v1`

**Date**: 2026-10-06 (overnight round, T0 ≈ 00:18 CST, reporting at ~04:10)
**Question (luyin19)**: is encoding structure and semantics **jointly into one
125-D tuple** a wrong division of information, versus **separate structure
coding (phi65) + explicit semantic branch (q60) fused per tuple** — under COMP
component supervision, seed 0, matched bodies (297,539 params each), matched
J-incidence pooling and matched kappa-scaled W_loc injection?

## Conclusion (headline, read this first)

1. **No.** There is no robust evidence that the separate construction (F)
   beats the joint 125-D tuple construction (J). Every Stage-1 construction
   contrast crosses zero (C_D, C_M, G_F, interaction I), and the two Stage-2
   signals that qualified for purchase **failed the pre-registered split-B
   replication** under the frozen gates (which require both raw gains > 0):
   the frozen-random-dictionary candidate **reversed sign**, and the
   best-arm candidate's replicated calibrated gain turned out to be a
   **per-arm median-bias (calibration-offset) effect, not a per-molecule
   accuracy effect**. Stage 4 (full-10k retrain + the single official-valid
   read) was therefore **not purchased**, and official-valid was never read
   this round.
2. **The dictionary-vs-MLP gap on the joint tuple (J_D vs J_M) is dominated by
   run-to-run training noise.** The source round found the MLP better (G0 cal
   −0.0020, raw CI entirely below zero); this round's contemporaneous control
   under the identical frozen recipe found the dictionary better (G_J G0 cal
   +0.0030; **raw +0.0119, CI [+0.0081, +0.0157] entirely above zero**). Two
   matched runs, opposite sign verdicts: neither sign is a property of the
   encoder class in this interface; the earlier "MLP better in this
   interface" claim is downgraded to noise-scale.
3. **Task adaptation of the structure dictionary directionally hurts, and
   training it at all buys nothing over a frozen random dictionary** (A-split:
   F_D < F_D_RAND by −0.0025, REC_TASK < REC by −0.0024, REC ≤ RAND by −0.0011;
   all CI-crossing) — but the whole D family failed replication on B, so this
   stays a directional observation, not a purchase-grade fact.
4. **A material historical ERRATA was found and fenced**: the fulltrain
   confirmation round's official-valid (and, via the same loader pattern, the
   cycle-level round's official-test) body inference attached **train**
   incidence structures to held-out graphs. Correcting the structure improves
   the frozen COMP body's valid y-MAE **0.1174 → 0.1122**. Quotations of those
   historical held-out numbers should stop until a single re-scoring pass
   under the corrected loader (prepared this round, unused because Stage 4
   was not purchased; see DECISION.md).

## Arms and matched construction (all 297,539 params, seed 0)

| arm | local encoder | structure code | semantic code |
|---|---|---|---|
| J_D | tied-IHT(10, top-8) D[125,64] | joint 125-D tuple | joint (inside tuple) |
| J_M | SiLU A[64,125] | joint 125-D tuple | joint |
| F_D | tied-IHT D_S[65,64] | phi65 alone | B[60,64] SiLU branch |
| F_M | SiLU A_S[64,65] | phi65 alone | B[60,64] SiLU branch |
| F_*_I / _RAND / _REC(_TASK) | F variants | marginal-I weights / frozen random D_S / REC-pretrained (frozen or task-adapted) | same |

Identical: body/bridge/reader, J-incidence pooling, `W_loc` zero-init
injection with one-shot kappa match (R\* = RMS(kappa_D·e_JD_init)), Adam
1e-3/WD 1e-5/clip 5, batch 128, 240 epochs = 15,120 steps, soup 236-240,
per-arm single fit-side median bias, COMP loss `L_g + 0.5(L_ell+L_s)`.

## Stage-1 (split A dev, 2,000 graphs; paired bootstrap 2,000×, seed 20261010, shared indices)

| arm | G0 cal | overall cal | G0 raw | overall raw | bias |
|---|---|---|---|---|---|
| J_D | 0.090011 | 0.090622 | 0.090384 | 0.091019 | −0.0099 |
| J_M | 0.093049 | 0.094072 | 0.102292 | 0.103470 | −0.0459 |
| F_D | 0.092324 | 0.092430 | 0.092831 | 0.092877 | −0.0130 |
| F_M | 0.090645 | 0.091335 | 0.090651 | 0.091340 | +0.0002 |

Contrasts (positive = subtrahend better): **G_J** +0.0030 cal [−0.0006,
+0.0064], **+0.0119 raw [+0.0081, +0.0157]** (sign-flipped vs the source
round's −0.0020 cal / raw-CI<0); **G_F** −0.0017 cal [−0.0055, +0.0020];
**C_D** −0.0023 cal [−0.0059, +0.0015]; **C_M** +0.0024 cal [−0.0011,
+0.0059]; **I_derived** −0.0047 [−0.0119, +0.0025]. Dev k-counts: k=0 1931,
k=−1 60, k=−2 7, k≤−3 2.

## Stage-2 (split A, five arms, all 15,120 steps)

G0 cal: F_M_I **0.087580** (best arm of the round) / F_D_RAND 0.089859 /
F_D_REC 0.090969 / F_D 0.092324 / F_D_I 0.094993 / F_D_REC_TASK 0.093328.

- **F_M_I vs J_M: +0.0055 cal [+0.0022, +0.0087] and +0.0087 raw [+0.0053,
  +0.0125]** — met the full "clear signal" gate on A.
- **F_D_RAND vs J_M: +0.0032 cal [−0.0001, +0.0065], +0.0101 raw** — met the
  D-candidate qualification minimum.
- Marginal-I weights matter only in the M arm: F_M→F_M_I −0.0031 (J−I
  contrast); F_D→F_D_I +0.0027 (wrong direction).
- Task adaptation hurts: F_D vs RAND −0.0025; REC_TASK vs REC −0.0024;
  REC vs RAND −0.0011; F_D_REC vs J_M +0.0021 (qualified, not selected).

## Stage-3 (split B, fresh objects refit on B fit rows, 3 arms, fresh retrain)

| arm | G0 cal | overall cal | G0 raw | overall raw | bias |
|---|---|---|---|---|---|
| B_J_M | 0.093657 | 0.095061 | 0.095004 | 0.096467 | −0.0193 |
| B_F_D_RAND | 0.096200 | 0.097965 | 0.097094 | 0.098920 | +0.0142 |
| B_F_M_I | 0.088571 | 0.090073 | 0.112584 | 0.114072 | −0.0695 |

Pre-registered contrasts vs B_J_M (frozen purchase minimum: G0 cal ≥ .002,
overall cal ≥ .001, **both raws > 0**):

- **B_F_D_RAND vs B_J_M: −0.0025 G0 cal [−0.0061, +0.0011]**, overall cal
  −0.0029, both raws negative → **FAIL (direction reversed).**
- **B_F_M_I vs B_J_M: +0.0051 G0 cal [+0.0017, +0.0084]**, overall cal +0.0050,
  **G0 raw −0.0176 [−0.0214, −0.0137]**, overall raw −0.0176 → **FAIL: the
  replicated calibrated gain comes entirely from the per-arm median bias
  (−0.0695 vs −0.0193); raw per-molecule accuracy is significantly worse.**

**Stage-4 purchase decision: NOT PURCHASED** (no full-10k arms trained; the
official-valid read never executed; FULL objects were built label-free in 8 s
CPU and left unused; the clean valid-structure reader + frozen cycle-module
reuse machinery exists in the module but was never pointed at held-out data).

## Mechanism health (all stages; epoch-240 soup probes)

- Zero-injection of the local channel collapses G0 cal to 0.46–0.63 in every
  arm (incl. B arms: 0.63/0.46/0.61) — the local interface carries real
  signal in every construction.
- J→I weight swap moves G0 cal by ≤ 0.010 in every arm (interface healthy,
  not degenerate); J/I identifiability at init ~5.6–5.9e-4 (identifiable).
- Dictionary codes stay exactly top-8 sparse; D-family atom usage 58–64/64;
  frozen dictionaries drift 0.000 by construction; trained D drifts 71.5
  (A) vs init L2 88.8 with best-|cos| to own init atoms mean 0.719 and zero
  sign flips (moderate reshaping, not a reshuffle).
- REC pre-training (the round's single extra fit, 1,000 steps, label-free):
  relative reconstruction 0.607 → 0.0377; kappa_REC 12.266.

## Figures

- `figure_stage1_four_arm_effect.png` — four-arm endpoints + contrast forest +
  interaction.
- `figure_dictionary_task_division.png` — health-by-role map (nnz / atoms /
  injection RMS / drift).
- `figure_stage23_replication.png` — the two Stage-2 signals on A vs their
  B replication (cal vs raw) — the round's key figure.

## ERRATA (two)

1. **Sign-convention check bug in the source round** (no number changes): its
   `cal_g_identity_max_abs` (6.60/4.11) used the wrong sign convention in the
   *check*; with the correct convention the identity is exact (≤ 2.4e-7).
2. **Train-incidence-on-held-out-graphs bug in the fulltrain round's valid
   read** (numbers change): see Conclusion 4 and
   `valid_structure_check.{py,json,npz}`. The frozen predictions reproduce
   exactly (max_abs 0.0), so the bug is confirmed in the historical path, not
   in this round's re-implementation. Affects
   `zinc_component_supervision_fulltrain_confirmation_seed0_v1` valid numbers
   and every downstream reuse (cycle-round valid replay **and terminal test
   body inference** via the same loader pattern). This round's own held-out
   machinery uses each held-out graph's own incidence structure.
   **Sequencing disclosure**: the ERRATA check read official-valid once at
   ~03:40, *before* the ~03:53 purchase decision (which the brief allows only
   after a purchase); it scored frozen historical bodies only, was never
   consulted by any gate or selection, and is logged in `heldout_access.json`.
   The Stage-4 rejection rests solely on the frozen split-B gates.

## Uncertainties reduced vs opened

Reduced: (a) the J-gap is noise-scale (two matched runs bracket ±0.005 cal /
±0.012 raw — the realistic effect scale for this interface); (b) Stage-2's
apparent F-family signals are not split-robust, and the only replicated
component (F_M_I's calibrated gain) is a bias-offset artifact; (c) the
historical held-out inference defect is found, quantified and fenced.
Opened: nothing new is licensed for purchase; the luyin19 hypothesis
(separate coding helps) is now *disconfirmed at this scale/interface*, which
is itself the round's deliverable.

Scope guard: single seed, internal-dev folds A/B only, official-valid never
read this round, official-test never touched; no sweeps, no post-hoc
selection, no parametric rescue; all gates as frozen at Stage 0
(`PROTOCOL.md`).
