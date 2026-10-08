# REPORT — zinc_cssd_pair_joint_residual_v1

**Round question.** Keeping C's shared endpoint projection and the existing pair path
(the working base of the line — a directional single-seed lead, NOT a repeat-confirmed
upgrade), does adding a SHARED, endpoint-swap-symmetric joint residual that reads BOTH
full endpoints and the existing relation improve the complete y_raw and g of the
historical development set?  The purchase reason was an INTERFACE property, never a
proven error mechanism: the per-coordinate pair statistics
`[left+right, |left−right|, left*right*gate, relation]` cannot distinguish the endpoint
pair `{(1,0),(0,1)}` from `{(1,1),(0,0)}` (sum = abs-diff = (1,1), product = (0,0)),
although co-occurrence on one endpoint differs.  C's learning, the unary/other branches
and the relation itself may compensate — nothing was claimed about whole-model
identifiability or error localisation, and no mechanism audit was required first.

**Design (frozen at 518c984, design basis 5d33c9a).**  Two arms × body seed 0, ONE
from-scratch 240-epoch training each (15,120 steps, batch 128, Adam lr 1e-3, coupled
wd 1e-5, clip 5, FP32, no scheduler/AMP/early stopping; estimator = equal-weight FP32
mean of the full member states 236..240; fit = 8001 source-fold rows only during
training; dev = the 1999 historical development rows, touched once at the terminal):

| arm | pair branch | params | total |
|---|---|---|---|
| CTRL_C | exactly the previous round's untrained C factory `proj.build_arm_round("SHARED_MLP", …)` (init hash-anchored bitwise to the previous SHARED_MLP s0 init), no joint module | 325,187 | 325,187 |
| JOINT_RESIDUAL | same factory + `joint_residual` (ψ: no-bias Linear 144→64→48, input `[left, right, relation]`; `delta_ij = ½(ψ[l,r,rel]+ψ[r,l,rel])` via ONE `[2P,144]` call split+averaged; `h_pair = h_old + delta` after h_old, BEFORE the original bucket pooling; last layer an ordinary Linear with nothing after) | +12,288 (+3.78%) | 337,475 |

Private init seed 202610081 under `fork_rng`: W1 = default draw, W2 = EXACTLY zero ⇒
step-0 delta exactly zero, both arms start function-identical (bitwise vs CTRL_C AND vs
the previous round's C factory on CPU), every non-joint init tensor bitwise shared.
W1's first-backward gradient is zero BY CONSTRUCTION (never rescued); W2's gradient is
reachable at step 0 (1.42) and W1 is active with a non-zero delta by step 3 (0.0116).
The old h_old path, pair/relation encoders, distance gate, five-bucket pooling, reader,
C6 mask and COMP supervision are verbatim; no new pool, no pair→node write-back; the
masked forward defaults to `cm.C6_MASK` so the new branch can never be silently
bypassed; `joint_enabled=False` zeroes only delta.  Official valid/test never loaded.

**Pre-freeze verification (all passed, committed revision 518c984).**
source-checks (Slurm 56326, run 20261008-133514-1b8a5b48): stage-0 artifact/fold/basis/Q
hashes, historical schedule bit-identical, CTRL_C init == previous SHARED_MLP s0 init
hash `ac18cb28…`, JOINT non-joint init bitwise == CTRL_C, W2 exactly zero, fork_rng RNG
neutrality.  pretrain-checks (Slurm 56327, run 20261008-133535-63691789): CTRL_C
forward/component/aux parity BIT-EXACT vs the previous C factory; step-0 agreement
bitwise; endpoint-swap/pair-order/node-relabel/batch invariances (incl. delta symmetry);
hook-verified mask wiring (ψ reads the masked views and the SAME masked relation tensor
the pair input reads; `pair_zero_blocks` not bypassed; with `joint_enabled=False` every
probe-mask response is bitwise identical to CTRL_C); 3-step gradient smoke; save/reload
replay + wrong-factory strict load blocked; switch equivalence on a temporary
non-zero-W2 copy (off == CTRL_C bitwise, on differs, upstream aux untouched); the 2-D
collision demo.  GPU smoke (Slurm 56328, 50.8 s): both arms through the full train path.

## Execution (all through the registered runner, clean committed revisions)

* Formal trainings, GPU res2-cu124 c05, A100-PCIE-40GB, torch 2.5.1+cu124, driver
  525.85.12 — CTRL_C 20261008-134134-1359c532 (Slurm 56329, 1249 s, peak 383 MB),
  JOINT_RESIDUAL 20261008-134157-6a41f09b (Slurm 56330, 1275 s, peak 479 MB).
  Both completed 15,120/15,120 steps; roster 11/11 terminal checks OK (inits
  hash/bitwise anchored; schedule, frozen basis, Q, members, soups hash-verified).
* Terminal eval: ONE-SHOT.  First submission 20261008-142900-d84440fe (Slurm 56331)
  FAILED at the last check; rerun 20261008-143924-55f466e0 (Slurm 56332, 50.5 s,
  commit 7264ff2) completed; `terminal_eval.json` written once and never overwritten.

**Disclosed engineering incidents (evaluation layer only; no training semantics touched):**
1. The first terminal submission required the CTRL_C switch no-op delta to be BITWISE
   zero; on GPU two forwards of the IDENTICAL configuration already differ by
   cross-kernel float noise (measured d = 4.77e-07 on 64 dev rows, equal to the measured
   same-config repeat-forward noise and three orders below the round's noise marker
   1e-4).  The forward path is identical ops either way (the module is absent on
   CTRL_C); the bitwise no-op remains asserted on CPU in the pretrain checks.  Fixed in
   7264ff2 to compare against the noise marker with the repeat-forward noise reported
   alongside; rerun within budget.  `terminal_eval.json` was never written by the failed
   run, so no result was mixed or overwritten; the npz exports of the failed run are
   deterministic recomputations of the same soups.
2. `rr pull` of the declared result paths lands in `.rr/pulled/<run_id>/result/…`
   rather than directly inside the round's results directory; artifacts were copied from
   there into their exact declared paths (same retrieval-path class as the previous
   round's incident 3) and the two training/terminal run-store entries were retrieved
   with `rr pull --path` and promoted locally.

## Results (dev = 1999 historical rows; delta = JOINT − CTRL_C, negative = improvement)

| arm | dev y_raw | dev g_raw | fit y_raw | fit g_raw | gap dev−fit (g) | b_y | seconds | peak GPU |
|---|---|---|---|---|---|---|---|---|
| CTRL_C | **0.11804** | **0.09001** | **0.04003** | **0.03187** | 0.05813 | −0.00446 | 1249 | 383 MB |
| JOINT_RESIDUAL | 0.11843 | 0.09148 | 0.04314 | 0.03506 | 0.05642 | +0.01337 | 1275 | 479 MB |

* Paired deltas: dev y **+0.00039** (+0.33 % relative, CI95 [−0.0030, +0.0040]), dev g
  **+0.00147** (+1.64 %, CI95 [−0.0020, +0.0050]) — BOTH point estimates worse; both
  CIs cross zero (canonical-SMILES-group-paired bootstrap, 2000 draws, seed 20261008,
  shared picks across arms and metrics; molecule resampling only).
* Fit deltas: y +0.00312, g +0.00319 — JOINT fits WORSE.  The fit-dev gap(g) shrinks by
  0.0017 (and gap(y) by 0.0027) ONLY because fit degraded by more than dev; the dev
  error got worse on both metrics.  The "gap improvement" is a fit-degradation artifact,
  not a generalisation gain.  Fit numbers are the fixed soups' fit MAE (never a training
  optimum); the end-of-training online train losses (curve.json train_L_g 0.0703 vs
  0.0700) are a different quantity and were never ranked against them.
* k-group cut (≤23 / 24..27 / ≥28 nodes, analysis only): the harm concentrates in the
  dominant k=0 group (in-group g 0.0894 → 0.0908); tiny tail groups (k≤−2, n=9) move
  within their huge variance.  No per-molecule audit was purchased.
* y_cal (descriptive only, never a positive signal): 0.11803 vs 0.11716 — JOINT's larger
  fit-median bias (+0.01337) happens to cancel more dev bias; this does not change the
  reading (the primary y_raw got worse).

**Mechanism / dependence evidence (frozen interventions only):**
* Dictionary readout intact in BOTH soups: alpha-mean |dPred| p95 4.05 (CTRL_C) / 2.95
  (JOINT) >> marker 1e-4, response fraction 1.000.  JOINT did not detach from the
  sparse code (ΔMAE_y when replaced: +2.43 / +1.41 — both massively degraded).
* The joint branch is REAL and LOADED on the JOINT soup: disabling it (delta := 0,
  nothing else touched) moves 99.8 % of dev predictions beyond the noise marker
  (|dPred| p95 0.068) and WORSENS this checkpoint by ΔMAE +0.0055 (y) / +0.0058 (g);
  the trained delta is substantial relative to the old pair state (|δ|/|h_old| median
  0.34 over 531,246 dev pair rows; W2 Frobenius norm 0.93 ≠ 0).  So the negative is NOT
  "the branch never mattered": the model uses it, and it helps the checkpoint internally
  — it just fails to beat the same-round control on dev.  (The |δ|/|h_old| MEAN is
  destroyed by pairs with near-zero |h_old| (ε = 1e-12 in the ratio); the median is the
  robust statistic and is the one read.)
* CTRL_C switch no-op verified within float noise (4.77e-07 == the same-config repeat
  noise; bitwise on CPU).

**Cost (measured, never inferred from parameter counts).**  Aggregate GPU ≈ 44.6 min
(2673 s: smoke 51 s + two formal runs 2524 s + failed terminal 48 s + terminal rerun
51 s) against the frozen 2 aggregate-hour budget; the two arms ran serially on c05
(c06 was occupied; identical hardware/software regime per protocol).  +12,288 parameters
(+3.78 %) cost ≈ +2 % wall time and +25 % peak memory, NOT a proportional-time claim.

## Frozen exploratory reading

Branch **`joint-residual-configuration-not-retained`** (protocol section 8, case 3):
JOINT does not improve both dev metrics — both point estimates are worse — so THIS
configuration is not retained and its additional budget is closed.  No width/depth/
residual-scale/init sweeps, no rescue, no third arm; and no verdict on relation fusion
or on joint endpoint reads in general is licensed by one seed of one configuration.

**Direct answers to the round's questions:**
1. **How much did JOINT improve the complete y/g vs the same-round C, at what CI and
   cost?**  It did not: dev y +0.00039 [−0.0030, +0.0040], dev g +0.00147
   [−0.0020, +0.0050] (worse at the point estimate, CIs cross zero).  Cost: +12,288
   parameters (+3.78 %), ≈ +2 % wall time, 1275 s vs 1249 s.
2. **How did fit/dev move; is the gap reduction just fit getting worse?**  Fit worsened
   (+0.0031 y / +0.0032 g) AND dev worsened (+0.0004 y / +0.0015 g); the gap(g)
   reduction of 0.0017 is entirely the fit degradation — not a generalisation gain.
3. **Is the dictionary still read; does the new branch really carry predictions; what
   is the MAE direction of disabling it?**  Yes (alpha p95 2.95 >> 1e-4, response 1.000);
   yes (toggle response 99.8 %, |δ|/|h_old| median 0.34); disabling WORSENS the JOINT
   checkpoint's own dev MAE (+0.0055 y / +0.0058 g) — the branch is net helpful inside
   its own model yet the whole model is still worse than the control without it.
4. **Performance lead, mixed result, or configuration failure?**  Under the frozen
   rubric: a configuration failure for THIS candidate — not retained, budget closed.
   It is not evidence against the C base (this round's CTRL_C reproduced the previous
   round's C within +0.0014/+0.0007 dev, inside documented GPU trajectory variance:
   0.11804/0.09001 vs the 0.11659/0.08935 anchor).
5. **Which competing explanations remain; what was NOT tested?**  For the residual
   question of the line, everything stays open: no parameter-matched arm existed, and
   even a win would not have separated added capacity, implicit regularisation, training
   trajectory, endpoint-own nonlinear corrections or relation-own corrections.  What a
   negative here does NOT licence: any claim that shared symmetric joint endpoint reads
   cannot help (one seed, one width, one insertion point after h_old, one relation
   encoding), any claim about the bucket-routed R family (separately closed), and any
   reopening of RAW/DICT, cross-domain reuse or interpretability questions.  Un-tested
   here: second seeds, other insertion points (into the pair encoder/gate), matched
   capacity controls, mechanism localisation audits.

**Recommendation (for the researcher, no automatic purchase).**  The C base remains the
live lead of the line exactly as it was (single-seed directional, unconfirmed).  The
joint-residual add-on as specified here is closed.  If the interface-collision idea is
pursued again, it needs a different pre-registered design (e.g. a parameter-matched
redistribution of the +12k parameters, or conditioning at a different interface) — that
is a new purchase decision, not this round's to make.

## Artifacts

* Protocol: `tracks/ksvd/protocols/zinc-cssd-pair-joint-residual-v1.yaml` (frozen 518c984)
* Code: `tracks/ksvd/experiments/luyin16/zinc_cssd_pair_joint_residual_v1.py`,
  `tracks/ksvd/src/ksvd_research/runners/zinc_cssd_pair_joint_residual_v1.py`,
  `tracks/ksvd/configs/luyin16/zinc_cssd_pair_joint_residual_v1.yaml`
* Tests: `tracks/ksvd/tests/test_zinc_cssd_pair_joint_residual_v1.py` (14 passed)
* Results: `source_manifest.json`, `pretrain_checks.json`, `smoke.json`,
  `terminal_eval.json`, `runs/{CTRL_C,JOINT_RESIDUAL}_s0/` (init / epoch120/240 /
  members 236..240 / soup / last / resume states, schedule, curve, probes, manifests,
  fit+dev per-row predictions)
* Run records: 20261008-133514-1b8a5b48 (source), 20261008-133535-63691789 (pretrain),
  20261008-133746-d7cb9e4d (smoke), 20261008-134134-1359c532 (CTRL_C train),
  20261008-134157-6a41f09b (JOINT train), 20261008-143924-55f466e0 (terminal);
  the failed first terminal 20261008-142900-d84440fe is recorded in this report and in
  the fix commit 7264ff2, not promoted (status failed)
* Reproduce: commit 7264ff2 (fix) on top of 518c984 (frozen design); `rr deploy res-2 &&
  rr run res-2 <exp> -- uv run research run zinc_cssd_pair_joint_residual_v1 --set
  model.stage=train --set model.arm=<ARM> --set model.seed=0 --set runtime.device=cuda:0
  --mode scratch` (ARM in {CTRL_C, JOINT_RESIDUAL}); terminal via
  `--set model.stage=terminal-eval`; official valid/test never loaded.
