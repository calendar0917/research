# DECISION — `lcrc_bond_anchored_composition_route_8h_v1`

**Verdict: `NO_COMPOSITION_ROUTE_ADVANCE`**

Decided at `2026-11-18T02:58+08:00`, from the 16 formal units of
`analysis/summary.json` and the three Stage-C measurements, all on their
pre-registered scales. Nothing in this decision depends on Development Test.

---

## The verdict in one paragraph

A real-physical-bond-anchored two-path composition (a three-bond path event, its
anchor bond, the two *real* adjacent frozen S2 occurrences, 17 392 parameters = 7.6 %
of C1, exactly zero at initialisation, direction-invariant to the bit) **adds no
information that survives a matched equal-capacity control, does not improve Lipo
prediction, does not move towards GINE, and does not recover externally referenced
structure.** On Lipo Valid the real-pairing arm loses to its own control at two of
three end-to-end seeds (mean `gain_null` = **−0.0106**) and ties it at the one seed
where both arms are alive (+0.0051, CI [−0.030, +0.043]); with the teacher frozen it
is indistinguishable from the control (−0.0007, P = 0.28). The best Lipo Valid number
of the round belongs to the *control* (0.6944) and still leaves 86 % of the
C1 → GINE gap (0.0783 of 0.0908). The explanation side is not merely null: gating the
**anchor bond** moves the prediction no more than gating the peripheral bonds of the
same event (anchor 1.21 / 1.20 / 1.12 / 1.07 vs first hop 1.20 / 1.15 / 1.06 / 0.91
and last hop 1.11 / 1.13 / 1.04 / 0.94 across 2 / 5 / 10 / 25 % gating), and the
WC_LOGP atom-additive oracle `capture` of the
top-rated event is **at or below chance** (−0.0047, CI [−0.0089, −0.0003],
P(> 0) = 0.018, against a 4-atom random baseline of 0.1563). In the end-to-end model
the molecules that receive the largest composition contribution are the molecules
whose error **grows** (Spearman −0.559, p = 7e-36). The composition is
construct-accountable — `delta_comp = Σ_P c_P` holds exactly, event deletion is
monotone in `|c_P|` by 22–53× — and nothing more.

## Why this is not `INCONCLUSIVE_OR_BLOCKED`

Blocking and power were healthy: 0 failed gates out of the 14 pre-registered ones (plus
the numerical operator-separation gate), teachers replayed to
1.4e-6, 16/16 formal units completed, 21/21 `rr` runs accounted for, 2.75 GPU·h of 14,
2.4 h of 8. The null is **measured**, not unmeasured:

* the family of explanations that could rescue a small effect is closed off by an
  interval, not by absence of a run: the only *clean* end-to-end pairing has a CI of
  width 0.073 around +0.005, and the frozen-institution pair has a CI of width 0.005
  around −0.0007 (i.e. tight enough to exclude the pre-registered +0.010);
* the structural claim is rejected on measurements with intervals that exclude zero
  *on the negative side* (oracle enrichment) or that are exact identities
  (`|c_P − c_P(reversed)| = 0`, `pred(branch=False) = pred_base`, both exactly 0.0);
* the anti-correlation on the end-to-end model (p = 7e-36) is a positive statement
  about where the contribution goes, not an absence of signal.

What is genuinely unresolved is *not* the route but one instrument: the matched
control itself (see below). That is logged as a defect of the control, not as
ambiguity about the route.

## Why not `PERFORMANCE_ONLY` or `STRUCTURAL_CALIBRATION_ONLY`

* `PERFORMANCE_ONLY` would require a real gain somewhere with no structural insight.
  There is no real gain: the only positive Valid number in the round (+0.0013 on
  WC_LOGP) is 1 % of a residual on a task whose teacher already reaches R² = 0.9941,
  and the only consistent positive sign anywhere is the ESOL **Development Test**
  column (+0.0381 over three seeds), which is read-once material.
* `STRUCTURAL_CALIBRATION_ONLY` would require the external localisation to be
  positive while performance is flat. It is negative on the primary task and at
  chance on the transfer row.
* `JOINT_ROUTE_PROMISING` is excluded by every clean measurement above.

## The one instrument that failed, and what it costs

The `MARGINAL` control is **not trustworthy at 2 of 3 end-to-end seeds**: its
zero-initialised readout over a molecule-mean second-path input stayed at
`mean|Δ| = 0.0001 / 0.0000` for all 360 epochs (curve in
`results/e2e/lipo/MARGINAL_s*/curve.csv`). At those seeds `gain_null` measures
*trainability*, not structure, and JOINT's apparent loss is inflated. At seed 2 —
where the control trains (`|Δ| = 0.141`) — the pairing is clean and the difference
collapses to +0.0051 ± 0.037. **The conclusion does not soften:** the direction is
still not positive, and the structural measurements (anchor-bond ratio, oracle
capture, contribution-vs-error correlation) are computed on the JOINT model alone and
are untouched by the control's failure. But any successor round on this route must fix
the control *before* its `gain_null` is read as a structural contrast.

Second instrument that failed: **WC_LOGP cannot discriminate at this scale.** Its
teacher has R² = 0.9941, so both arms sit within 1 % of the residual and the branch is
two orders of magnitude smaller than on Lipo. Using it as an independent *performance*
task is uninformative by construction; it remains valid only as the atom-additive
oracle channel (C5), which is where this round used it to a negative result.

## What the round establishes for the next one

1. **Do not re-run this architecture at more seeds.** The pre-registered effect
   (+0.010 vs the control) is excluded at the frozen institution with a CI of width
   0.005, and the end-to-end mean has the wrong sign. More seeds buy nothing.
2. **A three-bond path with a shared anchor bond does not concentrate evidence.** The
   anchor is statistically indistinguishable from the two peripheral hops of the same
   event under the validated `CONDITIONAL_S1_S2_EVIDENCE_MASK`. Any future
   bond-anchored object must be justified by something other than "the bond is
   shared".
3. **The C1 → GINE gap is a tail trade, and that is where the next mechanism must
   aim.** GINE is *worse* than C1 on the 60 % of molecules C1 already predicts well
   (deciles 0–5, P(GINE better) ≤ 0.10) and better by +0.539 on the worst decile;
   the same shape reproduces on Development Test (−0.055 … −0.360 on the body,
   +0.339 on decile 9). A route that improves mean RMSE without repairing the body
   cannot close this gap. The paths3-quartile concentration seen on Valid
   (+0.172) did **not** reproduce on Test (−0.014) and is withdrawn.
4. **Capacity, not connectivity, is what moved the number.** Every *live* branch
   reached Train RMSE 0.17–0.23 — below the best Valid of any historical model — and
   paid for it on Valid, in both arms. The next test of this family has to make
   Train-side expressiveness a controlled variable, not a by-product.

## Artefacts

`REPORT.md` (full report) · `PROTOCOL.md` + `configs/frozen.yaml` +
`analysis/protocol_freeze.json` (pre-registration, `protocol_hash a722872b…`) ·
`analysis/preflight.json` (15/15 gates) · `analysis/path3_support.json` ·
`analysis/paired_comparisons.json` + `analysis/tables/{units,paired_valid,paired_test}.md` ·
`analysis/branch_utilization.json` · `analysis/explanation_eval_{frozen_joint,e2e_joint_s0}.json` ·
`analysis/oracle_localisation.json` · `analysis/stage_decisions.json` ·
`analysis/tables/stage_d.md` · `analysis/compute_ledger.json` · `analysis/rr_runs.json` ·
`analysis/engineering_fixes.json` · `analysis/protocol_additions.json` ·
`results/<institution>/<task>/<variant>_s<seed>/` (checkpoints, epoch-0 predictions,
per-molecule `base`/`delta`/`pred`, per-event `c_P` ledger, curves, status).
