# Representation candidates for the next round (max two)

This round is a *pilot*: it changes splits, objectives and selection, and
deliberately **does not** change the representation. What follows is therefore a
proposal list with explicit tiers, expected effect sizes derived from P0, and
falsification criteria — not a result.

The tier vocabulary is fixed:

* **T1 Representation gap** — information the task needs that the frozen input
  *provably cannot express* (a collision).
* **T2 Inductive-bias augmentation** — information or relations that exist in the
  molecule but are not given to the model; added as structural features/relations
  so the encoder does not have to rediscover them.
* **T3 External physicochemical prior** — a small, deterministic, label-free set of
  descriptors computed outside the model and appended as inputs.

Labelling rule for next round: any variant that adds chemistry-derived inputs is
**`CHEM_AUGMENTED`**, never "LCRC-A". The frozen `lipo_v1` LCRC-A line stays
untouched so that every comparison keeps a clean control.

---

## Candidate 1 — T3: a 10-dimensional physicochemical prior block (global)

**What.** Append one global block per molecule, all computed by RDKit from the
parsed mol, all label-free and deterministic:

```
TPSA, HBD, HBA, log1p(MW), Crippen logP, rotatable_bonds,
formal_charge_count_signed, aromatic_ring_count, max_ring_size, fraction_sp3
```

Standardised with the *Train-only* mean/scale of that run (the existing
`fit_norm` path, `valid_or_test_used: false`), concatenated to `global` (42 → 52).

**Why this and not something else.** It is the only place where "high task-need"
and "provably absent" coincide more than once
(`feature_gap_matrix.csv`): TPSA is HIGH for ESOL/Lipophilicity/FreeSolv/BBBP,
HBD/HBA HIGH for the same four, MW HIGH for Lipophilicity/BACE/BBBP/HIV, logP HIGH
for ESOL/Lipophilicity/BBBP. None of the four is in the spec. MW is a linear
functional of composition the model *could* learn but does not get for free.

**Evidence strength: MEDIUM (not higher, and the Sprint-01/02 scoreboard is the
reason).**

* *Supporting*: the frozen classical baseline — whose inputs are exactly ECFP4
  plus 217 RDKit descriptors, i.e. this block and more — wins on 3 of 7 datasets
  (BBBP 0.7208 vs 0.7110, Tox21 0.8521 vs 0.8353, HIV 0.7624 vs 0.7524) and loses
  on 4 (ESOL, Lipophilicity, FreeSolv, BACE). So the descriptor block is *not*
  uniformly better; its advantage is concentrated, and small (0.010–0.017 AUC) on
  the three binary tasks.
* *Against*: on the Lipophilicity repeats, where the descriptor prior should pay
  most, the comparison is split — XGB better on Validation by 0.047 (repeat 123)
  and 0.002 (repeat 124, a tie), while on repeat 122's Test LCRC-A won by 0.075
  despite a 0.058 Validation deficit. That contradiction says the *measurement*
  is currently too noisy to attribute this gap to the representation.
* *Redundancy*: much of the descriptor set is *not* orthogonal to the 42 global
  fields (size, ring counts, aromatic fraction, H-count aggregates).

**Bound on the effect (from P1/P0, not from hope).** The LCRC-A Valid deficit to
the baseline is 0.047 (repeat 123), 0.002 (repeat 124) and 0.058 (repeat 122); the
Test deficit on the three datasets the baseline wins is 0.010–0.017 AUC. So the
*maximum* plausible gain is bounded by those numbers — and, decisively, the
split-to-split spread of the neural arm itself is ±0.054 RMSE (sd over repeats
122/123/124), which is larger than the gap being chased. A candidate that "wins"
by less than that is indistinguishable from split noise.

**Cost.** ~2 GPU-hours for a Lipophilicity + ESOL + BBBP trio at 360 epochs
(dropout/width change forces a fresh `ρ` init; the audit's width guard must be
re-run, see the PHYSCHEM6 lesson).

**Test plan (falsifiable).** Run on Lipophilicity repeats 122/123/124 and ESOL,
same recipe/seed, 2 seeds. Predict: Valid RMSE improves on ≥2 of 3 Lipophilicity
repeats by ≥0.02 *and* the improvement survives the 123/124 holdout direction.
Kill criterion: if the mean improvement over three repeats is < one fold-to-fold
spread (≥0.05 RMSE), the block is redundant with the global fields and should be
dropped rather than tuned.

**Most likely false positive.** Descriptor leakage of the target itself: logD is
strongly related to Crippen logP, so a Lipophilicity-only gain is *not* evidence of
a general prior. The ESOL/FreeSolv arms are the honest test.

---

## Candidate 2 — T2: tetrahedral chirality as a local relation

**What.** Two changes, both in the adapter/spec, neither in the model:

1. a per-atom 3-dim block appended to `atom_feat`: `chiral_tag ∈ {unspecified, R,
   S}` after CIP assignment on the parsed mol (UNK for unassigned);
2. a per-bond flag (`bond_feat`, +1 dim) marking a bond whose one heavy-atom
   substituent is a chiral centre, so that the *local environment* of the
   stereocentre, not just the atom, carries the information.

**Why.** P0 *proves* the collision: two BACE enantiomers (CIP `R` / `S`) produce
byte-identical raw artifacts and identical input tensors, and the frozen
`BACE_s0` checkpoint returns identical output (`8.033543586730957`) for both. This
is the only T1-grade gap found — it is implemented here as a T2 augmentation
because a chirality *parity* is an inductive bias about local geometry, not an
external descriptor.

**Evidence strength: LOW-MEDIUM** — strong on existence, weak on consequence:

* *Existence*: proven constructively and confirmed on real data — 114/115
  Lipophilicity, 30/32 BACE, 21/23 BBBP, 102/139 Tox21 constitutional-twin pairs
  have identical model input.
* *Consequence is bounded*: the Lipophilicity Train RMSE floor from colliding
  labels is **0.0279** against a Train RMSE of 0.28 and a Test RMSE of 0.57–0.74,
  i.e. at most ~5 % of the error. On BACE only 60 of 1210 Train rows sit in a
  colliding pair (5 %), where 57 % of pairs carry different labels — enough to
  matter for a handful of points, not enough to explain a 0.83 AUC.
* *Counter-evidence*: LCRC-A already **beats** the descriptor baseline on BACE
  (0.8254 vs 0.7969 Test), the dataset with the highest chirality need. If
  chirality were the binding constraint, that would be surprising.
* Note that HIV and ESOL contain **zero** stereocentres in this build: two of the
  seven datasets cannot benefit at all.

**Cost.** ~1.5 GPU-hours (BACE + Lipophilicity 122/123 + Tox21), plus an adapter
change that invalidates the existing caches (they must be rebuilt and their
fingerprints re-frozen).

**Test plan (falsifiable).** BACE is the pre-registered target. Predict: Test AUC
improves by ≥0.01 on BACE with the *same* split/seed/recipe, and the improvement
is confined to the 383 chiral molecules (a subgroup analysis that is
pre-registered, not searched). Kill criterion: no improvement on BACE ⇒ the
chirality gap is real but not load-bearing, and it should stay documented rather
than implemented.

**Most likely false positive.** A gain that comes from extra capacity (3 extra
atom channels) rather than from chirality: the control must be an *equally wide*
random/uninformative block, otherwise the comparison confounds width with
information.

---

## What is explicitly NOT proposed

* **T1 as a standalone item** — the surviving T1 gap is chirality, which is
  implemented as Candidate 2. No other T1 gap exists: `atom_type`/`bond_type` are
  redundant documentation tensors, S1/S2 are joint codes with no dead field, and
  the vocabulary is far from saturated (23–59 of 400 path codes are ever used).
* **Anything to "fix" 3D conformation, tautomers or binding pose** — these are
  `NOT_DETERMINABLE` from 2D constitution. Proposing them as representation work
  would be dishonest; they need new data (conformers, pH, assay context) or a
  different task framing.
* **S2 additive-context widening** — the pairing test shows the middle↔end
  relation is only expressible through the rank-32 multiplicative term, so
  widening it is a plausible *architectural* experiment, but it is Tier-0
  (capacity), not a chemistry gap, and P0 found no evidence that the rank is
  binding (path vocabulary usage is 6–15 %).
* **More seeds / longer training / bigger model.** Those are robustness levers,
  not representation levers, and this round already shows (P1, P2) that they are
  not where the variance lives.

## Ordering

Run **Candidate 1 first**. It is cheaper per unit of expected effect, it cannot
produce a wrong-chemistry artifact (all four descriptors are standard), and its
failure mode (redundancy) is diagnosed by the same three-repeat harness this round
just built. Candidate 2 should follow only if BACE remains the only dataset where
LCRC-A beats the classical baseline *after* Candidate 1 — otherwise the honest
conclusion is that chirality is a documented, bounded limitation.
