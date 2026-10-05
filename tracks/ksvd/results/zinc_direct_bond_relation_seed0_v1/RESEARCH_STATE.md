# RESEARCH_STATE — next entry (2026-10-05, direct-bond relation round)

This page organizes the research log; it does not supersede `tracks/ksvd/results/zinc_direct_bond_relation_seed0_v1/*` (see those files for numbers and provenance). All values are from verified artifacts; official-valid/test were not read in this round.

## 0. One-sentence verdict on this round

**Strict status: `NOT_COMPLETED`.** Completed O/T pair: small positive calibrated direction for T (G0 cal +0.000415, overall +0.000543), but 0/3 frozen gate conditions pass; CIs include zero and are not inside ±0.003. Separately, two partial training jobs (55999→epoch 140, 56000→epoch 40) were cancelled mid-training on the mistaken assumption that identical `CUDA_VISIBLE_DEVICES=0` meant the same physical GPU; the later complete jobs were relaunched fresh (not resumed) and began after the minute-90 cutoff. No confirmation compute follows; the dev rows are reused and are not fresh confirmation evidence.

## 1. Where the direct-bond relation round sits

* Source: frozen fresh-fold M (a5400de / prereg 2e4ee3f), internal `g=y−c`, one seed.
* Intervention: T appends primitive 4-D direct-bond beta to the 15-D source relation (path-bond-mean 14:18 excluded); O carries four zero columns. Same init/schedule/RNG, source C6 mask retained.
* Beta verified train-only: 10,000 graphs, 249,279 adjacent pairs, category mass `[0,185060,63548,671]`, three bond classes; `pair_relation[:,19:23]` exactly matches raw edges.
* Completed arms: 15,120/15,120 steps each, seed 0, 240 epochs; 297,883 params each; init hashes identical (`9678be159fdc…`); RNG/schedule/GID streams match source. O new-column grad exactly 0; T nonzero. Raw-soup replay ≤1.91e-6.
* Primary dev (k=0): O cal 0.095018, T cal 0.094603 → O−T +0.000402. Overall dev cal: O 0.096880, T 0.096337 → +0.000543. Raw: G0 −0.000097, overall +0.000162. Source M dev cal G0 0.097383 / overall 0.099304.
* Frozen gate: 0/3 (G0 gain ≥0.003 ✗, overall gain ≥0.003 ✗, G0 CI lower > 0 ✗). Bootstrap witnesses pass; practical equivalence rejected. → `DIRECTIONAL_NOT_CONFIRMED` for the numbers; `NOT_COMPLETED` for protocol.

## 2. Mechanism/interpretation boundary

Do not read the small point gain as mechanism evidence. Beta is a label-free input feature; identical init/schedule/RNG plus the zero-gradient-on-O/one-nonzero-on-T checks show the interface is wired and receives task signal, which is an engineering/learning-path check, not a generalization result. Source M dev `g≈0.10` (G0 ~0.097) remains the reference; the residual bottleneck is not localized here. The conditional beta-to-graph-marginal forward diagnostic was **not** run because T did not pass the gate. No architecture/dictionary/pooling/t hyperparameter/seed/fold/rescue attempt was layered on this contrast.

## 3. Current open items (from this round, not closed)

* **Process compliance (highest priority):** record and prevent mid-round re-launch-after-cutoff and partial-job cancellation decisions. Any future remote run must capture allocation-resolved GPU UUIDs *before* cancellation rationale and enforce resume-from-checkpoint vs fresh-start rules explicitly.
* **Input-information vs interface distinction:** a future design should separate "direct-bond feature available" (this round's input) from "capacity/path to use it" (architecture) to avoid reinterpreting an interface feature as an information bottleneck.
* **Training-randomness uncertainty:** the row bootstrap measures row uncertainty only; any buy should add explicit seed/schedule or regime coverage, not just relabel rows.

## 4. What this round does not revise

It does not revise the source fresh-fold M verdict, nor the local-tuple / dictionary/MLP, pooling-scale-count, or cycle-tail lines recorded earlier. It is a single feature-applicability contrast, not a bottleneck audit, and was treated as such.

## 5. Work rules in force (self-imposed, reinforced by this round)

* Exactly the pre-registered arms; no extra architecture/dictionary/pooling/hyperparameter/seed/fold/rescue or "just one more" run.
* No dev-score access during training; dev analysis post-training only; no selection on dev.
* No new compute after the cutoff; no compute if both jobs cannot be scheduled before the minute-90 stop on distinct allocation-resolved device UUIDs.
* Any future purchase must be a *new* preregistered question with an independent evidence source, not a relabeled confirmation of these reused rows (see `NO_CONFIRMATION_NOTE.md`).
