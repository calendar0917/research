# E2E-DictEnv H1 clarity audit — DECISION

Round `e2e_dictenv_h1_clarity_audit`, commit `72b490a3672af5a6efa514342e11f27295645608`.
Evidence: `REPORT.md`, `summary.json`, `frozen_interventions*.json`,
`adaptation/`, `matched_cpu/` in this directory; durable analysis in
`tracks/ksvd/notes/e2e_dictenv_h1_clarity_audit_analysis.md`.

## Accepted

1. **The Phase A information map is verified**, not inferred: reader input 302 = 97 + 165 + 32 + 8,
   `global_context62` contains a 32-coordinate graph chemistry marginal, `anchor62` contains
   32 coordinates of patch chemistry marginals, and the used `pair_relation` is
   `raw[0:14] + raw[18:19]` (raw 14–17 never reach the model). All provenance checks reproduce
   the tensors bit-for-bit; the audit path is bit-identical to the parent at an identity mask.
2. **Anchor chemistry marginals are load-bearing and not free to delete.** Root identity
   `+0.449`, patch atom mass `+0.442`, patch bond mass
   `+0.193`, size `+0.149`, all confirmed by mean fill
   (no OOD inflation), and still `+0.0151`
   behind a matched continuation after 40 epochs of adaptation. **The H1 main line is not a
   "minimal root-relative information" model; do not describe it that way.**
3. **The relation channel is bundle-read and boundary-dominated**: boundary 3-D block
   `+0.554`, overlap `+0.089`, distance
   `+0.074`, log path count `+0.0044`. "Distance-only"
   is rejected as a simplification (it deletes the dominant block).
4. **Second moments are genuinely used, counts are dormant.** Unary second
   `+0.369` (shuffle) though only `+0.235` (fill) and
   `+1.005` (zero, OOD-inflated); unary/pair counts
   `+0.0002` / `+0.0054`.
5. **Node binding is a bag, edge binding is a pairing.** Content
   `+0.348` vs `α↔atom` correspondence `+0.0122`;
   `role↔bond-type` correspondence `+0.093` (about all of the combined effect).
6. **Candidate `C6` — the minimal cleaned H1 — is accepted as the removal result of this round
   (CPU regime, seed 0 only).**
   Mask: graph atom histogram + graph bond histogram + relation log path count + unary count +
   pair count. Matched CPU retraining: BASE `0.143298` →
   C1 `0.133530` → **C6 `0.128499`**
   (`-0.014799`), same initialisation, same data order.
7. **Probe discipline**: zeroing an input block is an OOD corruption, not clean information
   removal. Three rows are flagged `zero_probe_inflated`, and R1/R5 additionally expose an
   all-or-nothing bundle effect (removing one relation block hurt more than removing all four).
   A "which block is load-bearing" table built only from zero probes would have been wrong here.

## Rejected / not authorised

1. **Replacing the H1 line with `C6` on the strength of this round.** Only one CPU seed was run
   and the untouched architecture is itself
   `0.0197` worse on CPU than the
   historical GPU H1 (0.123549), so this round cannot speak to the
   GPU regime at all. A confirmatory claim
   requires a preregistered GPU round with >=2 seeds on both arms.
2. **Removing the anchor marginals** (`C2`/`C3`) and **removing the second moments** (`C5`):
   rejected for this checkpoint, `+0.0216` /
   `+0.0233` /
   `+0.0389` behind the matched continuation.
3. **"Distance-only relation"** (`C4`) — registered before the data, then rejected by Phase B
   and therefore never trained.
4. **Tuning anything in this round**: no hyper-parameter search, no architecture search,
   no second seed, no official test. All were out of scope.

## Next round (if the line continues)

Preregister one GPU round with two arms (`BASE` GPU, `C6` GPU) x 2 seeds, 320 epochs, identical
initialisation and data order, official-valid only, plus the frozen probe table re-run on each
new checkpoint. Gate: `C6` must not be worse than `BASE` by more than 0.003 on the seed-mean
soup, and the frozen GS1/EG2/PS3/PS6/RS4 deltas must reproduce on the new checkpoints. Do **not**
run that round on CPU: at 12.6 s/epoch/process
the two-arm two-seed design costs ~1.5 h
wall clock at concurrency 3, and the regime offset would make the comparison uninformative
against the historical number.
