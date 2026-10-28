# DECISION TREE — non-local architecture tree v1

Updated: 2026-10-29 02:30 (+08:00). GPU used so far: 2.08 GPU·h (SCE 0.5947 + GEF 0.6738 + JSSP pretrain 0.4518 + JSSP downstream 0.3562 + timing 0.002).
Wall elapsed: ~3.5 h of the 7–8 h target.

## Observation 1 (Phase 0, CPU)

- SCE: 260,336 params (+22,817); GEF: 250,524 params (+13,005). Trunk initialisation is bit-identical
  to historical LCRC-A seed 0 (`11c698ae…`), verified per tensor.
- All 9 CPU gates pass: bridge enumeration vs networkx (300 molecules, 0 mismatch), endpoint-exchange
  and renumbering invariance ≤2.4e-7, bond-direction invariance 1.2e-7, batch split 1.2e-7,
  save/reload bit-identical, GEF field matches an independent float64 brute-force solve
  (K error 8.5e-8, c error 7.2e-8, inverse residual 2e-7, K off-diagonal ≥ 0.33) and is exactly zero
  for a uniform source; SCE term is exactly zero with no bridge.
- Structural witness passes: on a 4-path with the same local-state multiset but a different
  correspondence, both SCE and GEF change their output.
- Bridge coverage on official valid: 1000/1000 molecules have ≥1 bridge (mean 9.77, max 18), including
  all 251 r2-unseen molecules. So SCE is not limited by missing bridges.
- GEF λ = 0.1 fixed before training by a label-free reach rule (mean reach 3.02 graph steps, p90 5.79,
  mass beyond 3 hops 49%, max condition number 62.5); no valid MAE was used.
- Label-free redundancy at init: existing pooled features linearly reconstruct the SCE term at
  R²=0.738 (permutation control −0.31) and GEF term at R²=0.466 (−0.22). Not a duplicate of global
  statistics at init, but SCE's 74% is a warning to re-probe after training.

## Observation 2 (Phase 1, seed 0, matched epochs)

Both arms were launched in parallel at 00:26 on `res-gpu0`/`res-gpu1` with the exact R0 recipe
(600 epochs, 47,400 updates) and a 600-epoch scheduler.

| epoch | A valid | SCE valid | GEF valid | A best-so-far | SCE bsf | GEF bsf |
|---:|---:|---:|---:|---:|---:|---:|
| 150 | 0.111352 | — | — | 0.111352 | — | — |
| 200 | 0.090508 | 0.0920 (193) | — | 0.090508 | 0.08946 (163) | — |
| 250 | 0.085453 | — | 0.08995 | 0.084089 | 0.08396 (288) | 0.08675 |
| 300 | 0.080345 | 0.09110 | 0.08470 | 0.080345 | 0.08396 | 0.08470 |
| 400 | 0.078219 | 0.08340 | 0.08306 | 0.077575 | 0.08267 (363) | 0.08236 (421) |
| 500 | 0.077244 | 0.08317 | — | 0.076899 | 0.08236 (434) | 0.08236 |

Current reading: both arms train normally and fit the train set at least as well as A
(online train MAE at epoch ~450: SCE 0.004, GEF 0.008; A at 450 ~0.007), but their best valid
plateaus at ~0.0824 vs A 0.0766 (+0.0058). The two very different mechanisms converge to the same
plateau, which points to a shared non-local/extra-capacity penalty rather than one broken operator.
No evidence of a faster late slope. Not yet ≥0.010 worse, so the run was allowed to finish.

## Observation 3 (Phase 1 final, matched valid-1000)

Both runs completed normally (exit 0), fully fit the official train set, and were scored against the
frozen A_s0 best checkpoint on the same valid rows (`analysis/stage1.{json,md}`).

| arm | best valid | Δ vs A | Δ drop-0172 | Δ drop-fixed-10 | ordinary Δ | unseen-ord Δ | seen-ord Δ | tail Δ | wins | GPU·h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCE-Real-s0 | 0.082364 @434 | +0.005782 | +0.002115 | +0.003022 | +0.000459 | +0.002798 | −0.000282 | +0.082353 | 0.512 | 0.5947 |
| GEF-Real-s0 | 0.082361 @421 | +0.005780 | +0.004057 | +0.005084 | +0.004359 | +0.011721 | +0.002025 | +0.026218 | 0.499 | 0.6738 |

Both mechanisms converge to the **same** plateau, within 3e-6 of each other, despite different
operators. Neither has a broad improvement: SCE's median row is slightly better (−0.00083) but its
mean is worse and the unseen-ordinary group is worse; removing valid:0172 does not change the sign.
GEF is worse in every group. SCE's ordinary mean is +0.00046 and its unseen-ordinary +0.0028, so the
pre-registered AMBER condition ("broad ordinary/unseen improvement") is not met; the full gap is not
carried by one tail row (drop-0172 still +0.0021). Both are **RED**.

## Observation 4 (mechanism reading)

The identical +0.00578 plateau is a shared penalty of adding a late non-local scalar channel to the
600-epoch LCRC-A recipe, not a single broken operator. Both terms stayed active and both runs reach the same terminal train fit
(final ≈0.001; A 0.00131), but at their best-valid checkpoints their train MAE is 0.0094/0.0094
versus A's 0.00527: the selected solutions are not better-fitting, they generalise worse. The
failure is not optimisation collapse. Extra capacity around a sharp local optimum costs more than the
new information returns. No mechanism control is worth buying without a performance sign; SCE/GEF
are closed and will not be tuned.

## Phase 3 — JSSP opened (pre-registered H3)

Design frozen before GPU: official train 10k structures only, no labels; same backbone and init as
the historical SSL (verified encoder init hash `91af980a…`), same data order (`seed+202`), same
7,560 updates × batch 128, Adam 1e-3 / wd 1e-5 / clip 5, last-step encoder state. Only the mask
changes: joint contiguous 2-bond spans (atoms + physical bonds), all derived features rebuilt from
the masked raw graph. Fixed span budget `max(1, round(0.05 n_atoms))` gives 14.0% atoms (matched to
the old 15.1%) and 8.8% bonds (below the old 15.0%, because a 2-bond span has 3 atoms : 2 bonds);
this budget difference is recorded, not tuned. CPU gates pass: span enumeration vs brute force
(100/100), span closure, hidden-truth swap leak-free (100 atoms + 100 bonds), coverage, loss
decreases, downstream base path is bit-identical to A_s0 (`11c698ae…`, 237,519 params).

Downstream: identical R0 recipe, 600 epochs, seed 0, `--head base --ssl <JSSP ckpt>`; comparator is
the frozen A_s0 (the exact historical `old SSL -> A` run). Decision rule for this phase: buy seed 1
only if JSSP→A improves full MAE by ≥0.003 with a paired CI excluding 0 and no reversal in
ordinary/unseen; a smaller but broad improvement buys at most one seed-1 confirmation; otherwise
close H3 and stop with the negative result. No λ/width/ratio/epoch sweep.

## Observation 5 (Phase 3, JSSP -> A, seed 0)

Pretraining matched the historical SSL exactly in initialisation (`91af980a…`), visible rows (10k),
data order (seed+202), updates (7,560), batch, optimiser and checkpoint rule; only the mask changed.
It used 14.2% atoms and 8.9% bonds (old: 15.0%/14.9%), 0.452 GPU·h. All CPU gates passed.

Downstream A (600 epochs, 0.356 GPU·h) finished at best valid **0.083658 @452**, final 0.084057,
T100 0.084023 — **+0.007077** vs A_s0, worse in every group (ordinary +0.001970, unseen-ordinary
+0.004565, seen-ordinary +0.001148, tail +0.080536, drop-0172 +0.005760). Paired CI95
[+0.000403,+0.014810] excludes zero on the worse side; win rate 0.504. RED, no seed 1.

The joint-span arm is therefore not a candidate. Caveat: JSSP also reduced bond-mask coverage from
15% to 8.9% (a 2-bond span has 3 atoms : 2 bonds while the atom budget was held at ~15%); the result
closes the frozen JSSP recipe, not a clean causal claim that "jointness alone" is harmful. No ratio
control was bought because the arm showed no performance sign.

## Final verdict (all three pre-registered hypotheses)

| hypothesis | arm | valid best | Δ vs A | verdict |
|---|---|---:|---:|---|
| H1 SCE | separator energy, real sides | 0.082364 | +0.005782 | **RED** |
| H2 GEF | one global equilibrium solve | 0.082361 | +0.005780 | **RED** |
| H3 JSSP | joint 2-bond span pretraining | 0.083658 | +0.007077 | **RED** |

No arm met GREEN or AMBER; no mechanism control, seed 1 or official-test read is purchased. The
three unrelated interventions all land 0.0058–0.0071 above A_s0, and all reach the same terminal
train fit (≈0.001; A 0.00131). SCE/GEF terms remain active and non-redundant after training (linear
R² from existing pooled features 0.29 / 0.44; term std 0.061 / 0.107), so the failure is not an
inactive or collapsed branch: the new variation does not generalise. **STOP**; see `REPORT.md` and `NEXT_DECISION.md`.

Budget: 2.08 GPU·h used of soft 14 / hard 16; ~11.9 GPU·h unspent. Wall ~3.5 h. No test access.
