# E2E-DictEnv-P2-ABS — analysis

Round **E2E-DictEnv-P2-ABS** · protocol `e2e_dictenv_p2_abs` · study
`zinc-context-gap` (Workstream Z) · formal-run commit
`122db5dcd45e28f5a6dc3b6314a18e1ea328deae`.
Pre-registration: [`e2e_dictenv_p2_abs_preregistration.md`](e2e_dictenv_p2_abs_preregistration.md).
Implementation: [`e2e_dictenv_p2_abs_implementation.md`](e2e_dictenv_p2_abs_implementation.md).

**No official ZINC test was loaded in this round.** All decisions are frozen on
official-valid Top-5 soup MAE.

## 1. Question

Holding the clean primitive-only dictionary core fixed, what is the best
absolute official-valid MAE it can reach, and which pre-registered capacity /
weighting axis produces the gain?  The round was a *staged* 8-run budget, one
winner per stage, no Cartesian grid, no reader / LR / activation / dropout /
attention / MP / recurrence / `patch_cont` / `atom_shell` / `bond_shell` /
multi-dictionary sweep.

## 2. Complete results (7 of the 8-run budget used)

| stage | tag | decoder | d_e | K | s | λ | horizon | params | raw best | soup valid MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Z0 = P1 ref | p1 | 48 | 32 | 8 | 33.959 | 240 | 97,865 | 0.136478 | 0.131975 |
| A | Z1 | p1 | 48 | 32 | 8 | 33.959 | 320 | 97,865 | 0.133575 | 0.127907 |
| A | Z2 | p1 | 48 | 32 | 8 | 16.979 | 240 | 97,865 | 0.140523 | 0.134484 |
| A | Z3 | p1 | 48 | 32 | 8 | 8.490 | 240 | 97,865 | 0.138742 | 0.133574 |
| B | **H1** | h1 | 48 | 32 | 8 | 33.959 | 320 | 97,487 | 0.129284 | **0.123549** |
| B | H2 | h2 | 48 | 32 | 8 | 33.959 | 320 | 110,335 | 0.139304 | 0.135015 |
| C | E64 | h1 | 64 | 32 | 8 | 33.959 | 320 | 99,855 | 0.135865 | 0.129702 |
| D | K64S8 | h1 | 48 | 64 | 8 | 12.432 | 320 | 107,247 | 0.145981 | 0.138777 |
| — | K64S12 | — | — | — | — | — | — | — | not triggered | — |

Frozen winner `final_config.json`: **H1** —
`{decoder: h1, d_e: 48, K: 32, s: 8, lambda_factor: 0.25, horizon: 320}`,
Top-5 soup members `[293, 309, 313, 316, 317]`, 2,081 s, 207 MB peak.

```
ZINC_BEST_CLEAN_DICTIONARY_CORE_VALID_MAE = 0.123549   (target band 0.120-0.125)
P1 reference 0.131975  ->  delta -0.008426
```

## 3. Selection path

```
Stage A  {Z0 0.131975, Z1 0.127907, Z2 0.134484, Z3 0.133574} -> Z1
Stage B  {Z1 0.127907, H1 0.123549, H2 0.135015}              -> H1
Stage C  {H1 0.123549, E64 0.129702}                          -> H1
Stage D  {H1 0.123549, K64S8 0.138777}                        -> H1
         K64S8 improvement -0.015229 < 0.001 -> K64S12 NOT triggered
```

7 runs of the pre-registered maximum of 8; the conditional run was withheld by
its own frozen gate.

## 4. What produced the gain

**(a) Horizon, at constant λ.** Z1 and Z0 share the same absolute λ (33.959) and
the same architecture; only the horizon differs (320 vs 240). Z1 improves
0.131975 → 0.127907, and its best epoch is 312 — i.e. the P1 horizon was
genuinely truncating the run. This is the first material absolute gain in this
family.

**(b) Reconstruction-weight direction.** At a fixed horizon of 240 the ordering
is monotone in λ: Z0 (33.959) 0.131975 < Z3 (8.490) 0.133574 < Z2 (16.979)
0.134484. Weighting reconstruction more heavily is better for MBAN, not worse.

**(c) A shared compact dictionary-slot decoder.** H1 replaces the flat
638→102→48 decoder with per-slot shared encoders (node slot 96→64→48, edge slot
48→48→32, anchor 62→32) and a 368→128→48 fusion, with **fewer** parameters
(97,487 vs 97,865). It gains a further 0.127907 → 0.123549 on top of Z1 at
identical λ and horizon. The wider H2 (110,335 params) is clearly worse
(0.135015), so the gain is not "more capacity".

**(d) What did not work.** Widening the edge binding `d_E` 48→64 (E64) degrades
to 0.129702, and raising the dictionary to K=64 (K64S8) degrades to 0.138777.
Both add capacity through the dictionary-mediated path and both lose, so the
winning configuration is the *smallest* of the tested decoders.

## 5. Diagnostic: the K64 regression is a coding effect, not a capacity effect

Reporting-only follow-up (no new training run, no budget consumed; artifact
`results/e2e_dictenv_p2_abs/k64_diagnostic.json`).  On 47,588 official-train
`phi65` rows, the frozen tied-IHT coder with 10 steps is compared against the
K-SVD family's own exact OMP coder, and against longer IHT rollouts:

| dictionary | s | OMP (exact) normalised err | IHT-10 | IHT-30 | IHT-100 |
|---|---|---|---|---|---|
| SDB K32 | 8 | 1.452e-05 | 0.010084 | 0.002856 | 0.001655 |
| K64/s8 | 8 | 3.907e-05 | 0.026816 | 0.007372 | 0.001896 |
| K64/s12 | 12 | 4.979e-06 | 0.015776 | 0.002141 | 0.001007 |

Three conclusions:

1. **The frozen coder is the bottleneck of the whole family.** The 10-step tied
   IHT is ~700x worse than OMP on the SDB K32 dictionary (0.0101 vs 1.45e-05)
   while hitting exact top-8 on every row.  H1's record 0.123549 valid MAE was
   therefore produced by a dictionary whose exact sparse code it cannot
   actually compute.
2. **K64 loses in the coder, not in the dictionary.** K64/s8 has a comparable
   (slightly larger) OMP error but a 2.7x worse IHT-10 error (0.0268 vs 0.0101);
   at 100 steps it overtakes K32 at 10 steps (0.0019 vs 0.0101).  The K64S8 run's
   degraded MAE is consistent with an under-converged coder rather than with a
   worse dictionary.
3. **The `lambda_base` recalibration is what carried the defect into the run.**
   `rec_init` *is* the IHT-10 error, so K64's worse coding immediately lowered
   `lambda_base` to 49.73 and λ to 12.43, i.e. straight into the low-λ region
   where Stage A was already worse.  The pre-registered calibration is behaving
   exactly as specified, but at K=64 it converts a coder deficit into a weighting
   change.

This is why the K64S8 row above **must not** be read as "K=64 dictionaries are
worse".  Changing the IHT step count is explicitly outside the frozen sweep
surface of this round, so an IHT-steps round would require a new pre-registration.

## 6. Caveats

* **K64S8 is λ-confounded.** The pre-registration mandates recalibrating
  `lambda_base` after a K change; the K64 calibration gives λ = 12.432 (vs
  33.959), which lands in the low-λ region where Stage A was already worse.  The
  K64 regression therefore mixes a dictionary-capacity effect with a weighting
  effect and must not be read as "K=64 dictionaries are worse for MBAN".
* **Two of the three Stage-A axes are confounded with horizon.** Z1 is
  `λ × 1.0, horizon 320`; Z2/Z3 are `λ × 0.5 / 0.25, horizon 240`. The λ
  ordering above is read only *within* horizon 240 (Z0, Z2, Z3), which is
  internally consistent, but the round does not separate "λ = 33.959 is optimal"
  from "longer horizons help".
* **No MolHIV-style mechanism interventions** (dictionary zero / assignment
  shuffle) were run — they are explicitly out of scope for this tuning round.
  The H1 winner inherits P1's mechanism evidence and has not re-established it
  independently.
* **Valid-only.** The official ZINC test was not loaded, so 0.123549 is a valid
  MAE under the frozen split; the frozen P1 valid→test relation (0.131975 →
  0.107593) must not be assumed to transfer to H1.

## 7. What the winner changes relative to the frozen P1 core

| item | P1 | H1 (P2-ABS winner) |
|---|---|---|
| horizon | 240 | 320 |
| decoder | flat `638→102→48` | shared per-slot `96→64→48` / `48→48→32` / `62→32`, fusion `368→128→48` |
| params | 97,865 | 97,487 |
| φ65 / dictionary / coding | unchanged | unchanged (K32, s8, IHT 10, SDB `D` sha `925d573a…`) |
| λ | 33.95873017865987 | 33.95873017865987 (unchanged) |
| relation | 15-D pure topology | unchanged |

Everything else — the frozen φ65 coordinate, the K-SVD dictionary, exact top-8
tied IHT, the pure-topology relation, the backend — is untouched.

## 8. Closure

`final_config.json`, `REPORT.md`, `DECISION.md` and this note record the round.
No ZINC test read, no claim about mechanism, no architecture search beyond the
pre-registered 8-run budget.
