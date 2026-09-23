# FEC-D1 — analysis

Round **FEC-D1** (*Localized Sparse Structural Dictionary Binding*), protocol
`fec_d1`, study `zinc-context-gap`.
Pre-registration [`fec_d1_preregistration.md`](fec_d1_preregistration.md);
implementation [`fec_d1_implementation.md`](fec_d1_implementation.md);
prior-artifact audit [`fec_d1_prior_artifact_audit.md`](fec_d1_prior_artifact_audit.md).

Formal run: commit `0ee273c`, remote A100 (GPU1 after GPU0 migration), seed 0,
240 epochs/arm, no early termination. **Official ZINC test never loaded.**

## 1. Frozen verdict

```
FEC_D1_LOCAL_BINDING_SUPPORTED_DICTIONARY_NOT_SPECIFIC
```

Gate summary:

| quantity | value | gate | threshold | result |
|---|---|---|---|---|
| `M_B` frozen FEC-S1 best-checkpoint replay | 0.136782497 | — | — | baseline guard PASS (recorded 0.136782507, Δ 1.0e-08) |
| `M_D` Dict32 localized-binding soup | 0.131537382 | — | — | — |
| `M_P` PCA32 localized-binding soup | 0.130231331 | — | — | — |
| `M_shuffle` assignment shuffle (5 perms) | 0.144669035 | — | — | — |
| `G_D = M_B − M_D` | **+0.005245115** | A | ≥ 0.003 | **PASS** |
| `G_dict-specific = M_P − M_D` | **−0.001306051** | B | ≥ 0.002 | **FAIL** |
| `G_assign = M_shuffle − M_D` | **+0.013131653** | C | ≥ 0.010 | **PASS** |

The frozen classifier maps `A ∧ ¬B` to
`FEC_D1_LOCAL_BINDING_SUPPORTED_DICTIONARY_NOT_SPECIFIC` (verdict table
§16 of the pre-registration).

## 2. What each gate says

**Gate A (material local gain) — PASS.**
Adding the rank-8 localized binding to the frozen FEC-S1 best checkpoint lowers
official-valid MAE from `0.136782` to a Top-5 weight soup of `0.131537`
(`−0.005245`). The improvement is an order of magnitude above the seed-0 replay
noise (`1.0e-08`). So the *placement* is not vacuous: localizing a rooted
pure-topology 32-D structural coordinate into the rooted shell and binding it
to atom chemistry during environment formation carries predictive information
the shared environment did not have.

**Gate B (dictionary specificity) — FAIL.**
The matched dense control PCA32(`phi_v`) — same dim 32, same frozen
`phi65 → 32-D` map ("train-fit affine PCA" from the same SDB artifact), same
statistic, same branch, same FEC-S1 base, same init seed, same optimizer, same
batch order — reaches a **better** soup than the SDB K=32/s=8 dictionary:
`M_P = 0.130231` vs `M_D = 0.131537`. Dict is `−0.001306` *worse*. The gate
required the dictionary to be better by ≥ 0.002; the sign is opposite. The
sparse, chemistry-free pure-topology dictionary is therefore **not** the
ingredient that produces the Gate-A gain. Any 32-D rooted structural coordinate
appears to work at least as well.

**Gate C (assignment mediation) — PASS.**
Permuting the `alpha_v ↔ q_v` correspondence within each root/shell (keeping
the node set, the shell structure, the `alpha` multiset, the `q` multiset, and
the FEC-S1 base input unchanged) raises MAE by `+0.013132` on average, with a
mean absolute prediction shift of `0.0629` and a maximum of `1.047`. So the
gain is genuinely mediated by *which chemistry is bound to which structural
coordinate at which localization*, not by an extra additive capacity term, a
change of input distribution, or an untyped pooling artefact.

Combining B and C: the mechanism is real and assignment-mediated, but it is
**not dictionary-specific**. The honest reading is
"a localized, chemistry-bound, assignment-sensitive rooted structural
coordinate helps; the sparse SDB vocabulary has no advantage over its matched
dense PCA compression of the same `phi65`."

## 3. Mechanism checks (evaluation only)

* **Branch neutralization**: zeroing `W2` returns the prediction to the frozen
  base (`branch_neutralized_valid_mae = 0.136782511` vs replay `0.136782506`;
  `neutralization_max_abs_pred_shift = 9.54e-07`; `restores_base = true`).
  This confirms the branch is a true additive refinement with an exact `Δe = 0`
  neutral element.
* **Shuffle rows** (`valid_mae` / mean |Δpred| / max |Δpred|):

  | seed | valid MAE | mean shift | max shift |
  |---|---|---|---|
  | 101 | 0.146836 | 0.064551 | 0.804096 |
  | 202 | 0.142833 | 0.062217 | 0.749137 |
  | 303 | 0.140604 | 0.056068 | 0.648016 |
  | 404 | 0.147456 | 0.067443 | 1.047274 |
  | 505 | 0.145617 | 0.064467 | 0.909831 |
  | mean | **0.144669** | 0.062949 | 1.047274 |

* Learning dynamics (both arms, 240 epochs, no early stop):
  * Dict32: best valid `0.131555 @238`; Top-5 soup members `[59,63,101,210,238]`
    (individual `0.131619, 0.132026, 0.131909, 0.131971, 0.131555`), soup
    `0.131537`.
  * PCA32: best valid `0.130414 @140`; Top-5 soup members `[59,98,140,166,172]`
    (individual `0.130828, 0.131160, 0.130414, 0.131091, 0.130520`), soup
    `0.130231`.
  * Train MAE descends smoothly (`0.0884 → 0.0759` Dict, `0.0885 → 0.0766`
    PCA); no instability.

## 4. Historical anchor (context only, pre-registered as such)

The matched base for the gates is the frozen FEC-S1 **best checkpoint**
(`0.136783`), because the frozen protocol only allows soup over the *current
run's* branch checkpoints and forbids re-running the historical soup members.
The historical FEC-S1 seed-0 Top-5 soup (`0.130422`) was recorded and used here
**only as external context**:

* `M_D − anchor = +0.001116` (Dict32 branch soup is *worse* than the historical
  soup);
* `M_P − anchor = −0.000191` (PCA32 branch soup is marginally better).

This matters for interpretation: the +0.005245 Gate-A gain over the single
best checkpoint is real, but it is largely the same improvement that weight
souping the *unmodified* FEC-S1 run already delivered (best `0.136783` → soup
`0.130422`). Against the strongest frozen FEC-S1 anchor, the new branch is at
best neutral (PCA) and slightly negative (Dict). So even setting Gate B aside,
this is not a large new absolute improvement over FEC-S1; it is a
comparable-magnitude refinement whose distinguishing content is structural
coordinate binding, not a stronger baseline.

## 5. Relation to prior rounds

* **SDB-v0 Stage-4** (whole-graph residual of `alpha` on a strong S0) added
  only `+0.002381`, below its material gate. FEC-D1 relocates `alpha` from a
  graph-level residual to a *localized, chemistry-bound, per-environment*
  statistic and recovers `+0.005245` over the frozen single checkpoint — i.e.
  localization + binding does more than the graph-level residual. This
  corroborates the localization thesis while refuting dictionary specificity.
* **FEC-D0** stopped the *R11 within-shell residual subrole* design
  (`FEC_D0_BASIS_HAS_NO_HELDOUT_SUBROLE_SIGNAL`). FEC-D1 is a different
  placement (a validated reusable `phi65 → K32/s8` dictionary, localized into
  rooted chemical environments and bound to atom chemistry), and it was
  explicitly authorized as such; see the prior-artifact audit. The FEC-D0
  verdict is not contradicted here — FEC-D1 never used the R11 residual basis.
* **FEC-S1** remains the strongest single frozen artifact. FEC-D1's PCA32 soup
  (`0.130231`) is the best official-valid number in this repo's static line so
  far, but by `0.000191` over the FEC-S1 soup, which is not material.

## 6. Six mandatory answers

* **Q1 — was the exact SDB `R65 → K32/s8` dictionary reused unchanged?**
  Yes. `results/sdb_v0/dictionary.pt` (`D` 65×32, sha256
  `925d573a58083c3c20f36980dec9bba77dff592b30ea1abf11098ca3a7b7487a`), K=32,
  s=8, `DICT_SEED=20260924`; coding via `sdb_v0.omp_codes → tccd_v0.omp_codes`
  (exact top-s). G0 verified `phi65` bit-identical to the FSAR cache, `alpha`
  deterministic and exactly `l0 = 8`; no refit, no K/s sweep.
* **Q2 — is the dictionary still pure-topology, with chemistry entering only at
  environment formation?** Yes. G1 (chemistry relabelling leaves `phi`/`alpha`
  unchanged, max abs diff `0.0`) and G2 (`alpha_v` is node-centric and shared
  across roots) pass. Chemistry enters only through `q_v` in the binding
  statistic, which is computed *during* environment formation and injected into
  the local environment, never read by the dictionary.
* **Q3 — does the localized Dict binding give a material gain over the matched
  frozen base (≥ 0.003)?** Yes: `G_D = +0.005245` (Gate A PASS).
* **Q4 — is the Dict32 binding better than the matched dense PCA32 binding by
  ≥ 0.002?** **No**: `M_P − M_D = −0.001306` (Gate B FAIL); PCA32 is better.
* **Q5 — does breaking the `alpha ↔ q` assignment (within root/shell) worsen
  MAE by ≥ 0.010?** Yes: `G_assign = +0.013132` (Gate C PASS).
* **Q6 — verdict?**
  `FEC_D1_LOCAL_BINDING_SUPPORTED_DICTIONARY_NOT_SPECIFIC`.
  Reading: localized, chemistry-bound, assignment-sensitive structural
  coordinates help the frozen shared environment; the already-validated sparse
  pure-topology dictionary is **not** required for that help.

## 7. Correctness gates G0–G9

All 10 passed on the remote formal commit (local CPU and remote agree):

| gate | result |
|---|---|
| G0 SDB code identity | PASS — `phi` bit-identical (3/3 mols, 71 nodes), alpha repeat diff 0.0, max l0 = 8 |
| G1 chemistry purity | PASS — relabelling max abs diff 0.0 |
| G2 root localization | PASS — node-centric alpha diff 0.0; 118 shared nodes change shell across roots |
| G3 binding reference | PASS — float64 hand-loop max abs diff 9.9e-09 |
| G4 centering / assignment | PASS — shuffle changes statistic (max 10.91); both multisets preserved |
| G5 exact baseline containment | PASS — W2 = 0 ⇒ bit-identical adapter output and prediction |
| G6 no readout bypass | PASS — W2=0 stat has no effect; active branch changes prediction only via the adapter |
| G7 gradients | PASS — W2 grad 0.00897 at step 0; W1 grad 0.0 at step 0 and 0.00111 after one step |
| G8 strict-static purity | PASS — static contract; pre-pair local output bit-identical under pair mutation |
| G9 official-test blocker | PASS — `test` access raises; encoded cache `official_test_loaded=false` |

Parameter accounting: base 66,170 frozen; branch 21,696 trainable
(`W1` 21,504 + `W2` 192) = +32.8 %; parameter-matched across arms.

## 8. Honest limitations

* **Single seed (0).** No seed 1 was run (not authorized). The `M_P − M_D`
  margin is `0.0013`, larger than the seed-0 replay noise but from one run;
  it is a directional result.
* **Historical-soup caveat.** As in §4, `M_D` and `M_P` are both within
  `±0.0011` of the historical FEC-S1 soup, so Gate A is a statement about the
  frozen single-checkpoint base, not a new absolute record. The pre-registered
  gate is reported as frozen; this context is added, not substituted.
* **No official test.** `official_test_loaded = false` everywhere; the claim is
  in the official-valid regime only.
* **Gate B interpretation.** The failure means "the sparse dictionary is not
  better than matched dense PCA32", not "the localized binding is useless".
  Gate A and Gate C both pass, so the *binding mechanism* is supported; only
  the dictionary-specific value is rejected.

## 9. Decision

Per the pre-registered rule, `A ∧ ¬B` is a **STOP** for the dictionary
performance route. The construct that survives is the localized
chemistry-bound structural coordinate; the construct that does not is the
sparse SDB dictionary as a distinct source of value. Recorded as
`decision-fec-d1-stop-dictionary-not-specific-20260924`. No rescue (no new K/s,
no refit, no task-coupled D, no graph residual, no reader widening, no seed 1,
no test access) is licensed by this result.
