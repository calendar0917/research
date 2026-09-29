# Pre-registration — `e2e_dictenv_sem108_v1` (CSSD-Sem108)

Round: **E2E-DictEnv-Sem108-v1** (`e2e_dictenv_sem108_v1`), study
`zinc-context-gap`, track `tracks/ksvd`.
Candidate name: **CSSD-Sem108 / Shell-Resolved Primitive Semantic Interface**.
Prior-artifact audit: [`e2e_dictenv_sem108_v1_prior_artifact_audit.md`](e2e_dictenv_sem108_v1_prior_artifact_audit.md).

CPU only · official ZINC test never loaded · write-then-follow, no post-hoc
change to any constant, gate, threshold, probe or case in this document.

---

## 1. Single question

Is the ~`0.13` ZINC valid plateau caused by the primitive chemistry being
compressed too early — `anchor62 -> anchor_encoder -> 32D` — *before* the
environment decoder, rather than by a weak dictionary fusion function?

## 2. Hypothesis and falsification

**H-sem108.** Feeding the raw standardized shell-resolved primitive semantic
blocks `[Sem108 ; size2]` (110-D) directly into the environment fusion, and
deleting the 62→32 anchor compression, moves the valid soup outside the current
plateau (`<= 0.123` promising, `<= 0.120` strong) while the sparse dictionary
stays genuinely load-bearing.

**Falsified if** `M_S > 0.127` (NO_GO band), or the semantic interface is not
load-bearing (`G_sem_shuffle < 0.005`), or the dictionary correspondence is not
established (`G_corr < 0.003`), or a correctness gate fails.

## 3. Exactly one architectural change

```
OLD (frozen parent):  anchor62 -> anchor_encoder(62->32->32) -> 32D
                      fusion( [32 | 3x48 node | 6x32 edge] = 368 -> 128 -> 48 )

NEW (candidate):      [ Sem108 ; size2 ] = 110D  -> fusion directly
                      fusion( [110 | 3x48 node | 6x32 edge] = 446 -> 114 -> 48 )
```

* `Sem108 = patch_cont[:, 0:108]` = `atom_shell` (3×28=84) ∥ `bond_shell` (6×4=24),
  the exact FEC-S0 shell descriptor blocks resolved from the current code.
* `size2 = anchor[:, 60:62]` (= `patch_cont[:, 140:142]`, bit-identical on the
  standardized split) — retained so the candidate keeps the parent's
  `[log1p n, log1p m]` information.
* The anchor compression path (anchor encoder and the 32-D compressed anchor)
  is **deleted**.  All other computation is the frozen parent: common coordinate
  `q=1`, residual sparse dictionary, IHT-10, node/edge chemistry bindings,
  shells/shellpairs, relation/distance/global/topology backend, moment pooling,
  reader, `H1_LAMBDA`, Adam protocol, Top-5 soup.
* No RNDB, no message passing, no Sem108-only control, no new fine descriptor,
  no `root_atom` / `incident_bonds` / topology-scalar inputs.

## 4. Zero-training audits (Phase A, before any candidate training)

**A1 — exact Sem108 identity.**  Resolve the block geometry from
`fec.SHELL_BLOCKS` / `zpp.SHELL_PAIRS` / `zpp.SHELL_WIDTH` /
`audit.ANCHOR_GROUPS` in the *current* implementation and assert the expected
layout (raising on any drift).  Rebuild the descriptor for sampled official
train/valid molecules with `fec.FactorizedFeatureTransform` and require:
`patch_cont` bit-identical to the durable encoded cache, and the independent
role×primitive reconstruction of `atom_shell`/`bond_shell` exact.  Fit split is
**official train** and never includes valid/test; record mean/scale sha256.

**A2 — `Sem108 ↔ anchor62` marginal relations** (full official train + valid,
destandardized): root-atom `argmax` agreement, `n_nodes × Σ atom_shell` vs
`atom_mass`, `n_edges × Σ bond_shell` vs `bond_mass`, and `size2` identity, each
with `max_abs_error` / `fraction_exact` (tol `1e-5`).  Pass requires
`fraction_exact = 1.0` on both splits.

**A3 — frozen T1 coarse146 block ablation.**  Only if the exact T1 soup state
exists locally; otherwise record `T1_BLOCK_AUDIT_UNAVAILABLE`.  **No T1
retraining, no substitute T1 run.**  (Status is fixed by the prior-artifact
audit: unavailable.)

**A4 — redundancy / collision audit.**  Deterministic feature-equivalence
structure of `Sem108` vs `anchor62` (exact and rounded keys; distinct counts,
per-anchor collision counts, reverse single-anchor fraction).  Recorded; A4
does not gate a stop, it bounds how much new information Sem108 can carry.

**Phase-A stop rule.**  Proceed only if A1 is established, the scaler fit split
is official train, A2 passes and T1 does not *strongly* reject shell semantics.
Otherwise STOP with no training.

## 5. Parameter matching (closed form, one shot)

Parent local interface `= anchor_encoder + fusion = 3072 + 53424 = 56496`
params.  The candidate keeps every parent parameter except anchor encoder and
fusion; only the fusion hidden width `H` is free:

```
P_candidate(H) = (446 + 1 + 48) H + 48 = 495 H + 48
H* = (56496 - 48) / 495 = 114.036...  ->  H = 114
P_candidate = 56478, gap = 18, total = 97709 vs parent 97727
relative |Δ| = 18 / 97727 = 0.0184%  (bound ±2%, preferred ±1%)
```

Chosen **once** at import (`sem.CLOSED_FORM`); not tuned, not swept.

## 6. Correctness gates G0–G13 (all must pass before training)

| gate | requirement |
|---|---|
| G0 | `Sem108` geometry established from current code (`atom_shell 84`, `bond_shell 24`, `sem_dim 108`) |
| G1 | scaler fit split official train; valid not in fit; mean/scale sha256 recorded |
| G2 | A2 pass on train+valid (`fraction_exact = 1.0`) |
| G3 | `anchor[:,60:62]` and `patch_cont[:,140:142]` bit-identical; interface last 2 columns equal them |
| G4 | shared parent state-dict params (everything except the replaced `fusion.*` and the removed `anchor_encoder.*`) bit-identical; parent-equivalence model full forward (env + prediction) bit-identical |
| G5 | node slots bit-identical to parent |
| G6 | edge slots bit-identical to parent |
| G7 | relation / global / topology / environment / prediction bit-identical to parent in the equivalence model; reader / pair encoder / distance gate reused classes |
| G8 | `cm.C6_MASK is cssd.CSSD_MASK`, `cm.c6_equivalence_check()`, `cm.arm_mask(cssd.CSSD_SPEC) is cm.C6_MASK` |
| G9 | no `psi_*` parameter and no `psi_A`/`psi_E` attribute (no RNDB) |
| G10 | interface width 110; perturbing `patch_cont[:,108:146]` (root/incident/scalars) leaves predictions bit-identical; perturbing `patch_cont[:,0:108]` changes them |
| G11 | parameter ratio within ±2% (preferred ±1%) |
| G12 | node-relabelling invariance on real valid molecules (`<= 1e-4`, float32 reduction-order tolerance) |
| G13 | official-test blocker raises on `official_test_loaded=True`; every payload carries `false` |

Any failure ⇒ STOP; no training.

## 7. Smoke (≤100 optimizer steps, trainability only)

8 epochs × 1024 train / 512 valid = **64 optimizer steps**, no architecture
selection from the result.  Checks: finite loss / predictions; non-zero
gradients at init and after the smoke for `D`, `fusion.W1`, `fusion.W2`,
`W_A_S`, `W_E_S`; Sem108 perturbation changes predictions.
A smoke failure stops the round (no width / init / lr rescue).

## 8. Formal run — exactly one trajectory

320 epochs, seed 0, 8 CPU threads, Adam `lr=1e-3`, `wd=1e-5`, batch 128,
clip 5.0, train shuffle offset 91011, eval offset 91012, identical Top-5 soup on
valid MAE.  One seed only; no second run, no early-stop tuning, no
hyper-parameter change after observing the curve.

## 9. Frozen inference probes (soup state only, no retraining)

| probe | content |
|---|---|
| **S1** | `sem_block_zero` — zero the whole standardized Sem108 (= train-mean neutralization), size2 retained |
| **S2** | `use_sem_row_shuffle` — permute the whole Sem108 row across the roots of each molecule, 5 seeds `101/202/303/404/505`, size2 retained per root |
| **S3** | `sem_atom_zero` and `sem_bond_zero` — atom84 / bond24 blocks separately |
| **D1** | residual `alpha -> 0` (diagnostic only, not a gate) |
| **D2** | node-assignment shuffle (frozen parent semantics), 5 seeds |
| **D3** | edge-assignment shuffle (frozen parent semantics), 5 seeds |

`G_sem_shuffle = mean_S2(MAE) - M_S`; `G_sem0 = M_S1 - M_S`;
`G_atom`, `G_bond` from S3.
`G_node`, `G_edge` from D2/D3; **`G_corr = max(G_node, G_edge)`**;
gates `G_corr >= 0.010` clear / `>= 0.003` directional.

## 10. Performance bands and decision table

Bands of `M_S`: `<= 0.120` **strong**; `<= 0.123` **promising**;
`0.123 – 0.127` **within existing band**; `> 0.127` **NO_GO**;
`> 0.135` material regression (flagged wherever it occurs).

| case | condition | verdict label |
|---|---|---|
| A | `M_S <= 0.120`, `G_sem_shuffle >= 0.005`, `G_corr >= 0.010`, dictionary health pass | `CSSD_SEM108_STRONG_SINGLE_SEED_SUPPORTED` |
| B | `0.120 < M_S <= 0.123`, `G_sem_shuffle >= 0.005`, `G_corr >= 0.010` | `CSSD_SEM108_PROMISING_SINGLE_SEED` |
| C | `M_S <= 0.123`, `G_sem_shuffle >= 0.005`, `G_corr < 0.003` | `SEM108_BACKBONE_SIGNAL_DICTIONARY_INCREMENT_NOT_ESTABLISHED` |
| D | `0.123 < M_S <= 0.127`, `G_corr >= 0.010` | `DICTIONARY_INCREMENT_RETAINED_BUT_NO_NEW_TASK_BAND` |
| E | `M_S > 0.127` | `CSSD_SEM108_NO_GO` |
| boundary | task band + `G_sem_shuffle` pass with `0.003 <= G_corr < 0.010` | `CSSD_SEM108_TASK_SIGNAL_WITH_DIRECTIONAL_DICTIONARY_INCREMENT` (labelled boundary; no strong claim) |

Case A additionally requires dictionary health (`active >= 24`,
`effective >= 8`, `top1 <= 0.75`).  Single seed, no matched baseline rerun:
all historical comparisons remain unmatched context.

## 11. Stop discipline

After the single 320-epoch run and the frozen probes, the round **stops**.
No seed 1, no rescue run, no width/init/lr/lambda/horizon change, no retrained
baseline, no second candidate, no official-test read.  Any follow-up requires a
new pre-registration.

## 12. Result layout

`tracks/ksvd/results/e2e_dictenv_sem108_v1/`

```
historical_references.json        audit/{sem108_identity, anchor_relation, t1_block_ablation, audit_decision}.json
preflight.json  preregistration_snapshot.json  parameter_audit.json
correctness.json                  smoke.json
run_seed0.json  curve_seed0.csv   soup.json
mechanism/{sem_meanfill, atom_block, bond_block, dict_zero,
           node_assignment_shuffle, edge_assignment_shuffle,
           sem_root_shuffle, dictionary_health}.json
summary.json  REPORT.md  DECISION.md
```

Every JSON carries `protocol_version`, `git_commit`, `device = cpu` and
`official_test_loaded = false`.
