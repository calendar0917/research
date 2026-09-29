# Analysis — `e2e_dictenv_sem108_v1` (CSSD-Sem108)

Round: **E2E-DictEnv-Sem108-v1** (`e2e_dictenv_sem108_v1`), study
`zinc-context-gap`, track `ksvd`.
Candidate: **CSSD-Sem108 / Shell-Resolved Primitive Semantic Interface**.
Pre-registration: [`e2e_dictenv_sem108_v1_preregistration.md`](e2e_dictenv_sem108_v1_preregistration.md)
Prior-artifact audit: [`e2e_dictenv_sem108_v1_prior_artifact_audit.md`](e2e_dictenv_sem108_v1_prior_artifact_audit.md)

Verdict: **Case D — `DICTIONARY_INCREMENT_RETAINED_BUT_NO_NEW_TASK_BAND`**
(`M_S = 0.123705`, best soup of the line's unmatched single runs, but inside
the existing 0.123–0.127 band and 0.000705 above the pre-registered
"promising" threshold of 0.123).

CPU only, `official_test_loaded = false` everywhere, exactly one seed-0
trajectory.  Formal-run commit `f836b04` (audit `69435e3`, analysis additions
`c841c6d`).

---

## 0. Question

Is the ~`0.13` ZINC valid plateau caused by the primitive chemistry being
compressed too early — `anchor62 -> anchor_encoder(62->32->32) -> 32D` — before
the environment decoder, rather than by a weak dictionary fusion function?

Single architectural change, single hypothesis, falsification test.  No
baseline rerun, no sweep, no seed 1.

---

## 1. What stayed frozen

Parent object `CSSD-q1` (`CSSDModel`, `tracks/ksvd/experiments/luyin16/e2e_dictenv_common_subspace_dictionary_v1.py`),
`cm.H1_CONFIG` (`d_e=48`, `K=32`, `s=8`, `lambda=33.95873017865987`,
`horizon=320`, `dict_kind=sdb32`), `cm.C6_MASK` (`cssd.CSSD_MASK is cm.C6_MASK`,
`cm.c6_equivalence_check()` true), train-only common subspace `q1`
(`rms=[5.082852828320509]`), residual dictionary `Dbar_perp`, IHT-10, node/edge
shell definitions, pair relation, distance bucket, global branch, topology
branch, moment pooling, reader, Adam `lr=1e-3`, `weight_decay=1e-5`, batch 128,
grad clip 5.0, Top-5 soup protocol.  Nothing was retuned.

The candidate keeps the parent parameters **bit-for-bit**: of the shared
state-dict keys (everything except the replaced `fusion.*` and the removed
`anchor_encoder.*`), all 45 are `torch.equal` to `cssd.build_cssd_model`
(`correctness.json`, G4).

---

## 2. New architecture (exactly as pre-registered)

```
OLD:  anchor62 -> anchor_encoder(62->32->32) -> 32D
      fusion([32 | 3x48 node | 6x32 edge] = 368 -> 128 -> 48)

NEW:  [ Sem108 ; size2 ] = 110D (raw standardized blocks) -> fusion directly
      Sem108 = patch_cont[:, 0:108] = atom_shell (3x28 = 84) | bond_shell (6x4 = 24)
      size2  = anchor[:, 60:62] (= patch_cont[:, 140:142])
      fusion([110 | 3x48 node | 6x32 edge] = 446 -> 114 -> 48)
```

No RNDB (`psi_*` absent, G9), no message passing, no new fine structural
descriptor, no `root_atom` / `incident_bonds` / topology-scalar inputs (G10).
Not a re-typed copy: `SEM108Model` subclasses `CSSDModel`, deletes the anchor
encoder and re-creates only the fusion width; the whole shell routing /
index-add / encoders / backend / reader are the parent's code paths
(correctness G5–G7 bit-identical).

Implementation `tracks/ksvd/experiments/luyin16/e2e_dictenv_sem108_v1.py`;
runner `tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_sem108_v1.py`.

---

## 3. Parameter budget (closed form, one shot)

| item | value |
|---|---|
| parent local interface | `anchor_encoder 3072 + fusion 53424 = 56496` |
| candidate fusion `446 -> H -> 48` | `495 H + 48`, `H = 114` -> 56478 |
| gap | 18 |
| parent total | 97727 |
| candidate total | **97709** (delta -18) |
| relative | **0.0184 %** (bound ±2 %, preferred ±1 %) |

`sem.CLOSED_FORM` is computed once at import; no width sweep was performed.

---

## 4. Zero-training prior audit (Phase A)

`audit/audit_decision.json` -> **PROCEED**.

* **A1 — Sem108 identity.** Geometry resolved from the current implementation
  (`zpp.SHELL_PAIRS`, `fec.SHELL_BLOCKS`, `audit.ANCHOR_GROUPS`), raising on any
  drift; `atom_shell 84`, `bond_shell 24`, `sem_dim 108`.  On 64 official-train
  and 64 official-valid molecules the rebuilt descriptor is **bit-identical** to
  the durable encoded cache (`cache_max_abs_diff = 0.0`) and the independent
  role×primitive reconstruction of `atom_shell`/`bond_shell` is exact
  (`semantics_max_abs_diff = 0.0`, 1493/1493 and 1518/1518 patches).  Split fit
  is official train only (`valid_in_fit = false`), 231664 train patches,
  mean/scale sha256 recorded.
* **A2 — `Sem108 -> anchor62` relations** (full train 231664 rows / valid 23083
  rows, destandardized): root-atom `argmax` agreement `1.0` (max error 0.0);
  `n_nodes × Σ atom_shell` max error `5.7e-7` / `5.2e-7`, fraction exact `1.0`;
  `n_edges × Σ bond_shell` max error `6.5e-7` / `6.4e-7`, fraction exact `1.0`;
  `size2` bit-identical (`max_abs_error 0.0`, fraction `1.0`).  `a2_pass = true`.
* **A3 — T1 block ablation.** `T1_BLOCK_AUDIT_UNAVAILABLE`: the exact T1
  soup/selection tensors are not durable (git-ignored, deleted); no T1
  retraining, no surrogate.  T1's `0.12577` remains a whole-interface
  reference only.
* **A4 — redundancy / collision.** `Sem108` strictly refines `anchor62`:
  anchor keys are injective from Sem108 (each Sem108 key maps to one anchor),
  7016 distinct anchors vs **10893** distinct Sem108 keys on train; 77.4 % of
  train rows share an anchor with at least two distinct Sem108 values (valid
  68.2 %).  So the direct interface carries strictly more primitive information
  than the compressed anchor — but only marginally more *distinct* patches
  (median 1 Sem108 per anchor; mean 1.55).

---

## 5. Correctness gates G0–G13 (all PASS)

Recorded in `correctness.json` (14/14 gates, `all_passed = true`):

* shared parent params bit-identical (45 keys, 0 mismatches);
* parent-equivalence model full-forward bit-identical to `CSSDModel` for
  coord / node slots / edge slots / relation / global / topology / environment /
  prediction (all `0.0`);
* `anchor[:,60:62] == patch_cont[:,140:142] == interface[:,108:110]` bit-identical;
* forbidden T1 blocks invisible (`patch_cont[:,108:146] += 5` gives prediction
  diff `0.0`); Sem108 perturbation moves the prediction (`0.1314`);
* C6 mask identity, no `psi_*`, parameter ratio 0.0184 %, official-test blocker
  raises;
* node-relabelling invariance on real valid molecules: `max_abs_pred_diff
  1.74e-05 <= 1e-4` (float32 reduction-order tolerance; relabelling permutes
  the `index_add_` order).

14 focused tests pass (`tracks/ksvd/tests/test_e2e_dictenv_sem108_v1.py`).

---

## 6. Smoke (trainability only)

`smoke.json`: 8 epochs × 1024 train / 512 valid = **64 optimizer steps**.
Finite loss/predictions; `grad_D`, `grad_fusion.W1/W2`, `grad_W_A_S`,
`grad_W_E_S` all non-zero at init and after the smoke; Sem108 perturbation
changes predictions.  `best_valid_mae` 0.669095.  No stop condition fired.

---

## 7. Training — the single seed-0 trajectory

`run_seed0.json`, `curve_seed0.csv`.  320 epochs, seed 0, 8 threads, wall
**1916.8 s**.  No early stop.

| epoch | valid MAE | grad(D) | fusion W1 grad | active atoms |
|---|---|---|---|---|
| 1 | 0.606535 | 2.64e-2 | 6.5e-1 | 31 |
| 20 | 0.263892 | 7.10e-2 | 1.1 | 30 |
| 40 | 0.210591 | 5.93e-2 | 8.4e-1 | 31 |
| 80 | 0.160435 | 3.92e-2 | 6.1e-1 | 32 |
| 160 | 0.147581 | 2.45e-2 | 4.4e-1 | 31 |
| 240 | 0.138572 | 2.38e-2 | 5.0e-1 | 32 |
| 320 | 0.143351 | 4.53e-2 | 9.8e-1 | 31 |

* best valid `0.131313` @ epoch 309;
* Top-5 soup members `[238, 255, 257, 282, 309]`;
* **soup `M_S = 0.123704927947314`**.

Dictionary health (`mechanism/dictionary_health.json`, full valid, residual
slice): 32/32 active atoms, effective atoms 20.71, top-1 activation share
**0.823** (above the 0.75 Case-A health threshold), residual reconstruction
relative 0.97696 (full-descriptor diagnostic; the optimised residual term is
~4e-5), dictionary movement 5.24, task gradient norm on `D` 0.0234.

---

## 8. Frozen interventions (no retraining)

`soup.json` states; all probes evaluated on the frozen soup state with the
parent weights / dictionary / common coordinate / backend / reader retained.

| probe | valid MAE | delta vs `M_S` |
|---|---|---|
| `M_S` (soup) | 0.123705 | — |
| **S1** `sem_block_zero` | 1.430745 | **G_sem0 = +1.307040** |
| **S2** Sem108 row shuffle (5 seeds) | mean 0.616964 | **G_sem_shuffle = +0.493259** (`0.4811 – 0.5054`) |
| **S3a** `sem_atom_zero` (84D) | 1.221278 | G_atom = +1.097573 |
| **S3b** `sem_bond_zero` (24D) | 0.604106 | G_bond = +0.480401 |
| **D1** residual `alpha -> 0` | 0.190254 | G_dict0 = +0.066549 (diagnostic only) |
| **D2** node assignment shuffle (5 seeds) | 0.123705 | **G_node = 0.000000** |
| **D3** edge assignment shuffle (5 seeds) | mean 0.161950 | **G_edge = +0.038245** (`0.0344 – 0.0428`) |

`G_corr = max(G_node, G_edge) = +0.038245 >= 0.010` -> *clear incremental*
dictionary correspondence.

**Semantic mechanism (strong).**  Zeroing the 108-D Sem block (equivalent to
replacing it with the train mean) destroys the model (`+1.307`), and shuffling
the message across roots within each molecule costs `+0.493`; both atom and bond
sub-blocks are individually load-bearing (`+1.098`, `+0.480`).  The direct
interface is the strongest input channel in the round's line so far.

**Dictionary mechanism (edge-only).**  `D1` (`alpha -> 0`) costs `+0.067`,
positive but well below RNDB's `+0.289`.  The **node** correspondence is
*exactly* zero, and the reason is measurable in the soup state (§9).  The
**edge** correspondence is alive and above the clear gate, so the
pre-registered `G_corr` survives on the edge path alone.

---

## 9. Node-binding collapse (observation, not an intervention)

`mechanism/binding_health.json` (inference-only parameter diagnostic of the
frozen soup state):

| parameter | absmax | note |
|---|---|---|
| `W_A_S` | `7.05e-38` | float32 denormal -> node slots identically 0 |
| `W_A_C` | `7.05e-38` | same |
| `node_encoder.0.weight` | `7.04e-38` | zero input -> zero gradient -> same annihilation |
| `W_E_S` | `0.724` | alive |
| `W_E_C` | `0.579` | alive |
| `node_encoder.2.weight` | `0.0095` | alive (input is `SiLU(bias)`, not the dead slots) |
| `D` | norm `5.02` | alive |

The node dictionary↔atom binding was healthy in the 8-epoch smoke state
(soup norms `||W_A_S|| = 1.58`, `||W_A_C|| = 1.47`) and is annihilated in the
raw (best-epoch), soup and final states; the dead first node-encoder layer
shows the collapse cascades once the node slots are exactly zero.  The code path is bit-identical to the
parent at initialisation (G4/G5), so this is a **learned redundancy collapse**,
not a defect: once the fusion can read the raw `atom_shell` chemistry directly,
the node path — which carries the same chemistry through the compressed
dictionary coordinate — becomes redundant, its data gradient vanishes, and
Adam's coupled L2 decay drives its weights to the float32 underflow floor.
The parent CSSD-q1 soup keeps `|W_A_S|max = 1.45`; RNDB `0.78`; the frozen C6
controls `1.03–1.36` — this collapse is specific to the direct-interface
architecture.

Consequences: `G_node = 0.0` is a *structural* zero, not a weak effect, and the
node assignment shuffle carries no information about this solution.  The
`G_corr` gate is therefore driven entirely by the edge path, and the "dictionary
increment" language in Case D must be read as edge-resolved dictionary
correspondence plus the residual-zero diagnostic, not as a joint node+edge
result.

---

## 10. Task interpretation

* `M_S = 0.123705` -> band **`SEM108_WITHIN_EXISTING_PERFORMANCE_BAND`**
  (`0.123 < M_S <= 0.127`), missing the pre-registered promising threshold
  (`<= 0.123`) by `0.000705`.

| reference (unmatched) | soup |
|---|---|
| **CSSD-Sem108 (this round)** | **0.123705** |
| T1 tuned (fine coarse146 interface) | 0.125765 |
| training-protocol-audit control `lr=1e3` | 0.127428 |
| FINAL-CLEAN C6 | 0.128499 |
| CSSD-q1 seed 0 | 0.130028 |
| RNDB seed 0 | 0.133117 |

`G_hist = 0.130028 - 0.123705 = +0.006323` against historical CSSD-q1.  This is
the best soup of the line's single seed-0 runs, but **every comparison is
historical and unmatched** (no baseline was rerun), so it is descriptive
context only, not an architecture claim.

Reading the single question: *removing the early anchor compression does not
break the plateau*.  The direct shell-resolved interface is the strongest
input channel measured in this line (S1/S2 far above every gate) and the
candidate reaches the low end of the existing band, but it does not open a new
band: the pre-registered strong/promising thresholds (0.120/0.123) are not
reached.  The plateau is therefore not explained by "the decoder never sees the
primitive chemistry"; feeding it the raw blocks changes the *mechanism* the
model uses (node dictionary binding is switched off) more than it changes the
task number.

---

## 11. Case classification (pre-registered table)

Frozen conditions: `M_S = 0.123705` (band 0.123–0.127),
`G_sem_shuffle = +0.493 >= 0.005`, `G_corr = +0.038 >= 0.010`.

That is exactly **Case D**:
`DICTIONARY_INCREMENT_RETAINED_BUT_NO_NEW_TASK_BAND` — no new task band, but
the dictionary correspondence gate is retained.  No boundary flag was needed.
The node-collapse observation is recorded as a `mechanism_note` in
`summary.json`, `REPORT.md` and `DECISION.md`; it does not change the case
(which is defined on `G_corr = max(G_node, G_edge)`), but it narrows the
"dictionary increment" claim to the edge path.

---

## 12. Limitations

1. **One seed, no matched baseline.**  All task comparisons are unmatched
   historical context; `+0.006323` vs CSSD-q1 is at the edge of the round's
   descriptive resolution and cannot be read as a gain.
2. **Band boundary.**  `M_S = 0.123705` is inside the existing band and only
   `7e-4` above the promising threshold; it does not falsify the plateau.
3. **Node collapse is unexplained causally.**  The observation is solid
   (denormal weights, smoke alive, parent alive), but *why* the collapse happens
   (pure redundancy vs coupled-weight-decay dynamics) is not identified by this
   round; no ablation isolates it.
4. **`G_corr` is edge-only** because of the collapse; the round cannot claim a
   joint node+edge dictionary correspondence.
5. **Dictionary health caveat.**  Top-1 usage share 0.823 exceeds the 0.75
   Case-A health condition (irrelevant for the Case-D verdict, but the
   dictionary is concentrated).
6. **A3 unavailable.**  No per-block attribution of T1's `0.12577`; the round's
   interface reasoning is supported by A1/A2/A4 and the intervention probes,
   not by a T1 block ablation.

---

## 13. Next-round shapes (recorded only; nothing authorized or implemented)

* **Matched node-binding amputation control.**  Train the same Sem108
  architecture with the node dictionary binding removed (or initialised to
  zero) under the identical protocol, >= 3 seeds, same soup; this separates
  "the direct interface *replaces* the node dictionary path" (mechanism) from
  "the collapse is an optimizer accident" (trainability).  Only a new
  pre-registration authorizes it.
* **Multi-seed confirmation of the band position.**  Seed 1/2/3 of the exact
  frozen candidate would place `0.1237` on a distribution; not authorized here.
* **Anchor-vs-Sem108 capacity-matched comparison.**  A same-round matched pair
  (compressed anchor interface vs direct Sem108 interface, identical params and
  soup) would isolate the compression effect without historical comparison.
* **Sparse-usage regularisation** to move top-1 usage back under 0.75 would be
  a *new* hypothesis, not a rescue of this round.

No post-hoc rescue, no width / init / lr / lambda / horizon change, no seed 1,
no baseline rerun, official test never loaded.
