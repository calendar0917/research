# PEC-v0 — analysis (Pure Environment Composition on ZINC)

Round: **PEC-v0**, protocol `pec_v0`, study `zinc-context-gap`.
Pre-registration: [`pec_v0_preregistration.md`](pec_v0_preregistration.md)
(+ **Amendment A1**, per-occurrence rooted bases; + **Amendment A2**,
[`pec_v0_gate1_decision.md`](pec_v0_gate1_decision.md)).
Prior-artifact audit: [`pec_v0_prior_artifact_audit.md`](pec_v0_prior_artifact_audit.md).

Local CPU only. Commit `883e529b159cae49fd1a9d279db2b40b8ff5e77c` (implementation
committed on top of it, see the run artifacts). **Official ZINC test was never
loaded. Official valid was never read** (Gates 0-2 use official-train internal
splits only). No historical baseline was retrained.

Results: `results/pec_v0/{gate0.json,gate1_label_free.json,gate2_small_train.json,
GATE1_REPORT.md,GATE2_REPORT.md,decision.json,DECISION.md}`.

---

## 1. What was asked

Can a molecule be represented as **chemical primitives bound to reusable rooted
structural roles → frozen local chemical environments → one read-only static
composition**, with **no message passing, no recurrence and no mixed
chemistry bypass**, and is a **sparse reusable dictionary** a better structural
role coordinate than an ordinary dense one?

---

## 2. The object (frozen, Amendment A1)

For every root `i`, radius-2 induced patch `P_i`:

```
b^V_{iv} in R^11   audited FSAR explicit rooted NODE basis  (topology only)
b^E_{ie} in R^15   audited FSAR explicit rooted EDGE basis  (topology only)
alpha^V_{iv} = SparseDict_V(b^V_{iv})          K=16, s=4, tied-IHT, 10 steps
alpha^E_{ie} = SparseDict_E(b^E_{ie})          K=16, s=4
r^V_{iv} = [one_hot(shell,3) ; alpha^V_{iv}]   (19)
r^E_{ie} = [one_hot(shellpair,5); alpha^E_{ie}](21)
B^V_i = sum_{v in P_i} r^V_{iv} (x) a(q_v)     19x28
B^E_i = sum_{e in Q_i} r^E_{ie} (x) c(b_e)     21x4
E_i   = H([one_hot(q_i,28); vec(B^V_i); vec(B^E_i); S_i]) -> R^48   (frozen)
rho_ij in R^18  pure topology (S0 relation minus path_bond_mean & adjacent bond)
c_ij  = F([E_i+E_j; |E_i-E_j|; E_i*E_j; rho_ij]) -> R^48             (once)
y_hat = G([mean_i E_i; max_i E_i; mean c_ij; max c_ij; global_topology])
```

Purity is structural: `E_i` is computed from a molecule's own occurrences before
any pair tensor exists; `c_ij` is graph-level-pooled and never written back.

---

## 3. Gate 0 — correctness (CPU, data-free + toy graphs)

**8/8 checks pass** (`results/pec_v0/gate0.json`):

| check | observation |
|---|---|
| purity | chemistry change leaves `node_basis`, `edge_basis`, shells, `alpha` bit-identical |
| chemistry isolation | `atom_idx` is chemistry-only; changing chemistry leaves topology roles identical |
| assignment sensitivity | chemistry-placement shuffle changes `E_i` (max abs `0.0991`) |
| environment freeze | changing pair tensors / another molecule leaves a fixed `E_i` **bit-identical** |
| static contract | environment module called **once**; a BAG forward (no pair pass) is finite; mutating `pair_rho` leaves every `E_i` bit-identical |
| relabel invariance | prediction max abs diff `0.0` |
| sparse correctness | `max l0 = 4` for both roles (`s=4`) |
| gradients | finite, non-zero for node dict, edge dict, environment MLP, pair composer, reader |
| parameter accounting | sparse `94 049` = dense `94 049`; coarse `94 036` (0.014 % mismatch) |

No forward-access audit violation: no typed token, no `patch_cont`, no
`path_bond_mean`, no adjacent-bond chemistry, no global typed histogram, no raw
atom/bond readout outside the binding.

## 4. Gate 1 — label-free dictionary (frozen verdict **FAIL**)

Split: fit = official-train 8000, monitor = official-train 2000. K-SVD cost
`899.8 s`. Codes are tied-IHT at exactly `s = 4`.

| role | `E_rec` monitor (K-SVD) | `E_rec` random | K-SVD advantage | used atoms | argmax-used | effective atoms | top-1 mass |
|---|---:|---:|---:|---:|---:|---:|---:|
| node (`b^V`, R^11) | **0.0369** | 0.3305 | 9.0x | 12 / 16 | 7 | 6.0 | 0.575 |
| edge (`b^E`, R^15) | **0.0543** | 0.5299 | 9.8x | 11 / 16 | 3 | 4.5 | 0.488 |

PCA-16 reconstruction is exact (`E_rec` ~1e-31), i.e. the bases are
low-dimensional — the sparse dictionary is best read as a **discretised role
coordinate**, not as a compression of a high-dimensional signal.

Coarse-role recoverability (train-fit logistic probe, 50 k monitor rows):
`alpha^V -> shell` macro-F1 **1.000** (majority 0.217, margin **+0.783**);
`alpha^E -> shellpair` macro-F1 **1.000** (majority 0.180, margin **+0.820**).

**Frozen criteria:** 7/10 pass. Failures: `node_random_ratio` 0.1115 > 0.10,
`edge_random_ratio` 0.1024 > 0.10, `edge_used` 11 < 12. Verdict **FAIL**, kept
verbatim.

**Honest reading** (see `pec_v0_gate1_decision.md`): the two ratio misses are
+0.0115 / +0.0024 absolute over an uncalibrated self-set threshold, and the
K-SVD dictionary is still ~9-10x better than random on reconstruction with exact
`l0 = 4`; `edge_used = 11` is expected for an overcomplete `K=16 > dim=15`
dictionary. The **user-specified** Gate-1 STOP condition (can the dictionary
preserve the coarse rooted structural role) passes with a very large margin.
A design weakness is recorded: because the occurrence basis row already contains
the shell one-hot, the `alpha -> shell` probe is close to tautological and its
PASS is weak evidence.

**Amendment A2** therefore proceeded to Gate 2, with the frozen FAIL reported
unchanged and no `K`/`s`/epoch/threshold/architecture rescue.

## 5. Gate 2 — cheap internal task screen (**PASS**)

Train 2000 official-train molecules, dev 500 official-train molecules
(positions 8000..8500), dictionaries re-fit only on the 2000 train molecules
(224.9 s). Frozen protocol: Adam `lr 1e-3`, `wd 1e-5`, batch 64, L1, clip 5,
60 epochs, best-dev checkpoint, fixed equal-weight **Top-5 epoch-checkpoint
soup**. One seed (0). Parameters: C0 `94 036`, CD `94 049`, CK `94 049`.
**Absolute MAE is not comparable to the strict-static S0 band** (2 000 vs
10 000 training molecules); this is a relative screen only.

| arm | best dev MAE | **Top-5 soup dev MAE** | chem-shuffle MAE | chem-shuffle degradation | neutral-dict MAE |
|---|---:|---:|---:|---:|---:|
| **C0** coarse-only | 0.470864 | 0.466904 | 1.501883 | +1.031019 | — |
| **CD** DenseRole | 0.476846 | 0.467108 | 1.235357 | +0.758511 | — |
| **CK** SparseDict | **0.465776** | **0.461811** | 1.301115 | +0.835338 | 2.175492 |
| CK + **BAG** (unary only) | 0.500130 | 0.496982 | — | — | 2.880178 |
| CK + **SHUFFLE** (env↔pair broken) | 0.475933 | 0.473408 | — | — | 2.885856 |

Frozen criteria (all pass):

| criterion | frozen | observed | verdict |
|---|---|---|---|
| `chem_shuffle_material` | `>= 0.02` | **+0.8353** | PASS |
| `true_beats_bag` | `>= 0.003` | **+0.035171** | PASS |
| `true_beats_shuffle` | `>= 0.003` | **+0.011597** | PASS |
| `dict_not_worse_than_dense` | `CK-CD <= 0.002` | **-0.005297** (sparse better) | PASS |
| `dictionary_alive` | `> 0` | neutral-dict MAE 2.175 vs 0.462 | PASS |

Gate-2 verdict: **`GATE2_PASS_BUY_SEED0`**.

Mechanistic reading:

* **Environment claim supported.** `E_i` is load-bearing: shuffling where each
  chemistry sits (fixed topology and fixed atom/bond multisets) destroys the
  model (`+0.835` MAE on CK, `+0.759` on CD, `+1.031` on C0). The environment is
  a genuine role ⊗ primitive binding, not a chemistry histogram.
* **Composition claim supported.** With the *same* environments and the *same*
  pair relations, breaking only the environment-to-pair correspondence (SHUFFLE)
  costs `+0.0116`; removing the pair pass entirely (BAG) costs `+0.0352`. So the
  read-only static composition carries real information.
* **Dictionary claim (screen level) not refuted.** CK is slightly *better* than
  the exactly parameter-matched CD (`0.46181` vs `0.46711`) and better than
  coarse-only C0 (`0.46181` vs `0.46690`). The neutral-dictionary intervention
  moves the prediction from MAE 0.462 to 2.175, so the sparse code is
  load-bearing rather than decorative.

## 6. Gate 3 — NOT run (stop before the GPU purchase)

The pre-registration requires Gate 3 only if all previous gates pass. The
**frozen Gate-1 verdict is FAIL**, and Amendment A2 authorized only the CPU
Gate-2 screen. A seed-0 official-valid GPU run is therefore **not** purchased
autonomously. This is a deliberate, conservative stop: the round is paused with
a positive screen, not concluded.

If the user authorizes Gate 3, the runner is ready: two arms
(`C0_or_CD` best non-dictionary control and `CK`), one seed (0), official valid
only, `remote-research-runner` workflow (clean commit → preflight → deploy →
GPU smoke → detached formal run → pull → local analysis), no official test.

## 7. Purity audit (final)

* No message passing, no pair→centre / centre update, no recurrence, no relation
  refresh, no attention anywhere in `pec_v0.py`.
* `E_i` is bit-identical under pair mutation (Gate 0), so the environment freeze
  contract holds literally.
* The only chemistry route into the reader is through `E_i`; the pair relation
  is 18-D pure topology with `path_bond_mean` and adjacent-bond chemistry
  deleted; the dictionary input is the chemistry-free FSAR basis.
* No historical baseline was loaded, retrained or read.

## 8. Cost

| stage | wall time | device |
|---|---:|---|
| cache (10 000 + 1 000 molecules) | 94.3 s | local CPU |
| Gate 0 | < 5 s | local CPU |
| Gate 1 (two K-SVD fits + reporting) | 899.8 s | local CPU |
| Gate 2 (two K-SVD fits + 5 trained arms) | ~1 100 s | local CPU |
| formal GPU | **not run** | — |

`official_test_loaded = false`, `official_valid_read = false` in every artifact.

---

## 9. Answers to the five required questions

1. **Environment formation** — structurally supported. `E_i` is a stable,
   chemistry-placement-sensitive role ⊗ primitive binding; the frozen sparse
   dictionary preserves the coarse rooted role perfectly (macro-F1 1.0) and the
   `b^V`/`b^E` axes reconstruct to `E_rec` 0.037 / 0.054.
2. **Composition** — supported. TRUE static composition beats BAG by `0.0352`
   and SHUFFLE by `0.0116` on the frozen 2 000/500 internal screen.
3. **Dictionary** — not worse than parameter-matched DenseRole at screen level
   (`CK 0.46181` vs `CD 0.46711`, i.e. sparse is `0.0053` better), and
   load-bearing (neutral-dictionary MAE 2.175). The Gate-1 frozen numeric
   verdict is FAIL, so this is **not** yet a claim that the sparse dictionary is
   necessary.
4. **Purity** — yes. No MP, no recurrence, no mixed bypass, verified by the
   Gate-0 freeze/purity/contract checks.
5. **Performance** — not established. The formal official-valid band was not
   purchased; the internal 2 000-train dev MAE (`0.4618`) is not comparable to
   the strict-static S0 band.

**Terminal verdict: withheld.** The frozen Gate-2 verdict is
`GATE2_PASS_BUY_SEED0`; a terminal verdict from the allowed vocabulary requires
the Gate-3 official-valid seed-0 run. If Gate 3 confirms the screen, the
indicated terminal verdict is `PURE_SPARSE_DICT_ENV_COMPOSITION_SUPPORTED`;
`ENV_COMPOSITION_SUPPORTED_DENSE_NOT_DICT` is contradicted by the screen;
`ENVIRONMENT_SUPPORTED_COMPOSITION_NULL` and `PURE_ENV_COMPOSITION_NOT_VIABLE`
are contradicted by the screen.

## 10. Limitations

* Gate 2 uses 2 000 training molecules; it cannot rank against the strict-static
  S0 band and says nothing about the absolute band.
* Single seed at Gate 2; the CK-CD gap (`0.0053`) is small relative to the
  documented seed spread (~0.004-0.007) and must not be over-read.
* The dictionary was **frozen** (detached K-SVD) at Gate 2 while the dense
  control was trainable; a task-coupled dictionary is untested
  (`TASK_COUPLED_DICT_SUPPORTED` is not claimed).
* The `alpha -> shell` probe is near-tautological (Amendment A1 consequence) and
  is weak evidence.
* Gate 1's frozen numeric verdict is FAIL; Amendment A2 is a documented
  threshold-design decision, not a rescue.
