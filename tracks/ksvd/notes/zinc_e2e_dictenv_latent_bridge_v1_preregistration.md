# Pre-registration — `e2e_dictenv_latent_bridge_v1` (SEM108 + local task dictionary bridge)

Round: **E2E-DictEnv-Latent-Bridge-v1** (`e2e_dictenv_latent_bridge_v1`), study
`zinc-context-gap`, track `tracks/ksvd`.
Candidate name: **SEM108 + local task dictionary bridge**.
Core module: `tracks/ksvd/experiments/luyin16/e2e_dictenv_latent_bridge_v1.py`.
Stage module: `tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_latent_bridge_v1.py`.
Delivered design / NumPy reference package (vendored, byte-identical):
`tracks/ksvd/experiments/luyin16/latent_bridge_reference/v1_20261001/`
(`zinc_dictionary_next_step.md`, `latent_dictionary_bridge_reference.py`,
`validate_dictionary_bridge.py`, `dictionary_bridge_audit.json`).

CPU only · official ZINC test never loaded · write-then-follow, no post-hoc
change to any constant, gate, threshold, probe or case in this document.

Background (not a claim).  The immediately previous candidate HierRel was
stopped at official-valid soup MAE `0.182343`; the previous Sem108 candidate
reached `0.123705`.  The delivered negative-result analysis concluded that the
HierRel attempt changed too many things at once and that the next cheap step was
to protect the already-effective Sem108 representation, insert one shared
low-dimensional task dictionary into every local relation path, and guard the
existing representation with numerical acceptance checks before paying for a
full run.

---

## 1. Single question

Does one shared low-dimensional task dictionary on the Sem108 local fusion
output — with the Sem108 inputs, structural dictionary, common coordinate,
local encoders/fusion, static pair range, distance buckets, global/topology
auxiliaries, reader, loss and protocol all unchanged — improve the absolute
official-valid soup MAE, and is the new dictionary actually learned and used?

## 2. Hypothesis and falsification

**H-bridge.**  A per-object soft-threshold dictionary code
`h[48] -> alpha[96] -> E[48]` inserted at the fusion output, read by both the
unary pooling and the static pair computation, moves the valid soup outside the
Sem108 background (`<= 0.120` promising, `<= 0.115` strong) while `D_L` / `V_L`
receive task gradient, move away from their initialisation and remain
load-bearing under the frozen zero-code / permutation probes.

**Falsified if** `M_S > 0.1233` (STOP the current candidate), or the acceptance
gates fail, or the dictionary does not learn / is not used.  A `0.120 < M_S <=
0.1233` band is recorded as borderline; no strong claim is made.

The background Sem108 `0.123705` is context only: the candidate adds 9,216
parameters and changes the architecture, so any difference is **not** a causal
increment.

## 3. Exactly what is kept and what is added

Kept bit-for-bit from the frozen Sem108 candidate: every shared parameter name
and initial value (including the structural dictionary `D`, common coordinate
`U` / `common_rms`, node/edge bindings, fusion with hidden width 114), the
`[Sem108 ; size2]` interface, the static pair scope, the five distance buckets,
the moment pooling, the global/topology auxiliaries, the reader, `H1_LAMBDA`
and the parent training protocol.

Added — one module, one insertion point:

```
h_i   = Sem108 fusion output (48)                     # unchanged
rho_i = sqrt(mean(h_i^2) + 1e-12),  x_i = h_i / rho_i
Dbar_L = column-normalised D_L                        # 48 x 96, unit columns
L      = 1.05 * eigmax(Dbar_L Dbar_L^T) + lambda2     # detached solver step
alpha^0_i = 0
alpha^{t+1}_i = soft_threshold(alpha^t_i
                  + eta * ((x_i - alpha^t_i Dbar_L^T) Dbar_L - lambda2 alpha^t_i),
                  eta * lambda1)
eta = 1 / L,  lambda1 = 0.05,  lambda2 = 0.01,  t = 0..15 (fixed)
E_i   = rho_i * (alpha_i V_L)                          # 48
```

* `D_L` [48, 96] and `V_L` [96, 48] are the only new parameters (9,216 total).
* Initialisation: private NumPy RNG `seed = 0`; `Q = QR` of a 48x48 Gaussian
  matrix (sign-corrected); `D_L = [I48, Q]`; `V_L = D_L.T`.  The private stream
  never consumes or reorders the parent torch stream, so every shared parameter
  keeps the frozen Sem108 initialisation.
* The 16-step solve is an **unrolled ISTA encoder**, not an exact elastic-net
  optimum.  The step is a detached solver setting; the coding computation keeps
  the task gradient.  `rho_i` is computed from the current object only (no
  moving scaler) and keeps gradient.  All-zero `h` returns all-zero `E` without
  NaN.
* Sparsity is controlled by the soft threshold only: no fixed `l0` budget, no
  top-k truncation, no softmax/attention.  Code density is variable and is
  reported as a distribution, never as a fixed `s`.
* The bridge accepts no `edge_index`, `graph_id`, neighbour state or graph
  hidden state, and performs no write-back.  All local/pair paths read `E`
  directly; there is **no** `h + correction` residual bypass and no new raw
  feature bypass.  The existing low-dimensional global/topology auxiliaries are
  retained unchanged.
* This is a task-driven latent local dictionary: the entries are reusable latent
  environment directions, not visualised chemical fragments.  It borrows the
  task-driven dictionary idea; it is not a reproduction of the full Mairal et
  al. optimisation, and it is not Deep Sets.

## 4. Loss and training

Unchanged parent loss `MAE + H1_LAMBDA * structural_reconstruction`; the frozen
`lambda`, mask and structural code definition are untouched.  The new dictionary
receives task gradient.  No new high-dimensional reconstruction auxiliary, no
HierRel-style double reconstruction loss.  The low-dimensional
`||x - alpha Dbar_L^T||^2` is a diagnostic only and is never added to the outer
loss; `lambda1` / `lambda2` define the inner code, not an outer loss weight.

## 5. Acceptance gates (A, zero training, before any candidate training)

Run on one real official-train batch (32 molecules), same shared initial state,
eval mode:

| gate | requirement |
|---|---|
| A1 | every shared state-dict entry is bit-identical to the Sem108 model |
| A2 | parameter contract exact: body 97,709 + bridge 9,216 = 106,925 |
| A3 | identity mode (`lambda1 = lambda2 = 0`, initial `V_L`): `E` vs Sem108 `E` relative L2 `<= 1e-6`; final prediction max diff `<= 1e-5` |
| A4 | formal-init vs Sem108 `E` total relative squared error `<= 0.05` (and row p95 recorded) |
| A5 | `D_L` / `V_L` MAE-only gradients finite and non-zero; one Adam step changes both; one small SGD step decreases the same-batch MAE |
| A6 | the bridge is called exactly once per forward on both the plain and the masked path |
| A7 | `E` enters the unary pooling (exact equality with the mask-aware pool) and the pair projection input (exact equality) |
| A8 | node relabelling invariance `<= 1e-4` on real valid molecules |
| A9 | batch-offset invariance `<= 1e-5` |
| A10 | bond endpoint swap invariance `<= 1e-6` |
| A11 | changing `pair_relation` / `pair_bucket` cannot change `E` (no relation write-back; exact 0) |
| A12 | clean zero-code intervention: `E = 0`, predictions finite, and the intervention changes the predictions (both paths live) |
| A13 | the official-test blocker raises on `official_test_loaded = True` |

The identity mode is audit-only and is never a training or candidate
configuration.  Any failure ⇒ STOP; no training.

## 6. Short paired trainability check (B, scratch/dev only)

Fixed 1,024 official-train molecules, parent Sem108 and candidate from scratch,
40 epochs each, shared initial shared state and identical data order, parent
loss, CPU 8 threads, **no official valid and no official test access**.  Report
the paired train task MAE curve, initial/final bridge MAE-only gradients, code
density, initial-to-final `E` drift and throughput.  The result is scratch/dev
evidence only and is **not** a scientific stopping rule: a candidate not
beating the parent in 40 epochs does not block the formal run; only
non-finite/dead-gradient/wiring failures do.  No `K`, `lambda`, `LR` or
architecture rescue is allowed after seeing B.

## 7. Formal run — exactly one trajectory

`uv run research run zinc_e2e_dictenv_latent_bridge_v1 --study
zinc-context-gap --mode screen`.

official train 10,000 / valid 1,000; **test never instantiated**; CPU 8 threads;
seed 0; batch 128; Adam `lr=1e-3`, `wd=1e-5`, clip 5.0; 320 epochs; parent
fixed LR; parent Top-5 soup (full-model parameter average including `D`, `D_L`,
`V_L`; fixed buffers must agree).  No rerun of the older 320-epoch runs, no
dense/PCA control, no extra seed, no longer horizon.  Prepare reuses only the
required Sem108 / SDB / common-subspace objects; the old Joint709 prepare is
never invoked.  Smoke: 8 epochs × 1,024 train / 512 valid (64 optimizer steps),
trainability only.

Interrupt ⇒ `INCOMPLETE`, recorded as such, not as a failure.

Endpoint reporting additionally evaluates the soup model on the official train
split in eval mode with the identical loader protocol (same mask, same batch
size, no shuffling) so that the reported train and valid MAE share one
measurement convention.  This is an endpoint diagnostic only and is never a
selection criterion.

## 8. Performance bands and decision table

Bands of `M_S`: `<= 0.115` **strong**; `<= 0.120` **promising**;
`0.120 – 0.1233` **borderline**; `> 0.1233` **STOP the current candidate**.

| case | condition | verdict label |
|---|---|---|
| A | `M_S <= 0.115`, dictionary learned and load-bearing | `LATENT_BRIDGE_STRONG_SINGLE_SEED_SIGNAL` |
| B | `M_S <= 0.120`, dictionary learned and load-bearing | `LATENT_BRIDGE_PROMISING_SINGLE_SEED` |
| C | `0.120 < M_S <= 0.1233` | `LATENT_BRIDGE_BORDERLINE` |
| D | `M_S > 0.1233` | `LATENT_BRIDGE_STOP_CURRENT_CANDIDATE` |

"Learned" = `D_L` and `V_L` moved from the frozen initial frame and both still
receive a non-zero MAE-only task gradient.  "Load-bearing" = the zero-code
`delta_mae >= 0.005` and the mean within-molecule permutation `delta_mae > 0`.
Single seed: no significance claim for small differences.

## 9. Frozen inference probes (soup state only, no retraining)

| probe | content |
|---|---|
| **S1** | zero the whole bridge code (`alpha -> 0`, hence `E -> 0`) |
| **S2** | within-molecule code permutation, 5 seeds `101/202/303/404/505` (row-local bridge ⇒ identical to permuting the bridge input within each molecule) |
| **S3** | keep the trained encoder and head, reset only `D_L` / `V_L` to the initial frame |
| **D1** | code density distribution (mean / p50 / p95 per-object nonzero count) — variable density, never a fixed `l0` |
| **D2** | low-dimensional reconstruction `||x - alpha Dbar^T||^2` (diagnostic only) |
| **D3** | MAE-only task gradient norms of `D_L` / `V_L` on a valid batch |

Every intervention reports both `delta_mae` and the true
`delta_pred_rms = sqrt(mean((intervened - baseline)^2))` over the identical
evaluation loader.  (The previous round's `delta_pred_rms` field was
`RMS(intervened) - RMS(baseline)`, which can be negative; that field is not
reused.)

The zero-code / permutation probes only show that the channel carries
information; they do **not** show that dictionary learning beats a matched
trained control.  The reset-to-init probe is an inference diagnostic only.

## 10. Stop discipline

After the single 320-epoch run and the frozen probes, the round **stops**.  No
seed 1, no rescue run, no `K` / `s` / `lambda` / steps / LR / `rho` /
architecture change, no matched control arm added after seeing the result, no
official-test read.  A follow-up (confirmation seed, matched trained control,
dense/PCA bridge) requires a new pre-registration and only after `M_S <= 0.120`
with a learned dictionary.

## 11. Result layout

`tracks/ksvd/results/e2e_dictenv_latent_bridge_v1/`

```
historical_references.json
audit/{sem108_identity, anchor_relation, t1_block_ablation, audit_decision}.json
preflight.json  preregistration_snapshot.json  parameter_audit.json
correctness.json  smoke.json
run_seed0.json  curve_seed0.csv  soup.json
mechanism/{bridge_probes, dictionary_health}.json
summary.json  REPORT.md  DECISION.md
dev/{acceptance, short_trainability}.json     # scratch/dev only, not in the chain
```

Every JSON carries `protocol_version`, `git_commit`, `device = cpu` and
`official_test_loaded = false`.
