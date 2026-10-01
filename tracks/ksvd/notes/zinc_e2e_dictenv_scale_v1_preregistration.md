# Pre-registration — `e2e_dictenv_scale_v1` (unified Small/Full task-dictionary scaling)

Round: **E2E-DictEnv-Scale-v1** (`e2e_dictenv_scale_v1`), study
`zinc-context-gap`, track `tracks/ksvd`.
Candidate name: **Sem108 + widened task dictionary (Full, m = 3)**.
Core module: `tracks/ksvd/experiments/luyin16/e2e_dictenv_scale_v1.py`.
Stage module: `tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_scale_v1.py`.
Delivered design / NumPy reference package (vendored, byte-identical):
`tracks/ksvd/experiments/luyin16/dictionary_scaling/v1_20261001/`
(`zinc_dictionary_scale_plan.md`, `zinc_dictionary_scale_agent_prompt.md`,
`dictionary_scale_reference.py`, `latent_dictionary_bridge_reference.py`,
`validate_dictionary_scaling.py`, `dictionary_scale_audit.json`,
`source_manifest.json`, `SHA256SUMS.json`).

CPU only · official ZINC test never loaded · write-then-follow, no post-hoc
change to any constant, gate, threshold, probe or case in this document.

Background (not a claim).  The immediately previous candidate
`e2e_dictenv_latent_bridge_v1` reached official-valid soup MAE `0.121058`
(borderline, unconfirmed); Sem108 before it reached `0.123705`.  The delivered
scaling plan asks whether *widening the effective environment and relation
capacity inside the same local object, the same dictionary coding rule and the
same static relation form* can move the valid MAE into a band worth further
investment, without touching the fixed Sem108 stem.

---

## 1. Single question

Does one width multiplier applied to the task path only — fusion, shared task
dictionary input/output, static-relation endpoints/encoders and the graph
reader — improve the absolute official-valid soup MAE of the Sem108 local
object, while the structural dictionary, bindings, slot encoders, semantic
interface, static relation scope, global/topology auxiliaries, loss and
training protocol stay unchanged?

## 2. Hypothesis and falsification

**H-scale.**  Full (m = 3, 408,651 parameters) reaches official-valid soup
`M_S <= 0.115` with the shared task dictionary learned and load-bearing.

**Falsified if** `M_S > 0.120` (close this capacity route), or the acceptance
gates fail, or the dictionary does not learn / is not used.  `0.115 < M_S <=
0.120` is recorded as limited signal and this round does **not** widen further.

Small (m = 1) is the frozen latent-bridge candidate (`0.121058`) and is **never
retrained**.  The backgrounds `0.121058` (latent bridge) and `0.123705`
(Sem108) are unmatched context only: Full adds parameters, so a difference is
**not** a causal increment.

## 3. Exactly what is kept and what is multiplied

One implementation, one immutable `ScaleSpec` (`m in {1, 2, 3}`; `m = 2` is
audit-only and never a training candidate).

```
d = 48*m                      # fusion output = decoded environment width
K = 2*d                       # D_L[d, K], V_L[K, d]
p = 16*m                      # static relation endpoints / pair output
fusion        = 446 -> 114*m -> d      (SiLU, original layer count)
pair_projection = Linear(d, p, bias=False)
relation_encoder = _MLPBlock(15, 32*m, p, dropout=0.05)
distance_gate  = Embedding(5, p)
pair_encoder   = _MLPBlock(4*p, 64*m, p, dropout=0.05)
reader         = GenericReader(2*d+1 + 5*(2*p+1) + 32 + 8, (13*m, 13*m))
```

| module | Small m=1 | Full m=3 |
|---|---:|---:|
| fusion | 446→114→48 | 446→342→144 |
| D / V | 48×96 / 96×48 | 144×288 / 288×144 |
| relation endpoint | 48→16 | 144→48 |
| topology relation encoder | 15→32→16 | 15→96→48 |
| static pair encoder | 64→64→16 | 192→192→48 |
| graph reader | 302→13→13→1 | 814→39→39→1 |
| parameters | 106,925 | 408,651 (task dictionary 82,944, body 325,707) |

Kept bit-for-bit (never multiplied): the structural dictionary `D`
(K32/s8/IHT10), common coordinate `U` / `common_rms`, node binding width 96,
edge binding width 48, node slot encoder `96→64→48`, edge slot encoder
`48→48→32`, 3 node shells / 6 edge shell-pairs, the `[Sem108 ; size2]` (110)
interface, the global encoder `62→32→32`, the topology encoder `25→16→8`, the
C6 mask, the five distance buckets and the full-graph auxiliary channels.

* `m = 1` is built through the untouched `LatentBridgeSEM108` constructor:
  identical parameter names, identical initial values and identical RNG
  consumption as `build_latent_bridge_model`.  No "construct new then replace"
  for Small.
* `m = 3` starts from that same Small skeleton and replaces only the seven
  task-path modules in a separate, recorded torch generator stream
  (`scale_seed = 0`); every fixed module keeps its Small initial value.
  Full trains from scratch; no Small checkpoint is loaded.
* No new graph objects, no extra in-graph propagation steps, no `h` residual
  bypass, no extra raw-feature bypass.  All unary and pair consumers read the
  same `E`; relations aggregate to the graph level only, never back to nodes.

## 4. Loss, coding and training

Unchanged parent loss `MAE + H1_LAMBDA * structural_reconstruction`.  The task
dictionary is the existing 16-step unrolled soft-threshold ISTA: row-RMS
normalisation, column-normalised `D_L`, detached conservative step
`1/(1.05*eigmax(Dbar Dbar^T)+lambda2)`, `lambda1 = 0.05`, `lambda2 = 0.01`,
`alpha^0 = 0`, `E = rho * alpha @ V_L`; both `D_L` and `V_L` receive the task
gradient.  Initialisation: private NumPy RNG `seed = 0`; `D_L = [I_d, Q_d]`
(sign-corrected QR of a Gaussian), `V_L = D_L.T`.  The private streams never
consume or reorder the parent torch stream.

No new loss term, no hard top-k, no temperature, no orthogonality penalty, no
dropout/weight-decay search, no learning-rate change.  Adam `lr = 1e-3`,
`wd = 1e-5`, gradient clip 5.0, batch 128, shuffle offsets `+91011` (train) /
`+91012` (eval) from the frozen `zinc_e2e_dictenv_p2_abs` / `p1` constants.

## 5. Acceptance gates (S, zero training, before any candidate training)

One real official-train batch (32 molecules), same seeds, eval mode:

| gate | requirement |
|---|---|
| S1 | Small (m=1) state dict is bit-identical to the frozen `LatentBridgeSEM108` (names, shapes, values); plain and C6-masked forward max diff `<= 1e-5`; parameter count exactly 106,925 |
| S2 | Full parameter count exactly 408,651 and `<= 420k`; captured widths `h/E/alpha/u/pair/reader_input = 144/144/288/48/48/814`; every fixed module parameter and buffer bit-identical to Small |
| S3 | eval-mode containment witness: Small embedded block-diagonally / Net2Wider into Full gives plain and masked prediction max diff `<= 1e-4` |
| S4 | Full identity mode (`lambda1 = lambda2 = 0`, initial `V_L`): `E` vs `h` relative L2 `<= 1e-6`; formal-init `E` vs `h` relative squared error `< 0.05`; repeated coding exact and finite |
| S5 | MAE-only gradients of `D_L`, `V_L`, both fusion layers, pair projection and both pair-encoder layers finite and non-zero; one Adam step moves `D_L` / `V_L`; one small SGD step decreases the same-batch MAE |
| S6 | the task bridge is called exactly once per forward on both the plain and the C6-masked path; no `h` bypass; `E` enters unary pooling and the pair projection input exactly; node relabelling `<= 1e-4`; batch offset `<= 1e-5`; endpoint swap `<= 1e-6`; relation changes cannot change `E` (exact 0); zero-code gives `E = 0` with finite predictions |
| S7 | the official-test blocker raises on `official_test_loaded = True`; no test split is loaded |

The identity mode is audit-only and is never a training or candidate
configuration.  Any failure ⇒ STOP; no training.

## 6. CPU timing gate (before any formal training)

Real forward/backward timing on real official-train batches through the exact
parent loss, optimizer and C6 mask (CPU, 8 threads, no GPU, no remote host).
Steady-state medians exclude the first two batches.  The predicted 320-epoch
total uses the exact official batch counts (10,000 train → 79 batches,
1,000 valid → 8 batches per epoch) times a `1.10` safety factor.

* authorised when `320 * (79 * steady_train + 8 * steady_eval) * 1.10
  <= 14,400 s` (4 h);
* otherwise the round **stops at the completed implementation**: the smoke and
  timing records are delivered and no formal MAE is reported.  This is a
  resource decision, not an algorithmic failure; the width is not silently
  reduced, the ISTA step count is not reduced and the epoch budget is not
  truncated.

A short smoke (3 epochs × 1,024 train molecules, **no valid access**, ≤ 100
optimizer steps) confirms finite loss and live task-path gradients before the
formal run.

## 7. Formal run — exactly one trajectory

`uv run research run zinc_e2e_dictenv_scale_v1 --study zinc-context-gap --mode
screen`.

Full only (m = 3), from scratch, seed 0, scale_seed 0; official train 10,000 /
valid 1,000; **test never instantiated**; CPU 8 threads; batch 128; Adam
`lr=1e-3`, `wd=1e-5`, clip 5.0; 320 epochs; frozen fixed LR; Top-5 parameter
soup (full-model average including `D`, `D_L`, `V_L`; fixed buffers must
agree).  No Small retraining, no second seed, no matched control arm, no
ensemble, no dense/random/coarse-group arm, no official Test read.

Interrupt ⇒ `INCOMPLETE`, recorded as such, not as a failure.  Diagnostics
(1/80/160/240/320) never gate or stop the run; only numerical errors or real
resource anomalies can terminate it.

Endpoint reporting evaluates the soup model on the official train split in eval
mode with the identical loader protocol (same mask, batch size, no shuffling)
so the reported train and valid MAE share one measurement convention; this is
an endpoint diagnostic and never a selection criterion.

## 8. Resource-decision bands (frozen before training)

| Full soup valid MAE `M_S` | interpretation |
|---|---|
| `<= 0.110` | strong single-seed absolute signal; record as candidate, no automatic extra run |
| `0.110 – 0.115` | promising; a next round of paired seeds / training-recipe validation is justified |
| `0.115 – 0.120` | limited signal; this round closes without further widening |
| `> 0.120` | no clearly better region; the capacity route is closed |

These are budget gates, not significance tests.  If Full only lowers train MAE
while valid does not move, more `K` or width is not supported; if both fall
together, paired seeds / recipe validation become the next question.

## 9. Frozen inference probes (soup state only, no retraining)

| probe | content |
|---|---|
| **P1** | zero the whole task-dictionary code (`alpha -> 0`, hence `E -> 0`) |
| **P2** | within-molecule code permutation, 5 seeds `101/202/303/404/505` (row-local bridge ⇒ identical to permuting the bridge input within each molecule) |
| **P3** | keep the trained encoder and head, reset only `D_L` / `V_L` to the initial frame |
| **D1** | code density distribution (mean / p50 / p95 per-object nonzero count over 288 entries) — variable density, never a fixed `l0` |
| **D2** | low-dimensional reconstruction `||x - alpha Dbar^T||^2` (diagnostic only) |
| **D3** | MAE-only task gradient norms of `D_L` / `V_L` and Frobenius movement from the frozen frame |
| **D4** | readout block layout derived from `d` / `p` (never the frozen 48/16 indices) |

Every intervention reports both `delta_mae` and the true
`delta_pred_rms = sqrt(mean((intervened - baseline)^2))` over the identical
evaluation loader.  The probes establish that the channel carries information;
they do **not** establish that dictionary learning beats a matched trained
control.

## 10. Stop discipline

After the single 320-epoch Full run and the frozen probes, the round **stops**.
No seed 1, no rescue run, no `K` / `s` / `lambda` / steps / width / LR change,
no matched control arm added after seeing the result, no official-test read.
A follow-up (paired seeds, matched control, recipe validation) requires a new
pre-registration and only after `M_S <= 0.115`.  Reaching `≤ 0.110` is recorded
as a candidate, not as an automatic next run.

## 11. Result layout

`tracks/ksvd/results/e2e_dictenv_scale_v1/`

```
historical_references.json
audit/{audit_reuse, sem108_identity, anchor_relation, t1_block_ablation, audit_decision}.json
preflight.json  preregistration_snapshot.json  parameter_audit.json
correctness.json  timing.json  smoke.json
run_seed0.json | train_skipped.json
curve_seed0.csv  soup.json
mechanism/{bridge_probes, dictionary_health}.json
summary.json  REPORT.md  DECISION.md
```

Every JSON carries `protocol_version`, `git_commit`, `device = cpu` and
`official_test_loaded = false`.
