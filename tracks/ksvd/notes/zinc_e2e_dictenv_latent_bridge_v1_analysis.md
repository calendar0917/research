# Analysis — ZINC E2E-DictEnv-Latent-Bridge-v1 (SEM108 + shared low-dimensional task dictionary, seed 0, local CPU)

Round: `zinc-e2e-dictenv-latent-bridge-v1` (task package
`/home/calendar/Downloads/zinc_dictionary_next_step/`, audit revision
`65d4b8fb82261e9ec0a94bbf02d7ccc36e14219f`).

Formal screen run: `20261001-220925-9bbcc5c9` (promoted record
`records/runs/20261001-220925-9bbcc5c9.json`), formal-run commit
`9d4cbedf69adf6e15d4dbfa2cdefab65306b8f61`, preregistration commit `039355e`
(runner mapping fix `34d7c0e`, endpoint patch `9d4cbed`), preregistration sha
`00faf4d52b98` recorded by preflight.  Official test never instantiated
(`official_test_loaded = false`).

## 1. Verdict

`M_S = 0.121058` (soup valid MAE) → band
**LATENT_BRIDGE_BORDERLINE** (0.120 < M_S ≤ 0.1233), single seed-0 trajectory,
no matched control, no official-test read.

- best single epoch `0.129975` @ 309, Top-5 soup members
  [285, 296, 303, 307, 309] with member valid MAE
  0.131193 / 0.131298 / 0.131306 / 0.130226 / 0.129975.
- first/last valid `0.588556` / `0.140250`; training-loop final train
  `0.085155`, minimum `0.080422` (epoch 310).
- same-protocol (eval-mode, no-shuffle) soup train MAE `0.055133`, train→valid
  gap `+0.065926`.
- wall `2434.8 s` train (`7.609 s/epoch`) + `61.3 s` interventions/analysis;
  peak RSS `2313.1 MB`.

The candidate is *not* confirmed by its own preregistration: the confirmation
rule required `M_S ≤ 0.120` **and** a learned dictionary before considering
seed 1 and matched controls.  It is also not terminated by the stop rule
(`M_S ≤ 0.1233` triggers "current candidate unworkable"; `0.121058` sits
0.001058 above the promising edge and 0.002242 below the stop edge).  The round
therefore closes at "borderline, unconfirmed" — see section 6.

## 2. What this round tested

The previous round's hierarchical static relation dictionary was live but lost
to the Sem108 background (soup 0.182343 vs 0.123705) on absolute performance.
This round kept the Sem108 local encoding, static relations and readout exactly
as they are and inserted one shared, low-dimensional task dictionary at the
Sem108 fusion output: `h[48] -> rho-normalised 16-step unrolled ISTA -> α[96]
-> E = rho · α V_L [48]`, with `D_L [48,96]` / `V_L [96,48]` (9,216 new
parameters, body 97,709 unchanged).  No graph index, no message passing, no
residual bypass, no new raw features, no high-dimensional K-SVD, no refitting
of Joint709.

Deliverables: bridge core `experiments/luyin16/e2e_dictenv_latent_bridge_v1.py`,
stage driver `experiments/luyin16/zinc_e2e_dictenv_latent_bridge_v1.py`,
control-plane runner `runners/zinc_e2e_dictenv_latent_bridge_v1.py`, config
`configs/luyin16/zinc_e2e_dictenv_latent_bridge_v1.yaml`, vendored reference
package `experiments/luyin16/latent_bridge_reference/v1_20261001/` (sha256s in
its README), 10 unit tests, and the pre-registration
`notes/zinc_e2e_dictenv_latent_bridge_v1_preregistration.md`.

## 3. A — zero-training implementation acceptance (`dev/acceptance.json`)

All 13 gates passed on one real official-train batch of 32 molecules before any
training; scope line: implementation-level acceptance, no training, no valid
selection, test never loaded.

- decoder identity (`V_L = [I; Q]`, `D_L = [I; Q]ᵀ`, `λ1 = λ2 = 0`):
  `E` equals `h` with relative L2 `1.23e-07`; model prediction max diff
  `8.3e-07` vs the Sem108 parent on the same batch.
- formal-init check: the composite `D_L∘V_L` maps `h` to itself with relative
  squared error `0.00236` at the frozen unit-column init.
- task gradients at init: `D_L 1.03`, `V_L 1.07` (MAE-only); an Adam delta
  moves parameters 0.068; a small SGD step lowers loss `1.604 → 0.963`.
- the bridge is called exactly once on both the unary and pair path; unary/pair
  outputs match the Sem108 parent exactly; within-graph relabelling changes
  nothing (`1.57e-05`), batch offsets `4.5e-08`, endpoint swap `0`.
- relation readout does not write back (delta `0`); zeroing the bridge code
  makes `E = 0` while the surrounding deltas stay live; the official-test
  blocker raises.

Historical note: this stage ran before the endpoint patch and is stamped
`git_commit 65d4b8f` — the accepted behaviour it certifies is unchanged by the
patch (which touched only the analysis-stage reporting).

## 4. B — 40-epoch paired short trainability (scratch/dev, no claim)

Same data order and seeds, 1,024 train molecules, 40 epochs, no valid/test:
parent final `0.362907` (min 0.326184, 0.576 s/epoch) vs candidate final
`0.317501` (0.771 s/epoch); finiteness, learned-parameter and shared-init
identity checks passed.  This is scratch/dev evidence only: it shows the
inserted bridge trains, not that it is better.

## 5. C — formal 320-epoch screen and mechanism

Endpoint (frozen soup): see section 1.  Bridge mechanism on the frozen soup
state, inference only (`mechanism/bridge_probes.json`):

| probe | delta MAE | delta pred RMS |
|---|---|---|
| zero the bridge code (`E = 0`) | +1.185274 | 1.696537 |
| within-molecule code permutation (5 seeds) | +0.288035 | 0.473635 |
| reset `D_L`/`V_L` to frozen init (encoder/head kept) | +1.231997 | 1.809348 |

- learned: `D_L` / `V_L` Frobenius movement `5.7622` / `5.6371` from the frozen
  frame, with non-zero MAE-only task gradients `0.2694` / `0.2401` at the
  endpoint (0.72 / 0.76 already at epoch 1).
- code density at the endpoint: mean non-zero `89.10 / 96`
  (fraction 0.928, p50 89, p95 94, max 96) — variable density, no fixed l0, no
  top-k.  The soft-threshold regime (`λ1 = 0.05` on ρ-normalised coordinates)
  barely zeros atoms: the code is not sparse at convergence, so this round
  calls the bridge a **low-dimensional task dictionary with soft-threshold
  coding**, not a sparse code.
- diagnostic low-dimensional reconstruction `h ≈ E`: mean relative `0.0048`,
  p95 relative `0.01085`; the reconstruction term is never optimised.
- structural dictionary health (unchanged sub-model): active atoms 31/32,
  effective atoms 18.82, top-1 usage share 0.8190, valid reconstruction
  relative residual 0.9770.

Interpretation.  The intervention deltas are large, but they only establish
that the inserted channel carries a lot of prediction (by construction, zeroing
it removes the whole local encoding from the readout) — they do not show that
dictionary learning beats a matched trained alternative, and no such control
was run.  The "learned" statement is real (both factors moved and receive
non-zero MAE-only gradients), but 92.8 % of code entries are active, so the
dictionary is behaving as a near-dense low-rank re-encoding at this λ regime.

## 6. What this round does and does not claim

- Does claim: one frozen architecture reached `M_S = 0.121058` (borderline
  band) with a learned, load-bearing bridge under the local CPU seed-0
  protocol; the mechanism numbers in section 5 are exact endpoint measurements.
- Does **not** claim any causal gain over Sem108 (`0.123705` is an unmatched
  background: +9,216 parameters, different model, single seed), any sparse
  coding (measured code density contradicts it), or any generalisation benefit
  (same-protocol train `0.055133` vs valid `0.121058`; training-loop fit at
  epoch 320 is essentially identical to Sem108's `0.085076`).
- Does not claim significance: one seed, no seed-variance estimate, and the
  distance to both band edges (0.001058 / 0.002242) is far below anything this
  design could resolve.

## 7. Round outcome and next step

Per the pre-registration's confirmation rule, a borderline endpoint closes the
round without seed 1, without a matched control, and without tuning
`K / s / λ / steps`.  The two honest readings are: the inserted low-dimensional
task dictionary is *usable* (trained, load-bearing) and *not demonstrated to be
valuable* (borderline absolute, near-dense codes, no control).  A future round
may revisit only through a new pre-registration; the two candidates for such a
round are (a) a confirmatory design for this exact candidate (seed 1 plus a
parameter-matched dense/random adapter and paired seeds) to test whether the
borderline band and the +0.00265 background difference are real, or (b) a
different local object / learning objective, per the task package's section 6
warning against iterating `K / s / λ` on the same representation.  Neither is
authorised by this round.

## 8. Provenance and limitations

- Runs: dev acceptance (audit revision `65d4b8f`), short trainability (dev
  evidence), formal screen `20261001-220925-9bbcc5c9`; two earlier local
  attempts of 2026-10-01 were discarded before training or interrupted for the
  endpoint patch (`22cdedb9` completed with `PREPARE_OR_INCOMPLETE` due to a
  runner stage-mapping bug; `d50e4759` was stopped at ~epoch 3 for the
  same-protocol train-MAE endpoint patch).  Neither contributed results.
- `references` stage reconfirmed the Sem108 background `0.123705`; `audit`
  decision `PROCEED` with `T1_BLOCK_AUDIT_UNAVAILABLE` (T1 state tensors are
  not durable; recorded as a limitation, no T1 retraining); `preflight`
  parameter audit exact (body 97,709 + bridge 9,216 = 106,925) and prereg sha
  enforced; `correctness` all gates passed; `smoke` passed (8 epochs,
  best valid 0.665411).
- Historical numbers (T1 0.125765, FINAL-CLEAN-C6 0.128499, CSSD-q1 0.130028,
  P1-sparse 0.131975, Sem108 0.123705) are background only: different
  architectures, parameter counts and protocols.
- CPU regime (`runtime.device=cpu`, 8 threads), one geometry only
  (`d = 48`, `K = 96`, 16 unrolled ISTA steps, `λ1 = 0.05`, `λ2 = 0.01`), one
  seed, one authoring machine; no matched arms; the official test split was
  never instantiated.
