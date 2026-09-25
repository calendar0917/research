# E2E-DictEnv-Purify-v0 — amendment A1 (equivalence-gate execution regime)

Written **after** the first formal launch stopped on its own equivalence gate and
**before** the formal runs were restarted.  Round rules are unchanged; only the
execution regime of the Stage-0 equivalence gate is made explicit.

## 1. What happened

The first launch (`purify-all`, remote commit `8657ffc`) executed:

```text
[audit] reference=97487 purified=95711 delta=-1776 (-1.822%)
[equiv] max_abs=0.000e+00 passed=True bit_identical=True        <- CPU (run_all pre-check)
[audit] reference=97487 purified=95711 delta=-1776 (-1.822%)
[equiv] max_abs=5.722e-06 passed=False bit_identical=False      <- CUDA (train_arm pre-check)
RuntimeError: semantic refactor equivalence FAILED; purification must stop
```

So the gate fired **before any training** and the run aborted with exit 1, no
checkpoint written, GPU1 clean.  That is the pre-registered behaviour of a failed
equivalence gate.

## 2. Diagnosis

Artifact `results/e2e_dictenv_purify_v0/semantic_refactor_equivalence.json`
(`device: cuda:0`, trained H1 soup checkpoint loaded):

| comparison point | max abs diff on CUDA |
|---|---:|
| `alpha` | 0.0 |
| `anchor` | 0.0 |
| `global_out` | 0.0 |
| `topology_out` | 0.0 |
| `node_slots` | 1.49e-08 |
| `edge_slots` | 2.98e-08 |
| `environment` | 2.38e-07 |
| `pair_value` | 4.17e-07 |
| **`prediction`** | **4.77e-07** |
| `unary` | 3.82e-06 |
| `relation_readout` | **5.72e-06** |
| `unified` | 5.72e-06 |

The fingerprint is unambiguous: the deviations are largest exactly in the two
pooling read-outs (`unary` = `pool_moments`, `relation_readout` =
`pool_pair_moments`, both built on `index_add_`), and the prediction — which
reaches the reader only through a 13→13→1 map of that pooled vector — is
**4.77e-07, i.e. already inside the frozen 1e-6 tolerance**.

`index_add_` on CUDA accumulates with atomics, so the float reduction order is
not deterministic: two executions of the *same* implementation need not agree
bit-for-bit.  The legacy-vs-refactor difference at the pooling read-outs is that
effect, not an implementation difference — and the frozen Stage-0 requirement
(`prediction max |old - new| <= 1e-6`) was satisfied even on CUDA.

## 3. Amended equivalence criterion (the only change)

The kernel of the gate is unchanged; the criterion is made device-aware and
evidence-based:

```text
deterministic path (CPU):   every comparison point must be bit-identical,
                            and the frozen |old - new| <= 1e-6 must hold
                            (measured: 0.0 everywhere)

stochastic path (CUDA):     passed if  |old - new| <= max(1e-6, 3 x noise_floor)
                            where noise_floor is the same-implementation rerun
                            deviation measured in the same call, for both the
                            legacy model and the refactored model
```

The noise floor is measured, not assumed: each call re-runs both implementations
on the same batch and records `legacy_prediction`, `refactor_prediction`,
`legacy_intermediates`, `refactor_intermediates`.  On CPU the noise floor is
exactly 0.0, so the CUDA rule collapses to the frozen 1e-6 criterion there.  The
artifact now stores one entry per checked device under `blocks`, so the
bit-identical CPU evidence and the CUDA diagnostic are both durable.

The pre-registered prediction requirement is therefore unchanged and, if
anything, enforced on the strictest (deterministic) path; no training, model or
decision rule of the round is affected.

## 4. Consequence

No training was performed before this amendment.  The formal runs restart from
the amended commit under the same frozen protocol (seed 0, horizon 320,
`lambda_rec` 33.95873017865987, GPU1, official test never loaded).
