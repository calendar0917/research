# Compact-v4-smallhead — 2-round Weight-Tied Recurrent Pair–Centre

> **Question.** compact-v4 computes the structural pair relation `q_ij` exactly
> once. The centre states absorb `q^(0)`, but the updated context never acts back
> on the pair relation. Is one of the current bottlenecks the *one-shot*
> nature of the structural relation computation?
>
> **Verdict.** **RECURRENCE REPLICATED (2/2 seeds) but the gain is mostly
> depth, not relation persistence.** The 2-round weight-tied variant adds
> **exactly zero parameters** (82,115 = 82,115), reuses the shared pair encoder
> and centre update in both rounds, and recomputes `q^(1)` from the updated
> `h^(1)`. Official-valid MAE improves on both seeds (seed0 `+0.006958`,
> seed1 `+0.004619`, mean `+0.005789`, gate `+0.003`); the frozen one-shot
> official **test** improves on both seeds (mean `+0.004505`:
> `0.119456 → 0.114951`). **Decisive control (seed0):** a depth-only stale-`q^(0)`
> variant with the *same* two centre updates and *same* 82,115 params reaches
> `0.140504` vs baseline `0.145334` and refresh `0.138376`. So `+0.004830` of the
> `+0.006958` total gain comes from the **extra centre depth alone**, and only
> `+0.002128` is attributable to recomputing the relation. At matched epochs the
> refresh-over-depth delta is not even consistently positive. **The specific
> hypothesis that the bottleneck is the one-shot relation computation is
> therefore only weakly supported; the dominant effect is more sequential
> centre processing.** A second caveat: seed1's gain appears only *after* its
> baseline early-stopped (best epoch 234/240, horizon warning).
>
> Code: `tracks/ksvd/experiments/luyin16/zinc_compact_v4_recurrent_pair_centre.py`.
> Tests: `tracks/ksvd/tests/test_compact_v4_recurrent_pair_centre.py` (19 pass).
> Results: `tracks/ksvd/results/compact_v4_recurrent_pair_centre/`.

---

## 1. Motivation and relation to P2

P2 (one-shot relation refresh) replaced the final pair summary `q^(0)` with one
parameter-shared refresh `q^(1) = Q(P(h^(1)), r)` but explicitly did **not** let
`q^(1)` feed a further centre update (`q^(1)` never produced `h^(2)`). On the
99,613-param compact-v4-hinge backbone P2 was a sub-threshold positive
(`Δ = +0.001963`, gate `+0.004`).

This experiment asks the strictly stronger question:

> if the relation state is allowed to **persist across rounds** — recomputed
> from the updated centres and fed back into another centre aggregate — does the
> bottleneck move?

The variant is built on the current best model, **compact-v4-smallhead**
(82,115 params), and is deliberately parameter-neutral so any gain cannot be
attributed to extra capacity.

## 2. Exact computation

```
h^(0)   = patch_encoder(...)                                       R^48
q^(0)   = Q(P(h^(0)_i), P(h^(0)_j), r_ij, gate_ij)                 R^16
A^(0)   = per-(centre,bucket) [mean(q0) 16 ; std(q0) 16 ; log1p(cnt) 1] x 5
h^(1)   = h^(0) + U([h^(0) ; A^(0)])                               R^48
q^(1)   = Q(P(h^(1)_i), P(h^(1)_j), r_ij, gate_ij)   <-- recomputed, shared
A^(1)   = per-(centre,bucket) moments of q^(1)
h^(2)   = h^(1) + U([h^(1) ; A^(1)])                               R^48
R       = [unary_moments(h^(2)) 97 ; pair_moments(q^(1)) 165 ; global 32 ; topology 8] = 302D
```

`P = pair_projection`, `Q = pair_encoder`, `U = center_update` are the **same
tensor objects in both rounds** (weight tying). The relation descriptor
`relation_encoder(pair_relation)` and the distance gate are static inputs and are
computed once. The small raw head `GenericReader(302,(13,13))` is unchanged.

## 3. Parameter accounting / weight tying

| model | total | head | Δ params |
|---|---:|---:|---:|
| matched baseline (compact-v4-smallhead) | 82,115 | 4,135 | — |
| recurrent pair–centre | 82,115 | 4,135 | **0** |

Forward invocation counts per batch (proves tie + two rounds):

| module | baseline | recurrent |
|---|---:|---:|
| `pair_projection` | 2 | 4 |
| `pair_encoder` | 1 | 2 |
| `center_update` | 1 | 2 |

## 4. Sanity / mechanism checks (all 24 pass)

`results/compact_v4_recurrent_pair_centre/sanity.json`:

* one-round recurrent pathway is **bit-identical** to the parent `encode`;
* at init (`U` is zero-initialised) the 2-round model is **bit-identical** to the
  baseline function (`max|ΔR| = 0.0`), so both start from the same function;
* forcing the centre update to zero collapses the model back to baseline
  exactly (`max|ΔR| = 0.0`);
* with a perturbed `U`, `q^(1) ≠ q^(0)` (max drift `0.266`, cosine `0.9893`),
  `h^(1) ≠ h^(0)`, `h^(2) ≠ h^(1)`, and `∂q^(1)/∂h^(1) ≠ 0`
  (`max 19.64`) — round 2 genuinely depends on the updated centre state;
* the trailing `log1p(count)` block of the centre context is **bit-identical**
  between rounds, and the reconstructed incidence counts match the pair index
  exactly (mask / bucket / batch indexing is correct in round 2);
* state-dict shapes, `R` dimension (302) and head are unchanged.

## 5. Training protocol (inherited verbatim)

Adam, lr `1e-3`, weight decay `1e-5`, batch `128`, max epochs `240`,
patience `40`, no scheduler, gradient clip `5.0`, L1 loss, best official-valid
checkpoint, single stage. Matched baseline = the same code path /
`build_smallhead` / shared initial tensors; it reproduced the existing
smallhead runs **exactly** (seed0 `0.145334@176`, seed1 `0.138059@143`).
Official test is fit on official train only (identical to selection), one
frozen evaluation of the selection checkpoints (`terminal_test.json`).

## 6. Results

### 6.1 Validation

| seed | baseline valid | recurrent valid | Δ (base − rec) | best epoch (base / rec) | epochs (base / rec) |
|---:|---:|---:|---:|:---:|:---:|
| 0 | 0.145334 | 0.138376 | **+0.006958** | 176 / 199 | 216 / 239 |
| 1 | 0.138059 | 0.133440 | **+0.004619** | 143 / 234 | 183 / 240 |
| mean | 0.141697 | 0.135908 | **+0.005789** | — | — |

Pre-registered gate `Δ ≥ +0.003` passes on both seeds.

### 6.2 Epoch-matched caveat

| cutoff | seed0 Δ | seed1 Δ |
|---:|---:|---:|
| ≤ 100 | +0.003742 | −0.006394 |
| ≤ 143 | +0.008398 | −0.005222 |
| ≤ 183 | +0.004422 | +0.001197 |
| final (≤ n) | +0.006958 | +0.004619 |

Seed0 is consistently better. Seed1 is **worse at every matched epoch up to the
baseline's own early stop** and only overtakes later; the final seed1 best epoch
is 234/240 (`horizon_boundary_warning = True`). This is the main threat to a
clean mechanism claim.

### 6.3 Official test (frozen one-shot, selection checkpoints)

| seed | baseline test | recurrent test | Δ (base − rec) |
|---:|---:|---:|---:|
| 0 | 0.121821 | 0.119353 | +0.002468 |
| 1 | 0.117092 | 0.110549 | +0.006543 |
| mean | 0.119456 | 0.114951 | **+0.004505** |

Both seeds keep the valid direction on test (so seed1's late gain is not a
validation-only artefact), but the mean recurrent test MAE `0.11495` is still
above the `<0.10` target.

### 6.4 Depth-only stale-`q^(0)` control (seed0)

The control has the **same 82,115 parameters and the same two centre updates**
as the recurrent model; only round 2 reuses the stale `q^(0)` instead of
recomputing `q^(1)`.

| model (seed0) | valid MAE | best epoch | Δ vs baseline |
|---|---:|---:|---:|
| matched baseline (1 centre update, `q0`) | 0.145334 | 176 | — |
| stale control (2 centre updates, `q0`) | 0.140504 | 154 | **+0.004830** |
| recurrent (2 centre updates, `q1` refreshed) | 0.138376 | 199 | +0.006958 |

Decomposition of the total seed0 gain:

* extra centre depth alone: **`+0.004830`** (≈69% of the total)
* relation refresh over depth: **`+0.002128`** (≈31% of the total)

Epoch-matched decomposition (`Δdepth = baseline − stale`,
`Δrefresh = stale − recurrent`):

| cutoff | Δdepth | Δrefresh |
|---:|---:|---:|
| ≤ 100 | +0.000525 | +0.003217 |
| ≤ 140 | +0.009146 | +0.000888 |
| ≤ 154 (stale best) | +0.007443 | **−0.000927** |
| ≤ 176 | +0.004830 | **−0.000407** |
| final | +0.004830 | +0.002128 |

`Δrefresh` is only positive early and at the final (post-stale-early-stop)
checkpoint; at the stale model's own best epoch the recurrent model is
*slightly worse*. The refresh benefit is small and late, not a stable per-step
representation gain.

## 7. Interpretation

* The 2-round recurrence **does help** under the pre-registered protocol
  (valid mean `+0.0058`, frozen test mean `+0.0045`, 2/2 seeds).
* **But the dominant cause is the extra centre update, not relation
  persistence.** In the seed0 control, `+0.004830` of the `+0.006958` gain is
  already obtained by a stale-`q^(0)` 2-round model; only `+0.002128` is
  attributable to recomputing `q^(1)` from `h^(1)`. At matched epochs the
  refresh-over-depth delta is not consistently positive (negative at epochs
  ≤154/≤176).
* **Therefore the original hypothesis — "the bottleneck is the one-shot
  structural relation computation" — is only weakly supported.** The honest
  falsification verdict is: *more sequential centre processing is the main
  effect; persistent relation refresh adds a small, late, non-robust increment.*
* The seed1 late-convergence caveat (gain only after the baseline early-stops,
  horizon warning) is a second independent reason not to over-claim the
  refresh mechanism.
* The next cheapest disambiguation is **not** to stack another module. It is the
  pre-registered **pair-to-pair interaction through a shared intermediate `k`**,
  which would test whether relations can exchange information directly rather
  than only through the centre aggregate, and a matched-horizon protocol that
  separates depth/late-convergence from representation.

## 8. Recommendation

Because the recurrence result is positive only *as extra depth*, the specific
persistence claim should be treated as **weak/no-go**. Do **not** add more
centre rounds, attention, path features, extra statistics, or per-round weights
to this variant. The single most informative next experiment is the
pre-registered **shared-intermediate-`k` pair-to-pair interaction** test
(`q_ij` updated through a shared latent `k` rather than only through the centre
aggregate), with the same parameter-neutral, matched-baseline, matched-depth
and matched-horizon discipline. If that also fails to beat the stale/depth
control, the persistence line should be closed in favour of depth/regularisation
work.

## 9. Follow-up mechanism validation (stale seed1 + late-refresh seed0)

Two low-cost controls were added to separate three explanations of the T=2 gain:
(a) centre refinement / computation depth, (b) the relation refresh itself, and
(c) the refreshed relation *feeding back* into the centre. No architecture sweep
and no official test was run.

### 9.1 Task A — stale seed1 replication

Same protocol, seed1, `recurrence_mode="stale"` (two centre updates, `q^(0)`
reused in round 2), 82,115 params.

| model (seed1) | valid MAE | best epoch | Δ vs baseline |
|---|---:|---:|---:|
| matched baseline | 0.138059 | 143 | — |
| **stale** (2 updates, stale `q0`) | **0.145111** | **131** | **−0.007052** |
| recurrent (2 updates, refreshed `q1`) | 0.133440 | 234 | +0.004619 |

**The seed0 ordering does not replicate.** On seed0 the stale control sat
between baseline and recurrent (`baseline > stale > recurrent` by MAE); on
seed1 the stale control is clearly **worse than the baseline**
(`stale > baseline > recurrent`). The seed0 "most of the gain is just extra
centre depth" reading is therefore not robust: the stale/depth-only effect is
seed-specific and can be negative, while the recurrent gain over baseline is
positive on both seeds (`+0.006958`, `+0.004619`).

### 9.2 Task B — one-shot late-refresh variant, seed0

Computation (all tensors weight-tied, **zero** new parameters):

```
h^(0)  = patch_encoder(...)
q^(0)  = Q(P(h^(0)_i), P(h^(0)_j), r_ij, gate_ij)
A^(0)  = pool_pairs_to_centres(q^(0))          # formed exactly once
h^(1)  = h^(0) + U([h^(0) ; A^(0)])
h^(2)  = h^(1) + U([h^(1) ; A^(0)])            # SAME A^(0), no refresh
q^(1)  = Q(P(h^(2)_i), P(h^(2)_j), r_ij, gate) # one late refresh
R      = [unary(h^(2)) ; pair_moments(q^(1)) ; global ; topology]  # unchanged head
```

The centre path is **bit-identical to the stale control** (same `A^(0)` tensor
replayed twice, `_pool_pairs_to_centres` invoked once); only the pair readout
consumes `q^(1)`. This isolates "a better final pair representation" from
"the refresh re-entering the centre".

| model (seed0) | valid MAE | best epoch | params |
|---|---:|---:|---:|
| matched baseline | 0.145334 | 176 | 82,115 |
| stale (2 updates, stale `q0`) | 0.140504 | 154 | 82,115 |
| **late-refresh** (stale centres + refreshed `q1` readout) | **0.142952** | **150** | **82,115** |
| recurrent (refreshed `q1` fed back + readout) | 0.138376 | 199 | 82,115 |

Key deltas: `late − stale = −0.002448` (late-refresh is **worse** than stale),
`recurrent − late = +0.004576` (recurrent clearly better).

### 9.3 Sanity checks

`results/compact_v4_recurrent_pair_centre/sanity_late_refresh.json` (21/21 pass):

* parameter neutral: 82,115 = baseline = stale = recurrent; head 4,135; state-dict shapes identical;
* both centre updates replay the **same** `A^(0)` object
  (`last_center_context1 is last_center_context0`, exact numeric equality);
* `_pool_pairs_to_centres` is invoked **once** for late-refresh (= baseline),
  twice for stale/recurrent — `q^(1)` never re-enters centre aggregation;
* `q^(1)` is recomputed from `h^(2)`: `q^(1) ≠ q^(0)`, `∂q^(1)/∂h^(2) ≠ 0`, and a
  manual `Q(h^(2))` recomputation matches the stored `q^(1)` exactly;
* weight tying intact: `pair_projection` ×4, `pair_encoder` ×2, `center_update` ×2;
* centre path (`h^(1)`, `h^(2)`, `A^(0)`, `A^(1)`) is bit-identical to the stale
  control; switching the same weights to `stale` mode reproduces stale exactly;
* forcing `U = 0` collapses late-refresh to the baseline function exactly.

Existing `sanity.json` (24/24) and `tests/test_compact_v4_recurrent_pair_centre.py`
(25 pass) confirm the baseline / stale / recurrent paths are unchanged.

### 9.4 Interpretation and next step

* `late-refresh` does **not** beat `stale` (it is slightly worse), and
  `recurrent` beats `late-refresh` by `+0.0046`. The refreshed relation is not
  valuable merely as a better final pair representation; its benefit requires
  re-entering the centre through the second update.
* Combined with Task A (stale is not robustly positive across seeds), the
  evidence points to the **feedback / persistence** direction rather than to
  adding centre depth.
* Recommended next experiment: **persistent relation / direct pair-to-pair
  composition** (parameter-neutral, matched baseline, matched depth, matched
  horizon), not a K=3 centre-refinement sweep.

## 10. Direct pair-to-pair composition (shared intermediate `k`)

Follow-up to §9: allow relations to interact directly through a shared
intermediate `k` *before* the refreshed relation re-enters the second centre
aggregation.

### 10.1 Computation

```
h0 -> q0 -> A0 -> h1 -> q1 = Q(h1)          # first recurrent round unchanged
q1*_ij = q1_ij + phi( q1_ij , mean_k psi(q1_ik, q1_kj) )
q1* -> second centre aggregation -> h2 -> existing readout/head
```

* `psi`: `Linear(48 -> 24) -> ReLU -> Linear(24 -> 16)` on
  `[q_ik + q_kj ; |q_ik - q_kj| ; q_ik * q_kj]`;
* `phi`: `Linear(32 -> 24) -> ReLU -> Linear(24 -> 16)` on `[q_ij ; mean_k psi]`,
  last layer zero-initialised so the model starts exactly at the recurrent
  function;
* `q1*` replaces `q1` for **both** the second centre aggregation and the pair
  readout; `P/Q/U` remain weight-tied; no attention / multi-head / new relation
  descriptor / tokenizer / readout / head / optimizer change.

### 10.2 Shared-`k` indexing

Pairs inside one graph are a complete graph enumerated as `(a, b)`, `a < b`, so
the local pair index is `p(a,b) = a*n - a(a+1)/2 + (b-a-1)`. For each graph
size `n` a local triple pattern `(left, right, target)` is cached once and
reused across steps; per forward the only work is offsetting by the per-graph
pair offset and one gather/scatter. No dense `(i,j,k)` tensor is materialised;
only real `(i,k)`/`(k,j)` pairs are enumerated. The implementation was timed
first and was **not** judged on the naive dense construction.

Indexing scale (a 128-molecule batch of official-valid):

| quantity | value |
|---|---:|
| pairs | 34,876 |
| triples `(i,k,j)` | 804,384 |
| triples per pair | 23.06 |
| avg pairs / graph | 264.8 (valid), 266.8 (train) |
| avg triples / graph | 6,030 (valid), 6,106 (train) |
| max triples / graph | 23,310 (`n = 37`) |

### 10.3 Parameters

| model | total | head | added |
|---|---:|---:|---:|
| recurrent | 82,115 | 4,135 | — |
| pair-to-pair `psi` | 1,576 | — | 1,576 |
| pair-to-pair `phi` | 1,192 | — | 1,192 |
| pair-to-pair total | **84,883** | 4,135 | **+2,768** |

`sanity_pair_to_pair.json`: **17/17 pass** — init and forced-zero both collapse
to recurrent exactly; `q1* != q1` with non-zero `d(composition)/d(gathered q)`;
no cross-graph leak (0 rows); endpoints reconstruct `(i,k,j)` exactly; per-pair
counts equal `n-2`; residual shapes correct; no NaN/Inf; recurrent path intact.

### 10.4 Validation results

| seed | baseline | recurrent | pair-to-pair | Δ (rec − p2p) | p2p best epoch |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.145334 | 0.138376 | **0.136987** | **+0.001388** | 213 |
| 1 | 0.138059 | 0.133440 | **0.133374** | **+0.000066** | 167 |
| mean | — | — | — | **+0.000727** | — |

Matched-epoch behaviour:

* seed0: pair-to-pair is *worse* at every matched cutoff up to epoch ~176 and
  only turns positive at epoch 183; the win is late.
* seed1: pair-to-pair is *better* at every early/mid cutoff (+0.003 to +0.009)
  but the recurrent model catches up by its late best (0.13344 @234), so the
  final delta is ~0.00007.

The two seeds therefore show opposite matched-epoch profiles, and the final
mean delta `+0.00073` is inside the pre-registered `±0.001` no-signal band.

### 10.5 Cost

| seed | recurrent wall | p2p wall | epoch ratio | p2p peak RSS |
|---:|---:|---:|---:|---:|
| 0 | 1,777 s | 7,499 s | ~4.2x | ~3.01 GB |
| 1 | 1,849 s | 6,355 s | ~3.4x | ~3.01 GB |

Recurrent peak RSS is ~2.23 GB, so pair-to-pair costs ~1.35x peak memory and
~3.5–4.2x wall-clock.

### 10.6 Verdict

`decision_pair_to_pair_replication.json`: **`NO_INDEPENDENT_SIGNAL`**.
Seed0 was a weak positive, seed1 was a near-exact tie; the mean is inside the
noise band. Because the operator also carries +2,768 extra parameters, the
absence of a final gain is not rescued by capacity. Per the pre-registered
rule, **stop: do not run hyper-parameter sweeps and do not add more complex
pair-to-pair variants (attention, pair-to-pair layers, hidden-size or
aggregation sweeps, T=3/T=4, path/cycle features).**

