# FEC-S0 — REPORT

**Round:** FEC-S0 (Factorized Environment-Composition S0), zero-training
function-preserving equivalence audit of historical strict-static S0.
**Study:** `zinc-context-gap`. **Protocol:** `fec_s0`.
**Lineage:** `d4de88e`, clean worktree. **Device:** CPU only.
**Official ZINC test:** never loaded.
**Training / HPO / dictionary:** none.

---

## Question

> Is historical strict-static S0 already, *as a function*, an explicit

> chemical primitives + structural roles → frozen local environments →
> read-only static composition → y

> model whose factorizations were merely folded ahead of time into mixed
> handcrafted input tensors?

## Frozen historical object

* `StrictStaticPairModel` (`build_s0(0)`), `center_context=False`,
  `residual_mode="none"`, 66 228 params, runtime width 302.
* checkpoint `results/zinc_static_dictionary_pair/states/s0_seed0_selection_state.pt`
  (sha256 `8d81f129…8fcf03`); recorded best valid `0.14567435123870381`,
  Top-5 soup `0.140794` (member states not persisted).
* encoded inputs `results/zinc_static_dictionary_pair/cache/encoded_{train,valid}.pt`.

## Method

Every S0 input tensor was rebuilt from raw molecular primitives (atom/bond
categories + untyped graph) through explicit role × primitive bindings
(`experiments/luyin16/fec_s0_factorization.py`).  The original S0 modules and
weights were reused unchanged.  Historical and factorized inputs were run
through the same frozen model for the same molecules; tolerances were frozen
before the run.

## Results

### Input identity

| tensor | train (512) | official-valid (1 000) | max abs | bit-equal |
|---|---|---|---:|---:|
| `patch_cont` raw / standardized (146) | exact | exact | 0.0 | 1.0 |
| `pair_relation` (23, 5 blocks) | exact | exact | 0.0 | 1.0 |
| `global_context` raw / standardized (62) | exact | exact | 0.0 | 1.0 |
| `topology_features` (25) | — | exact | 0.0 | 1.0 |
| `typed_token` / `parent_token` ids | — | 1000/1000 | exact | 1.0 |
| rebuilt certificate bytes | — | 23 083/23 083 pat. | exact | 1.0 |

Standardizer/vocabulary reproduction on full train: `max_abs = 0.0`,
`bit_equal_fraction = 1.0`, token-id `match_fraction = 1.0`.

### Intermediate + prediction identity (official-valid)

`h`, `q`, relation encoder, global encoder, `R`, prediction: **all
`max_abs = 0.0`, bit-equal fraction 1.0** (stricter than the pre-registered
`1e-6` fallback).

### Performance inheritance (fixed checkpoint re-evaluated)

historical MAE `0.1456743378872634` = factorized MAE `0.1456743378872634`;
recorded `0.14567435123870381` (`|Δ| = 1.34e-8`).  Soup not reproduced (member
states absent).

### Static purity

no MP, no recurrence, no pair→centre, no context writeback; environment
formation before pair composition; `h` bit-identical under pair mutation;
relabel-invariant (2.4e-7).  All pass.

### The one blocking path

`typed_token`/`parent_token` are an **aliased, non-injective certificate**
(the historical `pynauty` certificate omits color labels) answered by a
per-key learned table (36 420 params).  Aliasing on official-valid:
12.57 % of unique tokens mix distinct root atoms; **66.32 %** of patch
occurrences lie in such alias buckets.  Reconstructible from primitives, but
not a shared role × primitive factorization.

## Frozen verdict

```
FEC_S0_LOCAL_FACTORIZATION_BLOCKED
```

Case C: the local environment cannot be *fully* factorized because a reachable
local chemistry path (`typed_token`/`parent_token`) is an aliased exact-token
memory rather than a shared role × primitive environment.  Numerically the
function is bit-identically reproducible; the block is specifically about the
factorization claim, not about equivalence.

## Scope statement

This is a model-factorization / computational-equivalence result, not a claim
about real chemistry.  No architecture, weight or protocol was changed; the
dictionary route stays closed; the official test stays unopened.
