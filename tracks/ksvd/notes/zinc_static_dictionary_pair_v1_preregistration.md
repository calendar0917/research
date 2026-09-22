# Preregistration — ZINC strict-static dictionary-pair v1 confirmation

Round name: **ZINC-strict-static-dictionary-pair-v1-confirmation**.
Study: `zinc-context-gap`.
Status: **frozen before any formal seed-1 run**.

This round answers exactly one question:

> Does the seed-0 dictionary-specific signal of
> `zinc-static-dictionary-pair-v0` replicate on seed 1 **after** removing the
> unequal-checkpoint-opportunity / late-training confound?

It does **not** implement the dictionary-conditioned pair kernel and it does
**not** change any architecture hyperparameter.

## 1. What is inherited, unchanged

The architecture contract of v0 is inherited verbatim through
`tracks/ksvd/experiments/luyin16/zinc_static_dictionary_pair.py`:

```text
center_context = False
center_update  = None          # no pair -> centre aggregation -> local update
no relation refresh; every pair is evaluated exactly once
```

and the residual dictionary is still exactly

```text
K = 64
rank = 32
tau init ~= 0.20
gamma init = 0.1
```

Same tokenizer (historical aliased rooted-topology token), same local
descriptor, same topology hinge (25 -> 16 -> 8), same small raw head
(302 -> 13 -> 13 -> 1), same optimizer (Adam), LR (1e-3), weight decay (1e-5),
batch (128), data split (official train 10000 / official valid 1000), shuffle
seed convention (`seed + 91011` / `seed + 91012`), branch-seed convention
(`head_seed=0`, `branch_seed=1`), and parameter matching (`H = 53`,
5,189 vs 5,201 branch parameters). No architecture hyperparameter changes.

## 2. Phase A — zero-training common-horizon audit (completed before this freeze)

Read from the durable v0 seed-0 curve CSVs
(`results/zinc_static_dictionary_pair/curves/*.csv`), never hardcoded:

```text
H_common = min(epochs_run) = min(204, 210, 240) = 204
```

| arm | epochs run | best_full | best_epoch_full | best_common (<=204) | best_epoch_common | mean_best5_common |
|---|---|---|---|---|---|---|
| S0 | 204 | 0.145674 | 164 | 0.145674 | 164 | 0.146560 |
| S-Dense | 210 | 0.145958 | 170 | 0.145958 | 170 | 0.148202 |
| S-Dict | 240 | 0.136050 | 225 | **0.139009** | 193 | 0.140259 |

```text
M0c = S0   best valid within [1, 204] = 0.145674
MDc = Dens best valid within [1, 204] = 0.145958
MKc = Dict best valid within [1, 204] = 0.139009

base_gain_common = M0c - MKc = +0.006666   (gate >= 0.004)
dict_gain_common = MDc - MKc = +0.006949   (gate >= 0.002)
```

Running-best and raw valid MAE at fixed epochs (all three arms have them):

| epoch | S0 runbest | Dense runbest | Dict runbest | S0 valid | Dense valid | Dict valid |
|---|---|---|---|---|---|---|
| 160 | 0.146690 | 0.147034 | 0.141850 | 0.161724 | 0.149637 | 0.157296 |
| 180 | 0.145674 | 0.145958 | 0.141850 | 0.147840 | 0.164513 | 0.142348 |
| 200 | 0.145674 | 0.145958 | 0.139009 | 0.161139 | 0.154092 | 0.141724 |
| 204 | 0.145674 | 0.145958 | 0.139009 | 0.149951 | 0.153815 | 0.146437 |

**Phase-A verdict: `COMMON_HORIZON_SIGNAL_SURVIVES` (A1).** The dictionary
already beats both S0 and the matched dense control inside the common horizon,
so the seed-0 direction is not created by the extra 36 epochs S-Dict received.

**Restricted Top-5 weight soup: unavailable from saved states.** The v0 runner
kept only the single best-checkpoint `selection_state.pt` per arm; per-epoch
Top-5 snapshots were an in-memory cache and were never written. Forming the
restricted soup would require retraining seed 0, which this round forbids, so
only curve statistics are reported for Phase A. The "best-5 validation MAE
mean" is **not** presented as a weight soup.

**How much of the seed-0 margin depends on epochs 205–240?** S-Dict's full best
(0.136050 @225) is 0.002959 lower than its common-horizon best (0.139009 @193),
while S0 and S-Dense do not improve after their best. So roughly
`0.002959 / 0.009624 = 30.7 %` of the seed-0 primary `S0 - Dict` margin came
from epochs 205–240; the remaining `~69 %` is already present at `H_common`.

## 3. Seed 1 — the only new full-training arms

Exactly three arms are authorized, in this order:

```text
S0       seed 1
S-Dict   seed 1
S-Dense  seed 1     (conditional, see 5.)
```

### 3.1 Forced full horizon + shadow early stopping (single run per arm)

Each seed-1 arm is trained for the **full 240 epochs**; training is never
terminated early. A *shadow* early-stopping tracker runs alongside and exactly
simulates the v0 protocol without ever stopping the optimizer:

```text
patience = 40
shadow best = best official-valid checkpoint within the prefix
stop at the first epoch where (epoch - shadow_best_epoch) >= 40
```

The shadow tracker snapshots its Top-5 weights at the shadow stop moment and
freezes them. Therefore one run yields two legitimate views:

* **equal-horizon view** — best valid / best epoch / Top-5 soup over epochs
  1..240 (identical checkpoint opportunities for all three arms);
* **shadow view** — best valid / best epoch / stop epoch / Top-5 soup over
  `[1, shadow_stop_epoch]` (the inherited early-stop protocol).

No arm is trained twice.

### 3.2 Primary and secondary metrics

```text
equal-horizon primary : M0_1, MD_1, MK_1 = best valid MAE over 1..240
Gbase_1 = M0_1 - MK_1
Gdict_1 = MD_1 - MK_1
equal-horizon Top-5 weight soup : corroborating metric
shadow view                      : reported for comparison with the seed-0 original protocol
```

## 4. Pre-registered seed-1 interpretation

* **B1 — strong replication**: `Gbase_1 >= +0.004` and `Gdict_1 >= +0.002`, and
  the Top-5 soup is in the same direction.
  `SEED1_DICT_SPECIFIC_REPLICATION`.
* **B2 — directional replication**: `Gbase_1 > 0` and `Gdict_1 > 0` but the
  double gate is not met. `SEED1_DIRECTIONAL_REPLICATION`.
* **B3 — generic-capacity ambiguity**: S-Dict beats S0 but does not beat
  S-Dense. `SEED1_GENERIC_CAPACITY_OR_AMBIGUOUS`.
* **B4 — no replication**: S-Dict does not beat S0. `SEED1_NO_REPLICATION`.

No rescue, no retuning, no architecture edit is allowed in response to B2/B3/B4.

## 5. Conditional purchase of S-Dense seed 1

After S0 seed 1 and S-Dict seed 1 finish, a lightweight provisional check is
run before spending the third run:

```text
S0 - Dict on equal-horizon best   and   S0 - Dict on equal-horizon Top-5 soup
```

* If `max(...) >= +0.002` (dictionary maintains a clearly positive direction on
  at least one main view), buy **S-Dense seed 1** as the matched control.
* If S-Dict is **not better than S0 on either view**, stop with
  `SEED1_DICT_NOT_REPLICATED` and do not buy S-Dense.
* If S-Dict is better on both views but below `+0.002`, still buy S-Dense so the
  B-gates are evaluable.

The dense control is mandatory whenever a dictionary-specific claim is on the
table; a good S-Dict seed-1 result never justifies skipping it.

## 6. Two-seed synthesis (reported, not averaged into one benchmark)

* **Original-protocol paired summary** — seed 0 original results paired with the
  seed-1 **shadow** results: `mean(S0 - Dict)`, `mean(Dense - Dict)` and sign
  consistency across the two seeds.
* **Equal-opportunity evidence** — seed-0 common-horizon diagnostic (`<= 204`)
  and the seed-1 full equal-horizon 240-epoch view, reported separately. The two
  horizons are **not** averaged into a single number.

Seed 0 is **not** retrained this round.

## 7. Dictionary mechanism diagnostics (report-only, not a GO gate)

For S-Dict seed 1: active atoms, argmax-used atoms, effective count, assignment
entropy, top-8 mass, max average mass, tau init/final, gamma init/final,
dictionary coherence (mean/max |cos|), mean `||gamma * delta||`, mean `||h0||`,
dictionary/Wq/Wv grad norms, and the inference ablation "residual forced to
zero" (mean/max `|prediction shift|`). The matched dense arm gets the same
residual-disable ablation.

MAE plus the matched dense control remains the primary evidence; mechanism
diagnostics never select a model.

## 8. Non-negotiable integrity gates (run before full training)

```text
center_context == False
center_update is None
_pool_pairs_to_centres monkeypatched-to-raise -> forward succeeds
pair encoder exactly once
relation encoder exactly once
pair mutation leaves local h_i bit-identical
prediction changes under relation mutation
pair-order invariance
dictionary gradients alive
```

Any failure stops the round before full training; the architecture is not
silently changed.

## 9. Explicitly forbidden this round

```text
dictionary-conditioned pair kernel
alpha_i/alpha_j pair features
new relation branch
message passing
centre update
relation refresh
second pair evaluation
K / tau / rank / gamma sweep
top-k / sparsity / entropy / orthogonality / reconstruction regularization
new tokenizer / topology features / larger head / new loss / scheduler search
seed 2 / seed 3
official test
```

Even if seed 1 is strongly positive, the pair kernel is deferred to a separate
future preregistration.

## 10. Budget and execution regime

```text
3 new full-training runs maximum (2 if the provisional check stops early)
240 forced epochs per arm
recommended GPU plan:
    GPU0: S0    seed 1
    GPU1: S-Dict seed 1
    (parallel)
    then S-Dense seed 1 on an idle GPU if purchased
remote A100-SXM4-40GB, commit recorded per run
official test never loaded
```

Durable artifacts:

```text
tracks/ksvd/notes/zinc_static_dictionary_pair_v1_preregistration.md   (this file)
tracks/ksvd/notes/zinc_static_dictionary_pair_v1_analysis.md
tracks/ksvd/results/zinc_static_dictionary_pair_v1_confirmation/
tracks/ksvd/tests/test_zinc_static_dictionary_pair_v1.py
records/claims/...
records/decisions/...
STATE.yaml
```

## 11. Phase-A freeze statement

This preregistration is committed **before** the seed-1 runs. Phase A was
computed from the existing seed-0 curves only, with no new training. The
seed-1 gates (`B1`–`B4`), the forced-240 protocol, the shadow tracker, the
conditional S-Dense rule, and the forbidden list above are frozen at this
commit.
