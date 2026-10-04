# EVIDENCE_SCOPE — zinc-chemistry-dictionary-vs-mlp-seed0-v1

This file fixes what earlier rounds already answered and what this round is
allowed to claim.  It is written **before** any new model is trained.

## 1. Already answered — do not re-buy

| earlier round | execution | what it established | what it did **not** establish |
| --- | --- | --- | --- |
| `zinc_task_dictionary_and_cycle_witness_seed0_v1` | `4797d06` / execution `77c40dda9d5b730a64ea653b63b053a11460993d` | On a **new model-internal fold**, fresh-only compressed body, matched ``D_y``/``M_y`` bridges: new-dev overall cal MAE `D 0.122960` / `M 0.117032`; G0 (k=0) cal `D 0.104300` / `M 0.103305`; G0 gain `-0.000995`, CI `[-0.006240, +0.003491]`; gate `INCONCLUSIVE`. | Nothing about whether removing the ring/cycle supervision (`y -> g`) changes the comparison. |
| `zinc_cycle_head_size_input_repair_seed0_v1` | `35471dc` | N/E repair cut the T25 conflict classes 4 -> 1, but the five-row extreme fit only moved `14.2209 -> 14.1539` and dev overall cal gain was `+0.000135`; class `INPUT_REPAIRED_NOT_FIT`. | Not evidence that the representation capacity is exhausted; the "fit the five rows" question is closed, not the "does the body generalise" question. |
| `zinc_overnight_interface_and_tail_seed0_v1` | `8b4b9b4` | A severity-weighted tail head fitted the five extreme fit rows to ~`0.097` but damaged G0 and dev overall cal; that weighting scheme is closed. | Not evidence that every alternative body/target treatment is closed. |
| `zinc_full_cycle_target_decomposition_v1` | legacy | Oracle decomposition (`O`: train on `g = y - c`, evaluate `h(x)+c`) answered whether a *specific full model* was hindered by joint ring supervision; it used a different model family/fold and cannot be used as a paired dictionary-vs-MLP fact. | Not a matched `D`/`M` pair; not a generalisation claim about the compressed bridge. |
| luyin19 recording (`docs/luyin/luyin19.txt`) | transcript | The advisor's stated purpose of the shared dictionary is to learn a *general structure-attribute relation* and transfer it to unseen graphs; direct structure x semantic fusion is the open issue, and the current system only had a topological dictionary with simple concatenation fusion. | The transcript is a research-direction discussion, not a result; nothing in it is validated by this round. |

## 2. Exactly what this round buys

One single-variable question:

> With identical inputs, parameter counts, initialization, schedule, optimizer
> and data stream, does the **current fused task-dictionary bridge** ``D_g``
> generalise the **remaining chemical target** `g = y - c` better than the
> matched bias-free MLP bridge ``M_g``, on the same new internal fold?

The only scientific change relative to the paired y round is the supervision
target: `y` (which contains the cycle component `c`) is replaced by `g`
(cycle component removed).  `c` and `g` are **training-side decomposition
labels**; `g` alone is not deployable on its own.  This round does not train a
new cycle head and never adds `c` at evaluation time to claim deployment
performance.

## 3. What this round explicitly does **not** test

1. **Not** the original luyin19 architecture.  The compressed skeleton does not
   contain the structural dictionary `D`, the `U`/IHT coding path, the slot
   binding or the structural-semantic cross modules.  A win or loss of the
   fused task-dictionary bridge does not by itself confirm or refute "shared
   structure-semantic relations transfer to unseen graphs".
2. **Not** all dictionaries or all MLPs.  One implementation of each function
   class, one seed, one fold, one λ/width/step setting frozen from the prior
   round.
3. **Not** a deployable `y` prediction.  Overall-row y-MAE and g-MAE are not the
   same task; only the `k=0` (G0) endpoint admits the constant-offset bridge
   `y = g + c0`.
4. **Not** a hyper-parameter search: no new seed, no new fold, no width/ISTA
   step/λ/WD/loss scan, no early stopping, no dev-based selection.
5. **Not** a 10k confirmation and **no** official-valid / official-test access.

## 4. Boundary of conclusions

* If `D_g` wins on G0 with a CI-separated calibrated gain, the supported claim
  is only "the current task bridge, under decomposed chemical supervision, is
  worth further study"; it is not a new `y` baseline and not a luyin19 proof.
* If removing ring supervision helps the body but the dictionary still does not
  win, the gain is attributed to target decomposition, not to the dictionary.
* If neither effect is confirmed, the conclusion is about *this* bridge pair
  and *this* decomposition; it does not license "the problem is the input".
* Descriptive `y`-arm vs `g`-arm deltas are reported as such; they are not a
  second deployment comparison.
