# compact-v4 ordered shortest-path bond sequence — Phase A/C audit (2026-09-13)

> **Question.** The frozen best model is the compact-v4-smallhead **T=2
> weight-tied recurrent pair–centre** (82,115 params; seed0 valid 0.138376,
> seed1 valid 0.133440). The relation primitive `q_ij = Q(h_i, h_j, r_ij)` is
> the last unmodified structural component. Its path-related descriptor keeps
> the shortest-path length and the bond **composition averaged over shortest
> paths**, but discards the **order** of the bond types along the path. Does
> restoring that order help?
>
> **Verdict.** **NO-GO / DEGRADED (seed0 only; official test never loaded).**
> The Phase-A audit finds genuine, substantial path-order aliasing (55.0% of
> official-valid pairs, rising from 46% at `d=3` to >90% at `d>=11`). A minimal
> ordered bond-sequence encoder was added to the *unchanged* T=2 recurrent
> skeleton (`+2,328` params, `84,443` total), and it **degraded** seed0
> official-valid MAE to **0.142814 @ epoch 120** versus the recurrent reference
> **0.138376** (`Δ = −0.004438`, i.e. worse). Per the pre-registered gate
> (`Δ >= +0.002` to continue), the direction is stopped: **no orderless matched
> control, no seed1, no atom sequence, no further path-architecture sweep.**
>
> Code: `tracks/ksvd/experiments/luyin16/zinc_compact_v4_ordered_path_bond.py`.
> Tests: `tracks/ksvd/tests/test_compact_v4_ordered_path_bond.py` (12 pass).
> Results: `tracks/ksvd/results/compact_v4_ordered_path_bond/`.

---

## 0. What was held fixed

Everything except the relation primitive is inherited verbatim from the frozen
T=2 recurrent pair–centre model:

```
h0 -> q0 -> A0 -> h1 -> q1 -> A1 -> h2 -> existing readout/head
```

* same modules (`pair_projection` / `relation_encoder` / `distance_gate` /
  `pair_encoder` / `center_update`), same weight tying across the two rounds;
* same `q_dim = 16`, same centre hidden width, same 302D readout, same small
  head `GenericReader(302, (13,13))`;
* same tokenizer / radius / patch construction / distance buckets / path count
  feature / global + topology channels;
* same optimizer (Adam lr 1e-3, wd 1e-5), batch 128, max 240 epochs, patience
  40, no scheduler, gradient clip 5.0, L1 loss, best-valid checkpoint.

The **only** change is `q_ij = Q(h_i, h_j, r_ij, s_ij)` with `s_ij` a small
ordered-path embedding concatenated to the pair-encoder input. The shared
baseline columns of the widened pair encoder are copied bit-exactly from the
recurrent init; `force_zero_path` makes the branch reduce to the recurrent
function exactly (sanity gate).

## 1. Phase A — path information audit

Audited on the official PyG ZINC `train` and `val` splits; **official test was
never loaded**. Grouping key = the path-related part of the descriptor only,
taken verbatim from the production `zpp._shortest_path_summary`:

```
(distance, shortest_path_count, bond_composition_mean)
```

Endpoint / patch-overlap features are deliberately **excluded**, so a collision
is a genuine loss of path information, not a masked endpoint difference.

| split | pairs | multiple shortest paths | bond-order collision | atom-sequence collision (bonus) |
|---|---:|---:|---:|---:|
| official-valid | 264,776 | 23.6% | **55.0%** | 99.9% |
| official-train | 2,668,346 | 23.1% | **57.3%** | 100.0% |

Collision = fraction of pairs whose descriptor group contains ≥2 distinct true
ordered bond-sequence signatures.

**Multiple shortest paths** are handled explicitly: each shortest path is
canonicalised to `min(seq, reverse(seq))`; a pair's true signature is the sorted
multiset of canonical sequences over *all* shortest paths. No pair exceeded the
enumeration cap (512); max observed shortest-path length 19 (valid).

## 2. Collision by path length (official-valid)

| d | pairs | bond collision | atom collision |
|---:|---:|---:|---:|
| 1 | 24,846 | 0.0000 | 1.0000 |
| 2 | 34,256 | 0.0000 | 1.0000 |
| 3 | 33,845 | 0.4631 | 1.0000 |
| 4 | 30,629 | 0.6266 | 1.0000 |
| 5 | 27,848 | 0.6712 | 0.9998 |
| 6 | 24,875 | 0.7189 | 0.9997 |
| 7 | 21,651 | 0.7613 | 0.9997 |
| 8 | 18,288 | 0.8043 | 0.9997 |
| 9 | 14,966 | 0.8449 | 0.9991 |
| 10 | 11,697 | 0.8784 | 0.9993 |
| 11 | 8,553 | 0.9028 | 0.9986 |
| 12 | 5,770 | 0.9265 | 0.9964 |
| 13 | 3,590 | 0.9415 | 0.9950 |
| 14 | 2,081 | 0.9390 | 0.9899 |
| 15 | 1,103 | 0.9474 | 0.9773 |
| 16 | 493 | 0.9412 | 0.9615 |
| 17 | 203 | 0.8867 | 0.8966 |
| 18 | 69 | 0.7101 | 0.7971 |
| 19 | 13 | 0.3077 | 0.3077 |

Reading: at `d=1,2` the descriptor is loss-less for bond order (`d=2` order is
inherently symmetric for an undirected pair). From `d=3` onwards order is
genuinely lost, and the loss grows with distance.

## 3. Phase B — ordered path encoder (actual implementation)

Input for every pair is a fixed-size float tensor

```
counts[pos, bond] = mean over shortest paths of 1[bond at position pos]
                    (positions >= PATH_MAX_LEN folded into the last slot)
shape [n_pairs, 12, 4]
```

Encoder (all shared across pairs and graphs):

```
x[pos] = sum_b counts[pos, b] * bond_embedding[b] + position_embedding[pos]
s      = MLP(flatten(x))      # Linear(96,16) -> ReLU -> Linear(16,8)
```

Then `pair_input = [left+right ; |left-right| ; product*gate ; relation ; s]`
and the existing `pair_encoder` is widened from 64D to 72D input. The first 64
input columns are copied bit-exactly from the recurrent pair encoder; the 8 new
columns are small-normal (`std=0.01`) to avoid the documented zero-init
branch-collapse.

No Transformer, no GRU/RNN, no attention, no atom sequence.

## 4. Reversal invariance & multiple shortest paths

* **Reversal invariance.** Each shortest path is canonicalised to
  `min(bond_sequence, reverse(bond_sequence))` before its counts are
  accumulated, so `(i,j)` and `(j,i)` produce identical `s_ij`. Unit-tested and
  shown in `sanity_ordered_path.json` (`canonical_collapses_reversal = true`).
* **Multiple shortest paths.** `s_ij` is the mean over the canonicalised paths
  (`counts` is the position-wise mean); aggregation is permutation-invariant and
  is unit-tested. The existing `log1p(path_count)` feature is retained in
  `r_ij`.

## 5. Parameters & runtime

| model | total params | head | Δ params |
|---|---:|---:|---:|
| T=2 recurrent reference | 82,115 | 4,135 | — |
| ordered path | **84,443** | 4,135 | **+2,328** |
| orderless control (not run) | 84,443 | 4,135 | +2,328 |

| run | epochs run | wall clock | epoch time | ratio |
|---|---:|---:|---:|---:|
| recurrent seed0 | 239 | 1,777 s | 7.44 s | 1.00× |
| ordered seed0 | 160 | 1,260 s | 7.88 s | **1.06×** |

Cost is small; the encoder is not a runtime problem. Peak RSS was not
instrumented this run; the added tensors are `[pairs, 12, 4]` float32 (~1.0 GB
for train) and the added model params are tiny.

## 6. Phase C — seed0 gate

| model (seed0) | official-valid MAE | best epoch | params | reference |
|---|---:|---:|---:|---|
| T=2 recurrent | 0.138376 | 199 | 82,115 | frozen |
| **ordered path** | **0.142814** | **120** | 84,443 | this run |
| **Δ (recurrent − ordered)** | **−0.004438** | | | worse |

Matched-horizon rolling-minimum check (both models under the same protocol):

| horizon | recurrent roll-min | ordered roll-min | Δ |
|---:|---:|---:|---:|
| ≤ 120 | 0.14729 | 0.14281 | +0.00448 |
| ≤ 140 | 0.14143 | 0.14281 | −0.00138 |
| ≤ 160 | 0.14143 | 0.14281 | −0.00138 |
| final | 0.13838 | 0.14281 | −0.00444 |

The ordered run early-stopped at epoch 160 (best 120); the recurrent reference
trained to 239 (best 199). Even at a matched horizon the ordered model is
slightly worse, so the negative verdict is not only an early-stop artefact. The
ordered model is also stopped before the reference's late dip, which is a
secondary caveat.

## 7. Sanity checks (all pass)

`sanity_ordered_path.json` (`all_checks_passed = true`):

1. `path_off_equals_recurrent` — forcing `s_ij = 0` reproduces the recurrent
   function bit-exactly (`max|ΔR| = 0.0`).
2. `path_on_changes_readout` — the path branch changes `R` at init.
3. `baseline_pair_encoder_columns_preserved` — first 64 input columns exact.
4. `ordered_orderless_param_matched` — 84,443 = 84,443.
5. `reversal_invariance.canonical_collapses_reversal` and
   `aggregation_permutation_invariant` — both true.
6. `counts_align_pair_index` — per-graph row counts equal `pair_index`.
7. forward/backward finite; `path_encoder_grad_max > 0`.
8. independent verification: all 4,969 audit-checked `d=1` adjacent-bond
   one-hots equal `counts[:,0,:]`; `valid_data[i]` matches raw molecule `i`
   (0 mismatches over 1,000).

## 8. Phase D — not triggered

`final_decision.json` → `phase_d.triggered = false`; the Phase-C verdict is
`DEGRADED`, i.e. no positive/weak ordered signal, so the matched orderless
control was **not** run. There is therefore no ordered-vs-orderless comparison
for this seed.

## 9. Output checklist

1. Phase-A audit: yes (train + valid).
2. Collision by length: yes (§2).
3. Ordered encoder implementation: yes (§3).
4. Reversal invariance / multiple-path handling: yes (§4).
5. Params & runtime: yes (§5, +2,328 params, 1.06× epoch).
6. seed0 ordered valid MAE / best epoch: 0.142814 @ 120.
7. Δ vs 0.138376: **−0.004438** (worse).
8. Orderless control triggered: **no**.
9. Ordered vs orderless: not run.
10. Final judgement: **暂无 path signal for this minimal ordered encoder**
    (degraded, not merely flat).

## 10. Interpretation

The hypothesis that the current relation primitive loses *path order* is
**audit-confirmed**: order aliasing is real and common. But the minimal ordered
bond-sequence encoder did **not** convert that missing information into a
validation gain on the frozen recurrent skeleton. Two readings remain open and
were deliberately **not** probed further under this pre-registration:

* the frozen recurrent skeleton may already compensate for the missing order
  through its endpoint/adjacency/overlap features and the global channel; or
* the missing information is real but not task-relevant at this model scale
  (consistent with the earlier stagewise representation-collision and
  function-basis NO-GOs).

Per the pre-registered gate, stop: do **not** run an orderless control, do
**not** add atom sequence, do **not** sweep encoder variants
(CNN/bigram/learned-order), do **not** open official test. Any future
path-order work would need a fundamentally different mechanism witness and its
own pre-registration.
