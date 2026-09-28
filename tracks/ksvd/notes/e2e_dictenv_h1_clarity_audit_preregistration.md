# E2E-DictEnv-H1-Clarity-Audit — pre-registration

Round **E2E-DICTENV-H1-CLARITY-AUDIT** · protocol `e2e_dictenv_h1_clarity_audit`
· study `zinc-context-gap` (Workstream Z).
Baseline artifact: P2-ABS winner **H1**
(`results/e2e_dictenv_p2_abs/states/H1_soup_state.pt`, Top-5 soup members
`[293, 309, 313, 316, 317]`, historical official-valid soup MAE
`0.12354862861608853`, commit `122db5dcd45e28f5a6dc3b6314a18e1ea328deae`).

This round is an **information-flow / mechanism / simplification audit** of the
already-validated H1 structure–semantic binding.  It is *not* a performance
search.  The output is a ranked map of which information channels the trained
H1 model actually reads, plus a decision about which channels can be simplified
in a cleaned architecture.

## 0. Hard constraints (frozen before any run)

1. **CPU only.** Every process sets `torch.set_num_threads(k)` and asserts
   `device == cpu`.  No CUDA, no SSH, no remote runner.
2. **No official ZINC test.** Only `encoded_train` (10,000) and `encoded_valid`
   (1,000).  Every artifact records `official_test_loaded = false`.
3. **Representation frozen.** `K = 32`, `s = 8`, `IHT steps = 10`, `d_e = 48`,
   SDB-v0 dictionary family, `lambda_rec = 33.95873017865987`, same optimizer /
   batch / seed / split / Top-5-soup rule as H1.  No dictionary, capacity,
   sparsity, LR, dropout, reader, attention, message-passing or recurrence
   changes.
4. **P1 / P2-ABS frozen implementations are not modified.**  The audit is a
   subclass + runner; `e2e_dictenv_p1.py` and `e2e_dictenv_p2_abs.py` stay
   byte-identical.
5. **Baseline contract.**  With no intervention active, the audit forward must
   be bit-identical to `P2Model.forward` on the same batch (asserted in a CPU
   test and re-asserted on the real valid split at run time).

## 1. Index provenance (verified from generating code, not from memory)

### 1.1 `anchor` (62-D, `e2e_dictenv_p1.build_anchor_raw`)

| slice | width | content | kind |
|---|---|---|---|
| `[0:28]`  | 28 | root atom one-hot `q_i` | chemistry (identity) |
| `[28:56]` | 28 | patch atom mass `sum_{v in P_i} q_v` (unconditioned) | chemistry (marginal) |
| `[56:60]` | 4  | patch bond mass `sum_{e in E_i} b_e` (unconditioned) | chemistry (marginal) |
| `[60:62]` | 2  | size `[log1p|V_i|, log1p|E_i|]` | topology (root-local) |

Stored standardized with an official-train-only per-coordinate
`(mean, scale)` (`results/e2e_dictenv_p1/anchor_stats.json`).

### 1.2 `global_context` (62-D, `zinc_long_range_proxy.global_feature_views`)

`global_all = concat([structure, attributes])`,
`structure = concat([short(15), long(15)])`,
`attributes = concat([atom_hist(28), bond_hist(4)])`:

| slice | width | content | kind |
|---|---|---|---|
| `[0:15]`  | 15 | `short`: n, m, density, degree mean/std/min/max, q25/q50/q75, frac deg==1, frac deg>=3, `m-n+1`, triangles, mean clustering | topology |
| `[15:30]` | 15 | `long`: shortest-path mean/std/min/max, q25/q50/q75/q90, eccentricity min/mean/std/max, frac dist>2/>3/>4 | topology |
| `[30:58]` | 28 | **whole-molecule atom-type frequency histogram** | chemistry marginal |
| `[58:62]` | 4  | **whole-molecule bond-type frequency histogram** | chemistry marginal |

### 1.3 `topology_features` (25-D, `zinc_topology_features.raw_vector(mode="hinge")`)

`[longest, n3..n10, n>10, mcb_count, mcb_max, mcb_mean, mcb_total, cycle_rank]`
(15) `+ [longest, longest^2, ReLU(longest-3)..ReLU(longest-10)]` (10).
Pure topology, graph-level, **not** dictionary-mediated, **not** root-local.

### 1.4 `pair_relation` → `P1_RELATION_INDICES` (15-D used slice)

Raw 23-D layout: `[0:5]` distance one-hot, `[5]` log distance, `[6:11]`
overlap (5), `[11:14]` boundary (3), `[14:18]` path-bond-mean (dropped),
`[18]` log path count, `[19:23]` adjacent-bond-type (dropped).
`P1_RELATION_INDICES = (0..13, 18)` gives the 15-D used vector:

| used index | group | raw indices |
|---|---|---|
| `[0:5]`  | distance one-hot (5 buckets) | 0..4 |
| `[5]`    | log shortest-path distance | 5 |
| `[6:11]` | overlap block (5) | 6..10 |
| `[11:14]`| boundary block (3) | 11..13 |
| `[14]`   | log path count | 18 |

### 1.5 Local environment readout blocks

`unary = pool_moments(E)` = `[sum_v E   (48) | sum_v E^2 (48) | log1p(count) (1)]`
= 97-D.  `relation_readout = pool_pair_moments(pair_value)` = per bucket
`[sum | sum of squares | log1p(count)]` × 5 buckets = 165-D.  Reader input is
`97 + 165 + 32 + 8 = 302`.

## 2. A. Phase A — static information-flow inventory (no training)

Record for every prediction-affecting variable: name, dim, source, `is_structure`,
`is_chemistry`, `is_structure_chemistry_correspondence`, `dictionary_mediated`,
`handcrafted_statistic`, `bypasses_dictionary`, `bypasses_local_binding`, entry
module.  Outputs:

```
results/e2e_dictenv_h1_clarity_audit/information_inventory.json
notes/e2e_dictenv_h1_clarity_audit_information_flow.md
```

## 3. B. Phase B — frozen H1 interventions (no training)

`B0` is the same-process CPU replay of the H1 soup checkpoint on official-valid.
All `ΔMAE`, `mean |Δpred|` and `corr(pred, B0)` are relative to that replay.

Frozen interventions are pre-registered here; no others may be added after the
first frozen run.  `E*` marks extended items (declared, lower priority).

### 3.1 Anchor

| id | zero groups of `anchor` |
|---|---|
| A0 | none (control: audit path with identity mask) |
| A1 | all four |
| A2 | `atom_mass`, `bond_mass` (keep root + size) |
| A3 | `root` |
| A4 | `atom_mass` |
| A5 | `bond_mass` |
| A6 | `size` |
| EA1 | `root`, `atom_mass`, `bond_mass` (keep size) |
| EA2 | `root`, `atom_mass`, `size` (keep bond mass) |

### 3.2 Graph-level global context

| id | zero slices of `global_context` |
|---|---|
| G0 | none |
| G1 | `[30:62]` (atom + bond histogram = chemistry marginal) |
| G2 | `[0:30]` (short + long structure) |
| G3 | all 62 |
| EG1 | `[30:58]` atom histogram only |
| EG2 | `[58:62]` bond histogram only |

### 3.3 Topology-25 bypass

| id | action |
|---|---|
| T0 | none |
| T1 | zero `topology_features` (25) |

### 3.4 Node / edge binding

| id | action |
|---|---|
| N0 | none |
| N1 | zero node-binding slot input (`u_n == 0`) |
| N2 | zero edge-binding role input (`g == 0`, bond type kept) |
| N3 | node assignment shuffle (5 seeds, P1 semantics: permute the code gather inside each `(root, shell)`, preserving alpha/q/node/shell multisets) |
| N4 | edge assignment shuffle (5 seeds, P1 semantics: permute endpoints inside each `(root, shellpair)`, preserving role/bond-type multisets) |
| N5 | N3 + N4 |
| N6 | zero dictionary coordinate `alpha` (global) |

### 3.5 Pair relation

| id | zeroed group(s) of the 15-D used relation |
|---|---|
| R0 | none |
| R1 | overlap + boundary + path count (distance-only) |
| R2 | overlap |
| R3 | boundary |
| R4 | path count |
| R5 | all 15 (no explicit relation) |
| ER1 | distance one-hot + log distance only (keep overlap/boundary/path count) |

### 3.6 Statistical readout

| id | action |
|---|---|
| P0 | none |
| P1 | zero unary second-moment block |
| P2 | zero pair second-moment blocks (all buckets) |
| P3 | zero unary + pair second moments |
| P4 | zero unary + pair count terms |
| EP1 | zero unary first moment only |
| EP2 | zero full unary pool |
| EP3 | zero full relation readout |

### 3.7 Extended graph-backend items (declared; only run if free)

| id | action |
|---|---|
| EB1 | zero `graph_hidden` (global encoder output) |
| EB2 | distance gate off (`gate = 1`) |
| EB3 | zero pair-projection output (pair inputs lose `E` entirely) |

Metrics per row: `baseline_mae`, `intervention_mae`, `delta_mae`,
`mean_abs_prediction_delta`, `prediction_correlation`, `runtime_seconds`,
`notes`.  Outputs: `frozen_interventions.csv` / `.json`.

## 4. C. Phase C — selection rules for training

Only channels with a *material* frozen effect (or a scientifically central
bypass claim) are promoted.  Pre-registered candidate order:

```
C1  remove graph-level chemistry marginal (G1)
C2  minimal anchor: root atom + size (A2)
C3  clean bypass model: C1 + C2
C4  simplified relation: clean + distance-only relation (R1)
C5  simplified statistics: clean + no second moments (P3)
```

Selection rule (frozen): promote a family only if its frozen `ΔMAE ≥ 0.005`
for "keep information" candidates, or if G1/A2/T1/relation items have
`ΔMAE ≥ 0.002`.  Channels with frozen `ΔMAE < 0.002` are classified
`weak / apparently dormant` and are *simplification candidates*, not
retraining candidates.

### 4.1 Tier 1 — warm-start adaptation (from the H1 soup checkpoint)

* budget 20 epochs (extend once to 40 only if the last 5 evaluations show a
  monotone decreasing gap and the candidate is still behind).
* **continuation control required**: the *unmasked* H1 forward continued for the
  same 20 epochs from the same checkpoint, same seed, same optimizer state
  (fresh Adam), same batch order.  A candidate is only ever compared against
  this matched continuation.
* stop early if candidate − continuation `> 0.02` MAE and the gap is not
  shrinking over the last 5 evaluations.
* differences `< 0.003` are reported `INCONCLUSIVE_AT_SCREENING_SCALE`.

### 4.2 Tier 2 — matched CPU retraining

* at most `1` CPU baseline (H1 architecture, from scratch) `+ 1..2` finalists,
  run in the same CPU regime (same seed 0, same splits, same optimizer, same
  batch 128, same λ, same horizon, same Top-5 soup rule, same thread count).
* the CPU baseline is *not* the historical GPU 0.123549; the comparison is
  candidate-vs-CPU-baseline only.
* horizon: 320 is affordable locally (measured ≈ 6–8 s/epoch) and is therefore
  pre-registered as the matched horizon.  If wall-clock forces a shorter
  protocol it must be declared before the first matched run and applied to
  *all* matched runs, and results then read as `CPU screening`.
* a candidate wins only if it beats the matched CPU baseline by more than the
  single-seed noise scale; `< ~0.003` is not read as a winner.

## 5. Stopping rule for the whole round

Stop before Tier 2 if Phase B already answers the structural questions with
`ΔMAE` that is either clearly negligible (`< 0.002`, removable) or clearly
load-bearing (`> 0.02`, not removable), and if no candidate combines a
materially removable channel with a plausible clean-architecture story.
Shell / shellpair audits are explicitly out of scope for this round unless
Phases A–C finish with budget left, and then only `S0/S1/S2` are allowed.

## 6. Language discipline

* Frozen intervention ⇒ "the trained H1 model is sensitive / insensitive to
  this path".
* Warm-start adaptation ⇒ "the representation appears recoverable / difficult
  to recover under short adaptation".
* Matched retraining ⇒ "this channel can be removed with limited / material
  cost under the tested CPU protocol".
* Never: "this information is fundamentally necessary / unnecessary".

## 7. Deliverables

```
results/e2e_dictenv_h1_clarity_audit/{runtime_budget,baseline_replay,information_inventory}.json
results/e2e_dictenv_h1_clarity_audit/frozen_interventions.{csv,json}
results/e2e_dictenv_h1_clarity_audit/adaptation/...
results/e2e_dictenv_h1_clarity_audit/matched_cpu/...
results/e2e_dictenv_h1_clarity_audit/REPORT.md
results/e2e_dictenv_h1_clarity_audit/DECISION.md
notes/e2e_dictenv_h1_clarity_audit_information_flow.md
```
