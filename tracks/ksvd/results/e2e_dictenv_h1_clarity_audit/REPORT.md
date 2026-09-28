# E2E-DictEnv H1 clarity audit — REPORT

Round `e2e_dictenv_h1_clarity_audit` (Workstream Z), protocol `e2e_dictenv_h1_clarity_audit`.
Code commit `72b490a3672af5a6efa514342e11f27295645608`.
Preregistration: `tracks/ksvd/notes/e2e_dictenv_h1_clarity_audit_preregistration.md`.
Phase A note: `tracks/ksvd/notes/e2e_dictenv_h1_clarity_audit_information_flow.md`.
Durable analysis: `tracks/ksvd/notes/e2e_dictenv_h1_clarity_audit_analysis.md`.

**Scope.** A mechanism / information-flow audit of the existing H1 (P2-ABS, `decoder=h1`,
`K=32`, `s=8`, IHT-10, `d_e=48`, `lambda_rec=33.95873017865987`) main line, run
**CPU-only** (`CUDA_VISIBLE_DEVICES=""`). No new architecture was invented, no hyper-parameter
was tuned, and the official ZINC test was never loaded (`official_test_loaded = false` in every
artifact of this round).

---

## 0. Provenance and the CPU baseline (B0)

| item | value |
|---|---|
| checkpoint | `tracks/ksvd/results/e2e_dictenv_p2_abs/states/H1_soup_state.pt` |
| checkpoint sha256 | `9d9e657292557e62…` |
| state sha256 | `13f39bc9dfa76cbe…` |
| dictionary | `sdb32` sha256 `b0c5da98aee57954…` |
| soup members | [293, 309, 313, 316, 317] |
| official-valid MAE (CPU replay) | **0.123548630** |
| historical GPU soup MAE | 0.123548629 |
| abs diff vs historical | 1.252e-09 |
| audit-path identity mask | bit-identical = True, max abs diff = 0.0 |
| independent P1 `evaluate` replay | MAE 0.123548630, max abs prediction diff 0.0 |
| parameters | 97487 |

Two facts make the rest of the round interpretable:

1. the masked audit model reproduces the parent forward **bit-for-bit** with an identity mask,
   and reproduces the historical GPU number to 1.3e-09;
2. re-running the whole frozen stage a second time reproduced all 88 recorded intervention
   rows (82 real interventions + 6 identity controls) with `max |Δdelta| = 0.0`.

The CPU baseline is *not* the historical baseline: it is rounded by CPU float ordering, so all
comparisons in this round are **within-regime only**.

---

## 1. Phase A — what information actually reaches the reader

Full detail in the information-flow note. Summary:

```
reader input 302 = 97 (unary moments of the dictionary environment E)
                 + 165 (distance-bucketed pair readout: 5 x [mean16, mean2_16, log1p(count)1])
                 + 32 (global_encoder(global_context62))
                 + 8 (topology_encoder(topology_features25))
```

* `global_context62 = structure_short(15) + structure_long(15) + atom_histogram(28) + bond_histogram(4)`
  — the **graph-level chemistry marginal is 32 of the 62 coordinates**.
* `anchor62 = root(28) + atom_mass(28) + bond_mass(4) + size(2)`; `atom_mass+bond_mass` are the
  **patch-level chemistry marginals**.
* the consumed `pair_relation` 15-D vector is `raw[0:14] + raw[18:19]`; raw coordinates 14–17 are
  never read by the model.
* `19` of 42 inventoried
  entries reach the reader **without** passing through the frozen dictionary, and
  `19` also bypass the local
  structure↔chemistry binding. All Phase A provenance checks pass.

---

## 2. Phase B — frozen intervention table

`ΔMAE` against the CPU baseline 0.123548630. `zero` = block set to 0 (or pooled value zeroed);
`fill` = external input block replaced by its training-set per-coordinate mean;
`*-shuf` = each molecule's / pair's / pooled row replaced by another molecule's row
(marginals preserved exactly). **The shuffle probes are the primary evidence; `zero` is retained
only as a cross-check because it corrupts the input out of distribution.**

Concrete proof that the `zero` probe is not a clean removal: removing *more* of the relation
caused *less* damage (`R3` +0.734 vs `R5` +0.164), and the
unary second moment looked like the most fragile block in the model (`P1` zero
+1.005) while its marginal-preserving shuffle puts it at
+0.369. Rows where zero exceeds the primary probe by more than 2x are
flagged in the table below.

| channel | verdict probe | delta | zero | fill | class |
|---|---|---|---|---|---|
| unary pool (all blocks) (`EP2`) | readout_shuffle | **+1.5426** | +1.0721 | +1.0656 | strongly_load_bearing |
| unary first moment (`EP1`) | readout_shuffle | **+1.4088** | +1.4351 | +0.9612 | strongly_load_bearing |
| anchor (full 62) (`A1`) | fill | **+1.0147** | +1.0167 | +1.0147 | strongly_load_bearing |
| relation: distance+overlap+boundary (no path count) (`R1`) | fill | **+0.6514** | +0.3876 | +0.6514 | strongly_load_bearing |
| anchor patch marginals (atom+bond mass) (`A2`) | fill | **+0.5624** | +0.5617 | +0.5624 | strongly_load_bearing |
| relation: boundary block (`R3`) | relation_shuffle | **+0.5541** | +0.7341 | +0.6825 | strongly_load_bearing |
| unary + pair second moments (`P3`) | readout_shuffle | **+0.4769** | +1.0569 | +0.3049 | strongly_load_bearing ⚠zero-inflated |
| anchor root atom identity (`A3`) | fill | **+0.4461** | +0.4490 | +0.4461 | strongly_load_bearing |
| anchor patch atom mass (`A4`) | fill | **+0.4404** | +0.4420 | +0.4404 | strongly_load_bearing |
| unary second moment (`P1`) | readout_shuffle | **+0.3692** | +1.0045 | +0.2346 | strongly_load_bearing ⚠zero-inflated |
| node structure-semantic binding (slot input) (`N1`) | zero | **+0.3476** | +0.3476 |   --    | strongly_load_bearing |
| pair composition (projected E) (`EB3`) | zero | **+0.3361** | +0.3361 |   --    | strongly_load_bearing |
| pair readout (all blocks) (`EP3`) | readout_shuffle | **+0.3013** | +0.4311 | +0.1931 | strongly_load_bearing |
| dictionary coordinate alpha (`N6`) | zero | **+0.2818** | +0.2818 |   --    | strongly_load_bearing |
| edge structure-semantic binding (role input) (`N2`) | zero | **+0.2726** | +0.2726 |   --    | strongly_load_bearing |
| topology25 cycle-spectrum bypass (`T1`) | graph_shuffle | **+0.2666** | +0.1447 | +0.1443 | strongly_load_bearing |
| anchor patch bond mass (`A5`) | fill | **+0.1901** | +0.1934 | +0.1901 | strongly_load_bearing |
| relation: full 15-D (`R5`) | relation_shuffle | **+0.1534** | +0.1638 | +0.1635 | strongly_load_bearing |
| anchor size (`A6`) | fill | **+0.1487** | +0.1488 | +0.1487 | strongly_load_bearing |
| pair second moment (`P2`) | readout_shuffle | **+0.1332** | +0.0911 | +0.0748 | strongly_load_bearing |
| edge role<->bond-type correspondence (`N4`) | zero | **+0.0930** | +0.0930 |   --    | moderately_used |
| relation: overlap block (`R2`) | relation_shuffle | **+0.0890** | +0.1115 | +0.1100 | moderately_used |
| graph-level global62 (full) (`G3`) | graph_shuffle | **+0.0774** | +0.0505 | +0.0508 | moderately_used |
| graph-level chemistry marginal (atom+bond histogram) (`G1`) | graph_shuffle | **+0.0574** | +0.0403 | +0.0404 | moderately_used |
| global encoder output (`EB1`) | zero | **+0.0552** | +0.0552 |   --    | moderately_used |
| distance gate (`EB2`) | zero | **+0.0376** | +0.0376 |   --    | moderately_used |
| graph-level topology summary (short+long) (`G2`) | graph_shuffle | **+0.0369** | +0.0219 | +0.0218 | moderately_used |
| node alpha<->atom correspondence (`N3`) | zero | **+0.0122** | +0.0122 |   --    | weakly_used |
| unary + pair counts (`P4`) | readout_shuffle:count(sum) | **+0.0056** | +0.1347 | +0.0033 | weakly_used ⚠zero-inflated |
| relation: log path count (`R4`) | relation_shuffle | **+0.0044** | +0.0309 | +0.0016 | weak_or_dormant ⚠zero-inflated |
| graph-level bond histogram (`EG2`) | fill | **+0.0030** | +0.0030 | +0.0030 | weak_or_dormant |

### Full frozen zero/mean-fill table

| id | category | zero | fill | description |
|---|---|---|---|---|
| `EP2` | readout_ext | +1.0721 | +1.0656 | zero full unary pool |
| `EP1` | readout_ext | +1.4351 | +0.9612 | zero unary first moment only |
| `A1` | anchor | +1.0167 | +1.0147 | zero full anchor62 |
| `R1` | relation | +0.3876 | +0.6514 | distance-only relation |
| `A2` | anchor | +0.5617 | +0.5624 | keep root atom + size only |
| `R3` | relation | +0.7341 | +0.6825 | remove boundary block |
| `P3` | readout | +1.0569 | +0.3049 | zero unary + pair second moments |
| `A3` | anchor | +0.4490 | +0.4461 | remove root atom identity |
| `A4` | anchor | +0.4420 | +0.4404 | remove patch atom mass |
| `P1` | readout | +1.0045 | +0.2346 | zero unary second moment |
| `N1` | binding | +0.3476 |   --    | zero node-binding slot input |
| `EB3` | backend_ext | +0.3361 |   --    | zero pair-projection output |
| `EP3` | readout_ext | +0.4311 | +0.1931 | zero full relation readout |
| `N6` | binding | +0.2818 |   --    | zero dictionary coordinate alpha |
| `N2` | binding | +0.2726 |   --    | zero edge-binding role input (bond type kept) |
| `T1` | topology25 | +0.1447 | +0.1443 | zero topology25 |
| `A5` | anchor | +0.1934 | +0.1901 | remove patch bond mass |
| `R5` | relation | +0.1638 | +0.1635 | no explicit relation |
| `A6` | anchor | +0.1488 | +0.1487 | remove size |
| `P2` | readout | +0.0911 | +0.0748 | zero pair second moments |
| `N4` | binding | +0.0930 |   --    | edge assignment shuffle |
| `R2` | relation | +0.1115 | +0.1100 | remove overlap block |
| `G3` | global | +0.0505 | +0.0508 | zero full global62 |
| `G1` | global | +0.0403 | +0.0404 | zero graph-level chemistry marginal (atom+bond histogram) |
| `EB1` | backend_ext | +0.0552 |   --    | zero global encoder output |
| `EB2` | backend_ext | +0.0376 |   --    | distance gate off (gate=1) |
| `G2` | global | +0.0219 | +0.0218 | zero graph-level topology part (short+long) |
| `N3` | binding | +0.0122 |   --    | node assignment shuffle |
| `P4` | readout | +0.1347 | +0.0033 | zero unary + pair count terms |
| `R4` | relation | +0.0309 | +0.0016 | remove log path count |
| `EG2` | global_ext | +0.0030 | +0.0030 | zero bond histogram only |

### Distribution-preserving shuffle probes (all rows as recorded)

| id | probe | delta | description |
|---|---|---|---|
| `GS1` | graph row shuffle | +0.0574 | shuffle graph atom+bond histogram rows |
| `GS2` | graph row shuffle | +0.0369 | shuffle graph structure (short+long) rows |
| `GS3` | graph row shuffle | +0.0774 | shuffle full global62 rows |
| `GS4` | graph row shuffle | +0.2666 | shuffle topology25 rows |
| `GS5` | graph row shuffle | +0.3303 | shuffle global62 + topology25 rows |
| `PS1` | readout row shuffle | +1.4088 | shuffle unary first-moment rows across graphs |
| `PS2` | readout row shuffle | +0.3692 | shuffle unary second-moment rows across graphs |
| `PS3` | readout row shuffle | +0.0002 | shuffle unary count rows across graphs |
| `PS4` | readout row shuffle | +0.1690 | shuffle pair first-moment rows across graphs |
| `PS5` | readout row shuffle | +0.1332 | shuffle pair second-moment rows across graphs |
| `PS6` | readout row shuffle | +0.0054 | shuffle pair count rows across graphs |
| `PS7` | readout row shuffle | +1.5426 | shuffle full unary pool rows |
| `PS8` | readout row shuffle | +0.3013 | shuffle full pair readout rows |
| `PS9` | readout row shuffle | +0.4769 | shuffle both second-moment blocks |
| `RS1` | pair row shuffle | +0.0738 | shuffle distance block rows across pairs |
| `RS2` | pair row shuffle | +0.0890 | shuffle overlap block rows across pairs |
| `RS3` | pair row shuffle | +0.5541 | shuffle boundary block rows across pairs |
| `RS4` | pair row shuffle | +0.0044 | shuffle log-path-count rows across pairs |
| `RS5` | pair row shuffle | +0.1534 | shuffle all used relation rows across pairs |

---

## 3. Answers to Q1–Q8

### Q1 — What information actually reaches the reader, and how much of it bypasses the dictionary?

302 coordinates. Only the 97-D `unary` block is dictionary-mediated: the frozen IHT code `α`
enters through the trilinear node slots `(α·W_A^S)*(q·W_A^C)`, then `m_v/m_e`, then `env_mlp`.
The other 205 coordinates are hand-written summaries (anchor 62, relation 15 inside the pair
encoder, graph 62→32, topology 25→8) and are concatenated into the reader directly.
The audit confirms this matters: zeroing the node-slot input changes MAE by
`+0.348` and zeroing `α` by `+0.282`, while
breaking the `α↔atom` correspondence (N3, node assignment shuffle) moves it by only
`+0.0122`.

### Q2 — Which anchor coordinates are genuinely needed?

**All four groups, and this is the strongest single result of the round.** Every probe agrees
(`zero ≈ fill`, no OOD inflation) because the anchor is used as content, not as a noisy input:

| anchor intervention | ΔMAE (zero) | ΔMAE (fill) |
|---|---|---|
| A1 full anchor62 | +1.0167 | +1.0147 |
| A2 drop patch chemistry marginals (keep root+size) | +0.5617 | +0.5624 |
| A3 drop root identity only | +0.4490 | +0.4461 |
| A4 drop patch atom mass only | +0.4420 | +0.4404 |
| A5 drop patch bond mass only | +0.1934 | +0.1901 |
| A6 drop size only | +0.1488 | +0.1487 |
| EA1 keep size only | +1.0093 | +1.0079 |
| EA2 keep patch bond mass only | +0.8526 | +0.8524 |

The model genuinely reads *which chemistry is here*, both globally (root identity
`+0.449`) and locally (patch atom histogram `+0.442`),
plus patch bond composition `+0.193` and size `+0.149`.
Root and patch atom mass are almost equally important and largely redundant with each other
(A2 < A3 + A4). Tier 1 adaptation confirms this is not free to remove: removing the patch
marginals after 20 epochs is `+0.0216` behind
the matched continuation, and still
`+0.0151` after 40 epochs.
The "minimal root-relative information" reading of the anchor is therefore **not** supported by
this checkpoint: the anchor is a load-bearing chemistry channel, not decoration.

### Q3 — Which parts of `pair_relation` (15 D) is the model actually sensitive to?

Distribution-preserving cross-pair row shuffle of each group separately:

| group | dim | ΔMAE (relation-shuffle) | reading |
|---|---|---|---|
| boundary (patch overlap / centre containment) | 3 | **+0.5541** | dominant |
| overlap | 5 | +0.0890 | moderately used |
| distance | 6 | +0.0738 | moderately used |
| log path count | 1 | +0.0044 | dormant |
| all 15 | 15 | +0.1534 | (consistent with R5 fill +0.1635) |

So 14 of the 15 coordinates are load-bearing, dominated by the 3-D boundary block. A
methodologically important side-finding: **the relation is read as an all-or-nothing bundle.**
Removing only the boundary block (zero +0.734, fill +0.682) hurts
far more than removing the whole relation (+0.163), because the encoder
receives a self-inconsistent relation that the reader cannot detect. This is exactly the
failure mode the zero probe alone would have hidden.

### Q4 — Is `global_context` (62 D) just a bypass?

Yes for 32 of its 62 coordinates, and only those 32 are materially used:

| intervention | ΔMAE (graph-shuffle) | ΔMAE (zero/fill) |
|---|---|---|
| GS1 graph chemistry rows (atom+bond histogram) | +0.0574 | +0.0403 / +0.0404 |
| GS2 graph structure rows (short+long) | +0.0369 | +0.0219 / +0.0218 |
| GS3 all 62 rows | +0.0774 | +0.0505 / +0.0508 |
| EG1 atom histogram only | — | +0.0393 |
| EG2 bond histogram only | — | +0.0030 |

The graph-level **atom-type histogram** carries a genuine, moderate effect
(`+0.057` when its rows are swapped between molecules) and it is a pure
chemistry shortcut: no topology, no dictionary, no structure↔chemistry pairing. The graph-level
bond histogram is dormant (`+0.0030`) — redundant with the patch-level
bond mass A5. Tier 2 shows this whole channel is *removable* (see Q8).

### Q5 — Can the relation be simplified to distance only?

No. Distance-only means deleting the boundary block, whose clean sensitivity is
`+0.554` — the largest single relation effect. The zero/fill numbers for
the distance-only ablation (`+0.388` / `+0.651`) are OOD
artefacts of the same bundle effect and should not be quoted as a clean measurement. Distance
information is also redundant by construction: the pair readout is already bucketed by distance
(5 buckets), and the distance gate EB2 alone is worth `+0.0376`.
The only defensible relation simplification is to drop the dormant log path count.

### Q6 — Can the moment readout be simplified?

Partially, and not in the way "second moments are redundant" would suggest:

| readout channel | ΔMAE (readout-shuffle) | ΔMAE (fill) | ΔMAE (zero) |
|---|---|---|---|
| unary first moment | **+1.4088** | +0.9612 | +1.4351 |
| unary second moment | +0.3692 | +0.2346 | +1.0045 ⚠ |
| pair first moment | +0.1690 | — | — |
| pair second moment | +0.1332 | +0.0748 | +0.0911 |
| unary count | +0.0002 | — | — |
| pair count | +0.0054 | — | — |

The second moments are **genuinely used** (`+0.369` unary,
`+0.133` pair), and Tier 1 adaptation says they are not free to remove
either (C5, `+0.0389`
behind the continuation). The two **count** blocks are dormant
(`+0.0002` / `+0.0054`): they duplicate information
already present in the mean and in the size scalars. Note the zero probe inflates the unary
second moment 2.7x (+1.0045 vs +0.3692) — an earlier
reading that the unary second moment was "the most fragile block" was an OOD artefact.

### Q7 — Node/edge binding: correspondence or bag?

| probe | ΔMAE |
|---|---|
| N1 node slot input zeroed (content) | +0.3476 |
| N6 dictionary coordinate `α` zeroed | +0.2818 |
| N3 node `α↔atom` correspondence shuffled | **+0.0122** |
| N2 edge role input zeroed (content, bond type kept) | +0.2726 |
| N4 edge `role↔bond-type` correspondence shuffled | +0.0930 |
| N5 node + edge correspondence shuffled | +0.0977 |
| EB3 pair-composition projection zeroed | +0.3361 |
| EB1 global encoder output zeroed | +0.0552 |
| EB2 distance gate disabled | +0.0376 |

The node branch behaves as a **bag over local codes**: its content is essential
(`+0.348`) but *which atom carries which code* is nearly irrelevant
(`+0.0122`, and it is inside the preregistered inconclusive band). The edge
branch is different: the `role↔bond-type` pairing is really used
(`+0.093`, 7.6x the node value, and almost all of the combined N5 effect).
So the *structure↔chemistry correspondence* is a real mechanism at the **edge** level only; at the
node level the environment is an unordered code multiset.

### Q8 — What is the minimal defensible architecture change, and does it cost anything?

Remove exactly the channels the frozen probes classify as non-load-bearing or redundant, and
nothing else:

* graph-level **atom histogram** (GS1 `+0.057` — used but redundant, it is a
  chemistry bypass) and **bond histogram** (EG2 `+0.0030` — dormant);
* the two **count** blocks of the readout (PS3 `+0.0002`,
  PS6 `+0.0054`);
* the relation **log path count** (RS4 `+0.0044`).

This is candidate `C6`. Under the matched CPU retraining protocol it is **better** than the
untouched architecture, not worse (see §5): soup MAE
`0.143298` → `0.128499`
(`-0.014799`), while removing only the graph atom
histogram (`C1`) gives `0.133530`
(`-0.009768`). Nothing else is removable: the
anchor, topology25, the relation bundle (minus path count), both first and both second moments,
the node-slot content and the edge-role content are all load-bearing.

---

## 4. Tier 1 — warm-start adaptation (is the loss recoverable by short adaptation?)

20 epochs from the H1 soup checkpoint, `Adam(1e-3, wd=1e-5)`, matched batches, seed 0, one
continuation control (`M0`). Deltas are against `M0`, the same-protocol continuation.

| 20 epochs | soup valid MAE | vs matched continuation | epochs | s/epoch |
|---|---|---|---|---|
| `C1` remove graph-level chemistry marginal… | 0.127073 |  +0.0040 | 20 | 7.1 |
| `C2` minimal anchor: root atom + size only | 0.144740 |  +0.0216 | 20 | 7.0 |
| `C3` clean bypass model: C1 + C2 | 0.146410 |  +0.0233 | 20 | 7.1 |
| `C5` clean model + no explicit second moments | 0.161993 |  +0.0389 | 20 | 7.0 |
| `C6` remove graph chemistry marginal + the four… | 0.128205 |  +0.0051 | 20 | 6.8 |
| `M0` continuation control: untouched H1 forward | 0.123116 |  +0.0000 | 20 | 7.1 |

Two extensions to 40 epochs for candidates with a visible recovery trend:

| run | soup valid MAE | best epoch | wall clock |
|---|---|---|---|
| `M0` (continuation control, 40 ep) | 0.123238 | 27 | 261 s |
| `C2` (minimal anchor, 40 ep) | 0.138306 | 33 | 281 s |
| → residual gap | **+0.015068** | | |

Reading:

* `C1`/`C6` are within `+0.0040` /
  `+0.0051` of the continuation: the removal of the
  graph chemistry marginal and of the dormant blocks is **recoverable under short adaptation**.
* `C2`/`C3` cost `+0.0216` /
  `+0.0233` and only close to
  `+0.0151` after 40 epochs: the anchor
  chemistry marginals are **not recoverable under short adaptation**.
* `C5` (no second moments) costs
  `+0.0389`: not recoverable either.
* the continuation control itself improves on the soup
  (0.123116 vs 0.123549), which is exactly why every
  adaptation number above is quoted against it and not against the frozen baseline.

---

## 5. Tier 2 — matched from-scratch CPU retraining (removal claims)

320 epochs, from scratch, identical initialisation and data order for all three arms, 4 threads
per process, 3 processes concurrently.

| arm | mask | soup valid MAE | best epoch | final epoch | Δ vs CPU baseline | wall clock |
|---|---|---|---|---|---|---|
| `BASE` | none (H1 architecture) | 0.143298 | 257 | 0.156006 | +0.000000 | 4023 s |
| `C1` | − graph chemistry marginal | 0.133530 | 317 | 0.152226 | -0.009768 | 4018 s |
| `C6` | − graph chemistry marginal − dormant blocks | **0.128499** | 300 | 0.153126 | **-0.014799** | 4003 s |
| historical GPU H1 (outside this regime) | none | 0.123549 | — | — | — | GPU |

Conclusions:

1. **The graph-level chemistry marginal is a removable bypass.** Deleting it *and* four dormant
   statistic blocks improves the matched from-scratch CPU result by
   `0.0148` in MAE, an effect ~4.9x the
   preregistered inconclusive band, with identical initialisation and data order.
2. This is a **CPU-regime screening result on one seed**, and it must not be compared to the
   historical GPU number: the untouched architecture itself is
   `0.0197` **worse** (higher MAE)
   than its own historical GPU run, so the whole regime is offset. The honest claim is
   "removing these channels does not hurt, and helps, in the tested CPU regime", not
   "the cleaned model beats H1 on ZINC".
3. The arm-matching contract that this comparison rests on — one parameter hash per seed
   regardless of mask, and one mask-independent shuffled batch order — is pinned by
   `test_matched_arms_share_initialisation_and_parameter_shapes` and
   `test_matched_arms_use_the_same_batch_order` in
   `tracks/ksvd/tests/test_e2e_dictenv_h1_clarity_audit.py` (38 focused CPU tests, all passing).
4. The improvement is consistent with the frozen evidence: the deleted blocks are either dormant
   or covered by an existing channel (patch-level bond/atom mass, size scalars, mean pooling),
   i.e. the graph chemistry marginal is **redundant**, and in this regime the extra 32 input
   coordinates act as shortcut capacity rather than as new information.

---

## 6. Runtime, resources, and what was deliberately skipped

| item | value |
|---|---|
| cores / RAM | 16 cores, 27.2 GiB total, 17.9 GiB available at preflight |
| thread setting | 4 threads/process (`OMP_NUM_THREADS=4`); epoch time 8.4 s @2t, 7.3 s @4t, 6.3 s @8t — sublinear |
| train RSS | 1.80 GiB/process → max 9 concurrent from RAM, 4 from cores |
| valid inference | 0.296 s per 1000 molecules, RSS 1.43 GiB |
| Phase A inventory | no training, seconds |
| frozen stage (88 recorded rows, 82 real + 6 identity controls) | 41.8 s cumulative |
| Tier 1 | 6x20 epoch runs = 842 s + 2x40 epoch = 542 s (per-process sum) |
| Tier 2 | 3x320 epoch at concurrency 3: 12.6 s/epoch/process, ≈ 67 min wall clock |
| isolated forward replica | `OMP_NUM_THREADS=1`, 1159 molecules/s vs 1375 @4t; a reproduction harness must use the same thread count to compare timings |
| deliberately skipped | no GPU/SSH (`res`) at any point; no hyper-parameter search; no second seed; no architecture search; no official test |

## 7. Limits of this round

1. **Single seed** for Tier 1/2. The Tier 2 gap is ~5x the preregistered inconclusive band and
   the same-seed arms share initialisation, but it is still one seed.
2. **CPU regime only.** No claim is made about the historical GPU H1 number
   (see §5 point 2).
3. Frozen probes are **causal interventions on a fixed checkpoint**, not retrained removals.
   Only §5 supports removal claims; §4 supports "recoverable under short adaptation"; §2–§3
   support sensitivity/insensitivity statements.
4. The `zero` probe is an OOD corruption. Three rows where it inflates the effect by >2x are
   flagged; relation ablations (R1–R5) are additionally distorted by an all-or-nothing bundle
   effect and are reported only alongside the shuffle probes.
5. `N3` (+0.0122) is small but above the <0.003 noise scale, so it is
   reported as *nearly insensitive*, not as proven-absent; the paired count probe
   (+0.0056) is inside the band and is reported as
   *insensitive*.
