# SSRC Phase 2b — frozen protocol

**Standard ZINC-12k performance validation of the frozen Phase 2a F/M candidates**

| field | value |
|---|---|
| protocol_id | `ssrc-phase2b-zinc-v1` |
| track | `structural-sufficiency` (main line) |
| status | **FROZEN_PRE_GPU** |
| freeze commit | the commit that first adds this file (`git log --follow -- results/phase2b/PROTOCOL.md`) |
| upstream | Phase 2a report (`results/phase2a/REPORT.md`, verdict `CAPACITY_ONLY_INCREMENT`), P3/R3 frozen artifacts in `tracks/structure-contribution` |
| budget cap | **2.5 A100 GPU-hours**, all stages (smoke, failed runs, formal runs) |
| host | `res` (single box, 2×A100-SXM4-40GB), existing `rr` workflow |

This protocol is written and committed **before any Phase 2b GPU second, before any
new official-valid score, and before any official-test access**. After the freeze
commit, only implementation bug fixes are allowed; no arm, width, depth, learning
rate, epoch budget, metric, subgroup or threshold may change. No result of this
round may be used to revise this document; deviations go in `REPORT.md`.

---

## 1. Question and scope

Phase 2a found, on the **internal** Fit 8,000 / Monitor 1,000 split, that inserting
a small per-atom stage after `AtomEncoder` improves MAE over the frozen P3E encoder
by ≈0.0055, while replacing that stage with two GINE rounds on real physical bonds
adds nothing measurable (`Gain_MP` = +0.0038 / −0.0001 across seeds). Those numbers
are **not** official ZINC-12k test numbers: they use Fit-only SSL, an 8k supervised
set, and an internal Monitor split.

This round asks exactly one performance question:

> Does the Phase 2a improvement survive when the two frozen candidates are trained
> on the **standard** ZINC-12k supervised protocol (official train 10,000, official
> valid 1,000 as the development split), with the **historical full-10k masked-SSL
> encoders** and the **historical P3/R3 recipe**, against the **historical frozen
> P3 checkpoints** measured on the same official valid/test splits?

It is a performance-validation and route-decision round. It is **not** new method
development. No new module, gate, regulariser, width, depth, learning rate, pretext
task, input feature, external data, or split is introduced.

Explicit non-goals: retraining GraphGPS/ESA+/GRIT/GSN or any public model; ZINC-Full;
parameter or hyper-parameter search; selecting a seed or recipe on any score;
continuing to Phase 2c automatically.

---

## 2. Standard benchmark reference table (fixed before any Phase 2b run)

All rows are **official ZINC-12k test MAE** (1,000-molecule test split of
`ZINC(subset=True)`), lower is better. ZINC-Full (250k molecules, ESA+ 0.011) is a
**different benchmark** and is never placed in this table.

| model | official test MAE | params | supervised training data | structural prior / extra data | evaluation | source |
|---|---:|---:|---|---|---|---|
| **BJOC-P3/R3** (this repo, 2 seeds) | **0.069822** (0.069304 / 0.070340) | 413,505 | official ZINC-12k train 10,000 only | masked atom/bond SSL on the **same 10,000 structures** (no labels, no external molecules); BJOC atom/bond/junction/Null-composer encoder; no RWSE/ring features | PyG ZINC subset test 1,000, raw-y MAE, single model per seed, seed-mean reported; **historical exposed test** | `bjoc_r3_frozen_terminal_test_v1/REPORT.md`, `official_test_metrics.json` |
| GraphGPS (GINE + Transformer + RWSE-20) | 0.070 ± 0.004 | 423,717 | official ZINC-12k train 10,000 | RWSE-20 positional/structural encoding; no pretraining, no external data | ZINC-12k test, MAE, 10 runs, mean ± s.d. | Rampášek et al., arXiv:2205.12454, Table 3 + Table B.1 |
| GRIT (no MP, RRWP) | 0.059 ± 0.002 | ~500k budget | official ZINC-12k train 10,000 | RRWP relative random-walk positional encoding; no pretraining | ZINC-12k test, MAE, 4 runs | Ma et al., arXiv:2305.17589, Table 1 |
| CSA-rings (no MP) | 0.056 ± 0.002 | ~500k budget (GraphGPS framework) | official ZINC-12k train 10,000 | RWSE + explicit ring-membership encoding (max ring size 18); no pretraining | ZINC-12k test, MAE, source's runs | Maziarka et al., arXiv:2304.10933, Table 1 |
| **ESA+** | **0.052 ± 0.001** | not stated in Table 6 (paper Fig. 2G: ESA has among the lowest parameter counts of the compared transformers) | official ZINC-12k train 10,000 | ESA edge-set attention + masked edge attention; **ESA+ = ESA + RWSE + ring-membership node features**; no pretraining, no external data | ZINC-12k test, MAE, paper's repeated-run protocol (Table 1 states 5 runs) | Buterez et al., *Nat Commun* 16 (2025), DOI 10.1038/s41467-025-60252-z, Table 6 |
| GIN (reference MPNN) | 0.526 ± 0.051 | ~500k budget | official ZINC-12k train 10,000 | none | ZINC-12k test, MAE, 10 runs | GraphGPS Table 3 (citing Xu et al.) |
| GatedGCN (reference MPNN) | 0.282 ± 0.015 | ~500k budget | official ZINC-12k train 10,000 | none | ZINC-12k test, MAE, 10 runs | GraphGPS Table 3 (citing Bresson & Laurent) |
| PNA (reference MPNN) | 0.188 ± 0.004 | ~500k budget | official ZINC-12k train 10,000 | none | ZINC-12k test, MAE, 10 runs | GraphGPS Table 3 (citing Corso et al.) |
| GSN (substructure counts + MPNN) | 0.101 ± 0.010 | ~500k budget | official ZINC-12k train 10,000 | precomputed graphlet/substructure counts as node features; no pretraining | ZINC-12k test, MAE, 10 runs | GraphGPS Table 3 (citing Bouritsas et al.) |
| CIN (higher-order MPNN) | 0.079 ± 0.006 | ~500k budget | official ZINC-12k train 10,000 | simplicial/cellular complexes built from the graph; no pretraining | ZINC-12k test, MAE, 10 runs | GraphGPS Table 3 (citing Bodnar et al.) |

Reading rules for this table:
- It is a **landscape**, not a controlled comparison: parameter counts, seeds, and
  structural priors differ. No ranking beyond the reported means is claimed.
- P3's own number is a **2-seed seed-mean of single models**, not a 10-run mean.
- P3 and F/M use masked-SSL on the official train structures (labels untouched).
  GraphGPS/GRIT/CSA/ESA+/GSN/CIN rows use no pretraining and no external data.
- The Phase 2a internal Monitor number **0.085345 must not appear in this table**
  and is never compared to official test numbers.

---

## 3. Data protocol

- `ZINC(root=data/ZINC, subset=True)`:
  - **official train**: 10,000 molecules. Supervised labels and masked-SSL
    structures both come from these 10,000 only.
  - **official valid**: 1,000 molecules. Development evaluation; already exposed
    historically (recipe tuning, full-10k transfer). It is the only split used for
    candidate promotion.
  - **official test**: 1,000 molecules. Terminal endpoint only, under §7.
- Supervised training uses **all 10,000 official-train rows**, including the rows
  that Phase 1a–2a called Monitor/Reserved. Those internal labels therefore stop
  being a held-out set in this round. No Phase 2b model is scored on the internal
  Monitor and no Phase 2b number is called "internal held-out".
- Reused frozen inputs (hashes in `configs/phase2b.yaml`):
  - `objects_train10k.pt` (10000), `objects_valid.pt` (1000), both built by the
    historical full-10k round from raw PyG `x/edge_index/edge_attr`;
  - `prep_train10k.json` (train-10k-only global-feature mean/scale);
  - `encoded_train.pt` for the raw supervised targets (the same file the historical
    full-10k/P3/R3 rounds used).
- No new atom/bond/junction/global descriptor. No label-derived quantity (including
  the cycle-penalty component or the label formula) is an input.
- Official test structures are loaded for forward passes only after promotion, and
  test labels only after both finalist prediction files are frozen and hashed.

---

## 4. Historical P3 reuse — audit required before any comparison

The reference is the **historical frozen P3** (BJOC-Null, 413,505 parameters,
full-10k masked-SSL encoder, R3 360-epoch cosine, last-5 soup). It is **reused, not
retrained**. Before training F/M, the audit must verify, read-only:

1. File hashes of `soup_R3_seed0.pt` = `a021d36e…` and `soup_R3_seed1.pt` =
   `b622824a…` match the hashes recorded by the R3 round and the terminal-test round.
2. Each soup loads into the current `BJOC-Null` class, has exactly 413505 parameters,
   no forbidden module, no new tensor.
3. For each seed, the R3 record's `init_full_sha256` (`d06d5e16…` / `c79e6b02…`),
   `init_group_state_sha256` (`d9ffbd15…` / `98d11fe9…`), `init_encoder_sha256`,
   `encoder_used_sha256` (`7165eb90…` / `54a232ed…`), R3 recipe values and batch-order
   hash (`3c964a71…` / `8710751a…`) are reproduced by this round's rebuild path.
4. The full-10k SSL encoder files `pretrain_encoder_s0.pt` (`006fabc1…`) and
   `pretrain_encoder_s1.pt` (`ff8d6474…`) reproduce the recorded post-SSL encoder
   hashes when loaded into the rebuilt trunk.
5. Replaying the two P3 soups on the 1,000 official-valid structures reproduces the
   historical valid MAE 0.079101 / 0.080586 (|Δ| ≤ 1e-6) and, where the historical
   `.npz` is present, max |Δ| ≤ 5e-5; the regenerated arrays must match the frozen
   historical **array** hashes `0fff2981…` / `30ac789c…` bit-for-bit.
6. Historical official-test prediction files `test_P30.npz` (`dd1dc3c4…`) and
   `test_P31.npz` (`b8699768…`) match their freeze record; their frozen array
   hashes are `4a217c2b…` / `4a339b09…` and their frozen test MAE is 0.069304 /
   0.070340 (seed-mean 0.069822). The arrays are regenerated from the frozen soups
   at the post-promotion test stage and must match those array hashes exactly.
7. The model / object / masked-pretraining source files are byte-identical to the
   hashes recorded when P3 was evaluated (terminal-test `code_hashes`).
8. The official-test structure fingerprint (`processed_pt` hash `bd7a1ceb…`,
   `structure_sha256` `75668e78…`) is reproduced by the test split before any
   finalist test forward pass.

If any item fails, the round records `PROTOCOL_OR_REPRODUCTION_UNRESOLVED` and does
not claim a strictly paired comparison. A minimal P3 replay is allowed **only** if
the frozen checkpoints cannot be reused; it is not planned and is charged to the
2.5 GPU-hour budget.

Disclosure: **P3's valid and test numbers and predictions are already-exposed
historical evidence.** They are not a new blind test and cannot be presented as one.

**Implementation note (added before the first GPU job, 2026-10-19).** The historical
`valid_*.npz`, `test_*.npz`, `valid_y_read_once.npy` and `test_y_read_once.npy`
files are gitignored and are absent from the compute checkout. The checkpoints,
object files and SSL encoders are present. Therefore the audit **regenerates** the
P3 valid predictions on CPU from the frozen, hash-verified soups and requires exact
equality with array-level SHA-256 hashes computed from the historical files
(`frozen_inputs`-adjacent `historical_prediction_provenance` and
`label_provenance` in `configs/phase2b.yaml`). At the post-promotion test stage the
same regeneration is applied to the test structures **before any test label read**,
and the finalist predictions are then frozen. Valid/test labels are re-derived from
the official PyG split at score time and must hash to the historical label arrays.
This is a portability fix with no change to splits, models, recipe, metrics,
promotion rule or test rule; it preserves exact numerical reuse rather than
approximate replay.

---

## 5. Frozen candidates F and M (unchanged from Phase 2a)

Both are the Phase 2a modules inserted after `AtomEncoder` and before
`BondEncoder`; the rest of the P3 trunk and reader is unchanged.

| arm | module | construction | new params | total params | adjacency |
|---|---|---|---:|---:|---|
| **F** | `NodeFFN-2` | 2 × pre-norm residual FFN (96→128→96), per atom | 49,984 | 463,489 | never reads `bond_index` / `bond_feat` |
| **M** | `LocalMP-2` | 2 × GINE round on both directions of every physical bond, shared 7→96 bond projection, same residual/norm shape, no gate | 50,752 | 464,257 | reads real physical bonds |

Frozen: hidden 128, 2 rounds, initialisation offsets 1013/2027, no zero-init gate,
no global attention, no bond-state update, no new descriptor. **The
`code/phase2a/phase2a_models.py` file is not modified.** A CPU loop-parity gate
reproduces the Phase 2a training-loop semantics and a replay gate reproduces the
frozen Phase 2a F/M Monitor predictions from their soups.

---

## 6. Training protocol — exactly four runs

F seed 0, M seed 0, F seed 1, M seed 1.

- Data: all 10,000 official-train rows, raw-y L1.
- Per-seed initialisation:
  - scratch `BJOC-Null` parameters from `bp.build_bjoc_null(seed, norm)` — must hash
    to the historical R3 `init_full_sha256` for that seed;
  - then the **seed-matched** full-10k masked-SSL encoder
    `pretrain_encoder_s{seed}.pt` — must hash to the historical
    `encoder_used_sha256` for that seed;
  - then the deterministic local-module initialisation from Phase 2a.
- Recipe: **R3** — Adam, lr 1e-3 cosine → 1e-5 per step, weight decay 1e-5,
  batch 128, gradient clip 5.0, **360 epochs** (79 steps/epoch, 28,440 updates),
  last-5-epoch parameter-mean soup, shuffle seed `seed + 101`.
- Batch order must reproduce the historical R3 full-order hash
  (`3c964a71…` seed 0, `8710751a…` seed 1). This is a hard assertion.
- **No** early stopping, **no** valid-based epoch selection, **no** soup selection,
  **no** seed-0-based change to seed 1, no regularisation/gating/structural channel.
- Smoke: a 2-epoch GPU run of each arm on a small row prefix only checks numerics
  and timing; smoke weights are never scored and never promoted. The formal budget
  must cover all four 360-epoch runs, otherwise the round stops as unresolved rather
  than reporting an undertrained model.

Arms are trained **before** any official-valid label is read in this round.

---

## 7. Official-valid evaluation (development split)

All six prediction files (P3 seed 0/1 regenerated from the frozen historical soups,
F seed 0/1, M seed 0/1) are frozen and hashed **before** the valid label read; the
P3 arrays must equal the historical array hashes, and each F/M file carries its
soup and environment record. Primary metric: **raw-y MAE** on the
1,000 official-valid molecules. Secondary: RMSE, median/p90/p95/p99 absolute error,
per-seed values, the two-seed mean, same-seed paired differences, molecule-level
paired bootstrap CI (5,000 resamples, unit = molecule, seed-averaged error vector),
fraction of molecules improved.

Subgroups (descriptive; never used to redefine the primary metric):
- ordinary `y > -3` and tail `y <= -3` (the historical valid/test convention);
- cycle-penalty proxy: `max(len(c) for c in nx.cycle_basis(G)) > 6`, recomputed from
  the PyG bond graph. This is an order-sensitive proxy for the label's cycle term,
  not an input and not the RDKit large-ring audit; the complement is the main
  "label-artefact-free" readout;
- `valid:0172` (`y = -20.3405`), the historically known extreme row;
- top-1 / top-5 / top-10 P3-absolute-error molecules removed.

Tail risk is reported with paired MSE-difference bootstrap CIs and per-molecule
contribution, **not** with the Phase 2a absolute `ΔRMSE ≤ +0.005` gate, which was a
single-molecule statistic at n = 1000.

Cost audit: parameters, forward GEMM FLOPs on one fixed 128-graph batch, mean/median
step time, training wall time, inference wall time on 1,000 molecules, and GPU-hours.
The historical P3 training wall from its R3 record is reported as the baseline cost
(same recipe, different run, not a controlled timing comparison).

### 7.1 Pre-registered promotion rule (no test unless this fires)

For a candidate `X ∈ {F, M}`, with `G_s = MAE(P3, seed s) − MAE(X, seed s)` on
official valid and `G_mean = (G_0 + G_1) / 2`:

A candidate **qualifies** only if **all** of:

1. `G_mean ≥ 0.003` (practical band; the same magnitude the task pre-registers);
2. `G_0 > 0` **and** `G_1 > 0` (no seed contradicts the other);
3. removing the single most candidate-favourable molecule (largest `d_i`, where
   `d_i = 0.5·[(|e_P3,s0,i|−|e_X,s0,i|) + (|e_P3,s1,i|−|e_X,s1,i|)]`) leaves
   `G_mean ≥ 0.002`;
4. removing the ten largest P3-absolute-error molecules leaves `G_mean ≥ 0.002`;
5. the cycle-penalty-proxy **complement** gain is `> 0` (the gain is not carried by
   the label's long-cycle term).

If both qualify:
- choose **M** only if `Gain_MP = MAE(F) − MAE(M)` has seed-mean `≥ 0.002`,
  **both** per-seed values `> 0`, and the complement subgroup does not reverse
  (`> 0`). Otherwise choose **F** (the computationally simpler arm).
- exactly **one** finalist is selected. Never send both to test and pick the better.

If neither qualifies, the round **stops at official valid**. No test prediction is
generated, no test label is read, and the report says so.

No RMSE threshold, no valid-based epoch or seed selection, and no post-hoc change of
`0.003` / `0.002` after seeing valid is permitted.

---

## 8. Terminal official-test evaluation (conditional, one shot)

Only the single finalist, with its two frozen seeds and frozen soups, is forwarded on
the 1,000 official-test structures. Order is mandatory:

1. select the finalist on valid under §7.1;
2. lock arm, seeds, soups, recipe and all hyper-parameters;
3. write both test prediction files and hash them;
4. only then read the official-test labels and score;
5. compare against the historical frozen P3 test predictions per molecule.

Test readout: per-seed MAE, seed-mean MAE, paired per-molecule delta vs P3
(seed-averaged), bootstrap CI, better fraction, ordinary/tail, penalty-proxy
complement, top-k removals. Decision labels for the performance dimension:

- `STANDARD_ZINC_GAIN_CONFIRMED` iff test seed-mean gain vs P3 `≥ 0.003`, **both**
  per-seed gains `> 0`, and the paired bootstrap CI95 lower bound `> 0`;
- `NO_PRACTICAL_ADVANTAGE` iff that conjunction fails (test-level practical gain not
  established).

**No result of the test read may change the model, seed, recipe, threshold or
trigger a follow-up training run.** The test score is a terminal benchmark
description; the official test has been used by earlier experiments in this project,
so this is not a first blind test.

---

## 9. Verdict dimensions

Multiple labels may hold; they must be separated by dimension.

Performance:
`STANDARD_ZINC_GAIN_CONFIRMED`, `VALID_GAIN_NOT_REPLICATED`, `NO_PRACTICAL_ADVANTAGE`

Mechanism:
`CAPACITY_GAIN_WITHOUT_MP_SPECIFICITY`, `MP_INCREMENT_SUPPORTED`

Protocol/reproduction:
`PROTOCOL_OR_REPRODUCTION_UNRESOLVED`

Mapping is mechanical from the frozen numbers:
- valid: neither candidate qualifies → `VALID_GAIN_NOT_REPLICATED` (performance);
  if the finalist is F, or M is not separated from F → 
  `CAPACITY_GAIN_WITHOUT_MP_SPECIFICITY`; if M is separated from F in both seeds with
  the §7.1 thresholds → `MP_INCREMENT_SUPPORTED`.
- test: §8 conjunction true → `STANDARD_ZINC_GAIN_CONFIRMED`; else
  `NO_PRACTICAL_ADVANTAGE` (performance at the standard endpoint).
- any §4 audit failure or non-finite/diverged formal run that cannot be resolved
  within budget → `PROTOCOL_OR_REPRODUCTION_UNRESOLVED`.

---

## 10. Budget, artefacts, isolation

- Hard cap **2.5 A100 GPU-hours** for everything (smoke + failures + four formal
  trainings + prediction/scoring). Expected demand ≈1.0–1.2 h; unused budget is fine.
  If four formal runs cannot fit, stop and report the blocker; do not substitute a
  shorter schedule and call it a result.
- New artefacts only under `results/phase2b/`. Historical files under
  `tracks/structure-contribution/results/*` are opened read-only and never rewritten.
- Every stage refuses to overwrite an existing artefact. Prediction arrays are frozen
  with a file hash and an array hash before labels are read.
- Disclosure: Phase 1e loaded all 10,000 official-train labels, so the internal
  Reserved split was historically exposed. In this round all 10,000 train labels are
  legitimately supervised training data and are not a held-out set. Official valid is
  an exposed development split. Official test has been read by earlier project
  experiments and is not a first blind test.

---

## 11. What would make the comparison non-paired

The round must not claim a strictly paired causal comparison if any of the following
fails: same per-seed trunk init hash, same seed-matched full-10k SSL encoder hash,
same 10k row set and batch order, same R3 recipe/epochs/soup rule, same P3 checkpoints
and model code hash, or a failed valid replay. Any such failure is reported explicitly
and maps to `PROTOCOL_OR_REPRODUCTION_UNRESOLVED` unless it is a benign reporting
gap that does not affect the numbers.
