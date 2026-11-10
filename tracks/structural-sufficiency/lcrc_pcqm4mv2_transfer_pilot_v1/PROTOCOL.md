# PROTOCOL — LCRC-W2 on PCQM4Mv2 (transfer pilot v1)

Round id: `lcrc-pcqm4mv2-transfer-pilot-v1`
Status: **frozen before the first formal training step**. §12 is an
append-only amendment log; nothing else in this file may change after the first
formal step.

Research question, model, data split, scale, recipe and stopping rules were fixed
by the execution instruction. This document records them, the measured dataset
facts, and the gates that must pass before a formal step is taken.

## 1. The question

Can the frozen **no-iterative-message-passing** LCRC-W2 (one atom encode, one
bond encode, one-shot S1/S2 local content, per-atom contribution `f`, global
correction `rho`, four-head attention readout) learn the PCQM4Mv2 2D molecular
graph regression task at all, and does increasing the training set from
300,000 to 1,000,000 molecules produce a meaningful Validation MAE improvement?

Only two formal runs. No W3, no DYNAMIC / CPC / RHIC arm, no new module, no
hyper-parameter search, no extra seed.

## 2. Data (official, 2D only)

| item | value |
|---|---|
| dataset | `ogb.lsc.PCQM4Mv2Dataset(root=data/ogb, only_smiles=True)` |
| task | HOMO–LUMO gap regression, eV |
| metric | MAE (lower is better), official `PCQM4Mv2Evaluator` |
| total molecules | 3,746,620 |
| official splits | `train` 3,378,606 · `valid` 73,545 · `test-dev` 147,037 · `test-challenge` 147,432 |
| partition | pairwise disjoint, union == `range(3,746,620)` (verified) |
| raw file | `pcqm4m-v2/raw/data.csv.gz`, sha256 `ab89a4707d5e37f19c3b5e69d1c5cbb7fc1a9665d38783c0c89df6675f34286b` |
| split file | `pcqm4m-v2/split_dict.pt`, sha256 `d913e5541af6a6279abfe23ecf0fdbe4bc86785e1d8fc24a8f3ef422a01d3722` |
| archive | `pcqm4m-v2.zip`, md5 `65b742bafca5670be4497499db7d361b` (= the md5 documented in `ogb/lsc/pcqm4mv2.py`) |
| provenance | official DGL-hosted OGB-LSC archive; downloaded once, then read locally |

Labels: `train` min 0.3755 / mean 5.6895 / median 5.5838 / max 47.0240 /
std 1.1621 eV; `valid` min 0.7347 / mean 5.4451 / max 15.9350 / std 1.2500 eV;
both splits fully finite. Hidden (`test-dev`, `test-challenge`) labels are never
read, written or scored: the label artifact stores Train+Valid only and NaN
everywhere else, and no code path indexes a hidden row.

Only SMILES-derived 2D information is used. No DFT 3D geometry, no other
dataset, no pretrained weight (`only_smiles=True`, `uses_3d_geometry=false`).

## 3. Fixed training subsets

One NumPy `default_rng(42).permutation(3_378_606)` of the official Train row
ids, generated once:

- `subset_300k` = first 300,000 permuted rows,
- `subset_1m` = first 1,000,000 permuted rows,
- therefore `subset_300k ⊂ subset_1m ⊂ official_train` **structurally** (prefix
  property), with original official row ids preserved.

Consequences: the 1M feature cache is built once in permutation order, and the
300K run is exactly cache rows `0..299_999` of it (no second cache, no second
subsampling). Index files, hashes and the containment proof are written to
`analysis/pcqm_subset_plan.json` (+ `cache/pcqm_subset_rows.npy`,
`cache/pcqm_permutation.npy`). The 3,378,606-row permutation is never used as a
training set here.

RNG separation: sampling 42 · batch order 0 · model init 0 · monitor-row
selection 43.

## 4. Model (migrated, not redesigned)

`bs_model.BSModel` (= historical `TULCRC(variant="C1")` +
`nmp_model.AttentionPoolReadout`) built with `bs_common.config_for_scale(2)` and
the frozen initialisation (FULL→C1 remap, readout from the private generator
`seed*1_000_003+17`, `readout.fc2` zeroed). W2 widths, unchanged:

| width | W2 (scale 2) |
|---|---:|
| `atom_hidden` / `atom_out` | 256 / 192 |
| `bond_hidden` / `bond_out` | 320 / 192 |
| `stat_rank` | 64 |
| `local_hidden` / `width` | 384 / 128 |
| `rho_hidden` | 128 |
| attention scorer hidden | 64 |
| attention pool hidden | 176 |
| heads | 4 |
| **outputs** | **1 (regression)** |

Migration changes (complete list): output width 128 → 1; loss masked BCE → L1 on
the standardised gap; PCQM4Mv2 SMILES/labels/split instead of the PCBA graph
dataset; training/evaluation/cache/plumbing adapted.

Parameter accounting is analytic and asserted at build time
(`analysis/pcqm_smoke.json`): base 815,330 + readout 123,749 = **939,079**
(vs 994,324 historical; the only difference is the output layer). The same
builder at `n_out=128` still produces exactly 994,324 parameters, which is
checked in every formal run's audit. No parameter is added to compensate.

Computation path (audited by hooks, must be exactly 1 call each per forward):
one `atom_encoder`, one `bond_encoder`, one `local` (S1/S2), one `f`, one `rho`,
one `readout`. No hidden-state propagation along bonds, no second pass.

## 5. Input features (frozen `lipo_v1` schema)

`SMILES → mb2_rdkit adapter → MolecularGraph → LCRC S1/S2 feature cache`, i.e.
the `lcrc_cross_task_preparation_v1` `lipo_v1` definition, unchanged: 51-d atom
features, 13-d bond features, 9 atom scalars, 43 graph statistics, S1 joint
`(neighbour type, bond type)` codes (80 categories), S2 typed paths
`(middle type, bond1, bond2)` (400 categories) with joint `(end type, bond2)`
codes and the real triangle-closure flag. No extra descriptor, fingerprint,
graph statistic or vocabulary is added.

Deterministic codes (`int32`), no truncation of S2: every record of every
molecule is materialised (the maximum per-molecule S2 count is recorded).
Disconnected molecules are supported by the schema with finite statistics.

Cache: shards (4,000 rows each, multi-process) → memory-mapped `.npy` directory
(`npy-directory-v1`), read-only shared by both runs. Only the 1M Train subset
and the full official Validation split are built; the remaining Train rows and
the hidden Test splits are not.

## 6. Pre-training audit gates

Must all pass before a formal step (`analysis/pcqm_*_cache_verify.json`,
`analysis/pcqm_train1m_oov_audit.json`, `analysis/pcqm_smoke.json`):

1. dataset version / total / split keys / row uniqueness / label finiteness,
2. atom-class and bond-class coverage with the `UNK` share **and the elements
   that land in `UNK`**,
3. parse-mode ledger; any SMILES the adapter cannot process blocks the cache
   (no dropping, substituting or renumbering),
4. no non-finite structural feature, max S2 count recorded,
5. exact rebuild of sampled molecules from their SMILES (all stored fields),
6. all-atom S2 count identity `#S2(v) == Σ_{w∈N(v)}(deg(w)-1)` and
   `#S1 == 2·#bonds`,
7. dimensions match the spec; model has one output; analytic parameter count,
8. atom renumbering does not change the graph-level prediction,
9. L1 loss / backward / optimizer step produce no NaN/Inf,
10. de-normalisation is exactly invertible; official MAE == independent NumPy
    MAE,
11. no Validation or hidden row in any training batch,
12. cache mmap read, checkpoint save and full restore (optimizer + RNG + order).

## 7. Normalisation (shared, Train-only)

- Feature standardisation of `atom_scalar` and `global`: mean/std computed
  **once** from `subset_300k` (cache rows 0..299,999) and used by both runs.
  Constant-on-fit columns use scale 1.0 and are recorded. Validation/Test
  statistics are never used.
- Regression target: `z = (gap_eV − mean_gap_300k) / std_gap_300k` (ddof 0), the
  same fixed constants for both runs. L1 loss on `z`.
- Inference: `gap_pred_eV = z_pred * std_gap_300k + mean_gap_300k`. **Every
  reported MAE is computed in eV.**
- Constant-median baseline: `median(gap)` of `subset_300k`, evaluated on the full
  official Validation split (CPU only).

## 8. Recipe (identical for both runs)

seed 0 · 40 epochs · batch 128 · AdamW · peak LR 3e-4 · weight decay 1e-3 ·
dropout 0.10 · betas (0.9, 0.999) · eps 1e-8 · 1 epoch linear warmup
3e-5 → 3e-4 · per-step cosine to 1e-5 over the remaining steps · gradient clip
5.0 · FP32 · no early stopping · fresh initialisation per run (the 300K
checkpoint never initialises the 1M run).

Steps per epoch = ⌈rows/128⌉ and the scheduler horizon is that number × 40, so
both runs really run 40 epochs:

| run | rows | steps/epoch | total steps |
|---|---:|---:|---:|
| `PCQM_W2_300K` | 300,000 | 2,344 | 93,760 |
| `PCQM_W2_1M` | 1,000,000 | 7,813 | 312,520 |

Batch order: a fresh `default_rng(0)` permutation every epoch; the per-epoch
hashes are precomputed and verified at run time and on resume. Same seed, same
model init (the initial state fingerprint is compared across the two runs).

No mid-run change to batch size, LR, dropout, loss or epochs is permitted.

## 9. Evaluation and checkpoints

After **every** epoch:

1. inference on the **full official Validation split** (73,545 molecules),
2. official `PCQM4Mv2Evaluator` MAE (eV) plus an independent
   `mean(|y_pred − y_true|)` cross-check on the same predictions,
3. MAE on the fixed 20,000-row Train monitor subset (identical rows in both
   runs; drawn from `subset_300k` with `default_rng(43)`),
4. training L1, LR, times, step-time percentiles, GPU peak memory, RSS.

Checkpoints: `checkpoint_best_valid.pt` (lowest Validation MAE; ties → earlier
epoch), `checkpoint_epoch40.pt`, `checkpoint_last.pt` (full resume state:
model + AdamW + RNG + plan prefix). Curves: `train_curve.csv`, `run.json`.
Hidden Test data are never used for selection or reporting.

## 10. Execution, budget and stopping rules

- Hosts: `res` (single box, process backend) through `rr`; GPU1 preferred, GPU0
  only if a live occupancy check shows it free (no co-tenant interference).
- GPU budget: target 6 A100-hours, hard cap 9. Measured cost projection from the
  smoke step is recorded before the first formal step. If the projection
  exceeds the hard cap: run `PCQM_W2_300K` first and start `PCQM_W2_1M` only if
  it still fits; otherwise stop and report the resource limit. Epochs, subset
  size and evaluation frequency are never reduced to manufacture a result.
- Recovery: resume from `checkpoint_last.pt`; earlier results are never
  discarded and the recipe is never silently changed.
- Stop after the two runs: no full 3.38M-molecule run, no architecture or
  hyper-parameter search.

## 11. Deliverables

`REPORT.md` (first page: the required comparison table), the two epoch-1..40
Validation curves, best/epoch-40 metrics, 300K→1M improvement (absolute and
relative), the last-10-epoch trend, a *positioning-only* comparison with the
public full-data GCN / GIN / GIN-virtual numbers, the parsing/OOV audit, cache
size and build/run performance, code version, commands, checkpoint hashes and
the resource ledger. Large caches/checkpoints stay git-ignored with their real
paths and SHA256 recorded.

Explicitly out of scope: W3 / DYNAMIC / CPC / RHIC, extra seeds, extra
architectures, hyper-parameter search, leaderboard submission, any Test MAE.

## 12. Amendments (append-only)

- 2026-11-10 — protocol written. Measured dataset facts (§2) and the recipe
  (§8) come from `analysis/pcqm_data_audit.json` and the instruction; no formal
  model step had been taken.
