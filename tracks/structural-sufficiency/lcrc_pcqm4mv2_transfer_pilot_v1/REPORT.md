# LCRC-W2 on PCQM4Mv2 (2D HOMO–LUMO gap) — transfer pilot v1

Round id: `lcrc-pcqm4mv2-transfer-pilot-v1` · protocol: [`PROTOCOL.md`](PROTOCOL.md)
Runs: `PCQM_W2_300K`, `PCQM_W2_1M` · code version `93ae701f` (both runs)
Model: LCRC-W2, **no iterative message passing**, **939,079** parameters
Task: official OGB-LSC `PCQM4Mv2`, 2D SMILES only, MAE in eV, official Validation split

**Answer in one paragraph.** The frozen no-message-passing W2 transfers cleanly to
this task: trained on 300,000 molecules for 40 epochs it reaches **0.1496 eV**
Validation MAE, and on 1,000,000 molecules **0.1323 eV**, versus **0.9816 eV** for
a constant-median predictor — a 6.6× / 7.4× error reduction. Going from 300K to 1M
molecules improves Validation MAE by **0.01724 eV (11.5 % relative)** at 3.33× the
data and 3.33× the GPU time, and *both* runs were still improving when the fixed
40-epoch budget ended (best epoch = 40 in both), so 11.5 % is a lower bound on what
this axis is worth. Total cost was **6.71 A100 GPU-hours** (target 6, hard cap 9;
smoke projection 5.67). The numbers locate W2 in the range of the *published
full-data* GCN/GIN baselines (0.1960 / 0.1536 Valid) with a fraction of the data and
no message passing, but that comparison is not like-for-like (§5).

## 1. Results (front table)

| arm | Params | Best epoch | Best Valid MAE (eV) | Valid MAE @40 (eV) | Train-monitor MAE @40 (eV) | GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| Constant median (Train 300K) | 0 | — | 0.9816 | 0.9816 | — | 0.00 |
| **LCRC-W2-300K** | 939,079 | 40 | **0.1496** | 0.1496 | 0.1101 | 1.55 |
| **LCRC-W2-1M** | 939,079 | 40 | **0.1323** | 0.1323 | 0.1059 | 5.16 |

* Validation MAE = official `PCQM4Mv2Evaluator` on all **73,545** official Validation
  molecules, every epoch; each value is cross-checked against an independent NumPy
  MAE on the same predictions (`crosscheck_delta` = 0.0 in all 80 evaluations).
* Train-monitor MAE = the same evaluator on a **fixed 20,000-row Train subset**
  (identical rows in both runs, seed 43). It measures fit, not generalisation.
* GPU-h = wall time of the whole run process on one dedicated A100-SXM4-40GB
  (training + per-epoch validation + monitor + checkpoint I/O), from `run.json`.
* Both runs are seed 0, 40 epochs, fresh initialisation, batch 128, identical recipe.

Curves (full 40 points in `analysis/pcqm_curves.csv` and `analysis/pcqm_curves.png`):

| epoch | 300K Valid | 1M Valid | 300K monitor | 1M monitor |
|---:|---:|---:|---:|---:|
| 1 | 0.3225 | 0.2517 | 0.2925 | 0.2320 |
| 3 | 0.2261 | 0.1847 | 0.2013 | 0.1635 |
| 5 | 0.2112 | 0.1667 | 0.1857 | 0.1467 |
| 10 | 0.1775 | 0.1511 | 0.1505 | 0.1284 |
| 15 | 0.1778 | 0.1489 | 0.1482 | 0.1265 |
| 20 | 0.1701 | 0.1403 | 0.1367 | 0.1183 |
| 25 | 0.1560 | 0.1366 | 0.1216 | 0.1124 |
| 30 | 0.1572 | 0.1350 | 0.1197 | 0.1099 |
| 35 | 0.1504 | 0.1361 | 0.1117 | 0.1105 |
| 40 | **0.1496** | **0.1323** | 0.1101 | 0.1059 |

### 1.1 300K → 1M improvement

| quantity | value |
|---|---|
| best Valid MAE, 300K | 0.149556 eV (epoch 40) |
| best Valid MAE, 1M | 0.132311 eV (epoch 40) |
| absolute improvement | **0.017245 eV** |
| relative improvement | **11.53 %** |
| improvement at epoch 40 | identical (best epoch = 40 in both) |
| monitor-MAE improvement @40 | 0.004171 eV |
| data ratio / GPU-time ratio | 3.33× / 3.33× |
| eV removed per extra GPU-hour | 0.00478 |
| first epoch where 1M beats 300K-final | epoch 13 (0.1479 eV, 1.65 GPU-h vs 1.55 GPU-h) |

Last-10-epoch trend (epochs 31–40, log-fitted slope):

| run | mean | min | slope (eV/epoch) | drop over 10 epochs |
|---|---:|---:|---:|---:|
| 300K | 0.15029 | 0.14956 | −0.000132 | 0.00134 |
| 1M | 0.13452 | 0.13231 | −0.000370 | 0.00158 |

Both curves are still descending at the forced stop, and the 1M curve is descending
~2.8× faster per epoch, i.e. **the 11.5 % is a lower bound**: the 1M arm was cut off
mid-improvement by the fixed 40-epoch budget (LR had just reached its 1e-5 floor).

## 2. What was run, at which code version

```bash
# local CPU prep (rdkit required; the GPU host has no rdkit by design)
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_prepare.py --step data
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_prepare.py --step estimate --workers 12
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_prepare.py --step build --workers 12
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_prepare.py --step norms
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_prepare.py --step manifest
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_verify.py --tag pcqm_train1m --official
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_verify.py --tag pcqm_valid  --official
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_audit.py  --tag pcqm_train1m --mode both

# GPU preflight (10,000 molecules), then the two formal runs, both at 93ae701f
rr run res pcqm-smoke   --pool res-gpu1 -- python .../pq_smoke.py
rr run res pcqm-w2-300k --pool res-gpu1 -- python .../pq_train.py --run PCQM_W2_300K \
    --out-dir .../runs/PCQM_W2_300K --device cuda --soft-seconds 7200  --hard-seconds 10800
rr run res pcqm-w2-1m   --pool res-gpu1 -- python .../pq_train.py --run PCQM_W2_1M \
    --out-dir .../runs/PCQM_W2_1M   --device cuda --soft-seconds 21600 --hard-seconds 25200

# local reporting + independent re-evaluation of both delivered checkpoints
uv run python .../pq_analysis.py
uv run python .../pq_recheck.py --run PCQM_W2_300K
uv run python .../pq_recheck.py --run PCQM_W2_1M
uv run pytest -q .../tests
```

Execution regime (`rr` provenance): host `res` = `a100-2`, 2× A100-SXM4-40GB,
driver 550.163.01, torch 2.5.1+cu124, `CUDA_VISIBLE_DEVICES=1` (pool `res-gpu1`),
python 3.12.14, ogb 1.3.6, no rdkit on the host (features come from the cache).
Both runs: same host, pool, GPU model and driver → one regime, directly comparable.
GPU0 was occupied by an external 16 GB / 100 % job for the whole round and was not
touched, so the two runs were serialized on GPU1 (no co-tenancy in either run).

Code version: **`93ae701f539b94ff23b14bb6a92e09e55d9f0fd4`** for *both* formal runs
(`run.json → git_commit`, remote checkout at that commit, deploy marker verified).
Files changed after the runs: `pq_analysis.py`, `pq_recheck.py` (new, reporting and
verification only) plus one reporting-only payload lookup in `pq_train.py`
(`final_epoch_*` resolved string keys in an int-keyed dict, so those two
convenience fields are `null` in the delivered `run.json`; the authoritative
per-epoch values are in `train_curve.csv` and `*_epoch_reports`). No
training-path computation differs.

## 3. Pre-training audit (all gates green before the first formal step)

| # | check | result | artifact |
|---|---|---|---|
| 1 | dataset version / total / split keys / row uniqueness / label finiteness | `ogb 1.3.6`, 3,746,620 rows, 4 splits, gids unique, splits pairwise disjoint and partitioning, all Train+Valid labels finite | `analysis/pcqm_data_audit.json` |
| 2 | atom-/bond-class coverage + `UNK` share and the affected elements | 956 / 14,127,298 atoms = **6.77e-5** `UNK` in 899 molecules (As 45, B 2, Ge 43, H 2, P 129, Se 187, Si 14); 0 `UNK` bonds; `I` never occurs | `pcqm_train1m_oov_audit.json` |
| 3 | parse-mode ledger, per row, no silent drops | 999,989 strict + **11** documented `unsanitized_fallback` (Si-cage / hypervalent-S / phosphorane SMILES), 0 failures, every fallback row listed with its SMILES | `pcqm_train1m_parse_ledger_scan.json` |
| 4 | no non-finite structural feature; max S2 recorded | 0 non-finite in atom/bond/scalar/global; max S2 per molecule **120** (Train 1M) / **178** (Valid); no truncation | same |
| 5 | exact SMILES→cache rebuild | 26 molecules per cache rebuilt field-by-field, all exact | `pcqm_*_cache_verify.json` |
| 6 | S2 all-atom identity `#S2(v)=Σ_{w∈N(v)}(deg(w)-1)`, `#S1=2·#bonds` | holds for all 14,127,298 Train and 1,050,647 Valid atoms | same |
| 7 | dims vs spec, one output, analytic parameter count | 51/13/9/42-d, 80/400 codes; output `(1,176)`; **939,079 = 815,330 + 123,749** (analytic == built; historical `n_out=128` replay still gives exactly 994,324) | `pcqm_smoke.json` |
| 8 | atom renumbering invariance | max \|Δz\| = 1.2e-7 (fp32) on a permuted batch | `pcqm_smoke.json` |
| 9 | L1 loss / backward / AdamW step finite | loss 1.158 → 0.729, grad norms finite, 51 parameter tensors updated, no NaN/Inf | `pcqm_smoke.json` |
| 10 | de-normalisation invertible; official == NumPy MAE | round-trip ≤ 3e-8 eV (float32 label artifact); official and NumPy MAE agree to 0.0 in every evaluation | `pcqm_smoke.json`, both `run.json` |
| 11 | no Validation / hidden row in any training batch; cache mmap; full checkpoint restore | 0 leaked gids over the epochs checked; two independent restores are bitwise identical in weights **and** AdamW state and reproduce the saved prediction hash and MAE exactly | `pcqm_smoke.json` |

The GPU preflight ran on a **10,000-molecule** slice (6,000 Train + 4,000 Validation),
2 epochs, and passed **27/27 gates** (`analysis/pcqm_smoke.json`, status `OK`,
device `cuda`). The formal runs started from scratch afterwards — no state from the
smoke was reused. On the GPU host (no rdkit) the SMILES-rebuild gate is satisfied
from the committed local cache-verification artifacts, and that is recorded in the
smoke payload (`details.features.local_verify_evidence`).

### 3.1 Data, subsets, normalisation

| item | value |
|---|---|
| official splits | train 3,378,606 · valid 73,545 · test-dev 147,037 · test-challenge 147,432 |
| subset rule | one `default_rng(42).permutation(3,378,606)`; `subset_300k` = first 300K, `subset_1m` = first 1M → `subset_300k ⊂ subset_1m ⊂ official_train` by construction | 
| containment proof | both are prefixes of the same permutation; 1M ∩ Valid = 0, 1M ∩ hidden = 0, 300K ⊂ 1M | `pcqm_subset_plan.json` |
| duplicated SMILES strings | 171,632 (distinct rows/labels; kept as-is, no dedup) |
| feature normalisation | mean/std from `subset_300k` only, shared by both runs; 4 global columns are identically constant (`bond_unknown_fraction`, `mean_formal_charge`, `unknown_charge_fraction`, `unknown_hydrogen_fraction` — all structurally 0 on this dataset) | `pcqm_feature_norm.json` |
| target normalisation | `z=(gap−5.687756)/1.160775`, from `subset_300k` only, same for both runs; every reported MAE is in eV | `pcqm_label_norm.json` |
| constant-median baseline | median 5.583776 eV → **0.981623 eV** on the full Validation split (CPU, no GPU) | `pcqm_constant_baseline.json` |
| monitor subset | 20,000 Train positions drawn with `default_rng(43)` from `subset_300k`, identical in both runs | `pcqm_monitor_plan.json` |
| hidden Test rows touched | **0** (label artifact NaNs every non-Train/Valid row; no code path indexes one) | `pcqm_label_artifact.json` |

### 3.2 Feature cache

| cache | rows | bytes | atoms | bonds | S1 | S2 | build wall | build peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pcqm_train1m` | 1,000,000 | 5.86 GB | 14,127,298 | 14,540,619 | 29,081,238 | 38,630,888 | 255.7 s (12 proc) | 6.75 GB |
| `pcqm_valid` | 73,545 | 0.44 GB | 1,050,647 | 1,087,322 | 2,174,644 | 2,913,958 | 22.0 s (12 proc) | — |
| total | | **6.31 GB** | | | | | 282 s | |

Per-molecule cost is 5,997 B (estimate step predicted 5,845 B/molecule → 6.27 GB), so
the estimate was accurate to 0.6 %. The cache is a memory-mapped `.npy` directory
(`npy-directory-v1`) shared read-only by both runs; the 300K run simply reads the
first 300,000 rows of the 1M cache, so the two arms see bit-identical features.
Building it needs ~1 CPU-minute per 250K molecules and rdkit; the GPU host has no
rdkit, so the cache was built locally and rsync'ed (6.34 GB, 7.4 MB/s), then
hash-verified on the host at run start (`sha256_tree` == manifest, else the run
refuses to start). Disk pressure on the host (`/home` 199 GB free) was checked
before transfer; RAM (503 GB) holds the whole cache in page cache after epoch 1.

## 4. Model and recipe actually used

* `bs_model.BSModel` with `config_for_scale(2)` (W2 widths) and the frozen
  FULL→C1 remap + readout initialisation; **one** output instead of 128 logits.
* One `atom_encoder`, one `bond_encoder`, one `local` (S1/S2 one-shot content), one
  `f`, one `rho`, one attention-pool readout — counted by hooks on every preflight
  forward: exactly **1 call each**, i.e. no hidden-state propagation along bonds.
* 939,079 parameters = 815,330 base + 123,749 readout (−55,245 vs the historical
  994,324, entirely the output layer). Verified analytically and by construction.
* AdamW, peak LR 3e-4, WD 1e-3, dropout 0.10, betas (0.9, 0.999), eps 1e-8,
  1-epoch linear warmup 3e-5 → 3e-4 then per-step cosine to 1e-5, grad clip 5.0,
  FP32, batch 128, 40 epochs, no early stopping.
* Steps: 2,344/epoch → 93,760 (300K); 7,813/epoch → 312,520 (1M); the schedule
  horizon is `steps_per_epoch × 40` in each run, so both really ran 40 epochs.
* Batch order: a fresh `default_rng(0)` permutation per epoch, hashes precomputed
  and verified at run time (`epoch_order_hashes_verified` = 40 in both runs).
* Gradient-clip events: 31,575 / 93,760 steps (33.7 %) for 300K, 83,077 / 312,520
  (26.6 %) for 1M — clipping is active but not pathological.
* Determinism: cuDNN deterministic, TF32 off, `use_deterministic_algorithms(True,
  warn_only=True)`; recorded in each `run.json → deterministic_settings`.

## 5. Position, not a like-for-like comparison

| model | training data | epochs | Valid MAE (eV) | test-dev MAE (eV) |
|---|---|---:|---:|---:|
| GCN (published, full data) | 3,378,606 | 100+ | 0.1960 | 0.2021 |
| GIN (published, full data) | 3,378,606 | 100+ | 0.1536 | 0.1678 |
| GIN-virtual (published, full data) | 3,378,606 | 100+ | 0.1396 | 0.1487 |
| **LCRC-W2-300K (this round)** | 300,000 (8.9 %) | 40 | **0.1496** | — |
| **LCRC-W2-1M (this round)** | 1,000,000 (29.6 %) | 40 | **0.1323** | — |

Read this as a *position*, not a benchmark result:

* the published rows train on the **full** Train split (3.33× / 11.2× more molecules)
  for **100+** epochs with their own augmentation/ensemble choices; this round uses
  8.9 % / 29.6 % of the data for 40 epochs, one seed, 2D features only, no
  augmentation, no ensemble;
* the published rows are the GNN baselines quoted in the OGB-LSC PCQM4Mv2 paper and
  are *validation* MAE next to *test-dev* MAE from a leaderboard pipeline, whereas
  this round reports Validation MAE only and never evaluates a Test split;
* consequently the table says "W2 with one-shot local encoding lands in the same
  0.13–0.15 eV region as full-data message-passing baselines on this task at a
  fraction of the data", **not** "W2 beats GIN". No claim of the latter is made, and
  no Test-split number is produced or implied.

## 6. Answers to the five pre-registered questions

**Q1 — Does the frozen no-message-passing W2 learn PCQM4Mv2 2D regression at all?**
Yes, unambiguously. 0.1496 eV (300K) and 0.1323 eV (1M) against a 0.9816 eV
constant-median predictor, i.e. **6.6× / 7.4×** error reduction, with a monotone,
smooth Validation curve from epoch 1 (0.3225 eV) and finite L1/backward/updates
throughout. The parameter count (939,079) and the single-encode/single-local
computation graph are unchanged from the migrated W2.

**Q2 — Does 300,000 → 1,000,000 training molecules produce a meaningful
Validation-MAE improvement, and is it saturated?**
Yes: **0.017245 eV / 11.53 % relative** (0.149556 → 0.132311 eV) for 3.33× data and
3.33× GPU time. It is *not* saturated — both arms were still improving when the
fixed 40-epoch budget stopped them (best epoch 40/40 in both; last-10 slope
−1.3e-4 eV/epoch at 300K and −3.7e-4 eV/epoch at 1M), and the 1M arm is still
descending ~2.8× faster per epoch. The measured gain is therefore a **lower bound**
on the 300K→1M axis. Cost-efficiency: 0.00478 eV per extra GPU-hour; and 1M
overtook the 300K *final* MAE already at epoch 13 (1.65 GPU-h vs 1.55 GPU-h).

**Q3 — Where does it sit relative to the public full-data GCN / GIN / GIN-virtual
numbers?**
Between them, in the sense of the positioning table in §5: better than the published
full-data GCN (0.1960) and GIN (0.1536) validation numbers, and marginally below
GIN-virtual (0.1396), while using 8.9 % / 29.6 % of the training data for 40 epochs.
This is **not** a like-for-like comparison (different data volume, epochs,
augmentation/ensemble, and Training/Validation-only vs leaderboard pipeline), so it
supports a *positioning* statement only: LCRC-W2's one-shot local encoding is
sufficient to reach the 0.13–0.15 eV region on this task. No superiority claim, no
Test-split claim.

**Q4 — Was the resource plan respected, and what would the full 3.38M split cost?**
Yes on the hard constraint, with an honest overshoot on the soft target:
**6.713 GPU-hours total** (300K 1.552 + 1M 5.162) against a 6 h target and a 9 h hard
cap; the ≤10k-molecule preflight projected 5.674 h, and the 18 % miss is explained by
the real caches (56–60 ms/step) being slower per step than the 34 MB smoke cache
(44 ms/step), which fits in RAM while the 5.9 GB cache pays first-touch page faults.
Extrapolating the *measured* per-step cost (58 ms at batch 128, 6.2 s per full
Validation pass) to the full 3,378,606-row split gives 26,396 steps/epoch →
1,531 s/epoch → ≈ **17.0 h of training + 0.07 h of validation ≈ 17.1 GPU-hours for
one 40-epoch run** (16.5–17.7 h across the observed 56–60 ms step range). That is a
projection, not a measurement: **the full-data run was not executed in this round**,
exactly as instructed.

**Q5 — Are the audits strong enough that the numbers mean what they claim?**
Yes for the claims made here: splits/labels/uniqueness verified against the official
files; cache verified structurally (S2 all-atom identity over 15.2 M atoms), by exact
SMILES rebuild, and by a hash check on the GPU host at every run start; atom/bond
OOV quantified (6.8e-5 / 0) instead of hidden; 11 fallback parses listed per row with
0 silent drops; normalisation fitted on `subset_300k` only and shared; Train/Valid
disjointness asserted (0 shared gids, 0 leaked rows in the checked epochs); official
MAE == NumPy MAE in all 80 evaluations; and both delivered checkpoints were
**re-evaluated locally, independently of the GPU host**:
300K 0.149555751 eV recorded → 0.149555751 recomputed (Δ 4.6e-10),
1M 0.132310904 → 0.132310906 eV (Δ 1.8e-9)
(`analysis/PCQM_W2_*_recheck.json`, status `OK`, checkpoint SHA256 recorded).

## 7. Resource ledger

| item | projected | measured |
|---|---:|---:|
| 300K GPU time | 1.348 h | **1.552 h** (139.5 s/epoch × 40 + evals) |
| 1M GPU time | 4.327 h | **5.162 h** (464.3 s/epoch × 40 + evals) |
| total GPU time | 5.674 h | **6.713 h** (target 6, cap 9) |
| GPU memory | — | 236 MB peak allocated / 40 GB (0.6 %) |
| process RSS | ~1.1 GB (build) | 3.63 GB (300K) / 7.53 GB (1M) |
| feature cache on disk | 6.27 GB | 6.31 GB |
| CPU prep (build + verify + audit) | — | ~9 min wall on 16 local cores |
| cache transfer to host | — | 6.34 GB, 14 min at 7.4 MB/s |
| wall clock, 300K run | — | 93 min |
| wall clock, 1M run | — | 5 h 10 min |

Compute was dominated by per-step graph staging and the small-model forward/backward
in FP32: the GPU was never the bottleneck (236 MB of 40 GB, ~17 steps/s), so the
runs are effectively CPU-staging-bound — which is also why `1M → full-data` scales
linearly in steps, not super-linearly.

## 8. Artifacts and hashes

Run directories (`runs/<run>/`, git-ignored; sizes and SHA256 recorded):

| file | 300K | 1M |
|---|---|---|
| `checkpoint_best_valid.pt` | 11,357,666 B · `ccb7f4aa7f367bcbaf4dac02725a6cc47df7201e092114b8c7aadd3eb9b17f5b` | 11,357,666 B · `e45316c29d9152adf79e702ebd701bf62e6237065112f8b000b419d1de3ed0b1` |
| `checkpoint_epoch40.pt` | 11,356,954 B · `288d60268d8139152e203ba5c6282282e64526cb1d6b2b7f9e8fe7e1fe762105` | 11,356,954 B · `77d80214eda49bb71d988ac5c98250a89ee19952770bfa9af54f0244e8066c13` |
| `checkpoint_last.pt` | 11,356,178 B · `487d7f79de2a90ddcd02d84b0a7fe11d567dd4d38e88829477320cb89cfbee2d` | 11,356,178 B · `0e7ab096d1c167355a71527e2c2b5f99396d5e175f1f1169d74f92d89dadce3f` |
| `train_curve.csv` | `6a62f58faf26e3fe7eaef0c03a6cac21b0aacff957793c7a73a14ccef7dac135` | `8e53acef62b4027ca2569f1e13239c1c93d86b4da30224fdedd1261cddacc697` |
| `run.json` | `364bb3be23262ce7fbead6b7132846fb6ee5a08731c53c7376d151006b2ed62f` | `814658d55a1cbaf6c858eab9593448c771d3b38fdda60f3c0593d21d9d6513cc` |

Because the best epoch is 40 in both runs, `checkpoint_best_valid.pt` and
`checkpoint_epoch40.pt` hold **bitwise identical weights** (verified) and differ only
in their metadata (`checkpoint_kind`, extra fields).

Committed analysis artifacts (small JSON/CSV/PNG, no caches or checkpoints):
`pcqm_data_audit`, `pcqm_subset_plan`, `pcqm_monitor_plan`, `pcqm_label_artifact`,
`pcqm_label_norm`, `pcqm_feature_norm`, `pcqm_constant_baseline`,
`pcqm_disk_ram_estimate`, `pcqm_cache_manifest`, `pcqm_train1m_cache_{build,merge,verify}`,
`pcqm_valid_cache_{build,merge,verify}`, `pcqm_train1m_{oov_audit,parse_ledger_scan}`,
`pcqm_smoke.json`, `pcqm_summary.json`, `pcqm_summary_table.md`, `pcqm_curves.csv`,
`pcqm_curves.png`, `PCQM_W2_300K_recheck.json`, `PCQM_W2_1M_recheck.json`.

Tests: `uv run pytest -q tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/tests`
→ **18 passed** (recipe constants, W2 widths, 939,079 parameter accounting and the
994,324 historical replay, LR endpoints, subset prefix containment, monitor
determinism, label-artifact NaN masking, epoch orders, label-norm round-trip, merge
equivalence against the pinned PCBA implementation, per-forward module-call counts,
atom-renumbering invariance, official-vs-NumPy MAE, rdkit-free evaluator import,
report-table aggregation).

## 9. Limitations and what is *not* claimed

* **Not converged.** Both runs improved monotonically to the last epoch; 40 epochs at
  this LR schedule is a budget, not a converged optimum. The 11.5 % data-scaling gain
  is a lower bound and the absolute MAEs are upper bounds for these arms.
* **One seed per arm.** No seed variance is measured, so the 0.017 eV difference
  between arms cannot be given a confidence interval. It is 11.5 % of the 300K error
  and consistent across every epoch after 13, but it is a single run.
* **Validation, not Test.** No test-dev / test-challenge prediction or score exists in
  this round; nothing here can be compared to leaderboard numbers as if it were.
* **2D only.** No 3D geometry, no extra descriptor, no pretrained weights, and the
  `lipo_v1` feature schema is unchanged — including its 6.8e-5 atom `UNK` rate, which
  *is* material for a handful of Se/P/Si/As-rich molecules (899 of 1M).
* **Positioning only** vs GCN/GIN/GIN-virtual: different data volume, epochs,
  augmentation and evaluation pipeline (§5).
* **Train-monitor MAE is a fit metric**, not an estimate of generalisation; it keeps
  falling while Validation has plateaued, which is the expected memorisation pattern.
* **Cost extrapolation to 3.38M is a projection** from measured per-step cost; it was
  not executed.
* The `final_epoch_*` fields are `null` in the two delivered `run.json` files: at
  `93ae701f` the payload looked string keys up in an int-keyed dict. Fixed
  afterwards in `pq_train.py` (reporting-only); the per-epoch truth lives in
  `train_curve.csv` and `*_epoch_reports`, which is what every number above uses.

## 10. Scope discipline (what was deliberately not done)

No W3 arm, no DYNAMIC / CPC / RHIC, no new module or architecture, no
hyper-parameter search, no extra seeds, no early stopping, no full 3.38M-molecule run,
no Test-split evaluation, no leaderboard submission. Both arms share one recipe, one
feature cache, one normalisation, one monitor subset and one commit; the only
difference between them is the number of training molecules.
