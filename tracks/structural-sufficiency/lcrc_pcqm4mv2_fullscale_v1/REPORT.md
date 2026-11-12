# `lcrc_pcqm4mv2_fullscale_v1` — full-Train W2/W3

Round `lcrc-pcqm4mv2-fullscale-v1`. Frozen plan: `PROTOCOL.md`.

**Status: COMPLETE (training + Validation + diagnostics + Test-dev file).
Test-dev MAE is `PENDING_OFFICIAL_EVALUATION`; the file is
`SUBMISSION_READY_NOT_SCORED`. Nothing was uploaded to OGB.**

- Host: `res` (`a100-2`), 2× NVIDIA A100-SXM4-40GB, 104 CPU cores, 503 GB RAM,
  driver 550.163.01, Python 3.12.14, torch 2.5.1+cu124, RDKit 2026.09.1.
- Code revision used by both formal runs:
  `639e890e0cb8a528d8b45b1fba8e8d0852ab0e16` (clean worktree).
- Only the two planned runs were trained: `PCQM_W2_FULL` and `PCQM_W3_FULL`,
  from scratch, seed 0, 40 epochs, no early stopping, no extra seeds/widths,
  no architecture or recipe search.

## 1. Runs

| Run ID | Width | Parameters | Train molecules | Epochs | Seed | Status | Steps | GPU-hours |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `PCQM_W2_FULL` | 2× | 939,079 | 3,378,606 | 40 | 0 | COMPLETED | 1,055,840 | 11.7867 |
| `PCQM_W3_FULL` | 3× | 2,038,167 | 3,378,606 | 40 | 0 | COMPLETED | 1,055,840 | 11.6276 |

Both runs used the frozen recipe: batch 128, AdamW `(0.9, 0.999)`, eps `1e-8`,
weight decay `1e-3`, dropout `0.10`, peak LR `3e-4`, one-epoch warmup
`3e-5 → 3e-4`, per-step cosine to `1e-5`, FP32 with TF32 off, gradient clip
`5.0`, L1 on standardised gap, E1 vectorised collation with the original
per-step synchronisation. Identical data order, seed and schedule; only width
differs.

## 2. Data and cache identity (OBSERVED)

Official installed `ogb` PCQM4Mv2 dataset, official `get_idx_split()`:

| split | rows |
|---|---:|
| train | 3,378,606 |
| valid | 73,545 |
| test-dev | 147,037 |
| test-challenge | 147,432 |

- `raw/data.csv.gz` SHA256 `ab89a4707d5e37f19c3b5e69d1c5cbb7fc1a9665d38783c0c89df6675f34286b`
  (3,746,620 lines including header).
- `split_dict.pt` SHA256 `d913e5541af6a6279abfe23ecf0fdbe4bc86785e1d8fc24a8f3ef422a01d3722`.
- Full-Train row plan (historical seed-42 permutation): 3,378,606 unique
  official Train rows, SHA256
  `81e7ec71665f8ffca32c630fbac14407aca872c49604544966b54595d2d7c7eb`;
  first 1,000,000 rows equal the historical 1M subset; no Validation or Test
  row is present.
- Rebuilt full `lipo_v1` cache (reliable full rebuild, not prefix injection):
  SHA256 `b58a15e752c4f661734dba4d25d1a383426135510ea0aeae1b4d032713d30469`,
  19,795,851,045 bytes, 3,378,606 molecules, 47,711,634 atoms,
  49,105,363 bonds, 98,210,726 S1 codes, 130,457,228 S2 paths.
- Build: 845 chunks × 4,000 rows, 48 workers, 338.8 s. Parse modes: 3,378,573
  strict + 33 `unsanitized_fallback`, 0 failures, 499 multi-component
  molecules (same adapter and fallback rule as the historical caches).
- **1M-prefix control: OK.** Every value field of the first 1,000,000 rows is
  bitwise equal to the historical `pcqm_train1m` cache
  (`d8f24e860d5535e86c2db9dc62ffe2f1be74560779c658cb42ab57e97164dffc`):
  atom/bond features, scalar features, atom/bond types, global features,
  component/ring counts, neighbour and path pointers/codes/closures, `gids`,
  SMILES, molecule ids, and sampled collated batches (E1 vs frozen collate).
- Cache verification: structure audit OK, tree SHA equals manifest SHA, prefix
  control OK. Shards (20.8 GB) removed after merge.

Historical 1M caches, labels, monitor rows, normalisation constants, evaluator
and checkpoints were used read-only and were not modified.

## 3. Correctness gates (OBSERVED, GPU)

`analysis/fs_verify.json`: **12/12 gates OK** on the real full cache.

- Parameters: W2 939,079, W3 2,038,167; W1 analytic 259,831; built widths match
  the frozen width rules; W2 builder bitwise equals the historical W2 state.
- E1 input batches equal the frozen collation field-by-field (batch sizes 1, 2,
  5, 34; no extra/missing fields).
- Row alignment: 3,378,606 full-cache rows match the official Train row plan;
  no Validation/Test row enters the optimizer.
- One-shot forward: exactly one atom encoder, one bond encoder, one S1/S2 local
  computation, one readout; no iterative message passing.
- Finite real training: 100 steps at both widths, finite loss/grad/update.
- Checkpoint save → restore → continue: loss delta `0.0`, model and AdamW state
  bitwise equal, RNG hashes equal.
- Official evaluator MAE vs independent NumPy MAE on the full Validation split:
  delta `0.0` (untrained model, 0.595587840 eV both).
- Frozen feature/label normalisation fingerprints and inverse round-trip OK.
- Identity gate: clean commit `639e890e...`, full-cache/row-plan/split hashes
  recorded, all `fs_*.py` hashes recorded.

Independent CPU re-score of the two best checkpoints:

| run | recorded best | recomputed | delta |
|---|---:|---:|---:|
| `PCQM_W2_FULL` | 0.128821302505 | 0.128821302166 | −3.39e-10 |
| `PCQM_W3_FULL` | 0.118775494697 | 0.118775494686 | −1.06e-11 |

## 4. Throughput preflight and resources (OBSERVED)

250 measured steps after 30 warmup steps, batch 128, real full cache, one
process per GPU (`analysis/fs_bench_*.json`, summary `fs_preflight.json`).

| measurement | run | GPU | mol/s | ms/step | process CPU cores (mean) | RSS | GPU util (mean) | GPU mem |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| standalone | W2 | 0 | 2,920 | 43.83 | 14.5 | 11.4 GB | 17.8% | 762 MiB |
| standalone | W3 | 1 | 3,044 | 42.05 | 15.2 | 11.4 GB | 21.2% | 818 MiB |
| concurrent | W2 | 0 | 3,048 | 42.00 | 14.6 | 11.4 GB | 18.7% | 760 MiB |
| concurrent | W3 | 1 | 2,705 | 47.32 | 13.2 | 11.4 GB | 19.5% | 818 MiB |

Aggregate concurrent throughput 5,753 mol/s vs 5,964 mol/s standalone
(−3.5%); W2 +4.4%, W3 −11.1%. GPU utilisation was only ~19–21% because the
frozen E1 path is CPU/copy bound; host load average was 55–72 (other tenants),
~59 GB page cache and ~474 GB available RAM at preflight. Concurrency was
therefore kept (no material efficiency loss, halves wall time), and the
decision plus the external-load caveat are recorded in `fs_preflight.json`.

Corrected budget before launch: W2 ≈ 12.3 h, W3 ≈ 13.9 h → ≈ 26.2 GPU-hours at
the concurrent rates (target 26, hard ceiling 38). Actual: **23.4144 GPU-hours**
for the two 40-epoch runs (W2 11.7867, W3 11.6276), within the target.
One-off preprocessing was 338.8 s of feature build (48 workers) plus the merge
and hashing pass; Test-dev end-to-end inference added 53.1 s. Peak training RSS
was 20.7 GB per process; peak GPU allocation was 223 MB (W2) / 311 MB (W3).

Preflight/publication caveats: the standalone/concurrent numbers are short
samples under an externally loaded host; they are used only for the
concurrency decision, not as formal benchmark claims.

## 5. Full-Validation results (OBSERVED)

| model | Train size | Parameters | Best Valid MAE (eV) | Best epoch | Valid@40 (eV) | Monitor@40 (eV) | GPU-hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| W2-1M historical | 1,000,000 | 939,079 | 0.132311 | 40 | 0.132311 | 0.105891 | 5.162 |
| W3-1M historical | 1,000,000 | 2,038,167 | 0.127809 | 39 | 0.127875 | 0.097213 | 3.447 |
| **W2-FULL** | 3,378,606 | 939,079 | **0.128821** | 18 | 0.135414 | 0.117057 | 11.787 |
| **W3-FULL** | 3,378,606 | 2,038,167 | **0.118775** | 35 | 0.123822 | 0.102832 | 11.628 |

Last-5-epoch means: W2 Validation 0.134401 eV, monitor 0.116113 eV; W3
Validation 0.123318 eV, monitor 0.102606 eV.

Full 40-epoch curves: `analysis/fs_curves.csv`, `analysis/fs_curves.png`
(Validation, Train-monitor, train L1(z), step p50); machine-readable summary:
`analysis/fs_summary.json`, table: `analysis/fs_summary_table.md`.

Deltas:

- FULL W3 − W2: best **−0.010046 eV** (−7.80% relative); epoch 40
  **−0.011592 eV** (−8.56%); last-5 mean −0.011083 eV.
- 1M → FULL (best): W2 **−0.003490 eV**, W3 **−0.009034 eV**.
- 1M → FULL (epoch 40): W2 **+0.003103 eV** (worse), W3 **−0.004053 eV**.
- Train-monitor minus Validation at epoch 40: W2 −0.018357 eV, W3 −0.020991 eV.

## 6. Read-only grouped diagnostics (OBSERVED)

`analysis/fs_diagnostics.json` / `.md`; descriptive only, the official metric is
never recomputed after dropping rows.

- Official Train has 3,256,262 distinct SMILES among 3,378,606 rows; the 1M
  historical subset has 987,772 distinct SMILES.
- Validation rows whose SMILES also occurs in official Train: **5,375 / 73,545
  (7.31%)**; 1,253 Validation molecules share a SMILES with another Validation
  molecule (1,216 duplicate strings).
- UNK-atom molecules: 230 (0.313%). W2 MAE 0.592918 eV with UNK vs 0.127365 eV
  without; W3 0.445766 eV vs 0.117750 eV without.
- Atom-count bins (W3 MAE): 0–20 0.1145 (n=70,269), 21–25 0.1864, 26–30
  0.2678, 31–35 0.3523, 36–40 0.8752, 41–50 0.4809, 51+ 2.9149 (n=1).
- True-gap bins (W3 MAE): [0,4.5) 0.1381, [4.5,5) 0.1045, [5,5.5) 0.1012,
  [5.5,6) 0.0998, [6,100) 0.1349.

## 7. Interpretation (INTERPRETATION)

- At full Train scale the width effect is no longer marginal. W3 beats W2 on
  best Validation MAE by 0.0100 eV and at epoch 40 by 0.0116 eV, exceeding the
  pre-registered 0.005 eV margin, and the sign is consistent across the best
  checkpoint, the last-5 mean and the late-epoch curve. On the historical 1M
  subset the same contrast was only −0.004436 eV at epoch 40 and was labelled
  `NO_WIDTH_SIGNAL`.
- More data helped both widths at their best checkpoints, but much more so W3
  (−0.0090 eV) than W2 (−0.0035 eV). W2's late curve is unstable (best at
  epoch 18, then 0.1354 at epoch 40); W3's best is at epoch 35 and remains
  0.1238 at epoch 40. The W2-FULL epoch-40 regression is an observation, not a
  reason to re-tune or re-select: model selection used best Validation MAE only.
- The Train-monitor gap is negative and larger for W3, consistent with the
  monitor rows being in-optimizer Train data, not a second validation set.
- Cost: W3 buys 0.0100 eV for +1,099,088 parameters (+117% over W2) and
  essentially the same GPU time (11.63 vs 11.79 h in this run; the frozen E1
  pipeline is data-bound, so width is nearly free at this scale).
- Published context, **not reproduced here**: on PCQM4Mv2 Validation, commonly
  reported 2D full-data numbers are roughly GIN ≈ 0.124, GIN-VN ≈ 0.120,
  GraphGPS ≈ 0.117, GRIT ≈ 0.115 eV. Our 0.1188 eV sits between GIN-VN and
  GraphGPS. The comparison is only indicative: different architectures,
  positional/3D inputs, training budgets and evaluation code; we did not rerun
  any published baseline.

## 8. Not established (NOT ESTABLISHED)

- Single seed (0), fixed 40 epochs, official CID split. No seed variance,
  confidence interval or epoch-budget sweep for the width contrast.
- No causal claim that width alone produces the difference at other scales;
  only W2 and W3 at 3,378,606 rows were trained.
- No Test-dev/Test-challenge label was read, inferred or scored. The real
  Test-dev MAE is unknown and must come from the official server.
- No claim that 0.1188 eV is a leaderboard result; it is official-split
  Validation MAE from our own pipeline.
- Published baseline numbers above are literature values, not measurements
  made in this round.
- The throughput numbers are short preflight samples on a shared host, not a
  controlled GPU benchmark.

## 9. Next research decision (NEXT RESEARCH DECISION)

Keep W3-FULL (`checkpoint_best_valid.pt`, epoch 35, SHA256
`786f3a5d745599b762e5882385ff84c9a74d88bae4f8a8c214588d8845d5a826`) as the
selected model and the round's width result. The next step, if authorised, is a
single pre-registered seed-1 confirmation of the FULL W2/W3 contrast (not an
architecture search, not more epochs, not an ensemble). Do not start W4 or a
new architecture based on this one seed. The Test-dev file may be submitted
only after explicit user authorisation.

## 10. Test-dev (OBSERVED; not scored)

Selection was frozen only after both runs completed, using best Validation MAE
(tie → fewer parameters): **`PCQM_W3_FULL`**, epoch 35, 2,038,167 parameters,
Validation MAE 0.118775494697 eV, checkpoint SHA256
`786f3a5d745599b762e5882385ff84c9a74d88bae4f8a8c214588d8845d5a826`
(`analysis/fs_model_selection.json`).

Test-dev was rebuilt from the official `split['test-dev']` row ids in ascending
original order, SMILES only from the raw CSV (`usecols=['idx','smiles']`; the
label column is never parsed):

- rows 147,037, row-order SHA256
  `03918fd3cb8a80a2857dc610ce60a94d7e6d003815f75b05b2308ff5d14d5cfe`;
- feature cache SHA256
  `8f8b885b1d0ed9a809c9832a86694aa5b8b23aff1686bb07983445bac809c150`
  (3 unsanitized fallbacks, 0 failures, no Train/Valid overlap);
- inference 3.389 s (43,386 mol/s, batch 512, single A100);
- end-to-end (raw SMILES → build → merge → GPU inference → `.npz`): **53.13 s**,
  within the official 4-hour inference limit;
- prediction (float64 eV): min −0.559042, max 13.975916, mean 5.489007,
  std 1.252045; all finite;
- raw `predictions_testdev_eV.npy` SHA256
  `16cacf6afab69f67a21a91b535bc1ddfa44fee797d199f64e3562a889695fd1d`.

Official OGB submission (`PCQM4Mv2Evaluator.save_test_submission`, mode
`test-dev`):

- file: `runs/PCQM_TESTDEV_SUBMISSION/y_pred_pcqm4m-v2_test-dev.npz`
  (durable copy committed at `submission/y_pred_pcqm4m-v2_test-dev.npz`);
- SHA256 `c4887c9178c627b64183edc4b7b17a93a44ab10b5158c839315e380b9683b6e4`;
- 496,280 bytes; key `y_pred`, shape `(147037,)`, dtype float32, all finite;
- submission verification: **`SUBMISSION_READY_NOT_SCORED`**
  (`analysis/fs_testdev_submission_verify.json`); row order and checkpoint
  hash re-checked independently.

**No leaderboard upload was performed.** Official submission requires a user
account and explicit authorisation; the Test-dev MAE remains
`PENDING_OFFICIAL_EVALUATION`.

## 11. Reproduction

```bash
# from the repo root, revision 639e890e...
rr deploy res
rr run res fs-build --cpus 64 --mem 200G --time 02:00:00 -- \
  python tracks/structural-sufficiency/lcrc_pcqm4mv2_fullscale_v1/code/fs_build.py --stage build --workers 48
rr run res fs-merge --cpus 64 --mem 200G --time 02:00:00 -- \
  python tracks/structural-sufficiency/lcrc_pcqm4mv2_fullscale_v1/code/fs_build.py --stage merge
rr run res fs-verify --pool res-gpu0 --cpus 16 --mem 32G --time 03:00:00 -- \
  python tracks/structural-sufficiency/lcrc_pcqm4mv2_fullscale_v1/code/fs_verify.py --device cuda
# preflight (optional) then the two formal runs:
rr run res fs-w2-full --pool res-gpu0 --cpus 16 --mem 40G --time 18:00:00 -- \
  python tracks/structural-sufficiency/lcrc_pcqm4mv2_fullscale_v1/code/fs_train.py \
  --run PCQM_W2_FULL --device cuda --resume auto --soft-seconds 59400 --hard-seconds 63000
rr run res fs-w3-full --pool res-gpu1 --cpus 16 --mem 40G --time 18:00:00 -- \
  python tracks/structural-sufficiency/lcrc_pcqm4mv2_fullscale_v1/code/fs_train.py \
  --run PCQM_W3_FULL --device cuda --resume auto --soft-seconds 59400 --hard-seconds 63000
# analysis, selection, Test-dev
python .../fs_analysis.py --stage summary
rr run res fs-recheck --pool res-gpu0 -- python .../fs_analysis.py --stage recheck --device cuda
rr run res fs-select --pool res-gpu0 -- python .../fs_infer.py --stage select
rr run res fs-e2e --pool res-gpu0 --cpus 32 --mem 48G --time 01:00:00 -- \
  python .../fs_infer.py --stage end2end --device cuda --workers 32
rr run res fs-verify-sub --pool res-gpu0 -- python .../fs_infer.py --stage verify-submission
```

RDKit is not in the locked project environment; the remote venv was given
`rdkit==2026.09.1` with `uv pip install --python .venv/bin/python` after the
last deployment. Training itself does not need RDKit (the cache is frozen);
only the Test-dev rebuild does.

## 12. Artifact inventory

| artifact | SHA256 |
|---|---|
| full Train cache | `b58a15e752c4f661734dba4d25d1a383426135510ea0aeae1b4d032713d30469` |
| historical 1M cache (control) | `d8f24e860d5535e86c2db9dc62ffe2f1be74560779c658cb42ab57e97164dffc` |
| Train row plan | `81e7ec71665f8ffca32c630fbac14407aca872c49604544966b54595d2d7c7eb` |
| W2 best checkpoint | `000f2b61edc3b586a0269d1685618a99897639b1cc91f54b4ef5916fa9a56f6f` |
| W3 best checkpoint (selected) | `786f3a5d745599b762e5882385ff84c9a74d88bae4f8a8c214588d8845d5a826` |
| W2 curve | `a23bfeeb0ea766704dc9d6a144b11b5098318ce0904f33fdd1d6bae83d886fc9` |
| W3 curve | `532a04319170204b5ca960f469ea469a8c08e73cb9d333d176d892782b8277de` |
| Test-dev row order | `03918fd3cb8a80a2857dc610ce60a94d7e6d003815f75b05b2308ff5d14d5cfe` |
| Test-dev feature cache | `8f8b885b1d0ed9a809c9832a86694aa5b8b23aff1686bb07983445bac809c150` |
| raw Test-dev predictions | `16cacf6afab69f67a21a91b535bc1ddfa44fee797d199f64e3562a889695fd1d` |
| official submission `.npz` | `c4887c9178c627b64183edc4b7b17a93a44ab10b5158c839315e380b9683b6e4` |

All other numbers and checks are in `analysis/` (`fs_verify.json`,
`fs_summary.json`, `fs_diagnostics.json`, `fs_model_selection.json`,
`fs_testdev_*.json`, `fs_recheck_*.json`, `fs_preflight.json`,
`fs_cache_prefix_control.json`, `fs_full_cache_manifest.json`). Caches,
checkpoints and the run directories stay git-ignored; the submission `.npz` is
committed under `submission/` for durability.
