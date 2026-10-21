# Test reporting-only pre-registration

**Protocol id:** `zinc-dual-channel-composer-v1-test-reporting`
**Created:** 2026-10-21T11:49+08:00
**Prior frozen result commit:** `e76cc7d` (`zinc dual-channel: complete L/C seed0 runs and valid analysis`)
**Authorization:** the user explicitly wrote “我授权了，做吧” on 2026-10-21, after the valid-only
report and decision C were already frozen.  This is the new pre-registration required by the
repository guardrail for a reporting-only re-opening of the spent official ZINC test split.

## Scope

This is a **terminal reporting-only read**, not a selector.

- Evaluate exactly three already-frozen best-valid checkpoints:
  - `P3B_s0` (`zinc_recipe_trunk_v2/runs/xBssl_P3N_s0/best.pt`)
  - `L_s0` (`zinc_dual_channel_composer_v1/runs/L_s0/best.pt`)
  - `C_s0` (`zinc_dual_channel_composer_v1/runs/C_s0/best.pt`)
- No training, no fine-tuning, no epoch change, no checkpoint change, no threshold change,
  no subgroup-driven routing, and no new arm.
- Test numbers will not alter the valid-only decision C, will not trigger seed 1, and will
  not be used to choose a model or recipe.
- The test split is already spent and is not a pristine blind test.  The report will say so.

## Frozen checkpoints

| model | checkpoint | sha256 | selected epoch | stored best valid MAE |
|---|---|---|---:|---:|
| P3B_s0 | `runs/xBssl_P3N_s0/best.pt` | `4c379c93214c003a6c58fd4add4a64ecb4fe23603233738fab18f3885abdc34f` | 377 | 0.0762237511 |
| L_s0 | `runs/L_s0/best.pt` | `fab04ec0d5fd56f338379d7dcb223e31208338a8a89a335a25a6f0c4f5226cd5` | 489 | 0.0785371522 |
| C_s0 | `runs/C_s0/best.pt` | `0a828c494f75138653ceb5677199f7b17e77943b5251bfd4b901406fe6739067` | 482 | 0.0823674316 |

The checkpoint hashes above must be verified at evaluation time; a mismatch aborts the read.

## Data

- Official PyG ZINC `subset=True` test split, 1,000 molecules, raw-y MAE.
- Frozen cache: `tracks/structural-sufficiency/zinc_arch_discovery_v1/cache/ad_test.pt`
- Cache sha256: `981e2b9a2588b300024947edac4a5053d054199051ca19e016d350ef3ece0dc1`
- Cache `labels_sha256`: `66fd23ecdb1b59c76cee597efc81272e0d2744eeac80744f4159d436e5388768`
- Before this pre-registration was committed, the cache file was loaded once only to confirm
  its presence, row count (1,000), and metadata; no test labels, predictions, metrics, or
  model outputs were inspected or computed.  All scoring is after this file is committed.

## Evaluation protocol

- CPU evaluation, `model.eval()`, batch size 256, no dropout, no augmentation.
- The exact P3N collate is used for all three models (`dcc_runner.collate_p3n`).
- P3B: `ad2_models.build_model("P3N", norm, 0, "cpu")`, load `state_dict`.
- L/C: `dcc_models.build_model("DCC_L"/"DCC_C", norm, 0, "cpu", organized_local=True/False)`,
  load `state_dict`.
- Report overall MAE, RMSE, median AE, p90/p95/max AE, mean signed error, positive-error
  fraction, and top-1/10/50/100 absolute-error shares.
- Paired bootstrap: 10,000 row resamples with RNG seed `20261021`; the same resamples are used
  for all three contrasts `L−P3B`, `C−P3B`, `L−C`; Δ = AE(first) − AE(second), negative means
  the first model is better.  CIs are descriptive fixed-checkpoint intervals on the spent test
  split, not cross-seed confirmation.

## Fixed subgroup semantics

Subgroups use the same definitions as the valid report, applied unchanged to test rows:

- `signature_unseen`: radius-2 rooted signature unseen in the 10k train split, i.e.
  `r2_unseen_atom_frac > 0`; train-only signature frequency.
- `signature_covered`: complement.
- `ordinary_y_gt_minus3`: `y > −3`; `tail_y_le_minus3`: `y ≤ −3`.
- `rings_ge4`: `n_rings ≥ 4`; `rings_ge4_and_signature_unseen`: conjunction.
- `P3B_test_worst100`: the 100 test molecules with largest P3B absolute error.  This is a
  descriptive hard-case monitor defined by a fixed rule, not a gate and not a selector.
- Unseen-vs-covered difference-of-differences uses the same formula as the valid report.

No test subgroup is used to choose an epoch, arm, or model; all three checkpoints are already
frozen.

## Outputs

- `analysis/test_reporting.json` — full metrics, contrasts, subgroups, hashes, provenance.
- `analysis/test_main_table.md` — compact main table.
- `analysis/test_per_molecule.csv` — row-level test predictions/errors and subgroup flags.
- `analysis/test_preds/{P3B_s0,L_s0,C_s0}_test.npz` — `row`, `pred`, `y` arrays.
- `TEST_REPORT.md` — Chinese reporting-only report.

One test scoring pass; no repeated peeking and no post-test model change.
