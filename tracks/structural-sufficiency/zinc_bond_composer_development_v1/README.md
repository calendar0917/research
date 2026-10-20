# zinc_bond_composer_development_v1

Small-budget internal development round: retain the complete original P3
reader, place a nonlinear per-bond branch only on the shared composer output
`c_e`, and ask whether B2 (4-head key attention) shows a useful single-seed
development signal over B0 (P3) and B1 (FFN capacity control).

**Status:** completed. **Decision:** `B2_DIRECTIONAL_BUT_BELOW_DEV_LINE`.
B2 is directionally better than B0 and B1, but the B2-vs-B0 Monitor gain is
0.001775, below the pre-agreed 0.003 line. Stop this candidate. See
`REPORT.md` for the full result and `TIMING_AMENDMENT.md` for the disclosed
timing-smoke deviation.

This is **not** a blind test, a confirmation, an official-valid/test evaluation,
a novelty claim, or a reopening of the closed S/T/R screen. The Monitor-1000
split is reused for development curves and metrics.

## Layout

```
PROTOCOL.md                 frozen protocol (pre-GPU)
TIMING_AMENDMENT.md         disclosed 2-epoch -> 5-epoch timing-smoke amendment
REPORT.md                   short result report
configs/                    frozen architecture / recipe / budget
code/
  bcd_common.py             frozen split, hashes, object/SSL loading
  bcd_models.py             B0/B1/B2 models and branch
  bcd_runner.py             prep -> gates -> smoke -> train -> analyze
  tests/test_bcd.py         lightweight CPU tests
cache/                      local git-ignored copies of frozen inputs
runs/B0_s0, B1_s0, B2_s0   train.json, curve.csv, soup.pt, monitor_pred.npz
curves/                     per-model development CSV
analysis/                   soup metrics and development decision
prep.json / cpu_checks.json
smoke.json                  2-epoch smoke (STOP_BUDGET)
smoke_probe_5ep.json        5-epoch timing probe (PASS)
gpu_ledger.json             process-wall GPU ledger
run_manifest.tsv            per-run hashes and metrics
wall_clock_correction.json  B0/B1 wall-field correction
runner_wall_check.json      conservative rr scheduler-wall audit
```

`cache/` is git-ignored. Run soups and Monitor predictions are retained with
their SHA-256 values in `run_manifest.tsv`.

## Commands

```bash
# from the repository root
UV="uv run python tracks/structural-sufficiency/zinc_bond_composer_development_v1/code/bcd_runner.py"

$UV prep
$UV gates
$UV smoke --device cuda --smoke-epochs 5 --out \
  tracks/structural-sufficiency/zinc_bond_composer_development_v1/smoke_probe_5ep.json

$UV train --kind B0 --device cuda
$UV train --kind B1 --device cuda
$UV train --kind B2 --device cuda
$UV analyze
```

Formal runs used the committed revision through `rr` on `res-gpu0` /
`res-gpu1`. The local cache is only for CPU development and tests; the remote
`res` checkout retains the phase1e frozen objects and SSL encoder at their
recorded hashes.

## Frozen inputs

| input | SHA-256 |
|---|---|
| split file | `fbca5fd5ac9c82c457bf4d78781fbe9936dd7068082377ff8cc2bf2309d72b49` |
| Fit rows (int64) | `7f66a23e038c3ad4879540f23d1e5a6cd6b8a789bbee8950a7b2c1d8b345ab15` |
| Monitor rows (int64) | `dc6cc4e9cb9d3cebac038b487f7e699e6c0aa52e5075e45825975b034509304f` |
| objects_fit | `bfbf96532029f8b5020e416bbf0f9fc7bdb257f5377265fc05b9c7c0f02d766c` |
| objects_monitor | `7a71850655ea606da51ee7bf27edb3c09340b25d701dee586b15a30b55652f0a` |
| Fit-only SSL encoder seed 0 | `0d045e6090062ac6e919c79c82c2cbc9a223c7974122c91d5b68a244d265bc0c` |

Canonical P3 post-SSL state: encoder `37282569…`, composer/reader `d9ffbd15…`.
The completed P3 soup is never loaded as an initialiser.

## Boundary

The prior `p3_bond_relation_bridge_v1` already found no gain from retaining P3
plus a one-hop bond-pair MLP residual. This round changes the interface and the
branch but is a related exploration, not a new main line. The result is a
negative single-seed development decision, not a universal claim about
attention or bond-composer architectures.
