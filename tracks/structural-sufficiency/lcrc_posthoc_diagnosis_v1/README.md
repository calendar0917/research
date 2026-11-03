# LCRC Post-hoc Diagnosis v1

Round ID: `lcrc-posthoc-diagnosis-v1`
Track: `tracks/structural-sufficiency/lcrc_posthoc_diagnosis_v1/`
Upstream commit: `6cf9ca4eb3ddd45701c73baea66e80642034b756`

This round is a **post-hoc, CPU-only re-analysis** of already-saved predictions.
It is **not** a preregistered experiment and **not** an independent validation.
All new Test/Audit results are labelled `POST_HOC_EXPOSED_EVALUATION`.

## What is here

| Path | Content |
|---|---|
| `PROTOCOL.md` | Fixed analysis definitions, written before the new statistics |
| `ERRATA.md` | P0 corrections to the previous report (counts, typed-S2 scope, CI definition) |
| `REPORT.md` | Results in the required order |
| `DECISION.md` | Q1–Q4 evidence judgements; no experiment is started |
| `code/` | Read-only parsers, metrics, bootstrap, structural features, auxiliary diagnostics |
| `tests/` | Alignment, rank-loss identity, component weighting, bootstrap reproducibility |
| `analysis/` | Committed aggregate tables and JSON |
| `_local/` | Gitignored: copied `test_predictions.npz` / `test_eval.json` / `run.json` and per-molecule intermediates |

## Data sources (read-only)

* Previous round predictions: `../lcrc_predictive_trace_benchmark_v1/runs/`
  (`mol_*`, `tu_NCI1_*`). Only saved `test_predictions.npz`, `test_eval.json`,
  `run.json` were copied to `_local/predictions/`; **no checkpoint was copied**.
* Previous round aggregates: `../lcrc_predictive_trace_benchmark_v1/analysis/`
  (`molecular_results.csv`, `statistics.json`, `mechanism_audit.json`, …).
* Local-edit Audit derived scores:
  `../lcrc_local_edit_error_diagnosis_v1/analysis/probe_audit_scores.json`.
  Raw Audit labels were **not** read; `Δy = d_pred − e_delta`.
* OGB MolHIV structure source:
  `../lcrc_molnet_benchmark_pilot_v2/analysis/source/ogb/ogbg_molhiv/`.
  Only the `smiles` column and official scaffold split row ids are read;
  `raw/graph-label.csv.gz` is not opened.
* ESOL structure source:
  `../lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench/ESOL/`
  (SMILES column + frozen `random122` split indices).
* NCI1 source: `data/TUD/NCI1/NCI1/raw/` (node counts + labels only to reproduce
  the frozen 5-fold split exactly).

## Reproduce

Use the repository CPU virtualenv. GPU is not used:

```bash
cd tracks/structural-sufficiency/lcrc_posthoc_diagnosis_v1
export CUDA_VISIBLE_DEVICES=""
V=/home/calendar/code/research/.venv/bin/python

$V code/ph_integrity.py       # gates; reads saved predictions only
$V code/ph_p0.py              # P0 errata + mean-seed bootstrap (~2 min)
$V code/ph_molhiv.py          # structural features + rank-error decomposition (~2.5 min)
$V code/ph_local_edit.py      # MODEL vs ZERO_CHANGE (~1 s)
$V code/ph_aux.py             # ESOL + NCI1 (~2 s)
$V code/ph_manifest.py        # source manifest + data-access ledger
$V -m pytest -q tests         # correctness tests
```

Or run the whole sequence with:

```bash
$V code/run_all.py
```

The copied prediction files are not committed. To rebuild `_local/predictions/`
from the remote `res` worktree, the files can be fetched read-only; this round
used the already-pulled copies and did not run inference.

## Constraints honoured

* GPU usage: **0**; `CUDA_VISIBLE_DEVICES=""`.
* No training / fine-tuning / checkpoint selection.
* No new Test inference; no call to `ptb_score.py`, `edit_probe.py --stage audit`,
  or `--all-frozen`.
* No new split; NCI1 fold reconstruction is hash-checked against the frozen
  `split_manifest.json`.
* No historical checkpoint or frozen protocol was modified.
* Missing artifacts: none (`MISSING_ARTIFACT` list is empty).

See `analysis/data_access_ledger.json` for per-product input mapping and
`analysis/source_manifest.json` for SHA-256 of every input and output.
