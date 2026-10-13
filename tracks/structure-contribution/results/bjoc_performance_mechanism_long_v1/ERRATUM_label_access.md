# Erratum: label-access provenance in `cscl-bjoc-performance-mechanism-long-v1`

Date: 2026-10-13
Author: research agent (round `cscl-bjoc-architecture-breakthrough-v1`)
Status: disclosure only — **no old experiment was rerun, no old result was rewritten**

## What this erratum records

This note corrects the label-isolation provenance of the previous round
(`cscl-bjoc-performance-mechanism-long-v1`, commit `3bbd9ee`, frozen
2026-10-13T05:50:02Z) and of the earlier round
(`cscl-bjoc-masked-pretraining-v1`).  In both rounds the Monitor block's raw
`y` was read during **object preprocessing**, i.e. before any model or
prediction for that round existed, because the object-building alignment check
included all Monitor rows.

### Round `cscl-bjoc-performance-mechanism-long-v1` (split B)

`tracks/structure-contribution/code/run_bjoc_performance_mechanism_long_v1.py`,
`cmd_objects` (commit `3bbd9ee`):

```python
# alignment check on fit + monitor rows only (locked y is not read)
enc = torch.load(ENCODED_TRAIN, ...)
check_rows = list(payload["fit_rows"][:200]) + list(payload["monitor_rows"])
max_diff = 0.0
for i in check_rows:
    max_diff = max(max_diff, abs(float(ds[i].y.reshape(-1)[0]) - float(enc[i].y.reshape(-1)[0])))
```

The frozen artifact `results/bjoc_performance_mechanism_long_v1/objects_manifest_B.json`
records `alignment_rows_checked: 1200` (200 Fit + 1000 Monitor rows) and
`alignment_max_abs_diff: 0.0`.  The read value was used **only** to compute
that diagnostic.

The same pattern exists in the earlier round:
`run_bjoc_masked_pretraining_v1.py` builds
`check_rows = fit[:200] + monitor` and reads `ds[i].y` there
(split A, distributional round).

### Declared Monitor read

The declared Monitor read in `_read_monitor_y_once` read the label array via
`bp.load_y_rows` and then **again** loaded the raw dataset rows to check the
alignment of that array (same value, used only for the diagnostic).  The
written `monitor_read_B.lock` contains the `prediction_manifest_sha256`, so the
period between "predictions frozen" and "labels available for scoring" is
provable for the scoring step; however the label values themselves had already
been loaded once during object building.

### Impact assessment

* Code-path analysis: the only consumer of the early read was the
  `alignment_max_abs_diff` float in the object manifest.  No training target,
  no early stopping, no architecture or threshold decision in either round
  consumed Monitor labels, and all reported Monitor scores were computed from
  frozen prediction files after the declared read.
* The rounds' protocol text claimed Monitor labels were first read at the
  declared read.  That claim is **false as stated**; the correct statement is
  that Monitor labels were *loaded* during preprocessing for an alignment
  diagnostic and again during the declared read, while all selection decisions
  used only frozen predictions.
* Statistical expectation of the reported numbers is unchanged; a potential
  (unused) information channel existed and is now disclosed.

## Correction of record and non-actions

* No old experiment was rerun.  Frozen models, predictions, locks, labels and
  reports of the previous rounds remain byte-identical.
* The previous round's `REPORT.md` is not rewritten; this erratum is the
  authoritative correction.
* The current round (`cscl-bjoc-architecture-breakthrough-v1`) fixes the
  defect structurally: `ab.LabelGuard` wraps the local dataset so that
  `.y` is unreachable for Monitor/Locked rows during preprocessing, SSL,
  training and prediction; the alignment check uses Fit rows only
  (`run_bjoc_architecture_breakthrough_v1.py`, `cmd_objects`:
  `check_rows = list(split["fit_rows"][:200])`), and tests exercise both the
  positive control (guarded `.y` raises) and the Fit-only alignment path.
