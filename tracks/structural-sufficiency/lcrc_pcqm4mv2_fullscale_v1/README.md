# `lcrc_pcqm4mv2_fullscale_v1` — PCQM4Mv2 full-Train W2/W3

Round `lcrc-pcqm4mv2-fullscale-v1`. Frozen plan: `PROTOCOL.md`; results:
`REPORT.md`.

* `code/fs_common.py` — frozen identities (3,378,606 rows, 1,055,840 steps,
  widths, budget, inherited read-only ingredients).
* `code/fs_build.py` — official split audit, full `lipo_v1` feature rebuild,
  mmap merge, and the bitwise 1M-prefix control against the historical cache.
* `code/fs_data.py` — full-Train task assembly (row plan, labels, monitor,
  normalisation, per-epoch order hashes).
* `code/fs_train.py` — formal training (E1 collation, epoch-boundary resumable
  checkpoints, full Validation + 20k monitor each epoch) and the throughput
  preflight.
* `code/fs_verify.py` — the twelve correctness gates.
* `code/fs_analysis.py` / `code/fs_diagnostics.py` — tables, curves, recheck and
  read-only grouped diagnostics.
* `code/fs_infer.py` — frozen model selection, official Test-dev features,
  inference and `y_pred_pcqm4m-v2_test-dev.npz`.
* `tests/test_fs_round.py` — CPU unit tests.

Code only; caches, checkpoints, curves and the submission stay git-ignored with
their SHA256 recorded in `REPORT.md` / `analysis/`.
