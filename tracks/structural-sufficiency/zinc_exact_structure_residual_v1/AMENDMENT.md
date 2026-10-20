# AMENDMENT — post-freeze issues and how they were handled

The protocol, config, vocabulary, prior, run matrix and decision rule were
frozen at commit `8df7986` before any formal trajectory. **No frozen scientific
constant was changed after the freeze**, no run was retrained, and no prediction
was replaced. This file records three implementation issues that surfaced after
the freeze, their blast radius, and the remediation that keeps the frozen
artifacts valid.

## A1. `valid_y_read_once.npy` absent on the remote checkout

* **Symptom:** the first `esr-freeze` job (`b71bc5d1`, commit `abbd97d`) exited 1
  with `FileNotFoundError` for
  `tracks/structure-contribution/results/bjoc_recipe_tuning_v1/valid_y_read_once.npy`.
* **Cause:** the label array is git-ignored (`*.npy`) and only the
  `valid_read.lock` (with its sha256) is tracked. The array existed locally but
  had never been transferred to `res`.
* **Remediation:** `valid_labels()` now re-materialises the array from the
  official raw valid split *only if the ignored copy is absent*, and accepts it
  only when its sha256 equals the committed lock hash
  `8f6ee636fd23ee977984e49cdc18f352a61bcf24a4987ae11402a402c41d4b45`. The
  reconstructed array is byte-identical to the previously recorded read; the
  flag `reconstructed_from_official_raw_valid = true` is stored in
  `analysis/metrics.json` and `protocol_freeze.json`'s label block.
* **Blast radius:** none on the science. The labels were already exposed in the
  previous round; this round reuses that single recorded read. No metric was
  computed until after `prediction_freeze.json` was written
  (`frozen_before_valid_read: true`), and official test was never loaded.

## A2. Score/analyze run-key mapping bug in the frozen runner

* **Symptom:** `esr-score-p1` failed with
  `KeyError: 'valid_E@0.1_s0'`; after fixing that, with
  `KeyError: 'drop_reference_worst_k'`.
* **Cause:** the frozen config registers human-readable names (`E@0.1`,
  `E@1.0`), while run directories use `E_eta0.1`, `E_eta1`. The frozen
  `cmd_score`/`cmd_analyze` used the registered string as a run key, and read
  `drop_reference_worst_k` from `decision` instead of `decision.family_vs_B`.
  This is stage plumbing, not a training or protocol constant.
* **Remediation:** the frozen runner was not edited (its code hashes are pinned
  by `protocol_freeze.json`, and `verify_freeze()` is called by every formal
  stage). Scoring/analysis live in the new `code/esr_score.py`, which imports
  the frozen helpers (`verify_freeze`, `_err`, `_contrast`, `valid_labels`,
  `parse_run_key`, `model_for_run`) and reads the registered config, `etas` and
  decision thresholds from the frozen config. The decision logic is the same
  four checks.
* **Blast radius:** none on the numbers. The two failed jobs (`8f390d42`,
  `8f201880`) are retained in the job history; the successful score job is
  `esr-score-p1-03059273` at commit `ce7d64a`.

## A3. `prior.mean_R_per_step = 0.0` in `train.json`

* **Symptom:** every residual run reports `prior.mean_R_per_step = 0.0`, while
  the per-epoch curve entries are non-zero.
* **Cause:** the training loop accumulated the per-step prior into
  `epoch_prior` but serialised `prior_sum`, which was never incremented. The
  affected field is a log-only aggregate.
* **Blast radius:** none on training, predictions or the decision. The field is
  written once at the end of training and is not read by the loss, the
  optimizer, the soup, prediction, scoring or the decision rule. The report uses
  (a) the exact prior recomputed from the frozen soup in
  `analysis/mechanism.json → prior_value`, and (b) the 20-epoch logged values
  (`prior_logged_curve_mean`, `curves/*.csv`).
* **Status:** the frozen runner is left as-is. A future round should fix
  `prior_sum += float(prior.detach())` before reusing this harness; the fix
  would change the frozen code hash and is intentionally not backported here.

## A4. Post-hoc diagnostics

`code/esr_diag.py` and `analysis/diagnostics.json` were added after the
decision. They read only frozen artifacts (soups, predictions, cached type
assignments, the exposed valid labels) and are explicitly not part of the
protocol. They do not feed any decision or reported primary metric.
