# DEVIATIONS — cscl-p3-sam-v1

## D-1. Amendment 1: smoke-only `lr_curve` key

The first post-freeze smoke (`p3sam-smoke-20261017-195017-dda865f7`) completed the 20-update SSL
smoke and then failed with `KeyError: 'curve'` while reading the P3 smoke epoch time.  The frozen
`bjoc_perfmech_long_v1.train_supervised` returns `lr_curve`, not `curve`.  One line in
`run_p3_sam_v1.py` was corrected; `p3_sam_v1.py` (model/SAM/loop) is byte-identical.  The failed
run had **no** P3/SAM optimizer step, prediction, or held-out label read.  `protocol_freeze.json`
was re-frozen as amendment 1 with the new driver hash.  See `AMENDMENTS.md`.

## D-2. Ledger accounting for the failed smoke and predictions

`gpu_budget_ledger.json` contains three manually appended entries after the formal runs:

* `smoke_failed` 18.0 s: the failed smoke in D-1 (20 SSL updates + one P3 epoch before the
  `KeyError`);
* `predict` 9.0 s for seed 0 and 9.0 s for seed 1: `cmd_predict` did not append ledger entries
  itself; the values are the measured process wall times (started 21:00:40, exited 21:00:48/49).

With these entries the total is **1.8480 h / 3.0 h**.  The formal SSL and training entries are
written by the driver directly.  No GPU time is omitted from the reported total.

## D-3. Remote compute-checkout lock file

`rr deploy` initially refused because the previous SAC round's `holdout_read.lock` was untracked on
the compute checkout (it is tracked locally).  The file was moved to
`/tmp/sac_v1_holdout_read.lock.pre_p3sam` on `res`; it was not deleted and the compute checkout
was not edited.  Deploy then fast-forwarded cleanly.

## D-4. Machine-local object tensors

As in the SAC round, the `.pt` object tensors are machine-local and gitignored.  `stage-inputs`
rebuilt them on `res` and verified semantic object hashes (`object_state_sha256`), row hashes and
the Fit-only standardiser against the frozen manifest; `.pt` byte equality is not claimed.

## D-5. No Locked read

Monitor failed the frozen gate (all 7 conditions false).  Per the protocol, `score-locked` was not
run.  No `holdout_read_locked.lock` exists and Locked labels were not read.  The Monitor lock was
created by `score-monitor` after all 8 predictions and checkpoints had been frozen.

## D-6. Split T is an exposed resample

The E/F/S historical splits together cover all 9,996 usable rows, so split T has
`historical_rows_seen_fraction = 1.0`.  T is a new identity-grouped resample with an explicit salt
and an asserted row hash, not an independent blind population.  This is disclosed in
`split_T.json`, `split_compare_T.json`, `PROTOCOL.md` and `REPORT.md`.

## D-7. P3 arm uses the frozen trainer directly

A does not use a copy of the P3 loop; it calls `bjoc_perfmech_long_v1.train_supervised` directly.
B uses the new `train_supervised_sam` mirror.  CPU gate G6 shows that the B loop with `rho=0` is
bitwise equivalent to the frozen trainer on synthetic molecules (same soup, batch order and update
count).  At training time the driver additionally asserts equal pre-SSL init hashes, equal SSL
encoder hashes, equal batch-order hashes and equal update budgets for the two real arms.
