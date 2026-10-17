# DEVIATIONS — cscl-p3-slim-v1

## D-1. Amendment 1: SSL metadata key read in `cmd_train`

After the protocol freeze, the first formal `train` launch for seeds 0 and 1 failed with
`KeyError: 'full_encoder_group_sha256'`.  Both jobs had already verified the split, rebuilt
the canonical P3 and Slim initialisations, verified all init hashes, loaded the common SSL
encoder into both arms, and compared the two arms' encoder hashes.  They failed only while
looking up the expected SSL hash in the `torch`-saved encoder state instead of
`ssl_U_s*.json`.

No supervised optimizer step, batch-order consumption, soup, prediction, Monitor/Locked label
read, or official-valid/test access occurred.  The two jobs consumed about 16 s and 17 s and are
in the ledger as `train_failed`.  The fix reads `ssl_U_s{seed}.json` for the expected
encoder-group hash.  `p3_slim_v1.py` is byte-identical; CPU G1–G8 and 8 pytest tests were rerun;
the freeze advanced to amendment 1 before any supervised trajectory.

## D-2. Amendment 2: post-SSL shared-hash check

The second formal `train` launch failed with `shared modules changed during SSL load`.  The
post-SSL four-module hash necessarily includes the newly loaded encoder weights, so comparing it
to the pre-SSL hash was wrong.  The jobs failed before any supervised optimizer step, soup,
prediction or label read; no partial train JSON or soup was written.  They consumed about 17 s
and 16 s and are in the ledger as `train_failed`.

The fix compares post-SSL shared hashes between the two arms, and separately compares the
composer-only hash to the pre-SSL composer hash.  `p3_slim_v1.py` is byte-identical; the freeze
advanced to amendment 2 before any supervised trajectory.

## D-3. Failed SSL seed-1 launch: GPU ordinal

The first `ssl --seed 1` launch used `--device cuda:1` on pool `res-gpu1`.  The pool exposes a
single GPU as `cuda:0`, so PyTorch raised `invalid device ordinal` before any SSL update.  The
job consumed about 11 s, is recorded in the ledger as `ssl_failed`, and was retried with
`--device cuda:0`.  No checkpoint or JSON was written by the failed launch.

## D-4. Ledger entries for prediction

`cmd_predict` did not append its own ledger entries.  The formal seed-1 and seed-0 prediction
jobs took about 9 s and 10 s (both arms, both zones); these are appended manually and labelled
`predict`.  The failed deployment-marker attempt for seed 0 launched no process and is not
charged.

## D-5. Split U is an exposed-pool resample

The historical E/F/S/T zones cover all 9,996 usable rows, so U has
`historical_rows_seen_fraction = 1.0`.  U is a new identity-grouped partition with an explicit
salt and a verified row hash, not an independent blind population.  This is recorded in
`split_U.json`, `split_compare_U.json`, `PROTOCOL.md` and `REPORT.md`.

## D-6. Machine-local object tensors

`objects_*_U.pt` are machine-local and gitignored.  `stage-inputs` rebuilt them on `res` and
verified semantic object hashes, row hashes and the Fit-only standardiser against the frozen
manifest; byte equality of the `.pt` files is not claimed.

## D-7. No Locked read

Monitor failed the frozen gate (M2/M3/M5 false).  `score-locked` was not run; no
`holdout_read_locked.lock` exists and Locked labels were not read.  The Monitor lock was created
only after all 8 predictions and all checkpoints had been frozen.

## D-8. P3 arm uses the new frozen-loop mirror

Both arms use `p3_slim_v1.train_supervised_model`, a prebuilt-model mirror of the frozen R3
loop.  CPU G3 shows that the P3 path produces the same 2-epoch soup, batch order and update count
as `bjoc_perfmech_long_v1.train_supervised` (max abs soup/forward difference 0 in the frozen
gate run).  This is a deliberate common-loop design, not a change to the P3 recipe.
