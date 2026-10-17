# AMENDMENTS — cscl-p3-slim-v1

## Amendment 1 — SSL metadata key read in `cmd_train`

**When:** after protocol freeze, during the first formal `train` launch for seeds 0 and 1.
Both jobs had already loaded the committed split/objects, rebuilt the canonical P3 and Slim
initialisations, verified every init hash, loaded the common SSL encoder into both arms, and
verified the P3/Slim encoder hashes against each other.  They then raised
`KeyError: 'full_encoder_group_sha256'` because `cmd_train` looked up the hash in the
`torch`-saved encoder state (which contains only the three encoder state dicts) instead of the
`ssl_U_s*.json` metadata.

**Impact:** no supervised optimizer step, no batch-order consumption, no soup, no prediction,
no Monitor/Locked/official-valid/test label read.  No partial `train_*.json` or `soup_*.pt`
file was written by the failed jobs.  The two failed GPU jobs consumed about 16 s each and are
recorded in `gpu_budget_ledger.json` as failed attempts.

**Fix:** one lookup change in `cmd_train`: load `ssl_U_s{seed}.json` and compare the model's
encoder-group hash to `ssl_meta["full_encoder_group_sha256"]`.  `p3_slim_v1.py` (model,
initialisation and training loop) is byte-identical.  CPU G1–G8 and the 8 pytest tests were
re-run; the freeze was advanced to amendment 1 before any supervised trajectory.

**Why this does not change the experiment:** the hash source is provenance metadata; the
encoder state loaded into the model, the model construction, the data and the training rule are
unchanged.  The fix only makes the intended equality assertion executable.

## Amendment 2 — post-SSL shared-hash check compared against pre-SSL hash

**When:** after amendment 1, during the second formal `train` launch for seeds 0 and 1.
Both jobs loaded the SSL encoder into both arms, verified the encoder-group hashes, and then
raised `shared modules changed during SSL load` because `cmd_train` compared the post-SSL
shared-module hash (which necessarily includes the new encoder weights) against the pre-SSL
shared-module hash.

**Impact:** no supervised optimizer step, no batch-order consumption, no soup, no prediction,
no Monitor/Locked/official-valid/test label read.  No partial `train_*.json` or `soup_*.pt`
file was written.  The two failed jobs consumed about 17 s each and are recorded in the ledger.

**Fix:** compare post-SSL shared hashes between the two arms, and separately compare the
composer-only hash against the pre-SSL composer hash.  The pre-SSL four-module hash remains the
initialisation audit value.  `p3_slim_v1.py` is byte-identical; no model, data or training rule
changed.
