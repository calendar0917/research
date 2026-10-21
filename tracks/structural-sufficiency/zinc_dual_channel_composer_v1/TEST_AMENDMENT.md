# Test reporting harness amendment

**Protocol:** `zinc-dual-channel-composer-v1-test-reporting`
**Date:** 2026-10-21
**Status:** transparency note; does not change any frozen checkpoint, model, or valid decision.

`TEST_PREREG.md` said one test scoring pass. The harness was actually invoked three
times after the pre-registration commit because of two analysis-code bugs:

1. **First invocation:** computed the three overall test MAEs, then aborted at the
   subgroup stage because `radius_signatures` was imported from `audit_common` instead
   of `audit_coverage`. No output files were written.
2. **Second invocation:** completed, but passed the radius-1 train signatures against
   radius-2 test signatures, making all 1,000 test rows appear “signature unseen”.
   Those subgroup outputs were rejected and not used.
3. **Third invocation:** corrected to radius-2 for both sides; this is the authoritative
   saved pass in `analysis/test_reporting.json`, `analysis/test_main_table.md`, and
   `analysis/test_per_molecule.csv`.

All three invocations produced the same overall test MAEs (P3B `0.0698648408`,
L `0.0662665548`, C `0.0700148196`). No checkpoint, epoch, recipe, metric definition,
or decision was changed after seeing test output. However, because the overall test
MAEs were printed already in the first invocation, the reporting read was **not
literally blind after that point**. The test split remains spent and this is a
descriptive reporting read, not a confirmatory blind test. `test_reporting.json`
therefore records `test_read_count = 3`.

The final authoritative pass is the only one used in `TEST_REPORT.md`.
