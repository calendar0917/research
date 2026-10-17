# AMENDMENTS — cscl-p3-sam-v1

## Amendment 1 (2026-10-17)

**Trigger.** The post-freeze GPU smoke run `p3sam-smoke` (run id
`p3sam-smoke-20261017-195017-dda865f7`) completed the 20-update SSL smoke and
then failed while reading the P3 smoke epoch time:

```
KeyError: 'curve'
```

The frozen `bjoc_perfmech_long_v1.train_supervised` returns `lr_curve`, while
the SAM trainer returns `curve`.  The smoke driver incorrectly used the SAM
key for the P3 arm.

**Change.** One line in `run_p3_sam_v1.py`:

```diff
-    p3_epoch = float(p3_info["curve"][-1]["seconds"])
+    p3_epoch = float(p3_info["lr_curve"][-1]["seconds"])
```

**Scope.** Smoke reporting only.  This path fires before any formal P3 or SAM
optimizer step, before any prediction, and before any held-out label read.  The
model, SAM primitive, frozen P3 trainer call, recipe, split T, CPU gates,
thresholds and decision rule are unchanged.  `p3_sam_v1.py` is byte-identical
to the original freeze.

**Re-freeze.** `protocol_freeze.json` amendment 1 binds the updated
`run_p3_sam_v1.py` hash.  No formal GPU trajectory or prediction existed at the
time of the amendment.
