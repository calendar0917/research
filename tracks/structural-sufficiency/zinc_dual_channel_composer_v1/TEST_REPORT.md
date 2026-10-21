# Official ZINC test — authorized reporting-only report

**Protocol:** `zinc-dual-channel-composer-v1-test-reporting`
**Pre-registration:** `TEST_PREREG.md` (`2bbf392`)
**Harness amendment:** `TEST_AMENDMENT.md` (`c18150d`)
**Prior valid-only decision:** `C_OWNERSHIP_ONLY_NO_PERFORMANCE` (`e76cc7d`)
**Authorization:** the user explicitly wrote “我授权了，做吧” on 2026-10-21, after the
valid-only decision and all checkpoints were frozen.

**This is not a pristine blind test.** The official ZINC test split is already spent.
The numbers below are descriptive reporting-only numbers; they did not change any model,
checkpoint, epoch, subgroup definition, or the valid-only decision.

---

## 1. Main test table

Official PyG ZINC `subset=True` test, 1,000 molecules, raw-y MAE. All three models are
their fixed valid-best checkpoints; no test-based selection.

| model | params | best valid | best ep | test MAE | test RMSE | median AE | p90 AE | p95 AE | max AE | mean signed | top-100 share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P3B_s0 | 413,505 | 0.076223751 | 377 | 0.069864841 | 0.147960938 | 0.035961390 | 0.150905415 | 0.223717274 | 1.920043468 | +0.002087527 | 0.472812 |
| L_s0 | 494,193 | 0.078537150 | 489 | **0.066266555** | 0.161322489 | 0.035739481 | 0.134011334 | 0.194864500 | 3.088657856 | +0.005970598 | 0.473059 |
| C_s0 | 494,193 | 0.082367432 | 482 | 0.070014820 | 0.157427757 | 0.037359208 | 0.141005361 | 0.222020257 | 2.195349693 | +0.003092653 | 0.478148 |

All three test MAEs are **lower than their valid MAEs** (test−valid: P3B −0.006359,
L −0.012271, C −0.012353). The test split is easier or differently distributed; this is
not a model improvement.

## 2. Paired test contrasts

10,000 paired row resamples, same resamples for all contrasts; Δ = AE(first) − AE(second),
negative means first is better. CIs are descriptive fixed-checkpoint intervals.

| contrast | mean Δ MAE | 95% CI |
|---|---:|---|
| L − P3B | **−0.003598** | [−0.008187, +0.001204] |
| C − P3B | **+0.000150** | [−0.003589, +0.004147] |
| L − C | **−0.003748** | [−0.007478, +0.000198] |

If one mechanically applied the valid performance threshold (`≤ −0.003`) to test, L would
look like it “passes” by 0.003598. **That would be invalid as a decision**: the test was
read after the decision was frozen, it is spent, all CIs include zero, and there is only
one seed. The test number is hypothesis-generating, not confirmatory.

The `L−C` direction and magnitude are similar to valid (−0.003748 vs −0.003830), so the
**ownership/grouping contrast replicates descriptively** on test, even though the
absolute performance level does not.

## 3. Fixed subgroups on test

Semantics are the same as the valid report, applied unchanged; `P3B_test_worst100` is a
fixed descriptive rule on P3B test AE, not a gate.

| group | n | P3B | L | C | L−P3B | C−P3B | L−C |
|---|---:|---:|---:|---:|---:|---:|---:|
| signature-unseen | 244 | 0.122947 | 0.129215 | 0.129762 | **+0.006269** | +0.006816 | −0.000547 |
| signature-covered | 756 | 0.052733 | 0.045950 | 0.050731 | **−0.006783** | −0.002001 | −0.004781 |
| ordinary `y>−3` | 923 | 0.063235 | 0.060392 | 0.064409 | −0.002843 | +0.001175 | −0.004017 |
| tail `y≤−3` | 77 | 0.149341 | 0.136686 | 0.137210 | −0.012655 | −0.012131 | −0.000524 |
| rings≥4 | 208 | 0.097206 | 0.090214 | 0.097461 | −0.006992 | +0.000255 | −0.007247 |
| rings≥4 & unseen | 51 | 0.200520 | 0.222111 | 0.214287 | +0.021592 | +0.013768 | +0.007824 |
| P3B test worst-100 | 100 | 0.330330 | 0.270850 | 0.311101 | **−0.059479** | −0.019229 | −0.040250 |

**Unseen-vs-covered difference-of-differences:** L−P3B **+0.013052**, C−P3B +0.008817,
L−C +0.004235. Positive means the first model’s advantage is larger in **covered** than
in unseen. On test, L’s overall gain over P3B is concentrated in **covered/common**
structures, while in the unseen group L is actually worse than P3B by +0.006269.

This is the opposite direction from the valid split, where L−P3B DoD was −0.003600
(relative advantage larger in unseen). The two splits therefore do **not** replicate an
unseen-specific benefit. The valid pattern was already tiny in absolute terms; the test
pattern goes the other way. There is no consistent evidence that the local channel
specifically helps the signature-unseen group.

The test tail and test P3B-worst100 also show L *better* than P3B, opposite to the valid
tail/worst100 tradeoff. This is a split-composition difference, not evidence that the
valid tradeoff disappeared.

## 4. Reconciliation with the valid-only decision

Valid-only result remains:

- L − P3B **+0.002313** (worse); C − P3B **+0.006144** (worse); L − C **−0.003830**
  (L better control). Decision **C — ownership-only, no performance**.
- Valid signature-unseen: L −0.000383 (near tie), C +0.004739; DoD L−P3B −0.003600.
- Valid tail: L +0.074998, C +0.076859; fixed worst100: L +0.006357, C +0.019153.

Test-only result:

- L − P3B **−0.003598** (better), C − P3B +0.000150 (tie), L − C **−0.003748**
  (L better control).
- Test signature-unseen: L +0.006269 (worse), C +0.006816 (worse); DoD L−P3B +0.013052
  (advantage in covered, not unseen).
- Test tail: L −0.012655, C −0.012131 (both better); test worst100: L −0.059479,
  C −0.019229.

The honest summary is: **the test split gives a more favorable overall L number but
does not replicate the valid subgroup story; it even reverses the unseen direction.**
The L−C grouping contrast is the only result that is directionally consistent across
valid and test (≈ −0.0038 both), but it still has a CI crossing zero and is single-seed.

## 5. What can and cannot be claimed

Can claim:

- With frozen valid-best checkpoints, on the spent official test, L has lower MAE than
  P3B (0.066267 vs 0.069865) and lower than C (0.070015); L−C is −0.003748.
- This test read is reporting-only; it did not select anything.
- The L−C grouping contrast is directionally consistent with valid, though not
  statistically resolved and not a cross-seed confirmation.

Cannot claim:

- That the dual-channel composer has a confirmed performance gain, or that the valid
  decision changes. Test is spent, single seed, post-hoc, and all CIs cross zero.
- That signature-unseen molecules are improved. Test says L is worse than P3B there
  (+0.006269), and the valid “advantage” was only −0.000383.
- That Null pooling is the generalization bottleneck, or that the generalization problem
  is solved.
- That the test number should trigger seed 1, a new arm, or a recipe search. It must not.
- That test and valid are exchangeable; test MAE is lower for all three models.

## 6. Harness note (important)

`TEST_AMENDMENT.md` records that the harness was invoked three times after the
pre-registration because of two analysis-code bugs. The first invocation computed the
same three overall test MAEs but aborted before writing outputs; the second wrote
subgroup diagnostics with a radius-1/radius-2 mismatch (all rows falsely “unseen”); the
third is authoritative. All three produced identical overall test MAEs. No checkpoint,
model, epoch, recipe, metric, or decision changed after seeing test output.

Because the overall test MAEs were printed before the final pass, this read was **not
literally blind after the first invocation**. This is one more reason the test numbers
are descriptive only. `analysis/test_reporting.json` records `test_read_count = 3`.

## 7. Artifacts

- `analysis/test_reporting.json` — full metrics, contrasts, subgroups, hashes.
- `analysis/test_main_table.md` — compact table.
- `analysis/test_per_molecule.csv` — row-level test predictions/errors and flags.
- `analysis/test_preds/{P3B_s0,L_s0,C_s0}_test.npz` — `row`, `pred`, `y`.

Checkpoint hashes:

| model | best.pt sha256 |
|---|---|
| P3B_s0 | `4c379c93214c003a6c58fd4add4a64ecb4fe23603233738fab18f3885abdc34f` |
| L_s0 | `fab04ec0d5fd56f338379d7dcb223e31208338a8a89a335a25a6f0c4f5226cd5` |
| C_s0 | `0a828c494f75138653ceb5677199f7b17e77943b5251bfd4b901406fe6739067` |

Test prediction hashes:

| model | `test_preds/*.npz` sha256 |
|---|---|
| P3B_s0 | `a03a503c8e95a7692b2febfe5711fee0fcd9554b3a8909607451582539c0c47f` |
| L_s0 | `f744c11846079449536f422bd87208c2e6d136429e8244079c0da9482f65cb74` |
| C_s0 | `6a1f93ca3885235d1866b06cb7034ee2f14dd6e4ac62f6e9f82871d6646bcc9e` |

`analysis/test_per_molecule.csv` sha256:
`252c5f10a660af7429370ad919a5244d87e2799968fad9647b7654e05e9c91d7`.

Test cache: `tracks/structural-sufficiency/zinc_arch_discovery_v1/cache/ad_test.pt`,
sha256 `981e2b9a2588b300024947edac4a5053d054199051ca19e016d350ef3ece0dc1`.
Label hash: `66fd23ecdb1b59c76cee597efc81272e0d2744eeac80744f4159d436e5388768`.

Reproduce the authoritative pass:

```bash
cd /home/calendar/code/research
uv run python tracks/structural-sufficiency/zinc_dual_channel_composer_v1/code/dcc_test_reporting.py
```

Decision remains **C — stop this configuration**. The test read is not a basis for
reopening it; if the L−C question is ever pursued, it requires a new pre-registration
and a fresh confirmation design (P3B/L/C seed-1 at minimum), not a test-driven rerun.
