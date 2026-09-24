# E2E-DictEnv-T1 — terminal official-test read

Round **E2E-DictEnv-T1** · protocol `e2e_dictenv_t1` · study `zinc-context-gap`.
[analysis](e2e_dictenv_t1_analysis.md) ·
[implementation](e2e_dictenv_t1_implementation.md).

This note is **reporting only**.  The configuration, λ, horizon, checkpoints,
soup members and the seed-1 pair were all frozen on train/valid before this read
(`architecture_freeze.json`, committed before the read).  No gate below may
change any decision; the round's verdict was already fixed as
`E2E_DICTENV_TUNED_MECHANISM_LOST`.

---

## 1. Unlock discipline

`official_test_unlock.json` (written by the single dedicated `unlock`
process, commit `3c13c816cede53e5e03d5933fdada5e908561201`):

```text
user_authorised_test_read                              true
purpose                                                terminal reporting only
project_wide_pristine                                  false
reason            historical unrelated ZINC official-test reads already exist
architecture_frozen_before_this_rounds_test_read       true
test_will_not_affect_any_model_config_checkpoint_decision  true
```

The test split was loaded exactly once (`n_test = 1000`), after every valid
decision was frozen.  This does **not** claim project-wide test purity.

---

## 2. Frozen objects on the official test

| object | test MAE | mean pred | std pred | params |
|--------|---------:|----------:|---------:|-------:|
| final Sparse seed 0 soup (`stage_b_lambda025`) | **0.098076** | 0.0197 | 2.0212 | 66132 |
| final Sparse seed 1 soup | 0.104108 | 0.0188 | 1.9961 | 66132 |
| final DenseTied seed 1 soup | 0.103006 | 0.0255 | 2.0048 | 66132 |
| v0 Sparse seed 0 soup (read-only historical) | 0.109373 | 0.0318 | 2.0127 | 66158 |

---

## 3. Mechanism-generalisation diagnostics (final Sparse seed 1)

```text
clean local test MAE            0.104108
zero-code test MAE              0.150156   G_dict-use^test = +0.046048
5x assignment-shuffle test MAE  0.117051   G_assign^test   = +0.012943
per-permutation                 0.118480 / 0.116007 / 0.115522 / 0.119871 / 0.115373
```

---

## 4. Reading (descriptive only)

* The tuned seed-0 Sparse state transfers best of the frozen objects on the test
  split (`0.098076` vs the v0 Sparse `0.109373`, a `+0.011297` test-side
  improvement), consistent with the valid-side gain.
* The **valid-selected seed-1 specificity does not replicate on test**: on valid
  Sparse (0.125496) beat DenseTied (0.131315) by `+0.005819`, but on test Sparse
  seed 1 (`0.104108`) is `0.001102` **worse** than DenseTied seed 1 (`0.103006`).
  This is reported as an observation; the valid-side specificity gate was a
  frozen decision input and no test number is allowed to reopen it.
* The two behavioural interventions are *larger* on test than on valid
  (`G_dict-use` +0.0460 vs +0.0179; `G_assign` +0.0129 vs +0.0076, i.e. above the
  0.010 gate on test but below it on valid).  The mechanism-loss verdict is a
  valid-set decision and stands; the test diagnostic is simply noted.
* Test MAE is systematically lower than valid MAE for every object, i.e. the
  held-out distribution is easier than the valid split; no gate is defined on
  this split and none was applied.

No model configuration, checkpoint, soup member, seed or hyper-parameter was
changed, added or re-selected after this read.
