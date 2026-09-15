# Structural-encoder mechanism decomposition — official-test closure (B-bag)

**Date:** 2026-09-15
**Branch:** `exp/structural-encoder-mechanism-decomposition`
**Selection code / runs:** `5b19132`
**Test-closure code:** `4802ba8`
**Run tag:** `sbpe-test` (GPU0 worker; evaluation itself runs on CPU)
**Remote:** `res` = `hxy@a100-2`, repo `/home/hxy/cy/research`
**Authorization:** explicit human authorization for this one terminal read
**Status:** one-shot official-test read performed; `official_test_evaluated = true`

This is the terminal closure of the B-bag mechanism control. It reuses the
pre-existing, validation-frozen seed0/seed1 selection states and fixed Top-5
soups, never re-selects an epoch, and never changes the soup rule. The official
ZINC test split was loaded exactly once. The earlier selection/mechanism note
(`structural_encoder_mechanism_decomposition.md`) remains valid-selection only;
this file is the only place in the study that reports official-test numbers.

## 1. Frozen assets at closure time

`official_test_freeze.json` records the frozen configuration before any test
read (`test_loaded_at_freeze_time: false`):

| seed | raw valid | Top-5 soup valid | Top-5 epochs |
|---|---:|---:|---|
| 0 | 0.129574 | 0.127382 | [199, 210, 212, 219, 209] |
| 1 | 0.128219 | 0.120229 | [174, 210, 198, 211, 204] |

Soup rule: K=5 lowest official-valid MAE, tie -> earlier epoch, equal-weight
arithmetic parameter mean. Total params 84,511. `official_test_unlock.json` was
written before the test load, so a second read is refused.

## 2. Official-test result (n = 1000)

| seed | raw selection test MAE | Top-5 soup test MAE |
|---|---:|---:|
| 0 | 0.109535 | 0.102662 |
| 1 | 0.106889 | 0.094821 |
| **2-seed mean** | **0.108212** (std 0.001871) | **0.098742** (std 0.005544) |

- The reported 2-seed soup is the equal-weight mean of the per-seed Top-5 soup
  test MAEs, matching the validation definition.
- Diagnostic only (not a single-model result): 2-seed **prediction** ensemble
  soup test MAE 0.089266, raw 0.096360.

## 3. Same-regime reference: frozen cell A test closure

Both were produced in the deterministic-A100 regime, same seeds, same split,
same frozen Top-5 soup rule:

| condition | params | valid soup (2-seed) | test soup (2-seed) |
|---|---:|---:|---:|
| cell A (typed lookup; frozen base) | 85,763 | 0.126368 | 0.106717 |
| **B-bag (this closure)** | 84,511 | 0.123806 | **0.098742** |

B-bag is better than frozen cell A on **both** valid (−0.002562) and test
(−0.007975) soup. The direction is consistent across the two splits.

## 4. What this does and does not establish

- **Does:** B-bag's validation advantage over frozen cell A transfers to the
  official test split; the connectivity-free bag encoder is a genuine
  improvement over the frozen base, not a validation artifact. The test result
  is consistent with the valid ordering.
- **Does not:** A2 and B-full were **not** evaluated on test (each terminal read
  is one-shot and only B-bag was authorized), so the test split cannot re-derive
  the `B-full < A2 < B-bag` valid ordering, nor the `Delta_connectivity`
  decomposition. The mechanism conclusion (connectivity supported, composition
  inconclusive, rank-1 channel) rests on the valid-selection evidence; the test
  numbers are a closure check, not a new selection signal.
- No tuning, checkpoint choice, or model decision used the test numbers.

## 5. Reproduce

```bash
# one-shot; refuses to run twice (official_test_unlock.json guard)
uv run python -m tracks.ksvd.experiments.luyin16.zinc_shared_bag_patch_encoder \
    test --tag sbpe --deterministic
```

## 6. Artifacts

- `results/shared_bag_patch_encoder/official_test_freeze.json`
- `results/shared_bag_patch_encoder/official_test_unlock.json`
- `results/shared_bag_patch_encoder/official_test_results.json`
  (`official_test_evaluated: true`, `test_used_for_selection_or_tuning: false`,
  `git_commit: 4802ba89...`)
- comparison reference:
  `results/compact_v4_recurrent_pair_centre_capacity_test_closure/official_test_results.json`
