# Compact-v4 T=2 recurrent pair--centre — H96 endpoint + frozen official test

Date: 2026-09-14
Protocol: `compact_v4_recurrent_pair_centre_hwidth_endpoint_v1`
Official test: loaded **exactly once**, only after the architecture freeze.

## Question

Does persistent centre-state capacity keep scaling from H64 (85,763 params)
towards a ~100k budget when only ``h_dim`` changes (``q_dim=16``, ``T=2``),
and does a frozen H96 transfer its validation gain to the official test?

## Phase A — H96 architecture / parameters

| model | h | q | total | head | backbone | unary | pair | ctx | R width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H64 | 64 | 16 | 85,763 | 4,551 | 81,212 | 129 | 33 | 165 | 334 |
| **H96** | 96 | 16 | **103,219** | **5,383** | **97,836** | 193 | 33 | 165 | **398** |

`delta = +17,456`.  H96 crosses the existing derived-width floors, so the
frozen architecture's own `patch_hidden` dependencies grow naturally:
`patch_encoder` hidden `max(h,64)` 64 -> 96, `global_encoder` hidden
`max(h//2,32)` 32 -> 48; `center_context_hidden` stays 60 and relation/pair
encoder hidden stay 32/64 (driven by `q`).  No new module.

## Phase B — sanity (all pass)

H48 builder bit-identical to frozen; H64 reference assets unchanged; only the
16 h-dependent tensors change shape; same-shape backbone bit-identical to
H64; `q_dim=16`; T=2 refresh; weight tying `{pp:4,pe:2,cu:2}`; refresh probe
`q1!=q0`; readout width 398; params exact; fwd/bwd finite non-zero.

## Phase C — H96 validation

| seed | raw best-valid | best epoch | train@best | gap | soup | top-5 epochs | epoch time |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | 0.124714 | 213 | 0.059224 | 0.065490 | **0.122036** | 213,185,184,156,182 | 8.77 s |
| 1 | 0.128182 | 205 | 0.076548 | 0.051634 | **0.124446** | 205,187,236,190,200 | 8.85 s |

Both seeds ran the full 240 epochs (no early stop, no horizon warning).
peak RSS ~2.42 GB.

## Phase D — freeze gate

| metric | H64 | H96 | improvement |
|---|---:|---:|---:|
| soup 2-seed mean | 0.130828 | **0.123241** | **+0.007587** |
| per-seed soup | 0.129843 / 0.131813 | 0.122036 / 0.124446 | +0.007807 / +0.007367 |
| raw 2-seed mean | 0.134720 | 0.126448 | +0.008272 |

`improvement +0.007587 >= 0.002` and both seeds same direction
(`CLEAR_CONTINUED_SCALING`) -> **freeze H96**.  No seed2 needed, no H112/H128.

## Phase E — frozen one-shot official test

Freeze record: `results/.../architecture_freeze.json`
(`selected_architecture=H96`, params 103,219, seeds 0/1, fixed Top-5 rule,
`test_status=not yet loaded`).  Only then was the test split loaded (unlock
record `official_test_unlock.json`).

| seed | raw selection -> test | Top-5 soup -> test |
|---:|---:|---:|
| 0 | 0.107781 | 0.106668 |
| 1 | 0.108134 | 0.103313 |
| **mean ± std** | **0.107958 ± 0.000249** | **0.104990 ± 0.002373** |

Diagnostic (NOT a single model): equal-weight 2-seed prediction ensemble —
raw 0.099401, soup 0.097830.

Historical H48 test raw mean = `0.114951`
(`results/compact_v4_recurrent_pair_centre/terminal_test.json`, selection
checkpoints).  H96 raw test improves by **+0.006993** over that reference.

## Scaling curve

| model | params | valid soup mean | test raw mean | test soup mean |
|---|---:|---:|---:|---:|
| H48 | 82,115 | 0.134644 | 0.114951 (historical) | — |
| H64 | 85,763 | 0.130828 | not tested (not frozen) | — |
| **H96** | 103,219 | **0.123241** | **0.107958 ± 0.000249** | **0.104990 ± 0.002373** |

`h2` centre-state effective rank (participation ratio): H48 24.36/16.77,
H64 28.62/26.99, H96 31.27/29.89 (seeds 0/1); dead-dim fraction 0.  The extra
centre dimensions are used, but the used *fraction* of the width falls
(50% -> 45% -> 33%), i.e. sublinear rank growth.

## Verdict

* **centre width continues scaling** (valid soup 0.134644 -> 0.130828 ->
  0.123241; H96 step larger than H64 step).
* **H96 is the ~100k preferred model** (103,219 params).
* **validation scaling transfers to test**: H96 raw test 0.107958 vs H48
  historical 0.114951 (+0.006993), though attenuated relative to the
  validation raw gain (+0.010577 vs H48).
* Distance to the `<0.10` target for a single model: soup 0.104990
  (+0.004990), raw 0.107958 (+0.007958).  The diagnostic soup ensemble is
  0.097830 but is not a single-model result.

Stop rule honoured: no H112/H128, q-width, new modules or optimizer sweep.

## Files

- `experiments/luyin16/zinc_compact_v4_recurrent_pair_centre_hwidth_endpoint.py`
- `tests/.../test_compact_v4_recurrent_pair_centre_hwidth_endpoint.py` (6 pass)
- `results/compact_v4_recurrent_pair_centre_hwidth_endpoint/`:
  `parameter_accounting.json`, `sanity.json`, `runs/`, `soup_h96_seed{0,1}.json`,
  `diagnostics.json`, `decision.json`, `architecture_freeze.json`,
  `official_test_unlock.json`, `official_test_results.json`, `report.json`.
