# Compact-v4 T=2 recurrent Q16 — minimal residual-magnitude gate

Date: 2026-09-14
Protocol: `compact_v4_recurrent_residual_gate_v1`
Module: `experiments/luyin16/zinc_compact_v4_recurrent_residual_gate.py`
Tests: `tests/test_compact_v4_recurrent_residual_gate.py` (14 pass)
Results: `results/compact_v4_recurrent_residual_gate/`
Official test: **never loaded.**

## Question

The canonical T=2 weight-tied recurrent pair–centre model (82,115 params) works,
but its training trajectory is noisy: the fixed Top-5 checkpoint soup improves
on 3/3 seeds (mean `+0.00314`), and the canonical 4-thread loop injects
irreducible execution noise of order `~0.003`. Does the simplest possible
*bound on the recurrent update magnitude* stabilise the dynamics and improve
generalisation without adding capacity?

## Change (exactly one parameter)

```
alpha = sigmoid(a)                         # a: one model-wide scalar
h^(t+1) = h^(t) + alpha * U(h^(t), A^(t))  # both T=2 rounds share alpha
```

* `a` is a single `nn.Parameter(torch.zeros(()))` → `alpha == 0.5` at init.
* `gate_update_magnitude=False` (default) adds **no** parameter and is
  bit-identical to the canonical ungated model.
* Placement: `PatchPathRecurrentPairCentreModel.__init__` creates
  `update_gate_logit`; `_encode_core` multiplies `delta` by
  `self.update_alpha()` at the single centre-update call site inside the
  `for round_index in range(rounds)` loop.
* Builder: `build_gated(seed)` = `build_recurrent(seed, gate_update_magnitude=True)`.
* Everything else (tokenizer, patch/relation/pair encoders, distance buckets,
  centre pooling/aggregation, readout, global/topology, small head, Adam,
  lr 1e-3, wd 1e-5, batch 128, 240 epochs, patience 40, no scheduler) is
  inherited verbatim. No channel/context gate, LayerNorm, attention, dropout,
  EMA/SWA, T=3, Q24/Q32, or new feature.

Parameter count: baseline `82,115` → gated **`82,116`** (`+1`); head `4,135`.

## Sanity (all 15 pass; `results/.../sanity.json`)

* params `+1` exactly; ungated named params == baseline; gated adds only
  `update_gate_logit`.
* `alpha(0) == 0.5`; a single gate parameter; exactly one `update_alpha()`
  application inside the round loop.
* with the centre update activated, forcing `a = -30` (`alpha < 1e-12`) collapses
  the `T=2` model to the baseline function (`max|Δ| = 4e-13`); a round-2-only
  ungated path would leave an `O(U)` gap — so **both rounds are gated**.
* round-1 update scales exactly with `alpha` (`0.68393975` observed vs
  `0.68393970` expected).
* gate gradient is non-zero and finite once the update is active.
* dropping the gate (`update_gate_logit = None`) reproduces the ungated recurrent
  encode **bit-identically**; shared init is bit-identical.
* canonical ungated seed0 checkpoint loads strictly and re-evaluates to
  `0.1406094916117727` — the ungated builder/checkpoint path is intact.
* no NaN/Inf in forward/backward.

## Phase 1 — gated seed0 / seed1

Protocol identical to canonical (Adam, lr `1e-3`, wd `1e-5`, batch `128`,
`max_epochs 240`, `patience 40`, no scheduler, best official-valid checkpoint).
4 threads, strict sequential, canonical stochastic mode (no deterministic
algorithms). Snapshots per epoch; locked Top-5 rule (5 lowest selection MAE,
ties → earliest epoch, equal-weight average, no k/weight search).

| seed | raw valid | best epoch | train@best | gap | Top-5 soup | soup gain | best alpha |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.138450** | 198 | 0.065444 | 0.073006 | **0.137714** | +0.000735 | **0.4592** |
| 1 | **0.130123** | 221 | 0.063513 | 0.066610 | **0.130028** | +0.000095 | **0.4692** |

Ungated canonical reference: seed0 raw `0.140609` @167 / soup `0.137078`;
seed1 raw `0.133440` @234 / soup `0.132210`.

### Deltas (ungated − gated, positive = gated better)

| metric | seed0 | seed1 | 2-seed mean |
|---|---:|---:|---:|
| raw valid | +0.002160 | +0.003317 | **+0.002738** |
| Top-5 soup | **−0.000637** | +0.002181 | **+0.000772** |

## Primary decision — no meaningful gate signal

Pre-registered primary metric = 2-seed Top-5 soup mean.

* gated soup mean `0.133871` vs ungated `0.134644` → improvement **`+0.000772`**,
  which is **≤ the `0.001` no-signal gate**.
* seed0 soup is slightly *worse* (`−0.000637`), seed1 soup clearly better
  (`+0.002181`): a seed-dependent direction, but the mean is inside the noise
  band. Not a strong positive (needs `≥ 0.002`).

**Operational decision: no signal → stop. Do not run gated seed2, do not sweep
`alpha` init / fixed alpha / vector gate / LayerNorm.**

## Stability diagnostics (`stability.json`)

Best-checkpoint seed0 vs seed1:

| metric | ungated | gated | direction |
|---|---:|---:|---|
| mean abs prediction disagreement | 0.089008 | 0.093403 | **worse** |
| prediction correlation | 0.997232 | 0.993188 | worse |
| residual correlation | 0.979447 | 0.969886 | worse |
| raw MAE spread | 0.009282 | 0.008327 | narrower |
| best-epoch spread | 67 (167/234) | **23 (198/221)** | **much tighter** |

Top-5 soup seed0 vs seed1:

| metric | ungated | gated | direction |
|---|---:|---:|---|
| mean abs prediction disagreement | 0.082384 | 0.082169 | ≈ equal |
| prediction correlation | 0.997208 | 0.996996 | ≈ equal |
| residual correlation | 0.980376 | 0.979614 | ≈ equal |
| soup MAE spread | 0.005852 | 0.007686 | wider |

The gate did **not** make the two seeds learn a more consistent function: best
model disagreement/correlation worsened, soup disagreement/correlation is flat.
The one clear stabilisation is best-epoch alignment (spread 67 → 23), which is a
trajectory-timing effect, not a functional-consistency gain.

## Learned alpha (`alpha_report.json`)

* best checkpoint: seed0 `0.4592`, seed1 `0.4692` (range `0.4592–0.4692`).
* Top-5 checkpoints range `0.4569–0.4699`; the soup alpha is `0.4593` / `0.4692`.
* both seeds learn to shrink the update magnitude consistently and
  monotonically: `0.5 → ~0.4565` (seed0) / `~0.4679` (seed1). The gate is
  genuinely trained and always converges slightly below the `0.5` init.

## Verdict

Primary metric (`2-seed Top-5 soup mean`) is **inside the pre-registered
no-signal band** (`+0.00077 ≤ 0.001`), with opposite per-seed soup directions.
The un-gated canonical soup already captures most of the available variance
reduction, so the gate's raw single-model improvement (`+0.00274`, positive on
both seeds) is not confirmed by the soup evaluator. Cross-seed function
consistency did not improve.

**Classification: `no meaningful gate signal` (operationally), with a
seed-dependent/inconclusive flavour on the soup metric.** Raw MAE improved on
both seeds and best-epoch spread collapsed (67 → 23), but the pre-registered
soup metric and the prediction-disagreement diagnostics do not support a
generalisation/stability claim.

**Close the direction.** No `alpha` init sweep, fixed-alpha sweep, channel/vector
gate, LayerNorm, EMA/SWA, or seed2 replication. Official test never accessed.
