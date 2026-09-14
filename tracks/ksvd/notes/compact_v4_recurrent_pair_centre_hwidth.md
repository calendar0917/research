# Compact-v4 T=2 recurrent pair--centre — centre-state width (H48 -> H64)

Date: 2026-09-14
Protocol: `compact_v4_recurrent_pair_centre_hwidth_v1`
Reference: canonical T=2 recurrent pair--centre, `h=48, q=16`, 82,115 params
Official test: **never loaded**.

## Question

`q_dim: 16 -> 24` gave no signal.  The effective computation loop is
`relation -> centre -> refreshed relation -> centre`; this audit changes
**only** the persistent centre state width:

```text
h_dim: 48 -> 64        q_dim = 16, T = 2
```

Everything else is inherited verbatim (tokenizer / patch construction /
relation descriptor / pair projection / pair encoder / ReLU / distance buckets
/ recurrent refresh logic / readout / small raw head / Adam / lr / wd / batch /
max epochs / patience / no scheduler / seeds).  No new module.

## Architecture difference / parameters

| model | h | q | total | head | backbone | unary | pair | centre ctx | R width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H48 (reference) | 48 | 16 | 82,115 | 4,135 | 77,980 | 97 | 33 | 165 | 302 |
| H64 | 64 | 16 | **85,763** | **4,551** | **81,212** | **129** | 33 | 165 | **334** |

`delta = +3,648` params.  Only these tensors change shape:

| tensor | H48 | H64 | delta |
|---|---:|---:|---:|
| `patch_encoder.layers.4.weight` | 48x64 | 64x64 | +1,024 |
| `patch_encoder.layers.4.bias` | 48 | 64 | +16 |
| `pair_projection.weight` | 16x48 | 16x64 | +256 |
| `center_update.0.weight` | 60x213 | 60x229 | +960 |
| `center_update.4.weight` | 48x60 | 64x60 | +960 |
| `center_update.4.bias` | 48 | 64 | +16 |
| `head.net.0.weight` | 13x302 | 13x334 | +416 |

`q_dim` and the pair channel are untouched; `center_context_width` stays 165.

## Sanity (all pass, `sanity.json`)

* H48 capacity builder is **bit-identical** to the frozen recurrent builder
  (state hash equal, output max diff 0.0) — H48 checkpoints unaffected.
* Only the 7 h-dependent tensors above change shape.
* Every same-shape **backbone** tensor is bit-identical to H48.  The head is
  re-drawn from the same `head_seed` after a width-dependent first Linear, so
  its later same-shape tensors legitimately differ.
* `q_dim` still 16; T=2 `refresh` flags intact; module call counts
  `{pair_projection:4, pair_encoder:2, center_update:2}` (weight tying).
* Perturbing the zero-init centre update gives `q1 != q0` (max drift 0.967,
  cosine 0.498), `h1 != h0`, `h2 != h1` — the refresh really recomputes from
  the updated centre state.
* Readout width 334, head in-features 334, `center_update[0].out=60`,
  `center_context_width=165` — no other hidden dim widened.
* Forward/backward finite and non-zero; deterministic rebuild.
* Exact params 85,763 / head 4,551.

## Training (canonical protocol, sequential)

Adam, lr 1e-3, wd 1e-5, batch 128, max_epochs 240, patience 40, no scheduler.
Snapshots saved per epoch for the fixed Top-5 soup (no RNG effect).

| seed | raw best-valid | best epoch | epochs | train@best | gap | epoch time | wall | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.134240 | 186 | 226 | 0.072252 | 0.061988 | 7.99 s | 1804.6 s | 2.42 GB |
| 1 | 0.135201 | 145 | 185 | 0.075013 | 0.060187 | 8.03 s | 1485.2 s | 2.42 GB |

H48 canonical raw: seed0 0.140609 @167, seed1 0.133440 @234; raw 2-seed mean
0.137025.  H64 raw 2-seed mean **0.134720** (`+0.002305`).

## Fixed Top-5 checkpoint soup (repo-locked rule)

Lowest selection-MAE 5 checkpoints, ties -> earliest epoch, equal-weight
parameter average, no `k` / weight search.

| seed | soup MAE | top-5 epochs | soup gain over best |
|---:|---:|---|---:|
| 0 | **0.129843** | 186, 214, 218, 215, 171 | 0.004396 |
| 1 | **0.131813** | 145, 183, 150, 146, 144 | 0.003387 |

| metric | H48 | H64 | improvement |
|---|---:|---:|---:|
| soup seed0 | 0.137078 | 0.129843 | +0.007234 |
| soup seed1 | 0.132210 | 0.131813 | +0.000396 |
| **2-seed soup mean** | **0.134644** | **0.130828** | **+0.003815** |

## Diagnostics

| quantity | H48 | H64 |
|---|---:|---:|
| raw 2-seed spread | 0.007170 | 0.000961 |
| soup 2-seed spread | 0.004868 | 0.001970 |
| best-epoch spread | 67 (167/234) | 41 (186/145) |
| seed prediction disagreement (raw) | 0.08901 | 0.08722 |
| seed prediction disagreement (soup) | 0.08238 | 0.07368 |

Centre-state effective rank (participation ratio), no dead dims:

| state | H48 seed0 | H64 seed0 | H48 seed1 | H64 seed1 |
|---|---:|---:|---:|---:|
| `h1` | 27.43 / 48 | 32.94 / 64 | 21.86 / 48 | 31.02 / 64 |
| `h2` | 24.36 / 48 | 28.62 / 64 | 16.77 / 48 | 26.99 / 64 |

The widened centre state is not dead: seed1's `h2` effective rank rises from
16.8 (of 48) to 27.0 (of 64), and both seeds use the extra dimensions.  `q1`
still shows a few dead relation dims (0.06 / 0.25), so relation width is still
not the binding channel.

Cross-architecture (same seed) prediction disagreement (raw): 0.0792 / 0.0924.

## Verdict

`2-seed Top-5 soup mean 0.130828` vs `H48 0.134644` -> **improvement
+0.003815**, above the pre-registered strong gate `+0.002`, and both seeds
move in the same soup direction.

**Centre-state capacity-limited.**  Caveat: the effect is concentrated in
seed0 (soup +0.0072, raw +0.0064); seed1 is essentially flat (soup +0.0004,
raw -0.0018).  The variance *spreads* shrink sharply on H64, which is
consistent with the extra centre capacity stabilising training rather than
adding a uniform per-seed gain.

Next authorised single step (not run here): one wider centre config near
100k--110k params, i.e. **h=96 (103,219 params, R width 398)**.

## Files

- `experiments/luyin16/zinc_compact_v4_recurrent_pair_centre_hwidth.py`
- `tests/.../test_compact_v4_recurrent_pair_centre_hwidth.py` (6 pass)
- `results/compact_v4_recurrent_pair_centre_hwidth/`:
  `parameter_accounting.json`, `sanity.json`, `soup_h64_seed{0,1}.json`,
  `diagnostics.json`, `centre_state_h48_reference.json`, `decision.json`,
  `report.json`, `runs/`, `curves/`, `states/`, `snapshots/`, `soup_states/`.
