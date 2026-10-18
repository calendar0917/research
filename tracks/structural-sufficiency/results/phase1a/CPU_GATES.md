# Phase 1a CPU gates — 8/8 PASS

日期: 2026-10-19  
协议: `ssrc-phase1a-lipo-v1`  
完整 JSON（本地证据，git-ignored）: `cpu_gates.json`  
运行: `uv run python tracks/structural-sufficiency/code/phase1a/phase1a_gates.py`

| # | Gate | Result | Key evidence |
|---|---|---|---|
| 1 | `split_structure` | PASS | Fit/Monitor/Reserved = 2352/504/504; small=253; scaffold group overlap all 0; row hashes match; Phase 0 candidate 逐行一致 |
| 2 | `label_isolation` | PASS | label vault retains 757 rows for n=253 (253 Fit + 504 Monitor), `reserved_labels_retained=0`; Reserved request and wrong-purpose request both `PermissionError` |
| 3 | `e2_collision_identity` | PASS | C24/caterpillar/grid: `E_2(d6)=E_2(d9)`; S/X prediction max abs diff 5.96e-8 / 0.0 / 1.49e-8 |
| 4 | `permutation_invariance` | PASS | patch order / atom relabel / patch edge order max abs diff ≤ 5.96e-8 for S and X |
| 5 | `input_isolation` | PASS | S/X batch keys exactly patch-only; G batch exactly graph-only; injected `full_edge_index` rejected |
| 6 | `shared_init_determinism` | PASS | S/X shared local encoder and head state_dicts tensor-identical; same seed reproducible; data permutation reproducible |
| 7 | `training_smoke` | PASS | S/X/G forward+backward on 32 rows, 2 epochs; finite losses; nonzero gradients |
| 8 | `cost_accounting` | PASS | parameters S=59,011, X=59,075, G=53,379; S/X=0.9989; forward FLOPs (fixed 64) S=622.3M, X=877.7M, G=170.5M; preprocess 0.48 ms/molecule |

## Caveats recorded before GPU

- FLOP counter is `torch.utils.flop_counter.FlopCounterMode`; PyG scatter ops may be
  uncounted, so these are lower bounds.
- X uses ~1.41× S forward FLOPs at matched parameters. This is a real compute
  difference and is reported; if X wins, extra compute is an alternative explanation.
- Preprocessing is CPU-only patch extraction from the committed label-free raw
  graph artifact; GPU hosts do not need RDKit.
- Phase 0 LOO used all 3360 official-train labels, so the internal Reserved split
  is not historically fully blind. Phase 1a still does not read or score it.
