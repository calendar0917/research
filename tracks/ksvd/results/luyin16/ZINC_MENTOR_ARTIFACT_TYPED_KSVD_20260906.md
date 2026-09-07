# luyin16-zinc-mentor-artifact-typed-ksvd-sta-cross-v1

Single-seed ZINC transfer of the mentor-aligned S + T + A K-SVD proxy.

- seed: `0`; official split sizes: `{'test': 1000, 'train': 10000, 'valid': 1000}`
- dictionary: train-only `K=64`, `T=8`; exact patch width `840`
- contexts: `ring5, ring6, ring_any, multi_ring, ring_boundary`

| view | dimension | train CV MAE | valid MAE | test MAE after train+valid refit |
|---|---:|---:|---:|---:|
| `sta_final` | 2907 | 0.496268 | 0.440145 | 0.467486 |

- best valid view: `sta_final`; best valid MAE `0.440145`
- selected-view test MAE: `0.467486`

The ZINC `ring_any` context is a cycle-membership proxy because ZINC has no OGB aromatic atom flag.
