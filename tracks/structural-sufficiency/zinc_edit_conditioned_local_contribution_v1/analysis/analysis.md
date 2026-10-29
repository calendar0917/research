# ECLC-v1 valid-only analysis

- Round: `zinc-edit-conditioned-local-contribution-v1`
- Created: `2026-10-29T23:41:42+0800`
- Decision: **`STOP_NO_COMPOSITION_GAIN`**
- Historical LCRC-A 4-seed valid: 0.075990; execution-drift band ≈ 0.0026

## Seed 0

| arm | run | best valid | ep | final | W20 | T100 | params | updates | GPU-h |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| REAL | `eclcREAL_s0` | 0.077558 | 514 | 0.078682 | 0.078498 | 0.078688 | 248368 | 47400 | 0.775 |
| NULL | `eclcNULL_s0` | 0.078881 | 520 | 0.079233 | 0.079184 | 0.079226 | 248368 | 47400 | 0.683 |

### Slices

| slice | n | REAL | NULL | REAL−NULL | 95% CI | REAL win |
|---|---:|---:|---:|---:|---|---:|
| all | 1000 | 0.077558 | 0.078881 | -0.001324 | [-0.005707, 0.002686] | 0.491 |
| ordinary_y_gt_-3 | 935 | 0.062548 | 0.062458 | 0.000090 | [-0.004119, 0.003899] | 0.489 |
| r2_unseen_ordinary | 225 | 0.131118 | 0.133636 | -0.002518 | [-0.018558, 0.011045] | 0.467 |
| seen_ordinary | 710 | 0.040818 | 0.039901 | 0.000917 | [-0.001663, 0.003471] | 0.496 |
| all_minus_row172 | 999 | 0.069343 | 0.070904 | -0.001561 | [-0.005872, 0.002389] | 0.491 |
| all_minus_worst10 | 990 | 0.062254 | 0.061655 | 0.000598 | [-0.002731, 0.003830] | 0.488 |
| tail_y_le_-3 | 65 | 0.293469 | 0.315130 | -0.021661 | [-0.049088, 0.003138] | 0.523 |

### Promotion conditions

| condition | pass | value |
|---|:---:|---|
| 1_full_valid_at_most_0.075 | no | value=0.077558; threshold=0.075000 |
| 2_full_delta_at_most_-0.003 | no | value=-0.001324; threshold=-0.003000 |
| 3_r2_unseen_ordinary_delta_at_most_-0.005 | no | value=-0.002518; threshold=-0.005000 |
| 4_not_worse_ordinary_and_minus_row172 | yes | ordinary_delta=0.000090; minus_row172_delta=-0.001561; threshold=0.001000 |
| 5_windows_and_base_path_active | yes | real_W20=0.078498; null_W20=0.079184; real_T100=0.078688; null_T100=0.079226; base_ablation_abs_mae_change=0.398513; base_ablation_threshold=0.000100; base_per_molecule_std=0.569828 |

Base ablation: MAE 0.077558 → 0.476070 (|Δ|=3.985e-01); per-molecule base std=0.569828.
