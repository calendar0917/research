# RBO v0 — frozen-checkpoint comparison (train-only evaluation sets)

RBO checkpoint: `/home/calendar/code/research/tracks/structural-sufficiency/zinc_rooted_branch_operator_v0/runs/rbo_minus_s0/best_monitor.pt` (epoch 439), parameters 331521
P3-minus reference: `/home/calendar/code/research/tracks/structural-sufficiency/zinc_composition_transfer_v1/runs/sup_minus_s0/best_monitor.pt` (epoch 424)

| set | n | RBO MAE | P3-minus MAE | Δ (RBO−P3) | 95% CI | median Δ | better | sign p |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| monitor | 1000 | 0.082757 | 0.080424 | +0.002333 | [-0.002976, +0.007703] | +0.000399 | 0.496 | 0.8003 |
| q_target | 350 | 0.128432 | 0.124316 | +0.004116 | [-0.008111, +0.016315] | +0.000635 | 0.494 | 0.8307 |
| q_background | 650 | 0.104169 | 0.100407 | +0.003762 | [-0.003208, +0.010778] | +0.000489 | 0.494 | 0.7537 |

P3-minus epoch-600 (secondary, no selection): monitor 0.080989, q_target 0.123690, q_background 0.098288

| r2-unseen atom fraction | set | n | RBO | P3-minus | Δ |
|---|---|---:|---:|---:|---:|
| <0.05 | monitor | 800 | 0.062354 | 0.061033 | +0.001321 |
| <0.05 | q_target | 152 | 0.085785 | 0.097002 | -0.011217 |
| <0.05 | q_background | 521 | 0.064507 | 0.062570 | +0.001937 |
| 0.05-0.15 | monitor | 147 | 0.134829 | 0.130297 | +0.004532 |
| 0.05-0.15 | q_target | 150 | 0.159555 | 0.155158 | +0.004396 |
| 0.05-0.15 | q_background | 95 | 0.253898 | 0.251943 | +0.001955 |
| 0.15-0.30 | monitor | 43 | 0.209910 | 0.177436 | +0.032475 |
| 0.15-0.30 | q_target | 40 | 0.171775 | 0.123852 | +0.047922 |
| 0.15-0.30 | q_background | 26 | 0.270444 | 0.248284 | +0.022159 |
| >=0.30 | monitor | 10 | 0.402772 | 0.481350 | -0.078578 |
| >=0.30 | q_target | 8 | 0.138472 | 0.067314 | +0.071158 |
| >=0.30 | q_background | 8 | 0.368714 | 0.284394 | +0.084320 |

| stratum | n mol | n keys | RBO | P3-minus | Δ |
|---|---:|---:|---:|---:|---:|
| support_4_7 | 98 | 43 | 0.187561 | 0.198141 | -0.010579 |
| support_8_15 | 144 | 33 | 0.113068 | 0.102253 | +0.010814 |
| support_16_25 | 115 | 14 | 0.094724 | 0.088247 | +0.006478 |

key level: 45/90 keys better, unweighted mean Δ -0.002715, query-weighted mean Δ +0.005088, median Δ -0.002059

| probe (frozen checkpoint) | M MAE | Q_target MAE | Q_background MAE | mean |Δpred| on M |
|---|---:|---:|---:|---:|
| full | 0.082757 | 0.128432 | 0.104169 | – |
| three_body_off | 0.111627 | 0.150461 | 0.146532 | 0.057925 |
| relation_off | 0.088712 | 0.131007 | 0.108376 | 0.024384 |
| typed_hist_off | 0.088808 | 0.129892 | 0.110177 | 0.029300 |
