# Supervision-support intervention (official train only)

* target keys: 90
* sizes: {'H': 801, 'Q_target': 350, 'D_plus': 451, 'Q_background': 650, 'Q_total': 1000, 'M': 1000, 'D_minus': 451, 'B': 7098, 'F_plus': 7549, 'F_minus': 7549}

| comparison | Q_target MAE plus / minus | Δ (plus−minus) [95% CI] | Q_background Δ | difference-in-differences | key-level Δ [95% CI] |
|---|---:|---|---:|---:|---|
| plus_minus_best_monitor_s0 | 0.08232 / 0.12432 | -0.04200 [-0.08588, -0.01621] | +0.00104 | -0.04303 | -0.10318 [-0.25717, -0.01979] |
| plus_minus_final_s0 | 0.08240 / 0.12369 | -0.04129 [-0.08473, -0.01566] | +0.00313 | -0.04442 | -0.10192 [-0.25461, -0.01912] |
| plus_minus_best_monitor_s1 | 0.08211 / 0.11593 | -0.03382 [-0.07570, -0.01001] | -0.00121 | -0.03261 | -0.09199 [-0.24274, -0.01149] |
| plus_minus_final_s1 | 0.07999 / 0.11582 | -0.03583 [-0.07795, -0.01185] | -0.00247 | -0.03336 | -0.09525 [-0.24637, -0.01433] |

Negative Δ = the plus (target-environment donor) fit has lower error.

Strata by target-key training support:

* **plus_minus_best_monitor_s0**: support_4_7: Δ=-0.0934 (n=98), support_8_15: Δ=-0.0201 (n=144), support_16_25: Δ=-0.0245 (n=115)
* **plus_minus_final_s0**: support_4_7: Δ=-0.0915 (n=98), support_8_15: Δ=-0.0214 (n=144), support_16_25: Δ=-0.0222 (n=115)
* **plus_minus_best_monitor_s1**: support_4_7: Δ=-0.0826 (n=98), support_8_15: Δ=-0.0168 (n=144), support_16_25: Δ=-0.0146 (n=115)
* **plus_minus_final_s1**: support_4_7: Δ=-0.0866 (n=98), support_8_15: Δ=-0.0169 (n=144), support_16_25: Δ=-0.0174 (n=115)
