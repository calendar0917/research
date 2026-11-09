# Report tables (auto-generated)

- round: `lcrc-pcba-bottleneck-diagnosis-v1`
- label: `POST_HOC_EXPOSED_TEST_DIAGNOSIS`

## T1. Frozen checkpoints and replay gates

| model | epoch | Valid AP (replay / recorded) | Test AP (replay / recorded) | Valid Δ | Test Δ |
|---|---|---|---|---|---|
| W2 | 24 | 0.265279072 / 0.265274028 | 0.253616863 / 0.253616863 | 5.0e-06 | 0.0e+00 |
| S3 | 29 | 0.271488153 / 0.271486502 | 0.250633382 / 0.250633382 | 1.7e-06 | 0.0e+00 |
| RC_REAL | 33 | 0.267714274 / 0.267712841 | 0.255940917 / 0.255940917 | 1.4e-06 | 0.0e+00 |
| BOC_FULL | 29 | 0.269954500 / 0.269949642 | 0.254142388 / 0.254142388 | 4.9e-06 | 0.0e+00 |
| BOC_BOND | 30 | 0.269908930 / 0.269907566 | 0.251666196 / 0.251666196 | 1.4e-06 | 0.0e+00 |

## T2. P1 prediction similarity (HIGH_SUPPORT tasks)

| pair | split | median rho [q1, q3] | median Jaccard K1 | median Jaccard K5 | pos in K1 (both / only new / only ref) | median Δpct of positives |
|---|---|---|---|---|---|---|
| BOC_BOND_minus_BOC_FULL | valid | 0.678 [0.600, 0.741] | 0.329 | 0.347 | 6920 / 2691 / 2730 | +0.0000 |
| BOC_BOND_minus_BOC_FULL | test | 0.674 [0.597, 0.743] | 0.332 | 0.353 | 6537 / 2661 / 2835 | -0.0000 |
| BOC_FULL_minus_RC_REAL | valid | 0.626 [0.562, 0.696] | 0.308 | 0.334 | 6729 / 2921 / 2877 | +0.0001 |
| BOC_FULL_minus_RC_REAL | test | 0.628 [0.558, 0.697] | 0.322 | 0.327 | 6464 / 2908 / 2837 | +0.0000 |
| BOC_FULL_minus_W2 | valid | 0.656 [0.579, 0.733] | 0.333 | 0.351 | 6904 / 2746 / 2669 | +0.0000 |
| BOC_FULL_minus_W2 | test | 0.659 [0.579, 0.734] | 0.330 | 0.338 | 6512 / 2860 / 2731 | +0.0002 |
| RC_REAL_minus_W2 | valid | 0.659 [0.600, 0.728] | 0.328 | 0.355 | 6819 / 2787 / 2754 | +0.0000 |
| RC_REAL_minus_W2 | test | 0.660 [0.599, 0.730] | 0.330 | 0.347 | 6547 / 2754 / 2696 | +0.0001 |
| S3_minus_W2 | valid | 0.739 [0.693, 0.797] | 0.392 | 0.410 | 7274 / 2345 / 2299 | +0.0001 |
| S3_minus_W2 | test | 0.740 [0.690, 0.799] | 0.390 | 0.401 | 6943 / 2328 / 2300 | +0.0000 |

Verdicts: `BOC_BOND_minus_BOC_FULL` → `H1_NOT_SUPPORTED`; `BOC_FULL_minus_RC_REAL` → `H1_NOT_SUPPORTED`; `BOC_FULL_minus_W2` → `H1_NOT_SUPPORTED`; `RC_REAL_minus_W2` → `H1_NOT_SUPPORTED`; `S3_minus_W2` → `H1_NOT_SUPPORTED`

## T3. P2 Train-only WL2 novelty groups

| split | group | molecules | positives | median heavy atoms | tasks with group-pos ≥20 |
|---|---|---|---|---|---|
| valid | SEEN | 32656 | 39304 | 27 | 102 |
| valid | PARTIAL_NOVEL | 9262 | 13765 | 25 | 86 |
| valid | HIGH_NOVEL | 1875 | 2895 | 21 | 37 |
| test | SEEN | 30728 | 37711 | 27 | 101 |
| test | PARTIAL_NOVEL | 10453 | 15245 | 25 | 87 |
| test | HIGH_NOVEL | 2612 | 3742 | 21 | 50 |

## T4. P3 novelty vs gain (task-matched shared main cells)

| pair | group | shared tasks | mean ΔAP Valid | mean ΔAP Test | decay (Test−Valid) | improved Valid → Test |
|---|---|---|---|---|---|---|
| BOC_FULL_minus_W2 | SEEN | 99 | +0.0053 | +0.0098 | +0.0045 | 57/99 → 66/99 |
| BOC_FULL_minus_W2 | PARTIAL_NOVEL | 84 | +0.0063 | -0.0055 | -0.0118 | 47/84 → 44/84 |
| BOC_FULL_minus_W2 | HIGH_NOVEL | 37 | +0.0131 | -0.0133 | -0.0264 | 26/37 → 15/37 |
| RC_REAL_minus_W2 | SEEN | 99 | +0.0039 | +0.0089 | +0.0050 | 59/99 → 72/99 |
| RC_REAL_minus_W2 | PARTIAL_NOVEL | 84 | +0.0011 | -0.0121 | -0.0132 | 42/84 → 35/84 |
| RC_REAL_minus_W2 | HIGH_NOVEL | 37 | +0.0042 | +0.0046 | +0.0003 | 20/37 → 21/37 |
| S3_minus_W2 | SEEN | 99 | +0.0075 | +0.0012 | -0.0063 | 69/99 → 54/99 |
| S3_minus_W2 | PARTIAL_NOVEL | 84 | +0.0014 | -0.0104 | -0.0119 | 47/84 → 33/84 |
| S3_minus_W2 | HIGH_NOVEL | 37 | +0.0116 | +0.0016 | -0.0100 | 20/37 → 17/37 |

Verdicts: `BOC_FULL_minus_W2` → `NOVELTY_EFFECT_UNRESOLVED`; `RC_REAL_minus_W2` → `NOVELTY_EFFECT_UNRESOLVED`; `S3_minus_W2` → `NO_CLEAR_NOVELTY_TREND`

## T5. P4 macro-AP decomposition (Test unless stated)

| pair | mean ΔAP Valid | mean ΔAP Test | high-support (≥20) mean ΔAP | improved (high) | low-support (≤19) sum ΔAP | low-support share of |Δ| | Valid+ → Test− | single largest ±ΔAP |
|---|---|---|---|---|---|---|---|---|
| BOC_BOND_minus_W2 | +0.0046 | -0.0020 | +0.0005 | 64/108 | -0.2973 | 39.2% | 27/127 | +0.116 (t62, n=11) / -0.232 (t108, n=7) |
| BOC_FULL_minus_W2 | +0.0047 | +0.0005 | +0.0032 | 61/108 | -0.2800 | 41.3% | 29/127 | +0.175 (t77, n=10) / -0.321 (t31, n=6) |
| RC_REAL_minus_W2 | +0.0024 | +0.0023 | +0.0009 | 65/108 | +0.1951 | 38.5% | 21/127 | +0.125 (t57, n=4) / -0.231 (t120, n=11) |
| S3_minus_W2 | +0.0062 | -0.0030 | -0.0009 | 60/108 | -0.2822 | 39.0% | 33/127 | +0.127 (t57, n=4) / -0.325 (t31, n=6) |

Low-support stratum = Test positives 5–19 (18 tasks) + 1–4 (1 task). Top support-asymmetry labels: `BOC_BOND_minus_W2` → `H3_SUPPORTED_BY_SUPPORT_ASYMMETRY`; `BOC_FULL_minus_W2` → `H3_SUPPORTED_BY_SUPPORT_ASYMMETRY`; `RC_REAL_minus_W2` → `H3_MIXED`; `S3_minus_W2` → `H3_MIXED`

## T6. Frozen task sets

| set | n |
|---|---|
| ALL_test | 127 |
| ALL_valid | 127 |
| HIGH_SUPPORT | 106 |
| LOW_SUPPORT_test | 19 |
| TEST_HIGH_BUT_VALID_LOW_SUPPORT | 2 |
| n_tasks | 128 |

Tasks excluded from HIGH_SUPPORT although Test positives ≥20: [9, 34] (Validation positives 13–14).

## T7. Structural support build

- Train molecules: 350343, Train atoms: 9001316, distinct WL2: 112623 (31154 appear in one molecule, 62463 in ≤4), build 165s
- parse modes (Train): {'sanitized': 350341, 'unsanitized_fallback': 2}
- labels read: False
