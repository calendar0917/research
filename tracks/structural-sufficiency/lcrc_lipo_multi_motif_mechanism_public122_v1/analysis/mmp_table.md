# 真实分子对（沿用 823 MMP 对，不新挖掘）

`Δf` = 冻结模型对**原分子**的预测差；`Δy` = 实验差；`|Δf-Δy|` 与零变化基线 `|Δy|` 按 eval_component 聚合。

| family | group | subset | pairs | components | MAE | zero-change MAE | Train-class MAE (coverage) | support |
|---|---|---|---|---|---|---|---|---|
| amide | E_HH_nonTest_train_pair | all | 2 | 2 | 0.4812 | 0.4150 | 0.4630 (1.00) | LOW_SUPPORT |
| amide | E_HH_nonTest_train_pair | clean | 1 | 1 | 0.8169 | 0.7700 | n/a (0.00) | LOW_SUPPORT |
| amide | E_T_train_endpoint | all | 6 | 4 | 0.1062 | 0.5175 | 0.2660 (0.75) | LOW_SUPPORT |
| amide | E_T_train_endpoint | clean | 12 | 5 | 0.2442 | 0.5813 | 0.8694 (0.60) | LOW_SUPPORT |
| amide | P_train | all | 15 | 9 | 0.2885 | 0.6028 | 0.2267 (1.00) | LOW_SUPPORT |
| amide | P_train | clean | 23 | 13 | 0.3926 | 0.9694 | 0.8462 (1.00) | SUPPORTED |
| amide | V_valid | clean | 1 | 1 | 0.2058 | 0.3500 | 0.3633 (1.00) | LOW_SUPPORT |
| amide | other | all | 2 | 2 | 0.3264 | 0.9000 | 0.0400 (0.50) | LOW_SUPPORT |
| amide | other | clean | 12 | 7 | 0.4283 | 1.1068 | 1.5078 (0.43) | LOW_SUPPORT |
| aryl_ether | E_HH_nonTest_train_pair | all | 1 | 1 | 0.1766 | 0.0400 | n/a (0.00) | LOW_SUPPORT |
| aryl_ether | E_HH_nonTest_train_pair | clean | 7 | 5 | 0.2581 | 0.5100 | 0.5705 (1.00) | LOW_SUPPORT |
| aryl_ether | E_T_train_endpoint | all | 1 | 1 | 0.1093 | 1.0400 | n/a (0.00) | LOW_SUPPORT |
| aryl_ether | E_T_train_endpoint | clean | 23 | 12 | 0.2112 | 0.6326 | 0.7172 (0.92) | SUPPORTED |
| aryl_ether | P_train | all | 8 | 6 | 0.2363 | 0.5608 | 0.8800 (1.00) | LOW_SUPPORT |
| aryl_ether | P_train | clean | 53 | 30 | 0.2732 | 0.4862 | 0.6631 (1.00) | SUPPORTED |
| aryl_ether | V_valid | clean | 1 | 1 | 0.1861 | 0.3500 | 0.3729 (1.00) | LOW_SUPPORT |
| aryl_ether | other | all | 2 | 2 | 0.1470 | 0.9350 | 1.4400 (0.50) | LOW_SUPPORT |
| aryl_ether | other | clean | 14 | 12 | 0.2632 | 0.4125 | 0.5211 (0.83) | SUPPORTED |
| arylamine | E_HH_nonTest_train_pair | clean | 3 | 3 | 0.2023 | 0.5067 | 1.5950 (0.33) | LOW_SUPPORT |
| arylamine | E_T_train_endpoint | all | 4 | 3 | 0.1099 | 0.6200 | 0.8500 (0.67) | LOW_SUPPORT |
| arylamine | E_T_train_endpoint | clean | 2 | 2 | 0.2328 | 0.6400 | 0.5433 (1.00) | LOW_SUPPORT |
| arylamine | P_train | all | 10 | 7 | 0.2204 | 0.5443 | 0.6233 (1.00) | LOW_SUPPORT |
| arylamine | P_train | clean | 23 | 16 | 0.3192 | 0.4558 | 0.4173 (1.00) | SUPPORTED |
| arylamine | other | all | 2 | 1 | 0.0836 | 1.1000 | 0.4525 (1.00) | LOW_SUPPORT |
| arylamine | other | clean | 7 | 7 | 0.2592 | 0.7371 | 0.9427 (0.71) | LOW_SUPPORT |
| ester | E_HH_nonTest_train_pair | all | 1 | 1 | 0.1766 | 0.0400 | n/a (0.00) | LOW_SUPPORT |
| ester | E_HH_nonTest_train_pair | clean | 1 | 1 | 0.0160 | 0.5000 | n/a (0.00) | LOW_SUPPORT |
| ester | E_T_train_endpoint | clean | 7 | 1 | 0.1267 | 0.5643 | 0.2917 (1.00) | LOW_SUPPORT |
| ester | P_train | all | 2 | 1 | 0.4337 | 1.0150 | 1.9287 (1.00) | LOW_SUPPORT |
| ester | P_train | clean | 5 | 4 | 0.3838 | 1.0463 | 2.0112 (1.00) | LOW_SUPPORT |
| ester | other | all | 2 | 1 | 0.5829 | 1.4000 | n/a (0.00) | LOW_SUPPORT |
| ester | other | clean | 5 | 3 | 1.0534 | 2.5700 | 2.7733 (1.00) | LOW_SUPPORT |
| sulfonamide | E_HH_nonTest_train_pair | all | 2 | 2 | 0.4812 | 0.4150 | 0.4330 (1.00) | LOW_SUPPORT |
| sulfonamide | E_HH_nonTest_train_pair | clean | 1 | 1 | 0.0731 | 0.3100 | 0.2967 (1.00) | LOW_SUPPORT |
| sulfonamide | E_T_train_endpoint | all | 1 | 1 | 0.0545 | 0.2500 | 0.6980 (1.00) | LOW_SUPPORT |
| sulfonamide | P_train | all | 9 | 6 | 0.3654 | 0.5325 | 0.9488 (1.00) | LOW_SUPPORT |
| sulfonamide | P_train | clean | 16 | 7 | 0.2755 | 0.4360 | 0.7708 (1.00) | LOW_SUPPORT |
| sulfonamide | other | all | 4 | 3 | 0.2968 | 1.3233 | 1.8275 (0.33) | LOW_SUPPORT |
| sulfonamide | other | clean | 5 | 3 | 0.3657 | 0.4789 | 1.7433 (0.33) | LOW_SUPPORT |
