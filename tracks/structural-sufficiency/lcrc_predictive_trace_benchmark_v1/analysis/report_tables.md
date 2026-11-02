# 自动生成的表格（`code/ptb_report.py`）

生成时间 2026-11-02T22:15:04Z，commit `02f230636fc4e3a287b64f6445bf1429ca773ba2`。

## A. 分子任务 Test 主指标（每格 3 seeds）

| dataset | metric | A0 mean±sd | C1 mean±sd | Δ C1−A0 | Δ 95% CI (rows) | n | status |
|---|---|---|---|---|---|---|---|
| ESOL | rmse | 0.5645 ± 0.0299 | 0.5534 ± 0.0299 | -0.0111 | [-0.0507, 0.0248] | 3/3 | NEW_EXPOSED_TEST_EXPLORATORY |
| FreeSolv | rmse | 1.0922 ± 0.0606 | 1.0711 ± 0.0732 | -0.0211 | [-0.0921, 0.1017] | 3/3 | NEW_EXPOSED_TEST_EXPLORATORY |
| BACE | roc_auc | 0.8168 ± 0.0066 | 0.8263 ± 0.0050 | +0.0095 | [-0.0259, 0.0453] | 3/3 | NEW_EXPOSED_TEST_EXPLORATORY |
| BBBP | roc_auc | 0.7041 ± 0.0083 | 0.7060 ± 0.0142 | +0.0019 | [-0.0169, 0.0283] | 3/3 | NEW_EXPOSED_TEST_EXPLORATORY |
| Tox21 | macro_roc_auc | 0.8353 ± 0.0104 | 0.8374 ± 0.0087 | +0.0022 | [-0.0023, 0.0153] | 3/3 | NEW_EXPOSED_TEST_EXPLORATORY |
| ogbg-molhiv | roc_auc | 0.7716 ± 0.0055 | 0.7611 ± 0.0135 | -0.0105 | [-0.0511, 0.0093] | 3/3 | NEW_EXPOSED_TEST_EXPLORATORY |


**Lipophilicity — read-only frozen history (no retraining)**


| unit | metric | A0 | C1 | Δ C1−A0 | label |
|---|---|---|---|---|---|
| repeat 122 | valid RMSE | 0.7066 | 0.7053 | -0.0013 | frozen history |
| repeat 122 | test RMSE | 0.5770 | 0.5784 | +0.0014 | EXPOSED_TEST_EXPLORATORY |
| repeat 123 | valid RMSE | 0.7610 | 0.7591 | -0.0019 | frozen history |
| repeat 123 | test RMSE | 0.5827 | 0.5667 | -0.0160 | EXPOSED_TEST_EXPLORATORY |
| repeat 124 | valid RMSE | 0.6373 | 0.6367 | -0.0006 | frozen history |
| repeat 124 | test RMSE | 0.5391 | 0.5420 | +0.0030 | EXPOSED_TEST_EXPLORATORY |

## B. 36 个分子单元逐条

| dataset | model | seed | best epoch | valid | test | train−valid | GPU-h | status |
|---|---|---|---|---|---|---|---|---|
| ESOL | A0 | 0 | 187 | 0.5613 | 0.5442 | -0.1664 | 0.022 | COMPLETE |
| ESOL | A0 | 1 | 116 | 0.5836 | 0.5988 | -0.1035 | 0.022 | COMPLETE |
| ESOL | A0 | 2 | 221 | 0.5355 | 0.5504 | -0.1549 | 0.021 | COMPLETE |
| ESOL | C1 | 0 | 136 | 0.5564 | 0.5320 | -0.1118 | 0.022 | COMPLETE |
| ESOL | C1 | 1 | 116 | 0.5816 | 0.5875 | -0.1059 | 0.022 | COMPLETE |
| ESOL | C1 | 2 | 202 | 0.5381 | 0.5406 | -0.1462 | 0.022 | COMPLETE |
| FreeSolv | A0 | 0 | 230 | 0.7929 | 1.0951 | -0.2989 | 0.011 | COMPLETE |
| FreeSolv | A0 | 1 | 272 | 0.7800 | 1.0302 | -0.2787 | 0.011 | COMPLETE |
| FreeSolv | A0 | 2 | 204 | 0.7491 | 1.1513 | -0.2526 | 0.011 | COMPLETE |
| FreeSolv | C1 | 0 | 161 | 0.7795 | 1.0835 | -0.1909 | 0.011 | COMPLETE |
| FreeSolv | C1 | 1 | 179 | 0.7673 | 0.9924 | -0.2509 | 0.011 | COMPLETE |
| FreeSolv | C1 | 2 | 204 | 0.7410 | 1.1373 | -0.2452 | 0.011 | COMPLETE |
| BACE | A0 | 0 | 68 | 0.7823 | 0.8107 | 0.1688 | 0.039 | COMPLETE |
| BACE | A0 | 1 | 117 | 0.7640 | 0.8237 | 0.2088 | 0.039 | COMPLETE |
| BACE | A0 | 2 | 242 | 0.7838 | 0.8161 | 0.2060 | 0.039 | COMPLETE |
| BACE | C1 | 0 | 98 | 0.7799 | 0.8313 | 0.1842 | 0.037 | COMPLETE |
| BACE | C1 | 1 | 73 | 0.7639 | 0.8214 | 0.1903 | 0.038 | COMPLETE |
| BACE | C1 | 2 | 251 | 0.7788 | 0.8263 | 0.2110 | 0.038 | COMPLETE |
| BBBP | A0 | 0 | 24 | 0.9733 | 0.7111 | -0.0227 | 0.044 | COMPLETE |
| BBBP | A0 | 1 | 24 | 0.9766 | 0.6949 | -0.0287 | 0.044 | COMPLETE |
| BBBP | A0 | 2 | 28 | 0.9731 | 0.7064 | -0.0171 | 0.044 | COMPLETE |
| BBBP | C1 | 0 | 39 | 0.9725 | 0.7180 | -0.0022 | 0.043 | COMPLETE |
| BBBP | C1 | 1 | 30 | 0.9768 | 0.6902 | -0.0263 | 0.044 | COMPLETE |
| BBBP | C1 | 2 | 28 | 0.9739 | 0.7097 | -0.0168 | 0.044 | COMPLETE |
| Tox21 | A0 | 0 | 73 | 0.8609 | 0.8290 | 0.0887 | 0.161 | COMPLETE |
| Tox21 | A0 | 1 | 47 | 0.8708 | 0.8473 | 0.0444 | 0.161 | COMPLETE |
| Tox21 | A0 | 2 | 64 | 0.8648 | 0.8295 | 0.0739 | 0.159 | COMPLETE |
| Tox21 | C1 | 0 | 56 | 0.8614 | 0.8356 | 0.0644 | 0.159 | COMPLETE |
| Tox21 | C1 | 1 | 47 | 0.8696 | 0.8470 | 0.0448 | 0.157 | COMPLETE |
| Tox21 | C1 | 2 | 64 | 0.8643 | 0.8298 | 0.0742 | 0.159 | COMPLETE |
| ogbg-molhiv | A0 | 0 | 53 | 0.8314 | 0.7653 | 0.1566 | 0.245 | COMPLETE |
| ogbg-molhiv | A0 | 1 | 32 | 0.8357 | 0.7736 | 0.1159 | 0.244 | COMPLETE |
| ogbg-molhiv | A0 | 2 | 29 | 0.8376 | 0.7759 | 0.1110 | 0.250 | COMPLETE |
| ogbg-molhiv | C1 | 0 | 53 | 0.8379 | 0.7586 | 0.1497 | 0.242 | COMPLETE |
| ogbg-molhiv | C1 | 1 | 32 | 0.8377 | 0.7757 | 0.1129 | 0.241 | COMPLETE |
| ogbg-molhiv | C1 | 2 | 23 | 0.8344 | 0.7491 | 0.0831 | 0.243 | COMPLETE |

## C. TU 结果

| dataset | model | test mean±sd | min/max | n units | best epoch | label |
|---|---|---|---|---|---|---|
| NCI1 | A0 | 0.8296 ± 0.0067 | 0.8175/0.8394 | 10 | 59.7 | NEW_INTERNAL_CV |
| NCI1 | C1 | 0.8241 ± 0.0081 | 0.8127/0.8382 | 10 | 59.5 | NEW_INTERNAL_CV |
| **NCI1 Δ** | C1−A0 | -0.0055 | [-0.0113, +0.0018] | — folds | outer-fold bootstrap | NEW_INTERNAL_CV |
| PROTEINS | A0 | 0.7592 ± 0.0298 | 0.7297/0.8117 | 10 | 34.4 | NEW_INTERNAL_CV |
| PROTEINS | C1 | 0.7533 ± 0.0281 | 0.7207/0.8027 | 10 | 43.1 | NEW_INTERNAL_CV |
| **PROTEINS Δ** | C1−A0 | -0.0059 | [-0.0180, +0.0031] | — folds | outer-fold bootstrap | NEW_INTERNAL_CV |
| IMDB-BINARY | A0 | 0.7370 ± 0.0276 | 0.7000/0.7850 | 10 | 44.0 | NEW_INTERNAL_CV |
| IMDB-BINARY | C1 | 0.7260 ± 0.0322 | 0.6850/0.7800 | 10 | 39.4 | NEW_INTERNAL_CV |
| **IMDB-BINARY Δ** | C1−A0 | -0.0110 | [-0.0310, +0.0075] | — folds | outer-fold bootstrap | NEW_INTERNAL_CV |
| IMDB-MULTI | A0 | 0.4810 ± 0.0262 | 0.4533/0.5433 | 10 | 37.1 | NEW_INTERNAL_CV |
| IMDB-MULTI | C1 | 0.4843 ± 0.0288 | 0.4500/0.5467 | 10 | 29.7 | NEW_INTERNAL_CV |
| **IMDB-MULTI Δ** | C1−A0 | +0.0033 | [-0.0073, +0.0130] | — folds | outer-fold bootstrap | NEW_INTERNAL_CV |

## D. 效率

| dataset | model | trainable params | param MB | epoch s | graphs/s | CUDA peak MB | cache MB | cache build s | long-tail ratio |
|---|---|---|---|---|---|---|---|---|---|
| ESOL | A0 | 247106 | 0.943 | 0.368 | 16396.7 | 20.3 | 4.8 | 0.04 | 1.34 |
| ESOL | C1 | 228674 | 0.872 | 0.188 | 19229.9 | 19.7 | 4.8 | 0.04 | 1.34 |
| FreeSolv | A0 | 247106 | 0.943 | 0.101 | 16341.7 | 14.2 | 1.7 | 0.02 | 1.30 |
| FreeSolv | C1 | 228674 | 0.872 | 0.092 | 15832.5 | 13.9 | 1.7 | 0.02 | 1.31 |
| BACE | A0 | 247106 | 0.943 | 0.649 | 8333.9 | 47.6 | 16.1 | 0.09 | 1.23 |
| BACE | C1 | 228674 | 0.872 | 0.377 | 8444.5 | 45.9 | 16.1 | 0.09 | 1.23 |
| BBBP | A0 | 247106 | 0.943 | 0.436 | 9627.0 | 48.2 | 14.8 | 0.08 | 1.99 |
| BBBP | C1 | 228674 | 0.872 | 0.428 | 9701.4 | 46.4 | 14.8 | 0.08 | 2.03 |
| Tox21 | A0 | 247106 | 0.943 | 1.597 | 15079.0 | 27.6 | 46.0 | 0.33 | 2.61 |
| Tox21 | C1 | 228674 | 0.872 | 1.582 | 15186.8 | 26.7 | 46.0 | 0.33 | 2.58 |
| ogbg-molhiv | A0 | 247106 | 0.943 | 8.771 | 6751.6 | 46.0 | 328.2 | 2.96 | 1.94 |
| ogbg-molhiv | C1 | 228674 | 0.872 | 8.852 | 10751.4 | 43.7 | 328.2 | 2.96 | 2.85 |

## E. 正确性门禁

| gate | result | detail |
|---|---|---|
| ? | FAIL | A0 247106 / C1 228674 / delta 18432 (single and Tox21) |
| ? | FAIL | A0 re-implementation reproduces the frozen upstream Lipophilicity seed-0 Validation predictions |
| ? | FAIL | dropout runs once per forward on the shared h and is the identity in eval |
| ? | FAIL | atom renumbering leaves the prediction unchanged |
| ? | FAIL | batch merge/split and graph order do not change per-graph predictions |
| ? | FAIL | S1/S2 records match independent brute-force enumeration |
| ? | FAIL | every trainable parameter receives a non-zero gradient |
| ? | FAIL | standardisation constants come from the Train part only; the trainer cannot read Test |
| ? | FAIL | saved best checkpoint reproduces the Validation metric |
| ? | FAIL | node renumbering leaves the feature multiset, global vector and logits unchanged |
| ? | FAIL | batch split/merge consistency for FULL and GIN; 2-class and 3-class logits have C columns |
| ? | FAIL | FULL/C1 share every init tensor; C1 deletes exactly the b_mean block |
| ? | FAIL |  |

## F. GPU 账本

| kind | units | GPU-h | mean GPU-h |
|---|---|---|---|
| local_edit | 6 | 0.399 | 0.067 |
| molecule | 36 | 3.172 | 0.088 |
| tu | 86 | 5.520 | 0.064 |
| **total** | 128 | **9.091** | — |
