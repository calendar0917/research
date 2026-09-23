# PEC-v0 — Gate 2 (cheap internal task screen)

Verdict: **GATE2_PASS_BUY_SEED0**

Train 2000 / dev 500 official-train molecules (official valid/test not read).

| arm | params | best dev MAE | Top-5 soup dev MAE | chem-shuffle MAE | chem-shuffle degradation | neutral-dict MAE |
|---|---:|---:|---:|---:|---:|---:|
| C0_coarse | 94036 | 0.470864 | 0.466904 | 1.501883 | +1.031019 | — |
| CD_dense | 94049 | 0.476846 | 0.467108 | 1.235357 | +0.758511 | — |
| CK_sparse | 94049 | 0.465776 | 0.461811 | 1.301115 | +0.835338 | 2.175492 |
| CK_bag | 94049 | 0.500130 | 0.496982 | — | — | — |
| CK_shuffle | 94049 | 0.475933 | 0.473408 | — | — | — |

Frozen criteria:

* `chem_shuffle_material`: True
* `true_beats_bag`: True
* `true_beats_shuffle`: True
* `dict_not_worse_than_dense`: True
* `dictionary_alive`: True

`official_test_loaded = false`

Frozen thresholds (pre-registered before the run):

* `chem_shuffle_material`: degradation >= 0.02
* `true_beats_bag`: TRUE - BAG >= 0.003
* `true_beats_shuffle`: TRUE - SHUFFLE >= 0.003
* `dict_not_worse_than_dense`: CK - CD <= 0.002
* `dictionary_alive`: neutral-dictionary MAE must differ materially from TRUE

Train 2000 / dev 500, one seed (0), dictionaries refit on the 2000 train molecules (224.9 s).  Absolute MAE is NOT comparable to the strict-static S0 band (10 000 training molecules); this is a relative screen only.
