# REPORT — CSCL-v0 formal round (seeds {0,1})

Protocol `cscl-v0` (notes/v0_protocol.md + addendum). Primary metric: soup dev y-MAE (raw units).
Data: internal grouped fit 8000 / dev 2000 split of official-train 10000; official valid/test never loaded.

| arm | seed | soup dev MAE | best dev MAE | fit MAE | params | wall s | device |
|---|---|---|---|---|---|---|---|
| additive | 0 | 0.43915 | 0.44778 | 0.37420 | 10771 | 74 | cuda:0 |
| additive | 1 | 0.41183 | 0.41745 | 0.33453 | 10771 | 125 | cuda:0 |
| relational | 0 | 0.46525 | 0.47201 | 0.37618 | 20308 | 96 | cuda:0 |
| relational | 1 | 0.46941 | 0.47146 | 0.34475 | 20308 | 121 | cuda:0 |
| shuffled | 0 | 0.43358 | 0.43876 | 0.37866 | 20308 | 116 | cuda:0 |
| shuffled | 1 | 0.42857 | 0.43473 | 0.37126 | 20308 | 106 | cuda:0 |
| opaque | 0 | 0.40005 | 0.40045 | 0.25962 | 26977 | 81 | cuda:0 |
| opaque | 1 | 0.40864 | 0.41475 | 0.30217 | 26977 | 57 | cuda:0 |
| xgb | 0 | 0.47714 | nan | 0.24261 | -1 | 3 | cpu |
| xgb | 1 | 0.47484 | nan | 0.24008 | -1 | 3 | cpu |

Seed means (soup): additive 0.42549, relational 0.46733, shuffled 0.43108, opaque 0.40434, xgb 0.47599

## Paired differences vs B (relational), molecule bootstrap 95% CI

| comparison | seed | mean | CI95 |
|---|---|---|---|
| additive-relational_s0 | | -0.02423 | [-0.04235, -0.00474] |
| additive-relational_s1 | | -0.05401 | [-0.07824, -0.03184] |
| shuffled-relational_s0 | | -0.03355 | [-0.05249, -0.01389] |
| shuffled-relational_s1 | | -0.03691 | [-0.05893, -0.01534] |
| opaque-relational_s0 | | -0.07196 | [-0.09590, -0.04592] |
| opaque-relational_s1 | | -0.06282 | [-0.08609, -0.04072] |
| xgb-relational_s0 | | +0.00513 | [-0.02010, +0.03237] |
| xgb-relational_s1 | | +0.00339 | [-0.02391, +0.03066] |

## Attribution stability (real ZINC, best-epoch B models)

- alpha_pearson_across_seeds: 0.7257
- alpha_spearman_across_seeds: 0.6636
- alpha_sign_agreement: 0.7824
- gamma_std_s0: 0.1793
- gamma_std_s1: 0.1701
- gamma_mean_abs_s0: 0.1244
- gamma_mean_abs_s1: 0.1585
- n_common_types: 170

## Provenance

- cscl-v0-a-s0 run=cscl-v0-a-s0-20261010-181009-7cf8bdbf commit=6549c04dc520 pool=res-gpu1 state=completed exit=None cuda_visible=1
- cscl-v0-a-s1 run=cscl-v0-a-s1-20261010-181759-2c781f8e commit=6549c04dc520 pool=res-gpu1 state=completed exit=None cuda_visible=1
- cscl-v0-b-s0 run=cscl-v0-b-s0-20261010-180740-c3f0d5b1 commit=6549c04dc520 pool=res-gpu1 state=completed exit=None cuda_visible=1
- cscl-v0-b-s1 run=cscl-v0-b-s1-20261010-182054-2006227b commit=6549c04dc520 pool=res-gpu1 state=completed exit=None cuda_visible=1
- cscl-v0-c-s0 run=cscl-v0-c-s0-20261010-181225-aae82a23 commit=6549c04dc520 pool=res-gpu1 state=completed exit=None cuda_visible=1
- cscl-v0-c-s1 run=cscl-v0-c-s1-20261010-182348-ef2fb47a commit=6549c04dc520 pool=res-gpu1 state=completed exit=None cuda_visible=1
- cscl-v0-d-s0 run=cscl-v0-d-s0-20261010-181529-6b06af9a commit=6549c04dc520 pool=res-gpu1 state=completed exit=None cuda_visible=1
- cscl-v0-d-s1 run=cscl-v0-d-s1-20261010-182638-76c60caf commit=6549c04dc520 pool=res-gpu1 state=completed exit=None cuda_visible=1

## Identifiability audit (real ZINC, best-epoch B; v0_protocol §6.4)

Soft constraints at lambda=0.01 are **only partially binding on real data** (in contrast to the synthetic task, where type means landed in alpha):
- seed s0: per-type mean |delta_bar_t| = 0.0684 (max 0.4682), gamma mean = +0.0495, |gamma| mean = 0.1232
- seed s1: per-type mean |delta_bar_t| = 0.0597 (max 0.4520), gamma mean = +0.1066, |gamma| mean = 0.1580

## Synthetic ground-truth check (structures = RINGCHAIN-v0, y = planted effects)

- unary direction agreement 0.75, alpha-theta spearman 0.875 (best-epoch model, lambda=0.01)
- real-pair |gamma| 0.1337 vs trap-pair |gamma| 0.0694 (real pairs were selected as the *highest co-occurrence* pairs — the hard case)
- known limitation: no novel-combination transfer cases (all 9 real pairs already co-occur at fit time); gamma magnitudes attenuated ~2.4x vs planted theta
- soup vs best-epoch: see notes/v0_protocol_addendum.md (soup destroys attribution geometry)
