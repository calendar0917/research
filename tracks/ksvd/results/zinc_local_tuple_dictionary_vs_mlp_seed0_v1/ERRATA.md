# ERRATA — wording and lineage corrections for `zinc_local_tuple_dictionary_vs_mlp_seed0_v1`

Written before the new M_J dev score; no old report or artifact is modified.

## E1. Actual previous-round training recipe (source/meta, not prose)

`J_meta.json` and the committed source show: training seed **0**, batch **128**, Adam lr 1e-3,
coupled WD 1e-5, global clip 5.0, 240 epochs, **15,120** steps, soup epochs **236–240**, IHT
starting from an **all-zero code** with **10** steps. The earlier `METHOD_CONTRACT.md` text
("batch 32", "seed `SEED + TRAIN_SHUFFLE_OFFSET`", "projected initial code") does not describe the
executed run. This round copies the executed recipe.

## E2. `20261004` is not the training seed

`20261004` is the fold seed (`build_fold`), the dictionary frame seed (`FRAME_SEED`), the
`kappa_sample` seed (`KAPPA_SEED`) and the bootstrap seed (`BOOT_SEED`) inherited from earlier
rounds. The training seed is 0. Report statements that call `20261004` a training seed are wrong
and are not propagated.

## E3. Old D_g / M_g wording

The posterior-bridge dictionary-vs-MLP round did not produce a separable conclusion. This round
does not replace that bridge; it replaces only the local tuple encoder before fusion layer 1, so
no D_g/M_g statement should be read into the new numbers.

## E4. Zero ablation wording

"Zeroing the local path degrades the model strongly" is a dependence statement, not an information
share. The intervention produces a large systematic offset and destroys co-adapted downstream
weights; the added fit-mean replacement is reported alongside precisely because it is a weaker,
mean-preserving intervention.

## E5. Mixed-12000 lineage

No historical `stage_refine` constant and no `label_effective_cycle` / `train_cycle_audit_label`
column is reopened, traced or refit. The frozen `fit_only_targets.npz` (`e2adf5f2…`) is the only
target source; the previous round's `ACCESS_LINEAGE_RESOLVED` audit is reused read-only.

## E6. Four-grid decomposition is descriptive

The frozen calibrated J/I intervention grid from the previous round

| state / operator | overall cal | G0 cal |
|---|---|---|
| J / J | 0.1019071074 | 0.1004162377 |
| J / I | 0.1019913913 | 0.1005100907 |
| I / J | 0.1045416120 | 0.1038224926 |
| I / I | 0.1046819255 | 0.1039915517 |

decomposes as `G = O + W` with point/CI-sized `O ≈ 0.00011 / 0.00013` and
`W ≈ 0.00266 / 0.00344` (overall/G0) — about 96% of the finite difference sits in `W`, which
mixes trained-weight state and state-specific calibration. This is one descriptive split, **not**
a unique causal decomposition: it does not prove that the real correspondence is worth only
~0.0001, and it does not establish that the correspondence inductive bias held during training.
