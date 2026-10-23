# Stage-1 decision note (seed 0)

Frozen protocol: `ssrc-zinc-native-relation-v1`; code commit `da86150a9e33` (code identical to the frozen `85b849734587`; only the pre-GPU manifest changed).

## Seed-0 completed runs

| run | best valid | best epoch | final valid | W20 | T100 | wall h |
|---|---:|---:|---:|---:|---:|---:|
| P_s0 | 0.078611 | 502 | 0.079085 | 0.079046 | 0.079079 | 0.307 |
| R_s0 | 0.084991 | 395 | 0.085834 | 0.085805 | 0.085833 | 0.582 |
| N_s0 | 0.084792 | 392 | 0.085819 | 0.085794 | 0.085879 | 0.513 |

## Best-checkpoint groups (old fixed masks)

| group | P_s0 | R_s0 | N_s0 | R−P gain | N−P gain | R−N gain |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.078611 | 0.084991 | 0.084792 | -0.006380 | -0.006181 | -0.000199 |
| seen_ordinary | 0.047160 | 0.049671 | 0.050742 | -0.002511 | -0.003582 | +0.001071 |
| unseen_ordinary | 0.134870 | 0.140910 | 0.140237 | -0.006039 | -0.005367 | -0.000673 |
| seen_tail | 0.192739 | 0.240879 | 0.246982 | -0.048140 | -0.054243 | +0.006103 |
| unseen_tail | 0.279424 | 0.331761 | 0.291536 | -0.052337 | -0.012112 | -0.040225 |
| ordinary_935 | 0.068267 | 0.071627 | 0.072278 | -0.003360 | -0.004011 | +0.000651 |
| tail_65 | 0.227413 | 0.277232 | 0.264804 | -0.049819 | -0.037391 | -0.012428 |
| ring_rich_213 | 0.116390 | 0.118578 | 0.115646 | -0.002188 | +0.000744 | -0.002932 |
| ring_rich_unseen_47 | 0.311235 | 0.343410 | 0.318835 | -0.032175 | -0.007600 | -0.024575 |

## Read against the pre-registered rules

- **H1 (R vs P, overall):** seed-0 R−P = **−0.006380** (R worse). The rule requires both seeds positive and mean ≥ +0.001; seed 0 alone already falsifies it. W20/T100 also move the wrong way (R−P sustained ≈ −0.00676/−0.00679).
- **H1 (R vs P, 225 unseen ordinary):** seed-0 gain **−0.006039** (R worse). The +0.005 group gate is not met.
- **H2 (R vs N):** seed-0 gain **−0.000199** (R marginally worse than the parameter-identical permuted control). Real relation labels show no advantage over permuted labels; both R and N are ≈0.0063 worse than P.
- **Tail diagnostics:** R is much worse on both tail strata (−0.048 seen tail, −0.052 unseen tail), so the overall cost is not confined to a negligible subgroup. Row 172 remains in every headline number.
- **Interpretation boundary:** this is a same-round, same-recipe, same-data contrast. The branch adds 7,185 parameters (420,690 total, <500k cap). The correct conclusion on seed 0 is that one native relation round did **not** improve the unseen local environments or overall MAE; it degraded both. This does not by itself identify which part of the branch caused the cost (optimisation interference vs message content), because R and N share the branch.

## Budget and continuation

- GPU hours used through seed 0: P 0.307 + R 0.582 + N 0.513 + pre-GPU 0.019 = **1.421 h** of the 5.0 h cap.
- Seed 1 is the frozen default completion item and tests reproducibility of the seed-0 negative; it is not a tuning response. Launching P_s1/R_s1 in parallel and N_s1 on the first freed GPU.
- No hyperparameter, relation-definition, epoch, or selection change is permitted between seeds. If seed 1 reproduces the negative, report and stop; do not search for a configuration that turns it positive.

## Seed-1 launch decision

**GO for the default seed-1 completion (P/R/N), with unchanged code and recipe.**
