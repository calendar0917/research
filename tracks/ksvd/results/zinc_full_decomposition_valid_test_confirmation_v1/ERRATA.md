# ERRATA — corrections to the prior narrative (c11342e / 87894a2)

Scope: this file corrects *narrative* errors in the earlier reports. It does
**not** rewrite any historical CSV/NPZ artifact and does **not** retrain any
historical model. All numbers below are re-read from the released files; the
current round's own results are reported separately.

1. **Severe-9 tail description was too strong.** Of the 9 reused inner-dev
   severe rows, 7 are `k=-2`. Five of those 7 have a cycle-head prediction error
   `|c - q| < 0.1` (`global train idx 1238, 4344, 4485, 5093, 7659`); only
   `2052` and `7507` are genuine `k=-2` misses. The earlier phrasing that "7/9
   `k=-2` rows were solved" is wrong: it is 5/7, and it is not 7/9 of the severe
   set.

2. **`k=-12`, `global train idx 2210`, is not a typical total failure.** It is
   substantially recovered: true `c ≈ -41.6205`, learned `q ≈ -40.4392` (seed 0)
   and `≈ -40.2663` (seed 1), i.e. `|c-q| ≈ 1.2-1.35`. It must not be used as the
   canonical "completely unlearnable" row.

3. **The residual `P`-vs-`O` error increment is concentrated, and those rows are
   old inner-dev rows.** About 95% of the extra error of `P` over the frozen
   oracle `O` is carried by `global train idx 2052, 5050, 7507` (two `k=-2`, one
   `k=-5`). Because the old 2,000-row dev is a *reused* train-side diagnostic
   set, these indices do **not** map to the same positions in the new
   official-valid / official-test splits, and must not be used to explain the new
   held-out numbers.

4. **`K` grouping narrative vs `group_table.csv`.** The overall constant-offset
   control values are `K_cal MAE = 0.2638855532` (seed 0) and `0.2599724312`
   (seed 1), matching `group_table.csv` `overall` rows. Any earlier report text
   that quoted different group counts / totals for `K` is superseded by the CSV.

5. **Signed-error convention was mixed.** Earlier tables used different sign
   conventions. This round fixes one definition everywhere:
   `residual = y − prediction` (and `e_g = g − h_raw`, `e_c = c − q`,
   `y − P_cal = e_g + e_c − b_P`).

These corrections limit the *mechanism attribution* of the old inner-dev round
(all extreme-tail rows are not unlearnable; the extreme tail is not fully
solved either) but they do not overturn the old main gate. The previous round's
gate was defined on the 2,000-row reused diagnostic dev and remains a historical
result; the present round re-tests the fixed method on the full 10k and the
official held-out splits.