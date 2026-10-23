# Main table — B / G / S, new holdout

`delta` = corrected MAE − base MAE (negative favours the correction).
The **C_dev-selected** head is the primary arm; every one of the 24 heads returned
`steps_used = 0`, i.e. the selection rule chooses *no correction*, so the selected
G and S columns equal B by construction. The `endpoint` columns are the descriptive
no-selection read at 1,500 steps (the fitted head actually applied).

| seed | package | n | n gated | B MAE | G sel | S sel | G end | S end | G−B end | S−B end | S−G end | S improves end | |h_S| end |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Q_A | 361 | 361 | 0.1975 | 0.1975 | 0.1975 | 0.2386 | 0.2336 | +0.0411 | +0.0362 | -0.0049 | 0.35 | 0.1005 |
| 0 | Q_B | 310 | 310 | 0.1291 | 0.1291 | 0.1291 | 0.1999 | 0.2514 | +0.0708 | +0.1223 | +0.0514 | 0.31 | 0.1900 |
| 0 | Q_bg | 400 | 114 | 0.1331 | 0.1331 | 0.1331 | 0.1449 | 0.1466 | +0.0118 | +0.0134 | +0.0017 | 0.12 | 0.0408 |
| 1 | Q_A | 361 | 361 | 0.1951 | 0.1951 | 0.1951 | 0.2440 | 0.2369 | +0.0489 | +0.0418 | -0.0071 | 0.32 | 0.1078 |
| 1 | Q_B | 310 | 310 | 0.1329 | 0.1329 | 0.1329 | 0.1840 | 0.2442 | +0.0511 | +0.1113 | +0.0602 | 0.32 | 0.1847 |
| 1 | Q_bg | 400 | 114 | 0.1345 | 0.1345 | 0.1345 | 0.1516 | 0.1495 | +0.0171 | +0.0151 | -0.0020 | 0.09 | 0.0391 |

## Bootstrap intervals on the endpoint deltas (per head init)

| seed | package | G−B [95% mol] | S−B [95% mol] | S−B [95% key-cluster] | clusters |
|---|---|---|---|---|---:|
| 0 | Q_A | [+0.0287, +0.0500] / [+0.0330, +0.0556] / [+0.0285, +0.0516] | [+0.0263, +0.0488] / [+0.0255, +0.0479] / [+0.0237, +0.0452] | [+0.0243, +0.0533] / [+0.0228, +0.0515] / [+0.0218, +0.0484] | 35 |
| 0 | Q_B | [+0.0571, +0.0904] / [+0.0551, +0.0955] / [+0.0516, +0.0811] | [+0.0910, +0.1700] / [+0.0795, +0.1304] / [+0.1016, +0.1666] | [+0.0424, +0.2486] / [+0.0540, +0.1673] / [+0.0574, +0.2283] | 37 |
| 0 | Q_bg | [+0.0034, +0.0190] / [+0.0043, +0.0253] / [+0.0042, +0.0179] | [+0.0051, +0.0238] / [+0.0029, +0.0226] / [+0.0046, +0.0240] | - | - |
| 1 | Q_A | [+0.0310, +0.0564] / [+0.0412, +0.0700] / [+0.0347, +0.0608] | [+0.0282, +0.0520] / [+0.0253, +0.0503] / [+0.0325, +0.0619] | [+0.0295, +0.0538] / [+0.0232, +0.0535] / [+0.0215, +0.0746] | 35 |
| 1 | Q_B | [+0.0327, +0.0613] / [+0.0401, +0.0757] / [+0.0357, +0.0666] | [+0.0655, +0.1331] / [+0.0885, +0.1527] / [+0.0878, +0.1461] | [+0.0216, +0.1990] / [+0.0533, +0.2077] / [+0.0409, +0.2048] | 37 |
| 1 | Q_bg | [+0.0076, +0.0235] / [+0.0106, +0.0286] / [+0.0091, +0.0270] | [+0.0058, +0.0257] / [+0.0091, +0.0260] / [+0.0025, +0.0232] | - | - |

## Head fitting and selection

Raw (`dev_mae`) and pre-registered (`dev_mae_trim5`) curves are both shown. Every listed arm (the C_dev-selected wd) has its minimum at step 0 under **both**, so the null is not a trim5 artefact. The other wd only kept its trim5 level (also equal to the base), not its raw curve.
| seed | head | wd | C_fit gated MAE | C_dev base raw | C_dev best raw (post-hoc) | C_dev base trim5 | C_dev best trim5 | selected step |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | G | 0.0 | 0.1202 | 0.1140 | 0.1140 | 0.0919 | 0.0919 | [0, 0, 0] |
| 0 | S | 0.0 | 0.1202 | 0.1140 | 0.1140 | 0.0919 | 0.0919 | [0, 0, 0] |
| 1 | G | 0.0 | 0.1230 | 0.1130 | 0.1130 | 0.0957 | 0.0957 | [0, 0, 0] |
| 1 | S | 0.0 | 0.1230 | 0.1130 | 0.1130 | 0.0957 | 0.0957 | [0, 0, 0] |

## Coarse baselines (no local structure at all)

| seed | package | constant (mean) Δ | constant (median) Δ | linear in f_base Δ |
|---|---|---:|---:|---:|
| 0 | Q_A | +0.0023 | +0.0001 | +0.0029 |
| 0 | Q_B | +0.0068 | -0.0005 | +0.0113 |
| 0 | Q_bg | -0.0001 | +0.0001 | +0.0007 |
| 1 | Q_A | +0.0029 | -0.0003 | +0.0048 |
| 1 | Q_B | +0.0076 | +0.0011 | +0.0158 |
| 1 | Q_bg | +0.0004 | +0.0000 | +0.0015 |

## Within-key diagnostic (molecule-disjoint 80/20 inside every C key family)

Held-out molecules share the exact radius-2 keys with their training molecules.
The **selected** column is the pre-registered rule (trim5). The post-hoc raw
minimum is shown for completeness and is **not** a selection rule.

| seed | head | base hold MAE | selected step | selected Δ | raw best step (post-hoc) | raw best Δ (post-hoc) |
|---|---|---:|---:|---:|---:|---:|
| 0 | G | 0.0966 | 0 | +0.0000 | 0 | +0.0000 |
| 0 | S | 0.0966 | 0 | +0.0000 | 50 | -0.0102 |
| 1 | G | 0.1002 | 0 | +0.0000 | 0 | +0.0000 |
| 1 | S | 0.1002 | 0 | +0.0000 | 0 | +0.0000 |

Oracle per-package constant (only bounds whole-package translations):

| cell | base MAE | best constant | gain |
|---|---:|---:|---:|
| Q_A_s0 | 0.1975 | +0.0005 | 0.000001 |
| Q_B_s0 | 0.1291 | +0.0145 | 0.001050 |
| Q_bg_s0 | 0.1331 | -0.0030 | 0.000078 |
| Q_A_s1 | 0.1951 | -0.0085 | 0.000279 |
| Q_B_s1 | 0.1329 | +0.0125 | 0.000789 |
| Q_bg_s1 | 0.1345 | -0.0025 | 0.000031 |

## Old supervision-support package (exploration, `sup_minus` epoch-600 base)

| seed | package | n | base MAE | base MAE gated | S selected Δ | G endpoint Δ | S endpoint Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Q_old | 225 | 0.1012 | 0.1012 | +0.0000 | +0.0909 | +0.1157 |
| 0 | Q_bg | 650 | 0.0983 | 0.1981 | +0.0000 | +0.0297 | +0.0337 |
| 0 | M | 1000 | 0.0810 | 0.1411 | +0.0000 | +0.0273 | +0.0360 |
| 1 | Q_old | 225 | 0.0995 | 0.0995 | +0.0000 | +0.0722 | +0.0727 |
| 1 | Q_bg | 650 | 0.1027 | 0.2058 | +0.0000 | +0.0283 | +0.0299 |
| 1 | M | 1000 | 0.0783 | 0.1298 | +0.0000 | +0.0291 | +0.0395 |
