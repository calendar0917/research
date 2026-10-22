# Fixed four-group valid errors (710 seen+ordinary / 225 unseen+ordinary / 39 seen+tail / 26 unseen+tail)

| run | all | seen_ordinary | unseen_ordinary | seen_tail | unseen_tail | drop row172 | max single-molecule share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E_s0 | 0.08155 | 0.04989 | 0.14082 | 0.19104 | 0.26883 | 0.07922 | 3.35% (row 917) |
| E_s1 | 0.07924 | 0.04689 | 0.13228 | 0.22191 | 0.28988 | 0.07391 | 6.82% (row 172) |
| R_s0 | 0.08021 | 0.04828 | 0.13942 | 0.17631 | 0.29552 | 0.07688 | 4.25% (row 172) |
| R_s0_k04 | 0.07853 | 0.04661 | 0.13301 | 0.18820 | 0.31433 | 0.07463 | 5.06% (row 172) |
| R_s1 | 0.07775 | 0.04798 | 0.13852 | 0.15065 | 0.25542 | 0.07416 | 4.71% (row 172) |
| R_s1_k04 | 0.07880 | 0.04595 | 0.13479 | 0.23089 | 0.26322 | 0.07274 | 7.79% (row 172) |
| S_s0 | 0.08149 | 0.04802 | 0.14164 | 0.20236 | 0.29371 | 0.07713 | 5.44% (row 172) |
| S_s0_k04 | 0.08036 | 0.04825 | 0.13358 | 0.18989 | 0.33237 | 0.07652 | 4.88% (row 172) |
| S_s1 | 0.07568 | 0.04930 | 0.13312 | 0.09899 | 0.26400 | 0.07497 | 3.03% (row 753) |

The last column is the largest single-molecule share of total absolute error.

Top-10 molecule contributions per run (row, group, y, |AE|):

* **E_s0**: 917(unseen_ordinary, y=-1.00, 2.730), 172(seen_tail, y=-20.34, 2.402), 753(unseen_tail, y=-9.19, 1.781), 935(seen_tail, y=-9.45, 1.322), 659(unseen_ordinary, y=-0.06, 1.264)
* **E_s1**: 172(seen_tail, y=-20.34, 5.402), 753(unseen_tail, y=-9.19, 2.778), 917(unseen_ordinary, y=-1.00, 2.116), 260(unseen_ordinary, y=-1.85, 0.876), 447(unseen_ordinary, y=-2.16, 0.802)
* **R_s0**: 172(seen_tail, y=-20.34, 3.405), 753(unseen_tail, y=-9.19, 2.782), 917(unseen_ordinary, y=-1.00, 2.023), 659(unseen_ordinary, y=-0.06, 1.216), 31(unseen_ordinary, y=-1.63, 1.032)
* **R_s0_k04**: 172(seen_tail, y=-20.34, 3.974), 753(unseen_tail, y=-9.19, 3.149), 917(unseen_ordinary, y=-1.00, 2.027), 659(unseen_ordinary, y=-0.06, 1.226), 260(unseen_ordinary, y=-1.85, 1.030)
* **R_s1**: 172(seen_tail, y=-20.34, 3.664), 917(unseen_ordinary, y=-1.00, 2.502), 753(unseen_tail, y=-9.19, 2.273), 616(unseen_ordinary, y=0.24, 0.908), 659(unseen_ordinary, y=-0.06, 0.883)
* **R_s1_k04**: 172(seen_tail, y=-20.34, 6.137), 753(unseen_tail, y=-9.19, 2.724), 917(unseen_ordinary, y=-1.00, 2.311), 659(unseen_ordinary, y=-0.06, 1.014), 447(unseen_ordinary, y=-2.16, 0.816)
* **S_s0**: 172(seen_tail, y=-20.34, 4.435), 753(unseen_tail, y=-9.19, 2.845), 917(unseen_ordinary, y=-1.00, 2.324), 659(unseen_ordinary, y=-0.06, 1.412), 260(unseen_ordinary, y=-1.85, 1.106)
* **S_s0_k04**: 172(seen_tail, y=-20.34, 3.918), 753(unseen_tail, y=-9.19, 3.538), 917(unseen_ordinary, y=-1.00, 2.009), 659(unseen_ordinary, y=-0.06, 1.246), 260(unseen_ordinary, y=-1.85, 1.106)
* **S_s1**: 753(unseen_tail, y=-9.19, 2.293), 917(unseen_ordinary, y=-1.00, 2.159), 659(unseen_ordinary, y=-0.06, 0.938), 172(seen_tail, y=-20.34, 0.786), 260(unseen_ordinary, y=-1.85, 0.786)

Row-172 sensitivity is kept in every table; no number is reported without it.
