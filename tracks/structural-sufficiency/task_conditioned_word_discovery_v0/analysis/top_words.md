# Selected structural words (D_TASK)

Decoded from `analysis/selected_words_D0.csv` / `_D1.csv`.
Full field indices follow the official OGB atom/bond feature order.
`*` marks the rooted atom; edges are `slot-slot:bondtype`.

## D0 top 20 by frozen score

| # | size/gran | df | scaffold df | score | fold corr | pattern |
|---:|---|---:|---:|---:|---|---|
| 0 | 3 FULL | 28 | 15 | 0.1692 | +0.447, +0.205, +0.169 | `*C(deg4,H2,sp3,ring) C(deg4,H1,sp3,ring) N(deg3,H0,sp2,ring) | 0-1:single, 1-2:single` |
| 1 | 4 FULL | 88 | 32 | 0.1626 | +0.265, +0.165, +0.163 | `*C(deg3,H0,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) N(deg2,H0,sp2,arom,ring) N(deg3,H0,sp2,ring) | 0-2:aromatic, 0-3:single, 1-2:aromatic` |
| 2 | 3 FULL | 28 | 15 | 0.1604 | +0.439, +0.199, +0.160 | `*N(deg3,H0,sp2,ring) C(deg4,H2,sp3,ring) C(deg4,H1,sp3,ring) | 0-2:single, 1-2:single` |
| 3 | 3 FULL | 28 | 15 | 0.1577 | +0.439, +0.190, +0.158 | `*C(deg4,H1,sp3,ring) C(deg4,H2,sp3,ring) N(deg3,H0,sp2,ring) | 0-1:single, 0-2:single` |
| 4 | 4 FULL | 88 | 32 | 0.1572 | +0.254, +0.179, +0.157 | `*N(deg2,H0,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) N(deg3,H0,sp2,ring) | 0-1:aromatic, 0-2:aromatic, 1-3:single` |
| 5 | 3 FULL | 26 | 15 | 0.1464 | -0.408, -0.159, -0.146 | `*C(deg4,H2,sp3,ring) C(deg4,H1,sp3,ring) C(deg4,H1,sp3,ring) | 0-2:single, 1-2:single` |
| 6 | 3 FULL | 96 | 54 | 0.1400 | +0.146, +0.150, +0.140 | `*C(deg3,H1,sp2,arom,ring) C(deg3,H0,sp2,ring) C(deg3,H0,sp2,arom,ring) | 0-2:aromatic, 1-2:single` |
| 7 | 3 FULL | 26 | 15 | 0.1398 | -0.429, -0.157, -0.140 | `*C(deg4,H1,sp3,ring) C(deg4,H2,sp3,ring) C(deg4,H1,sp3,ring) | 0-1:single, 0-2:single` |
| 8 | 3 FULL | 26 | 15 | 0.1398 | -0.429, -0.157, -0.140 | `*C(deg4,H1,sp3,ring) C(deg4,H2,sp3,ring) C(deg4,H1,sp3,ring) | 0-2:single, 1-2:single` |
| 9 | 4 FULL | 41 | 34 | 0.1381 | -0.138, -0.216, -0.147 | `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2) C(deg3,H0,sp2,arom,ring) S(deg1,H0,sp3) | 0-2:aromatic, 0-3:single, 1-2:single` |

Remaining top words:

- **10** 4 FULL, df=48, score=0.1331, corr=[+0.306, +0.133, +0.182]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2) C(deg3,H0,sp2,arom,ring) N(deg3,H2,sp2) | 0-1:single, 0-2:aromatic, 2-3:single`
- **11** 4 FULL, df=48, score=0.1331, corr=[+0.306, +0.133, +0.182]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2) C(deg3,H0,sp2,arom,ring) N(deg3,H2,sp2) | 0-2:aromatic, 0-3:single, 1-2:single`
- **12** 3 FULL, df=96, score=0.1314, corr=[+0.164, +0.157, +0.131]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,ring) C(deg3,H1,sp2,arom,ring) | 0-1:single, 0-2:aromatic`
- **13** 4 FULL, df=41, score=0.1307, corr=[-0.131, -0.192, -0.136]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2) C(deg3,H0,sp2,arom,ring) S(deg1,H0,sp3) | 0-1:single, 0-2:aromatic, 2-3:single`
- **14** 4 FULL, df=135, score=0.1296, corr=[-0.320, -0.155, -0.130]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) N(deg3,H0,sp2,ring) | 0-1:aromatic, 0-2:aromatic, 0-3:single`
- **15** 4 FULL, df=135, score=0.1296, corr=[-0.320, -0.155, -0.130]  
  `*N(deg3,H0,sp2,ring) C(deg3,H0,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) | 0-1:single, 1-2:aromatic, 1-3:aromatic`
- **16** 3 FULL, df=143, score=0.1290, corr=[+0.290, +0.129, +0.130]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) C(deg4,H2,sp3,ring) | 0-1:aromatic, 0-2:single`
- **17** 3 FULL, df=143, score=0.1290, corr=[+0.290, +0.129, +0.130]  
  `*C(deg4,H2,sp3,ring) C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) | 0-2:single, 1-2:aromatic`
- **18** 4 FULL, df=135, score=0.1282, corr=[-0.322, -0.154, -0.128]  
  `*C(deg3,H1,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) N(deg3,H0,sp2,ring) | 0-1:aromatic, 1-2:aromatic, 1-3:single`
- **19** 3 FULL, df=143, score=0.1280, corr=[+0.298, +0.128, +0.144]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) C(deg4,H2,sp3,ring) | 0-1:aromatic, 1-2:single`

## D1 top 20 by frozen score

| # | size/gran | df | scaffold df | score | fold corr | pattern |
|---:|---|---:|---:|---:|---|---|
| 0 | 4 COARSE | 743 | 373 | 0.1536 | -0.154, -0.179, -0.165 | `*C C(arom) N O | 0-1:single, 0-2:single, 0-3:double` |
| 1 | 4 COARSE | 743 | 373 | 0.1536 | -0.154, -0.179, -0.165 | `*C(arom) C N O | 0-1:single, 1-2:single, 1-3:double` |
| 2 | 4 COARSE | 743 | 373 | 0.1536 | -0.154, -0.179, -0.165 | `*O C C(arom) N | 0-1:double, 1-2:single, 1-3:single` |
| 3 | 4 COARSE | 743 | 373 | 0.1527 | -0.153, -0.192, -0.167 | `*N C C(arom) O | 0-1:single, 1-2:single, 1-3:double` |
| 4 | 4 FULL | 45 | 27 | 0.1457 | +0.192, +0.170, +0.146 | `*C(deg4,H1,sp3,ring) C(deg4,H1,sp3,ring) C(deg4,H2,sp3,ring) C(deg4,H2,sp3,ring) | 0-2:single, 0-3:single, 1-3:single` |
| 5 | 4 FULL | 176 | 121 | 0.1429 | -0.159, -0.219, -0.143 | `*C(deg3,H1,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) N(deg2,H0,sp2,arom,ring) | 0-1:aromatic, 0-2:aromatic, 2-3:aromatic` |
| 6 | 4 FULL | 176 | 121 | 0.1420 | -0.155, -0.223, -0.142 | `*C(deg3,H1,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) N(deg2,H0,sp2,arom,ring) | 0-2:aromatic, 0-3:aromatic, 1-2:aromatic` |
| 7 | 4 FULL | 48 | 29 | 0.1416 | -0.142, -0.167, -0.151 | `*C(deg3,H0,sp2,ring) C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) O(deg1,H0,sp2) | 0-2:single, 0-3:double, 1-2:aromatic` |
| 8 | 4 FULL | 45 | 27 | 0.1406 | +0.180, +0.149, +0.141 | `*C(deg4,H2,sp3,ring) C(deg4,H1,sp3,ring) C(deg4,H1,sp3,ring) C(deg4,H2,sp3,ring) | 0-1:single, 0-2:single, 2-3:single` |
| 9 | 4 FULL | 48 | 29 | 0.1395 | -0.139, -0.170, -0.150 | `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,ring) C(deg3,H0,sp2,arom,ring) O(deg1,H0,sp2) | 0-1:single, 0-2:aromatic, 1-3:double` |

Remaining top words:

- **10** 4 FULL, df=122, score=0.1301, corr=[+0.198, +0.135, +0.130]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) N(deg2,H0,sp2,arom,ring) N(deg2,H0,sp2,arom,ring) | 0-1:aromatic, 0-3:aromatic, 1-2:aromatic`
- **11** 4 COARSE, df=383, score=0.1263, corr=[+0.177, +0.126, +0.181]  
  `*C(arom) C(arom) N(arom) N(arom) | 0-1:aromatic, 0-3:aromatic, 1-2:aromatic`
- **12** 4 FULL, df=74, score=0.1235, corr=[-0.156, -0.123, -0.125]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) N(deg3,H2,sp2) | 0-1:aromatic, 0-2:aromatic, 1-3:single`
- **13** 4 FULL, df=74, score=0.1229, corr=[-0.154, -0.123, -0.124]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) N(deg3,H2,sp2) | 0-1:aromatic, 0-3:single, 1-2:aromatic`
- **14** 4 FULL, df=20, score=0.1222, corr=[-0.122, -0.138, -0.205]  
  `*C(deg4,H2,sp3,ring) C(deg4,H2,sp3,ring) C(deg4,H2,sp3,ring) N(deg3,H0,sp3,ring) | 0-3:single, 1-3:single, 2-3:single`
- **15** 4 FULL, df=20, score=0.1222, corr=[-0.122, -0.138, -0.205]  
  `*N(deg3,H0,sp3,ring) C(deg4,H2,sp3,ring) C(deg4,H2,sp3,ring) C(deg4,H2,sp3,ring) | 0-1:single, 0-2:single, 0-3:single`
- **16** 4 FULL, df=66, score=0.1221, corr=[+0.122, +0.147, +0.170]  
  `*C(deg3,H0,sp2,ring) C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) C(deg3,H1,sp2,arom,ring) | 0-2:single, 1-2:aromatic, 2-3:aromatic`
- **17** 4 FULL, df=71, score=0.1196, corr=[+0.159, +0.120, +0.147]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) N(deg2,H0,sp2,arom,ring) N(deg3,H2,sp2) | 0-1:aromatic, 0-2:aromatic, 1-3:single`
- **18** 4 FULL, df=71, score=0.1194, corr=[+0.162, +0.119, +0.141]  
  `*C(deg3,H0,sp2,arom,ring) C(deg3,H0,sp2,arom,ring) N(deg2,H0,sp2,arom,ring) N(deg3,H2,sp2) | 0-1:aromatic, 0-3:single, 1-2:aromatic`
- **19** 3 FULL, df=29, score=0.1193, corr=[+0.119, +0.199, +0.147]  
  `*N(deg2,H0,sp2,arom,ring) N(deg2,H0,sp2,arom,ring) N(deg3,H0,sp2,arom,ring) | 0-1:aromatic, 0-2:aromatic`
