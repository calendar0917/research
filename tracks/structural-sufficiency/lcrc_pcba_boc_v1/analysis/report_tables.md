# LCRC-BOC v1 — comparison tables

Test role: `EXPOSED_TEST_ENGINEERING_COMPARISON_SET`.  Decision bands are engineering bands, not significance tests.

## Primary table

| model | epoch | params | best Valid AP | Test AP (official) | independent AP | Valid→Test | GPU h |
|---|---:|---:|---:|---:|---:|---:|---:|
| W2 best-Valid | 24 | 994,324 | 0.265273 | 0.253616 | — | — | 1.3949 |
| W2 @40 | 40 | 994,324 | 0.261867 | 0.250985 | — | — | 1.3949 |
| RC-REAL best-Valid | 33 | 1,051,022 | 0.267718 | 0.255941 | — | — | 1.9186 |
| RC-REAL @40 | 40 | 1,051,022 | 0.266950 | 0.256166 | — | — | 1.9186 |
| BOC_FULL_best_valid | 29 | 1,352,596 | 0.269950 | 0.254142 | 0.254142 | -0.015807 | 2.4215 |
| BOC_FULL_epoch40 | 40 | 1,352,596 | 0.264009 | 0.250658 | 0.250658 | -0.013351 | 2.4215 |
| BOC_BOND_best_valid | 30 | 1,352,596 | 0.269908 | 0.251666 | 0.251666 | -0.018241 | 2.3242 |
| BOC_BOND_epoch40 | 40 | 1,352,596 | 0.267367 | 0.249054 | 0.249054 | -0.018312 | 2.3242 |

Primary band: **no_evidence** (<=0.258: no evidence).

## Support strata (Test positives per task)

* `BOND_vs_RC-REAL_5_19`: n=18 mean Δ=-0.011517822382058196 median Δ=-0.0021385953591465216 improved=7
* `BOND_vs_RC-REAL_ge20`: n=108 mean Δ=-0.0004669150522440655 median Δ=-0.0003741267664708317 improved=53
* `BOND_vs_RC-REAL_lt5`: n=1 mean Δ=-0.285141921703728 median Δ=-0.285141921703728 improved=0
* `BOND_vs_W2_5_19`: n=18 mean Δ=-0.0076215532511215755 median Δ=-0.00061425719681347 improved=7
* `BOND_vs_W2_ge20`: n=108 mean Δ=0.00045914649736972696 median Δ=0.0031621084681500433 improved=64
* `BOND_vs_W2_lt5`: n=1 mean Δ=-0.16013460767513132 median Δ=-0.16013460767513132 improved=0
* `FULL_vs_RC-REAL_5_19`: n=18 mean Δ=-0.010769213234252013 median Δ=-0.0009335761146992094 improved=8
* `FULL_vs_RC-REAL_ge20`: n=108 mean Δ=0.002284085051248649 median Δ=0.000999331066371234 improved=55
* `FULL_vs_RC-REAL_lt5`: n=1 mean Δ=-0.28124852118811 median Δ=-0.28124852118811 improved=0
* `FULL_vs_W2_5_19`: n=18 mean Δ=-0.006872944103315391 median Δ=-0.00011480142019974077 improved=8
* `FULL_vs_W2_ge20`: n=108 mean Δ=0.0032101466008624406 median Δ=0.0038721909707207652 improved=61
* `FULL_vs_W2_lt5`: n=1 mean Δ=-0.1562412071595133 median Δ=-0.1562412071595133 improved=0

## Channel norms / gradients (last 5 epochs mean)

* `FULL`: h0 1.1924, m 0.8754, r 0.8450, h 1.0820; grad L2 bond 0.0362, ring 0.0187, fusion 0.1497, W2 0.2446
* `BOND`: h0 1.2065, m 0.8600, r 0.0000, h 1.0648; grad L2 bond 0.0366, ring 0.0000, fusion 0.1694, W2 0.2815

## Zeroing sensitivity (official Validation only)

* `BOND`: both 0.269908, ring-zero 0.269905 (-0.000003), bond-zero 0.169700 (-0.100209), both-zero 0.169699
* `FULL`: both 0.269952, ring-zero 0.217645 (-0.052308), bond-zero 0.202970 (-0.066982), both-zero 0.157268

## Audits

* `structure_audit.json`: passed=True (16/16), sha `c3058aa82326`
* `model_audit.json`: passed=True (11/11), sha `051a62c17447`
