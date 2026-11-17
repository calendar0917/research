| unit | JOINT | MARGINAL | C1 anchor | GINE anchor | Δ(J vs M) | 95% CI | P(J<M) | Δ(J vs C1) | P(J<C1) | top1% share | frac better |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FROZEN/lipo/s0 | 0.7047 | 0.7039 | 0.7069 | 0.6161 | -0.0007 | [-0.0034,+0.0018] | 0.2830 | 0.0022 | 0.7675 | 0.0990 | 0.5214 |
| FROZEN/wc_logp/s0 | 0.1106 | 0.1119 | 0.1144 | - | 0.0013 | [-0.0002,+0.0027] | 0.9555 | 0.0038 | 0.9930 | 0.3559 | 0.4595 |
| E2E/lipo/s0 | 0.7172 | 0.6944 | 0.7069 | 0.6161 | -0.0227 | [-0.0559,+0.0103] | 0.0925 | -0.0103 | 0.2435 | 0.1140 | 0.5262 |
| E2E/lipo/s1 | 0.7209 | 0.7066 | 0.7069 | 0.6161 | -0.0143 | [-0.0458,+0.0190] | 0.2050 | -0.0140 | 0.2775 | 0.1377 | 0.5190 |

Δ = RMSE(MARGINAL) - RMSE(JOINT): **positive means JOINT is better**.
P(J<M) is the bootstrap probability that JOINT has the lower RMSE.
