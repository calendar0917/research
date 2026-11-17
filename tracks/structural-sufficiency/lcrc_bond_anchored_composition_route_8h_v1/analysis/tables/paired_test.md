| unit | JOINT | MARGINAL | C1 anchor | GINE anchor | Δ(J vs M) | 95% CI | P(J<M) | Δ(J vs C1) | P(J<C1) | top1% share | frac better |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FROZEN/lipo/s0 | 0.7303 | 0.7324 | 0.7293 | 0.6800 | 0.0021 | [-0.0007,+0.0051] | 0.9255 | -0.0009 | 0.3880 | 0.1043 | 0.4952 |
| FROZEN/wc_logp/s0 | 0.0880 | 0.0887 | 0.0934 | - | 0.0007 | [-0.0002,+0.0016] | 0.9270 | 0.0053 | 1.0000 | 0.1696 | 0.4429 |
| E2E/lipo/s0 | 0.7200 | 0.7277 | 0.7293 | 0.6800 | 0.0077 | [-0.0341,+0.0496] | 0.6500 | 0.0093 | 0.6965 | 0.1450 | 0.4881 |

Δ = RMSE(MARGINAL) - RMSE(JOINT): **positive means JOINT is better**.
P(J<M) is the bootstrap probability that JOINT has the lower RMSE.
