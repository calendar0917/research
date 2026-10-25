# Gate mass from frozen A (official valid, descriptive)

s(layer,head,query) = sum_key attention * gate, from the frozen legacy A checkpoint; no test, no new supervision.

## ho_A_s0 (epoch 364)

- mass mean 0.3489, median 0.0860, 5-95% [0.0002492, 1.7392], frac<0.01 0.214, <0.10 0.526.
- variance share (mass): layer/head 0.261, between molecules 0.206, within-molecule queries 0.533; log10: 0.283 / 0.218 / 0.499.
- molecule-mean mass R2 on [log bonds, rings, ring-atom frac, max cycle, unseen flag]: median 0.548 (range 0.132-0.770).
- within-molecule log-mass R2 on local bond features: median 0.023 (range 0.001-0.067).
- layer/head cell means: L0H0=1.144, L0H1=0.226, L0H2=0.114, L0H3=0.320, L1H0=0.410, L1H1=0.261, L1H2=0.259, L1H3=0.434, L2H0=0.206, L2H1=0.145, L2H2=0.107, L2H3=0.560.

## A_s0 (epoch 402)

- mass mean 0.1226, median 0.0030, 5-95% [2.274e-14, 0.8575], frac<0.01 0.610, <0.10 0.816.
- variance share (mass): layer/head 0.203, between molecules 0.213, within-molecule queries 0.584; log10: 0.827 / 0.062 / 0.111.
- molecule-mean mass R2 on [log bonds, rings, ring-atom frac, max cycle, unseen flag]: median 0.377 (range 0.111-0.640).
- within-molecule log-mass R2 on local bond features: median 0.015 (range 0.001-0.057).
- layer/head cell means: L0H0=0.132, L0H1=0.115, L0H2=0.581, L0H3=0.025, L1H0=0.003, L1H1=0.199, L1H2=0.079, L1H3=0.002, L2H0=0.009, L2H1=0.112, L2H2=0.146, L2H3=0.068.
