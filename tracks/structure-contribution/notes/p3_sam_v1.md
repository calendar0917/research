# P3-SAM v1 — flatness-bias follow-up to P3

**Protocol:** `cscl-p3-sam-v1`; **split:** T (explicit salt `cscl-p3-sam-v1`);
**decision:** `SAM_NO_USEFUL_GAIN`.

## Question

Can the held-out gap of the frozen P3-R3 model be reduced by changing only the optimisation
flatness bias (standard two-step SAM, fixed `rho=0.05`), without adding structure, capacity,
external data, regularisation, dropout or loss changes?

## Contrast

A = frozen P3 (413,505 params, Fit-only masked SSL, Adam, cosine, raw-y L1, 360 epochs, last-5
soup). B = identical architecture/init/SSL/samples/update budget, with global-L2 two-step SAM
outside Adam. First gradient is not clipped; the second gradient is clipped at 5.0 after the
second backward; parameters are exactly restored before the Adam step. Same seed: pre-SSL init,
SSL encoder, batch order and 22,680 updates were asserted equal.

## Data

Official ZINC-12K train 10k, four known input-audit rows excluded, new identity-grouped split T
7,996 / 1,000 / 1,000. Split salt and actual row hash are asserted; T has no zone hash collision
with historical E/F/S. All 9,996 rows were seen by E/F/S, so T is a resample of an exposed pool,
not an independent blind test. Official valid/test were never read. SSL used Fit structures only.

## Result (Monitor only)

| | P3 | P3SAM | delta |
|---|---:|---:|---:|
| seed-mean MAE | 0.087028 | 0.131579 | +0.044551 |
| per-seed MAE | 0.083518 / 0.090538 | 0.137359 / 0.125799 | +0.053841 / +0.035261 |
| RMSE | 0.183349 | 0.363610 | +0.180261 |
| ordinary MAE | 0.078878 | 0.116336 | +0.037459 |
| tail MAE (n=77) | 0.184726 | 0.314292 | +0.129567 |
| seed MAE spread | 0.007020 | 0.011560 | +0.004540 |

Paired molecule bootstrap CI for `P3SAM - P3`: **[+0.029202, +0.063349]**. All 7 frozen gate
conditions failed. Locked labels were not read; no full10k/test; no rho/seed/optimizer variants.

## Cost / fit

Soup Fit MAE: P3 0.001384/0.000993; P3SAM 0.008593/0.008615. Monitor−Fit gap widened from
0.085839 to 0.122975. Training wall 1,146.7 s (P3) vs 2,190.6 s (P3SAM), ~1.91×; 2 forward +
2 backward per update. Total GPU ledger 1.8480 / 3.0 h including smoke/SSL/predict.

## Interpretation boundary

This is a negative result for **this fixed SAM configuration**. It does not show that flat minima
are irrelevant or that the P3 bottleneck is flatness; it also adds no structural information.
The SAC/P3 historical decisions are unchanged.

See `results/p3_sam_v1/REPORT.md`, `DEVIATIONS.md`, `AMENDMENTS.md`, and `evidence_manifest.json`.
