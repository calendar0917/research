# MFIC v1 — analytic matrix-function quadratic composition

Protocol `cscl-bjoc-mfic-v1`; split V; report
`results/mfic_v1/REPORT.md`; decision **`MFIC_NO_USEFUL_GAIN`**.

## What was asked

Replace P3's 96-d `sum a_i^2` reader block with the 136 upper-triangle entries of
`C = Q^T K Q / max(1,n)`, where `K = sum_{k>=2} 0.65^k S^k`,
`S = D^{-1/2} A D^{-1/2}` from the physical heavy-atom adjacency, and
`Q = Linear(96->16)(h_i)`. Three arms: P3 (413,505), MFIC-R real K (425,297),
MFIC-U content-null `K_U = d*I + o*(11^T-I)` (425,297).

## CPU purchase gate: PASS

4-fold identity-grouped Ridge on Fit only, fixed 16-d train-fold PCA q, fixed
`alpha=1`: Base 0.496257 -> Matrix 0.395422, delta **-0.100835**, 4/4 folds
negative, trimmed-10 -0.092055, ordinary -0.090967, tail -0.228014. Math gates
and 20/20 model audit pass. Decision `MATRIX_FEATURE_CHEAP_SIGNAL`.

The probe is deliberately strongly regularised and uses fixed PCA q; it proves
K contains cheap structural signal, not that the learned model will use it.

## GPU Monitor: FAIL

n=1,000: P3 **0.087845**, MFIC-R 0.092489, MFIC-U 0.094765.

- R−P3 **+0.004644** (CI [+0.001032,+0.008585]; both seeds positive);
- ordinary +0.003361; tail (n=58) +0.025482; top-10 +0.003229;
- RMSE 0.243333 vs 0.206378 (+0.036955);
- R−U -0.002276 (CI [-0.005456,+0.000393]), below the pre-registered -0.003
  mechanism threshold;
- U−P3 +0.006920.

Only 6b (both seeds R−U negative) and 7 (participation) pass; 8/10 fail.
Fit MAE is lower for MFIC-R (0.000698/0.000811) than P3 (0.001178/0.001016):
the matrix arms overfit the Fit relation rather than generalise.

## Discipline

Split V is an exposed-pool resample (historical seen fraction 1.0), not blind.
Prediction freeze 18:35:06 UTC; Monitor read once at 18:37:23 UTC; Locked never
read (no `locked_read.lock`). No official valid/test, no full10k, no alpha / PCA
dim / rank / Reader / seed rescue. Budget 2.0911 / 3.0 GPU-h. No credit for the
Batatia et al. ICLR 2024 ZINC 0.063 (their model used GCN layers).

## Lesson

K's cheap signal under a fixed, regularised probe did not transfer to the
one-shot learned-q reader; the negative is specific to this frozen architecture
and recipe. Any regularisation/dimension/rank follow-up is a new pre-registered
route, not a rescue of this one.
