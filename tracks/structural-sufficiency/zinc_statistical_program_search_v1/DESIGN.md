# DESIGN — zinc-statistical-program-search-v1 (frozen before GPU)

Round: exploratory, discovery-first. Two independent bodies on the official PyG
ZINC subset (train 10k / valid 1k, raw-y MAE), ≤500k parameters, single model,
no recursive neighbourhood hidden-state message passing, **official test never
constructed or read**.

## 0. What is being asked

Can an explicit structural-statistics body, computed from static records rather
than from a recursive local GNN, beat the frozen DCC-L reference on the same
valid split and the same recipe? Development references (not re-trained):

- DCC-L seed 0: valid **0.078537** (epoch 489); historical 4-seed mean 0.075990.
- P3-Null seed 0: 0.079105 (413,505 params), seed 1: 0.076668.

Investment scale declared in advance (from the task): a stable entry into
0.073–0.075 with ≥0.002 improvement over the nearest matched reference is worth
a confirmation seed; 0.076–0.080 with a clearly cheaper body may keep one
declared child branch but is not a performance breakthrough; clearly worse and
plateaued after a healthy fit closes the body.

## 1. Shared front-end (both routes)

- P3 atom encoder (108→128→96) and bond encoder ((3·96+7)→160→96), **89,152**
  parameters. Bond input is endpoint-symmetric.
- Loaded from the seed-matched train-only masked-SSL checkpoint
  (`pretrain_encoder_s{0,1}.pt`) and fine-tuned; the unused junction encoder is
  not instantiated and is not counted. No supervised P3/DCC weights, no DCC
  composer/reader, no 513-d reader.
- New head parameters use a fixed independent RNG stream (`seed + 90731`). The
  two families are **not** parameter-matched; family comparison selects a route,
  it is not a one-component causal test. Seed 0 uses SSL-s0; a second supervised
  seed uses SSL-s1, so seed confirmation tests supervised-training randomness
  only (disclosed).

## 2. Route Q — rooted empirical-measure queries (Q-fixed)

For every real atom `r` as root, the real objects are every atom `u` and every
physical bond `e=(u,v)`:

- atom record `(a_u, dist(r,u))`; group by `dist ∈ {0,1,2} / {3,4} / {≥5, ⊥}`;
- bond record `(b_e, g_ru, g_rv)`; group by `min(dist(r,u), dist(r,v))`;
- every record has multiplicity 1; nothing is merged and then averaged.
- distance id `0..8`, `9` for `≥9`, `10` for unreachable; grouping always uses
  the untruncated distance. Counts and unreachable markers are kept.

Shared encoders produce key `k∈R^16` (tanh-bounded) and value `v∈R^8`:
`g_(r,u)=MLP([a_u, emb(dist)],104→64→32)`, `atom_memory=32→24`,
`bond_memory=MLP([b_e,g_ru+g_rv,|g_ru-g_rv|,g_ru·g_rv],192→64→24)`.
Four shared learned queries `q=2·tanh(raw)` read each group exactly:
`ℓ = logΣ_o c_o exp(q·k_o/√16) − logΣ_o c_o`, `m = Σ α_o v_o`, `s = Σ α_o v_o²`.
Empty groups return exact zeros (no `log 0`, no all-masked softmax NaN).

Per root: `u_r = MLP([a_r, 6 groups × 4 queries × (ℓ,m,s), counts/FitMean,
log1p(counts)], 516→128→96)`.
Graph readout: root sum and second moment, shared 96→32 projection of atoms and
bonds (first + second moments), global33, atom/bond counts → reader
`355→128→64→1`. Pooled sums are scaled only by fixed Fit mean sizes (no
per-graph mean substitution); counts are never normalised away.

Declared child branch (**not** run unless Q-fixed is competitive):
**Q-adaptive** — after the first read, `q¹_r = q⁰ + 0.25·tanh(W_q u_r)` and the
same static records are read a second time. No other root's updated hidden state
is ever a key/value; +6,208 parameters and ≈2× read cost.

Parameters: **Q-fixed 248,361** (front 89,152 + head 159,209); Q-adaptive
254,569.

## 3. Route F — content-conditioned rooted-forest marginals (F-content)

Real atoms are vertices and real physical bonds are edges. `w_e = 0.1 +
1.9·sigmoid(MLP(b_e)) ∈ (0.1,2)`; fixed scales `τ={0.25,1,4,16}`.

`L_τ = J diag(τ w) Jᵀ`, `A_τ = I + L_τ`, `Q_τ = A_τ^{-1}` (dense Cholesky solve,
sign of the incidence columns irrelevant).

- atom statistic `r_i,τ = Q_τ[i,i]`; bond statistic
  `p_e,τ = τ w_e (Q_uu+Q_vv−Q_uv−Q_vu)`; auxiliary `logZ_τ = logdet A_τ`
  (passed as `logZ/n_atoms`).
- `h^A_i = φ_A([a_i, r_i,·])`, `h^B_e = φ_B([b_e, p_e,·])`, `φ = 100→128→96`.
- Pool first/second moments of `h^A` and `h^B` only (a plain
  `Σr + Σp = N` identity is never used as a feature), plus `logZ/n`, global33,
  counts → reader `423→128→64→1`.

Declared child branch (**not** run unless F-content is competitive):
**F-fixed** (`w_e ≡ 1`, conductance module removed; 202,369 parameters).

Parameters: **F-content 205,506** (front 89,152 + head 116,354).

## 4. Recipe (identical for both bodies, declared before running)

AdamW, peak lr 1e-3 → min 1e-5 per-epoch cosine, warmup 500 updates,
weight decay 1e-5, grad clip 5, dropout 0, raw-y L1, batch 128,
valid batch 256, **600 epochs = 47,400 updates** (79 steps/epoch), full valid
every epoch, checkpoint = strict argmin(valid MAE) (earlier epoch wins ties).
Exactly the mature DCC-R0 recipe. No early-stopping LR reset: if a run is
stopped early, the state is reported as "stopped at epoch k with the full-600
cosine", never as a completed 600-epoch result.

Feature/scaling constants come only from official train structures (`fit_stats.json`):
mean atoms 23.083, mean bonds 24.846, per-group mean counts, global33 mean/scale
from the frozen P3 train-only prep. No valid/test labels are used anywhere in
prep.

## 5. Stop / buy rules (this round)

- Hard stops only for implementation error, leakage, budget violation or proven
  algebraic duplication. Initialization PR, initialization R² and early valid
  spikes are diagnostics, not gates.
- A body that fits train normally (online MAE falling, eval-mode train MAE well
  below valid) and is clearly worse than the reference with a flat valid curve
  after ~mid-horizon is closed; no decorative ablations.
- A body in the competitive band gets the second supervised seed (SSL-s1) before
  anything else. A declared mechanism control (Q-adaptive or F-fixed) is bought
  only if the first seed is competitive, and only one such control.
- ≤5 architecture configurations total in discovery; no width×rank×LR grid.
- Budget: 8 h wall (hard), 16 A100·h (hard), 14 A100·h soft. Per-run projected
  cost must be ≤2.25 GPU-h or the body is declared cost-unfit.

## 6. Data boundary

Official PyG `ZINC(subset=True)`: train 10,000 / valid 1,000. The test split is
never built, indexed or read in this round. Comparisons use full-1000-row valid
MAE, a 20-epoch-median window (W20), the late trend, eval-mode train MAE, and
per-molecule signed differences (candidate − reference, negative is better).
