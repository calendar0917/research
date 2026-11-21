# PROTOCOL — word response operator, s0 (lipo, seed 0)

Protocol id: `word-response-operator-s0-v1` (config `configs/wo_s0.json`, hashed into every run).
Design document: `Downloads/structural_word_response_operator_s0_design_and_prompt.md`
(2026-11-21 / s0). This file is the pre-registration; deviations are listed at the end.

## Question

Is a word represented as a *reusable multi-port boundary response function*
worth more than (a) a scalar/count word contribution, (b) independent per-word
computation, (c) a purely atomic/bond response model — on one frozen Lipo split,
one seed, with GINE only as an external performance reference?

## Data

* `ogbg-mollipo` official Train/Valid/Test = 3360/420/420. Test is never read.
* Vocabulary, canonicalisation, replay and full atom/bond attributes are the
  frozen Train-only word table of `word_sentence_port_route_v1` (148 types,
  max 6 atoms, all 9 atom + 3 bond columns, salts and components kept).
* Target scaler, `df`, `s_type` and the count vector are Fit-only.

### Fit / H / Valid (frozen before any label is read)

`code/wo_split.py` reads only `response_dictionary_task_transfer_s0_v1/analysis/pairs.csv`
structure columns (no `delta_y`/label column) and keeps pairs whose two ends are
both official Train. Graph components over these Train-Train pairs give the
component key = sorted official row ids; `SHA256("word-op-s0|" + key) mod 5 == 0`
selects candidate H components (Fit = remaining Train molecules, isolated
molecules stay in Fit). Evaluable transforms need ≥2 supporting Fit components,
an H-internal pair, and a core key never on a Fit pair.

`analysis/split_freeze.json` records the decision: `H_ENABLED`, 629 H molecules,
2731 Fit molecules, 105 evaluable pairs, 47 H components, 58 transforms; all four
structural conditions met (50–840 molecules, ≥40 pairs, ≥10 components,
≥8 transforms). `labels_read: false` at freeze time. H is a **new-round label
holdout**, not a fresh/blind set: the vocabulary itself has seen all official
Train structures.

## Model (no GNN, no message passing)

4 independent response channels; per-atom scalar latent per channel; one shared
scalar prediction.

* atom embedding: 9 columns × dim 4 → 36; bond: 3 × 4 → 12; both concatenated
  (not summed);
* plain 2-layer SiLU MLPs (hidden 32) for `d_a`, `k_ab`, `b_base`, `PairMLP`;
* non-edge pairs inside a word with shortest-path ≥2:
  `f_ab = [x_a+x_b (36), x_a*x_b (36), dist onehot 0..5 (6), log1p(n_i), log1p(m_i), cycle_rank_i]` (81 dims);
  `lambda = 0.5/(n_i-1) * sigmoid(PairMLP(f_ab)_h + s_type(i)*theta_type(i),orbit(a,b),h)`;
  PairMLP last-layer bias init −3;
* `Q_i = diag(d) + Σ_internal k_ab l l^T + Σ_nonedge lambda l l^T`, `l = e_a − e_b`,
  `d ≥ 0.5` via softplus, so `Q_i ≻ 0.5 I` by construction;
* word source `b_i = b_base + 0.25 s_type [tanh ψ(type,orbit(a)) − mean_r tanh ψ(type,orbit(r))]`;
* `c, S, h` from block elimination with Cholesky (`Q_II`), never an explicit inverse;
* `K = blockdiag(S_i) + Σ_cross k (e_p−e_q)(e_p−e_q)^T`, `F_h = Σc + ½logdet K − ½ h^T K^{-1} h`;
* `ŷ_std = β0 + β^T z_count + Σ_h α_h F_h / n0`, `n0` = Fit mean atom count,
  `α` init `[0.1,−0.1,0.1,−0.1]`, `β,β0` init 0. Count vector = atom/bond category
  counts + word-type instance counts + `N_atoms, N_bonds, N_components, cycle_rank`
  (raw counts, Fit column standardisation, zero-variance columns dropped; 188 kept).

Arms:

| arm | change |
|---|---|
| `FULL_WORD_OP` | full definition; trained in atom space, must equal boundary assembly |
| `LOCAL_WORD_OP` | cross-word `+k` kept on the diagonal, cross-word off-diagonal `−k` removed |
| `ATOMIC_OP_CAP` | same global Gaussian solve and `A_β`, `Q` from per-atom `d`/`b` and all physical `k` only; no pair term, θ, ψ; width chosen (before any metric) from `[32,48,64,96,128,192,256]` to match FULL's parameter count ±20% |
| `PORT_DIAG_WORD_OP` | conditional: `S_i → diag(diag(S_i))`; only after a saved GREEN decision |
| `ADD_WORD` | ridge on the same `z_count`, λ=1, intercept unpenalised |
| `FP_KRR` | Morgan r2/2048/chirality, Tanimoto kernel ridge, α=0.01 |
| `GINE_RAW_REF` | previous round's plain GINE (width 128, 5 layers, raw 9/3 columns), 240 epochs, refit on this Fit |

`GINE_RAW_REF` is the only file allowed to import GINE; it never enters the
operator forward pass, supervision, init or features.

## Recipe (operator arms share everything)

seed 0; batch 64; AdamW lr 2e-3, wd 1e-4, 5-epoch warmup, cosine to 2e-5; clip 5;
FP32, TF32 off; no dropout/EMA/SWA/pretraining; identical per-epoch Fit
permutations from `default_rng(20261121)`; FULL/LOCAL share bit-identical init.

`loss = MSE + (1/N_Fit)Σβ² + 1e-3·mean(s_w²(θ²+ψ²)) + 1e-4·Σα²` (ATOMIC drops
the word term; β0 unpenalised). Checkpoint selection: best official-Valid RMSE,
ties to the earlier epoch; best/final/recoverable state, curves and Fit/Valid
predictions by official row id are saved.

Horizon: 2 warm + 3 timed full epochs on the operator arms; if
`slowest_epoch × 160 × 1.25 > 18 min` all operator arms use 80 epochs, else 160.
`analysis/timing.json` and `analysis/pre_train_freeze.json` are committed before
the first formal run. GINE keeps its own 240-epoch recipe and is reported
separately.

## Checks before training (`code/wo_probes.py`, `analysis/checks.json`)

16 real molecules: molecule/word maps, all 9/3 columns, internal + crossing
bonds, boundary slots, batch offsets. FULL atom-space vs boundary assembly in
FP64 (≤1e-8) and FP32 (≤1e-4), covering empty I, empty B, 1/multi ports,
multiple crossing bonds, multi-component. Batch split and reversed order
invariance. Two-word merge before elimination equals per-word elimination
(≤1e-9). Bit-identical init for FULL/LOCAL/PORT_DIAG. Width selection for
ATOMIC before any metric. 32 Fit molecules / 60 steps health: non-zero
word-coefficient, PairMLP and α gradients, decreasing loss, no NaN, plus a
finite-difference check. Replay must reproduce the frozen word partition
(template ids, atoms, internal bonds) of the previous round exactly.

Smoke weights are never reused for formal runs.

## Decisions (Valid only, single seed)

`delta = 0.015 × Fit label std`, `band = 0.05 × Fit label std`.
GREEN: FULL better than LOCAL, ATOMIC and ADD by ≥delta and no worse than
`min(KRR, GINE) + band`. YELLOW: beats the three controls by delta but is behind
the strong reference by more than band. BASELINE_SIGNAL_ONLY: competitive but
does not beat both LOCAL and ATOMIC by delta. RED otherwise; missing required
controls = INCOMPLETE. Only GREEN unlocks one `PORT_DIAG_WORD_OP` run.

## Holdout (one shot, after all training and selection)

After the candidate/weight list is frozen and committed, H labels are opened once:
whole-molecule RMSE/MAE for every arm, `Δ_pred = F(B)−F(A)` vs `Δ_y` on the 105
evaluable pairs (MAE/RMSE/direction accuracy), per-transform component-mean
TABLE, and 500 paired bootstrap replicates over H components. No training,
no seed addition, no CI-driven extension.

## Resource protocol

T0 = 2026-11-21T13:14:52+08:00. 4 h wall, 4 GPU·h, ≤2 GPUs, stop launching at
T0+3 h, save and stop training at T0+3 h 20 min, ≤7 formal units. All runs on
`res` via `rr` (one GPU per job, `res-gpu0`/`res-gpu1`), each with the absolute
deadline. Timing, smoke and failed attempts count as resources. If `res` is
unreachable the pre-authorised fallback is local CPU (2 workers × ≤7 threads),
announced once, and the whole comparison stays in one regime.

## Deviations (filled during execution)

* see `REPORT.md` §deviations.
