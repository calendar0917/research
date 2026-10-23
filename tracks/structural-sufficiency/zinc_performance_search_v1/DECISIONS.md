# Decisions — zinc_performance_search_v1

Chronological, with the evidence that forced each one. Timestamps UTC.

## D1 (16:06) Round shape: performance-first, one round, two A100s on `res`
Budget 8 h wall (hard 00:06 UTC), 16 GPU-h cap, soft 15. Discovery closes
~20:30, freeze ~21:00, >=30 min for the terminal read. Two bodies from
history (A = `ad2_models.P3NullModel`, B = `dcc_models` L arm) plus one
attempt at a third (C). No ensembles, no seed averaging, no test use before
the freeze.

## D2 (16:10) Reuse of the two seed-0 anchors instead of re-running them
`xBssl_P3N_s0` (A-R0, 0.076224 @377) and `L_s0` (B-R0, 0.078537 @489) were
produced by exactly the code path, data caches and SSL encoder this round
uses (encoder sha256 `006fabc1...`). They are cited as seed-0 members of the
comparison and re-verified in this round (curve, predictions and MAE
recomputed from `predictions/valid_best.npz`). Only seeds 1-3 are new.

## D3 (16:15) Trainer extension kept additive
`ad_train.py` gained `min_lr`, `loss`, `smooth_l1_beta`, `ema`,
`recipe_override`; defaults reproduce the historical R0 exactly, which is what
makes D2 legitimate. No change to data, collate, or model code.

## D4 (16:40) EMA as an optional policy, then dropped
EMA (decay 0.999, start at 25 % of updates) was declared as an optional
single-trajectory stabiliser for A/B only. The first implementation copied the
freshly built model *into* the EMA buffer instead of the reverse, so every
EMA evaluation reported the untrained initialisation (constant valid
1.60004 for 600 epochs on both bodies). The bug is fixed in `ad_train.py`,
but re-running to obtain a legitimate EMA branch would cost 2 x 0.35 GPU-h and
would still leave the policy without seed-0/1 evidence at freeze time, so
**no EMA number is used or reported as a candidate in this round**; the two
affected runs are ranked on their online branch and their `*_ema` artifacts
are marked invalid.

## D5 (16:45) Arm C: GRIT first, on measured cost
The brief asks for a learnable explicit-relation body, preferably 10-layer
GRIT with RRWP 1..20 (473k params). I implemented and audited a dense port
(`zps_grit_dense.py`) that mirrors the vendored upstream parameter names and
shapes, verified RRWP against the frozen cache (max diff 2.4e-7) and verified
end-to-end parity against the sparse upstream with dropout off (forward
1.7e-6, gradient relative-norm error 9.6e-5, BN buffers 1.2e-7).

Measured cost then excluded it:
- dense, batch 32: 96 ms/step -> 31.0 s/epoch -> 2000 epochs = 17.2 h;
- dense, batch 128: 14.8 s/epoch -> 9.5 h for 2000 epochs;
- vendored sparse at 10 layers: ~9.5 s/epoch (8-layer ZSRT timing) -> >=4 h
  for 1500 epochs.
Both are over the round's 2 h per-run ceiling, and shortening GRIT to ~600
epochs would leave it far below its 625k-update recipe while still costing
1.5-2 h. Decision: **C = the directed-bond Transformer fallback**, native
width 64 / 4 layers / 4 heads (452,523 params), native recipe, 600-epoch
horizon (94,200 updates) so that one complete run fits in 1.5 h. The
deviation (1500 -> 600 epochs) is declared in `runs/dbtc_/dbt_s0/search.json`
and the historical full-horizon DBT run is reported as a reference row only.

## D6 (17:30) Ranking signal
Primary: best-valid MAE (strict `<`, earliest epoch on ties) — the same
signal every historical run used. Secondary diagnostics reported for every
run so a fragile winner can be recognised: W20 (minimum 20-epoch median of
valid MAE), final-epoch valid, train MAE at the selected epoch, and the
row-172 / top-10 sensitivity of the valid MAE. Seed-0/1 mean decides the
shortlist (tie: W20, then GPU-hours).

## D7 (17:35) One targeted extra config: R0L
With discovery cheap so far (~1.7 GPU-h), the one declared extra config is
R0L = R0 values with a 1200-epoch full cosine (94,800 updates). Rationale:
A-R0's best checkpoint is at epoch 377/600, i.e. the 600-epoch cosine is
still annealing when it peaks, so the horizon is the most plausible single
lever left for A. Everything else (lr, wd, loss, dropout) is R0.
