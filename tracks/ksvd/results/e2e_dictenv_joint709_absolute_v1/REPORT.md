# E2E-DictEnv-Joint709-Absolute-v1 — report

- protocol: `e2e_dictenv_joint709_absolute_v1` (study `zinc-context-gap`, seed 0)
- git commit: `d287b1d6bfbfddadd7733b908f085cee3a3f73c5`
- device: `cpu` (CPU regime, explicit `runtime.device=cpu`)
- official test loaded: `false` (never instantiated)
- candidate: `JOINT-SPARSE` — frozen K48/s12 joint dictionary over the train-only balanced 709-D input; coordinate width 49
- verdict: **JOINT709_STOP** (band `stop`)

## Absolute result (official valid 1000, Top-5 soup)

- Best valid MAE: **0.135177988** @ epoch 320
- Top-5 soup valid MAE: **0.132442119** (members [285, 315, 317, 318, 320])
- Official-train soup MAE (same protocol): 0.056583716; train−valid gap -0.075858403
- Band: `stop` — stop: the candidate did not enter a better absolute interval
- Thresholds: strong ≤ 0.115, promising ≤ 0.12, borderline ≤ 0.1233, else stop

## Training curve

- epochs run: 320 / 320 (completed: True)
- valid first 0.619420 → last 0.135178; min 0.135178 @ 320; last-20 mean 0.145034
- train last 0.086737 (min 0.083834)
- wall 2631.6s, 8.22 s/epoch

## Frozen dictionary diagnostics

- K-SVD: K=48/s=12, 10 epochs, seed 20260924, 231664 train rows, final fit MSE 1.165291, SHA-256 `541125249d2f069f…`
- Real joint reconstruction (tied-IHT top-12, the actual model path): valid 0.941186, train 0.895192
- Per-block relative error (valid): corr 0.9352, sem 0.8985, struct 0.9792
- Soup code usage: 40/48 atoms active, effective 24.55, exact l0 mean 12.00, top1 share 0.878

## Usage diagnostics (soup state, no retraining)

- zero the joint block: ΔMAE +0.000000 (0.132442)
- within-molecule row shuffle: seed 11 Δ+0.000000, seed 22 Δ+0.000000

The two probes are exactly zero because the **trained coordinate binding is inert**: in the
soup/best states all four binding matrices are denormal-zero (`|W_A_S|max ≈ 7.1e-38`,
`W_A_C`, `W_E_S`, `W_E_C` likewise), so the coordinate cannot change the prediction.
The channel is live *before* training (init-time prediction shift 8.0e-05 > 1e-6 gate,
non-zero joint binding gradients), and the decay dynamics were reproduced in a 6-epoch
subset probe (node pair ≈ ×0.85/epoch, edge pair ≈ ×0.975/epoch). The dead *node* pair is
pre-existing and already recorded (`claim-e2e-dictenv-jointbond-v1-key-level-fusion-inert-denormal-collapse-20260930`;
the route-1 / RoleCorr-v1 soups show the same 7e-38 floor, with only the edge pair alive
there and carrying their block sensitivity). The route-2-specific observation is that the
edge pair also reaches the floor here. The prior decay diagnostic
(`claim-e2e-dictenv-jointbond-decay-diagnostic-v1-frozen-parent-branch-trainable-decay-slows-not-blocks-20260930`)
shows weight decay alone does not block a live branch, so this is read as a sub-resolution
task signal on this coordinate rather than a global optimizer bug.
Consequence: 0.132442 measures the frozen protocol with an **inert dictionary
coordinate**, not the joint dictionary route's ceiling. See
`notes/e2e_dictenv_joint709_absolute_v1_analysis.md` §3. This round has no control arm,
so no increment / sparse-vs-dense / causal statement follows.

## Provenance

- correctness gates all passed: True; smoke passed: True; frozen objects verified: True; cache ordering: True; scaler train-only refit: True
- split fingerprint / SHAs: control-plane manifest + `artifacts/*.json`
- parameters: trainable 99469, total 135581
