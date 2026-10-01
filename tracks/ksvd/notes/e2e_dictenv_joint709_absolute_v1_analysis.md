# E2E-DictEnv-Joint709-Absolute-v1 — analysis (seed 0, local CPU)

Round: **E2E-DictEnv-Joint709-Absolute-v1** (`e2e_dictenv_joint709_absolute_v1`).
Pre-registration: `notes/e2e_dictenv_joint709_absolute_v1_preregistration.md` (frozen; write-then-follow).
Protocol: `zinc-context-gap` (study `zinc-context-gap`), `protocol_hash 52b50e0b…`.
Runner: `zinc_e2e_dictenv_joint709_absolute_v1`.
Formal screen run: `20261001-144617-89099a3f` (mode `screen`, `test_access: blocked`).
Prepare (scratch) run: `20261001-131619-843f4adc` (mode `scratch`, `test_access: blocked`).
Authoring revision: `d287b1d6bfbfddadd7733b908f085cee3a3f73c5` (dirty flag only from the
untracked result-dir snapshot `run_chain.sh`; the tracked tree equals the revision).
Device: CPU only (`runtime.device=cpu`, `torch_threads=8`, torch 2.5.1, no CUDA).
Official ZINC **test never loaded** (`official_test_loaded: false` in every artifact).

## 0. Decision (first line)

**`JOINT709_STOP`** — soup valid MAE **0.132442** falls in the pre-registered `stop`
interval (`> 0.1233`). The candidate does not enter a better absolute interval, so the
frozen rule terminates this route: no seed 1, no rescue, no K/s/LR/horizon sweep,
no dictionary fine-tuning.

Second, independent of the band rule: the run's *effective* model did not carry the
dictionary coordinate at all. At the soup state the coordinate binding matrices are
denormal-zero, so zeroing or row-shuffling the frozen joint code leaves the predictions
**bit-identical** (`ΔΔMAE = 0.0` exactly). The 0.132442 number is therefore the
absolute performance of the frozen protocol with an **inert dictionary coordinate**
(the Sem108 + frozen-fusion path), not an estimate of "what the joint dictionary route
could reach". Both statements are recorded; the pre-registered band rule fires `stop`
either way.

## 1. Absolute result (pre-registered endpoint)

| quantity | value |
|---|---|
| soup valid MAE (endpoint) | **0.1324421192816808** → band `stop` |
| soup members (Top-5, fixed rule) | epochs [285, 315, 317, 318, 320] |
| member valid MAE | 0.140583 / 0.137384 / 0.140607 / 0.137621 / 0.135178 |
| best single epoch | 0.13517798792017857 @ epoch 320 (last epoch) |
| valid first / last | 0.619420 / 0.135178 |
| valid last-20 mean | 0.145034 |
| soup train MAE (official train 10000) | 0.056584 |
| train min MAE | 0.083834 (last-epoch train 0.086737) |
| train–valid gap (soup train − soup valid) | −0.075858 |
| verdict / band | `JOINT709_STOP` / `stop` |

The best epoch is the **last** epoch and the top-5 members are all from the final 36
epochs: the curve had not flattened, it was still drifting downward at ~0.135. Soup
averaging over the tail gives 0.132442, i.e. the endpoint is 0.0027 better than the
best single state. Even if the tail drift were extrapolated generously the candidate
is ~0.010 MAE away from the `borderline` edge (0.1233), so the `stop` verdict is not a
matter of a few more epochs.

## 2. Frozen joint dictionary: what was actually built and how well it codes

All objects are train-only (fitted on the official train split, never on valid).

| item | value |
|---|---|
| input dim / atoms / sparsity | 709 / 48 / 12 |
| K-SVD: rows, epochs, seed | 231664 train rows, 10 epochs, seed 20260924 |
| K-SVD final fit MSE (OMP-coded) | 1.165291 |
| K-SVD wall time | 5194.1 s (prepare stage total 5248.1 s) |
| `dictionary_joint.pt` shape / sha256(f32) | [709, 48] / `541125249d2f069f5a81a7a335ac75d09b140977c3970be219aba9d9465b18a4` |
| scaler block weights (struct / sem / corr) | 0.1336 / 0.1179 / 0.0702 |
| valid block mean row²-norm after scaling | struct 1.034 / sem 0.797 / corr 0.878 |

Real coding diagnostics with the frozen operator (`tied_iht_codes`, exact top-12,
`eta = 1/(sigma²+eps)`, 10 iterations — the operator actually used in the coordinate):

| quantity | train | valid |
|---|---|---|
| tied-IHT(10) relative error | 0.895192 | 0.941186 |
| tied-IHT(10) per block (valid) | struct 0.979 / sem 0.899 / corr 0.935 | |
| OMP(12) reference (train subset, 16384 rows) | 0.646282 | — |
| extra diagnostic only: IHT 30 steps | 0.802 | 0.900 |
| extra diagnostic only: IHT 60 steps | 0.772 | 0.865 |

Interpretation (descriptive, no new experiment was run): the frozen K48/s12 dictionary
is a **weak code** for the scale-balanced 709-D environment vector. Even a 6× larger
IHT budget (60 steps) stays at 0.77/0.87 relative error and the OMP(12) optimum is
0.646, so the limit is the dictionary *capacity/geometry* at K=48 vs a 709-D input
(plus the per-row top-12 budget), not the 10-step budget alone. The corr and struct
blocks are the worst-coded; the sem block codes best (0.899/0.935 valid/train).

Code usage at the soup state (frozen dictionary, valid split): 40/48 atoms active,
effective atoms 24.55, exact `l0` mean 12.0, top-1 activation share 0.878 — i.e. the
codes are extremely concentrated on a few atoms, consistent with a coarse 48-atom
approximation of a 709-D signal.

## 3. Why zero/shuffle probes are exactly 0.0 — the coordinate channel is switched off

Pre-registered probes on the soup state (valid 1000, diagnostic only):

* zero the whole joint block of the coordinate → MAE 0.132442, `delta vs intact = 0.0` (exact);
* within-molecule row shuffle of the joint block, seeds 11 and 22 → MAE 0.132442,
  `delta vs intact = 0.0` (exact, both seeds).

This is not a broken probe. Reading the trained parameters directly:

| checkpoint | `|W_A_S|max` | `|W_A_C|max` | `|W_E_S|max` | `|W_E_C|max` |
|---|---|---|---|---|
| route-2 `JOINT-SPARSE` soup (this round) | 7.1e-38 | 7.0e-38 | 7.0e-38 | 6.8e-38 |
| route-1 `EXTRA-STRUCT` soup (increment-v2) | 7.1e-38 | 7.1e-38 | 0.541 | 0.590 |
| route-1 `CORR-ADD` soup (increment-v2) | 7.0e-38 | 7.0e-38 | 0.866 | 0.559 |
| route-1 `CORR-PCA-ADD` soup (increment-v2) | 7.1e-38 | 7.0e-38 | 0.895 | 0.523 |
| `RoleCorr-v1` CORR soup | 7.1e-38 | 7.1e-38 | 0.982 | 0.974 |

The node binding is the multiplicative pair `u = (c @ W_A_S) ⊙ (qc @ W_A_C) /
sqrt(D_A)`. All four binding matrices are denormal-zero (`~7e-38`, i.e. underflowed
through repeated shrinkage, not bit-zeros) in the route-2 soup state, so the coordinate
cannot influence the prediction, and both intervention probes are exactly 0.0.

Two facts from the frozen artifacts bound the explanation:

1. **It is not a wiring bug.** At initialization (untrained model, correctness stage)
   zeroing the joint block shifts predictions by 8.035e-05 (> the 1e-6 gate) and the
   joint binding rows receive non-zero gradients (1.6e-05). The channel is live before
   training; it is the *training* that removes it.
2. **This attractor is already in the record.** `claim-e2e-dictenv-jointbond-v1-key-level-fusion-inert-denormal-collapse-20260930`
   documents that the frozen parent's *node* binding sits at the same float32 denormal
   floor in the pre-existing Sem108 soup (`W_A_C`/`W_A_S`/`node_encoder.0` = 7.05e-38,
   `G_node = 0.0`); the read-only checkpoint check above confirms exactly that for every
   route-1 / RoleCorr-v1 soup state, where only the *edge* pair stayed alive
   (`W_E_S` 0.541–0.982) and carried the recorded block sensitivity. The route-2-specific
   observation is therefore narrow but real: **with the joint coordinate the edge pair
   also reaches the floor**, so both coordinate paths are inert at the endpoint
   (`W_E_S`/`W_E_C` = 7.0e-38 here versus 0.5–1.0 in the sibling runs).

Mechanism (diagnostic, not a new candidate): a 6-epoch / 2048-molecule probe with the
real frozen dictionary and the official factory shows the multiplicative pair shrinking
by a near-constant factor per epoch — node pair ≈ ×0.85/epoch, edge pair ≈ ×0.975/epoch
(e.g. `|W_A_S|max` 8.71e-02 → 3.60e-02 over epochs 1–6, `|W_A_C|max` 9.17e-02 →
5.00e-02). What the prior diagnostic round established is important for reading this:
`claim-e2e-dictenv-jointbond-decay-diagnostic-v1-frozen-parent-branch-trainable-decay-slows-not-blocks-20260930`
(`WD_SUPPRESSES_BUT_BOTH_LEARN`) showed under a frozen parent, with a fixed input and a
fresh *live* branch, that weight decay only slows learning (coupled L2 vs 0.0 differ by
3 %) and that sub-resolution gradients are not by themselves float32 precision failures.
So the correct reading here is not "Adam-L2 annihilates every channel" but: the joint
coordinate's edge signal was not strong enough to hold the multiplicative pair away from
the decay attractor that the node pair already sits in — the same failure mode, now on
the only channel route 1 had left alive. Either way it is *not* evidence that the joint
dictionary was given a fair structural role.

Pre-registration gap (recorded, not repaired): the prior round's mandated next shape
included mandatory endpoint liveness gates (`|w|max > 1e-3`, intervention delta ≥ 0.003)
before any task interpretation; this round's correctness gates only checked liveness
*before* training, and the endpoint liveness was established post hoc by the analysis
above. A future round must make the liveness check an acceptance gate, otherwise an
inert-channel trajectory is indistinguishable from a fair test at reporting time.

Consequences for the record (scope discipline):

* The candidate's absolute band is `stop`, exactly as pre-registered, from the soup
  valid MAE 0.132442. This is a valid record of the frozen protocol's outcome.
* The zero/shuffle probes may be cited **only** as a channel-usage diagnostic that
  explains the number; per pre-registration they cannot be used to claim "the joint
  dictionary is useless", "sparse is worse than PCA", or any causal/incremental
  statement, and they cannot be converted into a rescue attempt.
* Because the coordinate path is inert, this run does **not** measure the performance
  ceiling of the joint dictionary route. A future round that wants that number must
  address channel usage with its own pre-registered mechanism (e.g. a non-multiplicative
  binding), which is outside this round's authorization.

## 4. Training health and gates

* 320/320 epochs, no truncation, no early stop (0.619 → 0.135 valid).
* Correctness: all gates passed (`correctness_all_passed: true`), including
  `l0_max = 12`, coordinate shift under block zeroing 1.125, prediction shift 8.035e-05,
  re-label tolerance 1.19e-07, batch-composition offset 7.15e-07 (frozen tolerance 1e-5,
  identical sparse support — BLAS-level float rounding, documented in the round tests).
* Smoke: passed, best 0.842537 @ 3 epochs (2048/512), 1.62 s/epoch on the smoke subset.
* Full training: 8.224 s/epoch, wall 2631.6 s; screen run total 2680.8 s.
* Parameters: trainable 99469, total 135581, frozen dictionary elements 36112.
* Peak RSS: 3574165504 bytes (3.57 GB); device cpu, 8 threads.
* Split fingerprint (control plane): `58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a`;
  run fingerprint `8e47f7093bfb01a2…`; config hash `3c5d1c9ce5eb6a30…`.

Curve notes: valid MAE is monotone-improving until the last epoch (best = last); the
last-20 mean (0.14503) is above the last value, so the tail has a few unsettled epochs;
the soup is dominated by the last five states. Train MAE 0.0838–0.0867 vs valid 0.1352
at the same epochs is a ~0.05 absolute generalization gap, comparable in shape to the
route-1 arms (train min 0.0797–0.0824) whose valid soups were 0.1233–0.1264 — with the
caveat that in this round the coordinate path contributes nothing to that fit.

## 5. What may / may not be claimed

May be claimed (and is):

* The frozen candidate `JOINT-SPARSE` (joint 709-D input, K48/s12 dictionary, 49-wide
  coordinate, seed 0, 320 epochs) reached **soup valid MAE 0.132442** (best single
  0.135178 @ 320) under the frozen protocol, local CPU regime → pre-registered band
  **`stop`**.
* The frozen K48/s12 dictionary's real tied-IHT(10) coding error is 0.895/0.941
  (train/valid); OMP(12) reference 0.646; 60-step IHT 0.772/0.865.
* Descriptive mechanism: under the frozen objective the coordinate binding pair decays
  to denormal-zero (the node pair is already dead in the frozen parent and in every
  route-1/RoleCorr-v1 checkpoint — see
  `claim-e2e-dictenv-jointbond-v1-key-level-fusion-inert-denormal-collapse-20260930`;
  here the edge pair, alive in the sibling runs, reaches the floor as well), so the
  zero/shuffle probes are exactly 0.0 and the endpoint reflects an inert coordinate
  channel. The prior decay diagnostic
  (`claim-e2e-dictenv-jointbond-decay-diagnostic-v1-frozen-parent-branch-trainable-decay-slows-not-blocks-20260930`)
  rules out "weight decay alone blocks learning" as a general explanation, so this is
  read as a sub-resolution task signal on this coordinate, not as a global optimizer bug.

May not be claimed (explicitly out of scope):

* No increment/causality: not "better/worse than A/B/C/Sem108/RoleCorr", not "sparse
  beats PCA", not "the joint object contributes/damages".
* No claim that the joint dictionary route's ceiling has been measured — it has not,
  because the channel was switched off.
* No claim about official ZINC test; the test split was never loaded.

## 6. Budget and provenance

| stage | wall | notes |
|---|---|---|
| prepare (scratch `20261001-131619-843f4adc`) | 5248.1 s | K-SVD 5194.1 s; scaler/caches/verify_frozen |
| screen (`20261001-144617-89099a3f`) | 2680.8 s | correctness+smoke+320-epoch train (2631.6 s) + probes + analysis |

Round budget was ~3 h of machine-facing work; the executed chain used ~2 h 12 min of
compute (prepare 87 min + screen 45 min) plus tooling/reconnaissance, i.e. inside the
planned compute envelope (the earlier 52 s/epoch observation was K-SVD contention, the
clean rate is 8.22 s/epoch). No GPU/remote was used; nothing was truncated.

Artifacts (git-ignored, present in `results/e2e_dictenv_joint709_absolute_v1/`):
`summary.json`, `run_JOINT-SPARSE.json`, `soup_JOINT-SPARSE.json`, `interventions.json`,
`correctness.json`, `smoke.json`, `verify_frozen.json`, `reconstruction.json`,
`reconstruction_extra.json`, `joint_standardizers.json`, `readout_reference_state_route2.pt`,
`checkpoints/JOINT709-JOINT-SPARSE-seed0_{soup,raw}_state.pt`, `per_molecule_errors.npz`.
Tracked in git: `REPORT.md`, `DECISION.md`, `run_chain.sh`.

## 7. Exact reproduction

```bash
# prepare (train-only scaler/cache/frozen K48/s12 dictionary)
uv run research run zinc_e2e_dictenv_joint709_absolute_v1 \
  --study zinc-context-gap --mode scratch \
  --purpose "Joint709 prepare (train-only scaler/cache/frozen K48/s12 dictionary, no test access) CPU" \
  --set runtime.device=cpu --set model.stage=prepare
# formal screen (single candidate, no control arms)
uv run research run zinc_e2e_dictenv_joint709_absolute_v1 \
  --study zinc-context-gap --mode screen \
  --purpose "Joint709 dictionary absolute-performance CPU screen" \
  --set runtime.device=cpu
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_joint709_absolute_v1.py
```

## 8. Follow-ups (not authorized in this round)

1. If the joint-dictionary question is revisited, first pre-register a mechanism that
   keeps the coordinate channel live (mandatory `|w|max > 1e-3` / intervention-delta
   liveness gates, as already required by
   `decision-e2e-dictenv-jointbond-decay-diagnostic-v1-no-branch-rescue-optimizer-attractor-next-20260930`);
   a seed-1 confirmation of this round alone would only re-measure the inert-channel
   configuration.
2. Dictionary capacity (K ≫ 48) or a block-adaptive code budget would target the
   measured 0.9 reconstruction error, but both are new candidates requiring fresh
   authorization.
3. Any statement about the route relative to Sem108/RoleCorr anchors requires matched
   controls (Sem108-only, PCA48) and multiple seeds.
