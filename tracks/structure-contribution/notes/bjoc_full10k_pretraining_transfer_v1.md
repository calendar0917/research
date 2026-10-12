# cscl-bjoc-full10k-pretraining-transfer-v1 — design / pre-registration note

**Status (at freeze):** local CPU Phase A complete (official-train-10k structure
cache, train-only global standardiser, consistency audit, mask-token scan,
200/200 atom + 200/200 bond hidden-truth swaps, 100/100 rebuild cases, 10/10
correctness gates, 16 fast tests, full-10k B reuse audit, training manifest
frozen).  No GPU trajectory has run.  **Official valid is a previously exposed
development set; its labels are read exactly once after every prediction file is
hashed.  Official test is never loaded.**

Protocol: `configs/bjoc_full10k_pretraining_transfer_v1.yaml`.

## 1. Why this round exists

`cscl-bjoc-masked-pretraining-v1` showed that strictly leak-free masked
atom/bond pretraining of the three BJOC encoders improves locked-dev raw-y MAE
on a *new internal 8k fit split* (`P − B = -0.011984`, CI95
`[-0.017530, -0.005974]`, both seeds negative), and that the gain survives a
50%-extra-supervision scratch control (`P − C = -0.007713`).  It left the most
decision-relevant transfer question open: **does the same frozen pretraining
still help when the downstream model sees the complete official train 10000**
instead of 8k?  That is H1 of the current brief, and it is the only question this
round must answer before any new architecture is purchased.

The historical full-10k BJOC-Null official-test two-seed mean is `0.092233`
against a target band of 0.06–0.07.  This round runs on the official train split
and the exposed official-valid split only and makes **no extrapolation to the
official test**.

## 2. Design in one paragraph

No new internal split.  The complete official train 10000 is used for (a) SSL
pretraining on raw structures only and (b) raw-y supervised training.  Three
arms, all BJOC-Null at 413,505 parameters, raw-y L1, Adam 1e-3 / wd 1e-5 /
batch 128 / clip 5, 240 epochs = 79 steps/epoch = 18,960 updates, last-5 soup
(parameters mean of epochs 236–240):

| arm | encoder init | encoder LR | supervised updates |
|---|---|---|---:|
| **B** | scratch (historical full-10k checkpoints reused only if the frozen audit proves equivalence) | 1e-3 | 18,960 |
| **P** | frozen SSL encoder state; composer/reader from the same scratch init as B | 1e-3 | 18,960 |
| **Q** (conditional) | same pretrained encoders as P | 3e-4 (composer/reader 1e-3) | 18,960 |

The SSL recipe is copied unchanged from the validated round: 15% atoms + 15%
physical bonds, mask tokens atom 27 / bond 0 (verified free on the full 10k),
full rebuild of every BJOC field from the masked raw graph, `CE_atom + CE_bond`
(1.0/1.0), 7,560 updates at batch 128, last-step encoder state.  Because the
update count is fixed, the mean exposure per molecule drops from ~121 (8k fit)
to ~96.8 (10k); this is recorded but not searched.

## 3. Frozen reuse audit for B (Phase B)

The historical `bjoc_null_full10k_s0/s1_soup_state.pt` are reused **iff all**
hold (computed pre-GPU, `reuse_audit.json`):

1. the 10,000-object cache is bit-identical to the historical
   `bjoc_objects_v1.pt` (all four feature tensors and all four index tensors);
2. the train-only global standardiser is bit-identical;
3. the freshly rebuilt init-state hash equals the historical recorded
   `init_state_sha256` (seed 0 `d06d5e16…`, seed 1 `c79e6b02…`);
4. the recipe matches exactly (optimizer, LR, wd, clip, batch, epochs, last-5
   soup, shuffle seed `seed+101`, 413,505 params);
5. the frozen soup state replays the historical official-valid predictions
   bit-identically on the frozen structure cache.

All five held locally; B would contribute **zero new GPU training**.  **Amendment before any GPU run:** on the remote compute checkout the historical terminal-round prediction files are not part of the deployed revision, so the replay condition cannot be recomputed there; the audit therefore conservatively returns `RETRAIN_B` on the cluster.  The GPU chains retrain B seed 0/1 with the identical frozen recipe on the same GPU as P (adding ~16 GPU-min), which is the stronger matched-baseline guarantee; the locally verified historical replay (max `|dpred| = 0.0`) is recorded as supporting evidence only and is never used for selection.

## 4. Phase D representation diagnostics and the Q purchase gate

Before any valid label exists, the three encoder states (scratch init, SSL,
P fine-tuned) are compared on 2,000 fixed official-train structures:
parameter relative move per encoder, embedding scale, effective rank, and
linear CKA (SSL vs P) for atom / bond / junction representations.

**Q is purchased only if** (a) junction linear CKA (SSL → P) < 0.75 for at
least one seed, **or** (b) junction effective rank drops > 25% from SSL to P
with ≥ 4/5 repeated 1,000-molecule draws above 25% for at least one seed.
Otherwise `NO_ENCODER_PRESERVATION_PURCHASE`, Q is never trained, and no other
candidate is substituted.  A negative CKA / rank drop is a resource gate, not
evidence of harmful forgetting.

## 5. Evaluation discipline (Phase F)

1. Finish every authorised training run.
2. Run every frozen model forward on the official-valid **structures**; no label
   is read anywhere in the forward path.
3. Hash every prediction file into `prediction_freeze.json`; copy nothing after.
4. Read the official-valid labels **exactly once** in `score`.
5. Compute the pre-registered metrics; no further training or selection.

Primary metric: official-valid raw-y MAE as the **seed-mean of the two
independent single-model MAEs** (never an ensemble).  Secondaries: per-seed MAE
and RMSE, median/p90/p95/p99/max, ordinary (`y > -3`) and tail (`y <= -3`),
top-1/5/10-error-removed deltas, per-molecule paired improvement, 5,000-resample
molecule-level paired bootstrap, and the `valid:0172` decomposition.

## 6. Frozen decision cases

* `FULL10K_PRETRAINING_GAIN_CONFIRMED_ON_DEV` — P−B seed-mean ≤ −0.005, both
  seeds negative, ordinary and top-10-removed and all-rows-minus-0172 negative,
  bootstrap CI95 upper < 0, no protocol deviation.
* `FULL10K_PRETRAINING_GAIN_WEAK_OR_UNSTABLE` — mixed or extreme-row-driven.
* `FULL10K_PRETRAINING_NO_GAIN` — no consistent gain.
* `ENCODER_LR_ADAPTATION_USEFUL` / `ENCODER_LR_ADAPTATION_NOT_SUPPORTED` — Q,
  if purchased, must be negative on both seeds, seed-mean ≤ −0.003, ordinary
  improving, not extreme-row driven, without an RMSE blow-up.
* `BULK_GAIN_WITH_TAIL_COST` — a disclosed real trade-off.
* `INCONCLUSIVE_OR_BLOCKED` — engineering / gate / data blockage.

## 7. Budget

Wall ≤ 405 min, one GPU at a time, ≤ 3 GPU-hours.  Phase A is CPU-first; no
internal-split search; no extra data; no test access.

## 8. Results (completed 2026-10-13)

Frozen decision **`FULL10K_PRETRAINING_GAIN_CONFIRMED_ON_DEV`** with flags
`BULK_GAIN_WITH_TAIL_COST`, `ENCODER_LR_ADAPTATION_NOT_SUPPORTED`.

* Valid single-model MAE (two-seed mean): B `0.099202` → P `0.083549`
  (`P−B = −0.015653`, CI95 `[-0.021051, -0.009904]`; per seed `−0.015387`
  / `−0.015919`; ordinary `−0.017391`; top-10-removed `−0.015980`;
  `valid:0172`-removed `−0.016400`).  B was retrained in-round on the same
  GPU with the identical frozen recipe because the remote reuse audit could
  not recompute the historical replay (the amendment is recorded above).
* Pretext (7,560 updates, 96.8 exposures/molecule): atom CE `0.2126/0.2125`,
  acc `0.9230/0.9251`; bond CE `0.0193/0.0159`, acc `0.9939/0.9949`.
* Q was purchased (junction linear CKA SSL→P `0.1015/0.1268 < 0.75`) and is
  harmful: `Q−P = +0.019484` (CI95 `[+0.005177, +0.044905]`), tail
  `+0.182`, RMSE `0.541` vs `0.230`, `valid:0172` predicted at `−7.95/−0.27`.
* Tail is the binding residual: `y≤−3` (n=65) `P−B = +0.009341` while the
  ordinary 935 molecules improve; 61/100 of B's worst rows persist.  A
  count-conditioned whole-molecule scale calibration (Route B) is the single
  recommended next route; the sample-efficiency/external-data route (A) is
  rejected as the next step.

Full details in `results/bjoc_full10k_pretraining_transfer_v1/REPORT.md`.
