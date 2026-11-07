# PROTOCOL — `lcrc-pcba-atom-bond-ssl-pilot-v1`

**Frozen before any GPU execution of this round.** The definitions below may not
be changed after seeing any validation number; only infrastructure failures may
be repaired, and a repair may not alter the training definition.

Round id: `lcrc-pcba-atom-bond-ssl-pilot-v1`
Model under test: `A_SSL_INIT_S0` (original non-message-passing LCRC-A, seeded with
masked atom/bond self-supervised encoders).
Seed: `0`. Runs this round: **1 pretrain + 1 supervised fine-tune.**

---

## 1. Question

Does a single masked atom/bond reconstruction pretext on official-Train
structures produce an `atom_encoder` / `bond_encoder` pair that, transferred as
the *initialisation* of the original no-message-passing LCRC-A, improves epoch-40
official Validation macro AP relative to the historical seed-0 A0 run
(`0.2463560956`)?

This is an **initialisation experiment**. No SSL module is added to the
downstream architecture, and no message passing, path feature, auxiliary loss or
architecture change is introduced.

## 2. Hard constraints

| # | Constraint |
|---|-----------|
| 1 | seed `0` only; no multi-seed confirmation this round |
| 2 | exactly **one** SSL pretrain (10,000 updates, no mask-ratio / loss-weight / LR search) |
| 3 | exactly **one** downstream 40-epoch supervised run |
| 4 | final inference parameters must remain **287,644** (245,184 base + 42,460 new) |
| 5 | the historical A0 run is **not** retrained and not modified |
| 6 | no A1/A2/A3, no G, no V; no other arms |
| 7 | official **Test labels are never read or scored** (they are NaN in the frozen artifact) |
| 8 | no early stopping, no best-epoch selection; epoch 40 is the single comparison point |
| 9 | the pretrain recipe, mask ratio, gate and decision bands below are frozen |

Historical reference (read-only, **not** a same-round paired control):
`tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A`
— seed 0, 40 epochs, Validation macro AP `0.24635609609821457`, params `287644`,
Train AP40 `0.4196570175055804`, commit `b1b917dc4f72`.

## 3. Data

* Cache: `tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1/cache/pcba_v1`
  (`lipo_v1`, 51 atom features, 13 bond features, 9 scalars; 437,929 molecules,
  11,373,137 atoms, 12,309,186 bonds; `labels_present: false`). Read-only memory map.
* Splits: official OGB Train `350,343` / Valid `43,793` / Test `43,793`.
* **SSL probe** (evaluation only, never used for parameter updates): 4,096 Train
  molecules, `sorted by SHA256("pcba-ssl-probe-v1|" + decimal row id)`, mask
  `visit_index = 0` fixed.
* **SSL fit** = official Train minus probe = `346,247` molecules.
* Downstream: the frozen supervised recipe and the frozen seed-0 40-epoch
  batch-order plan `06f43729…`.
* Pretraining inputs are **structure only**: `atom_feat`, `bond_feat`,
  `bond_index` (plus `atom_ptr` / `bond_ptr` / `atom_type` / `bond_type` as
  targets/labels of the pretext task). The following cache arrays are never
  opened by the pretraining path: `atom_scalar`, `global`, `nbr_code`, `nbr_ptr`,
  `path_*`, `n_components`, `n_junctions`, `n_ring_systems`. The assay label
  artifact `pcba_labels_trainvalid.npy` is **never opened by any pretraining
  code path**; its path is recorded only so the runtime audit can prove this.

## 4. Pretext task (masked atom / bond reconstruction)

* Targets: `atom_type` ∈ 16 classes (CE), physical `bond_type` ∈ 5 classes (CE).
* Mask ratio: 15 % atoms and 15 % bonds **independently**, computed per molecule
  as `round_half_up(3/20 · n)` with a minimum of 1 when `n ≥ 1`.
* Mask token: the full 51-d atom feature vector / 13-d bond feature vector of the
  masked item is zeroed. For a masked bond, the full 51-d features of **both
  endpoints** are zeroed as well. Nothing else is hidden.
* Independent atom and bond mask streams:
  `SHA256(tag | seed | molecule_id | visit_index) → 128-bit seed`,
  `tag ∈ {"…|atom", "…|bond"}`; visit index grows per re-visit of a molecule.
  Mask RNG, data-shuffle RNG (`20261107`) and head-init RNG (`20261108`) are
  mutually isolated; encoder weights are **not** redrawn (they are the historical
  seed-0 scratch initialisation).
* Encoders keep the historical geometry: atom MLP `51→128→96`, bond MLP
  `3·96+13 = 301→160→96`. Contexts: atom `[a_v, mean_b(v), mean_a(v)]` (288-d),
  bond `[b_e, ctx_u+ctx_v, |ctx_u−ctx_v|]` (480-d) with leave-one-out means.
* Temporary heads: `288→160→16` and `480→160→5`. They are discarded after
  pretraining and never transferred.
* No message passing: one static atom encode and one static bond encode per view,
  static incidence mean aggregation only, no hidden-state iteration.
* Loss: `CE(masked atoms) + CE(masked bonds)`, equal weight (frozen).

### 4.1 Frozen pretrain recipe

```text
seed                = 0
updates             = 10000
batch_size          = 128
optimizer           = Adam
learning_rate       = 1e-3
weight_decay        = 1e-5
betas               = (0.9, 0.999)
eps                 = 1e-8
gradient_clip_norm  = 5.0
dropout             = 0
precision           = FP32
atom_mask_ratio     = 0.15
bond_mask_ratio     = 0.15
loss                = CE(atom) + CE(bond)
eval_every          = 2000 updates (Train loss, Probe atom/bond CE + accuracy,
                      class-marginal prior CE, counts, grad norms, step time)
export              = exactly update 10000
early_stop          = none; a Probe curve may not select an earlier checkpoint
checkpoints         = every 1000 updates, strict resume (optimizer, RNG,
                      shuffle and mask visit-index state)
```

## 5. Audits required before the formal pretrain

Implemented in `code/ssl_audit.py` (real tensors, not module names):

1. Pretraining loader never opens the assay label artifact.
2. All SSL rows are official Train; probe and SSL-fit are disjoint.
3. Masked targets appear only in the CE, never in the model input.
4. Changing the original 51-d feature and hidden class of masked atoms, then
   re-applying the atom mask, leaves the visible input bitwise unchanged.
5. Same for the 13-d bond features and both endpoint features of masked bonds.
6. `S1/S2`, `global`, `atom_scalar` and unmasked `atom_type/bond_type` never enter
   a reconstruction head (injection test: outputs are bitwise identical).
7. Atom renumbering permutes predictions and leaves the loss unchanged.
8. Swapping the endpoints of an undirected bond leaves the bond prediction unchanged.
9. No message passing (distance-2 perturbation has exactly zero effect; exactly
   two static encodes per forward).
10. Both encoders and both heads receive non-zero, finite gradients.
11. The original LCRC-A input dimensions and state-dict keys are unchanged.
12. Mask RNG, batch order, save/load and cache indexing are reproducible.
13. GPU smoke: no non-finite loss, gradient or logits; step time is measured to
    project the full phase against the remaining budget.

The formal pretrain may only start when every check passes
(`analysis/mask_leakage_audit.json → all_passed = true`).

## 6. Pretext quality gate (frozen)

After exactly 10,000 updates:

* Probe atom CE ≤ SSL-fit atom class-marginal prior CE − **0.05 nats**, and
* Probe bond CE ≤ SSL-fit bond class-marginal prior CE − **0.05 nats**, and
* both encoders differ from their seed-0 scratch state.

Otherwise the label is `PRETEXT_INVALID_OR_UNLEARNED` and the 40-epoch supervised
budget is **not** spent. No loss-weight, LR or mask-ratio rescue is permitted.

Passing the gate only shows that the pretext is learnable; it says nothing about
downstream benefit.

## 7. Transfer into the original A

1. Build the **complete original** LCRC-A with the historical builder at seed 0
   and require its base-initialisation fingerprint to equal the historical one.
2. Overwrite **only** `atom_encoder.*` and `bond_encoder.*`.
3. Keep the historical seed-0 scratch initialisation for `local.*`, `f.*`,
   `rho.*`, `readout.*`.
4. Never transfer the two reconstruction heads; never transfer SSL optimizer state.
5. All original A parameters are trainable downstream (encoders are not frozen).
6. Audit: overwritten keys + per-tensor SHA256, parameter change vs scratch,
   per-tensor equality of all non-transferred modules, `[8, 128]` forward probe,
   `287644` parameters, and inference without any pretext head.

## 8. The single downstream run

```text
model_name      = A_SSL_INIT_S0
seed            = 0
dataset         = ogbg-molpcba
split           = official Train/Valid
epochs          = 40
batch_size      = 128
optimizer       = AdamW (freshly created)
peak_lr         = 3e-4
warmup          = first epoch, 3e-5 -> 3e-4
scheduler       = historical per-step cosine to 1e-5 at epoch 40
weight_decay    = 1e-3
betas           = (0.9, 0.999)
eps             = 1e-8
dropout         = 0.10
gradient_clip   = 5.0
precision       = FP32
loss            = original observed-entry masked BCE
early_stopping  = false
batch order     = historical seed-0 plan 06f43729… (verified bitwise)
eval epochs     = 10, 20, 30, 40
primary metric  = epoch-40 official OGB Validation macro AP
```

Evaluation uses the historical `pc_metrics.validation_report` /
`Evaluator("ogbg-molpcba")` with the unchanged per-task NaN-mask rule. Only
diagnostics may look at epochs 10/20/30; epoch 40 is never replaced.

## 9. Decision rule (pre-registered, exploratory)

```text
Delta_AP40 = Validation_AP40(A_SSL_INIT_S0) − 0.2463560956

Delta >= +0.005            -> PROMISING_SINGLE_SEED
+0.002 <= Delta < +0.005   -> WEAK_SIGNAL
-0.002 < Delta < +0.002    -> NO_CLEAR_GAIN
Delta <= -0.002            -> NEGATIVE_TRANSFER
```

These are resource-allocation labels, not significance tests. Mandatory caveats:
there is no same-round A0, so implementation/environment drift cannot be
excluded; pretraining adds compute, so a gain cannot be attributed to
self-supervision rather than extra compute; a well-learned pretext with no
validation gain is a valid negative-transfer result; the historical Test-facing
A0 number is not this round's target.

## 10. Budget and execution

```text
Target : <= 5 A100 GPU-hours
Hard cap: <= 7 A100 GPU-hours   (preflight, failed attempts, pretrain, fine-tune)
```

All remote work goes through `rr` (`doctor`, `deploy`, `run`, `jobs`, `status`,
`logs`, `pull`) per `skills/remote-research-runner/SKILL.md`. Preferred pool:
`res-gpu1`; GPU 0 is not taken over if it still shows external load. A phase that
cannot finish inside the remaining budget is reported as `INCOMPLETE_RESOURCE`
rather than silently shortened.

## 11. Deliverables

`PROTOCOL.md`, `README.md`, `code/`, `tests/`,
`analysis/{data_audit,mask_leakage_audit,pretrain_probe,pretrain_run,encoder_transfer_audit,supervised_epoch40,run_registry,gpu_ledger,decision}.json`,
`analysis/per_task_ap.csv`, `REPORT.md`, plus the local (git-ignored) run
artifacts under `runs/`.
