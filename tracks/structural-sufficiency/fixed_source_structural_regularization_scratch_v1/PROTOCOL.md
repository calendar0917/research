# FSRP-SCRATCH v1 — structural-unit regularization applied from training step 0

Round ID: `fixed-source-structural-regularization-scratch-v1`
Seed 0 · PCBA official Train/Valid · Test never accessed · single GPU, one training unit.

## Question

The late-phase continuation pilot (`fixed-source-structural-regularization-pilot-v0`)
applied structural-unit message dropout only from epoch 25 onwards. Its matched-arm
gain (+0.00419 over `BASE_CONT`) was mostly avoidance of continuation decay; the gain
over the frozen epoch-25 start was +0.00172, below the pre-registered +0.003.

This round is a **new, independent hypothesis**: if the structural-unit constraint
acts from the first training step, does it shape the formation of the structural
representation and the computed function so that the resulting model beats the
original FIXED1 25-epoch model in absolute predictive performance?

No architecture work, no search.

## Model

Only `STRUCT_DROP_SCRATCH` is trained.

* Architecture: the unmodified DSCR `FIXED1` model (inherited read-only), i.e.
  `z1 = LN1(h0); m1 = Σ_j SiLU(z1[j] + e_ji); h1 = h0 + FFN1(z1, m1)`
  `z2 = LN2(h1); zs = LN2(h0); m2 = Σ_j SiLU(zs[j] + e_ji); h2 = h1 + FFN2(z2, m2)`.
* Initialisation: the **official FIXED1 seed-0 constructor state**
  (`ds_model.build_ds_model(..., arm="FIXED1", seed=0)`), fingerprints
  `initial_state_sha256 = 3d313189…50e5c` and `new_state_sha256 = f7c24ee4…1177`.
  No trained checkpoint is loaded; all parameters are trained end-to-end.
* Parameters: exactly 1,128,468 (audited per tensor group).
* Training-time constraint (inherited verbatim from `sr_model.py`):
  * `q = 0.10`, one Bernoulli draw per **real physical undirected bond** per batch;
  * both directions and both FIXED1 blocks use the *same* `keep_directed` tensor;
  * inverted-dropout scaling `1/(1-q)`;
  * independent `torch.Generator`, `mask_seed = 20261126`, restored in every checkpoint;
  * original `h0` dropout stays at 0.10;
  * **active from training step 0** — no warmup mask schedule, no q change, no change
    of mask grain;
  * inference is exactly the original FIXED1 computation (no mask, no scaling).

## Training

Complete 25 epochs of the frozen recipe:

* batch 128, AdamW (betas 0.9/0.999, eps 1e-8, weight decay 1e-3), **freshly
  initialised optimizer**, gradient clip 5, fp32, masked BCE;
* the frozen 40-epoch warmup+cosine schedule *prefix* only
  (`lr = scheduled_lr(step, phase=3)`, steps 0..68,449); the cosine is not rescaled;
* the frozen seed-0 order plan, epochs 1–25
  (global `06f4372…`, 25-epoch prefix `8c85725…`), identical to historical FIXED1;
* full official Validation AP/BCE every epoch;
* complete official Train AP/BCE at epochs 12, 20, 25 and for the best checkpoint;
* best (strict improvement on Valid AP) and last checkpoints saved with optimizer,
  RNG and mask-RNG state;
* no change of training length, no reaction to intermediate results.

## Correctness gates (must all pass before the formal run)

Inherited C1–C8 logic plus this round's specific confirmations:

| id | gate |
|---|---|
| C1 | initial parameters **and buffers** bitwise equal to the FIXED1 seed-0 constructor, 71 state tensors, fingerprint `3d313189…` |
| C2 | initial eval-mode logits exactly equal the FIXED1 initialisation (CPU, exact 0) |
| C3 | optimizer genuinely from scratch: no AdamW state entries, warmup-start LR 3e-5, step-0 scheduled LR |
| C4 | mask drawn on the first training forward (one draw per step), never in eval |
| C5 | the mask draw does not advance the model's global RNG stream (exact state compare) |
| C6 | 25-epoch plan = historical FIXED1 (global + prefix hashes, 2,738 steps/epoch) |
| C7 | eval-mode arithmetic exactly FIXED1 (no mask, no scaling) |
| C8 | Test never loaded/scored; no Test labels visible |
| C0a | `q = 0` reproduces FIXED1 forward, loss and every gradient entry exactly (CPU) |
| C0b | one physical-bond mask shared by both directions and both blocks; length = directed edges |
| C0c | realised drop fraction ≈ 0.10 inside a 3-sigma binomial band |
| T1 | timing preflight projects the 25-epoch run inside the remaining wall clock and the 1.3 GPU-hour cap |

## Reference (read-only, not retrained)

Historical FIXED1 seed-0, first 25 epochs of the same frozen recipe:
best/endpoint Valid AP 0.27584698 / 0.27584552, last-5 mean 0.27335849,
best Train AP 0.59953249, high-support (≥20 pos & ≥20 neg, 107 tasks) 0.29848664.
Values are re-read from
`direct_structural_computation_recurrence_v0/results/runs/FIXED1_seed0/` and gated
against these constants before use.

## Analysis

Complete 25-epoch Valid AP/BCE curve; best / endpoint / last-5 mean; Train AP/BCE at
the same training stages; high-support AP; per-task improved/degraded counts; paired
task bootstrap (B = 10,000, seed 20261126); sensitivity after removing the three
largest |Δ| tasks; Train–Valid gap on the same checkpoint over commonly scorable
tasks; realised mask drop fraction; actual resource use; and the three training phases
(1–12, 13–20, 21–25) to see whether the constraint slows early learning or buys a
late advantage.

Primary metric: official full-task macro Valid AP.

## Decision labels

* `SCRATCH_STRUCTURAL_REGULARIZATION_ADVANCE` — best Valid AP ≥ historical FIXED1
  best + 0.003, and endpoint, last-5 and high-support AP show no clear reverse
  movement (all ≥ −0.001).
* `WEAK_REGULARIZATION_SIGNAL` — positive late-phase or high-support signal without
  meeting the advance threshold.
* `FIT_REDUCTION_WITHOUT_GAIN` — Train fit lower than FIXED1 and no Valid gain.
* `NO_CLEAR_GAIN` / `REGRESSION` — no clear benefit, or clearly behind.

Single seed, single q: exploration decision only. Even a positive result cannot claim
that a structural mask beats ordinary message dropout (no such control here).
If there is no practical advance, the structural-dropout line stops: no q search, no
mask schedules, no longer training.

## Resource

GPU-hour target 1.0–1.2, hard cap 1.3; wall-clock hard cap 75 min including
implementation, deploy, checks, training and report; queue wait ≤ 5 min. Formal
training starts only if the timing preflight projects it fits.
