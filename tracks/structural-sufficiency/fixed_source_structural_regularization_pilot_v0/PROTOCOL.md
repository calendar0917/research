# FSRP v0 — FIXED1 structural-unit learning-constraint short validation

Round ID: `fixed-source-structural-regularization-pilot-v0`

## Question

With FIXED1 inference computation kept complete, does randomly masking real
structural computation units **during training only** improve standard
graph-level prediction generalization?

## Common start (read-only)

* `direct_structural_computation_recurrence_v0/runs/FIXED1_seed0/checkpoint_last.pt`
  (FIXED1 seed 0, epoch 25, global step 68,450, complete AdamW / RNG / order-plan
  / LR-schedule resume state; recorded Valid AP 0.27584552136653284).
* Model, optimizer moments, RNG, original 40-epoch order plan and the original
  40-epoch warmup+cosine LR schedule are restored. The horizon is continued
  25 -> 33 in this round's isolated trainer; the schedule is **not** rescaled.
* Both arms fork from the identical state.

## Arms (8 epochs each, epochs 26–33)

| arm | training computation |
|---|---|
| `BASE_CONT` | original FIXED1, no added constraint |
| `STRUCT_DROP` | FIXED1 + structural-unit message dropout, q = 0.10 |

`STRUCT_DROP` per training batch draws one Bernoulli keep mask per **physical
undirected bond** with an independent `torch.Generator` (seed 20261126), then

```python
keep = (rand(n_physical_bonds, generator=mask_rng) >= 0.10).float() / 0.90
keep_directed = torch.cat([keep, keep])          # exact directed_bonds() order
p = F.silu(z_source[src] + edge_msg) * keep_directed[:, None]
m = scatter_sum(p, dst)
```

* one draw per batch, never for eval;
* both directions of a bond share the draw; both FIXED1 blocks share the same
  `keep_directed` tensor;
* no new parameter, no auxiliary loss, no extra head, no change to S1/S2,
  degree, descriptors, `post_local`, or the h0 dropout (0.10);
* the training RNG of the model is untouched (separate generator);
* eval mode calls the inherited FIXED1 code path: no mask, no scaling.

## Recipe (inherited, unchanged)

batch 128, masked BCE, AdamW (moments restored), weight decay 1e-3, gradient
clip 5, fp32, original 40-epoch order plan, original 40-epoch cosine schedule
prefix. No Test run, no extra seed, no distillation, no ensemble.

## Evaluation

* complete official Valid every epoch (macro AP, official OGB evaluator; masked BCE);
* complete official Train at epochs 25, 29, 33 and at each arm's best checkpoint;
* epoch 25 start retained as an independent common baseline;
* report best (26–33), endpoint 33, last-3 mean, Train AP/BCE, Valid BCE,
  high-support tasks (Valid positive >= 20 and negative >= 20), per-task
  improved/degraded counts, paired task bootstrap (B = 10,000, seed 20261126),
  sensitivity after removing the three largest-|delta| tasks, and Train–Valid
  gap on the same checkpoint over commonly scorable tasks.

## Necessary checks

C1 common state/optimizer/RNG/LR/order plan; C2 identical initial eval logits
and Valid AP; C3 q=0 forward+gradient equality with FIXED1; C4 mask shared by
both directions and both blocks; C5 realised drop fraction ~10% with exact
bond mapping; C6 no mask in eval and eval arithmetic equals FIXED1; C7 both arms
1,128,468 parameters; C8 complete resume state and no Test access.

## Decision labels

* `STRUCTURAL_REGULARIZATION_PROMISING` — STRUCT_DROP best Valid AP beats both
  BASE_CONT and the epoch-25 start by >= 0.003, and endpoint, last-3 and
  high-support task AP are not worse than BASE_CONT.
* `FIT_REDUCTION_WITHOUT_GAIN` — training fit falls, Valid does not gain.
* `NO_CLEAR_GAIN` — positive practical threshold not met.
* `INCONCLUSIVE` — curves / task support / budget create material uncertainty.

Labels are short-horizon exploration decisions only. No plain message-dropout
control is run, so a positive result could not be attributed to the structural
organization of the mask. If no positive result: stop this regularization line;
do not add q, epochs, seeds or arms.

## Resource

GPU-hour target 0.9, hard limit 1.1; wall-clock hard limit 75 minutes including
implementation, queueing, checks, training and report. Two idle GPUs -> both
arms in parallel; otherwise sequential. Formal training starts only if the
timing preflight says both arms fit.
