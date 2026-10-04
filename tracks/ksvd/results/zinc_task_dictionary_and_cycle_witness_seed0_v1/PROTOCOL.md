# PROTOCOL — zinc-task-dictionary-and-cycle-witness-seed0-v1

Single seed (seed 0). Two fixed GPU arms plus a train-only CPU witness task.
No official-valid / official-test access. The main question is a *matched
current-implementation* comparison, not a claim about all dictionaries or all
MLPs.

## 1. Questions and completion criteria

**Main (GPU).** On the compressed unstructured-binding architecture, does the
current shared *task dictionary* bridge beat a parameter-matched ordinary local
MLP bridge on a new model-internal fit/dev fold?

**CPU.** What exactly are the samples that share the same `topology25` input but
have different ring targets? Is it distinct graphs compressed to the same
representation, a node-order-sensitive cycle helper, or insufficient evidence?
List the inspectable molecules, graphs and targets.

Completion criterion for the round: both questions get a verifiable answer.
An `INCONCLUSIVE` or `UNRESOLVED_PROVENANCE` outcome is a valid answer and is
not "fixed" by adding arms.

## 2. Budget and scope

| item | limit / fixed setting |
| --- | --- |
| wall clock | 180 min from first tool call; stop compute at 150 min |
| formal training | exactly two arms `D` and `M`, 240 epochs each, seed 0 |
| GPU | <= 1.2 GPU-hours total, at most two 1-GPU jobs in parallel |
| CPU | explicit <= 8 threads; cycle witness + old-result clarifications <= 45 min |
| remote | `rr`, host `res-2`, pool `res2-cu124` |
| data | official-train 10000 rows only; new fixed 8000/2000 internal fold |
| precision | FP32, no AMP / DDP; same node and software regime preferred |
| search | no hyper-parameter search, no third arm, no second seed |

If a full 240-epoch trajectory cannot be run inside the budget, formal training
is not started. Short trajectories are never reported as formal results.

## 3. New fold (generated once, frozen)

```python
perm = np.random.default_rng(20261004).permutation(10000)
fit_idx = np.sort(perm[:8000])
dev_idx = np.sort(perm[8000:])
```

The train-only cache is loaded in stable positional order. The fold is stored
with stable ids and SHA-256 of both index arrays; overlap is required to be
empty and the union to cover all 10000 rows. It is not redrawn for any reason.

This is a **new model-internal training fold**, not researcher-unseen data: the
old model may have trained on these dev rows, so warm starts, distillation and
old fitted scalers are forbidden.

## 4. Fresh-only rule

Both arms are constructed from `build_deploy_model(state=None)` and contain no
learned tensor. No `S_M`, `A0`, `C`, `T`, `O`, `H`, `Y` or old bridge state is
loaded for the main models. Old states appear only in the CPU clarification.
The 110-d first fusion layer is a new seed-0 initialization; no folded internal
bias from a trained slot path is used.

## 5. Arms

Common input/modules: Sem108 (atom-shell 3x28 + bond-shellpair 6x4) and size2
(110-d), fusion `110 -> 342 -> 144` with the current Full activations, static
local unary/pair aggregation, the existing distance/relation handling, global
C6, topology25 and reader. The bridge is inserted exactly once after fusion,
before every unary/pair consumer.

| arm | coding | parameters |
| --- | --- | --- |
| `D` | current task dictionary: `rho = sqrt(mean(h^2)+1e-12)`, `x = h/rho`, `Dbar = column_normalize(D_L)`, 16-step ISTA (`lambda1=0.05`, `lambda2=0.01`), `E = rho * alpha @ V_L` | `D_L[144,288] + V_L[288,144]` = 82,944 |
| `M` | same `rho/x`, `E = rho * W2 SiLU(W1 x)`, both biases off | `W1[288,144] + W2[144,288]` = 82,944 |

`M` weights start from the same fresh frame: `W1 = D_L_init.T`,
`W2 = V_L_init.T`. No column normalization, residual, LayerNorm, dropout or
bias is added to `M`. Both arms start from item-for-item identical non-bridge
parameters and buffers (verified per tensor by SHA-256). Initial function
inequality is expected and is not claimed away.

Expected accounting: body 184,667; bridge 82,944; total 267,611.

## 6. Training protocol

* seed 0; 240 epochs; batch 128; Adam (coupled L2), `lr=1e-3`, `wd=1e-5`;
  global gradient clip 5.0; loss exactly mean `L1(y)`; no scheduler.
* shared batch schedule generated before training from
  `torch.Generator(seed=101)` over the 8000 fit rows; SHA-256 recorded.
* true step count expectation: `240 * ceil(8000/128) = 15,120` per arm.
* epoch 236-240 parameter soup, no dev-based member selection; init / last /
  raw-soup states and the full curve are saved.
* one calibration per arm: `b = median(y - p_raw)` on fit in eval mode with the
  soup state; `p_cal = p_raw + b`; raw states never contain `b`.
* no dev early stopping, no extension of the worse arm, no optimizer/loss
  change, no extra MLP variant.
* the round does not enter any 10k confirmation and does not open
  official-valid/test.

## 7. Endpoints, gains and gate

Primary endpoint: new-dev G0 calibrated MAE. Reported in parallel: overall raw
and calibrated MAE, other severity groups, fit/dev predictions per row.

Gain is always `MAE(M) - MAE(D)`; positive means `D` is better.

Pre-registered categories:

| category | definition |
| --- | --- |
| `D_G0_SUPPORT` | G0 cal gain >= 0.003 and CI lower > 0; G0 raw gain > 0; overall cal gain >= -0.001 |
| `M_G0_SUPPORT` | G0 cal gain <= -0.003 and CI upper < 0; G0 raw gain < 0; overall cal gain <= +0.001 |
| `LOCAL_EQUIVALENCE` | both G0 and overall cal CIs fully inside [-0.003, +0.003]; no raw >= 0.003 effect in the opposite direction |
| `TRADEOFF` | one arm improves G0 while overall cal worsens by > 0.001 |
| `INCONCLUSIVE` | anything else, including close point values with wide CIs |
| `INVALID` | data / function / parameter / training / replay contract failure; no method conclusion |

Paired bootstrap: 1000 resamples, seed 20261004, shared per-resample graph
indices for G0 and overall; CIs are single-seed, current-fold descriptive
intervals.

Sensitivity (pre-registered, no gate change): delete the one dev row with the
largest `|err_D_cal| + |err_M_cal|` and report the gains on the rest.

Overall calibrated improvement >= 0.003 is labelled separately, with the raw
direction, and does not by itself support the dictionary mechanism.

## 8. CPU witness task

Uses the old CPU head's actual float32 `T25` input and the original 8000 fit /
2000 internal dev indices to reproduce the old numbers; named separately from
the new fold. Train-only.

1. Correct the old statistics: report repeated-class and conflicting-class
   counts/fractions separately; `global_min_l1` (all-fit class medians, overall
   L1), `global_opt_group_cost` (the same output per group), `group_only_min_l1`
   (per-group class medians), and the severity-weighted descriptive bound.
2. Reproduce the old ~0.0060697 and ~5.54946 exactly and state what each is.
3. Enumerate every conflicting class with members, stable ids, k/c values,
   `T25` values, audit cycle scores, and the two rows with maximum c difference
   per class. List all old-fit `k<=-3` rows and old internal-dev tail rows with
   the saved `q_U`/`q_B` predictions and errors.
4. Graph checks: node/edge counts, adjacency and attributes, component/ring
   facts, topology-only and typed isomorphism for each witness pair; save
   adjacency/label JSON/CSV and static figures.
5. Node renumbering: 16 permutations with seed 20261004, recompute the repo
   cycle helper under permuted node insertion order, record min/max/distinct
   and the basis; distinguish the repo diagnostic helper from the upstream
   official label process.
6. Classification per witness: `INPUT_ALIASING_WITNESS`,
   `ORDER_DEPENDENT_HELPER`, `INPUT_EQUIVALENT_TARGET_CONFLICT`,
   `UNRESOLVED_PROVENANCE`. One witness may support multiple local
   explanations; the classes are not collapsed into one global cause.

The CPU task may analyse the old internal dev but never changes the main fold
or configuration.

## 9. Stop / failure rules

Engineering failures (shape, device, replay, contract) may be fixed and the
code re-frozen; they are never an excuse to rerun a *different* configuration
after seeing scores. If the server is unavailable, only the CPU task is
delivered and the main comparison is truthfully marked not executed.
