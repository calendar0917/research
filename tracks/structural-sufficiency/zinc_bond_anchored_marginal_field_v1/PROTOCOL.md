# PROTOCOL — BAMF-v1: bond-anchored marginal field

**Protocol ID:** `zinc-bond-anchored-marginal-field-v1`  
**Track:** `tracks/structural-sufficiency/zinc_bond_anchored_marginal_field_v1/`  
**Status:** frozen after Phase-0 CPU gates and before any GPU job.  
**Design:** `DESIGN.md`; frozen numeric record: `configs/bamf_v1.json`.

This round asks one question only. It is not a hyper-parameter search, not an
architecture zoo, and not a licence to run the official test.

---

## 1. Sole hypothesis

> Under the condition that DCC-L's original local content is unchanged, can
> molecule-consistent latent-state inference convert long-range structural
> information into better molecular property prediction than independent
> per-bond inference?

The comparison of record is **G vs I**. The comparison against the matched
historical **DCC-L** is the practical floor. If G beats I but not L, the verdict
is `FACTOR_FEATURE_GAIN_NOT_GLOBAL`; if G beats L but not I, the verdict is
`GLOBAL_MECHANISM_BUT_NO_PERFORMANCE`.

## 2. Hard boundaries (all gates below are subordinate to these)

- Official PyG ZINC subset: train = 10,000, valid = 1,000, test = 1,000.
  **No labelled sample is ever added.** Only rows inside official train-10k are
  used for training, monitoring, or locked development scoring.
- Official valid and official test are **never loaded** in this round. The
  in-train splits are Fit8k / Monitor1k / Locked1k from the frozen BJOC split
  (`tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1/split_a.json`).
- Raw-y L1 only. No target formula, no RDKit Crippen/SA contribution, no
  artificial cycle penalty, as input or auxiliary supervision. Penalty labels
  are used only in post-hoc error attribution, never as a training target.
- ≤ 500,000 deployed parameters; one model; no ensemble, distillation,
  pseudo-labelling, or test-time adaptation.
- No recursive message passing of learned atom/bond hidden states. The exact
  elimination below is **algorithmic global inference** over a factor graph; it
  is not "no propagation", and it is not a learned GNN layer.
- Historical checkpoints, reports, and frozen decisions are read-only.
- This round's decision does **not** trigger auto-expansion training or official
  test scoring.
- **No grid search** over channel count C, state count S, potential amplitude,
  hidden width, learning rate, or training steps. The recipe and operator
  hyper-parameters are fixed here.

## 3. Data and frozen assets

| role | rows | source | use |
|---|---:|---|---|
| Fit8k | 8,000 | `ad_train.pt`, split `fit_rows` | supervised training only |
| Monitor1k | 1,000 | same cache, `monitor_rows` | per-epoch selection and seed-0/1 gates |
| Locked1k | 1,000 | same cache, `locked_rows` | exactly one read after prediction freeze |
| official valid | 1,000 | **not loaded** | — |
| official test | 1,000 | **not loaded** | — |

Frozen row hashes (from the BJOC split file):

- Fit8k: `1cf6a2ec73ff5dfc182d72a7d4cd0148d31f3ae2488ee9943132f1aa51d7455a`
- Monitor1k: `d493dc7fc547e062ee16301bb42f350d5eb6d5d773215254c0d84098a410933b`
- Locked1k: `9cbefd5e84cf7f4e8f7314280f35f636a0b18753a2f93a724b935e65a208cf6d`

Standardisation is fit-only, `fit_only_norm(train_cache, fit8k)`; its
`global_mean/global_scale` are bit-equal to the buffers in the frozen DCC-L
checkpoints (verified in Phase 0). P3 SSL encoders are the frozen
`pretrain_encoder_s{0,1}.pt` files; seed 0 and seed 1 use their own encoder.

Matched L baselines:

| seed | checkpoint | monitor MAE (re-evaluated) | sha256 |
|---:|---|---:|---|
| 0 | `zinc_dccl_bottleneck_v1/runs/fit8k_s0/best_monitor.pt` | 0.1231508795 | `6a145dc69e81aed9c1f41f572a6956f8a7351dacba9ddcaa2daf4e71930f5817` |
| 1 | `zinc_dccl_bottleneck_v1/runs/fit8k_s1/best_monitor.pt` | 0.1277715682 | `882fae17de42ef007b79750e6851e73dd414316a3c2efa15b2468892ce1ff35c` |

The L checkpoints were selected on Monitor1k by the historical `dcb_train.py`
loop; re-evaluation on Monitor1k reproduces the frozen predictions to
`1.9e-6` (float32).

## 4. Model (exact, audited)

Start from the historical DCC-L initial state (`DCC.build_initial_state`), then:

1. Copy every DCC-L tensor; keep `group_composer` and `reader` unchanged.
2. Widen **only** `local_composer.lins[0]` from 296 to 308 input columns:
   - first 296 columns = frozen DCC-L init columns;
   - last 12 columns = zero-init.
3. Add three shared potential heads:
   - `atom_potential = Linear(96, 8)`, applied to every atom state `a_v`;
   - `bond_potential = Linear(96, 12)`, applied to every bond state `b_e`.
4. `local_composer` input becomes
   `[b_e, Wl_e, Vl_e, log1p(n_b), bond_feat_e, m_e]`, where the first 296 dims
   are exactly DCC-L's organized-local input and `m_e` is the 12-dim marginal
   feature below.

Actual `named_parameters()` audit (Phase 0): **497,093** total.

| component | parameters |
|---|---:|
| DCC-L base | 494,193 |
| `atom_potential` | 776 |
| `bond_potential` | 1,164 |
| 12 new local-composer columns × 80 | 960 |
| **total** | **497,093** |

Initial BAMF state hashes (fit-only norm):

- seed 0: `fe4e98c31b1f02021a7f680a066fc80140d40289b0a0b30e6748d8795490c340`
- seed 1: `7205d7511775a4de7e87c0aee98d81262e17cc16784fb1e8df04c595b298e7b0`

G and I are built from the same state stream and have **bit-identical initial
tensors**; only the marginal-computation flag differs.

## 5. Operator

For each molecule, physical bonds define an undirected factor graph. Four
independent binary channels `c = 1..4`; two states `s ∈ {0,1}`.

```
log_phi_v[c,s]       = tanh( Linear_{96→8}(a_v) )[c,s]
raw_psi_e[c,0..2]    = tanh( Linear_{96→12}(b_e) )[c,0..2]
log_psi_e[c,0,0]     = raw_psi_e[c,0]
log_psi_e[c,0,1]     = raw_psi_e[c,1]
log_psi_e[c,1,0]     = raw_psi_e[c,1]          # symmetric
log_psi_e[c,1,1]     = raw_psi_e[c,2]
```

Exact **G** arm: eliminate all variables with a min-fill junction tree (max
induced width on official train-10k is re-computed, not cited: 3; `smax=4`,
`S=2^4=16`). Upward and downward sum-product messages produce calibrated
clique beliefs. The per-bond edge marginal is

```
p_e(c, s_u, s_v) = exp(belief_clique(e) restricted to s_u,s_v) / Z_clique
m_e = [ mu_u + mu_v - 1,
        |mu_u - mu_v|,
        4 * (mu_uv - mu_u * mu_v) ]              # 4 channels × 3 = 12
```

where `mu_u = p(s_u=1)`, `mu_v = p(s_v=1)`, `mu_uv = p(s_u=1,s_v=1)`.

Exact **I** arm: with the **same** `log_phi` and `log_psi`, compute each edge's
2×2 marginal independently:

```
p_e(c,·,·) = softmax( log_phi_u + log_phi_v + log_psi_e )
```

and apply the identical `m_e` formula. No cross-edge consistency.

The exact junction-tree pass is global algorithmic inference. It uses no
learned atom/bond hidden-state recursion and adds no recurrent GNN layer.

## 6. Recipe (identical for G and I)

```
AdamW, lr 1e-3 -> 1e-5 per-step cosine, warmup 500 updates,
weight_decay 1e-5, dropout 0, grad clip 5, FP32, no AMP,
raw-y L1, batch 128, eval batch 256,
Fit8k -> 63 steps/epoch, 753 epochs, exactly 47,400 updates,
batch order: Generator(seed+101) -> randperm(8000) each epoch,
best checkpoint: strict minimum Monitor1k MAE, earlier epoch wins ties.
```

No early stopping beyond the fixed 47,400 updates; no LR/width/amplitude search.
The potential heads are trained jointly with the rest of the model. The two
zero-init gates (reader's new columns and local composer's new columns) mean the
potential heads first receive non-zero gradient on the third forward pass
(0-based step 2); Phase 0 records this explicitly. This is a consequence of the
frozen zero-init protocol, not a detach bug.

## 7. Phase-0 CPU gates (must be green before GPU)

Artifact: `analysis/phase0_cpu.json`. Tests:
`uv run pytest -q tracks/structural-sufficiency/zinc_bond_anchored_marginal_field_v1/tests`.

| gate | requirement | Phase-0 result |
|---|---|---|
| P0.1 topology | re-compute min-fill width on all 10k; width ≤3, clique ≤4 | PASS: max 3, counts [0,55,9906,39] |
| P0.2 exactness | float64 dense enumeration, random 2–8 node graphs, logZ/node/edge marginals | PASS: 14 pytest cases, max error ≤1.8e-15 |
| P0.3 batching | batch == single; empty, single-bond, tree, ring, dense, disconnected | PASS |
| P0.4 invariance | atom renumbering, endpoint swap, bond permutation | PASS: ≤4.5e-16 |
| P0.5 numerics | extreme potentials ±100; finite, normalised | PASS |
| P0.6 gradients | finite-difference first derivative; second-order backprop through marginals | PASS |
| P0.7 modes | train/eval forward identical | PASS: 0.0 |
| P0.8 init/replay | G/I initial state identical; step-0 prediction equals DCC-L | PASS: max 4.8e-7 |
| P0.9 params | real `named_parameters()` ≤500k | PASS: 497,093 |
| P0.10 gradient reach | non-zero gradients into both potential nets after the zero-init gates | PASS: first non-zero at 0-based step 2 |
| P0.11 non-locality | six-node chain: distant `phi[5]` changes G edge (0,1), not I | PASS: G 2.88e-5, I exactly 0 |
| P0.12 real operator | Fit8k G/I difference stats + redundancy vs DCC local input | recorded; see below |

P0.12 is diagnostic, not a hidden tuning gate. On 1,024 Fit8k molecules
(25,718 bonds), mean absolute G−I feature difference is 0.0459 (max 0.587);
99.8% of entries differ by >1e-6. A molecule-disjoint ridge probe predicts G
features from the 296-dim DCC local input with mean test R² 0.791; it predicts
G−I with mean R² 0.525. Thus a substantial part of the marginal field is
locally predictable, and a substantial residual is not. This is carried into the
report as a redundancy warning, not used to change C/S/amplitude.

Any hard P0 failure stops the round immediately. No rescue by changing C, S,
potential amplitude, width, LR, or steps.

## 8. GPU gate and budget

Only after P0 is green and PROTOCOL/DESIGN/config are committed:

1. `rr doctor res`; deploy the frozen commit.
2. GPU smoke on `res-gpu0`: one short job that constructs the 10k plan, runs
   several G updates, and checks finite loss/gradients.
3. Full-epoch profile on `res-gpu0` (G) and `res-gpu1` (I), including the
   backward pass through the marginal computation. Record wall time, plan
   selection time, and peak memory.
4. If any **G** run is projected to exceed **2.5 A100·h** for 47,400 updates,
   emit `STOP_COMPUTE` and do not launch Phase 1.
5. Hard cap for the whole round: **12 A100·h**. No hidden retries.

`res-gpu0` and `res-gpu1` each host one job at a time. Do not run two jobs on
one GPU. Record every `rr` job id and GPU-hour cost, including failed/smoke jobs.

## 9. Phase 1: paired seeds, monitor-only gates

Step 1 (seed 0), launched in parallel on the two GPUs:

- `G_s0` on `res-gpu0`
- `I_s0` on `res-gpu1`

Both: same Fit8k rows, same norm, same init stream (G/I bit-identical), same
recipe, 47,400 updates. Monitor1k is the only validation read.

Seed-0 gates:

| gate | threshold |
|---|---|
| G_s0 − I_s0 (Monitor MAE) | ≤ −0.004 |
| G_s0 − L_s0 (matched historical Monitor MAE) | ≤ +0.002 |
| suggested diagnostic: any-unseen no-penalty subgroup same-direction improvement vs L | ≥ 0.005 |

If either hard gate fails: `STOP_NO_SIGNAL`. Do not run seed 1; do not read
Locked; do not retune C/S/amplitude/width/LR/steps.

Step 2 (seed 1) only if seed 0 passes:

- `G_s1` and `I_s1` with the same frozen protocol.

Then freeze:

- best-monitor and final checkpoints for G/I at both seeds;
- Monitor predictions and per-molecule errors;
- checkpoint/prediction SHA-256 hashes;
- a `FREEZE.json` with the exact commit, job ids, and hashes.

Only after that freeze, read Locked1k **once** and score the frozen
checkpoints. No model selection on Locked.

## 10. Final gates for a Phase-2 recommendation

A recommendation for a next-generation non-recursive structural computation
requires all of:

1. G − I improves at both seeds; mean ≤ −0.005 with a robust CI.
2. G − L improves at both seeds; mean ≤ −0.006.
3. Any-unseen no-penalty subgroup vs L improves by ≥ ~0.010.
4. After dropping the top-10 influential rows, G − L is still ≥ 0.002.
5. The effect is not produced by penalty extremes.
6. G is confirmed to use global marginals (not a disguised local feature).

Special verdicts:

- beats I only → `GLOBAL_MECHANISM_BUT_NO_PERFORMANCE`;
- beats L only → `FACTOR_FEATURE_GAIN_NOT_GLOBAL`.

If the gates fail, the round closes with an explicit STOP. There is no extra
experiment menu, no automatic seed expansion, and no official-test read.

## 11. Audit trail

Required records: code commit, environment (`python`, `torch`, CUDA/driver,
GPU), data/split/checkpoint hashes, plan hash, parameter audit, Phase-0 JSON,
GPU smoke/timing JSON, `rr` job ids, run `train.json`/`curve.csv`, frozen
checkpoint/prediction hashes, and Locked scoring record. `uv run research verify`
must pass before the round is called complete. `STATE.yaml` gets one audit
pointer to the final `REPORT.md`/`DECISION.md`.

## 12. Explicitly not allowed

- Changing C, S, potential amplitude, hidden width, LR, warmup, or update count
  after seeing Monitor/Locked results.
- Adding labelled rows, pseudo-labels, distillation, or an ensemble.
- Using official valid/test for selection or reporting.
- Describing the exact elimination as "no propagation" or as a learned
  recursive GNN.
- Reading Locked before the prediction freeze, or scoring it more than once.
