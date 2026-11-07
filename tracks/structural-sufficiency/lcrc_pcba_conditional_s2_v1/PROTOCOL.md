# PROTOCOL — LCRC Conditional S2 v1

Round id: `lcrc-pcba-conditional-s2-v1`
Status: **frozen before the first formal training step**. Only append-only
amendments (§12) are permitted after this point.

## 1. The single question

On the already-successful **W2** backbone (2× LCRC-A, 994,324 parameters,
epoch-40 official Validation AP **0.2618669181**), does replacing S2's scalar
path gate with a low-rank **vector** gate conditioned on
`(path type, path endpoint type)` produce a Validation AP gain beyond a
near-parameter-matched plain FFN?

This is a mechanism question with a capacity control, not a capacity scan. It is
not a search: exactly two new runs, one seed, the frozen 40-epoch recipe, the
fixed epoch-40 endpoint.

## 2. Frozen reference (never retrained in this round)

| item | value |
|---|---|
| model | `W2` = LCRC-A-Wide-2x, 994,324 parameters |
| epoch-40 official Validation macro AP | `0.26186691813350904` (protocol: 0.2618669181) |
| epoch-40 Train AP / Train BCE / Valid BCE | 0.5818745570283522 / 0.02628940533386738 / 0.04653792303564479 |
| best diagnostic epoch / AP (unused) | 24 / 0.26527331610028143 |
| source | `lcrc_pcba_backbone_scaling_v1/runs/W2` (read-only) |
| trunk init fingerprint | `5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d` |
| FULL trunk init fingerprint | `109756bd24b2ab8589012031e3cf0b0389b0d275227b35044e9381a45c00a5db` |
| global batch-order plan | `06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6` |
| consumed 40-epoch prefix | `1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58` |
| checkpoint SHA-256 | `7fbdbd16f7b2077e77cdb85488bb2506f2e4ee98333d40fbcf7558074e26598a` |

The historical W2 curve is read from
`lcrc_pcba_backbone_scaling_v1/runs/W2/train_curve.csv`; it is never re-run.
There is no W2 seed-1 run and no W2 retraining in this round.

## 3. The two new arms

Both arms start from the **same untrained W2 seed-0 state** and are trained
from scratch for 40 epochs. No warm start from the historical W2 epoch-40
checkpoint is allowed.

| arm | computation | new parameters | total parameters |
|---|---|---:|---:|
| `C2` | W2 + Conditional S2 vector gate | **10,752** | **1,005,076** |
| `F` | W2 + one local residual FFN | **10,664** | **1,004,988** |

Parameter difference: **88** (`10,752 − 10,664`), i.e. **0.82 %** of the C2
increment. This is a *near*-matched control, never described as exactly matched.

## 4. Base model (both arms)

The historical W2 width dict, passed explicitly to every constructor:

```
atom_hidden = 256      atom_out = 192
bond_hidden = 320      bond_out = 192
stat_rank = 64         local_hidden = 384
width = 128            rho_hidden = 128
attention scorer hidden = 64
attention pool hidden = 176
attention heads = 4    n_outputs = 128
```

Preserved unchanged (inherited read-only from `bs_model` / `tu_model` /
`nmp_model`): the atom encoder, the bond encoder, S1, the S2 path enumeration,
the S2 typed-path / endpoint / joint / closure statistics, the per-atom `f`,
the global `rho`, the four-head attention pooling, dropout and every graph-level
readout. The new modules perform one-shot local computation only; no updated
atom state is propagated along an edge.

### 4.1 C2 — Conditional S2 (exact definition)

The historical W2 S2 content computation is

```python
gate = self.path_gate(batch["path_cat"])                 # (n_paths, 1)
content_agg = zeros.index_add_(
    0, batch["path_atom"], a[batch["path_end"]] * gate
)
```

C2 changes exactly the gate:

```
g_p = g_scalar(c_p) + Δg(c_p, e_p)
Δg(c, e) = (E_c(c) ⊙ E_e(e)) V

E_c: Embedding(400, 16)      # c = path_cat, 34,369,144 cached paths, range [0, 393]
E_e: Embedding(80, 16)       # e = path_end63, range [0, 78]
V:   Parameter(16, 192)      # 192 = atom_out, the width of a[path_end]
```

```python
delta = (E_c(path_cat) * E_e(path_end63)) @ V
vector_gate = path_gate(path_cat) + delta
content_agg = zeros.index_add_(
    0, path_atom, a[path_end] * vector_gate
)
```

Every other line of `TULocalContent.forward` is the historical line, including
the path/joint/closure aggregation, the local MLP and the LayerNorm. There is
no new sigmoid, attention, normalisation, gate scale or learnable residual
coefficient.

Parameter arithmetic (exact):

```
400*16 + 80*16 + 16*192 = 6,400 + 1,280 + 3,072 = 10,752
994,324 + 10,752 = 1,005,076
```

The vector gate is `(n_paths, 192)` because the S2 content multiply acts on
`a` (`atom_out = 192`), not on the local MLP output (`width = 128`).

### 4.2 F — near-parameter-matched FFN control

Inserted **after** the local-content output and **before** the original
dropout; the original dropout is still called exactly once, at the historical
position:

```python
class LocalResidualFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128, 40)
        self.fc2 = nn.Linear(40, 128)
    def forward(self, h):
        return h + self.fc2(F.silu(self.fc1(self.norm(h))))
```

```
LayerNorm(128)  = 256
Linear(128, 40) = 5,160
Linear(40, 128) = 5,248
total           = 10,664
994,324 + 10,664 = 1,004,988
```

F never reads a bond, an S1/S2 code, or another atom; it is a local capacity
control. C2 and F add their module at different operators (S2 content vs local
representation), so the comparison is a **parameter-capacity** comparison, not
a same-computation-graph comparison. The report must not attribute every
difference to the path mechanism.

## 5. Initialisation (frozen)

1. Build the historical untrained W2 seed-0 state with
   `bs_model.build_model("W2", seed=0)` (the exact historical rule).
2. Copy **every** original W2 tensor into C2/F bitwise.
3. C2: `E_c`, `E_e` are drawn from a private generator
   (`seed*1_000_003+29`) as `Normal(0, 1/sqrt(16)) = Normal(0, 0.25)`;
   `V = 0`. The original scalar `path_gate` is retained unchanged (all ones at
   init).
4. F: `LayerNorm` defaults; `fc1` uses the standard `nn.Linear` initialisation
   (`kaiming_uniform(a=sqrt(5))`, bias `U(-1/sqrt(128), 1/sqrt(128))`) from a
   private generator (`seed*1_000_003+31`); `fc2.weight = fc2.bias = 0`.
5. New-module initialisation must not advance the global torch RNG. The
   *historical* W2 builder advances it (as it always did); the formal trainer
   calls `seed_everything(0)` after model construction, exactly as the W2
   training run did.
6. Consequence: `Δg = 0` at step 0 and the FFN branch is 0 at step 0, so the
   initial C2/F output must equal the original W2 output (CPU bitwise; CUDA
   within the historical atomic-accumulation tolerance).

Gradient consequence, recorded in advance: because `V = 0`, `E_c`/`E_e` receive
zero gradient on step 1 and non-zero gradient from step 2; because
`fc2.weight = 0`, `fc1` receives zero gradient on step 1 and non-zero from
step 2. `V` and `fc2` themselves receive non-zero gradient on step 1. This is
the expected zero-residual initialisation, not a training failure.

## 6. Training recipe (frozen, inherited from Backbone Scaling)

```
dataset = ogbg-molpcba        official Train = 350,343   official Valid = 43,793
seed = 0                      epochs = 40                batch_size = 128
steps_per_epoch = 2,738       total_optimizer_steps = 109,520
optimizer = AdamW             peak_lr = 3e-4
weight_decay = 1e-3           betas = (0.9, 0.999)       eps = 1e-8
warmup = 1 epoch linear       warmup_start_lr = 3e-5
scheduler = per-step cosine   final_lr = 1e-5
dropout = 0.10                gradient_clip_norm = 5.0
precision = FP32              loss = original observed-entry masked BCE
early_stopping = false
```

The schedule is exactly `sg_common.scheduled_lr(step, phase=3, kind="all")`.
The seed-0 batch order is the frozen historical plan
(`06f43729…`); both arms consume the same raw row id order. No training weight,
loss normalisation, negative sampling, evaluation definition, width, depth,
dropout or LR is changed.

## 7. Validation cadence (fixed by the round brief)

The previous round measured a full W2 Validation pass at **7.15 s** against a
**121.7 s** epoch (**5.9 %**), so this round fixes, for both arms:

```
evaluate official Validation at epochs 1..40 (every epoch)
primary comparison = epoch 40 only
diagnostic epoch 0 = one initialisation evaluation, no training step
```

The primary metric is the epoch-40 official OGB Validation macro AP; the best
epoch is recorded as a diagnostic only and is never substituted. If C2 peaks
early and falls by epoch 40, that is reported as a late-generalisation /
optimisation fact, not as a new official result.

## 8. Pre-training audit (must pass before any formal step)

All checks are in `analysis/initialization_audit.json` and
`analysis/architecture_audit.json`, produced by `code/cs2_audit.py` on the GPU
host before training:

1. C2/F original-W2 parameter subsets are bitwise equal to each other and to the
   freshly built historical W2 seed-0 state.
2. W2 trunk and FULL-trunk init fingerprints equal the historical record.
3. C2 adds exactly 10,752 parameters; F adds exactly 10,664.
4. Both models' CPU initial logits are bitwise equal to W2; CUDA maximum
   absolute difference is recorded against the historical 1e-4
   cross-implementation tolerance, and the full 43,793-row epoch-0 Validation
   AP matches W2 within 1e-4.
5. `path_cat ∈ [0, 399]`, `path_end63 ∈ [0, 79]`, `path_end` inside the atom
   range, over the whole frozen cache; the gate is exactly
   `[n_paths, 192]`.
6. C2 does not modify path enumeration, endpoint atoms, path counts or any
   S1/S2 statistic; the batch is bitwise unchanged after a forward.
7. C2/F make exactly one atom encode, one bond encode, one `local`, one
   dropout, one `f`, one `rho` and one readout call per forward; no updated
   atom state is re-propagated; `post_local` works with no bond/path field.
8. Non-zero masked-BCE gradients reach `V` on step 1 and `E_c`/`E_e` by step 2;
   non-zero gradients reach `fc2` on step 1 and `fc1` by step 2.
9. Both arms use the identical frozen batch-order plan and the identical
   Train-only standardisation.
10. Dropout is called once per forward; with one seeded forward the three
    models leave the torch RNG in the same state.
11. NaNs in the label matrix are masked exactly (the masked BCE equals the
    observed-entry-only BCE).
12. Official Test is never indexed, predicted or scored; Test labels are all
    NaN in the frozen artifact; no Test row appears in the plan.
13. A resumable checkpoint restores the model bitwise, the AdamW state, the
    scheduler step, the RNG state and the next batch rows.
14. Forward/backward are finite; no NaN/Inf weights, gradients or logits occur
    in a formal run (the trainer aborts on a non-finite event).

If any gate fails, the implementation is fixed before any formal step; no known
inconsistency is allowed to proceed.

## 9. Execution, GPU scheduling and budget

All remote execution through `rr` only (`rr doctor res`, `rr deploy res`,
`rr run res …`); no manual SSH, `nohup` or `sbatch`.

Scheduling preference: C2 → `res-gpu1`; F → `res-gpu0` **only if** a fresh
6 × 10 s `nvidia-smi` sample shows GPU0 free of sustained full-load
contention and with enough free memory for a ~200 MB peak. Otherwise both run
serially on `res-gpu1`. External processes are never cancelled, paused or
modified.

```
target ≤ 4 A100 GPU-hours      hard cap ≤ 6 A100 GPU-hours
```

The cap covers preflight, audit, training, Validation and failed retries. If
the preflight projects more than the hard cap, the round stops with
`BUDGET_NOT_FEASIBLE`; the dataset, epoch count or recipe is never reduced to
fit. A completed 40-epoch run is never rerun.

## 10. Metrics and decision labels

```
AP_W2 = 0.26186691813350904
AP_C2 = C2 epoch-40 official Validation macro AP
AP_F  = F  epoch-40 official Validation macro AP
D_C2  = AP_C2 − AP_W2
D_F   = AP_F  − AP_W2
D_CF  = AP_C2 − AP_F
```

Reported: Train AP40, Validation AP40, Train BCE40, Valid BCE40, Train−Valid
AP gap, AP at epochs 10/20/25/30/40, the full curve, the best diagnostic
epoch/AP, epoch25→40 and epoch30→40 changes, the 127 scorable assays' per-task
AP, improved/median/mean deltas for C2−W2 and C2−F, and the strata by
Validation positives ≥20 (107 assays), 5–19 and <5.

Pre-registered labels (resource decisions, not significance tests):

### `PROMISING_CONDITIONAL_S2`

All of:

```
D_C2 >= +0.005
D_CF >= +0.003
mean(C2−W2 delta over the 107 high-support assays) >= +0.003
C2 improves >= 60/107 high-support assays vs W2
```

and no early-peak-then-epoch-40 regression (best diagnostic epoch ≤25 with
AP40 at least 0.005 below the best diagnostic AP).

### `CAPACITY_ONLY_OR_MIXED`

Any of: C2 improves W2 but does not meaningfully outperform F; F improves
similarly to C2; C2 gains are mainly in very sparse assays; C2 improves early
but regresses by epoch 40. Each sub-condition is reported separately.

### `WEAK_OR_MIXED_C2`

`0.002 <= D_C2 < 0.005`, or `D_C2 < 0.002` with a broad stable high-support
pattern, or a positive `D_CF` below +0.003 when the other conditions are not
met.

### `NO_CLEAR_C2_GAIN`

`D_C2 < +0.002` without a broad stable high-support gain.

### `C2_REGRESSION`

`D_C2 <= −0.002`.

Precedence: regression → promising (all four + no early regression) →
`0.002…0.005` weak band → `< 0.002` no-clear/weak → `>= 0.005` capacity/mixed
when any promising condition fails. The `early-peak-then-regress` flag can
downgrade an otherwise promising result to `CAPACITY_ONLY_OR_MIXED`; this
precedence is fixed here, before any result is seen.

## 11. Explicitly out of scope

No W2 retraining; no seed 1; no extra arms; no iterative GNN / bond message
passing; no S3 / motif / RIE; no SSL or pretraining; no loss or hyper-parameter
search; no width/depth/dropout/LR change; no best-epoch substitution; no
official Test access; no mid-run structural or recipe change; no length
extension; no rank sweep after seeing results.

## 12. Deliverables

```
PROTOCOL.md  README.md  REPORT.md
code/cs2_common.py  code/cs2_model.py  code/cs2_train.py
code/cs2_audit.py   code/cs2_compare.py
tests/test_cs2_model.py  tests/test_cs2_training.py
analysis/initialization_audit.json   analysis/architecture_audit.json
analysis/training_parity.json
analysis/c2_epoch40.json             analysis/ffn_epoch40.json
analysis/paired_comparison.json      analysis/per_task_deltas.csv
analysis/gpu_ledger.json             analysis/decision.json
runs/C2/  runs/F/                    (git-ignored)
```

`runs/` keeps the final checkpoint, the resumable optimizer/RNG state, the
curves, per-task AP, the order plan and every SHA-256; the report records the
real recovery path.

## 13. Amendments (append-only)

| # | date | change | reason |
|---|---|---|---|
| 1 | 2026-11-08 | Initial freeze | before any formal step |
| 2 | 2026-11-08 | §9 applied: GPU0 not shared, both formal runs serial on `res-gpu1` | not a rule change — the co-tenancy clause failed: GPU0 was at 100 % utilisation in 6/6 fresh samples (external pid 3841932), while 24,222 MiB remained free. See `analysis/gpu_sampling.json` and `analysis/gpu_occupancy_samples.csv`. C2 21:39→23:13 CST, F 23:16→00:36 CST. |
