# PROTOCOL — LCRC PCBA Cross-atom Predictive Evidence (CPE) v1

Round id: `lcrc-pcba-cpe-v1`
Status: **frozen before the first formal training step**. After this commit only
append-only amendments (§13) are permitted.

## 1. The single question

On the already-successful **W2** backbone (2× LCRC-A, 994,324 parameters,
epoch-40 official Validation AP **0.2618669181**), with S1/S2 local
representation, the per-atom `f`, the global `rho` and the 4-head attention
pool readout all left untouched, does adding an **explicit second-order
predictive-evidence statistic between *different* atoms** produce a better
MolPCBA Validation AP than the **same-parameter, same-shape, same-init**
control that uses the *same-atom* product?

Two and only two arms, one seed, the frozen 40-epoch from-scratch recipe, the
fixed epoch-40 endpoint. No hyper-parameter search.

This round does **not** claim W2 cannot express cross-atom interactions. It
tests whether an **explicit interaction inductive bias** adds anything on top.

## 2. Frozen references (never retrained here)

| item | value |
|---|---|
| A0 seed0 epoch40 | parameters 287,644 · Validation AP 0.2463560961 |
| **W2 seed0 epoch40** | parameters **994,324** · **Validation AP 0.2618669181** |
| W2 Train AP40 / Train BCE40 / Valid BCE40 | 0.5818745570 / 0.0252153644 / 0.0465379230 |
| W2 best diagnostic epoch / AP (unused) | 24 / 0.2652733161 |
| C2 seed0 epoch40 (Conditional S2) | Validation AP 0.2609026157 |
| C2-FFN seed0 epoch40 | Validation AP 0.2614996012 |
| W2 checkpoint SHA-256 | `7fbdbd16f7b2077e77cdb85488bb2506f2e4ee98333d40fbcf7558074e26598a` |
| W2 source | `lcrc_pcba_backbone_scaling_v1/runs/W2` (read-only) |

The historical W2 curve, per-task AP and train-eval record are read directly
from `lcrc_pcba_backbone_scaling_v1/runs/W2/*`. W2 is never re-run.

Existing evidence taken as given (not re-tested): widening the W2 trunk pays;
S2 low-rank vector gates do not; conditional S2 is fully learned but does not
help; RIE motif-distance relations do not add absolute predictive gain; KSVD
matched-pair work shows relations are assay-specific, so a universal change
dictionary cannot be assumed.

Consequently this round adds **no** new motif, path type, structural radius or
S2 gate.

## 3. The two arms

Both arms are built from the **same untrained W2 seed-0 state** and trained
from scratch, seed 0, 40 epochs, identical order plan.

| arm | pair statistic | new parameters | total parameters |
|---|---|---:|---:|
| `P` | **cross-atom** `pair_cross` | 33,408 | **1,027,732** |
| `S` | **same-atom** `pair_self` | 33,408 | **1,027,732** |

The two arms have **identical** parameter names, shapes, values at step 0,
optimizer, order plan and validation cadence. The only computational
difference is which third statistic is emitted (`cross` vs `self`). Both arms
compute `pair_cross` *and* `pair_self` every forward, so the arithmetic and
memory cost is the same; only the selected tensor differs.

## 4. Base model (both arms) — W2, preserved exactly

```
atom_hidden = 256      atom_out = 192
bond_hidden = 320      bond_out = 192
stat_rank    = 64      local_hidden = 384
width (local output) = 128        rho_hidden = 128
attention scorer hidden = 64      attention pool_hidden = 176
attention heads = 4               n_tasks = 128
```

Construction: `bs_model.build_model("W2", spec, norm, 128, seed=0)` (the
historical builder, imported read-only), then every original W2 state tensor is
copied **bitwise** into the CPE model. Widths, S1/S2 features, path
enumeration, Train-only normalisation, `f`, `rho`, dropout and the 4-head
attention readout are unchanged.

The new module reads `h` **once**, from the one forward already performed:

```python
a, b, scal, h = model.encode_h(batch)          # historical: one atom encode,
                                               # one bond encode, one local, one dropout
base_logits   = model.post_local(h, batch)     # historical f + rho + 4-head readout
logits        = base_logits + evidence_residual(h, batch)
```

No atom is re-encoded, the original dropout is never called again, and no
updated atom hidden state is propagated along any bond. There is no iterative
message passing, no cross-atom attention, no cross-atom QKV and no implicit
re-propagation anywhere in the new module.

## 5. The new module (identical definition for both arms)

Evidence detectors are task-specific: the 128 output channels map one-to-one to
the 128 MolPCBA assays. They read only `h` (W2's 128-dim post-`local`,
post-dropout atom state); they never read edges, paths, motifs or another atom.

```python
linear_A = nn.Linear(128, 128, bias=True)      # 16,512 parameters
linear_B = nn.Linear(128, 128, bias=True)      # 16,512 parameters

evidence_A = torch.tanh(linear_A(h))           # [n_atoms, 128]
evidence_B = torch.tanh(linear_B(h))           # [n_atoms, 128]

sum_A  = segment_sum(evidence_A)               # [n_graphs, 128]
sum_B  = segment_sum(evidence_B)
sum_AB = segment_sum(evidence_A * evidence_B)
n      = segment_count(atoms)                  # [n_graphs]

mean_A = sum_A / n
mean_B = sum_B / n

pair_cross = (sum_A * sum_B - sum_AB) / (n * (n - 1))     # P uses this
pair_self  = sum_AB / n                                   # S uses this

residual = alpha_A * mean_A + alpha_B * mean_B + alpha_pair * pair_statistic
logits   = base_logits + residual
```

with, for every task `t`,

```
P_t(G) = ( (sum_i a_it)(sum_j b_jt) - sum_i a_it b_it ) / (n(n-1))  =  mean_{i != j} a_it b_jt
S_t(G) = ( sum_i a_it b_it ) / n                                    =  mean_{i == j} a_it b_it
```

* `n == 1` → `pair_cross` is **exactly zero** (the denominator is 0; the code
  masks it with `torch.where`, it does not divide).
* `n == 0` (illegal, must not occur in the frozen cache) → all statistics are
  exactly zero and the count is **reported** in every audit that touches it.
* `pair_cross` is invariant to swapping the A/B evidence streams.
* Complexity is `O(n_atoms * n_tasks)`; **no atom pair is enumerated**.

Per task three trainable scalars, all initialised to zero:

```python
alpha_A    = nn.Parameter(torch.zeros(128))
alpha_B    = nn.Parameter(torch.zeros(128))
alpha_pair = nn.Parameter(torch.zeros(128))
```

No extra MLP, attention, temperature, normalisation layer or learnable scale is
added. Both arms own exactly the same main-effect terms.

## 6. Exact parameter accounting

| term | arithmetic | parameters |
|---|---|---:|
| `linear_A` | 128×128 + 128 | 16,512 |
| `linear_B` | 128×128 + 128 | 16,512 |
| `alpha_A` | 128 | 128 |
| `alpha_B` | 128 | 128 |
| `alpha_pair` | 128 | 128 |
| **new total** | 33,024 + 384 | **33,408** |

`994,324 + 33,408 = 1,027,732` — **both** arms must equal exactly
**1,027,732** trainable parameters by `sum(p.numel())`. No padding, no filler
parameter, tolerance zero.

## 7. Initialisation and comparability

```
1. build the historical untrained W2 seed-0 state           (bs_model.build_model)
2. copy every original W2 tensor into the CPE model          (bitwise, no recompute)
3. linear_A, linear_B: standard PyTorch Linear uniform init
   (kaiming_uniform_(a=sqrt(5)) + U(-1/sqrt(fan_in), +1/sqrt(fan_in)))
   from a PRIVATE CPU generator with seed 37, same call order for both arms
4. alpha_A = alpha_B = alpha_pair = 0
5. new-module construction/initialisation must not advance the global torch RNG
   (construction runs inside torch.random.fork_rng(devices=[]), init uses the
   private generator only)
```

No trained W2 weight is ever loaded; only the freshly rebuilt untrained seed-0
state is used.

Required equalities, checked on CPU and recorded before any formal step:

```
W2_initial_logits == P_initial_logits        (bitwise on CPU)
W2_initial_logits == S_initial_logits        (bitwise on CPU)
P_initial_logits  == S_initial_logits        (bitwise on CPU)
W2 ⊂ P, W2 ⊂ S: every historical tensor bitwise equal
P.new == S.new: every new tensor bitwise equal
```

CPU must be bitwise; on CUDA the audited identity tolerance of the previous
rounds is used (`1e-4` max |Δ| on logits, because `index_add_` accumulates with
atomics).

Gradient-reachability expectation (a property of `alpha_* = 0`, not a failure):

```
step 1 : alpha_A / alpha_B / alpha_pair receive non-zero finite gradient
step 1 : linear_A / linear_B gradient is exactly zero
step 2+: linear_A / linear_B receive non-zero finite gradient
```

If this does not happen, the round is an **implementation failure** and stops
before any scientific interpretation.

## 8. Pre-training gates (all must pass on the GPU host)

Recorded in `analysis/architecture_audit.json` and
`analysis/initialization_parity.json`.

1. Exported architecture audit lists all 18 checks below; `all_passed: true`.
2. `P` and `S` parameters are exactly **1,027,732**; new = 33,408; base = 994,324.
3. Every original W2 init tensor is bitwise equal inside `P` and `S`.
4. All five new tensors are bitwise equal between `P` and `S` at epoch 0.
5. `P` and `S` differ only in the `pair_mode` attribute (`cross` / `self`);
   parameter names, shapes, dtypes and values are identical.
6. `atom_encoder`, `bond_encoder`, `local`, `f`, `rho`, `readout` each execute
   **exactly once** per forward; `drop` (dropout) fires **exactly once**.
7. No iterative GNN block, no cross-atom QKV, no second `local`/`readout` call:
   the evidence module's only cross-atom operations are `index_add_` reductions.
8. Graph logits are invariant to atom renumbering within each molecule
   (tolerance `2e-5`).
9. Swapping the A/B evidence streams leaves the second-order statistic unchanged
   (exactly zero difference).
10. `n == 1` molecules: `pair_cross` is exactly zero.
11. Several molecules in one batch produce **no** cross-molecule evidence
    interaction (per-molecule statistics identical to single-molecule batches).
12. `pair_cross` equals the explicit `i != j` double loop on small synthetic
    graphs (≤ 5 atoms, max |Δ| ≤ 1e-6 float32).
13. `pair_self` equals the explicit `i == j` loop on the same graphs.
14. Two-atom synthetic case: one evidence stream on atom 1 and the other on
    atom 2 → `pair_cross > 0`, `pair_self == 0`.
15. `tanh` outputs, graph reductions and residuals are finite everywhere on a
    real batch; no NaN/Inf.
16. Train/Validation/Test row isolation of the frozen order plan holds and no
    Test label is finite or indexed.
17. Epoch-0 logits of W2, `P` and `S` agree (CPU bitwise; CUDA ≤ 1e-4), and the
    epoch-0 official Validation AP of the three models agrees within 1e-4.
18. Gradient propagation is as specified in §7, and the full checkpoint
    (model + AdamW + scheduler step + RNG + batch plan) restores exactly.

## 9. Training protocol (inherited unchanged)

```
dataset = ogbg-molpcba, official OGB scaffold split
Train = 350,343  Valid = 43,793  Test = 43,793 (never evaluated)
seed = 0, from scratch, epochs = 40, batch_size = 128
steps_per_epoch = 2,738, total_steps = 109,520
optimizer = AdamW, peak_lr = 3e-4, warmup_epochs = 1, warmup_start_lr = 3e-5
scheduler = historical per-step cosine, final_lr = 1e-5
weight_decay = 1e-3, betas = (0.9, 0.999), eps = 1e-8
dropout = 0.10, gradient_clip_norm = 5.0, precision = FP32
loss = historical observed-entry masked BCE
early_stopping = false
```

Frozen seed-0 batch order: `global_plan_sha256
06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6`,
consumed-40 prefix `1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58`.
Same Train-only normalisation (`6ce50f98…`), feature spec
(`7270f767…`), split (`c872bd63…`) and label artifact (`5c3b5fbe…`) as W2.

No learning-rate search, no loss change, no additional regularisation, no data
subsetting.

## 10. Validation cadence, metrics, model selection

Validation is the **complete official Validation split (43,793 rows)**,
evaluated **every epoch 1…40** with `model.eval()`, `torch.inference_mode()`,
the original OGB `Evaluator`, and no RNG consumption.

```
primary metric   : official Validation macro AP at epoch 40
secondary record : AP10, AP20, AP25, AP30, AP40, full curve,
                   best observed AP + epoch (diagnostic only),
                   Train AP40, Train BCE40, Valid BCE40
```

The primary metric is epoch 40. A better intermediate epoch is **never**
substituted; the best-AP checkpoint is written but unused for selection.

The historical W2 curve is read from the frozen `backbone_scaling_v1` run; W2 is
never re-trained.

**The official Test split is never used** for selection, prediction or scoring
in this round.

## 11. Execution, budget

Only through the repository `rr` CLI (`rr doctor res`, `rr deploy res`,
`rr run res …`, `rr jobs`, `rr status`, `rr logs`, `rr pull`); no manual SSH,
`nohup` or `sbatch`, no interference with other users' processes.

Preference: `P → res-gpu1`, `S → res-gpu0` **if GPU0 is safe**. GPU0 has been
held at ~100 % utilisation by an external job in the two previous rounds; the
pre-registered rule is: share GPU0 only if it is idle or lightly loaded *and*
free memory exceeds the projected peak with margin. If not, both arms run
**serially on `res-gpu1`**. External processes are never cancelled, paused or
modified.

```
target   <= 4 A100 GPU-hours
hard cap <= 6 A100 GPU-hours
```

The cap covers the preflight, the audit job, both formal runs, all Validation
evaluation and any failed attempt/retry. If the cap cannot be met the round
reports `BUDGET_NOT_FEASIBLE`; data volume and stopping conditions are **never**
changed to fit. A failed formal run is resumed from its complete last-epoch
checkpoint (model + AdamW + scheduler + RNG + plan); a completed 40-epoch run
is never re-run.

## 12. Mechanism verification (not only AP)

After training, from the **epoch-40 checkpoints**:

1. `‖alpha_A‖`, `‖alpha_B‖`, `‖alpha_pair‖` (L2 and per-task profile).
2. Mean absolute **main-effect** residual `|alpha_A*mean_A + alpha_B*mean_B|`.
3. Mean absolute **pair** residual `|alpha_pair * pair_statistic|`.
4. Fraction of assays with a non-negligible pair coefficient, frozen here as
   `|alpha_pair[t]| >= 1e-2` (absolute logit units).
5. `pair_cross` and `pair_self` values on the **fixed 256 Validation
   molecules** defined in §12.1.
6. A **diagnostic-only** ablation of the trained `P`: full logits vs the same
   trained weights with `alpha_pair <- 0` at inference. Report the AP change and
   the logits difference. This ablation is **not** used to select a training
   configuration or checkpoint, and an AP change does **not** by itself
   establish a chemical causal mechanism — only that the branch contributes to
   the model's predictions.

### 12.1 Fixed diagnostic molecules (label-free, result-free selection)

The first **256 rows of `task.valid_rows` in the official split's own row
order** (`valid_rows[0:256]`), frozen by this protocol before any run. The row
ids and their SHA-256 are recorded in `analysis/evidence_mechanism_audit.json`.
No label and no model output participates in the selection.

## 13. Decision rules (pre-registered)

```
AP_W2 = 0.2618669181
AP_P  = P epoch-40 official Validation AP
AP_S  = S epoch-40 official Validation AP
D_base = AP_P - AP_W2
D_mech = AP_P - AP_S
```

| label | rule |
|---|---|
| `PROMISING_CROSS_ATOM_EVIDENCE` | `D_base >= +0.005` **and** `D_mech >= +0.003` **and** mean `(P − W2)` over the 107 assays with ≥20 Validation positives `>= +0.003` **and** `P` improves `>= 60/107` high-support assays |
| `CAPACITY_WITHOUT_CROSS_ATOM_ADVANTAGE` | both `P` and `S` beat W2 but `P` is not clearly better than `S` |
| `WEAK_OR_MIXED` | `P` gains on W2 but misses the full promising gate, or only low-support assays clearly benefit |
| `NO_CLEAR_GAIN` | `D_base < +0.002` and no broad high-support gain |
| `REGRESSION` | `D_base <= -0.002` |

These thresholds are **resource decisions, not significance tests**. The final
report must state, explicitly:

```
single-seed architecture exploration
historical W2 baseline
same-round paired P/S mechanism control
not independently confirmed
```

and must not use previously exposed historical Test numbers for candidate
selection.

## 14. Required per-task statistics

127 scorable assays (task 45 has no Validation positives):

```
P − W2, S − W2, P − S
number improved / regressed, mean and median AP delta,
top-5 / top-10 concentration of positive delta
strata by Validation positives:  >=20 (107 assays) · 5–19 (18) · <5 (2)
```

Also reported: whether Train AP keeps rising while Validation stalls; whether
`P` is broadly better than `S`; the measured training-speed difference between
the arms; the peak memory and runtime overhead of the new second-order
reduction; the three Validation curves; and the full provenance of recipe,
code, checkpoints and data order.

## 15. Deliverables

```
PROTOCOL.md  README.md  REPORT.md
code/cpe_common.py code/cpe_model.py code/cpe_train.py code/cpe_audit.py code/cpe_compare.py
tests/test_cpe_model.py tests/test_cpe_training.py
analysis/architecture_audit.json analysis/initialization_parity.json
analysis/paired_training_audit.json analysis/p_epoch40.json analysis/s_epoch40.json
analysis/paired_comparison.json analysis/per_task_deltas.csv
analysis/evidence_mechanism_audit.json analysis/gpu_ledger.json analysis/decision.json
runs/P/ runs/S/            (git-ignored; full checkpoints + resume state + SHA-256)
```

## 16. Explicitly out of scope

No re-training of W2, no W2 seed 1, no extra architecture, no seed 1 for `P`/`S`,
no hyper-parameter search, no lr/loss/regularisation change, no larger widths, no
new motif/path/radius/S2 gate, no iterative message passing, no cross-atom
attention/QKV, no Test use, no best-epoch substitution, no data subsetting, no
length extension, no follow-up run started automatically.

A negative result is an acceptable and expected outcome.

## 17. Amendments (append-only)

| # | date | change | reason |
|---|---|---|---|
|---|---|---|
