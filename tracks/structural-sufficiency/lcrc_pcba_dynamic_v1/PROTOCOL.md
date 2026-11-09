# PROTOCOL — LCRC-W2-DYNAMIC v1

Round: `lcrc-pcba-dynamic-v1`
Status of this document: **frozen before the first formal training step.**
Anything that changes after that is recorded in `REPORT.md` under
"Incidents and deviations", not edited here.

## 0. Scope and one-sentence question

> After the existing LCRC-W2 one-shot S1/S2 chemical-structure encoding, do two
> layers of real chemical-bond message passing that read the neighbours'
> *learned hidden states* meaningfully improve official MolPCBA scaffold-split
> predictive performance?

This round is a **diagnostic boundary experiment on the "no iterative MP"
constraint of LCRC**, not a next-generation LCRC method and not a
capacity-matched architecture comparison. Exactly **one** model is trained
formally: `W2-DYNAMIC`, seed 0, 40 epochs. No STATIC control, no W2 re-training,
no re-tuning, no additional seeds, no next-round experiment.

Frozen historical references (never retrained here; Test logits reused from the
frozen BOC round artifacts):

| Model | Parameters | Best Valid AP | Best-Valid Test AP | Valid@40 | Test@40 | GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 994,324 | 0.265273 | 0.253617 | 0.261867 | 0.250985 | 1.3949 |
| RC-REAL | 1,051,022 | 0.267718 | 0.255941 | 0.266950 | 0.256166 | 1.9186 |

## 1. Inherited, read-only code

* `lcrc_pcba_backbone_scaling_v1/code/bs_model.py` — `BSModel` (the W2
  architecture at scale 2), `build_model`, `permute_atoms`, `count_module_calls`.
* `lcrc_pcba_backbone_scaling_v1/code/bs_common.py` — W2 width dict, frozen
  from-scratch recipe, `scheduled_lr`, fingerprints, environment/provenance.
* `lcrc_pcba_backbone_scaling_v1/code/bs_train.py` — the audited training step
  shape (masked BCE, clip 5.0, AdamW) re-implemented with identical semantics.
* `lcrc_pcba_structural_gain_sprint_v1/code/sg_model.py` — `MessageFFN`, the
  already-validated pure-bond message formula (reused unchanged, at width 128 /
  hidden 256 instead of 64 / 128) and `initialize_new_modules`' private-generator
  initialisation **rule**.
* `lcrc_pcba_boc_v1/code/boc_test.py` + `analysis/test_prediction_manifest.json`
  — the frozen Test logits of W2/RC-REAL and the freeze/seal/score pattern.
* `lcrc_pcba_continuation_recipe_v1/code/cc_data.py` — the only code path that
  may open the official Test labels (`read_official_test_labels`) and the
  freeze seal.

The historical **V** model also contained a per-graph graph-context feedback
term. That term is **not** part of this round and its historical gain is never
attributed to bond message passing.

## 2. Model: `W2-DYNAMIC`

### 2.1 Main path

Created from the official W2 **seed-0 constructor** (`bs_model.build_model("W2",
...)`), i.e. the untrained W2 with the historical initialisation; no trained W2
checkpoint is loaded and no W2 parameter is frozen.

```python
a, b, scal = base._encode(batch)
h0 = base.drop(base.local(batch, a, b, scal))     # exactly one encode, one local
h1 = dynamic_block_1(h0, batch)
h2 = dynamic_block_2(h1, batch)                    # reads h1, never h0 again
logits = base.post_local(h2, batch)                # unchanged W2 readout
```

`h0, h1, h2` are all `[N_atoms, 128]`. `drop` and `local` are each called
exactly once; `post_local` is the inherited W2 readout (per-atom `f` + `rho` +
4-head attention pooling, 128-task head).

### 2.2 Dynamic bond messages

`edge_index = batch["bond_index"]` (`[E,2]`, each physical bond once),
`bond_feat = batch["bond_feat"]` (`[E,13]`). Two directions:

```python
src = cat(u, v); dst = cat(v, u)
edge_msg = bond_proj(cat(bond_feat, bond_feat))     # Linear(13,128), shared by both layers
```

Aggregation is strictly intra-molecular (edges come from `bond_index`).
No new artificial edge, no virtual node, no cross-molecule read.

### 2.3 Per-layer MessageFFN (two independent parameter sets)

For input `h` (the same `sg_model.MessageFFN` object, widths made explicit):

```python
z    = LayerNorm(128)(h)
msg  = zeros_like(z); msg.index_add_(0, dst, SiLU(z[src] + edge_msg))
delta = fc2(SiLU(fc1(z + msg)))                     # fc1: 128->256, fc2: 256->128
h_next = h + delta
```

* neighbour **sum**, no degree normalisation;
* no attention, no graph context, no extra pooling, no extra task head;
* `bond_proj` shared between layers; `blocks[0]`, `blocks[1]` independent;
* zero-bond graphs: `msg` is exactly zero and the forward stays finite.

### 2.4 Initialisation

* W2 part: the historical `bs_model.build_model("W2", seed=0)` state, verified
  per tensor; the new modules never touch W2's public initialisation RNG.
* New modules: private generator `seed*1_000_003 + 17` (the historical
  `sg_model` rule), `Linear` uniform `±1/sqrt(in_features)`, `LayerNorm`
  ones/zeros; `fc2.weight` and `fc2.bias` of **both** layers zeroed.
* Therefore, before any update, `W2_DYNAMIC(h) == W2(h)` exactly (eval mode;
  numerical tolerance 1e-4 for CUDA scatter reduction order).
* This is **not** frozen-residual training: both dynamic layers and every W2
  parameter are optimised from the first step. `fc2 = 0` only means that in
  step 1 the upstream dynamic parameters cannot yet receive gradient; this is
  audited at steps 1, 3, 10, 100.

### 2.5 Parameter budget (must match exactly)

| Component | Parameters |
|---|---:|
| W2 (trunk 848,096 + readout 146,228) | 994,324 |
| 2 × MessageFFN(128→256→128) incl. LayerNorm | 132,352 |
| `bond_proj` (13→128) | 1,792 |
| **total** | **1,128,468** |

If the measured total differs, the implementation is fixed before training; the
layer widths are never adjusted.

## 3. Correctness and mechanism gates (all before the first formal step)

1. W2 part bitwise identical (per tensor) to the W2 seed-0 constructor.
2. Total parameters exactly 1,128,468 (per-tensor audit).
3. Epoch-0 logits equal the untrained W2 logits, tolerance 1e-4.
4. Atom renumbering leaves logits unchanged (1e-4).
5. Reversing the stored direction of every physical bond leaves logits unchanged (1e-4).
6. Two molecules in one batch are fully isolated (perturbing one leaves the other's logits unchanged).
7. Both layers consume bidirectional real chemical bonds.
8. Layer 1 really reads `h0[src]`; layer 2 really reads `h1[src]` (hook capture comparison).
9. Chain-graph dependency test (`A—B—C`): changing `h0_A` must not change `h1_C` but must change `h2_C` after the second layer; executed with the real two-layer forward and deterministic non-zero test weights.
10. Independent pure-Python aggregation reproduces the module's `msg`.
11. No non-finite logits at initialisation, after training, or after checkpoint restore.
12. Every new tensor receives a usable gradient (audited at steps 1, 3, 10, 100).
13. Inherited Train-only normalisation, masked BCE, task order.
14. Official molecule row ids, label mask and Train/Valid/Test partition identical to the frozen assets.
15. No additional feature or non-official supervision enters the model (the batch keys the dynamic forward reads are exactly the keys W2 reads).

## 4. Training protocol (single formal run)

* `ogbg-molpcba`, frozen cache, official scaffold split.
* Train 350,343 / Valid 43,793 / Test 43,793, 128 tasks, seed 0, 40 epochs.
* batch 128, AdamW, weight decay 1e-3, betas (0.9,0.999), eps 1e-8.
* 1-epoch linear warmup 3e-5 → 3e-4, then per-step cosine 3e-4 → 1e-5 over the
  40-epoch horizon (109,520 updates).
* dropout 0.10 (the inherited W2 `drop`), gradient clip 5.0, FP32, masked BCE.
* The **same** 40-epoch batch-order plan as W2 seed 0
  (`global_plan_sha256 = 06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6`).
* One parameter group; W2 and dynamic parameters share the same LR. No freeze
  phase, no group-specific LR.
* Official Validation AP every epoch; `checkpoint_best.pt` (strict argmax of
  Validation AP, earliest wins ties) and `checkpoint_epoch40.pt` saved, plus a
  resumable `checkpoint_last.pt`.
* Recorded: per-epoch Train BCE, Valid AP, Valid BCE, step timings, GPU memory,
  gradient statistics of the new modules, per-epoch RNG hashes.
* Not permitted: changing LR, depth, hidden size, dropout, weight decay, epoch
  count, train subset, or the recipe in response to intermediate Validation AP.

## 5. Is the dynamic branch actually used?

Every epoch, on one fixed small Validation batch (first 32 official Validation
rows, eval mode, deterministic):

```text
R1 = RMS(h1 - h0) / RMS(h0)
R2 = RMS(h2 - h1) / RMS(h1)
Rtotal = RMS(h2 - h0) / RMS(h0)
```

plus `bond_proj` gradient norm, `blocks.{0,1}.fc{1,2}` gradient norms, `fc2`
weight norms, and the finite fraction of the dynamic output.

If, after training, the dynamic update is still numerically ~zero, the round is
reported as **`DYNAMIC_BRANCH_SUPPRESSED`** and *not* as a test of dynamic
capability. `Rtotal` magnitude is a usage indicator only — never evidence of
predictive ability.

Post-training, on the frozen best-Validation checkpoint, Validation-only
interventions (no retraining, no checkpoint reselection, no tuning):

* `message-off` (message term forced to zero) — sensitivity check;
* `message-shuffle` (inside each molecule, permute only the `src` of the dynamic
  edges; `dst` multiplicity, `h0` and all W2 static inputs unchanged).

Both create out-of-distribution inputs; they bound the model's sensitivity to
the real message correspondence and are **not** causal proof of a
message-passing generalisation gain, and they are never equivalent to the
historical independently-trained STATIC control.

## 6. GPU budget and scheduling

* Host `res`, single safe idle A100 (`res-gpu1`); GPU0 is inspected first and is
  never used while another user's process occupies it.
* One training job only; no second model in parallel.
* A ~100-real-batch profiling pass (real train step) precedes the formal run and
  projects the 40-epoch cost.
* Target ≤ 2.5 A100 GPU-hours, hard cap 3.0. Extra gates/inference stay small.
* If the projection exceeds the hard cap, numerically equivalent implementation
  optimisation is attempted; if it still cannot fit → report `RESOURCE_STOP`.
* Forbidden shortcuts: 20-epoch training, fewer training samples, structural
  changes.
* Interruptions are resumed from `checkpoint_last.pt` and the affected gates are
  repeated; incidents are recorded.

## 7. Selection and the official Test evaluation

The official Test split is already exposed (historical engineering comparison
set). Label `EXPOSED_TEST_ENGINEERING_COMPARISON_SET`. Only Train and Validation
labels are read during training.

After training:

1. unique best checkpoint by highest Validation AP; freeze best and epoch-40;
2. verify the W2 / RC-REAL reference identities (frozen BOC artifacts);
3. generate `W2-DYNAMIC` Test logits **before** any Test label is opened;
4. seal checkpoints, Test logits and the official Test row-id hash in a freeze
   manifest;
5. only then read the official Test labels through
   `cc_data.read_official_test_labels` and score with
   `ogb.graphproppred.Evaluator("ogbg-molpcba")` plus an independent per-task
   sklearn AP recomputation;
6. no model, epoch, seed or recipe change after any Test score.

## 8. Decision labels (fixed in advance)

Primary quantity: `ΔTest = Test AP(W2-DYNAMIC best-Valid) − Test AP(W2 best-Valid)`,
with Validation direction, high-support task coverage (≥20 positives) and the
dynamic-usage audit as supporting evidence.

| Label | Condition |
|---|---|
| `STRONG_DYNAMIC_SIGNAL` | ΔTest ≥ +0.010, ≥70/106 common high-support tasks improved, high-support mean Δ clearly positive (ref +0.006), Validation direction positive, dynamic branch confirmed used |
| `POSSIBLE_DYNAMIC_SIGNAL` | roughly +0.005…+0.010, or clear high-support improvement offset by low-support tasks; not yet sufficient |
| `NO_CLEAR_DYNAMIC_GAIN` | ΔTest ≈ ±0.002…0.003 or no broad task improvement |
| `DYNAMIC_REGRESSION` | clear degradation not caused only by a few very-low-support tasks |
| `MECHANISM_OR_OPTIMIZATION_INVALID` | no effective gradient / no cross-layer state dependency / a correctness gate failed |
| `DYNAMIC_BRANCH_SUPPRESSED` | dynamic updates remain ~zero after training |

Values between the bands are reported as measured evidence, not forced into a
label. One seed and an exposed Test split: **no claim that "Dynamic is proven
better than STATIC"**, and thresholds are engineering decisions, not
significance tests.

## 9. Interpretation boundaries

No capacity-matched STATIC control is trained in this round. Therefore even a
clear win over W2 cannot attribute the gain to iterative message passing alone
rather than to extra parameters / extra non-linear layers / joint adaptation.
Two MP layers also extend the effective receptive range of W2's local
representation, so this round cannot separate (a) longer-distance information
reachability, (b) learned neighbour-context interaction, (c) added capacity.
No STATIC, one-layer, four-layer, other-GNN or reproduction arm may be started
here.

## 10. Deliverables and stop rule

`PROTOCOL.md`, `README.md`, `code/`, `tests/`, `analysis/`, `REPORT.md` under
`tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/`.

`REPORT.md` separates **Observed** (scores, curves, per-task deltas, dynamic
dependency audit), **Interpretation** (which explanations fit, which cannot be
excluded) and **Decision** (worth continuing, and under which confirmation
conditions), and answers the five mandatory questions of the brief.

After the single training run, the freeze/score pass and the report, the round
**stops**. No next-round experiment is started automatically.
