# PROTOCOL — zinc bond-composer development v1

| field | value |
|---|---|
| protocol id | `zinc-bond-composer-development-v1` |
| round directory | `tracks/structural-sufficiency/zinc_bond_composer_development_v1/` |
| code | `.../zinc_bond_composer_development_v1/code/` |
| config | `.../configs/zinc_bond_composer_development_v1.yaml` |
| status at freeze | `FROZEN_PRE_GPU` |
| kind | internal small-budget architecture development |
| relation to closed rounds | independent side round; does not reopen S/T/R, does not modify Phase 2b or any frozen product |
| official valid/test | never loaded, parsed or predicted |
| Monitor | repeatedly used development set; not a blind test and not independent confirmation |
| budget | 0.75 A100-hours total occupancy, both GPUs summed |

This protocol is frozen before the first GPU second of this round. It is a
development probe, not a confirmatory experiment. It does not claim a novel
architecture, does not compare with GRIT/ESA+ paper test numbers, and does not
run an official-valid or official-test evaluation.

## 1. Question

The closed object-token S/T/R round replaced P3's 321→256→256→1 reader with a
612→96→96→1 reader and mixed three independent encoder representations
(`a`, `b`, `w`). Its negative result cannot separate "reader replaced" from
"representation interface changed".

This round asks a narrower, related question:

> If the **complete original P3 reader is retained**, and interaction is placed
> only on a **single shared per-bond composer representation** `c_e`, does a
> per-bond nonlinear FFN branch or a per-bond attention branch show a useful
> development signal against original P3?

The old `p3_bond_relation_bridge_v1` already tested a different related idea:
retain P3 and add a one-hop bond-pair MLP residual, with no gain. This round is
**not** a never-tested new main line and is not a novelty claim. It changes the
interface (`c_e` from the shared composer, global content attention, one
nonlinear pre-LN block) but remains a related exploration.

## 2. Data and lineage

* Fit-8000 / Monitor-1000: the frozen phase1b SSRC split.
  `fit_rows_sha256 = 7f66a23e…`, `monitor_rows_sha256 = dc6cc4e9…`; split file SHA-256 `fbca5fd5…`.
* Objects: the phase1e Fit-8000 and Monitor-1000 object caches, hashes
  `bfbf9653…` and `7a718506…`.
* Standardiser: the frozen Fit-only phase1e standardiser; no recomputation.
* Encoder: the phase1e seed-0 masked-SSL encoder trained on the **same Fit-8000
  rows only** (`pretrain_s0.pt`, SHA-256 `0d045e60…`).
* Supervised start: `bp.build_bjoc_null(0)` + `mp.load_encoder_state` with that
  encoder. The completed P3 soup is never used as an initialiser.
* Only Fit raw-y labels enter the supervised loss. Monitor raw-y labels are read
  repeatedly for development curves and metrics. Internal Reserved rows and
  official valid/test are never loaded.
* No new SSL, no external data, no target change, no new pretraining.

If any frozen input above is missing or hash-mismatched, the round stops and
reports the gap; it does not substitute a new encoder or rebuild a new SSL.

## 3. Models

### B0 — original P3

All 413,505 parameters, including the atom/bond/junction encoders, Null
grouping, `group_composer`, and the original 321→256→256→1 reader. B0 is
retrained in this round so the training and Monitor curves share one execution
regime with B1/B2.

### B1 — P3 + per-bond FFN branch (capacity control)

### B2 — P3 + per-bond key-attention branch (candidate)

For both B1/B2, start from the shared composer output per physical bond:

```
c_e = P3_group_composer(...)                       96
h0_e = Linear(96 -> 48)(c_e)
```

One independent pre-LN block:

```
U  = H + MIX(LN1(H))
H1 = U + FFN(LN2(U))
```

* `LN` affine=True, eps=1e-5; no final LayerNorm.
* `FFN`: 48→96→48, both layers biased, SiLU, dropout 0.
* `MIX` B1: `gamma ⊙ FFN_extra(x)`, `FFN_extra` 48→96→48, `gamma` 48-d init 1.
* `MIX` B2: standard 4-head Q/K/V/output self-attention, head dim 12, all valid
  physical-bond keys of the same molecule visible, dropout 0.
* No relation bias, distance bucket, atom/junction tokens, extra reader or
  final LN.
* `c'_e = c_e + Linear_out(48→96)(h1_e)`, with `Linear_out` weight and bias
  zero-initialised.
* Only `sum(c_e)` is replaced by `sum(c'_e)`. Reader input stays 321-d; the
  reader structure and all original parameters are retained. W/V are computed
  exactly as in P3.
* `c_e` contains the full-graph Null context. It is not a strictly local or
  causally interpretable per-bond contribution.

A molecule with no bonds returns an empty branch input / zero aggregate and
must not produce NaN. Padding keys never enter attention or aggregation.

## 4. Parameters and initialisation

| item | parameters |
|---|---:|
| input projection 96→48 | 4,656 |
| block: LN1 + MIX + LN2 + FFN | 18,960 |
| output projection 48→96 | 4,704 |
| **new per branch** | **28,320** |
| B0 | 413,505 |
| B1 | 441,825 |
| B2 | 441,825 |

B1's `FFN_extra` (9,360) + `gamma` (48) equals B2's MHA (9,408). Nominal
parameter equality does not imply equal effective capacity or compute.

* B0/B1/B2 original P3 parameters are bit-identical at step 0.
* B1/B2 input projection, LN1, LN2, second FFN and output projection are
  bit-identical; dedicated RNG streams initialise the modules.
* Zero initialisation only guarantees a common starting function. The first
  update can give the output projection a gradient while branch-internal
  gradients are zero; from the second update all branch groups must receive
  non-zero finite gradients.

## 5. Recipe

Exactly the matching internal R3 recipe:

| item | value |
|---|---|
| Fit rows | 8,000 |
| batch | 128 |
| steps/epoch | 63 |
| epochs / updates | 360 / 22,680 |
| loss | raw-y L1 |
| optimiser | Adam, lr 1e-3, per-step cosine → 1e-5 |
| weight decay | 1e-5 |
| grad clip | 5.0 |
| soup | mean of the last 5 epoch parameter states |
| early stopping / valid selection | none |
| precision | FP32, no AMP, no `torch.compile` |
| batch order | `torch.Generator(seed + 101)`, one permutation per epoch |

The seed-0 batch-order hash must equal the frozen R3 hash
`d142e307…`. Evaluation uses an independent loader and no RNG; it must not
change the training batch sequence.

## 6. Development curves and checks

Every 20 epochs (plus epoch 1), report full Fit-8000 raw-y MAE and Monitor-1000
raw-y MAE, using the same evaluation frequency for every model. The fixed
last-5 soup is the primary table. A best-Monitor epoch may be recorded as a
development observation, but it is not the primary result and is not an
independent validation.

Minimum CPU checks before GPU work:

1. parameter counts and reader layout;
2. bit-identical canonical P3 state across B0/B1/B2; bit-identical B1/B2 common
   branch modules;
3. B1/B2 at zero output projection reproduce B0 predictions and reader inputs;
4. atom renumbering, bond-key permutation and batch-composition invariance;
5. empty-bond molecules and batches produce finite outputs, no NaN;
6. two-update gradients: output projection at step 1, all branch groups from
   step 2, all finite;
7. Fit-only labels in the loss; Monitor labels only in evaluation; Reserved and
   official valid/test not loaded;
8. batch-order hash equals the frozen R3 hash.

## 7. Budget and run matrix

* Cap: **0.75 A100-hours**, sum of occupancy on both GPUs, including smoke,
  three formal runs, curves, evaluation and any failure/retry.
* Smoke: 2 epochs on a 1,024-row Fit prefix for B0/B1/B2. Projection uses the
  steady-state mean over updates 3–16, a 1.20 margin, the smoke wall and a
  300 s reserve. If the projection does not fit with at least 10 % headroom, the
  round stops and reports implementation plus projection; it does not shorten
  epochs or add budget.
* Formal: exactly B0, B1, B2, seed 0, one fixed soup each. No seed 1, no
  full-10k refit, no extra model from leftover budget.
* Schedule: two GPUs may run different single-GPU models concurrently; one GPU
  runs at most one training process at a time.

## 8. Primary table and decision

Primary: fixed last-5 soup raw-y MAE on Fit-8000 and Monitor-1000. Report
B0−B1, B0−B2 and B1−B2 with the sign convention `MAE(left) − MAE(right)`
(positive means the right model is better), plus wall time and parameters.

Robustness is limited to two pre-declared Monitor slices:

* ordinary region `y > -3`;
* drop the 10 Monitor rows with the largest B0 soup error, fixed by this
  round's B0, not per candidate.

A **development follow-up signal** requires all of:

* B2 vs B0 Monitor mean gain ≥ 0.003;
* B2 better than B1;
* ordinary-region gain positive;
* drop-B0-worst-10 gain positive.

This is a single-seed development decision line, not a confirmation or a
universal constant. If only a best-Monitor epoch is good while the fixed soup is
not, the conclusion is a training-selection/formulation issue, not "B2 beats
P3". If B1 helps but B2 adds nothing, the gain cannot be attributed to
attention. If there is no practical signal, stop this candidate without
changing rank, depth, heads or initialisation scale.

## 9. Interpretation boundary

Retaining the full reader and changing the interface from "sum of composer
outputs" to "sum of branch-corrected composer outputs" still changes more than
one thing relative to a hypothetical ideal ablation. Even a positive result
would not isolate "reader retained" from "representation interface changed".
The next step, if any, would be a separately designed ablation. No official
valid/test number is produced and no novelty claim is made.
