# Pre-registration — `e2e_dictenv_capacity_localization_v1`

Workstream Z (ZINC dictionary-environment line), CPU only, no GPU, no SSH, no
remote compute.  The official ZINC test split is **never** loaded
(`official_test_loaded = false` in every payload).

Round name: `e2e_dictenv_capacity_localization_v1`
Scientific question: **which expressive capacity does the current
dictionary-based structure-semantic model actually lack?**

* structure-semantic fusion capacity (candidate **F**),
* static environment-environment composition capacity (candidate **R**),
* graph-level invariant readout capacity (candidate **G**).

Implementation (frozen with this document):
`tracks/ksvd/experiments/luyin16/e2e_dictenv_capacity_localization_v1.py`;
focused tests:
`tracks/ksvd/tests/test_e2e_dictenv_capacity_localization_v1.py`.

Nothing in this document may be edited after the first screening process
starts.  No threshold, width, rank, seed, horizon or tie-break may be changed
after seeing a result.

---

## 0. Frozen context (not re-opened)

Frozen base:

```text
CAP-BASE = CSSD-q1
  C6 clean mask (no graph chemistry marginal / histogram, no counts,
                 no path_count relation)
  common structural coordinate c1 (train-only U, q = 1)
  + q1-orthogonal residual sparse dictionary (K = 32, s = 8, IHT-10)
  paired node structure-semantic binding
  paired edge structure-semantic binding
  full clean relation: distance + overlap + boundary
  first + second moment readout (node unary + per-bucket pair moments)
```

Frozen references (never re-trained, never used for selection beyond the
explicit bands in section 10):

| item | value |
|---|---|
| CAP-BASE CSSD-q1 seed-0 soup valid MAE | `0.13002798487985273` (recomputed and asserted in preflight) |
| FINAL-CLEAN sparse seed-0 soup valid MAE | `0.12849851670576026` (historical performance reference only) |
| CAP-BASE parameter count | `97 727` |
| CSSD soup checkpoint | `results/e2e_dictenv_common_subspace_dictionary_v1/training/checkpoints/CSSD-Q1-seed0_soup_state.pt` |
| train-only q1 subspace | `results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json` |
| data | frozen phi65 cache, official ZINC train 10 000 / valid 1 000 |

The CSSD seed-0 MAE being `+0.001529` *above* the historical FINAL-CLEAN sparse
number does **not** authorize a rollback of the structural representation; the
CSSD representation change was already supported at representation level and is
task-neutral, so it stays as the capacity base.

Not re-opened this round (frozen conclusions): sparse vs dense, IHT-10 vs
IHT-30/OMP, dictionary concentration / common-subspace q / q2, K, s, graph
chemistry shortcut, node/edge paired vs independence, distance-only relation,
explicit count removal.

---

## 1. Hard constraints

* CPU only.  `CUDA_VISIBLE_DEVICES=""`; every process constructs
  `torch.device("cpu")`; `cpu_only_guard` is called at every stage entry.
* No GPU, no CUDA training, no SSH, no remote compute, no `res` host.
* Official test never loaded; `official_test_blocker` is called on every
  payload.
* `docs/luyin/luyin19.txt` is not modified, deleted, moved or `git add`ed.
* Screening and full training use train + official-valid only.
* No hyper-parameter search: every candidate has exactly ONE frozen
  architecture; width/rank/head/block values below are final.

---

## 2. Budget philosophy

```text
at most 4 short adaptation runs (M0, F, R, G), each exactly 40 epochs
+ at most 1 full from-scratch seed-0 run (320 epochs)
= at most 480 training epochs total
```

The short runs are capacity *screening*; only a candidate that passes the
frozen gate in section 8 buys the single full 320-epoch trajectory.  If nobody
passes, the round stops after the 160 short epochs (screening is parallel, so
this is much less wall clock than 160 sequential epochs).

CPU concurrency: 16 cores / ~27 GiB RAM; wave 1 = `M0, F, R` (3 processes ×
4 threads), wave 2 = `G` (1 process × 4 threads).  The full winner runs as a
single process × 4 threads.  Concurrency is reduced automatically if memory or
CPU contention is observed.

---

## 3. Candidates (one frozen architecture each)

All candidates start from the *same* CAP-BASE seed-0 Top-5 soup checkpoint and
change exactly one stage of the frozen forward pass.  Everything else —
shell/shellpair assignments, CSSD code, `c1`, dictionary, relation, pair
encoder input, readout, optimizer, loss — is unchanged.

### 3.1 Candidate F — multi-rank factorized structure-semantic residual fusion

Screening priority is highest for F (the mentor's current direction).

Current binding: ``u = (z W_S) odot (q W_C)`` — one output channel per rank-1
bilinear interaction.  Candidate F adds a **residual multi-rank factorized
cross** with a frozen head count; it does NOT replace the product form with an
`MLP([z; q])`.

Frozen constants: `FUSION_HEADS = 4`, `FUSION_HEAD_DIM = 24`,
`FUSION_PROJ_INIT = 0.01`, `FUSION_INIT_SEED = 20261001`, divisor
`sqrt(d_h) = sqrt(24)`.

Node (per occurrence, `z` = structural coordinate row including `c1`,
`q` = atom-category one-hot row):

```text
u_base = (z W_A_S) odot (q W_A_C) / sqrt(96)
u_h    = (z W^h_NS) odot (q W^h_NC) / sqrt(24)          h = 1..4
u      = u_base + W_NP [u_1; u_2; u_3; u_4]             (no bias)
```

Edge (per bond occurrence, `g = [z_u + z_v; |z_u - z_v|; z_u odot z_v]`,
`b` = bond one-hot):

```text
e_base = (g W_E_S) odot (b W_E_C) / sqrt(48)
e_h    = (g W^h_ES) odot (b W^h_EC) / sqrt(24)          h = 1..4
e      = e_base + W_EP [e_1; e_2; e_3; e_4]             (no bias)
```

Output widths stay exactly `D_A = 96` (node slot) and `D_E = 48` (edge slot).
Shell/shellpair pooling is unchanged (per-occurrence sum via `index_add_`).
F does not touch pair composition or the readout.

### 3.2 Candidate R — stronger static environment composition

Only ``(E_i, E_j, r_ij) -> q_ij`` changes.  The current pair encoder output
`h^(0)` (width `PAIR_HIDDEN = 16`) receives **exactly two** residual blocks:

```text
z^(l)   = W1^(l) LN(h^(l))                               (width 256)
z'^(l)  = (1 + tanh(gamma^(l)(r))) odot z^(l) + beta^(l)(r)     [FiLM]
h^(l+1) = h^(l) + W2^(l) silu(z'^(l))                    (back to 16)
```

`r` is the existing encoded relation (`relation_encoder` output, width 16);
the relation encoder is called once per pair.  Frozen constants:
`RELATION_BLOCKS = 2`, `RELATION_HIDDEN = 256`,
`RELATION_RESIDUAL_INIT = 0.05`, `RELATION_INIT_SEED = 20261002`.
`gamma` and `beta` start at exactly zero (identity modulation).

Static contract (hard): each pair is computed exactly once; there is no
pair → node/environment write-back, no message passing, no center update, no
relation refresh, no second pair pass, no recurrence.  R does not touch the
environment, the CSSD code, or the readout.  The frozen fill/zero mask
semantics around the pair encoder are unchanged; only the pair state is
enriched.

### 3.3 Candidate G — learned invariant graph readout

Only the graph representation changes.  The existing first/second moments and
the existing global/topology blocks are kept; two gated DeepSets summaries are
appended:

```text
a_i    = sigmoid(g_E(E_i));  v_i    = v_E(E_i)
a_ij   = sigmoid(g_P(q_ij)); v_ij   = v_P(q_ij)
H_E    = sum_i  a_i  odot v_i  / (sum_i  a_i  + eps)
H_P    = sum_ij a_ij odot v_ij / (sum_ij a_ij + eps)
repr   = [ existing moments ; H_E ; H_P ; existing global/topology ]
```

Frozen: `READOUT_SUMMARY_DIM = 192`, `READOUT_SUMMARY_EPS = 1e-6`,
`READOUT_READER_INIT = 0.01`, `READOUT_INIT_SEED = 20261003`.

The reader input grows from 302 to `302 + 2*192 = 686`; the CAP-BASE reader
weights occupy the first 302 input columns **bit-identically** and only the
appended columns start near zero.  Permutation invariant by construction; no
attention between nodes, no pairwise attention, no Set Transformer, no message
passing.

---

## 4. Parameter budget rule (frozen)

```text
CAP-BASE                        = 97 727
preferred candidate total       = 120 000 .. 160 000
hard ceiling                    = 250 000
max/min added-parameter ratio   <= 1.5
```

Exact construction-time counts (verified by focused test
`test_parameter_budget_rule`):

| candidate | added | total | relative |
|---|---|---|---|
| F | 29 568 | 127 295 | +30.26 % |
| R | 34 400 | 132 127 | +35.20 % |
| G | 30 336 (25 344 summaries + 4 992 reader columns) | 128 063 | +31.04 % |

Ratio `max/min = 1.1634 <= 1.5`.  Widths were chosen **once** from the closed
parameter formulas (stated above) to land in the preferred band; there is no
width/rank/head/block sweep and no performance-based width selection.

---

## 5. Initialization rule for warm screening (frozen)

All four arms (including M0) are built with the exact CAP-BASE initialization
stream: `torch.manual_seed(seed)` advances identically to `CSSDModel`, and
every capacity parameter is initialized from an explicit local generator inside
`preserved_global_rng()`, so the global RNG stream after construction is
bit-identical to CAP-BASE (focused test `test_global_rng_stream_matches_cap_base`).

New modules use:

* F heads: default `nn.Linear` Kaiming-uniform (`a = sqrt(5)`); residual
  projections `W_NP`, `W_EP`: `U(-0.01, +0.01)`;
* R: `W1` Kaiming-uniform, `W2` Kaiming-uniform scaled by `0.05`,
  `gamma = beta = 0` exactly (identity modulation);
* G: `g_*, v_*` default `nn.Linear` init; appended reader columns
  `U(-0.01, +0.01)`.

Then the CAP-BASE soup state is loaded with `load_capacity_warm_state`, which
copies every shared tensor bit-identically and only leaves capacity-module
parameters (and, for G, the appended reader columns) at their frozen init.

Step-0 audit (frozen before training, not a selection criterion):

* valid MAE, mean/max |Δprediction| vs CAP-BASE;
* `||grad||` of every capacity parameter on the first deterministic train
  batch, in eval mode so the audit cannot consume the dropout RNG stream.

The audit passes iff all new-module gradients are finite and strictly positive
and `step0 MAE <= CAP-BASE MAE + 0.03`.  Measured (full valid, this
implementation):

| candidate | step0 valid MAE | Δ MAE | mean shift | max shift | grads |
|---|---|---|---|---|---|
| F | 0.131102 | +0.001074 | 0.01053 | 0.05211 | 18/18 finite > 0 |
| R | 0.130860 | +0.000832 | 0.01948 | 0.12942 | 20/20 finite > 0 |
| G | 0.129941 | −0.000087 | 0.00112 | 0.00300 | 8/8 finite > 0 |

Fixing an initialization that fails the audit is allowed (architecture must not
change); no candidate in this round needed a fix.

---

## 6. Screening protocol (frozen)

Four arms, identical everything except the capacity module:

```text
M0  CAP-BASE continuation control (no capacity module)
F   section 3.1
R   section 3.2
G   section 3.3
```

Per arm, in one fresh process:

1. `p2run._seed_everything(0)`;
2. build the arm model (base init = CAP-BASE stream; capacity init = local);
3. `load_capacity_warm_state` from the CAP-BASE soup checkpoint;
4. step-0 audit (section 5);
5. **fresh** `Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)` —
   including M0; no optimizer state is inherited;
6. train loader `shuffle=True` seed `0 + TRAIN_SHUFFLE_OFFSET`, eval loader
   `shuffle=False` seed `0 + EVAL_SHUFFLE_OFFSET`, batch `128`;
7. loss `L1(y_hat, y) + H1_LAMBDA * reconstruction_loss`, `H1_LAMBDA =
   33.95873017865987`; gradient clip `5.0`;
8. exactly **40 epochs**; every epoch logs train MAE, valid MAE,
   reconstruction loss, total loss, wall clock, and the new-branch gradient
   norm;
9. retain the weight state of every epoch in the window `[21, 40]`;
10. screening primary metric: Top-5 (by valid MAE) weight soup over epochs
    21–40, evaluated on the full official-valid split;
11. secondary metric: mean valid MAE over epochs 31–40;
12. also record best valid over the 40 epochs.

Screening metrics are exactly these; no single lucky epoch decides.

---

## 7. Screening deltas

```text
M0     = soup MAE of M0
Delta_F = F_soup - M0_soup
Delta_R = R_soup - M0_soup
Delta_G = G_soup - M0_soup        (negative = candidate better)
```

and the same for the last-10 secondary metric.

---

## 8. Qualification gate and winner rule (frozen)

A candidate is recorded as `CAPACITY_SIGNAL` iff **all** hold:

```text
Delta_soup   <= -0.004
Delta_last10 <= -0.003
training stable and finite, no mechanism-breaking bug
```

`Delta_soup <= -0.010` upgrades the record to `STRONG_CAPACITY_SIGNAL`.
The full 320-epoch run is bought only when the gate fires.

Winner (frozen, never changed after seeing results):

1. if no candidate passes → `NO_CLEAR_CAPACITY_LOCALIZATION`; stop; write
   `full/NOT_RUN.json`; no 320-epoch run;
2. if exactly one passes → it wins;
3. if several pass: smallest screening soup MAE wins, unless the best two are
   within `0.002`, in which case the frozen tie order **F > R > G** decides
   (F is the mentor's structure-semantic-fusion question, R second, G is the
   generic-capacity augmentation).

Exactly one full run.  No combined candidate (`F+R`, `F+G`, `R+G`, `F+R+G`) is
trained this round; no seed 1/2; no extra candidate.

---

## 9. Full winner run (only if bought)

* one new seed-0 trajectory **from scratch** (not warm-start), exactly
  `320` epochs;
* identical frozen protocol to CSSD-q1 seed 0: same split, seed, data order,
  Adam 1e-3 / wd 1e-5, batch 128, clip 5.0, reconstruction objective,
  dictionary settings, K=32/s=8/IHT-10, C6 mask, Top-5 soup over the whole
  320-epoch curve;
* the only change is the winner capacity module (its own initialization
  parameters are part of the architecture);
* single process × 4 threads;
* early stop is allowed **only** for: NaN/Inf; training divergence;
  dictionary collapse; a capacity-module gradient permanently zero; or valid
  MAE worse than the matched historical CSSD curve by `> 0.03` for `>= 40`
  consecutive epochs **and** no improving trend (defined as: the minimum valid
  MAE of the last 10 epochs is not lower than the minimum valid MAE of the
  previous 10 epochs).  "It improved only 0.002 by epoch 80" is not a stop
  reason.

Reporting: winner, seed, params, best valid/epoch, Top-5 soup + soup members,
Δ vs CAP-BASE (0.130028) and vs FINAL-CLEAN sparse (0.128499), wall clock.
No comparison is made against historical GPU H1 numbers.

---

## 10. Full-run interpretation bands (frozen)

```text
M_W > 0.127            -> FULL_CAPACITY_GAIN_NOT_ESTABLISHED
M_W <= 0.125           -> CAPACITY_DIRECTION_SUPPORTED_SINGLE_SEED
M_W <= 0.120           -> NEW_PERFORMANCE_BAND_SINGLE_SEED
M_W <= 0.110           -> MAJOR_CAPACITY_BOTTLENECK_IDENTIFIED
```

This round does **not** require reaching the mentor's ~0.06–0.07 strong-method
band.  The question is whether a capacity direction moves the model out of the
0.12–0.13 plateau.  A single-seed result is never stated as SOTA, stable across
seeds, or final architecture.

If the full winner fails (`M_W > 0.127`), the record is
`SHORT_ADAPTATION_SIGNAL_DID_NOT_TRANSFER` and there is **no post-hoc rescue**
(no rank/width/block/epoch/lr change, no F+R combination).

---

## 11. Post-full mechanism audit (frozen, cheap)

Only if the full run completes; on the frozen winner soup checkpoint, using the
official-valid split only, with the winner's training mask (C6) merged into
every probe mask:

* **P1 residual dictionary dependence** — distribution-preserving row shuffle
  of `data.dict_phi` (equivalently the structural coordinate) across nodes
  within each evaluation batch; 3 seeds; Δ MAE.
* **P2 node structure-semantic correspondence** — P1-semantics
  assignment-preserving node shuffle (`audit.prepare_shuffles(..., "node")`,
  `AuditMask(use_node_shuffle=True)`); 3 seeds; Δ MAE.
* **P3 edge role/bond correspondence** — the same with
  `use_edge_shuffle=True`; 3 seeds; Δ MAE.
* **P4 relation dependence** — cross-pair row shuffle of the relation bundle
  (`audit.permute_pair_rows`: `all`, `distance`, `overlap`, `boundary`); 3
  seeds; Δ MAE.
* **P5 winner branch disable** — set the new capacity branch off on the frozen
  checkpoint (`capacity_off`); record Δ MAE and prediction shift.

Candidate-specific diagnostics (no diversity loss, no rescue):

* winner F: per-head activation norm, per-head gradient norm, head-to-head
  output cosine (are the 4 heads learning different interactions?);
* winner R: residual block contribution norm, block-1/block-2 ablation,
  relation-conditioned modulation norm;
* winner G: learned summary norm, node vs pair contribution, gate
  distribution/entropy, disable-summary Δ.

---

## 12. Forbidden this round

Sparse vs DenseTied; CSSD q2; dictionary penalties; new handcrafted topology
descriptors; ring feature additions; K/s/IHT changes; learning-rate /
weight-decay / dropout sweeps; rank / head-count / block-depth / readout-width
sweeps; F+R, F+G, R+G, F+R+G; seeds 1/2; official test.  No post-hoc candidate
or threshold change after the first screening process starts.

---

## 13. Result layout

```text
tracks/ksvd/results/e2e_dictenv_capacity_localization_v1/
    preregistration_snapshot.json
    parameter_budget.json
    preflight.json
    init_audit.json
    screening/
        m0/  fusion/  relation/  readout/
            result.json, curve.csv, soup.json
        screening_summary.csv
        decision.json
    full/
        checkpoints/       (epoch states + soup state when the run fires)
        curve.csv
        soup.json
        final.json
        NOT_RUN.json       (when the gate does not fire)
    mechanism/
        probes.csv
        candidate_specific.json
    analysis_tables.md
    summary.json
    REPORT.md
    DECISION.md
```

## 14. Frozen implementation rule

The preregistration hash is written into `preregistration_snapshot.json` in
`preflight`.  The implementation is frozen before the first screening process;
after that only clearly-labelled harness/provenance fixes are allowed, and they
may not change a decision.  Shared helpers may be touched only in a
backward-compatible way with the related focused tests re-run.

---

## 15. Scientific questions the final report must answer

1. Which of structure-semantic fusion / environment relation composition /
   graph readout shows the main capacity signal (or none)?
2. If F wins: is the single-rank / low-rank bilinear fusion a bottleneck?
3. If R wins: is the one-shot shallow static pair composition a bottleneck?
4. If G wins: are the fixed first/second moments a readout bottleneck?
5. Does the short adaptation signal reproduce in the from-scratch 320-epoch run?
6. Does the new capacity still depend on the dictionary coordinate,
   structure-semantic correspondence and relation structure?
7. What is the honest Δparams → ΔMAE?
8. Does the model leave the 0.12–0.13 plateau?

Interpretation discipline: if F wins, write "increasing factorized
structure-semantic bilinear rank improved capacity, supporting a bottleneck in
structure-semantic interaction capacity", not "multi-head is better" or any
combination claim from a single isolated winner.
