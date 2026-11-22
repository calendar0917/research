# REPORT — attachment_conditioned_response_algebra_pilot_v1 (ACRA-v1)

Status: **complete**.  One seed (0), no hyperparameter search, one uniform
horizon (160 epochs) for all three arms.  Test labels were never read; no Test
metric is reported.  Total compute: **0.328 A100 GPU-hours** (8.2 % of the
4-hour budget), three runs, at most two GPUs in parallel.

Round directory:
`tracks/structural-sufficiency/attachment_conditioned_response_algebra_pilot_v1/`

| item | value |
|---|---|
| protocol commit (frozen plan) | `c7118fb8f926` |
| structure cache sha256 | `614b761af3d5cb120edbe530cdeb4981ba548681c6946102adc6777a74a5e555` |
| raw cache sha256 | `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c` |
| remote environment | `a100-2`, A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124, CUDA 12.4, Python 3.12.14 |
| labels | Train mean 2.1622, Train std 1.2115; Valid mean 2.1863, std 1.2029 |
| baseline (predict Train mean) | Valid RMSE 1.2181, MAE 1.0063 |

## 1. What was implemented

The architecture is the one frozen in `PROTOCOL.md`; nothing in the model was
changed after the protocol commit.

* **Local objects.** For every real undirected bond `{u,v}` exactly two directed
  connection events.  For `u→v` the bond is removed from the adjacency only;
  `S` is the induced radius-1 ball around `u`, `C` the induced radius-2 ball
  around `v`; `S`/`C` may overlap on rings.  The removed bond keeps its three
  raw attribute columns plus `is_bridge` and `in_ring`.  Every bond, including
  ring bonds, produces two events.  `C` is a local ball, never the full
  complement.
* **Shared structural representer.** All connected induced subsets of 1–4 atoms
  containing the root are enumerated, canonicalised with the parent round's
  root-fixed frame LUT (all winning permutations kept), compressed to unique
  frame + multiplicity (lossless), and encoded by one shared attribute-aware
  `FrameEncoder` (64d).  Multiplicity-aware mean pooling per size `k = 1..4`
  plus `log(1+count_k)` and local scalars gives a 268d vector; a per-side MLP
  (`268→128→64`) gives `z_S`, `z_C`.  No fragment-id embedding and no message
  passing are used.
* **Shared response operator.** `s = MLP_S([z_S,e_bond])`,
  `c = MLP_C([z_C,e_bond])`, `interaction = <Q s, K c>/√32`,
  `r = scalar_S(s) + scalar_C(c) + interaction`, with a 64d site vector `h`.
* **Composition.** `ŷ_std = β + Σ_v atom(v) + Σ_{u<v} ½(r(u→v)+r(v→u)) + E_rel`,
  where `E_rel = sigmoid(gate) · Σ_{e<f} pair(e,f)/max(1,m)` uses a symmetric
  low-rank pair function with distance buckets 1/2/3/4+ and a common-cycle flag.
  The identity `ŷ = β + E_atom + E_bond + E_rel` holds exactly (tested).
* **Loss.** Whole-molecule MSE only, plus the two fixed regularisers
  (`0.001·L_site`, `0.01·L_rel`); no cut-wise or response-level label.

Three arms, same data, initialisation hashes, recipe and horizon:

| arm | interaction | E_rel | parameters | Δ vs ADD |
|---|---|---|---|---|
| `ACRA_ADD` | no | no | 141,238 | — |
| `ACRA_COND` | yes | no | 145,398 | +4,160 (+2.9 %) |
| `ACRA_REL` | yes | yes | 157,823 | +16,585 (+11.7 %) |

Parameter counts differ by construction; no capacity matching was attempted
(pre-registered).

## 2. Main result

All numbers are label units on the official Valid split (420 molecules), at the
best-by-Valid-RMSE checkpoint.  "Train RMSE" is the Train error at that same
epoch.

| model | params | best epoch | Train RMSE | Valid RMSE | Valid MAE | gap (Valid−Train) |
|---|---|---|---|---|---|---|
| `ACRA_ADD` | 141,238 | 81 | 0.596 | **0.7700** | 0.5783 | 0.174 |
| `ACRA_COND` | 145,398 | 43 | 0.553 | 0.7856 | 0.5589 | 0.233 |
| `ACRA_REL` | 157,823 | 101 | 0.311 | **0.7166** | **0.5029** | 0.405 |
| `TRC_FACTOR_CUTRESP` (read-only) | — | — | 0.222 | 0.7096 | 0.5393 | 0.488 |
| `GINE_RAW_REF` (read-only) | — | — | 0.295 | 0.6406 | 0.4674 | 0.346 |
| Morgan-2 count + Ridge α=10 | — | — | 0.446 | 0.8972 | 0.6669 | 0.452 |

Molecule-level paired bootstrap (2000 resamples of the 420 Valid molecules;
positive = candidate better):

| comparison | ΔRMSE [95 % CI] | ΔMAE |
|---|---|---|
| COND vs ADD | −0.0155 [−0.097, +0.058] | **+0.0194** |
| REL vs COND | **+0.0690 [+0.015, +0.122]** | **+0.0560** |
| REL vs ADD | +0.0535 [−0.023, +0.129] | **+0.0755** |
| ADD vs TRC | −0.0605 [−0.135, +0.015] | −0.0390 |
| REL vs TRC | −0.0070 [−0.126, +0.093] | **+0.0364** |
| REL vs GINE | −0.0760 [−0.156, −0.006] | −0.0354 |
| ADD vs Morgan-Ridge | +0.1272 [+0.055, +0.203] | +0.0886 |

Only one seed was trained.  These intervals quantify evaluation-molecule
uncertainty, not training-seed uncertainty; a CI that includes zero is reported
as "not resolved", not as "equivalent".

## 3. Answers to the five questions

### Q1. Is the simple additive composition viable?

**Yes, in the sense that matters for this pilot.** `ACRA_ADD` — 141k parameters,
no message passing, no attention, no pretraining — reaches Valid RMSE 0.7700 /
MAE 0.5783.  That is 0.45 RMSE better than the Train-mean baseline (1.2181), and
0.13 RMSE better than a Morgan-radius-2 count fingerprint with Ridge (0.8972,
CI [+0.050, +0.208]).  It is statistically indistinguishable from the parent
round's per-cut response model TRC (0.7096; CI [−0.135, +0.015]), and
significantly worse than the external GINE reference (0.6406; CI [−0.194,
−0.063]).  So simple composition recovers most of the reference performance
without any neural message passing; the remaining GNN gap is ≈0.08 RMSE and is
concentrated in the large-molecule tail (Section 7).

### Q2. Does local context conditioning (the shared bilinear interaction) help?

**On absolute whole-molecule accuracy: no clear gain.  On relative/matched-pair
accuracy: yes, consistently, but not significantly.**  `ACRA_COND` is 0.0156
RMSE *worse* than `ACRA_ADD` on Valid (CI [−0.097, +0.058]) and has a larger
Train–Valid gap (0.233 vs 0.174), yet its MAE is 0.0194 better and its
matched-pair ΔMAE is 0.4797 vs 0.5055 (difference +0.026, CI [−0.035, +0.081]).
The interaction term is unambiguously *used*: zeroing it at test time in the
trained `ACRA_COND` checkpoint raises Valid RMSE from 0.648 to 0.928 in std
units, and the term's mean absolute value is 1.22× the scalar part.  The
parsimonious reading is that, at this capacity and horizon, the interaction buys
in-sample fit and somewhat better relative predictions but not held-out absolute
accuracy.

### Q3. Does the explicit relation term help?

**Yes, incrementally over `ACRA_COND`, with a resolved whole-molecule gain; the
total gain over `ACRA_ADD` is resolved only on matched pairs.**  `ACRA_REL`
improves Valid RMSE by 0.0690 over `ACRA_COND` (CI [+0.015, +0.122]), MAE by
0.0560, the matched-pair ΔMAE difference is 0.0628 (CI [+0.013, +0.117]) and direction
accuracy rises from 0.802 to 0.839.  Removing `E_rel` from the trained checkpoint raises RMSE
from 0.591 to 0.627 std units.  Against `ACRA_ADD`, ΔRMSE is +0.0535 but the CI
includes zero ([−0.023, +0.129]); on the 434 real Train×Valid matched pairs the
same comparison is resolved (ΔMAE difference +0.0886, CI [+0.012, +0.157]).
`ACRA_REL` is a tie with TRC on RMSE (CI [−0.126, +0.093]) and better on MAE
(+0.036), still behind GINE on RMSE.  Caveats: `ACRA_REL` has the largest
generalisation gap (0.405), `E_rel` is small in amplitude (mean |E_rel| 0.179
vs |E_bond| 0.916 in std units), and its effect on the local-multiset witness is
only 0.008 std (Section 5.3).  The gain is therefore more consistent with
statistical sharing/regularisation through pair terms than with recovering
information that is absent from the local terms.

### Q4. What is the biggest error class on unseen structural combinations?

**Combination novelty: molecules whose directed events contain local
(S, C, cut-bond) combinations that never occurred in Train — plus a heavy tail
from a few large, flexible, multi-ring molecules.**  On Valid, 80.6 % of
directed events have a seen combination, 19.4 % are unseen; 79/420 molecules
have all events seen and none has zero seen events.  For `ACRA_REL`, mean |error|
is 0.596 on the 35 molecules with >50 % unseen events vs 0.322 on low-novelty
molecules (Pearson novelty–|error| 0.220; for `ACRA_ADD` 0.103).

The error decomposition by novelty bucket shows where the relation term does and
does not transfer:

| unseen-event fraction | n | ADD RMSE | COND RMSE | REL RMSE |
|---|---|---|---|---|
| 0 | 79 | 0.647 | 0.617 | **0.511** |
| (0, 0.2] | 171 | 0.719 | 0.609 | **0.545** |
| (0.2, 0.5] | 135 | **0.891** | 0.950 | 0.921 |
| (0.5, 0.8] | 31 | **0.795** | 1.185 | 0.897 |
| (0.8, 1] | 4 | **0.435** | 0.232 | 1.090 |

REL's advantage is concentrated in the two low-novelty buckets and disappears
(or reverses) above 0.2.  The per-event response does **not** shrink on unseen
combinations: mean |r| is 0.098 on unseen vs 0.091 on the most frequent events
(REL), i.e. the operator is confidently wrong on novel local contexts.  The
single worst molecule (`row_id 1425`, 36 atoms, 40 bonds, cycle rank 5, one
phosphate/cyclopropane/thiophene-rich structure) has |error| 6.31 for REL (ADD
4.97, COND 6.68) and alone inflates the RMSE tail; all 420 Valid molecules
contain both ring and bridge bonds, so those two strata cannot be contrasted on
Valid (bridge combo seen rate 0.785 vs non-bridge 0.816 is the only available
support difference).

### Q5. What is the most promising next direction?

**Keep the additive/relational algebra and attack overfitting and novelty
calibration, not expressiveness.**  The only resolved whole-molecule gain came
from adding relations to the conditioned operator (Q3), and the relation ablation
is the only ablation that improves a trained arm.  But the model already fits
Train far better than Valid (gap 0.405 in the best arm; Train RMSE 0.311), and
its errors concentrate exactly where local support is thin.  The three
highest-value next steps, in order:

1. **Response shrinkage/calibration on unsupported local combinations** — make
   `|r|` depend on Train support of the event's `(S, C, cut-bond)` identity
   instead of being roughly support-independent (measured above).  This is a
   small, testable change to the response operator, not a new architecture.
2. **Capacity/regularisation control** for the frame encoder and the relation
   term (the 12k extra relation parameters produce a 0.405 gap and a tiny
   witness effect; per-bucket scalar terms or distance-normalised pair terms are
   cheaper alternatives to the generic bilinear sum).
3. **Matched-pair / relative supervision as an evaluation-first target** — the
   pair results (Section 6) are the most consistent signal in this round, and
   they are obtained without any pair-level training.

No GNN replacement or additional message-passing module is proposed; the round
stops here.

## 4. Mechanism diagnostics

Term scales and cancellation on Valid (std label units; `cancellation_index =
1 − mean|ŷ−β| / mean(|E_atom|+|E_bond|+|E_rel|)`):

| arm | mean abs E_atom | mean abs E_bond | mean abs E_rel | cancellation | signed mass cancellation of r | positive r fraction |
|---|---|---|---|---|---|---|
| ADD | 0.514 | 0.947 | 0 | 0.380 | 0.811 | 0.815 |
| COND | 0.556 | 0.955 | 0 | 0.432 | 0.732 | 0.814 |
| REL | 0.526 | 0.916 | 0.179 | 0.476 | 0.752 | 0.747 |

`E_bond` is the dominant term in every arm; `E_atom` is roughly half its size.
The composed prediction is substantially smaller than the sum of absolute terms
(cancellation 0.38–0.48), and a large fraction of the directed responses are
negative (18–25 %), so positive and negative local responses genuinely cancel.
Adding the relation term lowers the positive-response fraction and increases
cancellation, i.e. it does not simply add a positive offset.

Ablations of the trained checkpoints (same weights, one term zeroed at test
time, Valid RMSE in std units):

| arm | base | without interaction | without E_rel |
|---|---|---|---|
| COND | 0.648 | **0.928** | — |
| REL | 0.591 | **0.874** | **0.627** |

Both auxiliary terms are load-bearing for their own checkpoints; the large
interaction ablation effect is expected because the interaction was trained in,
but it confirms the term is not a dead branch.  The relation gate settles at
`sigmoid(gate)=0.231` (initialised at 0.12), and the site vector `h` is used
only by the relation term.

Gradient norms after training (Train batch, top-level modules) are finite and
non-zero for all branches in all arms; the per-branch check is in
`analysis/contribution_diagnostics.json`.  No branch is untrained or dead.

## 5. Relational capacity

### 5.1 Real molecules (Train + Valid, 3780 molecules)

Grouping molecules by the exact multiset of their per-bond local signatures
`(S identity, C identity, cut-bond attributes)` plus atom count finds **one**
group of size 2; those two molecules also have identical relation profiles, so
**no real molecule pair was found with an identical local multiset and a
different bond-pair relation profile**.  Within that group, the fitted model's
E_atom range is 0.0 and the E_bond/E_rel ranges are 2.4e-7 / 5.2e-8 (float
noise).  Structural support at this radius is therefore almost injective on this
dataset: real molecules with identical local inputs are rare, and when they
occur the global relations agree too.  This limits what a relation term can add
on Lipophilicity: there is little "local ambiguity" to resolve.

### 5.2 Exhaustive small graphs

All 992 connected graphs on 4–7 vertices (networkx atlas) with one common real
atom/bond attribute row: **no two graphs share the local-multiset multiset**
(the local radius-1/2 balls are already a strong invariant at this scale), so no
relation-only witness exists there.  This is a negative result about the probe,
not about the model.

### 5.3 Constructed witness (the relation term is relation-sensitive)

Because no small witness exists, one was constructed: in a cubic graph of girth
≥ 6, after removing a cut bond every radius-1/2 ball is a tree of the same
shape, so **every edge has the same local signature**.  The Pappus graph (18
vertices, 27 bonds, girth 6) and a deterministic girth-preserving 2-switch
variant are non-isomorphic but have an identical local multiset (one signature,
multiplicity 27 each).  Running the fitted `ACRA_REL` checkpoint on both:

| quantity | Pappus | 2-switch variant | Δ |
|---|---|---|---|
| E_atom (std) | 1.27281999 | 1.27281999 | **0.0** |
| E_bond (std) | 2.32980847 | 2.32980847 | **0.0** |
| E_rel (std) | −1.21441984 | −1.20633852 | **+0.0080813** |
| ŷ (std) | 1.96016717 | 1.96824849 | +0.0080813 |
| relation profile (bucket 3 / bucket 4) | 162 / 27 | 156 / 33 | different |

The local terms are bit-identical and the entire prediction difference is the
relation term (≈0.0098 label units).  This demonstrates that `E_rel` can in
principle distinguish graphs that are indistinguishable to every local term,
but the effect is small because the gate is 0.231 and the pair function is a
near-linear low-rank sum.  It is a capability demonstration, not a performance
claim.

## 6. Real matched pairs

The parent round's real replacement directory (same exact context, two observed
molecules) gives 4,646 Train×Train, 434 Train×Valid and 15 Valid×Valid pairs;
675 Test-touching pairs were dropped before any label was read.  Only
`predicted_delta = ŷ(G_b) − ŷ(G_a)` of the full model is evaluated.

| scope | arm | n | ΔMAE | 95 % CI | direction accuracy | Pearson(pred Δ, true Δ) |
|---|---|---|---|---|---|---|
| Train×Train | ADD | 4646 | 0.433 | [0.401, 0.466] | 0.774 | 0.794 |
| Train×Train | COND | 4646 | 0.404 | [0.368, 0.441] | 0.799 | 0.816 |
| Train×Train | REL | 4646 | **0.289** | [0.268, 0.312] | **0.856** | **0.912** |
| Train×Valid | ADD | 434 | 0.506 | [0.435, 0.593] | 0.783 | 0.825 |
| Train×Valid | COND | 434 | 0.480 | [0.394, 0.584] | 0.802 | 0.816 |
| Train×Valid | REL | 434 | **0.417** | [0.332, 0.519] | **0.839** | **0.862** |
| Valid×Valid | ADD | 15 | 0.595 | [0.369, 0.855] | 0.667 | 0.743 |
| Valid×Valid | COND | 15 | 0.553 | [0.311, 0.814] | 0.667 | 0.737 |
| Valid×Valid | REL | 15 | **0.492** | [0.317, 0.665] | **0.733** | **0.855** |

Paired cluster bootstrap of the *difference* in ΔMAE on the same 434
Train×Valid pairs (cluster = the Valid molecule of the crossing pair):

| comparison | ΔMAE difference | 95 % CI |
|---|---|---|
| COND vs ADD | +0.0258 | [−0.0350, +0.0807] |
| REL vs COND | **+0.0628** | [+0.0132, +0.1171] |
| REL vs ADD | **+0.0886** | [+0.0115, +0.1571] |

Representative pair (fixed rule: largest Train×Valid transform group with ≥3
pairs, then largest/median/smallest |Δy|; the same pair is shown for all arms):
`row 3918` (label 1.86) → `row 1661` (label 4.26), true Δ = +2.40.

| arm | predicted Δ | ΔE_atom | ΔE_bond | ΔE_rel |
|---|---|---|---|---|
| ADD | 0.895 | 0.519 | 0.376 | 0 |
| COND | 1.119 | 0.489 | 0.630 | 0 |
| REL | 1.251 | 0.503 | 0.601 | 0.147 |

All arms under-predict the true +2.40, and the relation term closes ≈0.15 of the
gap for this pair.  These are exact model-algebra decompositions, not chemical
causal contributions.

## 7. Failure cases and error structure

Deterministic top-3 lists are in `analysis/cases.json` (no cherry-picking):

| comparison | candidate better | candidate worse | mean squared-error reduction [95 % CI] |
|---|---|---|---|
| COND vs ADD | 219 | 201 | −0.0241 [−0.1705, +0.0844] |
| REL vs COND | 231 | 189 | **+0.1036 [+0.0216, +0.1905]** |
| REL vs ADD | 235 | 185 | +0.0795 |

The largest single case is `row 1425` (36 atoms, 40 bonds, cycle rank 5, 62.5 %
ring bonds): label −0.04, predictions ADD 4.93 / COND 6.64 / REL 6.27, with
squared errors 24.7 / 44.7 / 39.8 (70–80× the Valid mean squared error).  It is
a genuine outlier, not a representative error class: for REL vs COND it accounts
for ≈11 % of the total squared-error reduction, and for REL vs ADD it goes the
other way.  Size quartiles show REL is best in q1–q3 (q1 0.632, q2 0.543, q3
0.675, q4 0.969 RMSE; ADD 0.770/0.669/0.758/0.880), while `ACRA_ADD` is best in
the q4 stratum (33–100 atoms, RMSE 0.880 vs REL 0.969) — the large-molecule tail
is where the additional operators hurt.

## 8. What is not established

* **Single seed.**  No training-seed variance; all CIs are molecule-level
  bootstraps.  Differences of ≈0.01–0.05 RMSE should not be treated as stable
  model rankings.
* **Capacity is not matched.**  COND adds 4.2k parameters and REL 16.6k; part of
  the REL gain could be capacity rather than relations.
* **Only whole-molecule labels.**  No per-event or per-cut supervision, so
  "response quality" is inferred from composed predictions only.
* **Local-multiset test is mostly negative on real data** (one group of two
  molecules), and the positive relational witness is an 18-vertex synthetic
  graph with identical atoms; it demonstrates capability, not effect size.
* **Matched pairs are real replacements, not counterfactuals.**  The parent
  directory guarantees the same exact context, but the pair difference is still
  an observed replacement, and Δŷ is model algebra.
* **No Test metric, by design.**  Nothing in this report uses Test labels.

## 9. Deviations from the frozen protocol (all explained)

1. `analysis/relational_capacity.json` searches **all connected graphs on 4–7
   vertices** (protocol said 6–7) — a strict superset; no witness found.
2. Because the exhaustive search produced no witness, a **constructed
   18-vertex witness** (Pappus vs deterministic 2-switch) was added.  Its
   conditions (identical local multiset, different relation profile) are
   verified mechanically before the model is run.
3. An additional label-free analysis, `analysis/error_structure.json`, was added
   to answer the "biggest error class" question with molecule-level novelty
   buckets and size quartiles; it introduces no new model or tuning.
4. Harness facts: the structure cache (8.2 MB) is committed because the GPU host
   has no RDKit; the remote host had untracked parent-round result files, which
   were moved to `.rr-untracked-backup-20261122/` (verified byte-identical to the
   committed copies) before deploying.
5. Horizon: a timing smoke on the real remote GPU (2.5–3.0 s/epoch including
   full Train+Valid evaluation) fixed 160 epochs for all arms before the formal
   runs (`analysis/horizon.json`); actual total 0.328 GPU-hours vs the 0.36
   estimate.
6. JSON formatting: undefined non-finite diagnostics (Pearson/eta² of
   zero-variance terms, disabled relation gate) are stored as `null` in the
   analysis deliverables; the raw `results/ACRA_*/manifest.json` files keep the
   training script's `NaN` for the disabled gate.

## 10. Decision-rule outcome

Pre-registered rule: if only the additive arm works, do not add complex
operators; if the relation term helps, study structural composition; if the
interaction helps both whole-molecule and cross-molecule deltas, study response
regularities.

Observed: the additive arm is viable; the interaction is used and helps relative
predictions but not absolute Valid RMSE; the relation term gives the only
resolved whole-molecule improvement (over COND) and the largest matched-pair
improvement (over both).  The next round should therefore study **structural
composition under support-aware regularisation**, as in Q5 — not a new
architecture, and not a GNN replacement.  The round stops here.