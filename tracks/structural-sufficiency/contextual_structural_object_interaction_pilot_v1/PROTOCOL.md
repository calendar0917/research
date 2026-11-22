# CSOI-v1 — Contextual Structural Object Interaction (pilot)

Protocol id: `contextual-structural-object-interaction-pilot-v1`
Round dir: `tracks/structural-sufficiency/contextual_structural_object_interaction_pilot_v1/`
Parent round (read-only, reused code + two reused arms):
`tracks/structural-sufficiency/task_response_composition_s0_v1/`
Dataset: `ogbg-mollipo` (official scaffold split). Seed: `0`. Frozen before training.

This document is frozen **before** any formal training run. Everything below is a
commitment: model, arms, recipe, metrics, decision rules, budget. Deviations are
only allowed for the reasons listed in §9 (pre-declared exceptions) and must be
recorded in `REPORT.md`.

---

## 1. Question

The parent round compressed every rooted structural object (1–4 atoms, root-fixed
frame) into a **single 64-d fragment embedding before it ever sees its
counterpart**, then pooled the fragment embeddings. That early compression is
lossy in a specific way: an object's meaning can depend on the *other* side of
the cut (`C=O` next to an amine vs next to an ether). The pilot asks a single
falsifiable question:

> **Does keeping the fine-grained object token set and doing
> context-conditioned interaction *before* pooling beat early compression to one
> embedding per fragment, at equal data/recipe/parameter budget?**

Secondary question (only answered if the primary shows positive signal):
does an explicit chemical relation bias (bucket-pair + cut-bond) add anything on
top of content-only attention?

Non-goals: no new architectures, no extra arms, no hyperparameter search, no
literature review, no change to the parent's cut enumeration or molecule-level
aggregation, and **never reading official Test labels**.

## 2. Data view (identical to the parent)

- Cut enumeration, parse table, bond/atom attribute tables, scaffold split,
  train-only target scaler, train-only pair directory: **all reused verbatim**
  from the parent modules (`trc_data`, `trc_pairs`, `trc_prepare`) via
  `sys.path`; nothing in the parent round is modified.
- Split sizes: train 3360 / valid 420 / test 420 molecules. Test rows are kept
  only as an index; the training view (`RunData`) exposes `train` + `valid`
  **only** and asserts no test split exists. `manifest.json` records
  `test_policy: "no test labels read"`.
- Structural cache: `data/csoi_cache_lipo.npz`, rebuilt by the parent's
  `build_dataset` with `dataset_key="lipo"` (the parent loader silently falls
  back to the ESOL cache for any dataset string other than `lipo`; a raw-size
  guard now makes that impossible).
  Identity is enforced by **content digest** `46439b91dfd760255be4c53256f12687959d203b6740e55c0213d66476168825`
  (= the parent's cache content, verified equal), because npz zip timestamps
  make the file hash non-reproducible. Fragments: 29283, parses: 33438,
  frames: 71004.

## 3. Model (CSOI)

Shared across all CSOI arms — identical modules, identical initialisation,
identical data order, identical recipe:

1. **`FrameEncoder` φ**: full raw attributes (atom: 9 official columns, bond: 3
   official columns), masks, size bucket; **no template ids, no fragment ids**.
   Root-fixed atom order from the parent frame builder. Output 64-d per object.
   Shared by the S side and the C side.
2. **Token set** `X_F`: one token per rooted object in the fragment, ordered by
   (distance bucket, order). Token = `token_proj(φ(frame)) + bucket_emb[b] +
   mult_proj([w, log1p(w)])`, then LayerNorm. `w` = rooted-object multiplicity.
   Tokens are **not** pooled before interaction.
3. **Interaction** (1 layer, 4 heads × 16 dim, cross-attention S→C):
   `score_ij = (q_i·k_j)/sqrt(16) + log(m_j) [+ relation_bias(i,j)]`,
   `m_i = Σ_j α_ij v_j`, with a **null-context token** used when `C = ∅`.
   `relation_bias = rel_mlp([rel_ds(b_i) + rel_dc(b_j) + rel_dt(b_i+b_j) +
   rel_cut(cut_bond_emb)])` (CSOI_REL only).
4. **Cut response** `r_i = r_mlp([x_i, m_i, x_i*m_i, x_i-m_i, cut_emb])`,
   then multiplicity-weighted pooling over tokens:
   `f(S|C) = β + add_s(pool_S) + add_c(pool_C) + pool_w(r)`.
   `add_s`/`add_c` are the additive (no-interaction) branch.
5. **Graph level**: parent's cut enumeration and molecule-level **mean over
   parses**, with the parent's fallback for molecules without cuts (exactly the
   parent's aggregation, unchanged).

Arms (fixed, seed 0, no search):

| arm | description | status |
| --- | --- | --- |
| A `TRC_FACTOR_CUTRESP` | parent's factored additive model + CUTRESP aux | **reused** (`task_response_composition_s0_v1/results/LIPO_FACTOR_CUTRESP_R0`) |
| B `CSOI_ADD` | CSOI encoder/additive branch, **no** interaction, no attention params | trained |
| C `CSOI_GLOBAL` | + content-only cross-attention (no relation bias) | trained |
| D `CSOI_REL` | + relation bias (bucket pair, total distance, cut bond) | trained |
| E `GINE_RAW_REF` | external prediction coordinate (raw GINE, no ports/cuts) | **reused** (`.../GINE_RAW_LIPO_R0`) |

B/C/D share encoder, additive head, init, data order, and recipe; the *only*
differences are the attention modules (C, D) and the relation branch (D). Arm
contrasts are therefore pre-registered: B→C = value of interaction at all,
C→D = value of the explicit relation bias.

Parameter ledger (measured, `analysis/checks.json → params`):
B 69,080 / C 130,969 / D 132,973 (D−C = 2,004 = relation branch only).

## 4. Recipe (frozen, uniform across arms)

Parent recipe verbatim: AdamW, peak lr 1e-3, weight decay 1e-4, grad clip 5.0,
5 warmup epochs, cosine decay to 1e-5, batch 64 molecules + 16 aux pairs,
CUTRESP aux weight 0.25, train-only scaler and train-only pairs, best-by-valid
checkpointing, uniform 160-epoch cap. The horizon is **fixed once** from the
timing smoke (`analysis/timing.json`) and applied to all three trained arms;
`--epochs 0` reads it from that file so the horizon cannot drift per run.

## 5. Attention cost and the token guard

Measured on the full cache (`analysis/token_stats.json`): tokens/fragment
p50 358, p90 520, p95 567, **p99 712**, max 1399. p99 ≫ 128, so a 128-token cap
would be massively lossy (a deterministic stratified 128-token sample retains
< 95 % of the multiplicity mass on **91.1 %** of fragments, i.e. far above the
5 % blocker threshold).

Decision (recorded, not improvised): **no token cap is applied**; the full token
set is processed with **chunked ragged attention** (`attention_entry_chunk =
262144` entries) so no structural object is dropped. The guard statistics are
reported in `interaction_diagnostics.json` and the checks assert
(i) the guard is deterministic, (ii) it would be inadequate, (iii) training uses
the full token set. Full pass ≈ 190.4 M attention entries (train 151.1 M).

## 6. Metrics (Valid only; Test never read)

1. **Whole-molecule Valid RMSE / MAE** per arm, with a **paired bootstrap
   (2000 resamples over Valid molecules)** for every pre-registered contrast.
   Molecule-level error is the parent's mean-over-parses prediction.
   Train-seed uncertainty is *not* covered (single seed, as mandated) — the
   report says so explicitly.
2. **Pair delta metrics** on the train-only pair directory (Valid molecules
   only, pair cells restricted to Valid): delta MAE, delta RMSE,
   Pearson(delta_pred, delta_true), sign/direction accuracy, and
   component-equal MAE, each **by scope** (transform / component) and by
   support stratum.
3. **Support strata** (Train-only support index, `analysis/support.json`):
   seen/unseen S, seen/unseen C, 4-way combination, plus object-level coverage.
4. **Interaction diagnostics** (`interaction_diagnostics.json`): attention
   entropy, effective token count, relation-bias statistics, gradient norms per
   module group, the interaction contribution `f(S|C) - f_additive(S|C)`, and a
   **remove-bias-under-frozen-weights** probe (C vs D with D's relation branch
   zeroed, no retraining).
5. **Synthetic context-flip sanity task** (`analysis/synthetic_flip.json`): a
   constructed task where the correct response is defined by the *interaction*
   (same S, different C, different target) vs an additive-only rule. Reports
   whether the interaction arm can fit the non-additive rule at all (a
   capability check, not evidence about Lipo).
6. **Representation-compression diagnostics on the parent's TRC
   representations**: how much of the parent's fragment-embedding variance is
   explained by linear probes of additive vs interaction statistics (reported as
   a descriptive diagnostic, no causal claim).

## 7. Decision rules (pre-registered)

Let `σ_train = 1.2115` (train-label std, parent manifest).
`delta_task = 0.025 · σ_train = 0.0303` RMSE (the smallest difference this pilot
is designed to resolve); weak-signal band `0.01 · σ_train = 0.0121`.

Primary contrast **B→C** (`CSOI_GLOBAL` vs `CSOI_ADD`), Valid whole-molecule RMSE:

- **strong positive**: bootstrap 95 % CI of the paired RMSE difference excludes 0
  **and** point estimate ≥ `delta_task`.
- **weak positive**: CI excludes 0 and `0.01σ ≤ point < 0.025σ`.
- **no evidence**: CI includes 0, or point estimate < `0.01σ`.
- **negative**: CI excludes 0 in the wrong direction (interaction hurts).

Secondary contrast **C→D** (`CSOI_REL` vs `CSOI_GLOBAL`) uses the same rules and
is only interpreted if B→C is at least weak-positive.

A→B (reused parent arm vs `CSOI_ADD`) is a **calibration** contrast: it tells us
whether the CSOI encoder/additive re-implementation is on the same scale as the
parent, and is reported but **not** used to decide anything.

Conditional extension: ESOL (`CSOI_ADD`, `CSOI_REL`, `GINE_RAW_REF`) is run
**only if** Lipo shows a positive signal (strong or weak) for B→C. No other
dataset is touched.

## 8. Engineering gate (must pass before training)

`code/csoi_checks.py` writes `analysis/checks.json`; all 13 groups must be `ok`:

`inputs`, `renumber`, `tokens`, `branches`, `attention`, `batching`, `batch`,
`boundary`, `init`, `params`, `loss`, `guard`, `determinism`.

Notable facts they pin down: atom-renumbering invariance; parse/atom-set
fidelity; independent cache rebuild by fragment key; `CSOI_ADD` double-difference
≡ 0 and `CSOI_GLOBAL` ≡ `CSOI_REL` with the bias zeroed; fast chunked attention
≡ naive per-parse attention (≤ 1e-16) and chunk-size invariance; batch ≡ single
(≤ 1e-16); one real batch with finite loss and non-zero gradients in every module
group; split/pair/scaler train-only; identical init of shared modules across
arms; the parameter ledger; the cut-loss algebra; guard determinism; and
seed-controlled determinism.

*Bug found by this gate and fixed before training:* the attention softmax was
normalised **per parse** instead of **per query token**, making `f(S|C)` depend on
how many S objects the fragment happened to have. Caught by the naive-reference
check (0.105 ≠ 0), fixed, and re-verified (`vs_naive_reference_max_abs = 8.3e-17`).

## 9. Budget and pre-declared exceptions

- ≤ 5 A100 GPU-hours, ≤ 2 GPUs in parallel, ≤ ~6 h wall for the whole sprint.
- All GPU work goes through `rr deploy` + `rr run` on host `res`
  (pools `res-gpu0` / `res-gpu1`). No manual SSH, no `sbatch`, no `nohup`.
- Data prep, statistics, synthetic task, checks, and all analysis run **locally
  on CPU**.
- Pre-declared exceptions (recorded if used): (a) horizon reduced once from the
  timing smoke if 3 arms × 160 epochs would exceed 80 % of the GPU budget —
  the reduced horizon is applied uniformly; (b) a run may be restarted from
  scratch at most once if it dies for infrastructure reasons (same commit, same
  recipe).

## 10. Deliverables

`PROTOCOL.md` (this file, frozen), `REPORT.md` (decision-first),
`analysis/{checks,token_stats,structure_stats,timing,support,synthetic_flip}.json`,
`results/main_table.csv`, `results/summary.json`, `results/pair_metrics.json`,
`results/interaction_diagnostics.json`, `results/cases.json`,
`results/run_manifest.json`, plus per-run `manifest.json` / `curves.csv` /
`predictions_*.npz` / `train.log`.
