# PROTOCOL — lcrc-explicit-semantic-learning-8h-s0-v1

Frozen before the first formal run; changes are a new round. The machine-checked
core lives in `code/se_common.py` (the runner refuses to run on drift) and in
`configs/*.yaml` / `configs/run_matrix.csv`. This file states the rules; the
authoritative field values are in `protocols/lcrc-explicit-semantic-learning-8h-s0-v1.yaml`.

## 0. Test policy of this round

The round is explicitly authorised to evaluate the official Test split during
development (`test_policy: always`). Test is reported and may inform this
round's route judgement, but **selection is Validation-only**. Every Test read
is recorded in `analysis/test_access_log.json`.

## 1. Method change (the only change)

The method body stays LCRC: same explicit local objects, atom/bond/path
statistics, LocalMLP and readout. **No** message passing, attention, graph head,
D-MPNN backbone/teacher/student/distillation, and **no new loss**.

The six free embedding tables of the C1 local encoder are generated from two
shared composition tables `A:Embedding(16,16)`, `B:Embedding(5,8)` (296 params)
through small per-bank MLPs:

- `u(p) = [A(mid), B(b1), B(b2)]` (32-d), `v(e) = [A(atom), B(bond)]` (24-d)
- path banks `path_emb, path_u`: MLP(32→197→32)
- binary banks `nbr_emb, end_emb, path_v`: MLP(24→40→32)
- `path_gate`: `1 + MLP(32→23→1)` with the final Linear zeroed (gate ≡ 1 at init; no sigmoid)

Hidden activation SiLU, no output activation. No inter-bank shared output layer;
no additive low-order replacement of the MLPs.

### Codec (exact, verified by selfcheck)

- `N_BOND_TYPES=5`, `N_ATOM_TYPES=16`, `rank r=32`
- path code `p`: `mid=p//25`, `b1=(p%25)//5`, `b2=p%5`; 400 codes
- binary code `e`: `atom=e//5`, `bond=e%5`; 80 codes
- `path_end63` is a historical field name; the live schema has **80** end classes
  (ZINC 189/63 vocabularies are never used).

### Parameter accounting (exact)

- shared composition tables: 296
- two path MLPs: 25,674; three binary MLPs: 6,936; gate: 783 → generator **33,689**
- free-table params replaced: 33,680
- FS total = **228,683** (free-lookup C1 = 228,674; +9)

### Initialisation and scale

- Non-table state preserved from the C1 seed-0 state; generator initialised with an
  independent CPU `Generator(seed=17)`, not advancing the public training RNG.
- Shared composition embeddings `N(0,1)`; MLP Linear `U(±1/sqrt(in))`; gate final zeroed.
- For the five vector banks, at `θ0` compute the raw generator output over the whole
  fixed code grid and fix per-column affine buffers mapping `μ/std` to the original
  C1 initial table's per-column `μ/std` (`std` floor 1e-4). Constants come from
  initial weights + public schema, **never** refit on samples/labels.
- Gate is not standardised (always `1 + gate_net`).
- Tables regenerated every forward (cacheable within one forward, never across an
  optimiser update).

## 2. Data, recipe, metric

- **Lipo**: official `ogbg-mollipo` scaffold split, Train 3360 / Valid 420 / Test 420
  (verbatim from OGB `get_idx_split()`; `random122` is never used). Regression on
  raw logD; Train-only target normalisation; all results restored to raw logD.
- **ESOL**: official `ogbg-molesol` scaffold, 902/113/113, same label-free
  `lipo_v1` schema/codes; Train-fitted target norm; no ESOL tuning.
- **Recipe R (R3)**: AdamW, lr 3e-4 → 1e-5 per-step cosine, wd 1e-3, batch 128,
  dropout 0.10, clip 5.0, 360 epochs, FP32/TF32 off, row order seed 0 + shuffle_offset 917000.
- Recipes `W0` (wd 0) and `Rwd` (wd 1e-2) change only weight decay.
- Every epoch evaluates the **full** official Train/Valid/Test on the same weights.
- **Selection = Validation-only**. Primary point: 50-point common update grid
  `round(j·U/50)`, `U = 360·ceil(N_full/128) = 9720`. Auxiliary: best-Val epoch and
  last-5% window mean. Test is reported, never used to select.
- Determinism: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `deterministic_algorithms(True)`,
  TF32/benchmark off.

## 3. Four paired arms (initialisation matched)

| arm | free tables? | init | decode |
|---|---|---|---|
| LS | yes, then free | = FS θ0 materialisation | original code |
| FS | no (generator) | generator θ0 | true mid/atom/bond composition |
| LP | yes, then free | = FP θ0 materialisation | original code |
| FP | no (generator) | same generator/θ0 as FS | fixed composite-code bijection before decode |

- Bijection: `np.random.default_rng(20261114)` → `π400`, `σ80`. It is a composite-code
  bijection (not a per-atom/per-bond rename); it preserves class-identity information
  and destroys the real composition correspondence.
- Within each pair, epoch-0 predictions and all non-table weights are identical
  (verified CPU atol 1e-6 / GPU max|Δ|≤4.29e-06); gate init = 1.

### Quantities (RMSE; positive ⇒ composition sharing better)

- `Delta_sem = RMSE(LS) − RMSE(FS)`
- `Delta_perm = RMSE(LP) − RMSE(FP)`
- `DiD_semantics = Delta_sem − Delta_perm`

The four models are recomputed jointly on a fixed-weight common scaffold
perturbation; DiD intervals are never stitched from independent intervals.

Logical note: any trained F can materialise the six tables to a plain lookup and
reproduce its predictions, so F never gains more input information than the free
lookup. A positive FS-over-matched-LS must be explained by parameter sharing /
effective constraint / optimisation bias — not by "extra structure" or "more
capacity". If FP gains equally, real chemistry cannot be claimed as the source.

## 4. Budgets

- Wall: target 7.5h, hard 8h (includes implementation/deploy/train/analysis/deliver).
- GPU: soft 14, hard 16 A100·h (includes eval, smoke, retries, diagnostics).
- ≤ 2 GPU jobs, one per GPU; pools are logical `cuda:0`; no DDP.
- ≤ 23 formal research runs: P1 ≤ 11, P2 = 5, P3 ≤ 4, P4 = 3; untriggered branches are
  not run.

## 5. Budget runs (P3)

- Nested subsets of official Train by `sha256('LCRC-semantic-budget-v1|' + global_row_id)`
  ascending; fractions N100/N50/N25 = 1.0/0.5/0.25; Val/Test unchanged.
- Fixed update budget `U = 9720`, same per-step cosine and batch 128.
- Fit-only target normalisation uses only Fit labels; structure normalisation may use
  all official Train structures (label-free) and is disclosed.
- Each completed Fit epoch still evaluates the full official Train/Val/Test.
- Comparison uses the common R50 grid; best-per-epoch is auxiliary only.

## 6. Uncertainty

- 500 Murcko-scaffold-cluster bootstrap, `Exp(1)` weights shared by all models in
  a comparison (paired), seed 20261114; acyclic fallback = canonical SMILES.
- Intervals describe sample perturbation of fixed trained states, **not** training
  variance; no adding seeds or changing thresholds because an interval crosses 0.

## 7. Judgement labels (can co-occur, not mutually exclusive)

`INITIALIZATION_SIGNAL_ONLY`, `GENERIC_REPARAMETERIZATION`,
`SEMANTIC_SHARING_SUPPORTED_THIS_PANEL`, `REGULARIZATION_DEPENDENT`,
`SMALL_LABEL_REGIME_CANDIDATE`, `MECHANISM_SIGNAL_NO_PERFORMANCE_ADVANCE`,
`PERFORMANCE_ADVANCE_CANDIDATE`, `NO_USEFUL_SIGNAL`,
`INCOMPLETE_RESOURCE_OR_IMPLEMENTATION`.

### Evidence switch E (for a P4 amplification purchase)

1. In one full-Train recipe, FS beats matched LS by ≥2% Val **and** Test RMSE with MAE
   not worse >1%;
2. DiD_semantics > 0 on Val **and** Test and survives deleting the control's worst 1%
   on Test;
3. the wd0 pair or ESOL gives a same-direction (Val/Test) FS advantage;
4. FS also beats the best reachable lookup candidate (B_R / B_Rwd / LS, each Val-selected)
   by ≥1% on Val **and** Test.

E is a budget rule; a small-data interval crossing 0 is recorded as uncertain and does
not auto-add seeds.

## 8. Amplification (P4) — only if E holds and it fits the schedule

Prefer PCBA three-arm `B_native/LS/FS _W2` (r64 interface, official scaffold,
350343/43793/43793, 128 tasks, 40ep) only if data/interface ready, the true
profile ×1.25 for the slowest two in sequence + 30 min report margin fits before
T0+8h, and total GPU ≤ 16h. If PCBA cannot finish, fall back to official
`ogbg-molhiv` three-arm `B_native/LS/FS` (160ep, ROC-AUC focus). If E does not
hold, P4 is **not purchased**.

## 9. Deliverables

`PROTOCOL.md`, resolved config manifest, stage/queue log, `SOURCE_NOTES.md`,
code/tests, per-run identity/budget, curves/main table/paired/DiD, label budget,
support/gradient kernel, MMP/edit response, exported pure-lookup model, Chinese
`REPORT.md`, ≥5 informative figures. Large checkpoints/predictions stay in this
round's `analysis/results/`; Git tracks only durable small records; prior rounds
are never overwritten.
