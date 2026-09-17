# FSAR-R2-AR0 — fixed radius-2 explicit pure-topology coordinates + assignment residual (durable verdict)

Branch `exp/fsar-r2-ar0-zinc`.
Pre-registration: `notes/fsar_r2_ar0_preregistration.md` (20 sections).
Training revision `fc1312f`; eval-only revisions `8ad2358` (3-seed `decide()`
and the `analyze` paired-bootstrap stage).
**Official ZINC test never loaded** (`official_test_loaded = false` in every
JSON; the preprocessor reads only `train` / `val`).

This round is not a continuation of FSAR-v1/v2 or of the route recheck.  It is
not a SOTA attempt: the three models are deliberately tiny
(22,977 / 24,797 parameters) and use only a fixed radius-2 pure-topology
coordinate plus a whole-graph attribute marginal.  The absolute MAE levels
(~0.46–0.56) are far from the repo's full-backbone numbers and are **not**
benchmark claims.

---

## 0. The one question, and the answer

> Under a fixed radius-2 explicit pure-topology coordinate, a strong
> structure/attribute marginal baseline and one identical training protocol,
> does the real node structural-role ↔ node attribute assignment provide a
> stable predictive increment over the marginals **and** over a matched
> marginal-capacity control?

**Answer: YES — assignment is supported.**

`MB` (assignment residual `C`) beats both `M0` (strong marginal baseline) and
`MM` (matched marginal-capacity control on `P`) on **all three paired seeds**
`0/1/2`; every paired per-molecule bootstrap 95 % CI excludes zero with
`P(delta > 0) = 1.0`; and the trained `MB` is demonstrably sensitive to the
attribute assignment while `A` and `S` are exactly invariant under it.

The claim is exactly scoped to the design: a node-level **linear** assignment
residual on this fixed 65-D radius-2 explicit basis, at this training budget.

---

## 1. Information design

A molecule is `G = (T, X, E_attr)`.

| channel | content | dimension |
|---|---|---|
| `S` (`phi_v`) | pure-topology rooted radius-2 explicit coordinate | 65 |
| `A` (`a(G)`) | whole-graph atom/bond attribute multiset | 64 |
| `B` (`C`) | centred role↔attribute assignment residual | 65 × 28 |

`phi_v` = `[root_basis(11), mean_node(11), std_node(11), mean_edge(15),
std_edge(15), log1p(|V_{B2(v)}|), log1p(|E_{B2(v)}|)]`, using the frozen
`fsar_v2` explicit operator basis (root indicator, restrained shell
indicators, induced degree, shell-resolved neighbour counts, rooted walks
`A,A^2,A^3`; edge endpoint shell pairs, degree statistics, common neighbours,
shell-neighbour statistics).  Undirected induced edges are counted once.  No
chemistry enters `phi_v`; no C4/C5, RRWP or homomorphism primitive is added and
the radius is not swept.

`a(G)` = `[log1p(atom_counts)(28), atom_freq(28), log1p(bond_counts)(4),
bond_freq(4)]` — the whole-graph multiset only, never a structural position.

`q_v = onehot(x_v) ∈ R^28`, and with the sum-centred statistic

```
J = Σ_v φ_v q_vᵀ
P = (1/n)(Σ_v φ_v)(Σ_v q_v)ᵀ
C = J − P = Σ_v (φ_v − φ̄)(q_v − q̄)ᵀ        (no division by n−1)
```

The **sum-centred** convention is what makes `E_π[C] = 0` hold exactly under a
uniform random permutation of the node attributes.

Train-only per-coordinate RMS scaling (no dataset-mean subtraction):
`C̃ = C / D_C` with `D_C = sqrt(E_train[C²] + ε)`; coordinates whose raw train
RMS is `≤ 1e-9` are masked to zero (safe handling) and counted.

## 2. The three models

Shared `S` encoder `65 → 64 → 64` (SiLU), `z_S = [Σu, mean u, log1p(n),
log1p(m)] ∈ R^130`, and a shared strong base
`F0: [z_S, a] (194) → 64 → 32 → 1` (SiLU).

```
M0 : ŷ = F0([z_S, a])
MB : ŷ = F0([z_S, a]) + <W_B, C̃>     W_B ∈ R^{65×28}, no bias, zero-init
MM : ŷ = F0([z_S, a]) + <W_M, P̃>     W_M ∈ R^{65×28}, no bias, zero-init
```

Hard structural guarantees (unit-tested): `W_B`/`W_M` are plain bias-free
parameters; there is no MLP, activation or square after `B`; `C`/`P` reach the
prediction only through that one linear functional (forcing the statistic to
zero makes `MB`/`MM` reproduce the base exactly); zero-init makes `MB`/`MM`
predict exactly `M0` at step 0; the `S` and `F0` tensors are bit-identical
across the three variants at the same seed.

| model | S encoder | base F0 | binding linear | total | active path |
|---|---:|---:|---:|---:|---:|
| `M0` | 8,384 | 14,593 | 0 | **22,977** | 22,977 |
| `MB` | 8,384 | 14,593 | 1,820 | **24,797** | 24,797 |
| `MM` | 8,384 | 14,593 | 1,820 | **24,797** | 24,797 |

`dataset_dependent_vocabulary_params = 0`.  `MB` and `MM` are exactly
parameter- and architecture-matched; they differ only in the statistic.

## 3. Correctness (all pre-registered tests)

`tests/test_fsar_r2_ar0.py`: **17/17 pass** locally and the remote `sanity`
stage reports **17/17 checks, `all_pass = true`**.

* node relabel invariance of `M0`/`MB`/`MM`;
* `S` chemistry purity (and the implementation reads only adjacency);
* `A` assignment invariance; `P` assignment invariance;
* `C` assignment sensitivity (endpoint↔endpoint symmetric, endpoint↔internal
  different);
* exact permutation expectation `mean_π C = 0` (all 24 permutations of a
  4-node toy, `|mean| < 1e-10`);
* scalar `B` exact expectation `mean_π <W_B, C> = 0` (`< 1e-9`);
* batched torch statistics equal the numpy reference;
* no patch-copy inconsistency (one `phi` row per original node; per-graph
  attribute multiset preserved; `phi` untouched);
* undirected bond counting (3 undirected edges from 6 directed entries);
* train-only scaler (matches the hand-computed train RMS; changes when
  held-out data is added; is **not** a centred std, so no mean subtraction);
* gradient viability (`W_B`/`W_M` receive non-zero task gradient);
* no mixed bypass (zeroing the statistic reproduces the base; AST scan of the
  model's `batch.*` attribute accesses);
* architecture identity and zero-init equivalence;
* parameter counts.

Remote `gradient_audit` (deterministic CUDA, after 2 real optimizer steps):
`M0` active 22,977/22,977, `MB` 24,797/24,797 with `W_B` non-zero grad,
`MM` 24,797/24,797 with `W_M` non-zero grad.  Loss matches CPU
(1.24546 / 1.09349 / 1.09443).

## 4. Synthetic controls (both pass, run before any ZINC run)

**Positive (assignment-only).**  Fixed 4-node path, fixed multiset 1 N + 3 C,
label = N at endpoint vs internal.  `S` marginal and `A` marginal identical
across examples; only the assignment differs.

| model | valid MAE mean (3 seeds) | `||W||` values |
|---|---:|---|
| `M0` | 0.5653 | 0, 0, 0 |
| `MB` | **0.0216** | 0.0567, 0.0575, 0.0529 |
| `MM` | 0.5247 | 0.0331, 0.0781, 0.0218 |

`MB` learns the assignment; `M0`/`MM` cannot; `W_B` does not collapse.

**Negative (marginal-only).**  Label depends only on the topology marginal and
the attribute counts; attribute placement is randomised within each molecule.

| model | valid MAE mean |
|---|---:|
| `M0` | **0.0069** |
| `MB` | 0.0169 |
| `MM` | 0.0247 |

No repeatable `MB` advantage.  Both controls pass, so the formal ZINC round is
authorised.

## 5. Cheap feature audit (before training)

`feature_audit.json` (500 molecules):

* `phi` (65-D): effective rank 9.11; 11 zero-std coordinates (constant by
  construction, e.g. the root indicator); no NaN/Inf.
* `A` (64-D): effective rank 16.69; 24 zero-std coordinates (unused padded
  categories / rare types); no NaN/Inf.
* `C` (65×28): effective coordinates **1109/1820**; cross-molecule effective
  rank **1.40** (top singular fraction 0.89) — `C` is close to rank-1 across
  molecules, so the linear assignment readout has roughly one effective degree
  of freedom.  This is a descriptive diagnostic, not a design change.
* `P` (65×28): effective coordinates **1151/1820**.
* train-only scalers: `C_rms max 0.963`, `P_rms max 32.47`,
  `dataset_mean_subtracted = false`.
* `pearson(||C||, n) = 0.363`.

Transport determinism: the scaler was rebuilt from the pulled cache and matches
`sqrt(E_train[C²])` / `sqrt(E_train[P²])` exactly; freshly re-extracted
`phi`/`A`/`y` are bit-equal to the cached values for molecules 0, 5, 1234,
9999.

## 6. Training protocol and provenance

Inherited verbatim, identical for all three models: Adam, `lr 1e-3`,
`weight_decay 1e-5`, batch 128, L1 loss, grad clip 5.0, no scheduler,
`max_epochs 240`, `patience 40`, best-official-valid checkpoint, fixed
equal-weight Top-5 soup, `torch.use_deterministic_algorithms(True)`.

| item | value |
|---|---|
| GPU model | NVIDIA A100-SXM4-40GB |
| CUDA / PyTorch / Python | 12.4 / 2.5.1+cu124 / 3.12.14 |
| training commit | `fc1312f` |
| GPU 0 | unknown third-party python task, 35.8 GB / 94–97 % for the whole session — **never touched, never co-tenanted** |
| GPU 1 | ours, plus an unrelated GSN process that stayed at ~0.54 GB — recorded, never touched |
| our peak GPU memory | 77.3 MB (`M0`) / 100.7–100.8 MB (`MB`, `MM`) |
| DDP | none; one process per job |
| official ZINC test | never loaded |

Waves (two independent jobs in parallel on GPU 1): short smoke `M0/MB/MM`
(3 jobs), then `M0-s0 + MB-s0`, `MM-s0 + M0-s1`, `MB-s1 + MM-s1`, and after the
gate fired `M0-s2 + MB-s2`, then `MM-s2`.

**Determinism check.**  A full re-run of `MB` seed 0 under a separate tag
reproduced the soup MAE `0.4692853513918235`, the Top-5 epoch list
`[58, 46, 81, 65, 90]` and the selection-state SHA-256
`76b373a69ff32cb3432f7a170e044921` exactly.

## 7. Short runs (60 epochs, seed 0)

| model | best valid | Top-5 soup | branch norm | wall (s) |
|---|---:|---:|---:|---:|
| `M0` | 0.560572 | 0.559153 | 0.0000 | 96.0 |
| `MB` | 0.477389 | 0.470626 | 0.7467 | 93.6 |
| `MM` | 0.553141 | 0.546370 | 0.6361 | 94.0 |

All finite, normal learning curves, non-zero branch gradient; no collapse / NaN
/ constant-output bug.  (They are a trainability smoke only and did not select
anything.)

## 8. Formal results (valid MAE, fixed equal-weight Top-5 soup)

| model | seed | best valid | Top-5 soup valid | best epoch | Top-5 epochs | branch norm | wall (s) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `M0` | 0 | 0.556476 | 0.553790 | 65 | 65/89/66/87/98 | 0.0000 | 161.4 |
| `MB` | 0 | 0.477389 | **0.469285** | 58 | 58/46/81/65/90 | 0.7467 | 155.7 |
| `MM` | 0 | 0.538529 | 0.528340 | 108 | 108/80/96/101/148 | 0.8028 | 230.9 |
| `M0` | 1 | 0.535439 | 0.535570 | 226 | 226/230/209/228/208 | 0.0000 | 363.1 |
| `MB` | 1 | 0.464546 | **0.458746** | 149 | 149/171/121/133/148 | 1.0258 | 307.1 |
| `MM` | 1 | 0.551367 | 0.543241 | 52 | 52/56/90/61/92 | 0.7263 | 145.2 |
| `M0` | 2 | 0.549145 | 0.546893 | 87 | 87/94/112/126/114 | 0.0000 | 190.2 |
| `MB` | 2 | 0.467469 | **0.462365** | 114 | 114/153/112/95/145 | 0.9365 | 240.7 |
| `MM` | 2 | 0.529842 | 0.524873 | 202 | 202/198/231/190/234 | 1.0634 | 370.4 |

Paired deltas (positive = `MB` better):

| seed | `M0 − MB` | `MM − MB` | both > 0 |
| ---: | ---: | ---: | :---: |
| 0 | **+0.084505** | **+0.059055** | yes |
| 1 | **+0.076824** | **+0.084495** | yes |
| 2 | **+0.084528** | **+0.062507** | yes |

Paired per-molecule bootstrap (5,000 resamples over the 1,000 valid molecules,
seed 20260917), 95 % percentile CI:

| seed | `M0 − MB` mean [CI] | `MM − MB` mean [CI] | `P(delta>0)` |
| ---: | --- | --- | :---: |
| 0 | 0.084324 [0.067510, 0.101410] | 0.058914 [0.041431, 0.076366] | 1.0 / 1.0 |
| 1 | 0.076593 [0.060364, 0.092985] | 0.084304 [0.066661, 0.102764] | 1.0 / 1.0 |
| 2 | 0.084605 [0.067804, 0.101036] | 0.062607 [0.045553, 0.079732] | 1.0 / 1.0 |

Seed 2 was purchased exactly because the pre-registered gate fired on both
seeds 0 and 1 (`delta_base > 0` and `delta_capacity > 0`); `gate.json` records
`seed2_authorized = true`.  No width, radius, primitive or optimizer was
changed at any point.

## 9. Mechanism diagnostics (soup states)

`diagnostics` (full 1,000-molecule valid):

| model | seed | `||W||` | `W` sparsity | branch out std | base out std | branch live |
|---|---:|---:|---:|---:|---:|:---:|
| `MB` | 0 | 0.7773 | 0.391 | 0.673 | 1.406 | yes |
| `MB` | 1 | 1.0066 | 0.391 | 0.677 | 1.397 | yes |
| `MB` | 2 | 0.9620 | 0.391 | 0.653 | 1.389 | yes |
| `MM` | 0 | 0.7974 | 0.368 | 1.165 | 0.750 | yes |
| `MM` | 1 | 0.7841 | 0.368 | 1.177 | 0.724 | yes |
| `MM` | 2 | 1.0779 | 0.368 | 1.180 | 0.742 | yes |

No collapse: every branch is live with output std ≈ 0.65–1.18 and none of the
near-dead-channel pathology of the earlier FSAR rounds.  `MM`'s branch absorbs
a larger share of the variance (it can see graph size through `P`) yet still
loses to `MB`, so the `MB` gain is not "more live capacity".

**Evaluation-only assignment shuffle** (`witness`, 5 within-graph permutations
of the original-node attributes, soup state):

| model/seed | Δ`A` | Δ`S` | Δ`C` | actual MAE | MAE(`E[yhat_π]`) | `E[MAE(yhat_π)]` | mean abs pred change |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MB` s0 | **0.0** | **0.0** | 44,445 | **0.4693** | 0.8720 | 1.0195 | 0.7922 |
| `MB` s1 | **0.0** | **0.0** | 43,915 | **0.4587** | 0.8545 | 1.0034 | 0.7952 |
| `MB` s2 | **0.0** | **0.0** | 44,395 | **0.4624** | 0.8423 | 0.9774 | 0.7632 |

`A` and `S` are **exactly** invariant to the attribute assignment; `C` and the
prediction change materially; the permutation-mean prediction MAE
(`MAE(E[yhat_π])`, ≈ 0.84–0.87) and the mean per-permutation MAE
(`E[MAE(yhat_π)]`, ≈ 0.98–1.02) are both far worse than the actual MAE
(≈ 0.46–0.47).  The trained model uses the assignment.  This is a mechanism
diagnostic only; it does not replace the independently trained `M0`/`MM`
controls.

## 10. Verdict

**`ASSIGNMENT_SUPPORTED`** (per the pre-registered gate, three paired seeds).

> Under the current fixed radius-2 explicit pure-topology coordinates, the real
> node structural-role ↔ atom-attribute assignment provides a stable predictive
> increment beyond the structure/attribute marginals and beyond a matched
> marginal-capacity control.

What is **not** claimed:

* no SOTA / benchmark claim — all three models are marginal-only-scale and far
  from the full-backbone numbers;
* no claim that "structure-attribute binding matters in general" (only this
  node-level linear residual on this basis/budget);
* no claim about the historical `B-full`; it remains an external mixed-model
  reference and was never used to select a primitive or as a pure-topology
  teacher;
* no causal decomposition from the shuffle diagnostic.

## 11. Caveats and honest limits

* `C` is close to rank-1 across molecules (effective rank 1.40), so the linear
  assignment readout has roughly one effective degree of freedom.  The result
  says this single effective direction carries real assignment information, not
  that a rich 65×28 assignment tensor is used.
* `MM`'s `P` statistic is one specific marginal functional, not "all
  marginals"; the strong `M0` baseline and `A` cover the marginal side.  `MB`
  vs `MM` is the matched-capacity comparison; `MB` vs `M0` is the
  marginal-baseline comparison.  Both are positive on every seed.
* The round is on ZINC-12K official train/valid only, deterministic A100, three
  seeds; the effect is large (~0.06–0.08 MAE) relative to the training spread,
  so it is not a near-noise result.
* The synthetic positive control is a deliberately clean assignment-only
  construction; passing it establishes that the implementation can learn an
  assignment, not that ZINC assignment is equally clean.

## 12. What was explicitly not done

No official test; no radius sweep; no width/depth sweep; no new primitive
(C4/C5, RRWP, homomorphism); no attention/Transformer; no distillation; no
auxiliary loss; no adaptive gate; no validation-driven feature selection; no
training `S` from historical `B-full` hidden states; no mixed-descriptor
fallback; no seed hunting; no optimizer sweep; no checkpoint cherry-picking
beyond the frozen Top-5 rule.

## 13. Minimal next experiment

The node-level linear residual is supported.  The single smallest next step
that follows from this design (and was named in the pre-registration as the
follow-up for a positive outcome) is a **pair-level assignment residual**: the
same fixed pure-topology coordinates, but the assignment statistic is formed
over structural *relations* (which attribute pair sits on which structural
relation) rather than over single nodes, with the same strong marginal baseline
and a matched marginal-capacity control.  This is recorded here as the next
candidate and is **not** implemented in this round.

Do not revisit this round by adding new primitives, widening the encoder,
changing the radius, or reading the official test.

## 14. Reproduce

```
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 preprocess
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 audit
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 sanity
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 synthetic
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 gradient_audit --all --device cuda --deterministic
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 run --variant MB --seed 0 --device cuda --deterministic
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 diagnostics --variant MB --seed 0 --state soup
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 witness --variant MB --seed 0 --state soup
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 gate
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 analyze
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 decide
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_r2_ar0 report
```

Artifacts: `results/fsar_r2_ar0/` — `cache/`, `curves/`, `states/`,
`soup_states/`, `runs/`, `sanity.json`, `feature_audit.json`,
`parameter_accounting.json`, `gradient_audit_{M0,MB,MM}.json`,
`synthetic_controls.json`, `soup_*`, `diagnostics_*`, `witness_*`, `gate.json`,
`analysis_paired_bootstrap.json`, `decision.json`, `FORMAL_RESULTS.md`.
