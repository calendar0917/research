# PROTOCOL — zinc-support-transfer-probe-v1

Freeze written before the first formal GPU run. Round: `zinc_support_transfer_probe_v1`.
Base commit: `795bb39` (previous round closed, no leftover jobs). Host: `res`
(a100-2, 2 × A100-SXM4-40GB), one process per GPU.

## 0. Question, boundary, hypotheses

Final target unchanged: official ZINC subset raw-`y` **test** MAE < 0.060. This
round does not attempt a new trunk. It asks one narrow question:

> When the base model has **no supervision support at all** for the local
> structural types carried by a molecule, can a shared correction learned from
> *other* structures' genuine out-of-sample errors transfer to those molecules?

* **H1** The base model's signed error `r = y − f_base` on support-starved
  molecules contains a regularity that transfers across structural types.
* **H2** Local structural features add something beyond the prediction value,
  molecule size and support counts.
* **H3** The correction transfers to the full-10k main trunk and improves
  official valid.

H1/H2 internal gain does **not** imply H3; an H3 gain of 0.002 does not mean the
0.05x target is near. A negative result bounds the specific structural features,
corrector, support range and training condition used here — not "all structural
transfer is impossible".

Not reopened: T/N channels, Kekulé/aromatic training, exact-key lookup residual,
primary-trunk search, new attention trunks, hyper-parameter grids, external data.
Single model, explicit structural features, no recursive hidden-state update on
the local adjacency, total parameters (frozen included) ≤ 500,000.

## 1. Data roles (official train 10,000 only; built by `code/zst_split.py`)

Radius-2 / radius-1 content keys are the **old builder, unchanged**
(`zct_keys.py`: `k1(v) = (atom_type, sorted(bond_type, neighbour atom_type))`,
`k2(v) = (atom_type, sorted(bond_type, k1(neighbour)))`; `k1 = f(k2)` asserted at
build time). "Support" of a key = number of **molecules** in the stated set that
contain the key. Nothing here is an exact rooted induced-graph signature; the
content key is unchanged from the previous round.

Candidate keys: `k2` molecule support in 4..25 over the full 10k, radius-1
parent support ≥ 50. Fixed RNG `20261121`; no label, prediction or error enters
key choice.

| role | content | purpose |
|---|---|---|
| `F` | everything left after the roles below | base model supervised fit |
| `C_fit` | all molecules of key family `C` + 250 background molecules (no selected key) | correction training |
| `C_dev` | all molecules of key family `D` | correction selection |
| `Q_A` | all molecules of key family `A` | transfer evaluation, package 1 |
| `Q_B` | all molecules of key family `B` | transfer evaluation, package 2 |
| `Q_bg` | 400 molecules carrying no selected key | background evaluation |

Construction order `A → B → C → D → C_fit background → Q_bg → F = rest`, with
**whole-molecule** assignment: every molecule carrying any selected key goes to
that key's family, and a family never receives a molecule already assigned.
Family `B` first accepts only keys whose radius-1 parent is not used by `A`
(parent-overlap minimisation), then relaxes once to reach the quota; the residual
parent overlap is reported, not removed by force. No component is split to make
a quota; if a quota cannot be met, the shortfall is reported.

Hard isolation requirements, each verified in `analysis/split_gates.json`:

1. support in `F` of every `A`, `B`, `C`, `D` target key is **0**;
2. support in `F ∪ C_fit ∪ C_dev` of every `A`/`B` target key is **0**;
3. support in `F ∪ C_fit` of every `D` target key is **0**;
4. `A`/`B` target keys and molecules are disjoint; families are molecule-disjoint;
5. `Q_bg` never enters the base fit, the head fit or the selection; its own
   unseen-key composition is reported (a "background" molecule is **not**
   automatically fully supported);
6. role assignment is frozen and hashed before any new-query error is computed.

This is a **new holdout for this round**, not a blind test: these train molecules
were seen by earlier rounds' models. The base model is supervised from the
SSL-only initialisation — no supervised prefix of any kind is loaded. The reused
SSL encoder is `bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`
(sha256 `006fabc1…`), a **masked atom/bond reconstruction** pretext over the 10k
train structures with no labels. Results are therefore "supervised-support
transfer given a shared structural pretraining", not strict unseen-graph
inductive generalisation.

## 2. Base model `F` (two seeds)

`zinc_recipe_trunk_v2/ad2_models.P3NullModel`, 413,505 parameters, features and
recipe unchanged: recipe B AdamW 1e-3 → 1e-5, 500-step warm-up then per-step
cosine, wd 1e-5, betas (0.9, 0.999), eps 1e-8, clip 5, FP32, dropout 0, batch
128, **600 epochs**, steps per epoch = `ceil(|F| / 128)` recomputed from the real
`F` size. The endpoint is the **epoch-600 checkpoint**; no valid/C/Q-based
checkpoint selection, and official valid is not read at all in this phase. Both
seeds share the same seed-0 SSL encoder (repo precedent); the init state is
hashed so the parity with the historical seed-0 init can be checked.

## 3. Corrector: B / G / S

* **B** — the frozen base model, unchanged.
* **G** — generic support-conditioned correction. Context only: `f_base`,
  size / ring rank, atom composition, unseen atom count and fraction, radius-1
  parent support of the unseen atoms, and the fixed 33-dim global descriptor.
  No description of the flagged local environment.
* **S** — G's context plus a shared encoding of the unseen local environment:
  `psi([x_v, log1p n1(v), log1p n2(v)]) → 16`, with `x_v` the existing 108-dim P3
  atom descriptor and `n1/n2` the molecule support of the atom's `k1/k2` **in the
  base fit that produced the prediction**; sum and mean pooling over atoms with
  `n2 = 0`, plus unseen count and total atom count. No trainable table indexed by
  an exact `k2` id; `k2` is only used for support lookup and for role assignment.

`psi` 110→32→16; S readout 98→64→1 (10,481 params); G readout 64→160→1
(10,561 params, +0.8% — the two heads are capacity-matched without padding).
Base + head ≤ 425k total < 500k.

Prediction form, **both** G and S:

```
m_F(G) = 1[ molecule has ≥1 atom with n2 = 0 ]
f_corrected(G) = f_base(G) + m_F(G) * h(context, support_F, f_base)
```

The gate depends only on the base fit's support map, never on query identity or
on a target-key list. Fully supported molecules are unchanged **by design**;
that identity is not a result.

Head training: base frozen; target is the genuine out-of-sample residual
`r = y − f_base` of `C_fit` molecules with `m_F = 1`; loss is raw-`y` L1 on the
*corrected* prediction (`base + r̂`), i.e. the sign convention is `r = y − f_base`
and the correction is added. Last layer zero-initialised so the initial head
reproduces B exactly. Input standardisation statistics come from `C_fit` only;
columns whose C_fit standard deviation is below 1e-3 keep scale 1 instead of
`1/eps`, so an input column that is constant in `C_fit` cannot be amplified by
six orders of magnitude on another family. The reused P3 global normalisation is
disclosed. Fixed width/activation/lr (Adam 2e-3 with per-step cosine to 1e-4,
1500 full-batch steps, grad clip 1); `C_dev` corrected MAE is evaluated every 50
steps and the best-`C_dev` step is kept as the checkpoint (allowed selection).
The only hyper-parameter grid is at most two L2 strengths {0, 1e-3}, selected on
`C_dev` mean over head inits. Three head initialisations per (base seed, head)
are all reported; no "best init" cherry-picking.

*Pre-run amendment (A1).* The first version of this section specified Adam 3e-3 /
2000 steps / no step-wise `C_dev` checkpointing and a `max(std, 1e-6)` standard
scale. A 3-epoch CPU stub base (not a reported model) showed the head diverging
to |correction| ≈ 40 on `C_dev` because a column constant in `C_fit` was divided
by 1e-6. The scale floor and the `C_dev` checkpointing were fixed **before any
formal base model was trained**; no reported number exists under the old
settings.

Optional structural-alignment control (only if S clearly beats G): within
`C_fit`, shuffle the unseen-atom block ↔ residual-record correspondence inside
coarse strata (size tercile × base-prediction tercile × unseen-fraction
tercile), freeze the permutation, retrain the identical S, evaluate on the
normal query inputs.

## 4. Evaluation and decision

Primary, per base seed × package (`Q_A`, `Q_B`): MAE of B, G, S; `S−B`, `G−B`,
`S−G`; per-molecule paired AE differences; correction sign/size; the actually
modified `Q_bg` subset; the fully supported identity replay; drop-worst-1%
description (does not change the primary metric). Strata by training support,
parent support and unseen mass are reported descriptively, fixed before the read.

Uncertainty: molecule bootstrap (fixed predictions) and key-cluster bootstrap
(keys are molecule-disjoint inside a family, so a key is the structural cluster;
the number of effective clusters is reported). Two packages and two base seeds
are shown separately, never only their average.

Decision reference (not a mechanical universal gate; 0.02 is the old
label-support difference size, not a theorem):

* **A** — S beats B consistently on both packages and both seeds, clearly beats
  G, and the gain is not carried by one or two molecules → structural transfer
  supported → §5.
* **B** — G improves, S adds nothing → generic correction only; §5 only if the
  performance value is large enough, named honestly.
* **C** — old package gains, new package or the other seed does not reproduce →
  not transferable / unstable, stop expanding.
* **D** — head fits `C_fit` but query gain ≈ 0 and the interval excludes a
  worthwhile effect → stop this corrector hypothesis.
* **E** — head not trained, too few structural clusters, or the interval covers
  both zero and a useful gain → **uncertain**; at most one targeted supplement,
  and if the obstacle is structural coverage, more random seeds are not the fix.

## 5. Conditional official-valid check (only if §4 gives A or a large B)

Reuse the two previous-round `E` epoch-600 frozen checkpoints (full-10k
supervised, same architecture/features/SSL recipe; verified, not retrained).
Apply the frozen head to them with support statistics recomputed from their real
10k supervised fit; raw structural features and the scalar prediction interface
unchanged. Evaluate on official valid only ("one frozen valid-label read",
recorded with its hash): same-trunk original vs same-trunk + frozen head, the
old four groups, overall MAE and the changed fraction. No primary-trunk
retraining, no epoch re-selection, no seed averaging inside a reported cell.

If the internal result holds but full-10k valid does not, the support/
prediction/correction-magnitude distributions on both sides are reported before
any interpretation. A conditional cross-fitting extension (2–3 auxiliary fits,
most of train retained, the target keys' molecules removed, supervised from
SSL-only) is allowed **once** and only if the effect is large enough to be worth
it; auxiliary hidden vectors are never concatenated and auxiliary predictions are
never averaged at inference. Official test is read once, at the end, only for
the two frozen candidates plus their own B baselines, and never used to select.

## 6. Planned cost

Two `F` base trainings (600 epochs, |F| ≈ 7.5k, two GPUs in parallel):
≈ 0.25–0.35 GPU·h each, ≈ 20 min wall. Everything else (residual generation,
head training, evaluation) is CPU on the frozen predictions. No profiling job is
scheduled separately; the first epoch wall time is recorded inside the training
job. If a numeric or implementation fault appears, the job is stopped, fixed,
and the cost is booked.

## 7. Deviations

Recorded in this file and in `REPORT.md` §deviations. The split gates are
verified before the base training; a dataset-construction failure is fixed
before any result is reported, never papered over.

## A2 — selection statistic (written before the new-holdout C_dev was computed)

While preparing the old-package exploration (a CPU smoke run on `sup_minus_s0`
predictions that were already public in the previous round's report), `C_dev`
turned out to be a single-molecule-dominated statistic: in that package 5 of 110
molecules with `y ≈ −7…−9.6` and `|error| ≈ 7` carry **80%** of the `C_dev` MAE
(0.402 with them, 0.066 median without). These are the same permutation-unstable
label rows the repository documented in Phase 2a.

Therefore the **checkpoint/regularisation selection statistic is `C_dev` MAE
after dropping the 5% of `C_dev` molecules with the largest *base* absolute error**
(`dev_mae_trim5`). The drop rule is a function of (`f_base`, `y`) on the selection
split only — the correction never enters it, and the query packages are untouched.
The untrimmed `C_dev` MAE is reported next to it everywhere. This was fixed before
any new-holdout `C_dev` number existed, and it is the only change to §3's
selection rule.
