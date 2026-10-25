# zinc_dccl_bottleneck_v1 — pre-registered plan (written before any analysis run)

Round owner: local CPU audit (phase A) + one matched sample-size intervention (phase B).
Target model: frozen champion DCC-L (`B-R0`, 494,193 params, test 0.067319 ± 0.000709,
`../zinc_performance_search_v1/`), four checkpoints (s0–s3) with saved official-valid
per-molecule predictions. Official test is never constructed or read in this round.

## Question

Separate three explanations of the 0.067 platform, and name the one that deserves the next
purchase:

* **H1**: supervision/sample utilisation — too few labelled molecules per structural motif.
* **H2**: available structural information is not expressible / not routed into the learned
  function (interface/capacity of the forward pass).
* **H3**: a small set of structural regions fails by systematic extrapolation (a
  region-specific functional error, visible as a signed bias shared by all seeds).

## Phase A — fixed definitions (frozen before any group table is computed)

Data: official valid 1,000 rows (`ad_valid.pt` cache, sha recorded), official train 10,000
rows for every support statistic. Valid labels are used only for error measurement and for
the label-dispersion statistic of *train* neighbours; no selection of a threshold, neighbour
set or weight is made on valid labels or on valid errors.

Primary error readout: per-row absolute error averaged over the four DCC-L seeds;
signed error `e_k = pred_k − y`. Full-sample MAE = mean over rows of mean over seeds.

### Pre-fixed partitions

1. **MECE table 1 (12 cells)** — cross of
   * `U`: `r2_unseen_atom_frac > 0` (≥1 atom whose exact typed radius-2 signature never
     occurs in train 10k) vs `= 0`;
   * `T`: `y ≤ −3` (extreme tail, the historical definition) vs `y > −3`;
   * `R`: `n_rings ∈ {0–1, 2–3, ≥4}`.
2. **MECE table 2 (9 cells)** — cross of
   * `S`: atom-count tertile, cuts taken from the **train** `n_atoms` distribution;
   * `P`: support tier — `full` (no unseen r2 atom and no atom whose r2 signature has
     1–4 train molecules), `low` (no unseen, but some 1–4 support atom), `unseen` (any
     unseen r2 atom).
3. **MECE table 3 (8 cells)** — `U × T × S` (small/large split at the train median).

Thresholds are the historical ones (radius-2 exact typed signature, `y ≤ −3`, ring bins) or
train-distribution quantiles; they are not tuned on valid error.

### Confounder control

* Raw and partial Spearman between `AE` and `r2_unseen_atom_frac`, residualising ranks on
  `n_atoms`; then on `n_atoms + cycle_rank`; then on `n_atoms + cycle_rank + |y|`.
* Size-and-ring standardised ΔMAE (unseen − seen), i.e. weighted average of within-stratum
  Δ with strata = size tertile × ring bin; strata with <10 molecules on either side are
  reported but excluded from the standardised number.
* The same table recomputed after removing rows with `AE > 1` (single-molecule dominance
  check) and per seed.

### Seed structure (signed error)

* Per group and per seed: mean signed error, and the number of seeds agreeing on the sign.
  A group counts as *systematic* only if all four seeds share the sign and the magnitude
  exceeds the between-seed spread (min |mean_k| > 1.5 × sd_k).
* Cross-seed Pearson correlation of signed errors, overall and per group.

### Pre-fixed target-free similarity support

Two similarity definitions, both computed from structure only (atom types, bond types,
connectivity), fixed before any valid error is inspected:

* **Primary**: Weisfeiler–Lehman subtree kernel, 2 iterations with bond-type edge labels,
  concatenated level counts, cosine similarity. `k = 10` nearest train molecules.
* **Secondary**: Tanimoto on the *set* of exact typed radius-2 signatures. `k = 10`.

Per valid molecule, from its 10 nearest train neighbours:

* `sim_top10` = mean of the 10 cosine (Tanimoto) similarities; `sim_max`;
* `nn_ystd` = population std of the 10 neighbour labels; `nn_ymad` = mean |y_j − median|;
* `nn_pred` = similarity-weighted mean of the 10 neighbour labels (weights = raw
  similarities); `nn_err = |nn_pred − y|`.

`nn_pred` is the pre-fixed **kNN reference**: a nonparametric user of exactly the same
10k labels with no learned parameters. It is a *descriptive upper bound on what local
structural support alone can deliver*, never a model candidate.

Discriminating readout (H1 vs H2/H3):
`support` = `sim_top10` (tercile) × `label dispersion` = `nn_ystd` (tercile) →
per-cell MAE, share of total AE, kNN reference error. A large error mass in the
*high support / low dispersion* cell means the labels do support the answer and the
learned function still misses it (H2/H3); error mass concentrated in *low support /
high dispersion* means H1 (effective supervision) is the live explanation.

## Phase B — matched sample-size intervention (only if phase A is not decisive)

Reuse the frozen nested split of `tracks/structure-contribution/results/
bjoc_predictive_bottleneck_v1/split_a.json` (Fit8k / Monitor / Locked-dev, nested Fit4k),
which lives entirely inside the official ZINC train split. Train the **native DCC-L**
(builder `dcc_runner.make_builder("L")`, same 494,193 params, same seed-matched SSL encoder,
same R0 recipe, same recipe numbers) with

* identical optimizer-update budget T = 47,400 (= 600 epochs × 79 updates at 10k),
* `Fit4k`: 32 updates/epoch → 1,481 epochs; `Fit8k`: 63 updates/epoch → 753 epochs,
* per-fit standardization computed only from the corresponding Fit rows,
* checkpoint selection on **Monitor** only; Locked-dev read once at the end,
* official valid reported descriptively (no selection, no test).

Primary contrast: `Fit8k − Fit4k` on locked MAE (seed 0), plus monitor MAE; secondary:
official-valid level vs the historical 10k DCC-L (`0.075990` seed mean). seed 1 only if
seed 0's direction is clear. Hard cap 2.0 A100·h.
