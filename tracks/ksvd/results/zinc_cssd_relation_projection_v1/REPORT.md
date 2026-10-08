# REPORT — zinc_cssd_relation_projection_v1

**Round question.** With the verified frozen CSSD basis and its effective local access held
fixed (the DICT consumer of `zinc_cssd_consumer_replacement_v1`), does letting the EXISTING
distance buckets select the endpoint projection — *before* the local representation is
compressed — improve the complete y and g?  In the deployed consumer every environment pair
first shares ONE projection `P: E_i -> 48` (no-bias Linear 144->48); only then are the
endpoint sum / |difference| / product * distance-gate computed, the relation encoded, and the
pair input pushed through the pair encoder and pooled by the five existing buckets.

**Design (frozen at 404b7a5, design basis 7aec57f).**  Three arms x body seed 0, ONE
from-scratch 240-epoch training each (15,120 steps, batch 128, Adam lr 1e-3, coupled wd 1e-5,
clip 5, FP32, no scheduler/AMP/early stopping; estimator = equal-weight FP32 mean of the full
member states 236..240):

| arm | endpoint projection | projection params | total |
|---|---|---|---|
| BASE | shared no-bias Linear 144->48 (historical P0) | 6,912 | 297,539 |
| SHARED_MLP (C) | shared no-bias 144->180->48 SiLU MLP, fixed function-preserving init (W1=[P0; −P0; default rows], W2=[I; −I; 0] ⇒ C(E)=P0E at init, max abs 6e-8) | 34,560 | 325,187 |
| BUCKET_LINEAR (R) | five no-bias P_b[48,144], one per EXISTING distance bucket, each a bitwise copy of P0; endpoint views gathered by the pair's projection route | 34,560 | 325,187 |

The relation input, distance gate, pair readout pooling and frozen Q always read
`data.pair_bucket`; only the endpoint views are routed; the route override never writes
`data.pair_bucket`; E_i is generated before any pair conditioning and never updated by pairs.
Official valid/test never loaded; dev = the 1999 historical development rows (a repeatedly
used development comparison — never a new confirm).  Single body seed: directional screening
only, never a repeatability claim; the historical B72 A–E gates are not inherited.

## Execution (all through the registered runner, clean committed revision 404b7a5)

* source-checks + pretrain-checks: CPU (rr runs 20261008-120123-214b7394,
  20261008-120143-294331af; deterministic re-runs of the pre-freeze checks, all passed).
* smoke: GPU res2-cu124 c05 (20261008-104237-c97913db), all three arms through the full train
  path, 63 steps each, all checks passed.
* formal trainings: GPU res2-cu124 c05, A100-PCIE-40GB, torch 2.5.1+cu124 —
  SHARED_MLP 20261008-104553-2713167e (1144 s), BUCKET_LINEAR 20261008-110529-f448bce2
  (1091 s), BASE 20261008-112412-24ab5788 (837 s); Slurm quota gpu=2 ⇒ two waves, one body per
  GPU at a time.
* terminal eval: ONE-SHOT, 20261008-114950-f26cbe0d; roster 11/11 checks ok (BASE init ==
  historical DICT s0 init hash; C/R non-projection init bitwise == BASE; schedule bit-identical
  to the historical plan; frozen basis and Q hashes unchanged; members/soups hash-verified).

**Disclosed engineering incidents (no science touched):**
1. Two wiring bugs were caught by the fit-only pretrain checks BEFORE any training (route
   override built from pair-row indices instead of permuted bucket labels; override length
   from node counts instead of pair counts) and fixed pre-freeze; the committed checks now
   cover both.
2. First terminal submission used `--mode terminal`, which grants test access and was refused
   by the runner's non-terminal guard (exit 2 before any eval; nothing written).  Retried with
   `--mode scratch` — the same mode every non-terminal round of this line uses.
3. The first terminal `--result` declaration pointed at a non-existent `terminal/` subdir; the
   artifacts were retrieved via `rr pull --path` (exact file paths) and copied into the round's
   results directory; manifests and hashes verified against the promoted run records.

## Results (dev = 1999 historical rows; negative delta = improvement)

| arm | dev y_raw | dev g_raw | fit y_raw | fit g_raw | gap_dev−fit(g) | b_y | seconds | peak GPU |
|---|---|---|---|---|---|---|---|---|
| BASE          | 0.11920 | 0.09169 | 0.03797 | 0.02985 | 0.06184 | +0.01133 | 837 | 378 MB |
| SHARED_MLP (C)| **0.11659** | **0.08935** | 0.04345 | 0.03507 | 0.05429 | −0.01485 | 1144 | 383 MB |
| BUCKET_LINEAR (R) | 0.12824 | 0.10098 | 0.05580 | 0.04748 | 0.05350 | −0.04249 | 1091 | 378 MB |

Paired deltas (candidate − reference; negative = improvement) with group-paired bootstrap CI95
(2000 draws, seed 20261008, shared group picks across all three arms AND both metrics):

| contrast | Δy (dev) | Δy CI95 | Δg (dev) | Δg CI95 | Δy (fit) | Δg (fit) |
|---|---|---|---|---|---|---|
| R − BASE | **+0.00904** | [+0.00552, +0.01259] | **+0.00929** | [+0.00578, +0.01276] | +0.01782 | +0.01763 |
| C − BASE | **−0.00261** | [−0.00590, +0.00085] | **−0.00234** | [−0.00583, +0.00127] | +0.00547 | +0.00522 |
| R − C    | **+0.01165** | [+0.00783, +0.01524] | **+0.01163** | [+0.00789, +0.01518] | +0.01235 | +0.01241 |

* C improves both complete y (−0.00261, −2.2% relative) and g (−0.00234, −2.6%); both CIs
  cross zero — a directional signal at this seed, not a separable effect.
* R is WORSE than BASE on both metrics, and the harm is CI-separated from zero on both; it is
  also worse on FIT (+0.0178 y) — the degradation is present in the training optimum itself,
  not a fit–dev gap artifact.  R's final on-the-fly train Lg is the lowest of the three
  (0.0586 vs BASE 0.0708) while its soup fit MAE is the worst — the bucket-split family
  optimized the streaming objective differently and still landed worse; with one seed this is
  an observation, not a mechanism claim.
* Historical anchor (descriptive only, different GPU-trajectory draws): replacement DICT_s0
  dev y_raw = 0.12152; this round's same-recipe BASE lands at 0.11920 — the same-round paired
  comparison, not the historical number, carries the recipe effect.

## Mechanism checks (terminal, noise eta = 1.07e-6 ⇒ marker = 1e-4)

**Dictionary readout (intervention 1, the unique alpha-mean intervention).**  All three soups
remain clearly dictionary-reading: |Δpred| p95 = 2.37 (BASE), 6.66 (C), 2.69 (R), response
fraction 1.000 each, hugely damaging swaps (ΔMAE_y +1.11 / +3.76 / +1.27).  Nothing was
diluted by the projection changes; the readout is intact everywhere.

**Route usage (intervention 2, one fixed shuffle shared by all arms; 66.2% of the 531,246 dev
pair rows rerouted, 1999/1999 molecules covered; `data.pair_bucket` verified untouched).**

* BASE / C do not read the route: |Δpred| p95 = 3.6e-7 (pure numerical noise, ≪ marker) —
  the shared-mode forward is provably override-invariant.
* R really learned and really uses relation-dependent projections: its five P_b diverged
  strongly (relative Frobenius deviation from their mean 1.10–1.32 per bucket, mean 1.23;
  max pairwise 1.33), the endpoint views actually change under the shuffle (median relative
  view change 1.10 over the rerouted pairs), and shuffling DEGRADES R massively
  (|Δpred| p95 0.663, response 1.000, ΔMAE_y +0.188, ΔMAE_g +0.190).
  So the negative result is NOT "the route never mattered": R built genuinely different
  per-bucket viewpoints and pays for them — the route is causally loaded in the trained model.

## Reading (frozen single-seed rubric)

Branch: **shared-nonlinear-projection-directional-signal** — C (the shared nonlinear endpoint
projection, parameter-matched to R) improves both y and g vs BASE, while R (the
distance-split relational projection) does not improve; it is reliably worse.  At this seed
the hypothesis "different relations need different endpoint projections, bucket-routed" is
NOT supported; the data instead give a directional signal that a *stronger shared* endpoint
projection helps.  C's CIs cross zero, so even the positive side is only directional.

Scope and caveats, stated plainly:
* ONE body seed — nothing here is a repeatability, stability or significance claim; the
  molecule-resampling CI covers sampling only, not training-seed or basis uncertainty.
* C is a parameter-matched but different function recipe; "R worse than C" does not isolate a
  universal causal mechanism — initialization identity at step 0 does not equalize
  trajectories or implicit regularization (and R's late-phase trajectory visibly diverged).
* The route response is dependence evidence; the shuffle degradation cannot alone prove R's
  route split would have helped under any other recipe.
* dev is the historical development comparison set; y_cal and component MAEs are descriptive;
  the k=0 n_nodes bins (≤23 / 24..27 / ≥28) show the same ordering as the totals with no new
  subgroup story (C ≤ BASE ≤ R across all three bins, e.g. k0 in-group y: 0.0888 / 0.0912 /
  0.0994).

## Costs (measured, not inferred)

Three A100 trainings: 837 s (BASE) / 1144 s (C) / 1091 s (R) curve time; peak GPU 378–383 MB;
total wall clock 3072 s ≈ 51 min — inside the 2-hour budget including smoke (66 s), the
one-shot terminal (51 s) and two pre-freeze engineering fixes.  C/R cost +27,648 projection
parameters each (+9.3%); C trains ~37% slower than BASE in curve time, R ~30% — the measured
times, not the parameter counts, are the cost statement.

## What was purchased / excluded

* Purchased: a clean single-seed answer that bucket-routed endpoint projections hurt this
  frozen consumer while a stronger shared nonlinear projection directionally helps, with the
  route mechanism verified live (P divergence + causal shuffle response) and the dictionary
  readout intact in all arms.
* Excluded at this seed: "relations need their own endpoint projection views before
  compression" as an improvement lever in this architecture.  NOT excluded: the relational
  organization question in general (one seed, one architecture, one route family), and not
  re-opened: any historical negative recipe, bridge widths, windows, seeds beyond 0.

## Recommendation (for the researcher, no automatic purchase)

If this line continues, the live lead is the shared nonlinear endpoint projection (C): a
cheap, function-preserving-init upgrade of the pair front-end with a directional two-metric
improvement at 240 epochs.  Before any second seed, a pre-registered design would need to
state how it separates a ~0.002–0.003 dev effect from trajectory variance (e.g. the paired
two-seed same-round control standard of this line).  The bucket-routed variant is closed for
this architecture at this seed: its harm is CI-separated and mechanism-verified, and no
rescue re-initialization was performed or is proposed.

## Artifacts

* Protocol: `tracks/ksvd/protocols/zinc-cssd-relation-projection-v1.yaml` (frozen 404b7a5)
* Code: `tracks/ksvd/experiments/luyin16/zinc_cssd_relation_projection_v1.py`,
  `tracks/ksvd/src/ksvd_research/runners/zinc_cssd_relation_projection_v1.py`,
  `tracks/ksvd/configs/luyin16/zinc_cssd_relation_projection_v1.yaml`
* Tests: `tracks/ksvd/tests/test_zinc_cssd_relation_projection_v1.py` (14 passed)
* Results: `source_manifest.json`, `pretrain_checks.json`, `smoke.json` (in the smoke run
  record), `terminal_eval.json`, `route_shuffle_plan.npz`, `runs/{BASE,SHARED_MLP,BUCKET_LINEAR}_s0/`
* Run records: 20261008-120123-214b7394 (source), 20261008-120143-294331af (pretrain),
  20261008-104237-c97913db (smoke), 20261008-104553-2713167e (C train),
  20261008-110529-f448bce2 (R train), 20261008-112412-24ab5788 (BASE train),
  20261008-114950-f26cbe0d (terminal)
* Reproduce: commit 404b7a5; `rr deploy res-2 && rr run res-2 <exp> -- uv run research run
  zinc_cssd_relation_projection_v1 --set model.stage=train --set model.arm=<ARM> --set
  model.seed=0 --set runtime.device=cuda:0 --mode scratch` (ARM in {BASE, SHARED_MLP,
  BUCKET_LINEAR}); terminal via `--set model.stage=terminal-eval`.
