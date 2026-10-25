# SCM v1 — pre-registered protocol (frozen before any GPU run)

**Round:** `zinc_structural_conditional_moments_v1`
**Question.** Does letting the explicit graph structure define the conditional
statistics of the local bond content — and letting a content-conditioned
interaction learn only the extra covariance relative to those statistics, while
*writing the correction into the existing physical-bond contribution* — improve
raw-y MAE over the frozen DCC-L, more than an independent attention branch would?

**One sentence of what is new.** DCC-L is unchanged except that `c_loc` becomes
`c_loc + delta`, where `delta` is built from two fixed line-graph distance bands:
a centred structural difference `S` and a local-vs-global score/content
covariance difference `A`; the existing reader then consumes `sum(c_loc')` and
`sum(c_loc'^2)` exactly as before.

**Data.** official PyG ZINC `subset=True`, train 10,000 / valid 1,000, raw-y L1.
Official test is never constructed or read. Valid is an exposed development
split, not a blind test.

**Frozen constants** (`configs/scm_v1.json`, `code/scm_common.py`)
| item | value |
|---|---|
| bands (line-graph distance, self excluded) | `t1 = [1,2]`, `t2 = [3,8]` |
| shared low-rank width | 8 |
| per-band global scale | mean `n_t` per bond over the Fit10k raw graphs = 6.303114, 14.815448 (label-free, identical in both arms, no per-query `n_t` normalisation) |
| new module seed | `seed + 50731`, isolated RNG stream |
| new parameters | Q/K/V `96->8` with bias (2,328) + `Wout 32->96` (3,168) = 5,496 |
| deployment total | 499,689 (cap 500,000) |

**Arms (seed 0 only).**
* `S-only` — `A` is exactly zero in the training forward; `Q`/`K` are registered
  but not evaluated (disclosed: not strictly capacity-matched in effective use).
* `S+A` — `S` and `A` both active.
Both arms start from one shared initial state: the historical DCC-L seed-0
backbone (verified tensor-by-tensor against `f5c34c28…`) plus the same SCM
tensors. `Wout` weight and bias are zero-initialised.

**Recipe (identical to the historical DCC-L seed 0).** `ad_train.train`, recipe
variant B: AdamW, lr 1e-3 -> 1e-5 per-step cosine, 500-step warm-up, wd 1e-5,
clip 5, raw-y L1, batch 128, 600 epochs (79 updates/epoch, 47,400 updates),
full valid every epoch, strict best-valid selection (ties -> earlier epoch),
batch order from `randperm(10000, manual_seed(seed+101))`. SSL start
(`pretrain_encoder_s0.pt`, sha `006fabc1…`) — no supervised warm start.

**Comparison.** Frozen historical DCC-L seed-0 full-10k run
(`zinc_dual_channel_composer_v1/runs/L_s0`, best valid 0.078537…); not retrained.
Primary metric: official-valid raw-y MAE over all 1,000 molecules.

**Decision rules (implemented in `code/scm_analyze.py`, committed before the run).**
* `directionally_better` = paired `delta_best_valid < 0` **and** `delta_W20 < 0`
  **and** `delta_late(300-600) < 0` **and** the paired bootstrap CI95 upper bound
  `< 0`.
* material threshold vs same-seed DCC-L = `-0.003` MAE.
* Result 1 = S-only directionally better and A adds no gain.
* Result 2 = S+A better than S-only with CI95 upper `< 0` **and** S+A at most
  `-0.003` vs DCC-L.
* Result 3 = neither arm directionally better.
* Result 4 = S+A worse than S-only (`delta > 0`).
Anything else is labelled `WEAK_OR_MIXED_EXPLORATORY_SIGNAL`.

**Budget.** 2.5 A100·h total occupancy (two GPUs summed), including prep,
timing, training, evaluation and any retry. Profile both arms for 2 full epochs
before the formal runs and require `projected_total × 1.15 <= budget`;
otherwise stop instead of shortening the horizon.

**Not allowed this round.** No distance-range sweep, no rank/layer/reader-width
sweep, no seed 1, no third arm, no supervised DCC-L warm start, no external
labels, no ensemble, no target-derived features, no recursive hidden-state
message passing on the original or line graph, no official-test read, no
subgroup-based epoch selection, no new "hard molecule" partition derived from
baseline errors.

**CPU gate (must pass before GPU).** `analysis/cpu_checks.json` — historical
backbone identity, isolated RNG, Wout=0 function equality, the correction
entering both original sums, non-zero S/A response once the module is live,
two-step gradient flow, algebraic identities (uniform relation, constant score,
constant content, empty relation), no leakage, permutation/batch invariance,
zero-bond and disconnected boundaries, checkpoint round-trip, real-trainer
smoke, float64 reference agreement, and the synthetic 32-ring counterexample.

**Data feasibility (must pass before GPU).** `analysis/data_feasibility.json`.
Structural gates: far band non-empty, relation far from uniform, bands not
redundant. Degeneracy criteria D1–D5 on the real frozen DCC-L bond
representation. The single pre-registered magnitude line for `A` (1 % of the
`c_loc` rms) is **missed by band 2 at 0.88 %**; this failure is kept in the
record and the decision is disclosed as
`PROCEED_TO_GPU_DISCLOSED_GATE_5_MISS` (see that file).
