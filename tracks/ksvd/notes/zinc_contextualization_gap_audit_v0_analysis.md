# CGA-v0 — Contextualization Gap Audit — analysis

Protocol: `zinc_contextualization_gap_audit_v0`
Pre-registration: `notes/zinc_contextualization_gap_audit_v0_preregistration.md`
(Amendment A1 included there, frozen before the formal run).
Formal run: commit `c174ab36` (code) / remote A100 host `res`, deterministic
**CPU** inference, official ZINC train 10000 / official valid 1000.
Code: `experiments/luyin16/zinc_contextualization_gap_audit.py` (13 tests pass).
Results: `results/zinc_contextualization_gap_audit_v0/`.
**New full training runs: 0. Official test: never loaded.**

---

## 1. What was reused

| role | checkpoints | seeds | parameters | training revision |
|---|---|---|---|---|
| primary recurrent | `recurrent_seed{0,1}_selection_state.pt` | 0, 1 | 82,115 | `a3515e76` |
| matched one-shot baseline | `baseline_seed{0,1}_selection_state.pt` | 0, 1 | 82,115 | `a3515e76` |
| auxiliary B-Null | `lt_null_seed0_selection_state.pt` | 0 | 49,343 | local-token-null round |

`checkpoint_inventory.json` records the SHA-256 of each state file. The primary
comparison is the matched compact-v4 small-head family; DTX / SRDA / SDPK
numbers are not used as causal comparators.

## 2. Integrity gates (`export_integrity.json`)

| gate | seed0 | seed1 |
|---|---|---|
| **A** valid MAE vs recorded | 0.13837561 vs 0.13837560, `|Δ|=5.2e-9` | 0.13343998 vs 0.13343998, `|Δ|=2.4e-8` |
| baseline valid MAE vs recorded | 0.14533377 vs 0.14533376, `|Δ|=4.1e-9` | reproduced |
| **B** ordering (valid targets vs record) | max `|Δ|=0.0` | max `|Δ|=0.0` |
| **B** single-molecule vs batched `h0` | max `|Δ|=0.0` | max `|Δ|=0.0` |
| **C** `q0` reconstruction | max `|Δ|=0.0` | 0.0 |
| **C** `A0` reconstruction from `q0` | max `|Δ|=0.0` | 0.0 |
| **D** repeat-export `h0/h1/h2/A0/A1/yhat` | all bit-identical (0.0) | all 0.0 |
| `official_test_loaded` | false | false |

All primary gates pass. The exports are exact reproductions of the durable
checkpoint behaviour; no new forward computation was invented.

## 3. Audit A — local-neighbour contextual divergence

Train patches (231,664) are the database; valid patches (23,083) are queries;
k = 16. **Important discovered structure:** the h0 space is heavily duplicated —
72.3 % of valid patches have an *exact* h0 match in train (median 77 exact
duplicates). The preregistered `Jaccard(NN_h0, NN_h2)` is therefore
non-diagnostic (two draws from the same duplicate group overlap little), and
`Amp = d2/(d0+eps)` is degenerate at `d0 = 0`. Amendment A1 replaced these in
the decision with the matched-random retention ratio and added the
exact-local-state subset; both raw quantities are still reported.

| statistic (H0 space) | seed0 | seed1 |
|---|---:|---:|
| median `R_delta = dDelta/(√2 σ_δ)` | 0.182 | 0.104 |
| `retention_dDelta = mean dDelta_NN / mean dDelta_random` | **0.304** | **0.262** |
| `retention_d2 = mean d2_NN / mean d2_random` | 0.155 | 0.150 |
| mean / median `J` (descriptive, tie-confounded) | 0.213 / 0.103 | 0.203 / 0.096 |
| fraction exact h0 match | 0.723 | 0.723 |
| exact-match subset: `dDelta/(√2 σ_δ)` median | 0.176 | 0.082 |
| exact-match subset: `retention_dDelta` | 0.295 | 0.233 |

Raw h0-neighbours have deltas that disagree only ~26–30 % as much as
random-matched patches: the contextual shift is **substantially a function of
the local state**. The same holds for the raw local environment (Space X:
selecting neighbours by the standardized raw descriptor gives `R_delta`
0.20 / 0.12, and 44.3 % of those neighbours still coincide exactly in h0). The
`d0`-bin control shows amplification 1.3–2.1 and `R_delta` 0.18–0.26 on the
non-degenerate bins, i.e. the result is not an artefact of nearest-neighbour
quality.

Contrast with the preregistered divergence thresholds (`R_delta ≥ 0.5` or
retention `dDelta ≥ 0.6`): **neither seed shows substantial contextual
divergence.** (`DIVERGENCE = false` for both.)

## 4. Audit B — can local state predict the contextual refinement?

Target-free (no `y`). Target = `delta_h = h2 − h0`; predictor = mean over the
16 train h0-nearest neighbours.

| quantity | seed0 | seed1 |
|---|---:|---:|
| `MSE_mean` (constant predictor) | 0.02803 | 0.05651 |
| `MSE_local` (h0-16NN) | 0.00207 | 0.00218 |
| `MSE_context` (`[h0;A0]`-16NN) | 0.00104 | 0.00136 |
| **`P_local`** | **0.926** | **0.961** |
| **`P_context`** | **0.963** | **0.976** |
| `context_gain` (relative to residual) | 0.501 | 0.375 |
| **`context_abs_gain`** (fraction of δ variance) | **0.037** | **0.014** |
| `context_share_of_h2_variance` | 0.0085 | 0.0057 |
| optional `[h0;A0;A1]` `context_gain_A1` | 0.477 | 0.319 |

The local state alone explains **93–96 %** of the recurrent shift. Centre
context (`A0`) is genuinely informative — it halves the *residual* — but the
residual is small: context explains only **1.4–3.7 %** of the total shift
variance (0.6–0.9 % of the h2 variance). The recurrent contextual refinement
is therefore mostly a locally determined transformation with a small,
context-input-dependent remainder.

## 5. Audit C — local prototypes and contextual roles

Target-free spherical k-means, `K = 64`, fixed seed, train h0 only.

| quantity | seed0 | seed1 |
|---|---:|---:|
| prototypes used (`n_k ≥ 50`) | 64 | 64 |
| weighted-mean `S_k = trace Cov(δ|k)/trace Cov(δ)` | **0.184** | **0.128** |
| median `S_k` | 0.174 | 0.112 |
| `S_k` range | 0.05–0.55 | 0.01–1.56 |
| within-prototype h0 radius (median) | 3.06 | 3.67 |
| within-prototype δ radius (median) | 0.41 | 0.48 |

82–87 % of the recurrent-shift variance is **between** local prototypes, i.e.
already explained by the local atom; only 13–18 % is within-atom contextual
variance. A tight local environment does not map to a unique contextual state,
but it maps to a narrow distribution: the within-atom δ spread is ~0.4 σ_δ.
No `K` sweep, no prediction use of these features.

## 6. Audit D — is contextualization linked to the recurrent gain?

Per valid molecule: `gain = |y−ŷ_base| − |y−ŷ_rec|`; primary metric
`C_h = mean_i ‖h2_i−h0_i‖`.

| quantity | seed0 | seed1 |
|---|---:|---:|
| mean recurrent gain | +0.006958 | +0.004619 |
| Spearman(`C_h`, gain) | 0.0264 | 0.0649 |
| Spearman 95 % CI (10,000 bootstrap) | [−0.0355, 0.0919] | [0.0005, 0.1297] |
| Pearson(`C_h`, gain) | 0.0347 | — |
| Q1/Q2/Q3/Q4 mean gain | 0.0049 / −0.0035 / 0.0127 / 0.0137 | — |
| `gain_Q4 − gain_Q1` | +0.00877 CI [−0.0119, 0.0322] | +0.00435 |
| Spearman size-residualised | 0.0183 CI [−0.047, 0.080] | 0.0421 CI [−0.024, 0.106] |
| Spearman(`C_h`, size) | −0.349 | −0.332 |

The task link is **at best marginal and seed-unstable**: seed0's CI includes
zero (and its Q4−Q1 CI includes zero); seed1's Spearman CI low is +0.0005,
i.e. barely positive. This is not the consistent, non-zero task link that the
contextual-bottleneck claim needs. (Descriptive `C_h1`, `C_q`, `C_A` are in
`seed{0,1}_task_link.json`; `C_q` mean ≈ 0.031, `C_A` mean ≈ 0.349.)

## 7. Two-seed replication and verdict

Both seeds independently show: no substantial divergence, high `P_local`,
small absolute context gain, and a near-zero / unstable task link. The
preregistered decision rule returns

```text
C_CONTEXTUAL_STATE_LOCALLY_COMPILABLE
```

with `LOCAL_COMPILABLE_ABSOLUTE` true for both seeds and `TASK_LINK` true for
seed1 only (marginal). `DIVERGENCE` is false for both, so Case A cannot fire.
There is no seed conflict on the representation metrics; the only seed
difference is the fragile task link, which is reported rather than pooled.

## 8. Auxiliary B-Null seed0 (`bnull_seed0_auxiliary.json`)

The T=2 recurrent model with the molecule-dependent 16-D local token deleted
reproduces its recorded valid MAE (0.12687264 vs 0.12687270, `|Δ|=5.6e-8`) and
shows the **same structure**: `R_delta` 0.257, `retention_dDelta` 0.397,
77.1 % exact h0 matches, `P_local` 0.864, `P_context` 0.933,
`context_abs_gain` 0.069. Contextual splitting is not compensating for a weak
local token; it is a property of the pair–centre computation itself, and it is
again mostly locally predictable (the context residual is somewhat larger,
6.9 % of δ variance). B-Null stays auxiliary and is not used for the primary
verdict.

## 9. Direct answers to the round's questions

1. **Checkpoints / seeds reused:** recurrent seed0/seed1 and matched baseline
   seed0/seed1 (primary); B-Null seed0 (auxiliary).
2. **New full training runs:** `0`.
3. **Prediction reproduction:** yes — valid MAE reproduced to `|Δ| ≤ 2.4e-8`.
4. **Do h0-neighbours stay h0-neighbours in h2?** Formally the Jaccard is low
   (0.20–0.21), but that is a tie artefact of massive exact h0 duplication
   (72 % of queries). The tie-robust measure shows h0-neighbours stay much
   closer in h2 than matched random patches (d2 ratio 0.15, δ-disagreement
   ratio 0.26–0.30).
5. **Mean / median neighbour retention:** mean J 0.213/0.203, median 0.103/0.096;
   tie-robust `retention_dDelta` 0.304/0.262 and `retention_d2` 0.155/0.150.
6. **delta_h dispersion in tight h0 neighbourhoods:** exact-h0 patches disagree
   by 0.176/0.082 of `√2 σ_δ` (0.30/0.23 of the random-control disagreement).
7. **`P_local`:** 0.926 / 0.961.
8. **`P_context`:** 0.963 / 0.976.
9. **`context_gain`:** 0.501 / 0.375 (relative to residual); absolute
   `context_abs_gain` 0.037 / 0.014.
10. **Same direction across seeds?** Yes on divergence, `P_local`, `P_context`,
    case; the task link is seed-unstable (seed1 marginal, seed0 null).
11. **K=64 within-prototype contextual variation?** Yes but bounded: weighted
    `S_k` 0.184 / 0.128 — 13–18 % of δ variance is within-prototype.
12. **Spearman(C_h, gain):** 0.0264 / 0.0649.
13. **95 % CI:** [−0.0355, 0.0919] / [0.0005, 0.1297].
14. **Q4 − Q1 recurrent gain:** +0.00877 (CI [−0.0119, 0.0322]) / +0.00435.
15. **Size control:** direction preserved but null, 0.018 / 0.042, CIs include 0.
16. **B-Null:** same contextual split (P_local 0.864, context_abs_gain 0.069).
17. **Case:** `C_CONTEXTUAL_STATE_LOCALLY_COMPILABLE`.
18. **Bottleneck?** The data support **contextual states exist and are
    context-input-dependent**, but NOT that *contextual computation is the
    bottleneck*: the recurrent shift is 93–96 % locally predictable, the
    context-only increment is 1–4 % of the shift, and its task link is
    near-zero / seed-unstable.
19. **Local dictionary hypothesis:** a local atom is a reusable local-chemistry
    coordinate that already captures 82–87 % of the contextual-role variance;
    `local atom ≠ complete predictive state` only for the small (13–18 %)
    within-atom residual, which is not task-linked. A sixth local / pair /
    topology-cross dictionary is not the lever.
20. **Next step:** pursue a **bounded strict-static (local-only) compiler** is
    the direction the representation result authorizes, but with a low expected
    ceiling, because the context residual is small and its task link is null;
    the recurrent gain itself is more consistent with local depth. Do not invest
    in another local dictionary / topology cross.
21. **Official test:** never loaded (`official_test_loaded = false` everywhere).

## 10. Limitations

* `h0` is a learned local representation; the audit asks whether the *model's
  own* local state already determines its contextual refinement, which is the
  operationally relevant question, not whether a raw graph invariant does.
* `P_local` is a non-parametric upper bound on a local-only predictor; it does
  not prove that an end-to-end strictly-local model recovers the recurrent
  task performance (that is a separate, budget-matched training question).
* `context_abs_gain` is small but non-zero; the honest reading is "context is
  real but not the bottleneck", not "context is absent".
* B-Null is a single seed and auxiliary only.

## 11. Recommendation (no training this round)

STOP the local-dictionary / pair-dictionary / topology-cross line as a way to
recover the recurrent advantage. The evidence chain local-unpredictability →
context-predictability → task-linked gain fails at the first and third links.
If the strict-static direction is pursued at all, it should be **one** bounded
preregistered test of a strictly-local two-update compiler (per-patch, no
pair→centre context, matched depth and parameters), with a low prior; it should
not be another dictionary, ring feature, or cross.
