# P2 — Compact-v4 One-Shot Relation Refresh

> **Question.** The final compact-v4 graph representation mixes two computation
> stages: the unary branch uses the contextualized centre state `h^(1)` while the
> final pair branch pools the *stale* pre-context relation state `q^(0)`. If,
> after centre contextualization, the **same shared** relation encoder recomputes
> the relation from the updated endpoints — `q^(1) = Q(P(h^(1)), r)` — and only
> the final pair graph summary uses `q^(1)`, does compact-v4 improve?
>
> **Verdict.** **Decision Case B — ONE-SHOT RELATION REFRESH NO-GO** under the
> locked protocol. Stage A (zero-training, label-free) showed the refresh is
> **not** numerically negligible (median relation cosine `0.9827`, median graph
> pair-summary drift `0.1988`, mean `|Δprediction|` `0.0670`) and authorized
> exactly one seed0 run. P2 seed0 reached official-valid MAE **0.144457**
> (best epoch 156) vs the reused optimized compact-v4 seed0 reference
> **0.146420**, i.e. `Δ_P2,0 = +0.001963` — a **sub-threshold positive** strictly
> below the pre-registered `+0.004` gate. The refresh branch is genuinely alive
> and trained (refresh-off inference degrades valid to `0.1751`; `q1` path
> gradient non-zero), it adds **exactly zero parameters** (99,613 = 99,613), and
> the common-input bulk is safe. Per the budget rule seed1 was **not purchased**.
> Official test was **never loaded**.
>
> Code: `tracks/ksvd/experiments/luyin16/zinc_compact_v4_relation_refresh.py`.
> Tests: `tracks/ksvd/tests/test_compact_v4_relation_refresh.py` (24 pass).
> Results: `tracks/ksvd/results/compact_v4_relation_refresh/`
> (+ `answers_q1_q20.json`).

---

## 1. Motivation

P1 closed the minimal *learned pre-pool composition* hypothesis NO-GO
(`Δ_P1,0 = -0.000809`). P2 deliberately does **not** reopen that form. Instead it
isolates a different internal assumption of compact-v4: its final readout reads
the unary state **after** the one-shot centre update but reads the pair state
**before** it. That is a computation-**ordering** inconsistency, not a capacity
question, and it is the exact opposite of the P1 direction (P1 changed *how* the
relation is composed before the update; P2 leaves composition fixed and changes
which relation stage the final summary reads). P2 was explicitly pre-registered
as a separate study and P1's failure does **not** auto-trigger it.

## 2. Why P1 was closed

P1 added ~3% parameters (2,988) to replace the fixed `[mean;std;log-count]`
compression with a learned residual composer and did not clear `+0.004`. The
matched capacity control and seed1 were not purchased. The verdict was a clean
scientific NO-GO with a live, trained but unhelpful branch. Anything that merely
enlarges or re-parameterizes the same composition site is closed.

## 3. Relation-state staleness hypothesis

Compact-v4 computes the pair relation `q_ij` **once**, from the pre-update patch
states `h^(0)`. It then uses that same `q^(0)` twice: (i) to contextualize the
centres (the `A_i` moments that produce `h^(1)`), and (ii) as the final graph
pair summary. After the update the endpoints are `h^(1) ≠ h^(0)`, so the final
relation summary describes endpoints that no longer exist. P2's hypothesis is the
narrowest possible one about this:

> the final pair relation should live at the **same contextual stage** as the
> final unary state.

This is **not** the claim that multi-round message passing is better, and it is
**not** the claim that staleness is the unique bottleneck.

## 4. Exact baseline computation ordering

Verified from the real code
(`zinc_patch_path_pooling.PatchPathModel.encode`):

```
h^(0)_i   = patch_encoder(...)                                  in R^48
u^(0)_i   = P(h^(0)_i) = pair_projection(h^(0)_i)               in R^16
r_ij      = relation_encoder(pair_relation)                     in R^16
gate_ij   = 1 + tanh(distance_gate(pair_bucket))
q^(0)_ij  = Q([u_i+u_j ; |u_i-u_j| ; (u_i*u_j)*gate ; r_ij])    in R^16    (computed ONCE)
A_i       = per-(centre,bucket) [mean(q0) 16 ; pop_std(q0) 16 ; log1p(count) 1] x 5 = 165D
h^(1)_i   = h^(0)_i + center_update([h^(0)_i ; A_i])            in R^48
R_base    = [unary_moments(h^(1)) 97 ; pair_moments(q^(0)) 165 ; global 32 ; topology 8] = 302D
```

The final pair block (`165D`) is the `DISTANCE_BUCKETS x [sum ; sum_sq ;
log1p(count)]` moment pooling of `q^(0)`, implemented by
`PatchPathModel._pool_pairs`.

## 5. P2 computation ordering

P2 keeps the baseline computation exactly through the centre update, then adds
**one** parameter-shared evaluation of the same `P` and `Q` on `h^(1)`:

```
h^(0) -> q^(0) -> A -> h^(1)                (unchanged; q^(0) still drives A)
u^(1)_i   = P(h^(1)_i)                       (SAME pair_projection tensor)
q^(1)_ij  = Q([u_i+u_j ; |u_i-u_j| ; (u_i*u_j)*gate ; r_ij])   (SAME pair_encoder, gate, r_ij)
R_P2      = [unary_moments(h^(1)) 97 ; pair_moments(q^(1)) 165 ; global 32 ; topology 8] = 302D
```

`q^(1)` never feeds a centre update, so there is no `h^(2)`. The refresh count is
exactly **1**. There is no `alpha`, no `[q0;q1]` concat, and no independent
refresh encoder. `q^final = q^1` exactly.

## 6. Why this is not generic iterative message passing

The structure stays: patch state, explicit pair relation, centre context,
refreshed pair relation, graph summary. P2 is *not* a stack of propagation
rounds — it is a single re-evaluation of one explicit relation object after its
endpoints changed. No unbounded depth, no learned per-round operators, no
hidden-state propagation. It is **explicit relation consistency**: keep the
relation object explicit and ask it once more after the state it refers to has
changed.

## 7. Shared parameter design

The refresh reuses the *same tensors*: `pair_projection` (a single
`Linear(48,16, bias=False)`), `pair_encoder` (a single `_MLPBlock`), the same
`relation_encoder`, and the same `distance_gate`. There is no
`pair_projection_refresh` and no `pair_encoder_refresh`. The model subclass
overrides `encode` to compose the existing modules; it introduces no new
trainable tensor.

## 8. Parameter neutrality

From the instantiated model (`parameter_audit.json`):

| model | params |
|---|---:|
| compact-v4 baseline | 99,613 |
| **P2** | **99,613** |
| **Δ** | **0** |

`initialization_match.json`: P2 built from scratch under seed 0 reproduces
**every** shared baseline tensor exactly — 48 shared tensors, `max_abs_diff =
0.0`, no extra/missing keys, identical state hash `d2650caf…` for both. This is
from-scratch initialization matching, **not** warm-starting from the trained v4
checkpoint.

## 9. Stage A zero-training staleness audit

`staleness_audit_lock.json` fixes the audit before any number is seen. The audit
uses the **optimized-v4 seed0 selected checkpoint** (sha256 `60b7d297…`,
fingerprint verified) and the **optimized-manifold broad-state screen**
official-train `train_probe` split: **2000 molecules** selected by a
target-independent deterministic molecule-id sha256 hash (split seed
`optimized-manifold-broad-state-screen-v1-20260919`; recomputed assignment hash
matches the frozen manifest exactly, and the recomputed probe indices equal the
frozen list). No labels are read. `staleness_integrity.json` passes all 11 gates
(A0.1–A0.10): checkpoint fingerprint, exact refresh-off ↔ parent prediction
reconstruction (`0.0`), `h0/h1 = 48`, `q0/q1 = 16`, functional reconstruction of
`q1` from the shared tensors (`0.0`), identical pair ordering (`8452 rows` per
batch, `q0 rows == q1 rows`), pair summary implementation identity, and no extra
state keys.

## 10. Relation-level drift

`relation_drift.json` (528,582 relations over the 2000-molecule probe):

| metric | mean | median | p90 | p95 |
|---|---:|---:|---:|---:|
| normalized L2 `||q1-q0||/(||q0||+eps)` | 261010.3* | **0.1820** | 1.0000 | 2.0852 |
| cosine `cos(q0,q1)` | 0.7200 | **0.9827** | — | — |

D2 report (p10 / p05): `0.0 / 0.0`.

\* The raw mean is corrupted by degenerate rows: the shared pair encoder ends in
`ReLU`, so **21.7%** of `q0` rows are exactly zero (P1's independent note already
reported 86–97% zero *entries*). For those the spec denominator `||q0||+eps` is
near `eps`. Excluding degenerate `q0`, the normalized-L2 mean is **0.8301** and
the cosine mean is **0.9624**. The **median is the robust summary** and gate
selection uses it. Descriptive per-bucket cosine means: `0.8449 / 0.7452 /
0.7224 / 0.6587 / 0.7047` (buckets 0–4, i.e. distances 1,2,3,4,5+).

## 11. Graph-summary drift

`pair_summary_drift.json` (2000 graphs). The compare uses
`PatchPathModel._pool_pairs` **verbatim**.

| metric | mean | median | p90 |
|---|---:|---:|---:|
| `||S1-S0||/(||S0||+eps)` | 0.2099 | **0.1988** | 0.3021 |

Per-bucket descriptive pair-summary drift means: `0.1781 / 0.1770 / 0.1709 /
0.1961 / 0.2170`. Mechanical prediction shift `prediction_shift.json`:
mean **0.0670**, median 0.0453, p90 0.1340, max 1.9917 (true `y` not used).

## 12. Stage A decision

`stageA_decision.json`:

| gate | threshold | observed | pass? |
|---|---:|---:|:--:|
| median cosine >= | 0.995 | 0.9827 | ✗ |
| median pair-summary drift <= | 0.02 | 0.1988 | ✗ |
| mean prediction shift <= | 0.005 | 0.0670 | ✗ |

None of the three near-identity conditions holds → **not near-identity** →
**AUTHORIZE P2 SEED0**. This only buys one seed0 run; it is **not** evidence that
refresh improves MAE.

## 13. Architecture lock if authorized

`architecture_lock.json` (written before training, unchanged afterwards) fixes:
the refresh definition and ordering, the shared-tensor policy, `q_dim = 16`,
5 buckets, `refresh_count = 1`, no `h^(2)`, no mixing / no concat, `R = 302D`, the
unchanged centre pooling / topology branch / graph head, the parameter target
99,613 with `Δ = 0`, the training protocol, the `+0.004` architecture gate, the
`+0.002` bulk gate, and the seed policy (`[0]` then conditional `[1]`; 2/3
forbidden).

## 14. Initialization matching

`initialization_match.json`: `all_99613_params_identical = true`,
`max_abs_diff_initial = 0.0`, no extra/missing keys, `shared_state_sha256 ==
baseline_state_sha256 = d2650caf…`. Because P2 adds no tensor, all 99,613
initial values match the seed0 baseline exactly. At initialization `h^(1) = h^(0)`
(the centre update's final layer is zero-initialised), so the P2 forward is
bit-identical to the baseline before training — the architecture difference
emerges only as the centre update learns.

## 15. P2 seed0

One run, frozen protocol. Curve `curves/p2_seed0_curve.csv`.

| run | valid MAE | best epoch | epochs run | steps | params | wall |
|---|---:|---:|---:|---:|---:|---:|
| optimized compact-v4 seed0 (reference, reused) | **0.146420** | 169 | 209 | 16,511 | 99,613 | — |
| **P2 seed0** | **0.144457** | 156 | 196 | 15,484 | 99,613 | 1382 s |

Per-epoch diagnostics (recorded, do not change training RNG): `q0_norm`,
`q1_norm`, `q_cosine`, `pair_summary_drift`, `pair_encoder_grad_norm`,
`pair_projection_grad_norm`, `q0_grad_norm`, `q1_grad_norm`,
`checkpoint_selected`.

## 16. Architecture gate

`Δ_P2,0 = 0.146420 - 0.144457 = +0.001963`. Required `>= +0.004`. **FAIL** —
positive but sub-threshold. Per the mandate a sub-threshold positive (even
`+0.003`) still STOPs. No optimization pathology: no NaN; best epoch 156/196
(not boundary-pinned); early stop fired; training loss decreasing; pair-encoder
gradient and `q1`-path gradient non-zero. This is a clean scientific NO-GO, **not**
optimization ambiguity.

## 17. Bulk safety

`common_input_bulk.json` (target-independent: valid molecules whose train-derived
`rare_le5_ratio` is below the train 80th percentile; 756 of 1000 valid):
v4 **0.09472** vs P2 **0.09373**, diff **-0.00099** `<= +0.002` → **safe** (P2 is
marginally *better* on the easy bulk). No error-quintile / target-tail mining was
used.

## 18. Refresh dependency

`mechanism_diagnostics.json`. On the official-valid split, toggling the refresh
off on the *same trained checkpoint* degrades valid MAE from **0.144457** to
**0.175124**; mean `|Δprediction|` **0.0932**, max **1.6828**. This confirms the
refresh computation genuinely enters the prediction. It is a **branch-dependency
check only**: the head/backbone were trained for `q^(1)`, so the degradation does
**not** causally isolate staleness. Training-time diagnostics also show the
`q1` path is live: `q1_grad_norm` non-zero (max 0.0418) and `pair_encoder`
gradient non-zero (max 5.70).

## 19. Seed1 conditional replication

**Not run.** Seed1 is conditional on the seed0 architecture gate (`>= +0.004`),
which failed. No `stageC_*` artifacts and no `final_two_seed_summary.csv` were
created. Seeds 2/3 were forbidden throughout.

## 20. Compute overhead

`compute_audit.json`: a 128-molecule valid batch has 2,998 centres and 34,876
pairs (mean **272.5 pairs/molecule**). Forward wall-clock rose from **18.48 ms**
(baseline) to **29.81 ms** (P2), i.e. **+61.3%**, at **+0%** parameters. Peak RSS
for the audit process ≈ 2,026 MB. This is the explicit reminder that
"parameter-neutral" is **not** "compute-neutral": the refresh re-evaluates the
pair projection and pair encoder once per relation. (The baseline `q^(0)` pair
readout is also computed before being replaced, so the measured overhead is an
upper bound; a production implementation would skip the discarded `q^(0)`
readout.)

## 21. What is and is not proven

**Proven (seed0, locked protocol):**

* A one-shot, parameter-shared relation refresh is implementable with **zero**
  added parameters, is deterministic, and leaves the baseline pathway
  bit-identical when disabled.
* The pre-context vs post-context relation states are **not** numerically
  interchangeable: `median cos(q0,q1) = 0.9827`, median graph pair-summary drift
  `0.1988`, mean `|Δprediction|` `0.0670` on a label-free probe.
* Under the frozen optimized protocol the refresh does **not** clear the `+0.004`
  architecture gate: `Δ_P2,0 = +0.001963` (positive but sub-threshold).

**Not proven:**

* That relation staleness is harmless, or that it is the unique causal
  bottleneck. Stage A is a magnitude audit, not a performance test.
* Anything about mechanism isolation — refresh-off is a dependency check only.
* Any official-test number — test was never loaded.

## 22. Final verdict

**Decision Case B — ONE-SHOT RELATION REFRESH NO-GO.** `Δ_P2,0 = +0.001963 <
+0.004`, branch alive, parameter-neutral, bulk safe, no optimization ambiguity.
Only **one** new full training run was spent; seed1 was **not purchased**.
`results/compact_v4_relation_refresh/final_decision.json`.

## 23. Implication for future architecture work

This is the third consecutive clean nested NO-GO on local repairs of the
existing compact-v4 computation graph (P1 learned centre composer,
compact-v4-cell cycle object, P2 one-shot relation refresh). Each isolated a
single local assumption, changed it minimally with a pre-registered gate, and
failed to clear `+0.004`. Together they weaken the hypothesis that compact-v4 is
merely missing one small local computation-ordering fix. The next move should be
**paradigm-level**: broader representation-family / parameter-allocation
redesign (e.g. re-allocating the large identity-lookup embedding budget, or a
different readout/computation family) rather than another local patch. If a
future refresh-style idea is proposed, it must be its own pre-registered
hypothesis and must address compute cost (the measured `+61%` forward overhead at
zero parameter change is a real deployment concern). Do **not** stack P1 + P2 +
cells + attention at once, and official test stays locked.

---

## Q1–Q20

| # | answer |
|---|---|
| Q1 | `h^(0)=patch_encoder(...)`; `u^(0)=P(h^(0))`; `q^(0)=Q([u_i+u_j;\|u_i-u_j\|;(u_i*u_j)*gate;r])` computed once; `A` = per-(centre,bucket) `[mean(q0)16;pop_std(q0)16;log1p(count)1]x5 = 165D`; `h^(1)=h^(0)+center_update([h^(0);A])`; `R_base=[unary(h1)97;pair_moments(q0)165;global32;topology8]=302D`. |
| Q2 | `q` dim = **16** (`pair_hidden`). |
| Q3 | `h^(1) = h^(0) + center_update([h^(0) ; A])`, where `A` pools `q^(0)` per (centre,bucket). |
| Q4 | `q^(0)` is computed **before** centre contextualization; by the time the final pair summary is pooled, the endpoints are `h^(1)`, so `q^(0)` describes endpoints that no longer exist — a stale pre-context relation. |
| Q5 | Stage A `q0->q1` **median** normalized L2 = **0.1820** (raw mean dominated by 21.7% exactly-zero `q0` rows; robust mean excluding degenerate = 0.8301). |
| Q6 | **median** cosine = **0.9827** (mean 0.7200; excluding degenerate 0.9624). |
| Q7 | graph pair-summary **median** drift = **0.1988** (mean 0.2099, p90 0.3021). |
| Q8 | mechanical prediction **mean** shift = **0.0670** (median 0.0453, p90 0.1340, max 1.9917). |
| Q9 | **No** — none of the three near-identity conditions holds. |
| Q10 | **Yes** — P2 seed0 authorized (and run). |
| Q11 | **Yes** — P2 total params = **99,613** (Δ = 0). |
| Q12 | best valid = **0.144457**, best epoch = **156** (196 epochs, 15,484 steps, 1382 s). |
| Q13 | `Δ_P2,0 = +0.001963`. |
| Q14 | **No** (`+0.004` gate not passed). |
| Q15 | **Yes** — bulk safe (v4 0.09472 vs P2 0.09373, diff -0.00099). |
| Q16 | **Yes** — refresh path alive (refresh-off valid 0.1751; `q1`-path gradient non-zero). |
| Q17 | **+61.3%** forward wall-clock (18.48 → 29.81 ms) at 0 added params. |
| Q18 | **No** — seed1 not authorized. |
| Q19 | N/A (seed1 not run). |
| Q20 | **Case B — ONE-SHOT RELATION REFRESH NO-GO (sub-threshold positive).** |
