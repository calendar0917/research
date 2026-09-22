# Preregistration — TCCD-v7: Final Normalized-Moment Closure

Round name: **TCCD-v7** (*Final Normalized-Moment Closure*).
Study: `zinc-context-gap`.
Status: frozen before any formal experiment on 2026-09-22.

This is the **closing round** of the TCCD standalone-predictor route.  It does
not explore architecture.  It answers exactly one question:

> If the normalization / coordinate exposure that TCCD-v5 and TCCD-v6 proved
> to be the true source of the local gain is written **explicitly into the
> representation**, and the prototype vocabulary is then allowed to co-adapt
> end-to-end, does TCCD standalone performance enter a range that justifies
> continuing the line?

Either TCCD standalone gains a clear reason to continue, or the standalone
performance route is closed.  No rescue is authorized.

## 1. Accepted prior results (inputs to this round)

| round | result | value |
|---|---|---|
| TCCD-v2 | official-valid PrototypeREL best / Top-5 soup | `0.287337` / `0.261988` |
| TCCD-v2 | internal PrototypeREL (Gate A, seed 0) best / soup | `0.2862437069416046` / `0.26635152101516724` |
| TCCD-v2 | internal Dense-REL best | `0.410789` |
| TCCD-v3/v3b | stronger frozen local representation | no rescue |
| TCCD-v4 | `C^T S^2 C` two-hop | no practical increment |
| TCCD-v5 | frozen BASE best / soup | `0.28791576623916626` / `0.2783639132976532` |
| TCCD-v5 | frozen POST best / soup | `0.2608269155025482` / `0.2522333264350891` |
| TCCD-v6 | `G_FULL` (BASE − FULL soup) | `0.0261305868625641` |
| TCCD-v6 | RECON-NL soup | `0.252360463` (`rho 0.995135`) |
| TCCD-v6 | RECON-LINFACT soup (nonlinearity gap) | `0.252556831` (gap `0.000196368`) |
| TCCD-v6 | NOVEL-NL soup | `0.254013419` (`rho 0.931877`) |
| TCCD-v6 | Case | **R1** — readout / coordinate conditioning, not new information |

Consequence: TCCD-v7 adds **no new graph computation**.  It only makes the
already-proven conditioning explicit and deterministic.

## 2. Non-negotiable execution discipline

* Formal compute: remote A100 **GPU1 only**.  GPU0 is forbidden.
* **Official ZINC test is never loaded** in any stage, including the final one.
* Local development / Gate 0 / tests may run locally on CPU.
* Official-valid (1000 graphs) is allowed only if Stage B reaches Case P or G.
* Base architecture is the **exact TCCD-v2 PrototypeREL**:
  `x_v^714 -> z_v^64 -> c_v^64`; K = 64; cosine assignment; temperature
  `0.05 + 0.95*sigmoid(a)`; the five frozen relations
  `[R_int, R_b0, R_b1, R_b2, R_geo]`; the TCCD-v2 entropy + balance
  regularizers, optimizer, epochs (240), patience (40), batch (32),
  lr `1e-3`, wd `1e-5`, clip `5.0`, and the canonical Top-5 soup protocol.
* Forbidden in this round: stronger local encoder, new relation, `S^2`/`S^3`,
  pair MLP, PRE/POST pair enumeration, center update, GNN, message passing,
  attention, topology branch, deep reader, K sweep, tau sweep, regularizer
  sweep, seed rescue, official test.

## 3. Explicit normalized representation (the only change)

For a graph with `n` occurrences and frozen/learned assignment matrix
`C in R^{n x 64}`:

```text
mu   = (1/n) * sum_i c_i                 in R^64      (prototype mean)
m2   = (1/n) * sum_i c_i^{o2}            in R^64      (prototype second moment)
v    = max(m2 - mu^{o2}, 0)              in R^64      (prototype variance)
M_r  = C^T R_r C                         in R^{64x64}
s_r  = sum_{ij} R_r(i,j)                              (relation mass)
Mhat_r = M_r / max(s_r, eps),  and Mhat_r = 0 exactly when s_r = 0
```

`vec_sym` is the exact TCCD-v0/v2 symmetric upper-triangle vectorization
(`np.triu_indices(64)`, diagonal included, 2080 entries per relation).  Relation
semantics are unchanged.

* **NORM+MOM (primary candidate)**

```text
h_G^NM = [ mu, v, {vec_sym(Mhat_r)}_{r=1..5}, log(1+n), {log(1+s_r)}_{r=1..5} ]
dim(h_G^NM) = 64 + 64 + 5*2080 + 1 + 5 = 10534
```

* **NORM (frozen ablation only, Stage A only)**

```text
h_G^N = [ mu, {vec_sym(Mhat_r)}_{r=1..5}, log(1+n), {log(1+s_r)}_{r=1..5} ]
dim(h_G^N) = 64 + 5*2080 + 1 + 5 = 10470
```

`m2` is deliberately **not** concatenated next to `mu` and `v` (deterministic
redundancy).  The primary architecture is fixed to NORM+MOM and is **not**
changed after seeing any result.  NORM exists only to answer whether the unary
second moment adds an increment.

* **RAW (Stage-A reference)**: the existing TCCD-v2 representation
  `h_G^RAW = [sum_i c_i, {vec_sym(C^T R_r C)}]`, dim 10464.

* **Reader** for every Stage-A/B candidate:

```text
Linear(h_G, 1)
```

Linear reader only.  No MLP, no ReLU, no standardization layer, no extra
preprocessing.  The scientific object of the round is the representation.

## 4. Gate 0 (must PASS completely; otherwise STOP)

Run on CPU locally and on GPU1 remotely.

* **A. permutation invariance** — after a random node relabel (C rows and the
  relation matrices relabelled jointly), `mu`, `v`, `s_r`, `Mhat_r` and the
  final representation are invariant (tolerance `1e-5` on float32, relative to
  representation scale).
* **B. batching invariance** — a graph's representation is identical whether
  computed alone or inside a padded batch; padded rows contribute exactly zero.
* **C. algebra** — `sum_k mu_k = 1` within `1e-6`; `v = m2 - mu^{o2}` within
  `1e-6` (including the clamp path).
* **D. normalized composition mass** — for every graph and relation with
  `s_r > 0`, the **full reconstructed symmetric matrix** of `Mhat_r` sums to 1
  within `1e-3` relative.  The storage is the upper triangle only, so the test
  must rebuild `Mhat_r[k,l] + Mhat_r[l,k]` for `k != l`.  The same check is run
  on the real 10,000-graph cache, where the numerator comes from the frozen
  `h_base` and the denominator `s_r` comes from the frozen pair cache; this also
  verifies that the two independent sources of `M_r` and `R_r` agree.
* **E. zero-mass relation** — when `s_r = 0`, `Mhat_r` is finite and exactly
  zero.  Real-cache counts: relation 2 has 127 zero-mass graphs, relation 3 has
  9366 zero-mass graphs, so this path is exercised on real data.
* **F. no target leakage** — every feature depends only on `C`, the observed
  relation matrices `R_r`, and `n`.  Features are identical for two different
  label vectors; the Stage-A builder never loads `pair_*_y.npy`.
* **G. official test blocked** — `official_test_loaded == false` is asserted in
  every artifact; the runner has no code path that opens the official test.

Additional Gate-0 equivalence check (registered): the Stage-A feature builder
(which derives `M_r` from the frozen `h_base` and `s_r` from the frozen pair
cache) and the Stage-B tensor feature builder (which receives `C3` and `R_pad`
directly) must agree to `1e-4` absolute on a real-cache subset with the exact
frozen assignments, for both NORM and NORM+MOM.

FAIL of any Gate-0 check → STOP immediately.

## 5. Stage A — frozen representation closure

Frozen TCCD-v2 PrototypeREL checkpoint (`prototype_rel_seed0_best.pt`,
SHA-256 `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42`),
local encoder / prototypes / temperature frozen, exact internal split
`internal_split(10000)` with `SPLIT_SEED = 20260922` → 8000 train / 2000 dev.
Only the linear reader is trained, with the exact TCCD-v5 frozen-reader
protocol (`train_frozen_base_reader`: seed 0, Adam lr `1e-3`, wd `1e-5`,
clip `5.0`, batch 32, 240 epochs, patience 40, Top-5 soup).

Arms:

* **A0 = RAW** — the frozen v5/v6 reference.  Primary reference value
  `MAE_RAW = 0.2783639132976532` (Top-5 soup) taken from
  `tracks/ksvd/results/tccd_v5/stageA_seed0.json`
  (SHA-256 `2ab232c74219529ef3ddbdf1c6175d664756eb0dc7c0a0a30325d62a94b8aefe`,
  arm `BASE`, feature_dim 10464).  In addition, the exact RAW reader is
  **re-run in the same session** (`RAW_rescreen`) as a protocol-identity check.
* **A1 = NORM** — `h_G^N`, 10470 dims.
* **A2 = NORM+MOM** — `h_G^NM`, 10534 dims (primary candidate).

Primary metric for every arm: **Top-5 soup dev MAE**.  Best-checkpoint MAE is
reported as secondary evidence only.

Registered deltas:

```text
delta_norm      = MAE_RAW          - MAE_NM          (primary)
delta_norm_rerun= MAE_RAW_rescreen - MAE_NM          (conservatism check)
delta_mom       = MAE_NORM         - MAE_NM          (variance diagnostic)
protocol_drift  = |MAE_RAW_rescreen - MAE_RAW|
```

### Stage A decision (frozen)

```text
PASS  iff  MAE_NM <= 0.255
      and  delta_norm       >= 0.015
      and  delta_norm_rerun >= 0.015
      and  protocol_drift   <= 0.005
FAIL  otherwise
```

If `protocol_drift > 0.005` the frozen reference is not protocol-identical →
**PROTOCOL_DRIFT = FAIL** (STOP).  This rule is stated before the run so that
neither the frozen nor the re-run reference can be chosen after the fact to
inflate the gain.

A FAIL stops the round: **no end-to-end run, no official-valid run**.

## 6. Variance diagnostic (interpretation only)

```text
delta_mom = MAE_NORM - MAE_NM
material   iff delta_mom >= 0.005   -> prototype variance has an independent increment
negligible iff delta_mom <  0.005   -> the scientific attribution is normalization /
                                       scale exposure, not the second moment
```

Report only.  The Stage-B architecture stays NORM+MOM either way.

## 7. Stage B — one final end-to-end TCCD run (only if Stage A PASS)

From scratch, seed 0, on the same internal split (8000 train / 2000 dev):

```text
x_v -> z_v -> c_v -> h_G^NM -> Linear -> y
```

Trainable: local encoder `W`, prototypes `P`, temperature logit, linear reader.
Exact TCCD-v2 regularization (`lambda_local`, `lambda_balance` calibrated once
on the first 32 train graphs), exact TCCD-v2 optimizer / schedule / soup
protocol, reusing `tccd_v2.train_model` verbatim.  The frozen v2 assignments
are not used; no module is added; no new loss is added.

Matched reference (TCCD-v2 internal, Gate A, seed 0, same split and protocol):

```text
best 0.2862437069416046 / soup 0.26635152101516724
```

No additional baseline is re-run.  If provenance cannot be verified, exactly
one `exact RAW-E2E seed0` reference run is allowed; nothing else.

### Stage B hard decision (frozen)

```text
MAE_soup > 0.23            -> Case S : STOP standalone (hard)
0.20 < MAE_soup <= 0.23    -> Case P : PARTIAL
MAE_soup <= 0.20           -> Case G : STRONG GO
```

* **Case S**: the TCCD prototype-moment representation is not viable as a
  standalone high-performance predictor.  No seed 1, no tuning, no K sweep,
  no reader upgrade, no new moments, no structural rescue.
* **Case P**: one single full official-train → official-valid run is allowed.
* **Case G**: one single full official-train → official-valid run is allowed.

## 8. Official-valid run (only Case P or G)

Same architecture, same seed, official train 10000 → official valid 1000,
canonical Top-5 soup, GPU1.  Official test remains blocked.

```text
delta_abs = MAE_official_soup - 0.119818
```

Interpretation (frozen):

```text
MAE <= 0.15              competitive-ish: standalone route stays worth studying
0.15 < MAE <= 0.20       substantial but incomplete
MAE > 0.20               insufficient: standalone performance optimization CLOSED
```

No rescue after any of these bands.

## 9. Vocabulary health (reported after Stage B)

`active_prototypes`, `dead_prototypes`, `effective_prototype_count`,
`top1_usage_mass`, `top8_usage_mass`, `mean_local_assignment_entropy`,
`global_usage_entropy` (+ normalized), learned temperature, and semantic
coherence on the internal dev split (top-50 canonical-key concentration vs a
matched random baseline).  If performance improves but the vocabulary
collapses, a reusable-vocabulary success may not be claimed.

## 10. Claims this round may support (frozen wording)

If the round ends in STOP:

> Task-learned prototype vocabularies and assignment-sensitive composition are
> mechanistically supported, and explicit normalization materially improves
> decoding.  However, a representation based only on global prototype moments
> and normalized relation contractions remains insufficient to match strong
> molecular predictors on ZINC.

Explicitly **not** allowed: "prototype learning failed".  The correct boundary is

```text
prototype vocabulary works;
prototype-moment standalone predictor is insufficient.
```

If STOP, the only future use of prototypes authorized by this round is

> prototype vocabulary as an interpretable / reusable local coordinate system
> inside a stronger structural backbone

with a matched control (same backbone without prototypes, same backbone with
prototypes, prototype shuffle / zero intervention, vocabulary health).  No
hybrid may be implemented in this round.

## 11. Durable records produced by this round

* this preregistration;
* `results/tccd_v7/gate0.json` + test results;
* `results/tccd_v7/stageA_seed0.json` + `stageA_decision.json`;
* `results/tccd_v7/stageB_seed0.json` (if authorized);
* `results/tccd_v7/official_seed0.json` (if authorized);
* `notes/tccd_v7_final_normalized_moment_analysis.md`;
* claim / decision records;
* `STATE.yaml` update, including the explicit line
  `TCCD standalone performance route CLOSED.` and
  `No TCCD-v8 rescue is authorized.` if Case S / insufficient is reached.
