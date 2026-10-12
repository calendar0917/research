# cscl-bjoc-cross-bond-interaction-v1 — cross-bond (two-endpoint) interaction channel

**Status:** COMPLETED 2026-10-12. Single formal stage `pair-a-s0` (3 trajectories,
~0.32 GPU-h). Seed 1, the split-A locked read and split B were NOT purchased.
**Decision: `CROSSBOND_INFORMATION_BUT_NO_PREDICTIVE_GAIN`.**
Pre-registration: `configs/bjoc_cross_bond_interaction_v1.yaml` (committed 9f890dd
before any GPU run). Report: `results/bjoc_cross_bond_interaction_v1/REPORT.md`.

## 1. Question

The published BJOC-Organized predictor summarizes the two endpoint environments of a
physical bond by `W_b = Σ_{j∈b} w_j`, `V_b = Σ_{j∈b} w_j²`, `n_b`. Those summaries are
invariant to *which* junction sits on which endpoint. This round asks whether explicitly
preserving the two-endpoint (cross-bond) combination — the left-side mean junction vector
times the right-side mean junction vector — carries information the existing summaries
cannot express, and whether it buys held-out raw-y MAE.

Frozen intervention (one channel, no search):

```
q_b = sqrt(96) * mu_u(b) ⊙ mu_v(b),   mu_side = (Σ_{j on side} w_j) / n_side
```

- arms: N = BJOC-Null, O = BJOC-Organized (published recipe), P = O + q channel.
- every arm 428,865 parameters, identical init hash, identical batch order.
- only difference P vs O: `group_composer` input gains the 96 q columns
  (zero-initialised, so P ≡ O at init, verified exactly).
- recipe frozen: Adam lr 1e-3, wd 1e-5, batch 128, 240 epochs, clip 5, FP32, raw-y L1,
  last-5 soup, no scheduler. Split A salt `cscl-bjoc-crossbond-v1-discovery`.

## 2. Phase A — CPU information audit (hard GPU purchase gate)

**Decision artifact: `phase_a_gate.json` → `PHASE_A_PURCHASE_GRANTED`.**

### 2.1 Projection collision (is q outside the old projection?)

- Scalar witness: counts {1,2} vs {3,4} and {1,3} vs {2,4} both give W=10, V=30, n=4,
  but side products 21 vs 24. A systematic search over small integer assignments found
  **322 collision keys**.
- 96-D construction: an explicit pair of per-dimension partitions collides in (W, V)
  exactly (`W_equal=True`, `V_equal=True`) while **max|Δq| = 7.3485**.

### 2.2 Real molecules (2000-molecule audit, labels never read)

- `valid_pair_fraction = 0.8350` (208,149 / 249,279 bonds have at least one junction on
  each side; `n_u` classes 1/2/3 = 100,945 / 129,285 / 9,108).
- q has variance in **all 96 dims**; per-dim std median 2.17.
- Invariance gates exact: endpoint swap 0.0, junction endpoint swap 0.0, atom relabel
  3.6e-15.
- **Exact identity** (verified to 3.6e-15):

```
q / sqrt(96) = (W² − V) / (2 n_u n_v) − P_net
P_net        = [ (s_u² − Q_u) + (s_v² − Q_v) ] / (2 n_u n_v)
```

where `s` are side sums and `Q` side sums of squares, so `P_net` is the same-side
pair-product correction. `(W²−V)/2` is exactly re-expressible from the old summaries;
`P_net` is **not a function of (W, V, n)** — it is the functional information increment.
Measured on real training molecules: **functional new-variance share 0.3136,
new-L1 share 0.4347**. By `(n_u,n_v)` class: (1,1) 0.000/11,502 bonds; (1,2) 0.313/22,094;
(2,2) 0.247/6,914; (1,3) 1.80/712; (2,3) 0.565/577; (3,3) 0.443/24.

The simple degree/bond-type/ring-flag probe explains only 61% of q
(residual 0.389) — q is *not* determined by simple structure.

### 2.3 Disclosed amendment (before the freeze commit, before any GPU)

The first draft gated on the residual of a **linear** ridge probe from the composer input
(W, V, n, bond features) to q — measured **0.0148**, below the drafted 0.05 threshold.
Running the audit showed that number conflates *same-distribution linear predictability*
with *functional non-expressibility*: `P_net` is statistically predictable from V on this
feature distribution even though no function of (W, V, n) can express it. The exact
identity above is the correct operationalisation of the brief's criterion ("无法仅由 W/V
重建"). The governing non-degeneracy criterion is therefore the functional new-variance
share (threshold 0.10; measured 0.3136); the linear and simple probe numbers remain
reported (`reported_not_gating` in the gate JSON) and do not gate. This amendment was made
after seeing Phase-A outputs but **before the protocol commit and before any GPU job**.

### 2.4 Synthetic known-truth distinguisher

Two systems share the same (W, V, n) projection but differ in the side partition.
Projection-only model test MAE **0.1238**; projection + q **0.1008** (18.6% relative
improvement); the projection-only prediction gap on the collision is 5.54 vs q-model
gap 0.32. All symmetry checks hold.

### 2.5 Correctness gates and parameter matching

All **24 correctness gates pass** (`correctness_gates.json`): endpoint/junction/relabel
invariances, batched composition, side-assignment correctness, W/V/n semantics unchanged
vs published BJOC, N/O/P 428,865 params with equal init hash `b4f1de541f0f8692`, O ≡ P at
init (0.0 gap), new input columns get gradients in P (0.0066 vs 0.0 in O), no
GNN/attention/iteration, raw-y-only loss, official splits never touched, save/load replay
exact.

## 3. GPU stage — results

Single stage `pair-a-s0` (N, O, P; split A, seed 0). Run
`bjoc-cbic-pair-a-s0-20261012-190941-52050b03` on `res` / pool `res-gpu1`, commit
`9f890dd`, A100-SXM4-40GB, torch 2.5.1+cu124. Wall: 362 s / 364 s / 416 s.

| arm | fit MAE | monitor MAE | params | init sha |
|---|---:|---:|---:|---|
| N (Null) | 0.039239 | 0.098266 | 428865 | b4f1de541f0f8692 |
| O (Organized) | 0.038275 | **0.089843** | 428865 | b4f1de541f0f8692 |
| P (O + q) | **0.034149** | 0.091796 | 428865 | b4f1de541f0f8692 |

Paired bootstrap (5000 resamples, molecule unit, monitor, label-blind):

| contrast | Δ MAE (P − ref) | CI95 | frac P better |
|---|---:|---:|---:|
| P − O | **+0.001952** | [−0.003703, +0.007767] | 0.467 |
| P − N | −0.006471 | [−0.016748, +0.001645] | 0.504 |
| O − N (context) | −0.008423 | [−0.017744, −0.001120] | — |

Subgroups (P − O): ordinary y>−3 +0.004350 (n=931); tail y≤−3 −0.030399 (n=69);
top-10-removed +0.001587.

**Frozen monitor purchase rule** (seed-1): P−O ≤ −0.006 AND P−N ≤ −0.004 AND ordinary ≤ 0
AND top-10-removed ≤ 0 AND no new overfit signal. Result: `p_minus_o_le_threshold=False`,
`ordinary_not_worse=False`, `not_single_molecule_driven=False` →
**`CROSSBOND_NO_SEED_REPLICATION_PURCHASE`**. Therefore seed 1, the locked read and split B
were not purchased, and the split-A locked labels were **never read**.

## 4. Mechanism audit (CPU, labels never read, `mechanism_audit.json`)

- q columns are consumed by the trained model: mean |input weight| on the 96 q columns
  0.0202 vs 0.0167 on base columns (max 0.343 vs 0.212).
- zeroing q changes predictions by mean |Δ| 0.2407 (max 13.58); permuting q rows changes
  them by 0.2473 → the channel is active, not bypassed.
- endpoint swap of `bond_index` leaves q unchanged (gap 0.0).
- q magnitude is larger on ring bonds (0.157) than non-ring (0.126), similar for
  chain/branch endpoints (0.150/0.148).
- P reaches a visibly lower fit MAE (0.0341 vs 0.0383) while its monitor MAE is worse: the
  extra channel adds fit capacity that does not translate into held-out gain.

## 5. Decision and reading

**`CROSSBOND_INFORMATION_BUT_NO_PREDICTIVE_GAIN`** (frozen vocabulary).

- The channel is *real and non-redundant*: it is outside the (W, V, n) projection and its
  functionally-new component is ~31% of its variance on real training molecules.
- The channel is *implemented and used*: invariances exact, gradients live, zeroing moves
  predictions strongly.
- It *does not improve prediction*: the marginal contrast over the matched Organized
  control is slightly negative (+0.0020 monitor, CI crossing zero), ordinary molecules are
  worse, and the top-10-removed robustness check fails. P beats N only through the shared
  Organized pathway (P−N ≈ O−N, both ≈ −0.007/−0.008 with CIs crossing zero).
- Therefore local optimization of the one-shot BJOC module via an explicit cross-bond
  endpoint combination is not a promising route at this capacity/recipe; no second variant
  was tried (that would be a new pre-registration).

## 6. Disclosures / deviations

1. **Gate metric amendment** (section 2.3): drafted linear-probe gate failed (0.0148 < 0.05);
   replaced *before the freeze commit and before any GPU* by the exact functional
   decomposition (0.3136 ≥ 0.10); both numbers reported. No GPU evidence existed at the time.
2. **Remote prep step**: the results-directory JSONs are gitignored, so the driver's
   `objects` (prep) command was run once remotely through `rr` before training; its outputs
   (`split_a.json`, `prep_a.json`, `objects_summary.json`) are byte-identical to the local
   ones (sha256 compared).
3. **Analysis-only bugfix after the GPU run, before any locked read**:
   `_monitor_maes` tag parsing dropped the `_monitor` suffix (commit after training; model,
   recipe and gate values unchanged).
4. **Analysis-only decision-classifier fix**: `cmd_decision` originally treated a negative
   P−N as "weak signal"; since P−N is dominated by the shared Organized pathway, the
   classifier now requires the marginal P−O < 0 for `CROSSBOND_WEAK_SIGNAL`. With P−O > 0
   the outcome is `CROSSBOND_INFORMATION_BUT_NO_PREDICTIVE_GAIN`.
5. **No locked read**: split-A locked labels were never read (no `locked_read_A.lock`);
   split B was never trained.
6. P is `arm P = O + q` with identical init, recipe and batch order; the comparison is
   capacity-matched and seed-matched.

## 7. Budget

- formal trajectories: **3 of 12** (N, O, P × split A seed 0).
- GPU wall: train 1142 s + predict ~30 s + objects (CPU work on a GPU pool) 17 s
  ≈ **0.33 GPU-h of 2.5**.
- wall clock for the whole round (Phase A CPU + GPU + analysis + report) ≈ 4 h.
- no seed 1, no locked read, no split B, no model/recipe changes after results.

## 8. Observations for a future (not authorized) round

- The tail (y≤−3, n=69) point estimate moves in P's favour (−0.030) while the ordinary
  bulk moves against it (+0.004); this is a hypothesis, not a finding.
- A capacity-matched alternative would be to fine-tune only the new columns of a frozen O
  checkpoint (different protocol; not tested here).
- The `(1,2)`/`(2,2)` bond classes carry the new functional component; a future high-order
  channel that preserves more than side means (e.g. side-count-aware products) is a
  different pre-registration.
- Do not re-use the split-A monitor or any split-A labels for further model selection
  without a new pre-registration; the locked split-A labels remain unread.
