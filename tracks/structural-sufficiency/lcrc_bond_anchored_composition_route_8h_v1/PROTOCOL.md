# PROTOCOL — Bond-Anchored S2 Joint Composition (BAJC)

Round id: `lcrc-bond-anchored-composition-route-8h-v1`
Track: `tracks/structural-sufficiency/`
Wall clock budget: 7.5 h soft / 8 h hard. GPU budget: 14 GPU·h soft / 16 hard on
`res-gpu0` + `res-gpu1` (one GPU process per card).

This file, `configs/frozen.yaml`, `code/*.py` and `tests/*.py` are SHA-256 hashed
into `analysis/protocol_freeze.json` **before the first formal training unit**.
After that, only engineering fixes are allowed, and each must be reported in
`analysis/compute_ledger.json` with the affected units and the reason for the
rerun.

## 0. What is being tested, and what is not

One proposition, taken verbatim from the task book:

> For two real S2 occurrences that share a central **physical bond**, does a
> nonlinear joint computation *before* aggregation carry task information beyond
> keeping the same occurrences, the same parameter count and the same scale but
> destroying the real second-path partner correspondence?

Not tested here (forbidden, already bought in earlier rounds): three-hop counting,
fixed-motif pooling, internal/cross γ scaling, PATH-BIND/UNBOUND, static S3
blocks, plain additive heads, unconstrained architecture search. Three-hop paths,
path neural networks and higher-order graph models all have literature
precedents; the operator is **not** claimed to be novel.

## 1. Model

### 1.1 C1 is preserved bit-for-bit

Reused unchanged through `cr_models.ArmA` (read-only): atom encoder, bond encoder,
S1 neighbour embedding, typed S2 `type/end/joint/closure/content`, `LocalMLP`,
`LayerNorm`, dropout, `sum_v f(h_v)` and `rho([z_g, mean h, mean h^2, log1p n])`.
228 674 parameters; 213 376 in the trunk and 15 298 in the readout.

### 1.2 The composition object

For every molecule, all simple three-bond paths `P=(a,b,c,d)` with four distinct
atoms and physical bonds `(a,b),(b,c),(c,d)`. The middle bond `(b,c)` is the
anchor. `P` corresponds to the two real adjacent directed S2 occurrences
`p=(a,b,c)` and `q=(b,c,d)` and to the reversed pair `p_rev=(d,c,b)`,
`q_rev=(c,b,a)`. The occurrence map is rebuilt from the frozen cache with
`as_data.build_occ_map` and verified item by item against the cached
`path_cat / path_end63 / path_end / path_closure`; gate 3 additionally re-counts
the whole event pool with an independent adjacency enumeration.

Frozen conventions:

* each **undirected** path is one event, stored in the lexicographically smaller
  direction; the model output is exactly symmetric under reversal
  (`z_P = 0.5 (psi(x_{p,q}) + psi(x_{p_rev,q_rev}))`), so atom relabelling cannot
  change any number;
* `closure4 = 1` iff the terminal atoms `(a,d)` are additionally bonded;
* no functional group, SMARTS, label, size or ring filter; no truncation, no
  sampling, no dropping of large molecules (exact event chunking only).

Counts (frozen, from `analysis/path3_support.json`): Train 179 589 events over
3 360 molecules (p50 54, p95 78, max 188); Valid 23 334; Test 23 796. Every
molecule in every split has at least one event, and every Valid molecule has at
least two distinguishable partner events
(`fraction_molecules_distinguishable = 1.000`), far above the pre-registered
`LOW_PAIRING_SUPPORT` floor of 0.30.

### 1.3 Branch

```
r_p        = the frozen 200-d typed-S2 record of occurrence p
u(r)       = SiLU(Linear(200,32) r)          v(b) = SiLU(Linear(96,16) b)
x(p,q,e)   = [u(r_p), u(r_q), u(r_p)*u(r_q), v(b_e), closure4]           (113)
psi        = Linear(64,32) . SiLU . Linear(113,64)
z_P        = 0.5 * ( psi(x(p,q,e)) + psi(x(p_rev,q_rev,e)) )
c_P        = Linear(32,1,bias=False)(z_P) / sqrt(max(1, n_paths3(G)))
delta_comp = sum_P c_P                y_new = y_base + delta_comp
```

The nonlinearity runs before the per-path aggregation; the operator is **not**
reducible to "sum both sides, then multiply channel-wise". The readout weight is
initialised to exactly zero, so `delta_comp == 0` at initialisation and
`y_new == y_base` bit-for-bit. 17 392 branch parameters = 7.61 % of C1
(≤ 15 %). Branch dropout 0.10 draws from an isolated `torch.Generator`, so the C1
dropout stream is untouched.

### 1.4 MARGINAL control

Identical event list, anchor bond, first path, `closure4`, normaliser, parameter
shapes, parameter count, initialisation and batch order. The only change:

```
r_q_null = mean_{e in paths3(G)} r_second(e)      (per direction)
x_null   = [u(r_p), u(r_q_null), u(r_p)*u(r_q_null), v(b_e), closure4]
```

`u` is applied to the mean (i.e. `u(mean r)`, not `mean u(r)`), exactly as
specified. This removes the per-event partner identity and keeps the molecule-level
second-path marginal. It is a **partner-binding control**, not an ideal null
matched on every graph statistic: the mean replacement also changes per-event
variance, so a JOINT advantage cannot be attributed to one unique chemical
interaction.

## 2. Institutions

**F (FROZEN).** The selected, already trained C1 checkpoint of the task is loaded
(identity re-verified by SHA-256), every parameter and buffer is frozen and the
module stays in `eval()`. Only the branch is trained: AdamW lr 1e-3, wd 1e-5,
clip 5, batch 128, ≤ 180 epochs, branch dropout 0.10, loss = full standardised
MSE + `0.02 * mean(delta_comp^2)`. Minimum Valid RMSE selects the checkpoint and
**epoch 0 is a legal candidate**, so best-Valid better than C1 is a *constructive*
property of the selection rule and is never used as evidence for the mechanism.

**E (END_TO_END).** A fresh model from the canonical C1 seed-0 initial state (never
the trained teacher), C1 and branch trained together under the frozen Rf recipe:
360 epochs, AdamW, peak LR 1e-3, 5 warmup epochs, per-step cosine to 1e-5,
wd 1e-5, batch 128, dropout 0.10, fp32, clip 5, Train-only target normalisation,
minimum Valid RMSE selection with ties to the earlier epoch. Rf contains no branch
penalty, so E2E has none; JOINT and MARGINAL are treated identically.

Randomness: batch order is `np.random.default_rng(0)` permutations, one per epoch,
identical for every unit; the canonical C1 initial state and the canonical branch
initial state (`BRANCH_INIT_SEED = 20261118`) are byte-identical across JOINT and
MARGINAL.

## 3. Data and comparability

Main task: `ogbg-mollipo` official split (3 360 / 420 / 420), experimental logD in
raw units, RMSE. Computational task: RDKit WC_LOGP on the same 4 200 molecules and
the same official split, **its own** Train-only scaler — never pooled with logD.
Historical baselines are read-only anchors:

| model | Valid RMSE | Development Test RMSE |
|---|---:|---:|
| C1 (Rf, seed 0) | 0.7069 | 0.7293 |
| GINE-G0 (Rf, seed 0) | 0.6161 | 0.6800 |

The C1 teacher for both tasks is the frozen
`lcrc_explanation_task_relevance_sprint_v1` checkpoint
(`L_lipophilicity`, `L_WC_LOGP`); its per-molecule predictions are replayed to
1.5e-6 raw RMSE units before any training (gate 2). No historical model is
retrained. Old `random122` / `random42` / `public122` results never enter a main
comparison table.

## 4. Stages and triggers

* **Stage 0** — preflight, path3 cache, support statistics, 14 gates + one
  non-label numerical gate, then the freeze.
* **Stage A** — FROZEN × {JOINT, MARGINAL} × {lipo, WC_LOGP}, four units.
* **Stage B** — E2E × {JOINT, MARGINAL} × lipo × {seed0, seed1}, four units.
  Mandatory even if Stage A shows nothing.
* **Stage C** — only if a candidate signal fires on Lipo **Valid**:
  JOINT vs same-institution MARGINAL ≥ 0.010 RMSE, or JOINT vs the C1 anchor
  ≥ 0.015, or paired bootstrap `P(JOINT better) ≥ 0.80`, or a non-degenerate
  branch with non-reversed Train/Valid gains. E2E is confirmed if it triggers,
  otherwise FROZEN. Then: seed 2 in the confirmed institution, ESOL transfer, and
  the explanation evaluation (native event ledger + physical-bond evidence
  intervention + WC_LOGP oracle localisation + edit-pair support).
* **Stage D** — if nothing fires: a one-shot read-only integration (error
  localisation of the C1−GINE gap on four pre-specified slices, summary of the
  existing explanation evidence, and the mechanism ledger). No GPU.

These thresholds buy a confirmation experiment; they are not significance tests.

## 5. Explanation evaluation (Stage C only)

* **Constructive ledger** `delta_comp = sum_P c_P` is an architectural identity
  and never counts as a faithfulness result.
* **Physical-bond intervention** reuses the validated
  `CONDITIONAL_S1_S2_EVIDENCE_MASK`: S1 evidence weighted by the bond gate, every
  S2 occurrence by the product of its two bond gates, every composition event by
  the product of its three bond gates; `atom_feat`, `atom_scalar` and `global` are
  never gated. `P* = argmax_P |c_P|` is the pre-registered, label-free discovery
  rule; it is compared against matched random events from the same molecule's
  candidate pool (fixed RNG 20261118, up to 16 controls).
* **WC_LOGP oracle**: `capture(P) = sum_{v in atoms(P)} |oracle_c_v| / sum_v |oracle_c_v|`,
  matched-control enrichment, 2 000-draw molecule-level bootstrap. It is a
  *computed* Crippen contribution, never an experimental chemical cause.
* Edit pairs reuse the existing 823 Lipo catalogue read-only; non-Train support is
  reported as `LOW_SUPPORT` when insufficient and never upgraded to validation.

## 6. Decisions

Performance: `gain_base = RMSE(C1_frozen) − RMSE(JOINT)`, `gain_null = RMSE(MARGINAL) − RMSE(JOINT)`,
paired molecule bootstrap (2 000 draws, seed 20261118), CI95 and
`P(JOINT better)`. Mechanism: only JOINT vs MARGINAL can support a real
partner-binding increment; a gain over C1 without exceeding MARGINAL is labelled
`CAPACITY_OR_REPARAMETERIZATION_GAIN`. Explanation: external localisation evidence
must come from the WC_LOGP oracle on a matched candidate pool **and** be reported
alongside the physical-bond intervention. Combination verdicts:
`JOINT_ROUTE_PROMISING`, `PERFORMANCE_ONLY`, `STRUCTURAL_CALIBRATION_ONLY`,
`NO_COMPOSITION_ROUTE_ADVANCE`, `INCONCLUSIVE_OR_BLOCKED`.

Every number in this round is DEVELOPMENT. Test is read exactly once per frozen
model, after selection.
