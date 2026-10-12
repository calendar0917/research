# BJOC Cross-Bond Interaction v1 — REPORT

protocol `cscl-bjoc-cross-bond-interaction-v1` — frozen and committed (9f890dd) before any GPU run;
generated 2026-10-12.

**Final decision: `CROSSBOND_INFORMATION_BUT_NO_PREDICTIVE_GAIN`.**
The cross-bond channel is real, invariant and used by the trained model, but on split A seed 0 it
does not improve held-out MAE over the matched BJOC-Organized control (P−O +0.001952); the frozen
seed-1/locked/split-B purchases were therefore not made and split-A locked labels were never read.

## 1. Question and intervention

Does explicitly preserving the two-endpoint combination of junction environments of a physical bond
add prediction information beyond the published `W_b, V_b, n_b` summaries?

```
q_b = sqrt(96) * mu_u(b) * mu_v(b)     (elementwise; mu_side = mean junction embedding per side)
```

Arms (all 428,865 params, identical init `b4f1de541f0f8692`, identical batch order, raw-y L1,
last-5 soup): N = BJOC-Null, O = BJOC-Organized (published recipe), P = O + q. Only difference
P vs O: 96 zero-initialised q columns on the composer input (O ≡ P at init, exact).

## 2. Phase A — CPU information audit (hard purchase gate)

- decision: **PHASE_A_PURCHASE_GRANTED**
- projection_collision_exists: True — scalar witness (W=10, V=30, n=4, products 21 vs 24),
  322 collision keys; 96-D collision with `W_equal=True`, `V_equal=True`, `max|Δq| = 7.3485`
- real_valid_pair_fraction: True — 0.8350 (208,149 / 249,279 bonds; n_u/n_v classes 1/2/3 =
  100,945 / 129,285 / 9,108)
- real_q_dims_with_variance: True — 96/96 dims
- real_invariants: True — endpoint swap 0.0, junction endpoint swap 0.0, atom relabel 3.6e-15
- synthetic_q_model_improves: True — test MAE 0.1238 (projection-only) vs 0.1008 (+q);
  collision prediction gap 5.54 vs 0.32
- **exact identity** (gap 3.6e-15): `q/sqrt(96) = (W²−V)/(2 n_u n_v) − P_net`,
  `P_net = [(s_u²−Q_u)+(s_v²−Q_v)]/(2 n_u n_v)`; the `(W²−V)/2` part is exactly re-expressible from
  the old summaries, `P_net` is not a function of (W, V, n)
- **functional new-variance share 0.3136 / new-L1 share 0.4347** (governing non-degeneracy);
  by class (n_u,n_v): (1,1) 0.000 / 11,502 bonds; (1,2) 0.313 / 22,094; (2,2) 0.247 / 6,914;
  (1,3) 1.80 / 712; (2,3) 0.565 / 577; (3,3) 0.443 / 24
- reported not gating: linear view-probe residual 0.0148 (same-distribution linear predictability
  from W, V, n, bond features); simple degree/bond-type/ring-flag probe residual 0.389
- all 24 correctness gates pass; N/O/P parameter-equal; O ≡ P at init; no GNN/attention/iteration;
  official valid/test never loaded

### Amendment disclosure

The first draft of the pre-registration gated on the *linear* view-probe residual with a 0.05
threshold; the measured 0.0148 would have failed. Running the audit showed that number measures
same-distribution predictability, not functional non-expressibility (`P_net` is statistically
predictable from V on this distribution although no function of (W, V, n) can express it). The
governing criterion was changed to the exact functional decomposition (threshold 0.10; measured
0.3136) **after seeing Phase-A outputs but before the freeze commit and before any GPU job**; both
numbers remain reported in `phase_a_gate.json` under `reported_not_gating`. The change and its
timing are also recorded in the config file, the notes and this report.

## 3. GPU stage — split A seed 0 (`pair-a-s0`)

Run `bjoc-cbic-pair-a-s0-20261012-190941-52050b03`, host `res`, pool `res-gpu1`, commit `9f890dd`,
node a100-2 (CUDA_VISIBLE_DEVICES=1), torch 2.5.1+cu124, exit 0.

| arm | fit MAE | monitor MAE | params | wall |
|---|---:|---:|---:|---:|
| N | 0.039239 | 0.098266 | 428865 | 362 s |
| O | 0.038275 | **0.089843** | 428865 | 364 s |
| P | **0.034149** | 0.091796 | 428865 | 416 s |

Paired bootstrap (5000 resamples, molecule unit, monitor, labels never read before prediction
hashing):

| contrast | Δ MAE (candidate − ref) | CI95 | frac candidate better |
|---|---:|---:| ---:|
| P − O | **+0.001952** | [−0.003703, +0.007767] | 0.467 |
| P − N | −0.006471 | [−0.016748, +0.001645] | 0.504 |
| O − N | −0.008423 | [−0.017744, −0.001120] | — |

Subgroups (P − O): ordinary (y>−3, n=931) +0.004350; tail (y≤−3, n=69) −0.030399;
top-10-removed +0.001587. (P − N): ordinary −0.005819; tail −0.015262; top-10-removed −0.005719.

### Frozen monitor gate (seed-1 purchase)

| check | result |
|---|---|
| p_minus_o_le_threshold (≤ −0.006) | **False** (+0.001952) |
| p_minus_n_le_threshold (≤ −0.004) | True (−0.006471) |
| ordinary_not_worse (≤ 0) | **False** (+0.004350) |
| not_single_molecule_driven (top-10-removed ≤ 0) | **False** (+0.001587) |
| no_new_overfit_signal | True (P.mon−fit 0.0576 ≤ max(N,O gaps)+0.01 = 0.0691; P.fit ≥ O.fit−0.01) |

**decision: `CROSSBOND_NO_SEED_REPLICATION_PURCHASE`.** Seed 1, the split-A locked read and split B
were not purchased; `locked_read_A.lock` does not exist.

## 4. Mechanism audit (`mechanism_audit.json`, CPU, labels never read)

- q columns are consumed: mean |input weight| 0.0202 (q) vs 0.0167 (base columns), max 0.343 vs 0.212.
- zeroing q changes predictions by mean |Δ| 0.2407 (max 13.58); permuting q rows by 0.2473.
- endpoint swap leaves q exactly unchanged (gap 0.0).
- q magnitude: ring bonds 0.157 vs non-ring 0.126; chain vs branch endpoints 0.150 vs 0.148.
- P reaches a lower fit MAE (0.0341 vs 0.0383) while monitor is worse: the added channel absorbs fit
  capacity without held-out gain.

## 5. Final decision and reading

**`CROSSBOND_INFORMATION_BUT_NO_PREDICTIVE_GAIN`**

- *Information*: the channel is provably outside the old projection (collision + exact identity),
  and its functionally-new component is ~31% of its variance on real training molecules.
- *Implementation*: invariances exact, gradient live, strongly consumed; P ≡ O at init.
- *Prediction*: no gain over the matched Organized control (P−O +0.0020, CI crossing zero; ordinary
  bulk worse; robustness check fails). P−N is negative but reflects the shared Organized pathway,
  not the new channel.
- *Consequence*: the one-shot BJOC local module is not improved by this explicit cross-bond
  endpoint-combination channel at the frozen capacity/recipe. No second variant, rank search or
  retuning was performed (banned); the branch stops here.

## 6. Disclosures

1. Phase-A gate-metric amendment (section 2, before freeze commit, before any GPU).
2. The driver `objects`/prep step was executed once remotely because the results JSONs are
   gitignored; `split_a.json`, `prep_a.json`, `objects_summary.json` are byte-identical local vs
   remote (sha256 compared).
3. Analysis-only bugfix after the GPU run, before any locked read: `_monitor_maes` dropped the
   `_monitor` suffix while parsing keys. Model, recipe and gate values unchanged.
4. Analysis-only decision-classifier fix: `CROSSBOND_WEAK_SIGNAL` now requires the marginal P−O < 0
   (a negative P−N alone is dominated by the shared Organized pathway). With P−O > 0 the outcome is
   `CROSSBOND_INFORMATION_BUT_NO_PREDICTIVE_GAIN`. No locked label had been read.
5. Training ran on the committed revision `9f890dd`; the two analysis-only fixes above are the only
   post-freeze driver changes and do not affect training, predictions or gate numbers.
6. The tail subgroup improvement (P−O −0.030 on 69 molecules) is recorded as a hypothesis only.

## 7. Discipline

- official valid/test never loaded; historical official-test number 0.092233 not used for any
  decision; locked labels read exactly once per trained split (here: never).
- no second interaction definition, no bilinear matrix, no rank/hidden/location search, no
  GNN/attention/iteration, no component supervision, no teacher, no model change after results.
- capacity-matched (428,865 everywhere), seed-matched, batch-order-matched contrasts.
- budget: 3 of 12 trajectories; ~0.33 GPU-h of 2.5 (train 1142 s + predict + prep).

## 8. Artifacts

`phase_a_gate.json`, `projection_collision_audit.json`, `real_molecule_information_audit.json`,
`synthetic_cross_bond_test.json`, `correctness_gates.json`, `parameter_inventory.json`,
`protocol_freeze_manifest.json`, `training_manifest_pair-a-s0.json`, `train_{N,O,P}_A_s0.json`,
`checkpoint_{N,O,P}_A_s0.pt`, `prediction_manifest_A.json`,
`split_a_monitor_predictions.npz`, `stage1_decision_A.json`, `mechanism_audit.json`,
`scientific_decision.json` (all gitignored under `results/`).
