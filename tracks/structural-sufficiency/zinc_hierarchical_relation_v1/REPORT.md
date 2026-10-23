# REPORT — zinc_hierarchical_relation_v1

Protocol id: `ssrc-zinc-hierarchical-relation-v1`
Track: `structural-sufficiency`
Formal training commit: `d10c3f8` (code identical to frozen source commit
`5e9218e`; `d10c3f8` only adds the final evidence manifest). Post-run analysis
path fix: `e726657` (score/compare only; no training or prediction change).
Official ZINC test was **never loaded, predicted or scored**.

## 0. Short conclusion

Both candidates fail the pre-registered promotion gate. Arm A (hierarchy alone)
reaches best-valid raw-y MAE **0.085289** (epoch 434); arm B (hierarchy + one
shared low-rank relational interaction) reaches **0.083107** (epoch 450).
The frozen reference `P3B_s0` is **0.076224**. The fixed-top-10-removed gain is
**−0.000677** (A) and **−0.000692** (B), not **≥ +0.002**. Only the train-fit
criterion passes, and both arms fit *better* than P3B at their selected epoch
(train MAE 0.01113 / 0.01209 vs 0.01468), so this is not an underfitting story.

The round therefore closes: no seed 1, no null-control C, no hyper-parameter or
feature rescue. The failure is nevertheless informative:

1. The deficit is almost entirely the **65-molecule extreme tail** (`y <= -3`).
   On the other 935 molecules A is 0.00096 better and B is 0.00270 better than
   P3B; on the tail A is 0.1533 worse and B is 0.1447 worse.
2. The tail failure is **extreme-label shrinkage**. Tail slope of prediction on
   label is 0.531 (A) / 0.548 (B) vs 0.916 for P3B. One valid molecule
   (row 172, `y = −20.3405`) is predicted −8.27 / −8.44 by A/B vs −18.78 by
   P3B; that single row contributes **+0.0103 to +0.0105 MAE**. Excluding only
   that row, B would be 0.0713 and A 0.0733 (vs P3B 0.0747) — but that is a
   post-hoc diagnostic, **not** a gate result, and it cannot be used to promote
   either arm.
3. The claimed **support-modulated induced-patch path collapsed**. Learned
   support gates are ~0.0047/0.0026 (A) and ~0.0040/0.0023 (B) for node/edge
   sums. Zeroing patch-edge types changes valid MAE by **< 2e-5** (max prediction
   change 0.0019–0.0023). Forcing the gate open worsens MAE. Support still
   matters, but as a **direct scalar feature**, not through the claimed gate.
4. B's relational interaction **is used and is locally useful**. Null relation
   assignment worsens B's valid MAE from 0.08311 to 0.08900 (mean |Δpred| =
   0.0256). B is significantly better than A on radius-2-unseen molecules
   (+0.0200 MAE, 95% CI [0.0031, 0.0404]) and on ring-rich-unseen molecules
   (+0.0735, CI [0.0064, 0.1659]). The net is negative because B also loses
   0.0134 on radius-2-covered molecules and 0.1447 on the tail.

The result changes the research judgment in a specific way: more object
families or another relation layer is not the immediate bottleneck. The
blocking failure is **readout extrapolation on extreme labels**, plus a
fine-patch path that training learns to suppress. A relation mechanism can
still be worth one new, pre-registered test, but only if it is paired with a
readout that does not shrink extreme labels.

## 1. Propositions tested

- **P-A (hierarchy):** a non-recursive atom → bond → junction → rooted induced
  patch hierarchy, one shared coarse-to-fine refinement block, train-only
  support modulation of the fine patch contribution, and a vector/second-moment
  readout reaches best-valid `<= 0.073`, with fixed-top-10-removed gain
  `>= 0.002` and train MAE at best `<= 0.026`.
- **P-B (one relation interaction):** one shared low-rank (r = 16) pair
  interaction over graph-related junction/patch tokens, with real relation ids
  and no zero-init gate, improves the same hierarchy under the same
  initialisation, data order and recipe.
- **C (null control):** within-molecule permutation of relation assignment,
  prepared and CPU-tested but **not run formally** because neither A nor B
  passed.

The gate was frozen before the first formal GPU run in `PROTOCOL.md`,
`configs/hr_v1.json`, and `analysis/frozen_manifest.json`. It was not changed
after seeing results.

## 2. Why this round is not one of the five rejected mechanisms

| Rejected mechanism | What this round did instead |
|---|---|
| IJCM pure additive object prediction | Prediction pools **vectors, second moments and local cross moments** (`a_u⊙a_v`, `b_1⊙b_2`, `b_root⊙p`) through an MLP. There is no per-object scalar identity and no exact additive sum. |
| P3 zero-init gated one-hop relation residual | B has **no zero-init gate** and is **not** a residual on P3 output. One shared low-rank bilinear interaction rewrites the junction/patch tokens that feed the main readout; relation-type biases are part of the score. |
| Full pair composition / multi-layer object attention | One interaction round, rank 16, one shared score/value projection, no per-pair MLP, no intermediate-atom contraction, no attention stack, only graph-related pairs (~903/molecule). |
| Side-branch factorised structure features ignored by the readout | The hierarchy tokens **are** the readout input; there is no separate side branch. The CPU sensitivity gate required a non-zero response, and the formal probe measured the actual learned response. |
| Local correction on a frozen P3 residual | Both arms are trained from the frozen train-only masked-SSL encoders with a fresh readout. No supervised P3 reader or P3 checkpoint is loaded. |

## 3. Frozen data and CPU evidence

Data: official ZINC subset train 10,000 / valid 1,000; no test. Inputs are
label-blind raw atom/bond features, adjacency, frozen BJOC object features,
rooted induced patches (radius 2), 33 global topology features, graph
relations, and a 4-scalar train-only support key. Support key = radius-2
nested-WL bond-centred count, fit on train only (40,380 unique keys). No
per-key embedding or table.

CPU gates (`analysis/cpu_gates.json`, all pass):

| gate | result |
|---|---|
| parameters | A 438,585; B 445,578 (budget 500k) |
| shared A/B initial state | identical hash `ac9b8ed2…` |
| edge-direction invariance | max Δpred 0.0 |
| atom-renumbering invariance | max Δpred 4.8e-7 |
| batch vs single | max Δpred ≤ 1.1e-6 |
| empty / single-atom / ring / disconnected | finite, no cross-component pairs |
| train/valid isolation | train-only support; no test; no row/mol id feature |
| gradient coverage | zero zero-grad parameter tensors, both arms |
| A sensitivity | support zero Δ 0.043; patch edge zero Δ 0.022; junction zero Δ 0.068 |
| B sensitivity | real vs null relations Δ 0.0197 at init |
| hidden ids | only role (3) and relation (33) embeddings |
| CPU profile | 0.058 s (A) / 0.107 s (B) per 128-molecule fwd+bwd |

Data audit (`analysis/data_audit.json`): exact cache fidelity; 4,969 patches
from 200 valid molecules, 4,701 nested-WL keys, 96 keys (192 patches, 3.86%)
contain more than one **induced** patch signature; synthetic C6 vs 2×C3 have
identical 2-round WL colours but different induced edge signatures. Support
unseen fraction 8.4% (200 molecules) / 9.0% (all 1000). Mean 59.3 tokens and
903 related pairs per molecule.

Overfit diagnostic (`analysis/overfit_cpu.json`): 48 molecules, 400 steps,
final eval-mode train MAE 0.177 (A) / 0.093 (B) vs constant baseline 1.005.
This short CPU run is thread-order sensitive by a few hundredths; it is not a
promotion criterion.

GPU smoke/profile (`analysis/smoke_gpu0.json`, `profile_gpu0.json`,
`profile_gpu1.json`): 30-step smoke ~0.054 s/step, 0.23/0.31 GB. 128-molecule
fwd+bwd: A 0.032 s, 4.167 GFLOPs, 0.220 GB; B 0.037 s, 4.261 GFLOPs, 0.470 GB.
A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124.

## 4. Formal runs

Both runs used the frozen recipe: AdamW, peak LR 1e-3, min 1e-5, wd 1e-5,
warmup 500, per-step cosine, clip 5, batch 128, 600 epochs, 79 updates/epoch =
47,400 updates, raw-y L1, seed 0, shuffle seed 101, best-valid checkpoint with
strict `<` and earlier-epoch tie-break.

| | A (HCE) | B (HCE+SRI) |
|---|---|---|
| parameters | 438,585 | 445,578 |
| shared state hash | `ac9b8ed2…` | `ac9b8ed2…` |
| best valid MAE | **0.085289** | **0.083107** |
| best epoch | 434 | 450 |
| train MAE at best (eval mode) | 0.01113 | 0.01209 |
| final valid MAE (epoch 600) | 0.087373 | 0.084705 |
| final train MAE (epoch 600) | 0.000718 | 0.000711 |
| A100 GPU-hours | 1.0138 | 1.2049 |
| peak CUDA memory | 0.237 GB | 0.327 GB |
| best state SHA-256 | `c0ad8cba…` | `0f78fee7…` |
| file SHA-256 (`best.pt`) | `a2f46d0b…` | `ed916e1d…` |

The post-best train MAE collapse (to ~7e-4) while valid rises is classical
overfitting after the selected epoch; the best-valid rule correctly stopped at
434/450.

## 5. Gate and statistics

Frozen reference `P3B_s0`: valid MAE 0.076224 (npz SHA-256 `8dc5ba20…`),
train MAE at best 0.01468.

| gate check | A | B |
|---|---|---|
| best-valid `<= 0.073` | **false** (0.085289) | **false** (0.083107) |
| drop-top-10 gain `>= 0.002` | **false** (−0.000677) | **false** (−0.000692) |
| train MAE at best `<= 0.026` | true (0.01113) | true (0.01209) |
| **eligible** | **false** | **false** |

Fixed P3B top-10 rows: `[917, 753, 172, 659, 260, 678, 31, 224, 447, 229]`.
On all 1000 rows A is 0.00907 worse and B 0.00688 worse than P3B. Removing the
fixed top-10 rows changes this to −0.000677 / −0.000692 (still slightly worse);
the top-10 set removes several rows where A/B are *better*, so this is not the
same as removing row 172 alone.

Paired molecule bootstrap (10,000 resamples, seed 20261025; positive = candidate
worse):

| contrast | mean ΔAE | 95% CI | P(candidate better) |
|---|---|---|---|
| A vs P3B | +0.00907 | [−0.00534, +0.03249] | 0.263 |
| B vs P3B | +0.00688 | [−0.00843, +0.03094] | 0.324 |
| A vs B | +0.00218 | [−0.00275, +0.00784] | 0.211 |

Group bootstrap (`analysis/group_bootstrap.json`; positive = first worse):

| group | n | A vs P3B | B vs P3B | A vs B |
|---|---:|---|---|---|
| all | 1000 | +0.00907 | +0.00688 | +0.00218 |
| ordinary (`y > −3`) | 935 | −0.00096 | −0.00270 | +0.00174 |
| tail (`y <= −3`) | 65 | +0.1533 | +0.1447 | +0.0086 |
| radius-2 unseen | 251 | +0.00741 | **−0.01264** | **+0.02005** [0.0031, 0.0404] |
| radius-2 covered | 749 | +0.00962 | +0.01343 | **−0.00381** [−0.0069, −0.0008] |
| ring-rich | 213 | −0.01034 | **−0.02163** [−0.0443, −0.0037] | +0.01129 |
| ring-rich & unseen | 47 | −0.02008 | **−0.09358** [−0.1913, −0.0190] | **+0.07350** [0.0064, 0.1659] |
| P3B worst 100 | 100 | +0.03207 | +0.00563 | +0.02644 |
| drop fixed top-10 | 990 | +0.000677 | +0.000692 | −0.000016 |

The one significant positive mechanism result is B on ring-rich / ring-rich-
unseen molecules, and B over A on radius-2-unseen. The one significant negative
is B on radius-2-covered molecules (small, −0.0038). B's gains and losses are
both real; the tail dominates the net.

## 6. Per-molecule and extreme-tail decomposition

`analysis/error_decomposition.json` and `error_decomposition_extra.json`:

| group | n | P3B MAE | A MAE | B MAE |
|---|---:|---|---|---|
| all | 1000 | 0.076224 | 0.085289 | 0.083107 |
| tail (`y <= −3`) | 65 | 0.190540 | 0.343874 | 0.335261 |
| tail excluding row 172 | 64 | 0.169095 | 0.160582 | 0.154559 |
| ordinary | 935 | 0.068277 | 0.067313 | 0.065578 |
| radius-2 unseen | 251 | 0.151340 | 0.158751 | 0.138702 |
| all excluding row 172 (diagnostic) | 999 | 0.074735 | 0.073288 | 0.071278 |

Tail calibration:

| | P3B | A | B |
|---|---|---|---|
| tail bias (pred − y) | +0.1060 | +0.2452 | +0.2427 |
| tail slope of pred on y | 0.9163 | 0.5312 | 0.5481 |
| tail slope excluding row 172 | 0.9299 | 0.9406 | 0.9661 |

Row 172: `y = −20.3405`, 24 atoms, 3 rings, radius-2-unseen fraction 0.
P3B predicts −18.7775 (AE 1.563); A predicts −8.2659 (AE 12.075);
B predicts −8.4403 (AE 11.900). This is an **out-of-range extrapolation
failure**, not a local-structure error: row 172 is structurally ordinary
(3 rings, no unseen radius-2 atom) but has an extreme label.

Error-mass decomposition of the overall gap: A tail excess +0.009967,
ordinary −0.000901; B tail excess +0.009407, ordinary −0.002523. The tail
accounts for more than the entire net deficit in both arms.

## 7. Mechanism probes (post-hoc, frozen best checkpoints)

`analysis/probe_A_s0.json`, `probe_B_s0.json`.

### 7.1 The support-modulated fine-patch path collapsed

Learned support gates over 24,846 valid patches:

| | A | B |
|---|---|---|
| node gate mean / std | 0.00467 / 0.00120 | 0.00398 / 0.00135 |
| edge gate mean / std | 0.00257 / 0.00086 | 0.00230 / 0.00081 |
| node gate q05–q95 | 0.00255–0.00655 | 0.00221–0.00674 |

Zeroing / shuffling support changes predictions strongly (support zero:
A 0.13235, B 0.13143; support shuffled per molecule: A 0.09943, B 0.09874), so
support information is used. But the **gate** multiplying the induced patch
node/edge sums is nearly closed. Zeroing patch-edge types changes valid MAE
from 0.085289 to 0.085276 (A) and from 0.083107 to 0.083087 (B); max prediction
change is only 0.0023 (A) / 0.0019 (B). Forcing the gate open worsens MAE
(A 0.09203; B 0.08464), and half-open also worsens A (0.08610) but slightly
improves B (0.08279). The model learned that the fine patch sums are noisy and
suppressed them; the “support modulates the fine contribution” claim is **not**
what the trained model does. Support survives as a direct feature concatenated
into the patch MLP, not as a gate on induced structure.

Junction features still matter (A 0.08924, B 0.08442 when zeroed) and global
features matter strongly (A 0.10479, B 0.10567), so the model is not degenerate;
it simply bypasses the new fine patch path.

### 7.2 B's relation interaction is used

For B, replacing real relation ids with the within-molecule null permutation at
inference:

| | real | null |
|---|---|---|
| valid MAE | 0.083107 | 0.088997 |
| mean / max abs pred Δ | — | 0.02556 / 0.4559 |
| prediction correlation | — | 0.99983 |

Attention over related tokens: mean max weight 0.375, mean entropy 2.05 (over
sources with ~non-uniform segment softmax). Relations are not ignored; they
buy about 0.0059 MAE over the null. The relation mechanism is therefore a real
local effect, but not enough to offset the readout's tail failure.

## 8. Does this change the research judgment?

Yes, in three specific ways.

1. **The 0.076 platform is not broken by this hierarchy.** Adding induced
   patches, a shared coarse-to-fine stack, support features and a non-additive
   moment readout does not improve the official valid mean over P3B. The best
   A/B numbers are 0.0853/0.0831, not 0.05x and not below P3B.
2. **The fine-structure interface, as implemented, is not viable.** The
   support gate collapses and patch-edge ablation is negligible. This is
   evidence against *this* fusion of support and induced patches, not evidence
   that induced patches are useless; a future test must make the patch path
   necessary (e.g., remove the direct support shortcut or supervise/regularise
   the gate open) and show a non-zero structural ablation.
3. **The bottleneck has moved to extreme-label extrapolation.** Ordinary and
   ring-rich/unseen molecules are at or better than P3B in both arms; the
   entire net loss is the tail, dominated by one extreme label. More object
   families, more depth, or another relation layer would not address this.
   A readout that predicts the label scale (or a monotone/affine calibration)
   is the next thing to test, not a wider vocabulary.

## 9. Failure-mode classification

- **Fitting:** not the failure. Train MAE at best is 0.01113/0.01209, below the
  0.026 gate and below P3B's 0.01468; both arms overfit after the selected
  epoch (final train ~7e-4).
- **Generalisation:** the failure. Overall valid is worse, concentrated in the
  tail; B also loses slightly on radius-2-covered molecules.
- **Relation use:** not the failure. B's real-vs-null gap is 0.0059 MAE, and
  B is significantly better than A on radius-2-unseen / ring-rich-unseen.
- **Information / inductive bias:** partially the failure. The fine induced
  patch is effectively suppressed; the readout compresses extreme labels
  (tail slope 0.53/0.55 vs 0.92 for P3B). The model has enough information to
  fit train and to improve ordinary/ring-rich molecules, but not enough
  output-range inductive bias for row-172-like labels.

## 10. Is it worth continuing?

The round itself is closed. No seed 1, no C, no rescue under this protocol.
There is one worthwhile **new** falsifiable question:

> Is the tail failure caused by the readout's inability to extrapolate extreme
> labels (fixable by a label-scale-aware / monotone output head) or by missing
> structural information (not fixable without new inputs)?

A new protocol should pre-register a readout change (for example, predict a
standardised target and invert a train-fitted scale, or add a monotone
global-baseline residual) and test whether it removes the row-172/tail
shrinkage **without** losing B's ring-rich/unseen gain. In the same new round,
a mechanism control should remove the direct support concatenation (or
regularise the support gate open) to test whether the induced-patch path can be
made non-negligible. Neither test is run here.

## 11. Provenance, resources, discipline

- Formal training commit `d10c3f8` (`git_commit` in both `train.json`);
  frozen source code commit `5e9218e`; post-run analysis path fix `e726657`.
- Config SHA-256 `9bafebd6…`; SSL checkpoint SHA-256 `006fabc1…`.
- Train cache SHA-256 `6dfbfae0…`; valid cache `cf2f6fb3…`; stats `16bff3d7…`.
- P3B valid prediction SHA-256 `8dc5ba20…`; valid reference MAE 0.076224.
- CPU gates all-pass; frozen manifest and file hashes in
  `analysis/frozen_manifest.json`.
- GPU: 2× A100-SXM4-40GB on `res` (node `a100-2`), driver 550.163.01,
  torch 2.5.1+cu124. A 1.0138 A100-h; B 1.2049 A100-h; total 2.2187 A100-h,
  below the 4-hour round budget. Smoke/profile/prep were negligible.
- Official test never instantiated, loaded, predicted or scored.
- Historical reports and checkpoints were not modified or overwritten; this is
  a new track. The pre-registered gate was not redefined. Deviations are
  recorded in `analysis/deviations.md` (pre-formal support-key permutation fix,
  remote prep manifest plumbing, CUDA init fix, and post-run score/compare path
  fix); none changed the model, data, recipe, gate or any reported metric.

## 12. Raw evidence index

| path | contents |
|---|---|
| `analysis/frozen_manifest.json` | frozen code/config/protocol/cache hashes, parameters, gate |
| `analysis/prep.json` | cache and train-only statistics provenance |
| `analysis/cpu_gates.json` | all CPU gates and parameter counts |
| `analysis/data_audit.json` | patch-vs-WL, support coverage, relation stats, fidelity |
| `analysis/overfit_cpu.json` | controlled overfit diagnostic |
| `analysis/smoke_gpu0.json`, `profile_gpu0.json`, `profile_gpu1.json` | GPU timing/memory/FLOPs |
| `analysis/train_A_s0.json`, `train_B_s0.json` | full curves, recipe, environment, hashes |
| `analysis/curves/HR_A_s0_curve.csv`, `HR_B_s0_curve.csv` | per-epoch curves |
| `analysis/predictions/HR_A_s0_valid_best.npz`, `HR_B_s0_valid_best.npz` | frozen valid predictions |
| `analysis/checkpoints_sha256.txt` | checkpoint/prediction file hashes |
| `analysis/score_A_s0.json`, `score_B_s0.json` | gate, bootstrap, groups, per-molecule source |
| `analysis/per_molecule_A_s0.csv`, `per_molecule_B_s0.csv` | per-valid-row errors and group flags |
| `analysis/comparison.json` | A-vs-B bootstrap, prediction/error correlation |
| `analysis/group_bootstrap.json` | group-wise paired bootstrap |
| `analysis/probe_A_s0.json`, `probe_B_s0.json` | support-gate, patch/junction/global ablations, real-vs-null |
| `analysis/error_decomposition*.json` | tail/ordinary/outlier decomposition |
| `analysis/results_summary.json` | derived summary table (raw files remain authoritative) |
| `analysis/deviations.md` | every deviation, with reasons |

Checkpoints remain in `runs/HR_A_s0/` and `runs/HR_B_s0/` locally (gitignored)
and on `res`; they were not overwritten. The next round must use a new protocol
id if it changes the gate, readout, data contract or budget.
