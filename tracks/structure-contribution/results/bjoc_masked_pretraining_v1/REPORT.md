# cscl-bjoc-masked-pretraining-v1 — report

**Decision: `SELF_SUPERVISED_PRETRAINING_USEFUL`** (frozen rule
`configs/bjoc_masked_pretraining_v1.yaml`; decision written by the driver, not by hand).
One-sentence conclusion: *strictly leak-free masked atom/bond pretraining of the three existing
BJOC encoders on the same 8k Fit molecules, with no extra labels, molecules or architecture
change, improves locked-dev raw-y MAE by `-0.011984` (seed-mean, CI95 `[-0.017530, -0.005974]`,
both seeds negative, ordinary molecules `-0.013472`, top-10-error-removed `-0.013204`) and beats a
50%-extra-supervision scratch control (`P − C = -0.007713`, CI95 `[-0.013252, -0.001911]`), so the
gain is a generalization effect rather than pure optimization acceleration.*

Historical context (mandatory): the previous BJOC-Null full-10k **official-test seed-mean is
`0.092233`** against the user target band `0.06–0.07`. This round ran on the official **train**
split only, on a new internal grouped split, and **gives no extrapolation to the official test**;
internal locked MAE levels below are not comparable to official-test values and no claim is made
that any technique here reaches any specific official-test MAE.

## 1. Protocol, provenance, budget

Frozen before any GPU (commit `3a25521`, preregistration): split `cscl-bjoc-maskpretrain-v1-A`
(Fit8k 8000 / Monitor 1000 / Locked-dev 1000, canonical-SMILES grouped, no y used, exact sizes,
no duplicate group crossing sides); structure-only object caches + Fit8k-only standardiser hashes
in the config `frozen:` block; one pretext task (15% atoms + 15% physical bonds per molecule, mask
tokens atom `27` / bond `0`, full rebuild of every BJOC feature from the masked raw graph,
`L_SSL = CE_atoms + CE_bonds` 1.0/1.0, 7560 updates, batch 128, Adam 1e-3/wd 1e-5/clip 5, last-step
encoder state only); arms B (scratch) / P (same init + pretrained encoders) / C (conditional,
22680 updates); monitor purchase gate frozen in advance.

Provenance: all training ran on deployed commit `3a25521` (phase F) and `7261e77` (phase G). The
analysis-only commit `7261e77` (`git diff 3a25521..7261e77` = 10 insertions / 2 deletions in
`run_bjoc_masked_pretraining_v1.py`, both in analysis helpers: paired per-seed contrast keys and
the C batch-order provenance check) touched **no training path**; `training_manifest.json` was
regenerated at the analysis revision and the training-revision manifest is preserved as
`training_manifest_at_phase_f.json` (same module / split / cache hashes, different driver hash).

Budget actually used: 2 formal remote stages (phase F 84.5 min, phase G 20.2 min) = **104.7 min
wall on one A100** ≈ 1.75 GPU-h (limit 2 GPU-h), 2 remote + local CPU-only gates/audits,
total wall well inside 7 h 45 min; 8 formal-stage limit respected. No full-10k training.

## 2. Leak audits (all before GPU, re-run on the remote A100 node)

* `masked_feature_dependency_audit.json` — full leak-path map; mask-token slots free in Fit8k
  (atom codes 0–20, bond codes 1–3 used) → `MASK_TOKEN_SLOTS_FREE`.
* `mask_target_swap_audit.json` — deterministic minimal witness + **200/200 atom and 200/200 bond
  random hidden-truth swaps** on real Fit8k molecules: every BJOC / pretext-head input bit-identical
  while the hidden target changes; visible-change positive control changes only dependency cells →
  `MASK_SWAP_INVARIANT`.
* `masked_object_rebuild_audit.json` — 100/100: rebuild deterministic, diffs ⊆ dependency map, own
  slots hold only the mask token, hidden truth absent from own slot, both directed PyG records of a
  masked physical bond replaced, junction/global rebuild consistent → `MASKED_REBUILD_LEAK_FREE`.
* `pretext_leakage_gates.json` — **22/22 PASS** (including hidden-truth absence, double-edge sync,
  finite CE, non-zero encoder gradients, heads isolated, 413,505 downstream params, permutation
  invariance, endpoint symmetry, batch consistency, Fit8k-only, no-y pretraining process,
  save/load replay, mask-RNG reproducibility, official valid/test never loaded, no
  GNN/attention/iteration). `labels_read: false` inside pretraining; Fit8k/Monitor-only y is read
  outside pretraining for monitor/standardisation diagnostics only.

The locked-dev y was read **exactly once**, after all 12 prediction files (B/P/C × 2 seeds ×
monitor/locked) were frozen and hashed (`locked_prediction_manifest.json`, combined
`72160afc…`; `locked_read.lock` written immediately after the single read). No training or model
selection followed. Official valid/test were never accessed anywhere.

## 3. What the pretext task learned

| seed | final CE atom | acc atom | final CE bond | acc bond | wall |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2104 | 0.9268 | 0.0154 | 0.9952 | 1723 s |
| 1 | 0.1990 | 0.9289 | 0.0186 | 0.9946 | 1705 s |

Priors (Fit8k, visit 0): atom marginal CE 1.1181 nats (majority class 70.2%), bond 0.6051 nats
(74.2%). Both objectives beat the marginal baseline by a wide margin (`pretext_beats_prior = true`,
`M4_negative_transfer`). The bond objective is close to ceiling, i.e. a low-entropy but non-leaking
signal; the atom objective remains genuinely informative. Masked data path: 1.58 ms/molecule masked
rebuild on the remote CPU, projected 26.6 min/seed data preparation — the dominant wall cost of the
SSL arms and the reason phase F took 85 min.

## 4. Primary result — P vs B on the one locked read

Per-seed readouts (raw-y MAE; the monitor row is the training-time readout, descriptive only):

| arm | locked s0 / s1 | locked seed-mean | median | RMSE | p90 | monitor s0 / s1 |
|---|---:|---:|---:|---:|---:|---:|
| B scratch | 0.099356 / 0.108689 | 0.104023 | 0.0666 | 0.1663 | 0.2236 | 0.105501 / 0.120351 |
| C scratch, +50% updates | 0.095860 / 0.103643 | 0.099752 | 0.0659 | 0.1661 | 0.2117 | 0.105471 / 0.105396 |
| **P pretrained** | 0.091976 / 0.092102 | **0.092039** | 0.0564 | 0.1842 | 0.1917 | 0.104861 / 0.099289 |

Frozen primary contrasts on locked (paired molecule bootstrap 5000 resamples; negative = candidate
better):

| contrast | seed-mean Δ | CI95 | per-seed | ordinary y>-3 | tail y≤-3 | top-10 removed | better-mol frac |
|---|---:|---|---:|---:|---:|---:|---:|
| **P − B (pretraining)** | **-0.011984** | [-0.017530, -0.005974] | -0.007380 / -0.016588 | -0.013472 | +0.007483 (n=71) | -0.013204 | 0.589 |
| C − B (extra compute) | -0.004271 | [-0.008241, -0.000323] | -0.003496 / -0.005046 | -0.004379 | -0.002858 | -0.005305 | 0.542 |
| **P − C (pretraining vs +50% compute)** | **-0.007713** | [-0.013252, -0.001911] | -0.003884 / -0.011542 | -0.009093 | +0.010341 | -0.008526 | 0.555 |
| P − B monitor (descriptive) | -0.010851 | [-0.016577, -0.004743] | -0.000640 / -0.021062 | -0.013177 | +0.021030 | -0.013061 | 0.589 |

The frozen `SELF_SUPERVISED_PRETRAINING_USEFUL` rule fires on every condition: pretext beats the
marginal prior, both seed deltas negative, seed-mean `≤ -0.005`, ordinary molecules `≤ 0`,
top-10-removed `< 0`, bootstrap CI95 upper `< 0`, no leakage/protocol deviation, encoders actually
changed and fine-tuning started from the pretrained state (see §6). Fit-vs-locked gaps
(`fit_minus_locked`: B `-0.064/-0.052`, P `-0.048/-0.064`, C `-0.068/-0.065`) show the effect is not
a fit-quality artifact: P fits Fit8k *less* tightly than C yet generalizes better.

## 5. Compute control

The frozen monitor gate fired (`compute_control_gate.json`, `PURCHASED`, 7/7 checks: seed-mean
`-0.010851 ≤ -0.005`; per-seed `-0.000640 / -0.021062 < 0`; ordinary `-0.013177 ≤ 0`; top-10
removed `-0.013061 < 0`; CI95 upper `-0.004743 < 0`; no leakage; encoders changed and used). C was
then trained (22680 updates each, 360 epochs, walls 599 s / 568 s) with the same init and the same
batch-order prefix as B (`initialization_audit.json`: `C_same_init_as_B = true`,
`C_batch_order_prefix_matches_B = true` for both seeds).

C alone buys `-0.0043` locked (roughly a third of the pretraining effect) for +50% supervised
compute; **P still beats C by `-0.0077`**, so the pretraining benefit is not explained by extra
optimizer steps. (C is an extra-supervision control, not a FLOP-matched one — reported as such.)

## 6. Initialization / discipline evidence

`initialization_audit.json` per seed: B and P share the exact initial full-model hash, composer/
reader init hash and batch-order hash; P's fine-tuning started from the *frozen* pretrained encoder
state (`encoder_used_is_pretrained = true`, `encoder_moved_by_pretraining = true`); C shares B's
init and batch-order prefix. Batch orders use `torch.Generator(seed+101)` with identical
`torch.randperm` streams for every arm. `pretrain_*` JSONs carry `labels_read: false`; the
pretraining code path is additionally AST-scanned for `.y`/cache access by correctness gate 17.

## 7. Mechanism / diagnosis

`transfer_mechanism.json` (locked structures, label-blind):

* **Parameters moved far**: relative B→P move atom `1.26/1.24`, bond `1.48/1.48`, junction
  `2.07/2.07` (seed 0/1) — the pretext task reshapes all three encoders, most strongly the
  junction encoder.
* **Junction representation geometry**: effective rank of junction encodings rises from `5.0/5.0`
  (B) to `21.4/18.2` (P); atom effective rank barely changes (`12.6/13.5` → `13.2/13.9`); P atom
  activations are ~1.8× larger in scale and correlate only 0.23–0.28 with B.
* **Subgroups** (locked): gains are broad — ring-rich molecules (`n=996`) `-0.0073/-0.0169`,
  ordinary `-0.0067/-0.0202`; junction-rich (`n=497`) is mixed (`+0.0006` seed 0, `-0.0183`
  seed 1); rare-atom-env (`n=11`) mixed; rare-bond-env had `n=0` under the frozen definition.
* **Caveat — the extreme tail is not improved**: tail molecules (`y≤-3`, n=71) move `+0.0075`
  seed-mean (seed 0 `-0.0158`, seed 1 `+0.0307`), and P's locked RMSE (0.1842) is worse than B's
  (0.1663) even though MAE, median and p90 all improve. The benefit is a bulk/ordinary-molecule
  effect.

## 8. Answers to the eight required questions

1. **Mask leak-free?** Yes — 22/22 gates, swap invariance 400/400 randomized hidden-truth swaps,
   100/100 rebuild cases, `MASK_TARGET_LEAKAGE_BLOCKED` never triggered; all inputs to the model are
   functions of the masked raw graph only.
2. **Pretext beats trivial prior?** Yes — atom CE 0.199–0.210 vs 1.118 nats, bond 0.015–0.019 vs
   0.605 nats; bond task is near ceiling (acc 0.995) but non-leaking the atom task carries the
   learning signal.
3. **Improves raw-y generalization?** Yes — locked seed-mean `P − B = -0.011984` (CI95
   `[-0.017530, -0.005974]`).
4. **Exceeds compute control?** Yes — `P − C = -0.007713` (CI95 `[-0.013252, -0.001911]`); C alone
   `-0.004271`.
5. **Cross-seed?** Both seeds improve (locked `-0.007380`, `-0.016588`; monitor `-0.000640`,
   `-0.021062`).
6. **Ordinary molecules?** `-0.013472` locked (n=929).
7. **Extreme tail?** Not improved: `+0.007483` locked (n=71), seed-mixed; RMSE worse.
8. **Few outliers / top-10 removed?** Improves: `-0.013204`.

## 9. Recommendation — exactly one next route

**Route A: data-efficiency / external unlabelled structures.** The round shows that the *same* 8k
molecules carry a usable label-free structural signal once the encoders see masked raw graphs, and
that the gain survives a 50%-extra-supervision control; the historical bottleneck round shows the
plateau is dominated by labelled-sample quantity/utilisation. The highest-value next step is
therefore to scale the *now-validated, frozen, strictly leak-free* masked pretraining to a large
external unlabelled structure corpus (or a much larger unlabelled pool), keeping this round's exact
split, gate and locked-read discipline, with the primary contrast `P-external − P-internal` at
matched pretraining updates and the same B baseline. Route B (new molecule-level composition
architecture) has the weaker prior: junction ownership was already seed-unstable
(`O8 − N8 = -0.006419`, seed 1 `+0.0009`), and this round's junction-rich subgroup shows the same
seed-instability, while the pretraining route delivers a larger, control-surviving effect with no
architecture change.

## 10. Non-claims and limitations

Internal train-only split; no official-test extrapolation; single locked read; two seeds; tail
molecules and RMSE not improved; C is extra-supervision rather than FLOP-matched; `rare_bond_env`
had no locked rows under the frozen definition; pretraining used only the Fit8k raw structures and
no extra data. Whether the effect grows with external unlabelled data is untested and is the
recommended next route.

## 11. Artifact index (results/bjoc_masked_pretraining_v1/)

`split_a.json`, `objects_manifest.json`, `prep_fit8k.json`, `training_manifest.json`
(+`training_manifest_at_phase_f.json`), `masked_feature_dependency_audit.json`,
`mask_target_swap_audit.json`, `masked_object_rebuild_audit.json`, `pretext_leakage_gates.json`,
`pretraining_feasibility_audit.json`, `pretraining_seed{0,1}.json`, `pretrain_encoder_s{0,1}.pt`,
`train_{B,P,C}_seed{0,1}.json`, `soup_{B,P,C}_seed{0,1}.pt`, `initialization_audit.json`,
`scratch_vs_pretrained_monitor.json`, `compute_control_gate.json`, `monitor_metrics.json`,
`predictions/pred_{monitor,locked}_{B,P,C}{0,1}.npz`, `locked_prediction_manifest.json`,
`locked_read.lock`, `locked_y_read_once.npy`, `locked_metrics.json`, `paired_bootstrap.json`,
`transfer_mechanism.json`, `scientific_decision.json`. All result JSONs carry
`official_valid_loaded: false` and `official_test_loaded: false`.
