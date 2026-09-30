# BondAnchoredTriple-JointTune-v1 — analysis

Round `e2e_dictenv_bond_anchored_triple_joint_tune_v1` (BATJ-v1, study
`zinc-context-gap`), executed **locally on CPU** (8 threads) on 2026-09-30.  Frozen
specification: `notes/e2e_dictenv_bond_anchored_triple_joint_tune_v1_preregistration.md`
(commit `be2686e`), implementation commit `088b1ab`.  Control plane:
`zinc_bond_anchored_triple_joint_tune_v1_mainline`; run
`20260930-154827-2dc792ad` (status `completed`, mode `scratch`, `test_access:
blocked`, config hash `eaf93c43…2aea`, split fingerprint `58c69506…f28a`).

## Question

With the shared structural coordinates (dictionary `D`/`U`/`common_rms`, C6 mask,
semantic interface, dead-node binding `W_A_S`/`W_A_C`/`node_encoder`,
`global_encoder`, `topology_encoder`, all fixed normalisation buffers) frozen, and
hot-starting from the BAT-v1 M1 soup (`F` + 366-D Reader), does **jointly
fine-tuning representation + composition + three-environment operator + Reader**
reach `M_joint_soup <= 0.120` on official ZINC valid within the fixed seed-0
40-epoch budget?  This is a single-candidate mainline performance screen: not a
readout retrain on old features, not a matched control, not a mechanism test.

## Frozen pieces and hot start

| piece | value |
|---|---|
| Sem108 soup backbone | `results/e2e_dictenv_sem108_v1/checkpoints/SEM108-seed0_soup_state.pt`, full-file sha256 `6ec0fdef…4f89`, canonical state `7a721984…050a` (49 keys, 97 709 params, old 302-D Reader 4 135 params removed) |
| BAT-v1 M1 soup | `results/e2e_dictenv_bond_anchored_triple_v1/checkpoints/BAT-v1-seed0_soup_full_state.pt`, canonical state `f898a595…`, members `[50, 57, 43, 49, 71]`, member valid `[0.124576, 0.125264, 0.125437, 0.125595, 0.125666]` |
| standardisation | `mu_p/scale_p`, `mu_old/scale_old`, `mu_3/scale_3` loaded from the BAT soup; matched against `standardizers.json`; never refit |
| online hot start | `M_start = 0.12300291641423246` — exactly equal to the recorded BAT-v1 soup valid MAE (abs diff `0.0`, tolerance 1e-6) |
| parent replay (background) | `M_parent = 0.123704927947314` (historical unmatched parent soup replay, abs diff `0.0`) |
| frozen subset | unchanged before/after (`frozen_subset_unchanged: true`); both source checkpoint files byte-identical before/after (`source_files_unchanged: true`) |

## Trainable surface (measured)

34 tensors / **82 805 params**, unfrozen in place after freezing everything else:

| group | params |
|---|---:|
| `fusion` (446→114→48 + biases) | 56 478 |
| `pair_encoder` | 5 328 |
| `F` (48→64→32) | 5 216 |
| `edge_binding` (`W_E_S` + `W_E_C`) | 4 944 |
| `reader` (366→13→13→1) | 4 967 |
| `edge_encoder` | 3 920 |
| `relation_encoder` | 1 104 |
| `pair_projection` | 768 |
| `distance_gate` | 80 |

Frozen: 17 tensors / 20 952 params (including `D` 2 080, `W_A_S`/`W_A_C`,
`node_encoder`, `global_encoder`, `topology_encoder`); full registered model
103 757 params.  The removed old 302-D Reader (4 135 params) is not registered
and never enters the optimizer.  Only the exact whitelist is in the optimizer /
clip set (`optimizer_set_equals_whitelist: true`); frozen parameters have
`.grad is None`.

## Online-forward discipline and correctness (all gates passed before training)

- **Structure cache is data-only** (no parent forward, no cached `E`/`p_ij`/
  `z_old` as training input).  Cross-checked against the BAT-v1 cache: identical
  counts and **bit-identical** triple `ij`/`ik`/`jk` rows for both splits
  (train 5 511 568 triples, valid 546 830).
- `structure_audit_train/_valid`: `m*(n-2)` count rule, anchor counts, no
  out-of-range rows, no cross-graph offsets, brute-force enumeration agrees;
  0 zero-tuple graphs.
- `batch_mapping` (32 graphs / 18 129 triples): triple rows stay inside each
  graph's pair range; pair buckets match the BAT cache.
- `online_vs_cached` (32 valid graphs, fresh per-forward online path vs the BAT
  cache): pair token max diff `6.48e-07`, `z_old` max diff `1.91e-06`, `z3` max
  diff `5.96e-08`, final prediction diff `0.0` (tolerance 2e-5).
- `mode_forward`: the whole model keeps every backbone and dropout module in
  eval mode; forward under `train()` equals the eval forward exactly
  (max abs diff `0.0`), so the original backend dropout never activates while
  autograd stays enabled.
- `one_batch_gradient` (128 graphs / 70 669 triples): finite graph-L1 loss
  `0.0392`; all nine unfrozen groups have non-zero gradients (min group grad
  norm `0.0109`: `distance_gate`; largest `reader` `1.0532`, `fusion` `0.5403`);
  frozen grads `None`; grads cleared afterwards.
- `parameter_accounting`: 34 tensors / 82 805 params trainable, 103 757 total,
  old Reader −4 135.
- `hot_start`: `0.12300291641423246` vs expected, abs diff `0.0`.

## Result

| metric | value |
|---|---|
| `M_joint_soup` (Top-5 soup valid MAE, window 21–40) | **0.12315951417008182** |
| `M_start` (online hot start) | `0.12300291641423246` |
| `Delta_vs_start` | **`+0.00015659775584936364`** |
| `M_parent` (background replay) | `0.123704927947314` (`Delta_vs_parent −0.000545413777232176`, background only) |
| best valid, epochs 1–40 | `0.12297315568843624` @ epoch 6 |
| last-10-epoch valid mean (31–40) | `0.12425338722536106` |
| final train MAE | `0.04675897518694401` |
| soup members (21–40) | `[27, 25, 40, 38, 22]`, member valid `[0.123404, 0.123408, 0.123430, 0.123519, 0.123659]` |
| epochs / wall / peak RSS | 40/40 / 405.3 s (10.13 s/epoch) / 2 743.9 MB |
| device | local CPU, 8 threads, float32 |
| verdict | **`JOINT_TUNE_SCREEN_NO_STRONG_SIGNAL`** (0.12316 > 0.120) |

Curve landmarks (full curve in `results/…/curve_seed0.csv`): epoch 1
train 0.05708 / valid 0.123307; epoch 6 0.05337 / **0.122973**; epoch 10 0.05164 /
0.126127; epoch 20 0.04934 / 0.123853; epoch 30 0.04771 / 0.124101; epoch 40
0.04676 / 0.123430.  Train MAE falls ~18 % (0.0571 → 0.0468) while valid MAE
oscillates in the 0.1233–0.1261 band with no improving trend; every epoch in the
soup window is above the hot-start value.

## Reading

1. **The pre-registered question is answered negatively for this candidate.**
   Joint adaptation did not push the predictor to a new level: the fixed Top-5
   soup over epochs 21–40 is `+0.000157` *worse* than the hot start it began
   from and `0.00316` above the frozen 0.120 gate.  The only epoch nominally
   below `M_start` is epoch 6 (`0.122973`, i.e. `−0.000030`), still above the
   gate; the frozen soup window excludes it, and even including it would not
   change the verdict.
2. **The optimiser is using the extra freedom for train-only structure.**  Train
   MAE keeps falling through epoch 40 while valid does not improve past the
   hot-start band, consistent with the jointly adapted representation/composition
   fitting idiosyncratic train structure rather than adding generalising signal
   under this protocol.  This is a one-line observation from the recorded curve,
   not a mechanism conclusion.
3. **What this round does *not* establish.**  Because representation,
   composition, the three-environment operator and the Reader were adapted
   together with no matched control, no shuffle/ablation arm and no mechanism
   arm, the result cannot attribute anything to the triple operator alone, cannot
   attribute the flatness to the frozen dictionary coordinates, and cannot claim
   three-environment mechanism usage, dictionary necessity, or operator
   specificity.  The `Delta_vs_parent = −0.00055` is a background screen
   difference against an unmatched historical replay.
4. **Validity of the negative result.**  The round is complete (40/40 epochs) and
   valid: the online hot start reproduces the BAT-v1 soup exactly, all correctness
   gates pass, the frozen subset and both source checkpoints are byte-identical
   before/after, and the official test split was never instantiated
   (`official_test_loaded: false`, `test_access: blocked`).  The verdict follows
   the frozen rule with no re-thresholding.
5. **No rescue is purchased.**  Per the pre-registration, a positive result would
   have been report-and-stop and a negative result closes the candidate: no seed
   1, no learning-rate or unfreeze-scope change, no initialisation/normalisation
   change, no horizon extension, no M0 retrain, matched control, shuffle,
   ablation, branch-off or mechanism arm off this run.  A next step needs a new
   pre-registration and does not follow automatically.

## Provenance notes

- Device: local CPU only, 8 threads, float32; no remote A100 work was performed
  in this round (per user instruction).
- Execution order: pre-registration committed (`be2686e`) before the
  implementation (`088b1ab`); the formal `research run` executed the full stage
  chain `structure → references → preflight → correctness → train → analysis`
  and wrote `structure_report.json`, `references.json`, `preflight.json`,
  `correctness.json`, `run_seed0.json`, `curve_seed0.csv`, `soup.json`,
  `summary.json`, `REPORT.md`, `DECISION.md` plus the checkpoints
  (`epoch000_hotstart_full_state.pt` excluded from the soup,
  `epoch{027,025,040,038,022}_full_state.pt`,
  `soup_trainable_state.pt`, `soup_full_state.pt`).
- All scientific constants live in the algorithm module; the runner stages only
  orchestrate and record.  Nothing was tuned after observing valid results.
- The run is a single pre-registered trajectory; the result is not a
  statistical-significance statement.

## Recorded next shapes (not scheduled, not authorised here)

- A NEW pre-registration for a mechanism-bearing question, e.g. a matched
  control / liveness design that isolates the three-environment operator or the
  dictionary coordinates at equal parameter and optimiser budget, before any
  accuracy claim.
- Any dictionary-coordinate adaptation round must be pre-registered separately;
  this round deliberately held the coordinates fixed and therefore cannot speak
  to whether they are the binding constraint.
