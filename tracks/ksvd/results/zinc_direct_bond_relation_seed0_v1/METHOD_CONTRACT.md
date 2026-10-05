# METHOD_CONTRACT — `zinc_direct_bond_relation_seed0_v1`

This contract freezes implementation semantics, provenance, evaluation and outputs for the one O/T round. Read with `PROTOCOL.md`; no score-driven edits are allowed. Protocol, contract, source, runner and manifests must be committed before the two formal jobs.

## Model and state identity

* Base model is the exact frozen fresh-fold source M skeleton (`M_init_state.pt`, 297,499 params); the sole architecture change is the required first relation Linear expansion `15→19` into unchanged 96 hidden units.
* Construct with the exact source M seed-0 constructor, including its `joint` local tuple and `sum` pooling; preserve the resulting RNG state when replacing the first Linear. Then copy all source init state exactly, append 4 zero columns to `relation_encoder.layers.0.weight`, and preserve its original bias. Do not initialize new columns randomly or use a trained source soup.
* O and T have identical expanded state dicts and 297,883 parameters. O explicitly concatenates four zeros; T concatenates verified beta. Source, O and T initial outputs must match within `1e-5` on a fixed fit batch.
* Input uses exactly `P1_RELATION_INDICES = 0:14 ∪ {18}` plus beta. The path-count mask remains. Raw `pair_relation[:,14:18]` (`path_bond_mean`) never enters the model. Global atom/bond histogram mask and all other C6 semantics are unchanged.
* Beta is primitive unstandardized `[n_pairs,4]` float32. Construct from `pair_relation[:,19:23]`, then independently derive from raw official-train `edge_index/edge_attr` for all 10,000 training graphs and assert exact equality. Direct edge type has same code under both endpoint orientations. No labels are consulted to construct beta.
* Add `pair_beta` as a per-graph tensor before the existing PyG `env_collate`; PyG concatenates pair-aligned fields without node offset. Check batched concatenation, per-graph/single evaluation, stable global graph IDs, edge orientation, and reverse ordering. Do not infer beta from the batched global pair index without local graph offsets.
* The actual forward is a copy of `audit.AuditModel.forward`, differing solely by concatenating beta after the masked 15-D relation and before the existing relation encoder. Environment generation still depends on the original frozen local environment path, not beta; beta flows once through static pair composition and existing graph aggregation, no write-back or messages.

## Data, fit and target

* Source objects: `tracks/ksvd/results/zinc_local_tuple_fresh_fold_replication_seed0_v1/`; source model training lineage `a5400de`, preregistered fold objects `2e4ee3f`.
* Fold SHA256: fit `2a21cb8771f6e24cfb4a5cc50602cf0b4390db9c4367f8bb910052f9f0aebbcb`, dev `270ab4126b0f0413f1f6ff7ada2e9914bbaca91ae50cc65bbd8ca2673d8f9357`.
* Position schedule SHA256 `7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65`; global train graph-ID stream SHA256 `69187f13fba5def82ee2f28765c1c46ac4129844d3f76ac769e018d18ae097d2`.
* Load `zftd.load_train_only()` only (encoded train + env train); explicitly do not call generic split loaders or instantiate valid/test. Apply and verify the frozen fresh-fold fit-only preprocessing to the train cache before either arm. Target `g=y-c` and its fit-only constants come from frozen `fresh_targets.npz`.
* The held-out internal dev is part of official train and reused in previous research. Describe it as a reused internal partition, not an independent test. Do not access any official-valid/test molecule, label, cache, prediction or score.

## Formal recipe and execution

* seed 0, FP32, 240 epochs, batch 128, exactly 15,120 steps, Adam `lr=1e-3`, coupled `weight_decay=1e-5`, global grad clip `5.0`, L1 loss on identical frozen g targets; state soup is arithmetic mean of raw states from epochs 236–240.
* Same private schedule and exact source construction/training RNG streams in both arms. Any fit diagnostic, evaluator, bootstrap or report code must not consume/perturb the trajectory RNG.
* Only two formal runs O and T. Remote host/pool `res-2`/`res2-cu124` on c05/c06 A100 regime, same driver/Torch/CUDA regime, separate physical GPU UUIDs when concurrent. FP32, no AMP/DDP, per process max 4 threads; no more than 2 GPUs. `rr` is the only remote control plane.
* Total GPU allocation time ≤0.8 GPU-hours and wall time ≤120 min from start. No new compute after minute 90. One pre-formal fit-only GPU smoke is state-discarded. Every self-launched job must finish/cancel to a terminal state.
* Record `rr` run ID, Slurm job/allocation, node, driver, GPU UUID and PCI bus ID, Python, Torch/CUDA, CUDA visibility, per-arm wall/GPU timing, source/final commit and artifact checksums. A run on a different physical GPU is valid only if the GPU regime itself matches and is recorded.

## Checks and failures

Before formal jobs, require all seven protocol checks: source/O/T init equivalence; raw beta verification plus one-hot/non-edge/reverse/batching/order checks; actual relation input width/content and path-bond-mean exclusion; task backward (T new-column grad nonzero, O exactly zero); environment invariance/no write-back and exactly-once hook counts; real 19-D path in train/eval/replay; fixed schedule/calibration/paired-bootstrap/replay invariants. Smoke states/weights are discarded.

Any data identity, fold, target, preprocessing, init, schedule, global graph-ID stream, relation input, RNG, nonfinite training, incomplete step count, replay, resource, or execution-regime failure is recorded as `IMPLEMENTATION_FAILED` or `NOT_COMPLETED`; do not repair the condition after examining dev scores, add a trajectory, or issue a performance verdict. Small log/analysis corrections after training may be recorded as deviations but cannot affect inputs, trajectories, gate or classification.

## Evaluation code contract

* Only the arithmetic raw-soup checkpoint is primary. Compute one per-arm bias from fit only: `median(g_fit - raw_soup_fit_pred)`. Apply additively to raw soup predictions; no dev bias or labels in wrapper/calibrator.
* Gain sign is O error minus T error. Primary is dev k=0 calibrated MAE; mandatory companion is overall dev calibrated MAE. Gate: both gains ≥0.003 and G0 paired bootstrap 95% CI lower >0. Bootstrap 1000 draws, seed 20261007, same sampled rows across arms within each draw, stratified by endpoint population; zero and swapped-arm witnesses mandatory.
* Raw endpoints are reported and can qualify calibration dependence but are not a hard veto. Report calibration accounting identity, four k groups (n/MAE/sum abs/contribution), per-arm raw-to-cal change, fit/dev gap, and fixed largest-O-error deletion sensitivity. Do not call this identity causal decomposition.
* Practical equivalence requires both G0 and overall calibrated CIs wholly within ±0.003. Positive point values without gate remain `DIRECTIONAL_NOT_CONFIRMED`; no extra compute follows.
* Optional adjacent-beta-to-graph-marginal diagnostic is gated on the formal T candidate pass. It is one label-free forward intervention on T raw soup, no training. Report it as a distinction between native pair-specific compatibility and graph-marginal-compatible behavior—not as evidence of a trained marginal control or dictionary mechanism.
* `g` is an internal residual/chemical component `y-c`, not official ZINC y. The historical ~0.09 internal target is not official benchmark performance. No official valid/test data may be loaded.

## Outputs and single-file replay

Under `tracks/ksvd/results/zinc_direct_bond_relation_seed0_v1/`:

`PROTOCOL.md`, `METHOD_CONTRACT.md`, `REPORT.md`, `DECISION.md`, `EXECUTION.md`; frozen input copies/manifests and beta provenance; `input_checks.json`, `smoke_checks.json`; O/T init/last/raw-soup states, complete curves, probes, metadata, GPU runtime and raw fit/dev predictions; fit calibration biases; `analysis.json`, `gains.json`, `gate.json`, `main_table.csv`, `group_table.csv`, `group_gain_table.csv`, `per_graph_dev.csv`, replay checks and artifact manifest. If T passes only: native-vs-graph-marginal diagnostic artifacts and a compute-purchase design note; if T does not pass the gate, no marginal diagnostic is run and only a brief no-purchase note is written.

Commands (from repository root):

```bash
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --prepare
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --pre-checks
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --smoke --device cuda
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --train --arm O --device cuda
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --train --arm T --device cuda
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --analyze
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --replay
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --manifest
```

Remote `rr run` must invoke only one `--train --arm {O,T}` command per job and declare this result directory for pull. Analysis/replay occur locally after both terminal jobs are pulled; they do not train.
