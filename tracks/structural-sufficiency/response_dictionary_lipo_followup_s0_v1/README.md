# response_dictionary_lipo_followup_s0_v1

Lipo follow-up, seed 0, official splits. Two decisions and one read-only
replay:

1. **Is the RESP advantage specific to the product form?** One new formal unit,
   `JOINT`: the same parameters, the same canonical initial state and the same
   recipe as `RESP`, with the fusion operator replaced by a bias-free
   `concat(z_S,z_C) -> 16 (tanh) -> 1` joint MLP
   (`T = sum_j w_j tanh((U z_S + V z_C)_j/sqrt(2)) / sqrt(16)`).
2. **Does response-based word selection survive stronger compression?** A frozen,
   post-hoc stress test on the parent Lipo best-Val RESP (epoch 63): only the
   three `tau` classes with >64 Train candidates drop from K=64 to K=16; all
   three selectors and the frozen projection are reused.
3. **How much independent held-out edit support exists?** Read-only replay of
   the existing 823-pair catalogue with official splits (791 Train–Train,
   5 Train–Valid, 27 Train–Test), including the follow-up `JOINT` arm.

Design: [`PROTOCOL.md`](PROTOCOL.md) (frozen before the formal run).
Results: [`REPORT.md`](REPORT.md). Parent round:
`../response_dictionary_small_s0_v1/` (read-only, commit `185db246`).

## Layout

```
code/fu_common.py       paths to the parent round (read-only), follow-up config, protocol hash
code/fu_model.py        JOINT fusion + mode dispatch; same parameter set as RESP/ADD
code/fu_train.py        the parent training loop with an injected model factory (1-epoch check)
code/fu_joint_run.py    the single formal unit: pre-flight, 360 epochs, artefacts
code/fu_checks.py       parameter / init / w=0 / data / order / batch / smoke / loop-reproduction
code/fu_dict_k16.py     parent replay, K=16 dictionaries, fidelity, alpha, response dims, k-panel probe
code/fu_edits.py       823-pair edit replay on official rows and saved full F(G)
code/fu_analyze.py      tables, curves, paired bootstrap, labels
configs/fu_s0.json      follow-up decisions and expected identities
results/lipo_JOINT/     JOINT curve, checkpoints, predictions, preflight, summary
results/lipo_RESP/dictionary_stress_k16/   K=16 tables (parent weights, read-only)
analysis/               edit-replay tables
```

Heavy artefacts (`*.pt`, `*.npz`, `*.npy`, `run.log`, `*.png`) are not
committed; compact tables, manifests and the report are.

## Commands

```bash
# the single formal unit (via rr on res, pool res-gpu1; runs pre-flight first)
uv run python code/fu_joint_run.py --device cuda:0

# frozen dictionary stress test (CPU; reads the parent checkpoint)
uv run python code/fu_dict_k16.py

# edit replay (needs the JOINT predictions)
uv run python code/fu_edits.py

# aggregate
uv run python code/fu_analyze.py
```

`FU_RESULTS_DIR` overrides the follow-up result root (used for throw-away smoke
runs). Test is read under `test_policy: always` and reported as
`DEVELOPMENT_TEST`; it never selects a checkpoint, dictionary or recipe.
