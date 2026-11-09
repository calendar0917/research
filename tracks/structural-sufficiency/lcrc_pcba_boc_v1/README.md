# LCRC-BOC v1 — bond objects + ring systems composed into the atom representation

Round `lcrc-pcba-boc-v1` in `tracks/structural-sufficiency`.
Pre-registered design: [`PROTOCOL.md`](PROTOCOL.md). Result: [`REPORT.md`](REPORT.md).

The round asks whether a **one-shot, learnable joint composition** of
bond-centred local instances and ring-system instances, injected into the W2
**atom representation**, improves official `ogbg-molpcba` scaffold-split AP.
It is not a logit-residual experiment: object encoders see only projected atom
and bond codes plus static structure; there is no iteration or message passing.

Two arms, identical modules and parameter count (1,352,596):

| arm | ring content |
|---|---|
| `FULL` | real ring-system composition |
| `BOND` | exact zeros; ring modules instantiated but bypassed |

## Layout

```
code/
  boc_common.py       frozen constants, historical identities, IO helpers
  boc_structures.py   label-free packed structures + vectorised collate
  boc_model.py        W2 trunk + bond/ring objects + one fusion MLP
  boc_train.py        inherited train_step, 40-epoch driver, checkpoints
  boc_audit.py        CPU structural audit + small-batch model audit
  boc_profile.py      real-batch GPU profiling / 40-epoch projection
  boc_test.py         Validation replay, Test-logit freeze, post-freeze score
  boc_compare.py      comparison tables / decision-band analysis (post-score)
  boc_zeroing.py      Valid-only channel zeroing sensitivity (post-training)
tests/
  test_structure.py   hand-built graphs, bridges, rings, cache roundtrip
  test_model.py       init, invariances, no-recursion, gradients, BOND zeros
analysis/             audit/profile/selection/freeze/score JSON + tables
runs/                 gitignored checkpoints, curves, caches (regenerated)
```

## Reproduction order

Everything is label-free until step 6.

```bash
# 1. full structural cache (CPU; ~4 min, 1.65 GB)
uv run python code/boc_structures.py --out runs/structure_cache

# 2. audits (CPU structural + GPU small-batch)
uv run python code/boc_audit.py structural --sample-rows 2000
uv run python code/boc_audit.py model --device cuda

# 3. real-batch GPU profile (no training)
uv run python code/boc_profile.py --device cuda --arms FULL BOND W2

# 4. formal runs (one seed, 40 epochs, same order plan)
uv run python code/boc_train.py --model FULL --device cuda \
  --out-dir runs/FULL --soft-seconds 18000 --hard-seconds 21600
uv run python code/boc_train.py --model BOND --device cuda \
  --out-dir runs/BOND --soft-seconds 18000 --hard-seconds 21600

# 5. Validation-only checkpoint manifest
uv run python -c 'import boc_test as T; T.build_selection()'

# 6. freeze Test logits, then open labels once
uv run python code/boc_test.py freeze --device cuda
uv run python code/boc_test.py score

# 7. post-freeze analysis
uv run python code/boc_zeroing.py
uv run python code/boc_compare.py
```

On `res`, all GPU steps go through `rr` (GPU1 = FULL, GPU0 = BOND only when
free):

```bash
rr run res boc-build --pool res-cpu -- uv run python code/boc_structures.py --out runs/structure_cache
rr run res boc-audit --pool res-gpu1 -- uv run python code/boc_audit.py model --device cuda
rr run res boc-profile --pool res-gpu1 -- uv run python code/boc_profile.py --device cuda
rr run res boc-full --pool res-gpu1 --result runs/FULL -- uv run python code/boc_train.py --model FULL ...
rr run res boc-bond --pool res-gpu0 --result runs/BOND -- uv run python code/boc_train.py --model BOND ...
```

`boc_train.py` refuses to start unless `analysis/structure_audit.json` and
`analysis/model_audit.json` pass and bind the loaded cache hash.

## Key invariants

* W2 seed-0 **untrained** initialisation is bitwise identical; no trained W2
  weights are loaded.
* Same Train-only normalisation, batch order, seed 0, 40-epoch recipe and
  official masked BCE/AP as the historical W2.
* Fusion output is not zero-initialised.
* No object encoder reads `h0`/`h`; one `_encode`, one `local`, one fusion MLP.
* `FULL(ring_zero=True) == BOND` up to floating tolerance; BOND ring tensors are
  exactly zero.
* Test targets are read only by `boc_test.py score`, after the prediction
  freeze; Test is `EXPOSED_TEST_ENGINEERING_COMPARISON_SET`.

## Historical references

W2: 994,324 params, best Valid 0.265273 (epoch 24), Test 0.253616, 1.3949 GPU h.
RC-REAL: 1,051,022 params, best Valid 0.267718 (epoch 33), Test 0.255941,
1.9186 GPU h.  Decision bands: ≥0.270 strong, 0.265–0.270 meaningful,
0.258–0.265 small, ≤0.258 no evidence.

## Pre-formal corrections

Two implementation bugs were found by the pre-training tests and fixed before
any formal run: the batch pair-index offset (must use the bond's neighbour-block
start, not the molecule's), and the reverse bond orientation's side counts
(must swap `nL/nR`, `nLL/nRR` for `z_vu`).  Both are covered by the invariance
tests and do not change the frozen architecture or recipe.
