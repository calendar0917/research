# LCRC-RHIC v1 — rooted hierarchical co-occurrence composition

Round: `lcrc-pcba-rhic-v1` (track: `structural-sufficiency`)

One model, one seed, 40 epochs. `LCRC-RHIC` is the frozen historical LCRC-W2
seed-0 model with a single, one-shot residual inserted between
`drop(local(...))` and `post_local`:

```python
a, b, scal = base._encode(batch)
h0 = base.drop(base.local(batch, a, b, scal))
delta = RHIC(h0, b, path_index, bond_index)
h = h0 + delta
logits = base.post_local(h, batch)
```

CPC-v1's rooted two-bond path tokens are kept verbatim, then grouped in two
levels: inside a directed first-hop branch `(i, j)` the module keeps the mean
token and the pairwise co-occurrence mean of **distinct** tokens; across the
branches of one root it keeps the mean branch value and the pairwise
co-occurrence mean of **distinct** branch values. `q_b` and `cross_branch` are
zero for `n < 2`; roots with no branches get `delta = 0`; the final
`Linear(192,128)` is zero-initialised so epoch-0 logits equal the untrained W2.

* **Parameters: 1,117,253** = W2 994,324 + RHIC 122,929
  (token 40,561 + branch 26,368 + root 56,000).
* `PROTOCOL.md` — frozen pre-registration (architecture, recipe, audits, Test
  flow, decision rules).
* `REPORT.md` — Chinese research report (written after the formal run).

## Layout

```
code/
  rhic_common.py     frozen constants, inherited read-only imports, high-support set
  rhic_branches.py   branch ids, grouping, stable mu/q, shuffle, coverage
  rhic_model.py      RHICModule / RHICModel / init / audits / prediction
  rhic_audit.py      structure + preflight gates
  rhic_train.py      the single formal 40-epoch run
  rhic_test.py       Test freeze + one post-freeze scoring pass
  rhic_analyze.py    Validation-only mechanism diagnostics + decision
tests/               CPU correctness suite (21 tests)
analysis/            committed JSON evidence (structure, preflight, mechanism, …)
runs/                run artifacts (git-ignored, pulled from the GPU host)
```

## Commands

```bash
# CPU correctness suite
uv run pytest tests -q

# structural audits (frozen CPC cache, no labels)
uv run python code/rhic_audit.py --mode structure --device cpu \
  --cache ../lcrc_pcba_cpc_v1/runs/cpc_cache

# GPU preflight: init parity, one-shot hooks, gradients at 3/10/100,
# 100-batch profile, projected GPU-hours
uv run python code/rhic_audit.py --mode preflight --device cuda \
  --cache ../lcrc_pcba_cpc_v1/runs/cpc_cache

# the one formal run (through `rr`, after committing)
uv run python code/rhic_train.py --seed 0 --device cuda \
  --out-dir runs/LCRC_RHIC --soft-seconds 12600 --hard-seconds 14000

# freeze Test predictions, then the single post-freeze scoring pass
uv run python code/rhic_test.py --mode freeze --device cuda
uv run python code/rhic_test.py --mode score

# Validation-only mechanism diagnostics + decision
uv run python code/rhic_analyze.py --mode all --device cuda
```

## Remote execution (single A100, `res`)

From the repository root, with a clean committed worktree:

```bash
rr doctor res
rr deploy res

# 1) build the frozen CPC structure cache in the deployed checkout (~1 min,
#    CPU) and rerun the full structure audit there
rr run res rhic-structure --result tracks/structural-sufficiency/lcrc_pcba_rhic_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_rhic_v1/code/rhic_audit.py \
     --mode structure --device cpu \
     --cache tracks/structural-sufficiency/lcrc_pcba_cpc_v1/runs/cpc_cache

# 2) GPU preflight: epoch-0 parity, one-shot hooks, gradients at 3/10/100,
#    100-batch profile and the 40-epoch projection
rr run res rhic-preflight --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_pcba_rhic_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_rhic_v1/code/rhic_audit.py \
     --mode preflight --device cuda \
     --cache tracks/structural-sufficiency/lcrc_pcba_cpc_v1/runs/cpc_cache
rr pull rhic-preflight

# 3) the one formal 40-epoch run (hard wall 3.83 h, budget cap 4.0 h)
rr run res rhic-train --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_pcba_rhic_v1/runs/LCRC_RHIC \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_rhic_v1/code/rhic_train.py \
     --seed 0 --device cuda --epochs 40 \
     --out-dir tracks/structural-sufficiency/lcrc_pcba_rhic_v1/runs/LCRC_RHIC \
     --soft-seconds 10800 --hard-seconds 13800
rr pull rhic-train
```

Freeze, the single Test scoring pass and the mechanism diagnostics then run
locally on CPU from the pulled checkpoints; the official label file is local.
If the formal run is interrupted, resume only from `checkpoint_last.pt` with the
same `--out-dir` and `--resume`; never change the recipe.

## Artifacts

| Path | Contents |
|---|---|
| `analysis/structure.json` | cache identity, full-cache branch audit, small graphs, batch offsets, independent NumPy stats, witness, Train coverage |
| `analysis/coverage.json` | atom- and molecule-level Train structure coverage |
| `analysis/preflight.json` | epoch-0 parity, architecture hooks/invariances, gradient + cost gates |
| `runs/LCRC_RHIC/run.json` | the single run: curve, fixed epoch reports, hashes, provenance |
| `runs/LCRC_RHIC/checkpoint_best.pt` | best official-Validation checkpoint (SHA-256 pinned) |
| `runs/LCRC_RHIC/checkpoint_epoch40.pt` | epoch-40 endpoint (SHA-256 pinned) |
| `analysis/test_prediction_manifest.json` | sealed Test logits + reference logits, before any label read |
| `analysis/test_scores.json` | official + independent Test AP for all models, high-support table |
| `analysis/mechanism.json` | real / delta-zero / branch-shuffle Validation diagnostics |
| `analysis/decision.json` | pre-registered decision label and per-task evidence |

## Test discipline

Official Test targets are NaN in the training artifact and in every analysis
before the freeze. Selection, checkpoint choice and the mechanism gate use
official Validation only. Test labels are opened once, after the sealed
prediction manifest, through `cc_data.read_official_test_labels`. Test is an
exposed engineering comparison set, never a blind generalisation estimate and
never a model-selection signal.

## Status (round complete)

- [x] architecture, initialisation and parameter audit (CPU)
- [x] structural audits A–E (structure: `all_ok=True`)
- [x] CPU correctness suite (26 tests)
- [x] GPU preflight (`all_ok=True`, epoch-0 CUDA parity 1.9e-6, 24/24 grads, 1.92 h projected)
- [x] formal 40-epoch run: **COMPLETED**, best epoch 25, Valid **0.268734**, 1.918 GPU-h
- [x] Test freeze + single post-freeze scoring pass: Test best **0.258416**
- [x] mechanism diagnostics (real / delta-zero / branch shuffle) + decision
- [x] `REPORT.md` — decision label **`NO_CLEAR_ADVANCE`**

Bottom line: RHIC improves official Validation over CPC (+0.0028) but not Test
(−0.0039); the hierarchical branch co-occurrence is demonstrably used, yet the
extra capacity does not transfer across the official scaffold split.
