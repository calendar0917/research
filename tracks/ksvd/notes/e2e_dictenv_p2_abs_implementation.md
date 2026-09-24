# E2E-DictEnv-P2-ABS — implementation

Round **E2E-DictEnv-P2-ABS** · protocol `e2e_dictenv_p2_abs` · study
`zinc-context-gap` (Workstream Z).
Pre-registration: [`e2e_dictenv_p2_abs_preregistration.md`](e2e_dictenv_p2_abs_preregistration.md).

## 1. Code layout

| file | role |
|---|---|
| `tracks/ksvd/experiments/luyin16/e2e_dictenv_p2_abs.py` | configurable clean dictionary-core model (`P2Config` / `P2Model`), parameter accounting, frozen lambda constants |
| `tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_p2_abs.py` | runner: candidate registry, dictionary fit (K64), lambda calibration, training, stage selection, finalize |
| `tracks/ksvd/tests/test_e2e_dictenv_p2_abs.py` | focused CPU tests |

The runner reads the already-validated **P1 primitive environment cache**
(`results/e2e_dictenv_p1/cache/env_{train,valid}.pt`) read-only; it never
rebuilds it and never touches any P1 result file.

## 2. Model

`P2Model` mirrors `P1Model` exactly at the base configuration
(`decoder='p1'`, `d_e=48`, `K=32`, `s=8`) — the CPU test asserts the two
`state_dict`s are bit-identical for the same seed, and all downstream widths are
shared.

```
dictionary        D in R^{65xK}
node binding      W_A^S (K x 96), W_A^C (28 x 96)
edge binding      W_E^S (3K x d_e), W_E^C (4 x d_e)
decoder           p1: [anchor(62); A(288); E(6*d_e)] -> MLP -> 48
                  h1: shared node 96->64->48, shared edge d_e->48->32,
                      anchor 62->32, fusion 368->128->48
                  h2: shared node 96->96->64, shared edge d_e->64->48,
                      anchor 62->48, fusion 528->96->48
backend           unchanged (exact P1 static composer), 15103 params
```

Coding is always tied-IHT with exact top-`s`, 10 steps.  No dense arm, no
`patch_cont`, no message passing, no recurrence.

Parameter counts (accounting == actual, verified):

| config | decoder | d_e | K | s | params |
|---|---|---|---|---|---|
| Z0 / P1 | p1 | 48 | 32 | 8 | 97,865 |
| Z1/Z2/Z3 | p1 | 48 | 32 | 8 | 97,865 |
| H1 | h1 | 48 | 32 | 8 | 97,487 |
| H2 | h2 | 48 | 32 | 8 | 110,335 |
| E64 (p1) | p1 | 64 | 32 | 8 | 109,257 |
| E64 (h1) | h1 | 64 | 32 | 8 | 99,855 |
| K64s8 (p1) | p1 | 48 | 64 | 8 | 107,625 |
| K64s12 (h2) | h2 | 64 | 64 | 12 | 124,255 |

All are inside the pre-registered `80,000 .. 130,000` budget.

## 3. Frozen lambdas

```
lambda_base(P1) = 135.83492071463948
Z1 = 0.25   * base = 33.95873017865987      horizon 320
Z2 = 0.125  * base = 16.979365089329935     horizon 240
Z3 = 0.0625 * base =  8.489682544664968     horizon 240
```

For `K=64` the reconstruction scale changes, so `lambda_base` is recomputed
under the same rule (`L_task^init / L_rec^init` on the first 512 official-train
graphs with the initial model) and multiplied by the Stage-A chosen factor.  The
calibration is cached in `results/e2e_dictenv_p2_abs/lambda_k64s{s}.json`.

`K64/s8` and `K64/s12` each use a **matched** K-SVD fit (`sdb_v0.fit_ksvd`,
10 epochs, seed `20260924`) on the ZINC official-train `phi65` (231,664 nodes);
the `s8` fit is never reused for `s12`.

## 4. Training protocol

Identical to the frozen P1 protocol except for the pre-registered candidate
axes: seed 0, Adam lr 1e-3, wd 1e-5, batch 128, grad clip 5, no scheduler, fixed
Top-5 soup by official-valid MAE.  Per-candidate output paths:

```
curves/<TAG>_curve.csv   states/<TAG>_raw_state.pt   states/<TAG>_soup_state.pt
results/e2e_dictenv_p2_abs/<TAG>.json
```

## 5. Stage selection

```
stage_a_selection.json   min soup over {Z0, Z1, Z2, Z3}
stage_b_selection.json   min soup over {stage-A winner, H1, H2}
stage_c_selection.json   min soup over {stage-B winner, E64}
stage_d_selection.json   min soup over {stage-C winner, K64S8} (+ K64S12 if
                         K64S8 improves the pre-Stage-D best by >= 0.001)
final_config.json        winning config, hashes, metrics, official_test_loaded=false
```

## 6. Reproduce

```bash
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_p2_abs.py
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_p2_abs correct --device cuda
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_p2_abs smoke --device cuda
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_p2_abs train --candidate Z1 --device cuda
# ... Z2, Z3, select-a, H1, H2, select-b, E64, select-c, K64S8, select-d, finalize
```

Stages are resumable: a candidate whose run JSON exists is skipped.

## 7. Operational record

Every run records commit, device, candidate config, params, dictionary sha256,
lambda, per-epoch curve, wall clock and peak GPU memory.  Official ZINC test is
never loaded (`official_test_loaded = false` everywhere).
