# PEC-C1 — implementation / protocol note

Companion to [`pec_c1_preregistration.md`](pec_c1_preregistration.md).
Round `pec_c1` · study `zinc-context-gap`.

## Files

| path | role |
|---|---|
| `experiments/luyin16/zinc_pec_c1.py` | PEC-C1 runner (stages `cache dicts smoke train mechanism report all`) |
| `tests/test_pec_c1.py` | 29 targeted tests (the ten §11 checks + freeze + gate semantics) |
| `experiments/luyin16/pec_v0.py` | **unchanged** — PEC-v0 architecture, imported as-is |
| `experiments/luyin16/pec_v0_gate0.py` | **unchanged** — PEC-v0 correctness checks, re-used |

PEC-v0's source files are **not modified** by this round.

## What is reused verbatim

The runner imports `pec_v0` and therefore inherits, unmodified: the radius-2
patch, the FSAR explicit rooted occurrence bases (`b^V ∈ R^11`, `b^E ∈ R^15`),
shell / shellpair anchors, the 28/4 chemical primitives, the 18-D pure-topology
pair relation, the read-only static pair composer, the unary/pair pooling, the
reader, the tied-IHT encoder (10 steps, deterministic power-iteration step,
unit-normalized atoms, exact top-`s`), and `K_V = K_E = 16`, `s_V = s_E = 4`.

## What PEC-C1 adds

1. **Full-data regime** — dictionary fit on all 10,000 official-train
   molecules; training on all 10,000; selection on the 1,000 official-valid
   molecules.
2. **Genuine dictionary freezing** (pre-registration D1).  PEC-v0's
   implementation passed `model.parameters()` to Adam, which included
   `d_node`/`d_edge`, so the "frozen" K-SVD dictionary was in fact updated by
   task gradients.  PEC-C1 calls

   ```python
   model.d_node.requires_grad_(False)
   model.d_edge.requires_grad_(False)
   params = [p for p in model.parameters() if p.requires_grad]
   optimizer = torch.optim.Adam(params, ...)
   ```

   and *checks* the invariant at runtime: `role_drift_after_training` and
   `role_drift_after_soup` must be exactly `0.0`, otherwise the run aborts.
   `optimizer_trainable_params` is recorded, and a leak of a frozen role
   parameter into the optimizer list is a hard error.
3. **Frozen 240-epoch schedule** with no early stopping; a non-`smoke` run with
   any epoch count other than `240` raises.
4. **Official-valid checkpoint + Top-5 prediction soup** selection.
5. **Three evaluation-only mechanism interventions** on the CK Top-5 soup:
   chemistry-placement shuffle, neutral dictionary, composition relation
   shuffle.
6. **A guarded loader** — `_guard_split` raises for any split in
   `FORBIDDEN_SPLITS = ("test",)`, and it is the single entry point to ZINC.

## Declared deviations (see pre-registration §3)

* **D1** — CK's dictionary is frozen (`role_trainable = 0`).
* **D2** — CD's dense role map stays exactly PEC-v0's trainable
  `Linear(11→16)` / `Linear(15→16)`, initialized from `Dᵀ`
  (`role_trainable = 416`).

Both arms report `total = 94,049`.  CK is trainable `93,633`; CD is trainable
`94,049`; the difference is exactly the 416 role parameters.  The comparison is
**conservative against CK** and a Case-B verdict carries a confound note.

## Local validation already performed

* 41 targeted tests pass (`test_pec_c1.py` + `test_pec_v0.py`).
* `cache` stage: 10,000 train / 1,000 valid, 98.7 s, `official_test_loaded=false`.
* `smoke` stage (128 train / 64 valid, 2 epochs, CPU) for both arms:
  * CK `total 94,049 / trainable 93,633 / role_trainable 0`, drift after
    training and after soup **`0.0`**
  * CD `total 94,049 / trainable 94,049 / role_trainable 416`, drift
    `0.00394` after 2 epochs on 128 molecules (confirms the dense map is live)

## Remote execution

```bash
# 1. build the round's own cache and fit the two dictionaries (CPU, train-only)
python -m tracks.ksvd.experiments.luyin16.zinc_pec_c1 cache
python -m tracks.ksvd.experiments.luyin16.zinc_pec_c1 dicts

# 2. GPU smoke on GPU0 (timing + CUDA backward + parity), short
python -m tracks.ksvd.experiments.luyin16.zinc_pec_c1 smoke \
    --device cuda:0 --epochs 2 --smoke-train 512 --smoke-valid 256

# 3. two formal seed-0 arms, one per GPU, followed to completion
python -m tracks.ksvd.experiments.luyin16.zinc_pec_c1 train --arm CK --device cuda:0
python -m tracks.ksvd.experiments.luyin16.zinc_pec_c1 train --arm CD --device cuda:1

# 4. frozen decision
python -m tracks.ksvd.experiments.luyin16.zinc_pec_c1 report
```

`cache` and `dicts` are resumable (they skip work whose output already exists),
so a relaunch after an interruption continues instead of restarting.

Artifacts land in `tracks/ksvd/results/pec_c1/`:
`cache/`, `dicts/`, `states/`, `{CK,CD}_seed0.json`, `{CK,CD}_seed0_curve.csv`,
`{CK,CD}_seed0_best.pt`, `{CK,CD}_seed0_top5.pt`,
`{CK,CD}_seed0_{soup,best}_predictions.npy`, `pec_c1_decision.json`,
`REPORT.md`.
