# LCRC-S3 v1 — explicit three-hop structure, and a best-Validation selection protocol

One question: does a **one-shot explicit three-bond path (S3) encoding** add
predictive value to W2 (`LCRC-A-Wide-2x`, 994,324 parameters, no iterative
message passing) **beyond a near-parameter-matched FFN control**?

Two arms, from scratch, same seed, same batch plan, same recipe:

| arm | new module | new parameters | total |
|---|---|---:|---:|
| `S3` = LCRC-S3 | factored per-path encoder + rooted aggregation | 24,208 | 1,018,532 |
| `F` = LCRC-W2-LocalResidualFFN | `LayerNorm→Linear(128,93)→SiLU→Linear(93,128)` | 24,285 | 1,018,609 |

Reference: W2 with best Valid AP `0.2652733161` (epoch 24) and AP@40
`0.2618669181`, used **read-only** (checkpoint replay verified in
`analysis/w2_best_replay.json`).

The protocol, including the frozen decision rules, is in `PROTOCOL.md`. The
result and its scope are in `REPORT.md`.

## What is new here

1. **The S3 structure.** Every *simple* three-bond bond path `(r,u,v,x)` over
   four distinct atoms, with ordered bond types, two intermediate atom types,
   and `dist(r,x) ∈ {1,2,3}` — no walks, no dedup, no truncation, no sampling,
   no message passing. `analysis/structure_novelty.json` shows, exactly and
   label-free, that 93.5 % of atoms sit in a two-ball class that splits into
   several non-isomorphic three-ball classes (6 verified witness pairs + a
   synthetic cyclohexane/n-hexane witness), so S3 is *not* a re-encoding of the
   two-hop interface.
2. **A best-Validation model-selection upgrade.** Every epoch 1…40 is scored on
   official Validation; `e* = argmax AP_valid` (earliest wins ties); the
   selected checkpoint, its SHA-256, the full curve and the epoch-40 endpoint
   are all recorded. The historical W2 number is used exactly as recorded and
   is re-verified by replay, instead of being mixed with epoch-40 numbers.
3. **A capacity control that is not a straw man.** `F` is sized to within
   0.318 % of the S3 branch by an explicit parameter identity
   (`Δ(H) = 257H + 384`, `H = 93`) and sees only `h_raw`.

## Reproducing

CPU-only (structure, cost projection, gates; no model training):

```bash
uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_s3_v1/code/s3_audit.py \
    --modes structure,cache --device cpu
uv run --no-sync pytest -q tracks/structural-sufficiency/lcrc_pcba_s3_v1/tests
```

GPU (only through `rr`, host `res`; see `skills/remote-research-runner`):

```bash
rr doctor res
rr deploy res
# audit job (builds the S3 cache, then preflight/init/architecture/cost/profile/replay)
rr run res s3-audit --pool res-gpu1 \
    --result tracks/structural-sufficiency/lcrc_pcba_s3_v1/analysis -- \
    python tracks/structural-sufficiency/lcrc_pcba_s3_v1/code/s3_audit.py \
    --modes gpu-preflight,cache,init,architecture,cost,profile,w2-replay --device cuda
# formal training (two arms, one GPU each)
rr run res s3-S3 --pool res-gpu1 --result tracks/structural-sufficiency/lcrc_pcba_s3_v1/runs/S3 -- \
    python tracks/structural-sufficiency/lcrc_pcba_s3_v1/code/s3_train.py --model S3 --device cuda \
    --out-dir tracks/structural-sufficiency/lcrc_pcba_s3_v1/runs/S3 --eval-every 1 \
    --soft-seconds 14400 --hard-seconds 21600
rr run res s3-F --pool res-gpu0 --result tracks/structural-sufficiency/lcrc_pcba_s3_v1/runs/F -- \
    python tracks/structural-sufficiency/lcrc_pcba_s3_v1/code/s3_train.py --model F --device cuda \
    --out-dir tracks/structural-sufficiency/lcrc_pcba_s3_v1/runs/F --eval-every 1 \
    --soft-seconds 14400 --hard-seconds 21600
# comparison, decision, ledger
uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_s3_v1/code/s3_compare.py
```

The trainer rebuilds the S3 cache automatically if `runs/s3_cache/` is absent
(deterministic, Train ∪ Valid only, ≈110 s) and verifies the cache identity
before the first step.

## Code map

| file | role |
|---|---|
| `code/s3_paths.py` | torch-free exact S3 enumeration, closed-form count, rooted balls, exact VF2 ball isomorphism, path-type keys |
| `code/s3_common.py` | frozen constants (W2 identity, parameter arithmetic, gates, decision rules), S3 cache build/load, `collate_with_s3` = frozen `pc_data.collate` + S3 tensors, scope/isolation guards |
| `code/s3_model.py` | `S3Branch`, `FFNBranch`, `S3Model`, `FModel`, bitwise-W2 `build_model`, private-generator init, call counting, FLOPs |
| `code/s3_train.py` | 40-epoch trainer with per-epoch Validation, best-Validation selection, resumable checkpoints, curve/per-task CSVs, run record |
| `code/s3_audit.py` | stage-0 structure/cost/novelty audits + GPU preflight/init/architecture/cost/profile/W2 replay/ablation |
| `code/s3_compare.py` | selected records, `D_best_S3` / `D_best_F` / `D_mech`, per-task strata, decision label, GPU ledger |
| `tests/` | 46 CPU tests: definition invariants, parameter arithmetic, init parity, aggregation math, selection rule, checkpoint round-trip, decision labels |

## Non-goals (this round)

No multi-seed run, no S4/S5, no iterative message passing, no Test access, no
early stopping, no hyper-parameter search, no architecture comparison beyond the
two registered arms.
