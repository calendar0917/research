# REPORT — LCRC PCBA structural-gain sprint v1

Round: `lcrc-pcba-structural-gain-sprint-v1`  
Host: `res` (`a100-2`, 2 × A100-SXM4-40GB), driver 550.163.01, torch 2.5.1+cu124, CUDA 12.4, FP32  
Formal runs: 2026-11-05 16:41–21:48 UTC  
Status: **complete**; pre-registered label **`PCBA_025_REPRODUCED`**

## Result in one paragraph

The pre-registered screen promoted exactly one arm: **V = frozen C1 + two MP2
bond-message blocks + per-graph context feedback**, with 42,624 new parameters
(287,808 total). Continued from the frozen epoch-9 checkpoint under the common
40-epoch recipe, V reached **0.252658 Validation AP at the fixed epoch 40**
(control C: 0.238236; `D40 = +0.014421`, 105/127 tasks improved). Trained from
scratch for 40 epochs, V reached **0.257772 (seed 0)** and **0.258952 (seed 1)**
of 127-task official AP (mean **0.258362 ± 0.000590**), against paired controls
0.231945 and 0.226905 (`D = +0.025828` and `+0.032047`; 101/127 and 109/127
tasks improved). Both seeds clear the pre-registered confirmation rule (AP ≥
0.250 and gain ≥ 0.005 over the paired control), so the round is labelled
`PCBA_025_REPRODUCED`. The primary target ≥ 0.250 was met; the stretch target
≥ 0.260 was not (best single run 0.258952). No Test target was scored.

## 1. Question and pre-registration

The frozen C1 (245,184 parameters) encodes each atom locally and predicts
`sum_v f(h_v) + rho(G)`. The sprint asked which of three additions improves
OGB MolPCBA scaffold generalization within a fixed 40-epoch budget:

| Arm | Addition | New parameters |
|---|---|---:|
| C | none (control) | 0 |
| F | two pre-norm residual FFN blocks, per atom | 33,408 |
| M | F + directed physical-bond messages (`bond_proj`) | 34,304 |
| V | M + per-graph mean context fed back to each atom | 42,624 |

Everything was frozen in `PROTOCOL.md` before the first formal epoch: arms,
widths, learning rates, 40-epoch per-step cosine, evaluation epochs, Phase-1
floor `+0.004`, Phase-2 floor `AP ≥ 0.245` / `D40 ≥ 0.006`, Phase-3 double-seed
rule, and stop conditions. No threshold, arm, learning rate or comparison epoch
was changed after seeing any AP.

All arms share the frozen warm start (SHA-256 `b158f166…`, epoch 9,
`AP0 = 0.2374902968831333`), the same 40-epoch Train order plan, the restored
AdamW moments (step 24,642), and zero-initialised final Linear layers in the new
blocks. At step 0 the new residuals are exactly zero, so all four arms reproduce
the C1 function before training.

## 2. Correctness gates

All gates passed before formal training:

* Warm start: SHA-256, schema/metadata, 37 state tensors, bitwise load into a
  fresh C1; AdamW group 0 (`0..30`, 24,642 steps) restored bitwise, moments not
  reset, new parameters appended as a separate group.
* CPU tests: 15/15 (`tests/test_sg_model.py`, `tests/test_continuation.py`).
* Architecture audit: C = 245,184; F = 278,592; M = 279,488; V = 287,808
  parameters; F/M new-parameter difference 2.68% (≤ 5%); F/M shared FFN
  tensors have identical initial state.
* GPU preflight, both GPUs, all four arms: epoch-0 Validation AP reproduces
  `AP0` (within 1e-4), zero-residual identity with C1, permutation invariance,
  F-invariance with fixed atom states, M-response to real adjacency, V context
  isolation by graph, gradient flow into `bond_proj` and context projections,
  masked-BCE equality, and a 30-step speed smoke.
* No non-finite events in any formal run; peak allocated GPU memory ≈ 122–128 MB.

The first preflight attempt failed only the 1e-6 identity tolerance because
CUDA `index_add` reduction order produces ≈ 2–4e-6 differences. The tolerance
was calibrated to 1e-4 (context isolation 1e-5) **before any formal epoch**, and
both attempts are recorded in the ledger. This is a numerical gate calibration,
not a change to the model, data or promotion rule.

## 3. Phase 1 — four-arm screen (epochs 1–12)

Official Validation AP at the fixed evaluation epochs; `AP0` is the warm-start
value. No early stopping; epoch 12 is the pre-registered comparison.

| Arm | AP0 | AP4 | AP8 | AP12 | `D_base` | `D_control` | Tasks improved | ≥20-pos mean diff | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C | 0.237490 | 0.238267 | 0.236937 | 0.237321 | — | — | — | — | control |
| F | 0.237490 | 0.237739 | 0.236591 | 0.236557 | −0.000934 | −0.000765 | 51/127 | −0.000396 | reject |
| M | 0.237491 | 0.239256 | 0.239662 | 0.241275 | +0.003785 | +0.003954 | 81/127 | +0.005091 | reject |
| V | 0.237490 | 0.239000 | 0.239861 | **0.243552** | **+0.006062** | **+0.006231** | **92/127** | **+0.007238** | **promote** |

V satisfies every Phase-1 condition. F adds capacity but loses AP. M is positive
and improves 81 tasks, but misses both pre-registered `+0.004` floors by
0.000215 (`D_base`) and 0.000046 (`D_control`); it was **not** continued and no
post-hoc promotion was made. V's epoch-12 curve is still rising (0.2390 → 0.2399
→ 0.2436), so Phase 2 was the informative test.

## 4. Phase 2 — continuation to epoch 40

C and V resumed from the fixed **epoch-12** checkpoint (not from the epoch-4
"best" of C), with model, AdamW, scheduler, RNG and consumed-step state. The
comparison epoch 40 was fixed in advance.

| Run | AP12 | AP20 | AP30 | AP40 | Best epoch | Best AP | Train AP40 | Train BCE40 | Valid BCE40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 0.237321 | 0.237636 | 0.238060 | 0.238236 | 40 | 0.238236 | 0.4068 | 0.031054 | 0.046124 |
| V | 0.243552 | 0.249561 | 0.252350 | **0.252658** | 40 | 0.252658 | 0.4500 | 0.028976 | 0.045934 |

* `D40 = +0.014421`; `AP_W40 = 0.252658 ≥ 0.245`; 105/127 tasks improved;
  ≥20-positive subset (107 tasks) mean difference **+0.014048** (90 improved,
  17 regressed).
* Both runs improve monotonically to epoch 40; the best epoch equals the fixed
  comparison epoch, so the headline is not a best-epoch cherry-pick.
* After epoch 12, V gains +0.009106 while C gains +0.000915. The continuation
  test is the conservative one: the block was trained for only 28 epochs on top
  of a checkpoint that never saw it, and it still clears the Phase-2 floor.
* Validation BCE improves only ≈ 0.00019 while AP improves 0.0144. AP is a
  ranking metric; the gain is primarily a better ordering of positives, not a
  calibration shift. Train BCE does improve (0.03105 → 0.02898), so the block
  is fitting better rather than merely regularising.

Phase 2 satisfied every pre-registered condition (`PROMOTE_PHASE3`). D40 versus
the prior best non-C1 readout (0.240339) is +0.012318; versus `AP0` it is
+0.015167.

## 5. Phase 3 — double-seed from-scratch confirmation

Four 40-epoch runs: {C, V} × {seed 0, seed 1}. Within a seed, C and V share the
same base initialisation fingerprint (`1b1f1ad3…`), the same complete minibatch
order plan SHA-256 (`105a7637…` for seed 0), the same dropout seed and data
order; only V's added block differs. The learning-rate schedule is the
pre-registered 1-epoch warmup 3e-5 → 3e-4, then per-step cosine 3e-4 → 1e-5.
Epoch 40 is the primary comparison.

| Seed | C AP10/20/30/40 | V AP10/20/30/40 | C40 | V40 | D40 | V best epoch | Tasks improved | ≥20-pos mean diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.2015 / 0.2200 / 0.2296 / 0.2319 | 0.2209 / 0.2469 / 0.2574 / **0.2578** | 0.231945 | 0.257772 | +0.025828 | 40 | 101/127 | +0.030791 |
| 1 | 0.2006 / 0.2192 / 0.2256 / 0.2269 | 0.2187 / 0.2468 / 0.2565 / **0.2590** | 0.226906 | 0.258952 | +0.032047 | 40 | 109/127 | +0.037621 |

* V seed mean **0.258362**, population std **0.000590**; D seed mean
  **+0.028937**, std 0.003110. Both V seeds are ≥ 0.250 and both beat their
  paired control by ≥ 0.005 → **`PCBA_025_REPRODUCED`**.
* All four runs have best epoch = 40, so the fixed-epoch comparison and the
  best checkpoint coincide.
* The from-scratch gain is larger than the continuation gain (+0.026/+0.032 vs
  +0.0144). This is consistent with the new block co-adapting with the base
  representation when trained from the start; it also means the continuation
  result understates what the same architecture achieves with a fresh 40-epoch
  schedule. The two tests answer different questions and are reported separately.
* V seed-0/seed-1 gains agree in sign on 97/127 tasks (90 tasks improve in both
  seeds; Pearson 0.28 between the two seed gain vectors). The continuation gain
  and the seed-0 scratch gain also correlate positively across tasks
  (Pearson 0.21, rank 0.38). The effect is broad, not driven by a single task or
  seed.

## 6. What the arm comparison does and does not show

* Extra per-atom FFN capacity alone (F) did **not** help in this budget
  (AP12 0.236557, below C). The gain is not explained by added parameters per se.
* At epoch 12, MP2 alone (M, 0.241275) and MP2+context (V, 0.243552) are both
  above C; the context block adds +0.002277 over M in that single-seed screen.
  Because M missed the pre-registered floor and was not continued, the Phase-2/3
  result is for the **bundle V**, not for an isolated context effect. No
  post-hoc M continuation was run.
* The confirmation is a performance result on the official Validation split of
  one dataset. It is not a causal proof that bond messages or context cause the
  gain, and it is not a Test-set claim.

## 7. Budget, provenance and concurrency

| Stage | Runs | GPU-hours |
|---|---|---:|
| Phase 1 screen | C 0.4454, F 0.3577, M 0.4658, V 0.4072 | 1.6761 |
| Phase 2 continuation | C 0.9975, V 0.9072 | 1.9047 |
| Phase 3 seed 0 | C 1.4299, V 1.2776 | 2.7075 |
| Phase 3 seed 1 | C 1.4478, V 1.3058 | 2.7536 |
| GPU preflights (passing) | two files | 0.0145 |
| Failed attempts (charged) | first preflight pair, first Phase-2 pair | 0.0956 |
| **Total** | | **9.1520** |

Budget: target 15 GPU-hours / hard 16; actual **9.1520** (61% of target), at most
two GPUs concurrently. Wall clock from sprint start (≈16:07 UTC) to this final
report/analysis commit (≈21:56 UTC) is ≈5 h 49 min, below the 7.5 h target and
8 h hard cap. No GPU stage was started with < 30 min remaining.
Peak allocated memory was ≈ 122–128 MB per run; the runs are data-transfer- and
step-bound, not memory-bound.

Provenance:

* Phase 1 runs: commit `520f0793cf41…`; Phase 2/3 runs: commit `c39616c3…`;
  both deployed through `rr` and executed with `git_dirty: false`.
* Runtime recorded in every `run.json`: torch 2.5.1+cu124, CUDA 12.4,
  Python 3.12.14, A100-SXM4-40GB, FP32, matmul TF32 off.
* `analysis/run_registry.json` lists all 16 `rr` jobs (including the failed
  preflight and Phase-2 attempts); `analysis/gpu_ledger.json` includes training,
  preflight and failed-attempt entries.
* Warm-start SHA-256 and optimizer-step audit are stored in every run's
  `resume_audit`/`optimizer_state_audit`; per-task AP, curves and checkpoints
  are under `runs/` (git-ignored).

## 8. Incidents and deviations

1. **Preflight identity tolerance (before formal runs).** The first pair failed
   the 1e-6 logit-identity check at ≈ 2–4e-6, the expected reduction-order noise
   of CUDA `index_add`. The gate was recalibrated to 1e-4 (context 1e-5) and both
   GPUs passed all four arms. No model, data, LR or promotion parameter changed.
   The failed 41 s / 33 s allocations are charged in the ledger.
2. **Phase-2 first attempt (end of the first epoch, before any evaluation).**
   Phase 2 starts at epoch 13 and its first evaluation is epoch 20, so the
   per-epoch curve row called `int(tracker.best_epoch)` on `None` and raised
   `TypeError` after the first epoch of training, before any evaluation or
   checkpoint was written. The fix (commit `c39616c3`) makes the two
   pre-first-eval `best_*_so_far` fields `None`; it changes no training state,
   schedule or metric. Both arms were rerun (`sgs-p2-C2`, `sgs-p2-V2`); the
   failed JSONs are preserved in `analysis/failed_runs/`, and the one-epoch
   allocations are charged in the ledger.
3. **No post-hoc changes.** After seeing Validation AP, no arm, width, LR,
   epoch, threshold, comparison epoch or task set was changed. M was not
   continued despite being close; F was not retuned; no extra seed was added to
   Phase 2. The Phase-3 seeds and schedule were fixed in the protocol.
4. **Test discipline.** The frozen loader checks that the official Test label
   slice is entirely NaN; that check indexes the NaN mask (it is inherited from
   the frozen data path) but no Test logit, AP, BCE or per-task score is ever
   computed, saved or used in a decision. Every run records
   `test_evaluated: false`. The README wording is corrected accordingly.

## 9. Limitations

* One dataset and one architecture family; the result is Validation AP, not a
  Test-set estimate. The Test split remains unused and this round cannot claim a
  leaderboard number.
* The 40-epoch budget is short relative to the historical 360-epoch schedule.
  The from-scratch control is undertrained, so the absolute from-scratch APs are
  a budget-specific comparison, not a converged-model comparison. The
  continuation experiment (Phase 2) addresses the same question on the frozen
  checkpoint and gives the more conservative gain.
* V is not parameter- or compute-matched to C (+42,624 parameters, +17.4%;
  ≈ 41 ms/step vs ≈ 47 ms/step on the shared GPU). The comparison is "same
  recipe, added structural block", not iso-parameter or iso-FLOP.
* V bundles MP2 and context. The isolated context contribution is estimated only
  by the epoch-12 M-vs-V difference (+0.0023, one seed); M itself missed the
  promotion floor and was not continued.
* Validation AP is used for the pre-registered promotion decisions, so the
  reported numbers are selection-aware. The double-seed confirmation and the
  broad per-task support reduce, but do not eliminate, this concern.
* Sparse tasks dominate the tails: the largest single-task gains and losses are
  on tasks with < 10 Validation positives. The ≥20-positive subset is the
  robustness check (+0.0140, +0.0308, +0.0376 across the three comparisons).
* Only A100-40GB, FP32, one host; no multi-node or mixed-precision test.

## 10. Phase 4 (optional Lipophilicity probe)

Not run. At confirmation (21:51 UTC) the round had ≈1 h 45 min of wall clock and ample
GPU-hour headroom, but the Lipophilicity Fit-2352/Monitor-504
adaptation is a different feature/graph pipeline from the PCBA cache; building
and testing it would have consumed the window reserved for the PCBA analysis,
report and commit. Protocol §9 explicitly permits skipping when the data
adaptation would exceed the remaining budget, and prioritises the PCBA readout.
No Lipophilicity label was read or scored in this round.

## 11. Reproduction

Local analysis (after the `runs/` artifacts are pulled from `res`):

```bash
uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/tests
uv run python tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/code/sg_audit.py
uv run python tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/code/sg_analyze.py all
rr jobs --json > /tmp/rr_jobs.json
uv run python tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/code/sg_analyze.py registry --rr-jobs /tmp/rr_jobs.json
```

Formal executions (abbreviated; exact commands and guards are in `README.md`
and `PROTOCOL.md`):

```bash
rr deploy res
rr run res sgs-p1-C --pool res-gpu0 --gpus 1 --result .../runs/phase1_C -- \
  timeout 2800s uv run python -u .../code/sg_train.py --phase 1 --arm C --out-dir .../runs/phase1_C
rr run res sgs-p1-F --pool res-gpu1 --gpus 1 --result .../runs/phase1_F -- \
  timeout 2800s uv run python -u .../code/sg_train.py --phase 1 --arm F --out-dir .../runs/phase1_F
# then M / V on the same commit; then, only after PROMOTE_V:
rr run res sgs-p2-C2 --pool res-gpu0 --gpus 1 --result .../runs/phase2_C -- \
  timeout 4500s uv run python -u .../code/sg_train.py --phase 2 --arm C \
  --resume .../runs/phase1_C/checkpoint_epoch12.pt --out-dir .../runs/phase2_C
rr run res sgs-p2-V2 --pool res-gpu1 --gpus 1 --result .../runs/phase2_V -- \
  timeout 4500s uv run python -u .../code/sg_train.py --phase 2 --arm V \
  --resume .../runs/phase1_V/checkpoint_epoch12.pt --out-dir .../runs/phase2_V
# then the four Phase-3 runs, two per seed:
rr run res sgs-p3-s0-C --pool res-gpu0 --gpus 1 ... --phase 3 --arm C --seed 0 ...
rr run res sgs-p3-s0-V --pool res-gpu1 --gpus 1 ... --phase 3 --arm V --seed 0 ...
rr run res sgs-p3-s1-C --pool res-gpu0 --gpus 1 ... --phase 3 --arm C --seed 1 ...
rr run res sgs-p3-s1-V --pool res-gpu1 --gpus 1 ... --phase 3 --arm V --seed 1 ...
```

## 12. Files

| Path | Contents |
|---|---|
| `PROTOCOL.md` | frozen pre-registration |
| `code/sg_common.py` | constants, arms, LR schedule, decision thresholds |
| `code/sg_model.py` | C1 subclass, F/M/V blocks, init, AdamW restore |
| `code/sg_data.py` | PCBA loader, order plan, official-AP evaluation |
| `code/sg_train.py` | Phase 1/2/3 trainer, PinnedStager, fixed epochs |
| `code/sg_preflight.py` | GPU correctness gates |
| `code/sg_analyze.py` | promotion rules, ledger, registry |
| `analysis/phase1_screen.json` | Phase-1 decision and per-task evidence |
| `analysis/phase2_continuation.json` | Phase-2 decision and curves |
| `analysis/phase3_seed_confirmation.json` | double-seed confirmation |
| `analysis/gpu_ledger.json` | training, preflight and failed-attempt hours |
| `analysis/run_registry.json` | all `rr` jobs, commits, states |
| `analysis/initialization_audit.json`, `architecture_audit.json` | gate evidence |
| `runs/` (git-ignored) | checkpoints, curves, per-task AP, RNG state |
