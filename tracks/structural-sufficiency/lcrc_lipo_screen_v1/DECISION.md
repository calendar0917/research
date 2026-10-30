# DECISION — LCRC-A Lipophilicity screen v1

- **Date:** 2026-10-30
- **Round status:** `READY_FOR_GPU_AUTHORIZATION`
- **Scientific decision:** **DEFERRED — no verdict on any hypothesis.**
- **GPU:** not authorized, not used. No formal A/B/C run exists.
- **Monitor/Reserved/official labels:** not read, not scored.

## 1. Decision

The round is complete as a **pre-GPU engineering package** and stops here. It has not
produced evidence for or against LCRC-A, the S2 channel, GINE or ECFP4 on
Lipophilicity. Any verdict now would be fabricated.

The only decision taken is procedural:

> The implementation, frozen protocol, parameter table, budget estimate and CPU
> evidence gates are ready. Formal execution may begin **only** after the researcher
> explicitly authorizes GPU work for this exact frozen revision.

No next research round has been started, and none should start automatically after the
GPU stage. The post-GPU decision rule is pre-registered in `PROTOCOL.md` §6.

## 2. Preconditions — all met

| precondition | evidence | status |
|---|---|---|
| first-round assets unchanged and hash-verified | `analysis/data_audit.json`, gate G0 | met |
| Fit rows and labels one-to-one | gate G1, `fit_labels` audit | met |
| Monitor unreachable from the training path | gate G1 static/import scan + freeze gate | met |
| Reserved/official paths denied | gate G8 audit hook | met |
| A/B identical init, B S2 exactly off, A S2 gradients live | gate G2 | met |
| GINE real-adjacency message passing | gate G3 | met |
| finite outputs/loss/grads, save/restore, soup | gates G2–G4 | met |
| CPU smoke loss decreases for A/B/C | gate G5, `analysis/smoke.json` | met |
| ECFP4 rows aligned, XGBoost deterministic | gate G6 | met |
| evaluation order and classification rules unit-tested | gate G7 | met |
| protocol/config/code frozen with hashes | `analysis/protocol_freeze.json` | met |
| budget estimate within the internal cap | `analysis/budget_estimate.json` (0.734 h vs 1.5 h) | met |
| no GPU process started | `analysis/cpu_gates.json`, `analysis/smoke.json` | met |

The 53 pytest checks pass. The round has no failed formal seed because no formal seed
has run.

## 3. Authorization contract

GPU work is **not** authorized by this document, by the 4-hour external cap, or by the
budget estimate. It requires a new, explicit researcher action: creating
`analysis/gpu_authorization.json` that matches the current protocol freeze.

Minimum contents:

```json
{
  "authorized": true,
  "protocol_id": "lcrc-lipo-screen-v1",
  "protocol_freeze_sha256": "<sha256 of analysis/protocol_freeze.json>",
  "max_gpu_hours": 1.5,
  "authorized_runs": ["A_s0","A_s1","A_s2","B_s0","B_s1","B_s2","C_s0","C_s1","C_s2"],
  "researcher_note": "<explicit authorization and any additional limits>"
}
```

`run_screen.py train` will refuse if:

- the file is absent, `authorized != true`, or `protocol_id` differs;
- the authorization hash does not match the current freeze;
- `max_gpu_hours` is ≤0 or exceeds the frozen hard cap (1.5);
- the requested run is not in `authorized_runs`;
- any code/config hash has drifted, or the tracked worktree is dirty;
- the run directory already contains a `train.json` (no silent overwrite).

There is no `--force` bypass. Any change to code or config after the freeze requires
regenerating the freeze and a new authorization (and is recorded as a deviation).

## 4. Execution order after authorization

1. Deploy the committed revision with the repository `rr` runner. Do not start a
   manual SSH training process.
2. Run the nine neural jobs (`A/B/C × s0/s1/s2`), with the frozen recipe and no
   overrides. Record wall time, GPU hours, and any failure.
3. Produce label-blind predictions for every completed run
   (`predict-nn --device cuda`), plus the already-computed or re-computed local
   `D_r0..r2`.
4. `freeze-predictions` must succeed for all 12 keys and write
   `MANIFEST.json` + `MANIFEST.sha256`.
5. Only then run `evaluate` once, on CPU, which reads Monitor504 labels and writes
   `monitor_metrics.json`, `monitor_paired.json`, `monitor_diagnostics.json`.
6. Update `REPORT.md`/`DECISION.md` with raw results and exact hashes. Do not change
   the frozen recipe, metrics or decision rule afterwards.

If any gate or hash check fails at step 1–4, stop and report. Do not reduce the
comparison, drop a seed, or change the recipe to make the run fit.

## 5. Budget rules

- Frozen internal hard cap: **1.5 A100 GPU-hours**.
- Researcher-suggested external cap: **4.0 A100 GPU-hours** (a cap, not a target).
- Estimate: **0.734 GPU-h** for nine runs including 20% overhead and reserves.
- Every launch, including failed/aborted ones, is charged against the cap and recorded.
- A rerun of the same frozen config is allowed only after recording why the first run
  failed; it is not a new experiment and must not change seed/epochs/lr.
- Arm D is local CPU and consumes 0 GPU-hours.
- If measured cost exceeds the cap, stop after the current run and report; do not
  shrink the number of seeds or arms to fit.

## 6. Post-GPU interpretation (pre-registered, not yet applicable)

The evaluator emits exactly one of three cases, using only the frozen rules:

| outcome | interpretation | permitted next step |
|---|---|---|
| `CASE_1_S2_INCREMENT_SUPPORTED` | A learned the task, S2 shows a clear ≥0.03 RMSE increment over B, and A is not materially worse than C/D | Consider a separately pre-registered S2 mechanism/SSL study; still not a claim of unique mechanism or interpretability |
| `CASE_2_LCRC_COMPETITIVE_NO_CLEAR_S2_INCREMENT` | A's overall representation is competitive, but S2 has no clear increment | S2 must not be advertised as the core contribution; any follow-up must target the actual competitive component |
| `CASE_3_A_WEAKER_OR_DID_NOT_LEARN` | Scratch A is weaker than the references or fails to beat the Fit-mean constant | Report the negative result; no hyperparameter search, no SSL rescue inside this round |

Additional pre-registered restrictions:

- All three seeds are reported; no best-seed headline.
- Seed uncertainty and molecule-resampling uncertainty are separate; a positive
  mean with seed sign flips is not a stable increment.
- Train-vs-Monitor RMSE gap is reported; Monitor is a development set, not a fresh
  blind test.
- Reserved labels and official valid/test labels remain unread regardless of outcome.
- No result is compared with historical official valid/test rankings.

## 7. What would justify a next round

Only after the GPU stage, and only under a new pre-registration:

- **SSL:** justified only if scratch A is clearly underfit or CASE_3, and the frozen
  analysis identifies an optimization/representation gap that SSL could address. The
  existing BJOC evidence that SSL can help small-data regression is background, not a
  result of this round.
- **Mechanism ablation:** justified only if CASE_1 or a stable but non-material S2
  signal appears; it must isolate S2 content from effective capacity (B is not
  information-matched).
- **Independent data:** justified only after a stable Monitor result; Reserved or
  official test would be read once under a new frozen protocol, never post-hoc.

A negative or mixed result does **not** automatically trigger any of these. This
document does not authorize them.

## 8. Invalidation conditions

The round must be reported as invalid or amended (with the original committed) if any
of the following is later discovered:

- any Monitor/Reserved/official label was read before the prediction freeze;
- any code/config hash differs from the freeze without a recorded amendment;
- a seed, arm or run is missing without a recorded failure;
- checkpoint selection used Monitor or a best-seed criterion;
- the reported metric is not raw-scale RMSE/MAE on the frozen Monitor rows.

As of this decision, none of these has occurred: no formal prediction exists and no
Monitor label has been read.
