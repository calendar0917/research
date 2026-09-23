# PEC-v0 — decision

**Round** PEC-v0 · study `zinc-context-gap` · protocol `pec_v0` ·
implementation/analysis commit `883e529b159cae49fd1a9d279db2b40b8ff5e77c` (+ this
round's uncommitted-then-committed files).

`official_test_loaded = false` · `official_valid_read = false` ·
no historical baseline retrained.

---

## Frozen verdicts

| gate | frozen verdict |
|---|---|
| Gate 0 — correctness (CPU) | **PASS** (8/8 checks) |
| Gate 1 — label-free dictionary (CPU) | **FAIL** (7/10 criteria) |
| Gate 2 — cheap internal task screen (CPU) | **PASS** (`GATE2_PASS_BUY_SEED0`, 5/5 criteria) |
| Gate 3 — official-valid seed-0 (A100) | **NOT RUN** |

## Decision

**Do not purchase the Gate-3 seed-0 official-valid GPU run without explicit
authorization. Record the round as a positive CPU screen and stop here.**

## Because

1. The pre-registration makes Gate 3 conditional on *all* previous gates passing.
   The **frozen Gate-1 verdict is FAIL**. Amendment A2 was scoped to Gate 2 only
   and must not be silently extended to the expensive, hard-to-reverse resource
   decision.
2. Gate 2 nevertheless passed every one of its pre-registered criteria, with
   large margins on the two mechanism criteria:
   * chemistry-placement shuffle degrades MAE by **+0.835** (required ≥0.02),
   * TRUE composition beats **BAG** by **+0.0352** and **SHUFFLE** by **+0.0116**
     (each required ≥0.003),
   * the sparse dictionary is **not worse** than the exactly parameter-matched
     dense control (`0.46181` vs `0.46711`) and is load-bearing
     (neutral-dictionary MAE `2.175`).
3. Therefore the round is *paused*, not concluded: the screen is positive and
   the two plausibly-fatal outcomes are excluded, but the absolute-band claim
   requires Gate 3.

## Alternatives rejected

* **Buy the seed-0 GPU run now** — rejected: Gate 1's frozen numeric verdict is
  FAIL; buying now would be an unpreregistered rescue of a failed frozen gate.
* **Re-run Gate 1 with a relaxed threshold** — rejected: forbidden; the frozen
  criteria and verdict are reported verbatim and unchanged.
* **Rescue with `K=32/64`, more IHT steps, or a different basis** — rejected:
  explicitly forbidden by the brief.
* **Claim `ENV_COMPOSITION_SUPPORTED_DENSE_NOT_DICT`** — rejected as
  contradicted: the sparse arm is better than the dense arm at matched
  parameters.
* **Claim `PURE_SPARSE_DICT_ENV_COMPOSITION_SUPPORTED` now** — rejected: the
  pre-registered meaning of that verdict is a formal official-valid result; we
  only have a 2 000-molecule internal screen.

## Revisit if

* The user authorizes the Gate-3 seed-0 official-valid purchase (two arms:
  best non-dictionary control and `CK` sparse), **or**
* the user wishes to re-freeze the Gate-1 thresholds with a declared,
  pre-registered calibration (a new preregistration, not an amendment).

## If authorized, Gate 3 is ready

Clean committed revision → `remote-research-runner` preflight → deploy to
`/home/hxy/cy/research` → A100 smoke → two-arm seed-0 official-valid run
(detached, single job) → pull → local analysis. Official test stays closed.
