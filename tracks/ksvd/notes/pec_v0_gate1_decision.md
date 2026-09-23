# PEC-v0 — Gate 1 result and the decision to proceed to Gate 2 (Amendment A2)

Written after the `pec_v0` Gate-1 run (local CPU, commit `883e529`,
`results/pec_v0/gate1_label_free.json`, `GATE1_REPORT.md`) and **before** any
Gate-2 run. It does not modify the pre-registration; it records a decision and
its reason, as the pre-registration's gate vocabulary requires.

## 1. The frozen Gate-1 verdict is FAIL

The pre-registration (`pec_v0_preregistration.md` §6) froze ten criteria; three
failed:

| criterion | frozen | observed | verdict |
|---|---|---|---|
| `node_random_ratio` `E_rec(ksvd)/E_rec(random)` | `<= 0.10` | **0.1115** | FAIL |
| `edge_random_ratio` `E_rec(ksvd)/E_rec(random)` | `<= 0.10` | **0.1024** | FAIL |
| `edge_used` used atoms of 16 | `>= 12` | **11** | FAIL |

The other seven pass. This FAIL is recorded as-is and is **not** rewritten.

## 2. What the numbers actually say

| quantity | node | edge |
|---|---:|---:|
| `E_rec` monitor (K-SVD, s=4) | 0.0369 | 0.0543 |
| `E_rec` monitor (matched random) | 0.3305 | 0.5299 |
| K-SVD advantage over random | **9.0x** | **9.8x** |
| used atoms / 16 | 12 | 11 |
| argmax-used atoms | 7 | 3 |
| effective atom count | 6.0 | 4.5 |
| exact `l0 = 4` | yes | yes |
| PCA-16 `E_rec` (dense reference) | 5.5e-31 | 1.3e-31 |
| `alpha -> shell` probe macro-F1 | **1.000** (majority 0.217) | — |
| `alpha -> shellpair` probe macro-F1 | — | **1.000** (majority 0.180) |

The **user-specified Gate-1 STOP condition** ("if the dictionary cannot even
preserve the coarse rooted structural role, STOP") **passes decisively**: the
frozen `s=4` codes recover `shell` / `shellpair` perfectly, with a `+0.78` /
`+0.82` macro-F1 margin over the majority baseline, and the K-SVD dictionary is
~9-10x better than a matched random dictionary on reconstruction with exact
`l0 = 4` and full row coverage.

## 3. Honest reading of the three failures

* The two `random_ratio` misses are **0.0115 and 0.0024 absolute over a
  threshold this round's author invented without calibration data**. In `R^11`
  / `R^15`, four atoms out of sixteen already span a large subspace, so a random
  dictionary is a strong baseline and the `<= 0.10x` ratio was set aggressively.
  Both roles are still ~9-10x better than random, which is the substantive
  content of that criterion.
* `edge_used = 11` is expected for an **overcomplete** `K=16 > dim=15`
  dictionary: some atoms are redundant. `effective_atom_count = 4.5` and
  `argmax_used = 3` show the edge codes are concentrated, not dead.
* A design weakness discovered here: because the occurrence basis row already
  contains the shell one-hot, the `alpha -> shell` probe is close to
  tautological and cannot really fail. Its PASS is weak evidence and is reported
  as such.

## 4. Decision (Amendment A2)

Proceed to Gate 2. Reason:

1. The **user-specified** Gate-1 STOP condition (coarse rooted structural role
   recoverability) passes with a large margin; no dictionary has been shown
   unable to preserve the role axis.
2. The frozen FAIL is a **marginal miss on two uncalibrated self-set numeric
   thresholds**, not an observed mechanism failure. Rewriting those thresholds
   after seeing the data would be goalpost-moving; they are therefore left
   frozen and reported as FAIL in every artifact.
3. Gate 2 is the actual scientific test of the round (environment vs BAG /
   SHUFFLE, sparse vs dense). Stopping before it would answer nothing about the
   research question while the CPU cost is ~20 minutes.
4. The pre-registration's final verdict vocabulary presumes Gate 2 is reached;
   a Gate-1 numeric miss has no corresponding terminal verdict, which is
   consistent with the user brief making coarse-role recoverability (not the
   random ratio) the Gate-1 STOP condition.

Guardrails attached to A2 (non-negotiable):

* **No rescue**: no `K`/`s` change, no epochs change, no threshold change, no
  architecture change, no additional seeds. The frozen Gate-2 criteria in
  `pec_v0_preregistration.md` §7 are used verbatim and are not re-tuned after
  seeing Gate-2 numbers.
* The Gate-1 FAIL and all raw numbers stay in `gate1_label_free.json`, this
  note and the final decision record.
* If the Gate-2 screen returns `TRUE ≈ BAG`, `TRUE ≈ SHUFFLE`, or a materially
  worse `CK` than `CD`, the round STOPs and no GPU run is purchased.
* Official valid and official test stay unread through Gate 2.

## 5. Note on Gate-1 cost

`seconds = 899.8` for both K-SVD fits (1 135 990 node occurrence rows and
987 582 edge occurrence rows, 10 epochs each). This is the only expensive CPU
stage; Gate 2 re-fits on the 2 000 Gate-2 train molecules.
