# DECISION — zinc-task-dictionary-and-cycle-witness-seed0-v1

Status: **closed; main gate `INCONCLUSIVE` (no dictionary advantage), CPU
witness resolved as genuine input aliasing.**  No write-back, no merge, no
official-valid read, no 10k confirmation.  Frozen decision record; the
machine-readable summary is `analysis.json` + `gate.json`.

## Answers to the five closing questions

**1. On the fresh compressed-body fold, does the current shared task-dictionary
bridge beat the parameter-matched MLP bridge?  Where is any difference?**

No pre-registered category supports the dictionary.  New-dev overall
calibrated MAE is `D 0.122960` vs `M 0.117032` (gain `-0.005928`,
CI `[-0.014148, +0.001230]`, negative favours the MLP); the primary G0
endpoint is a tie (`0.104300` vs `0.103305`, gain `-0.000995`,
CI `[-0.006240, +0.003491]`); the raw endpoint has the same direction and the
same non-separation.  The point estimate crosses the 0.003 "overall
improvement" label in favour of the MLP but is not CI-separated, and the
per-row counts are nearly even (dev cal: 1009 better / 991 worse; max
`|delta| = 4.61`).  The one robust mechanism difference is capacity on the
severe tail: on **fit**, `k<=-3` cal MAE is `1.150` (M) vs `3.962` (D) and
`k=-2` is `0.229` vs `0.317`, and the dev gain is concentrated on the two
`k<=-3` dev rows.  So: the 16-step/288-atom ISTA dictionary code underfits the
extreme-tail mapping even in sample; it shows no generalisation advantage over
a same-parameter MLP.  This is one seed and one new internal fold; it is not a
claim about all dictionaries or all MLPs.

**2. What do the old internal-dev numbers now mean, and can they be used as
controls?**

They are background only.  `B 0.120130` / `R_DM 0.115736` were measured on the
**old internal dev** with a frozen body and loaded learned state; this round
uses a new fold refit from raw inputs with fresh weights, so fresh `D`
(`0.122960`) cannot be compared to old `B` directly.  The `A0 - B = +0.006011`
gap remains unattributable to unfreezing alone (extra epochs + dense
reconstruction path + optimizer trajectory); see `ERRATA.md`.  Old
`q_U 0.113180` / `q_B 0.152442` tail-head numbers are re-explained by the
witness classes: two of the five old fit extremes are members of input
conflict classes and three are not, and the old `5.54946` is a
severity-blind class-median cost, not a group floor.  No old arm is used as a
matched control anywhere in this round.

**3. CPU witness: what exactly are the "same topology25, different ring
targets" samples?**

Four exact `T25` classes (86, 232, 250, 259) with 10 rows total in the old fit
(0.125%); 190 repeated classes cover 98.2875% of rows but only these four
disagree on the target.  Witnesses: 86 (`train:3741` k=-1 vs `train:7491`
k=0), 232 (`train:0593` k=-5 plus `train:2447/2472/9913` k=0), 250
(`train:2232` k=-2 vs `train:3626` k=0), 259 (`train:1270` k=0 vs
`train:1424` k=-6).  Every witness pair is non-isomorphic (topology-only and
typed), so the compression genuinely merges differently-shaped molecules.
`group_only_min_l1 = 0.0` in every severity group: the conflicts exist only
across severity groups.  Flags: `INPUT_ALIASING_WITNESS` and
`INPUT_EQUIVALENT_TARGET_CONFLICT` for all four classes; class 232 is also
`ORDER_DEPENDENT_HELPER` (the repo stored-order cycle helper scores
`train:2447` as -5 while canonical/GVAE order scores 0); no class is
`UNRESOLVED_PROVENANCE`.  Node renumbering (16 perms, seed 20261004) moves the
stored-order helper on 7/8 witness molecules, which localises helper
order-sensitivity but does not prove the official labels wrong.  Corrected
statistics: `global_min_l1 = 0.0060697`, `global_opt_group_cost[k<=-3] =
5.54946`, old `conflict_row_fraction = 0.982875` is the repeated-row
fraction.

**4. What is truly reproducible vs exploratory?**

Reproducible from committed artifacts: the new fold and hashes, both full
training runs and their replay check (`1.67e-06`), the parameter audit, the
matched shared initialization (`5a8d8ff0...`, 34 tensors, zero mismatch), all
tables/CIs and the bootstrap self-tests, the CPU witness/bounds/renumber
results, and the J/M flip replay.  Exploratory: every performance delta — one
seed, one new internal fold, and 0.001-0.006-scale effects inside the fold's
interval; the `k<=-3` dev side is only 2 rows.  The `k<=-3` fit gap is a
robust in-sample observation of this configuration only.

**5. Budget and data boundaries; next single action?**

Kept: seed 0; new 8000/2000 fold frozen before training; exactly two arms;
max 2 parallel A100 GPUs; `1503.9 s = 0.42 GPU-hours` trained (limit 1.2);
CPU <= 8 threads; two Slurm jobs (`55894`, `55895`) both completed on `c05`
with exit 0 and none left running; official-test never loaded; official-valid
loaded 0 times; old result directories untouched.  Next single action:
re-run the identical `D` vs `M` protocol on a second fresh fold (or seed 1)
before anything else — with, as an option, an ISTA-capacity control
(param-matched by shrinking `V_L`) to test whether the tail gap is the 16-step
coding bottleneck.  Do not write back, do not touch official-valid, and do not
tune `lambda1/lambda2` or the tail on this dev.
