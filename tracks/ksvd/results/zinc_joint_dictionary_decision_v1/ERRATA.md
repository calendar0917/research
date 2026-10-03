# ERRATA — errata for prior ZINC reports, surfaced by zinc-joint-dictionary-decision-v1

This file records corrections to earlier ZINC documents that Stage A re-derived
from the raw inputs.  It does not rewrite history: the original files are left
untouched, and every claim below is reproducible from
`cycle_input_audit.py` / `cycle_input_decision.json`.

## E1. `zinc_overnight_bottleneck_v1/REPORT.md` §5 — group labels do not match the printed numbers

**Where.** `tracks/ksvd/results/zinc_overnight_bottleneck_v1/REPORT.md`,
section *"5. Error budget / group accounting"*, the `N0_s0` row of the error
table:

```
| `N0_s0` | 0.084080 (0.087130) | 0.004644 (0.154796) | 0.022482 (4.496366) | 0.111206 |
```

with the column header `| arm | G0 (965) | G1 (34) | G172 (1) | total |`.

**Problem.** The column labels are the frozen three-way *row-set* partition

* `G0` = 965 rows, cycle penalty 0
* `G1` = 34 rows, negative penalty excluding `valid:0172`
* `G172` = 1 row, `valid:0172`

but the two right-hand numbers are the **severity-stratum** contributions, i.e.
grouping by the penalty value:

* `0.004644` / `0.154796` = the 30 rows with penalty −1 only
* `0.022482` / `4.496366` = the 5 rows with penalty ≤ −2 only

The correct values for the *labelled* `G1` (34 rows) and `G172` (1 row) sets are
different, because the 5 severe rows (including 0172) are split across the two
labelled sets:

| set | n | MAE | total-error contribution |
|---|---|---|---|
| `G0` | 965 | 0.0871300 | 0.0840804 |
| `G1_ex172` | 34 | 0.2471608 | **0.0084035** |
| `G172` | 1 | 18.7222337 | **0.0187222** |
| total | 1000 | 0.1112061 | 0.1112061 |

The severity-stratum table that actually produced the printed numbers is:

| stratum | n | MAE | contribution |
|---|---|---|---|
| penalty 0 | 965 | 0.0871300 | 0.0840804 |
| penalty −1 | 30 | 0.1547958 | 0.0046439 |
| penalty ≤ −2 | 5 | 4.4963655 | 0.0224818 |
| total | 1000 | 0.1112061 | 0.1112061 |

**Verdict.** The `N0_s0` total (0.111206) and the G0 row are correct. The `G1`
and `G172` columns are the *severity* strata (30/5) printed under the *row-set*
labels (34/1). Both decompositions sum to 0.111206; they are not the same
partition. Any downstream argument that read "G1 (34) = 0.004644" as the error
contributed by the 34 non-0172 ring rows is therefore reading the wrong number;
the right one is 0.0084035.

Reproduced by `cycle_input_decision.json` → `A1.n0_s0_group_partition`,
`A1.n0_s0_priority_severity`, `A1.errata_report_section5_labels`.

## E2. "30 / 5" severity rows versus "34 / 1" row-set groups are not interchangeable

Related to E1 but stated separately because it affects gate wording elsewhere:
the *severity* strata used by the C3 gate (penalty 0 / −1 / ≤ −2) and the frozen
`G0 / G1_ex172 / G172` groups are different partitions of the same 1000 valid
rows.  Contributions must be reported against one partition at a time; mixing
the counts of one with the numbers of the other is the E1 error.  In this task:

* Stage C gates use the **severity** strata, computed on the 2000-row
  molecule-level **dev** split (not the official valid).
* Stage D (if reached) reports both partitions explicitly.

No other numeric discrepancy was found: A1 reproduced `raw = 0.11073645`,
`cal = 0.11120612` (published `0.11120613`, Δ ≈ 1e-8), and the train-fitted
median bias `−0.01204258` to 1e-8.

## E3. Scope note (not an erratum)

`cycle_input_decision.json` also confirms the A-stage inputs carry no split
leak: the topology25 key spaces `topo_raw` and `topo_model_input` induce the
**identical** partition of the 10000 official-train rows
(`raw_nrm_class_partition_identical = true`), and the all-train affine
standardiser has max residual 6.47e-7, so the normalised key is an affine image
of the raw key and cannot separate rows the raw key merges.
