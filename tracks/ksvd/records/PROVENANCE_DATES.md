# Record date provenance

This file documents durable-record timestamps that are **not** real experiment
dates, so that the anomaly is auditable instead of silently "cleaned up".

## Background

The track was bootstrapped in one large backfill commit (`9f35aaf`,
2026-09-12, "compact-v4 / SBCI audit backlog and state update") that committed
previously-uncommitted compact-v4-line work together with its claim/decision
records. A number of those records carry dates **later than the commit that
introduced them** — several are in the future relative to the real system clock
(2026-09-18). The date appears in three places that must stay consistent:
the filename suffix, the `created:` field, and the date embedded in the
`claim_id` / `decision_id`.

## Policy

* Records are **not** re-dated and **not** renamed. Renaming would break the
  id ↔ filename ↔ cross-reference chain that the rest of the records/notes rely
  on, and the true experiment date cannot be recovered from the repository.
* The anomaly is recorded here and enforced by `uv run research verify`:
  a future-dated record is an **error** unless its id is listed below.
* Git commit dates are evidence of *when a record existed*, not of when the
  underlying experiment ran; they are recorded here only as a lower bound.

## Recoverable vs unrecoverable

* `claim/decision-local-token-null-20260919` — first commit 2026-09-18, i.e.
  the record existed at least one day before the date in its name. The stage B
  result is referenced by the 2026-09-18 one-shot test read, so the real date is
  ≤ 2026-09-18. Kept as-is, no rename; documented below.
* The remaining records were all introduced by the 2026-09-12 backfill. Their
  content predates that commit, so their real dates are ≤ 2026-09-12 and cannot
  be pinned down further. Kept as-is; documented below.

No record in this list is used as a live pointer from `STATE.yaml`; the live
state points at claim/decision ids that are date-consistent.

<!-- The fenced block below is the machine-readable allowlist parsed by
     ksvd_research.runtime.integrity. Keep it in sync with the prose above. -->
```yaml
documented_future_dated_records:
  - record: claim-compact-v4-cell-minimal-falsification-nogo-20260921
    kind: claims
    filename_date: '2026-09-21'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: claim-compact-v4-sbci-nogo-20261007
    kind: claims
    filename_date: '2026-10-07'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: claim-local-token-null-20260919
    kind: claims
    filename_date: '2026-09-19'
    first_commit_date: '2026-09-18'
    reason: result existed by 2026-09-18; real date is <= 2026-09-18 but exact day unknown (kept, not re-dated)
  - record: claim-optimized-manifold-broad-state-inconclusive-20260919
    kind: claims
    filename_date: '2026-09-19'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: claim-raw-graph-patch-system-sufficiency-caseD-20260926
    kind: claims
    filename_date: '2026-09-26'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: claim-sample-efficiency-gain-localization-caseH-20261006
    kind: claims
    filename_date: '2026-10-06'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: claim-sbci-fit-generalization-triage-inconclusive-20261007
    kind: claims
    filename_date: '2026-10-07'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: claim-structural-computation-gap-analysis-20260920
    kind: claims
    filename_date: '2026-09-20'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: decision-compact-v4-cell-minimal-falsification-nogo-20260921
    kind: decisions
    filename_date: '2026-09-21'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: decision-compact-v4-sbci-nogo-20261007
    kind: decisions
    filename_date: '2026-10-07'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: decision-local-token-null-20260919
    kind: decisions
    filename_date: '2026-09-19'
    first_commit_date: '2026-09-18'
    reason: result existed by 2026-09-18; real date is <= 2026-09-18 but exact day unknown (kept, not re-dated)
  - record: decision-optimized-manifold-broad-state-inconclusive-20260919
    kind: decisions
    filename_date: '2026-09-19'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: decision-raw-graph-patch-system-sufficiency-caseD-20260926
    kind: decisions
    filename_date: '2026-09-26'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: decision-sample-efficiency-gain-localization-caseH-20261006
    kind: decisions
    filename_date: '2026-10-06'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: decision-sbci-fit-generalization-triage-inconclusive-20261007
    kind: decisions
    filename_date: '2026-10-07'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
  - record: decision-structural-computation-gap-analysis-caseA-20260920
    kind: decisions
    filename_date: '2026-09-20'
    first_commit_date: '2026-09-12'
    reason: backfilled in 9f35aaf; true experiment date not recoverable (kept, not re-dated)
```
