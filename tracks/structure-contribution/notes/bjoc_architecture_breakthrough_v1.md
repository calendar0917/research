# BJOC architecture breakthrough v1 — working notes

Protocol: `configs/bjoc_architecture_breakthrough_v1.yaml` (frozen before any
GPU trajectory).  Track: `tracks/structure-contribution`.
Status: pre-registration + local CPU validation; GPU results appended below as
they are produced.

## 1. Question

The frozen system is BJOC-Null + Fit-only masked atom+bond SSL + R3
(413,505 parameters; historical full-10k official-test seed-mean MAE 0.069822,
used only as background, never for selection).  The Null composer deliberately
keeps whole-molecule junction statistics and drops which junction touches which
bond, while the masked-SSL pretext head already consumes true incidence.  This
round asks two one-shot architecture questions on top of that frozen system,
with a matched in-split P3 baseline as the only comparison object:

* **Stage A / GLRC**: add a per-molecule tanh gate on the true-incidence local
  residual so the model can mix Null global statistics and local ownership
  adaptively (`+34` parameters, zero-init so the start is exactly BJOC-Null).
  *A1*: does it improve Monitor raw-y MAE?  *A2* (conditional purchase): if it
  does, is the gain explained by a fully-Organized incidence readout, or does
  adaptive local-global mixing add value beyond Organized?
* **Stage B / DLP** (conditional): distance-conditioned low-rank pair pooling
  (rank 32; unordered pairs at d=3, d=4, d>=5; mean of elementwise products;
  `+27,648` parameters, zero-init new reader columns).

If both fail their pre-registered Monitor gates, the round stops; no third
architecture is attempted.

## 2. Architecture specs

GLRC (413,539 params): `W'_b = W_null_b + tanh(theta0 + theta^T g_G) *
(W_local_b - W_null_b)`, likewise for `V`.  `g_G` is the existing Fit-only
standardised 33-dim global structure vector; `theta0=theta=0` at init gives
`lambda == 0` and bitwise-identical BJOC-Null forwards.  Conservation:
`sum_b dW_b = sum_b dV_b = 0` per molecule.

DLP (441,153 params): `h_i = P a_i` (`32 x 96`), bucket pool
`p_k(G) = mean_{i<j: d(i,j) in {3,4,>=5}} (h_i * h_j)`, appended to the frozen
reader input with zero-initialised new first-layer columns.  Distances are
precomputed from the raw structure; there is no pair iteration.

## 3. Discipline

* Official test is never loaded (no code path exists); official valid is
  exposed development, descriptive only.
* Splits are internal group-aware re-partitions of official train 10k
  (primary C, replication D, salts frozen in the config).  These molecules
  have been used in historical experiments; the splits are *not* independent
  external tests and per-split overlap with the historical A/B splits is
  computed and disclosed in `split_{tag}.json`.
* Monitor: each candidate prediction set is frozen and hashed before the
  single declared read; scoring rounds and selection counts are recorded in
  `monitor_rounds_{tag}.json`.  Locked: one read after `locked-freeze`.
* `ab.LabelGuard` makes Monitor/Locked `.y` unreachable during preprocessing,
  SSL, training and prediction.  This round is the structural fix for the
  label-access defect recorded in
  `results/bjoc_performance_mechanism_long_v1/ERRATUM_label_access.md`.
* Budget: 6.0 GPU-hours total; conditional purchases only per the frozen
  gates; ledger `gpu_budget_ledger.json`.

## 4. Local CPU validation (before any GPU run)

The full `prep` pipeline ran locally on 2026-10-13: frozen input audit OK
(six hash checks), split C built with the pre-registered row fingerprints and
zero duplicate groups crossing sides, guarded objects + pair cache built
(1,668,045 fit pairs / 202,709 monitor pairs over the three distance buckets),
Fit-only standardiser, protocol freeze written, and the Stage-A init-parity
audit passed (P3 init full/group/encoder hashes match the historical anchors
for seeds 0 and 1; R3 recipe constants and batch-order hash match).

Both CPU gates passed (`cpu_gate_glrc.json`, `cpu_gate_dlp.json`):

* GLRC: exact identity at `lambda=0` (base parameter hash equal, forward diff
  <= 1e-5), gate gradient active, active-vs-zero-lambda prediction difference
  nonzero, conservation `sum dW <= 3.9e-6`, ownership counterexample
  (identical Null statistics, different local residual 3.1e-2, different
  active prediction), atom/bond/junction/row/batch invariances <= 1.2e-7,
  413,539 parameters, no forbidden modules.
* DLP: brute-force pair-construction cross-check, information witness (same
  first/second moments, different pair features), zero-init identity with
  copied old reader columns, non-degenerate features, reader-new-column
  gradient at step 0 and pair-projection gradient from step 1, prediction
  change, recoverability probe R2 median 0.723 <= 0.90, invariances <= 1.5e-5
  (pair row order and endpoint swap exact), 441,153 parameters, no forbidden
  modules.
* The matched-loop equivalence test proves `ab.train_supervised_arch` with
  `model_kind="base"` reproduces the frozen `bp.train_supervised` soup and
  batch order bitwise on a smoke trajectory, so P3 and the candidates share
  one training path.

## 5. Remote execution plan

`rr deploy res` from the committed revision; remote `prep` (freeze + audit)
and `cpu-gates` re-run on the server; GPU work per seed on a single verified
idle GPU: Fit-only masked SSL (split C) -> P3 baseline -> GLRC; predictions
frozen and hashed -> single declared Monitor read -> GLRC gate.  Conditional
purchases in order: ORGANIZED contrast (iff GLRC passes); DLP seed 0 (iff GLRC
fails); DLP seed 1 (iff DLP seed 0 passes); D-split replication (iff either
candidate passes its primary gate); full-10k refit only under the frozen
condition.  GPU0/GPU1 occupancy is re-probed before any launch (an external
user occupied GPU0 earlier today).

## 6. Results

(to be appended: Monitor contrasts, classification decisions, replication,
conditional arms, Locked table, budget ledger)
