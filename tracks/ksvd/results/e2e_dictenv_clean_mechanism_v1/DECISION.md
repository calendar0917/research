# DECISION — e2e_dictenv_clean_mechanism_v1

Date: 2026-09-29 (CPU-only round, official test never loaded)
Preregistration: `notes/e2e_dictenv_clean_mechanism_v1_preregistration.md`
Evidence: `REPORT.md`, `analysis_tables.md`, `summary.json`, gate artifacts in `stage_*/`.

## Adopted

```text
CLEAN_BASE / FINAL-CLEAN = C6 mask
  (no graph atom/bond histogram, no path_count, no unary/pair count rows;
   anchor chemistry retained)
  + paired node binding (per-occurrence product)
  + paired edge binding (per-occurrence product)
  + sparse tied-IHT coding (s = 8, IHT-10)
```

No further simplification was adopted. The relation bundle stays complete
(distance + overlap + boundary).

## Because

* **Q1**: C6−BASE is negative on 3/3 matched seeds (−0.014799, −0.000116, −0.009229;
  mean −0.008048, max −0.000116, range 0.014683) → frozen gate `STRONG_SUPPORT`. The
  removal costs nothing anywhere; the C6 regime is also tighter (std 0.0043 vs 0.0071).
  The mean gain is dominated by the BASE seed-0 outlier, so the adoption is justified as a
  *safe cleaning*, not as a claimed improvement.
* **Q2**: the clean model reads fine correspondence, more strongly than BASE
  (frozen C6: edge +0.0401, node +0.0093 mean). From-scratch independence-null controls
  fail adoption for both roles: node mean +0.00434 but max +0.00925 (> 0.005); edge mean
  +0.00755 (> 0.003). Warm-start 20-epoch deltas agree (node +0.00261, edge +0.00833).
  "Shell marginals suffice" is rejected for edges and unconfirmed for nodes.
* **Q3**: distance-only relation costs +0.005661 at 320 epochs
  (`INCONCLUSIVE_KEEP_FULL`); the preregistered acceptable answer is the full bundle, and
  the data do not force anything below it.
* **Q4**: matched DenseTied is marginally better than sparse
  (`G_dict = −0.002936`, gate ≥ +0.003) → sparse-specificity not established; no dense
  extension. The sparse code remains a valid, load-bearing coordinate code
  (N6 code zero +0.27…+0.38) but is not shown to be specific.

## Not run (gates did not fire)

C1 fallback; seed-3 C6 extension; node/edge independence extensions beyond the prescribed
seeds 1–2; relation seeds 1–2; dense seeds 1–2; combined FINAL-CLEAN-SPARSE reference;
Stage E (shell simplification — its precondition "NODE-INDEP supported" is unmet); PCA32.

## Revisit only if

* a new preregistration asks about the **concentration** of the sparse code
  (top-5 atoms ≈ 60 % of activations, one atom active in nearly every row) and whether
  rebalancing it changes the Sparse/DenseTied comparison — with the matched dense control
  and ≥3 seeds;
* or a new mechanism question requires the C6 seed spread to be pinned down (a 4th seed
  would settle the near-threshold 0.0147 range), with the gate thresholds frozen in advance.
