# structural_computation_gap_analysis

Same-Budget Structural Computation Gap Analysis (static; no training, no official test).

Question: why can a similarly sized higher-order reference (CIN-small) use ~100K parameters
more effectively than optimized compact-v4, at the level of the real computation graph?

Verdict: **Case A - ONE REPRESENTATION HYPOTHESIS AUTHORIZED (design only)**.
Top-1 = incidence-preserving structural persistence of explicit higher-order (cycle) objects.

## Files
- `source_inventory.json` - primary-source traceability (CIN paper + official code + local code/records)
- `benchmark_comparability.json` - PARTIALLY VERIFIED (same data/split/target/metric; different protocol/budget)
- `compact_v4_computation_graph.json` - exact reconstruction from `zinc_patch_path_pooling.py`
- `reference_computation_graph.json` - exact reconstruction from the official CWN repo + paper Eq. 4
- `structural_object_lifecycle_v4.json` / `structural_object_lifecycle_reference.json` - side-by-side object lifecycles
- `parameter_allocation_v4.csv` (99,613) / `parameter_allocation_reference.csv` (130,945)
- `compute_profile_v4.json` / `compute_profile_reference.json` - object counts, operator applications, cost
- `gap_evidence_matrix.csv` - G1..G8 with for/against evidence and confounds
- `top1_hypothesis.json` - the single falsifiable representation-family hypothesis
- `minimal_falsification_plan.json` - minimal architecture + seed0 gates (design only)
- `final_decision.json` - Decision Case A (+ flagged sub-condition)
- `answers_q1_q20.json` - the 20 mandated questions
- `figures/` - Figure 1 (v4 graph), Figure 2 (reference graph), Figure 3 (lifecycle/persistence comparison)

Note: "Structural Persistence" and "Incidence Persistence" are analysis terms introduced in this audit.
