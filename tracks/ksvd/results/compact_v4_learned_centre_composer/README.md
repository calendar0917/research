# compact_v4_learned_centre_composer (P1) — results

Minimal falsification of the compact-v4 fixed statistical centre-composition
bottleneck.

**Verdict: Decision Case A — P1 MINIMAL LEARNED COMPOSER NO-GO.**

- P1 seed0 official-valid MAE **0.147229** (best epoch 151, 191 epochs,
  15,089 optimizer steps, 1,300 s) vs reused optimized compact-v4 seed0
  reference **0.146420** → `Delta_P1,0 = -0.000809`, far below the
  pre-registered `+0.004` architecture gate.
- Branch alive/trained: residual-zero inference ablation max|Δ| = 0.531;
  `phi_grad` max 6.0e-3, `rho_grad` max 0.177; learned residual norm
  0.078 → 0.81; residual/fixed ratio 0.373 at the selected checkpoint.
- All 17 Stage-0 integrity gates pass; fixed pooling reused verbatim; seed0
  baseline initial tensors matched exactly (48 shared tensors, max diff 0.0);
  common-input bulk safe (v4 0.09472 vs P1 0.09358).
- P1 = 102,601 params (99,613 + 2,988); `R` unchanged at 302D; centre-update
  input unchanged at 213D.
- Matched post-pooling capacity control **not purchased**; seed1 **not run**;
  seeds 2/3 forbidden; **official test never loaded**.

Artifacts: `architecture_lock.json`, `baseline_inventory.json`,
`parameter_audit.json`, `initialization_match.json`, `integrity_gates.json`,
`compute_audit.json`, `stage1_p1_seed0.json`, `stage1_decision.json`,
`common_input_bulk.json`, `composition_diagnostics.json`, `final_decision.json`,
`answers_q1_q20.json`, `curves/p1_seed0_curve.csv`, `figures/`.

Code: `../../../experiments/luyin16/zinc_compact_v4_learned_centre_composer.py`
Note: `../../../notes/compact_v4_learned_centre_composer.md`
Tests: `../../../tests/test_compact_v4_learned_centre_composer.py` (22 pass)
