# DECISION — zinc-upstream-portfolio-v1 (2026-10-02)

**Verdict: NO PURCHASED CANDIDATE. All three arms closed; fresh-320 not run.**

- CODE → `CODE_CONDITIONAL_STOP` (cached frozen-parent convex proxy: valid `0.115069` vs H39
  `0.115024`, gain `−0.000045`, groups 3/5; required `≤0.112` and `≥0.003`).
- DROP → warm-80 gate FAIL (cal `0.115925` vs control `0.114399`, gain `−0.001526`, groups 1/5).
- R3 → warm-80 gate FAIL (cal `0.113868` vs control `0.114399`, gain `+0.000531`, groups 4/5,
  ex-172 `+0.000442`; both `≤0.111` and `≥0.004` missed).

Not permitted without a new preregistration: extra seeds; scanning K / s / λ / dropout-p / shell
radius / LR / horizon; combining arms or ensembling; any 320-epoch run; reopening official test.

Revisit only if a new preregistration explains mechanistically why these frozen proxies cannot
move the endpoint and carries the same-checkpoint replayed parent as the matched baseline.

Rationale and full evidence: `REPORT.md`, `../..//notes/zinc_upstream_portfolio_v1_analysis.md`,
`pilot_summary.json`, `valid_predictions.npz`, `stage0_code.json`.