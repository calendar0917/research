# DECISION — SOC-v1

## 2026-10-03 — pre-registration

**Status: screen not started. GPU spend 0 A100·h.**

Architecture, arms, gates, recipe and budget are frozen in `PROTOCOL.md` and
`configs/soc_v1.json` before any GPU training. CPU gates pass:

* exact DCC-L initialisation (max abs diff 0.0; CTRL 2.4e-7),
* 512,888 parameters (R/T), 513,021 (CTRL),
* finite forward/backward, all closure parameters receive gradient,
* batch-composition invariance 3e-8,
* K=3 trajectory is non-trivial, K=1 vs K=3 parameter count identical,
* 8-molecule overfit 0.989 → 0.091 in 60 CPU steps,
* open-gate forward finite.

The screen will be read only against the pre-registered gates. No response to
Monitor beyond the stop/advance rule.

## 2026-10-28 — final decision

**STOP_NO_GAIN.** Both pre-registered screen arms fail; full-10k confirmation
and official-test read were not purchased.

| arm | seed-mean Monitor | Δ vs frozen L | 95% CI | gate |
|---|---:|---:|---|---|
| R (final state) | 0.125259 | −0.000203 | [−0.002516, +0.002076] | FAIL (bar 0.1215) |
| T (trajectory mean) | 0.127733 | +0.002271 | [−0.000270, +0.005042] | FAIL |

Mechanism check: the learned gates stayed at their 0.006693 initialisation, and
the train loss is minimised there. Forcing the trained closure open moves
Monitor by at most ~0.0022 and still does not beat the frozen DCC-L anchor. The
operator is real but far too weak; the SOC-trained backbone with the closure off
is worse than DCC-L, i.e. the branch perturbs optimisation and only partially
compensates.

Spend: 2.924 A100·h (R 1.450, T 1.472, smoke 0.001). No CTRL (conditional on a
pass), no full 10k, no test. See `REPORT.md` §5–§8 for the mechanism and the
next-step options (label-count slope; coverage-aware objective; no more
moment-closure variants).
