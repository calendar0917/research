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
