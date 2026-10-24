# Byte-identical pre-fix snapshot (baseline commit `ed70416`)

These files are extracted verbatim from
`tracks/structural-sufficiency/zinc_structural_factor_contraction_v0/code/`
at commit `ed70416` ("sfc-v0: SFC mathematical/computational feasibility
audit (CPU only)"). They exist only so the corrigendum checks can be re-run
against the **original** implementation and compared with the fixed one.

* `sfc_core.py` / `sfc_synth.py` — the two defective modules. Do not edit;
  the live fixed versions live one directory up.
* `sfc_checks.py` — the original T1 gate (not defective; kept so the snapshot
  is self-contained).

`code/sfc_corrigendum_checks.py --code-dir code/legacy_pre_fix` loads these
modules by path (never via `sys.path` shadowing) and records the pre-fix
failures in `analysis/t0_corrigendum_prefix.json`.
