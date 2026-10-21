# zinc-generalization-audit-v1

CPU-only audit of the ZINC official-valid generalisation plateau. No training,
no official-test read. See `REPORT.md` and `NEXT_STEP_PROMPT.txt`.

Quick start:

```bash
cd /home/calendar/code/research/tracks/structural-sufficiency/zinc_generalization_audit_v1
uv run python code/audit_inputs.py
uv run python code/audit_errors.py
uv run python code/audit_coverage.py
uv run python code/audit_topcases.py
uv run python code/audit_grouping_probe.py
uv run python code/proto_dual_channel.py
uv run python code/make_figures.py
```
