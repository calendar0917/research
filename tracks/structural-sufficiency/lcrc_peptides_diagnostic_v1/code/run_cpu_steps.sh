#!/usr/bin/env bash
# CPU-only reproduction of the diagnostic round (no GPU, no Test data).
#
#   bash tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/code/run_cpu_steps.sh
#
# Writes wall-clock seconds per step to analysis/cpu_steps.json.  The isolated
# ECFP/LightGBM interpreter can be overridden with ECFP_PY=...
set -euo pipefail

ROUND="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROUND/../.." && pwd)"
cd "$REPO"

ECFP_PY="${ECFP_PY:-/home/calendar/.cache/research-venvs/peptides-ecfp/bin/python}"
TSV="$(mktemp)"
trap 'rm -f "$TSV"' EXIT

timed() {
  local name="$1"; shift
  local start end
  start=$(date +%s.%N)
  "$@"
  end=$(date +%s.%N)
  awk -v n="$name" -v a="$start" -v b="$end" 'BEGIN{printf "%s\t%.2f\n", n, b-a}' >> "$TSV"
}

mkdir -p "$ROUND/analysis"

timed p0_learning_curves uv run --project "$REPO" python "$ROUND/code/diag_p0_curves.py"
timed p0_label_statistics uv run --project "$REPO" python "$ROUND/code/diag_p0_labels.py"
timed p0_row_identity uv run --project "$REPO" python "$ROUND/code/diag_row_identity.py"
timed p0_frozen_checkpoint_eval uv run --project "$REPO" python "$ROUND/code/diag_p0_checkpoint_eval.py" \
  --seeds 0 1 2 --device cpu --out-dir "$ROUND/runs/p0_frozen_eval"
timed p1_ecfp_lightgbm "$ECFP_PY" "$ROUND/code/diag_p1_ecfp.py"
timed aggregate uv run --project "$REPO" python "$ROUND/code/diag_aggregate.py"

uv run --project "$REPO" python - "$TSV" "$ROUND/analysis/cpu_steps.json" "$ECFP_PY" <<'PY'
import json, pathlib, sys, time
tsv, out, ecfp_py = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
steps = {}
for line in tsv.read_text().splitlines():
    if not line.strip():
        continue
    name, seconds = line.split("\t")
    steps[name] = float(seconds)
payload = {
    "round_id": "lcrc-peptides-diagnostic-v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "host_note": "local CPU workstation (no GPU); wall clock per pipeline step",
    "ecfp_interpreter": ecfp_py,
    "steps_seconds": steps,
    "total_seconds": round(sum(steps.values()), 2),
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload["steps_seconds"], indent=2, sort_keys=True))
PY

echo "done; per-step seconds in $ROUND/analysis/cpu_steps.json"
