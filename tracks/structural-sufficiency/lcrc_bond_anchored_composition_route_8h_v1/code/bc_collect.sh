#!/usr/bin/env bash
# Merge every pulled run snapshot of this round into results/ (one writer per
# unit directory, so a plain overwrite-rsync is exact).  The snapshots stay in
# .rr/pulled/<run_id>/ untouched as the primary evidence.
set -euo pipefail
round=lcrc_bond_anchored_composition_route_8h_v1
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../../.." && pwd)"
mkdir -p "$here/results"
shopt -s nullglob
for snap in "$repo"/.rr/pulled/bajc-*/result/tracks/structural-sufficiency/$round/results; do
  rsync -a --exclude '.gitkeep' "$snap"/ "$here/results"/
done
find "$here/results" -name status.json | sort | sed 's|.*/results/||'
