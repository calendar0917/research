#!/usr/bin/env bash
# Pull every completed run listed in the given queue files into the repo tree.
#   code/pull_all.sh code/queue_*.txt
set -u
ROUND="tracks/structural-sufficiency/structural_path_composition_route_v1"
args=()
for q in "$@"; do args+=("$(cd "$(dirname "$q")" && pwd)/$(basename "$q")"); done
cd "$(dirname "$0")/../../../.." || exit 1
for q in "${args[@]}"; do
  while read -r run_id _rest; do
    [ -z "$run_id" ] && continue
    case "$run_id" in \#*) continue;; esac
    if [ -f "$ROUND/results/$run_id/manifest.json" ]; then continue; fi
    out=$(rr pull "psc-$run_id" 2>&1)
    echo "$out" | tail -n 2
    d=$(ls -dt .rr/pulled/psc-"$run_id"-* 2>/dev/null | head -1)
    if [ -n "$d" ] && [ -d "$d/result/$ROUND/results/$run_id" ]; then
      mkdir -p "$ROUND/results/$run_id"
      cp -a "$d/result/$ROUND/results/$run_id/." "$ROUND/results/$run_id/"
      echo "synced $run_id"
    fi
  done < "$q"
done