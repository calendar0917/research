#!/usr/bin/env bash
# Sequential queue driver for one GPU pool.
#
#   code/queue.sh <pool> <queue-file> <log-file>
#
# Queue lines:  <run_id> <arm> <split> <seed> [epochs|-]
# It launches one rr job at a time on the given pool, waits for a terminal
# state, and records the outcome.  Safe to run two instances (one per GPU).
set -u
POOL="$1"; QUEUE="$2"; LOG="$3"
ROUND="tracks/structural-sufficiency/structural_path_composition_route_v1"
: >> "$LOG"

status_of() {
  rr status "psc-$1" --json 2>/dev/null | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    print(""); raise SystemExit(0)
print(d.get("state",""))
'
}

while read -r run_id arm split seed epochs; do
  [ -z "${run_id:-}" ] && continue
  case "$run_id" in \#*) continue;; esac
  args=(--run-id "$run_id" --arm "$arm" --seed "$seed" --split "$split" --device cuda)
  if [ -n "${epochs:-}" ] && [ "$epochs" != "-" ]; then args+=(--epochs "$epochs"); fi
  echo "[$(date +%H:%M:%S)] LAUNCH $run_id arm=$arm split=$split seed=$seed epochs=${epochs:--} pool=$POOL" >> "$LOG"
  rr run res "psc-$run_id" --pool "$POOL" --result "$ROUND/results/$run_id" --time 90m -- \
     python "$ROUND/code/psc_train.py" "${args[@]}" >> "$LOG" 2>&1
  st=""
  for _ in $(seq 1 1440); do
    st=$(status_of "$run_id")
    case "$st" in completed|failed|cancelled) break;; esac
    sleep 5
  done
  echo "[$(date +%H:%M:%S)] DONE $run_id state=$st" >> "$LOG"
done < "$QUEUE"
echo "[$(date +%H:%M:%S)] QUEUE COMPLETE $QUEUE" >> "$LOG"