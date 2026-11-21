#!/usr/bin/env bash
# Run a queue of training jobs, at most two at a time (the round's two-GPU budget
# maps to two local workers).  Usage: wp_queue.sh <jobs-file> [parallel]
# jobs-file lines: "<ARM> <SEED> [extra wp_run args...]"; '#' comments allowed.
set -uo pipefail

ROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${1:?jobs file required}"
PARALLEL="${2:-2}"
LOG="$ROUND_DIR/results/queue.log"

mkdir -p "$ROUND_DIR/results"
: >"$LOG.queue"
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' >"$LOG.queue"

running=0
while IFS= read -r line; do
  # shellcheck disable=SC2086
  set -- $line
  ARM="$1"; SEED="$2"; shift 2
  while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 5; done
  echo "[$(date -u +%H:%M:%S)] start $ARM seed=$SEED $*" >>"$LOG"
  ( bash "$ROUND_DIR/code/wp_launch.sh" "$ARM" "$SEED" "$@" >>"$LOG" 2>&1 ) &
  sleep 3
done <"$LOG.queue"
wait
echo "[$(date -u +%H:%M:%S)] queue done" >>"$LOG"
