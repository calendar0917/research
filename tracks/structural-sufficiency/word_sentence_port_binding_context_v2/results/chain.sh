#!/usr/bin/env bash
# Wait for the Stage B/C queue, then run Stage D.
set -uo pipefail
cd "$(dirname "$0")/.."
export WP_THREADS=7 WP_DATASET=lipo WP_DEVICE=cpu WP_EPOCHS=240 WP_DEADLINE=1795249951
while pgrep -f "wp_queue.sh results/stage_bc.jobs" >/dev/null; do sleep 20; done
sleep 10
bash code/wp_queue.sh results/stage_d.jobs 2
