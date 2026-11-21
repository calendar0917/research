#!/usr/bin/env bash
# Wait for the Stage D queue, then run the ESOL replication (all five Lipo arms).
set -uo pipefail
cd "$(dirname "$0")/.."
export WP_THREADS=7 WP_DATASET=esol WP_DEVICE=cpu WP_EPOCHS=240 WP_DEADLINE=1795249951
while pgrep -f "wp_queue.sh results/stage_d.jobs" >/dev/null; do sleep 20; done
sleep 10
exec bash code/wp_queue.sh results/stage_esol.jobs 2
