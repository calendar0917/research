#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export WP_THREADS=7 WP_DATASET=lipo WP_DEVICE=cpu WP_EPOCHS=240 WP_DEADLINE=1795249951
exec bash code/wp_queue.sh results/stage_d.jobs 2
