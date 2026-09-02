#!/usr/bin/env bash
# 运行 GPS + MOLHIV 的 paper-seeds-provisional 全部 4 个 seed (0,1,2,3)
# seed 0 可能已在运行；如果日志已存在且进程活着则跳过。
# 每个 seed 完成后自动调用 collect_gps_results.py 写入 registry.jsonl。
set -u

VENV="/home/calendar/code/research/tracks/hod-gnn-replication/.venv-gps/bin/python"
REPO="/home/calendar/code/research/tracks/hod-gnn-replication/code/vendor/GraphGPS"
RES="/home/calendar/code/research/tracks/hod-gnn-replication/results/paper-seeds"
COLLECT="/home/calendar/code/research/tracks/hod-gnn-replication/code/collect_gps_results.py"

LOG_PREFIX="$RES/gps-molhiv-seed"

seed_finished() {
    local s="$1"
    local log="${LOG_PREFIX}${s}.log"
    grep -q "> Epoch 99:" "$log" 2>/dev/null
}

seed_running() {
    ps aux | grep -F "seed $1" | grep -v grep | grep -v run_gps_molhiv >/dev/null 2>&1
}

for SEED in 0 1 2 3; do
    LOG="${LOG_PREFIX}${SEED}.log"
    if seed_finished "$SEED"; then
        echo "[queue] seed $SEED already finished, collecting"
        "$VENV" "$COLLECT" --log "$LOG" --method gps --dataset molhiv \
            --protocol paper-seeds-provisional --seed "$SEED" --train-seconds -1
        continue
    fi
    if seed_running "$SEED"; then
        echo "[queue] seed $SEED already running (waiting...)"
        while seed_running "$SEED" && ! seed_finished "$SEED"; do
            sleep 60
        done
        sleep 30
        echo "[queue] $(date) seed $SEED finished externally, collecting"
        "$VENV" "$COLLECT" --log "$LOG" --method gps --dataset molhiv \
            --protocol paper-seeds-provisional --seed "$SEED" --train-seconds -1
        continue
    fi
    echo "[queue] $(date) starting seed $SEED"
    cd "$REPO"
    "$VENV" main.py --cfg configs/GPS/ogbg-molhiv-GPS+RWSE.yaml \
        wandb.use False "seed ${SEED}" >> "$LOG" 2>&1
    echo "[queue] $(date) seed $SEED done, collecting"
    "$VENV" "$COLLECT" --log "$LOG" --method gps --dataset molhiv \
        --protocol paper-seeds-provisional --seed "$SEED" --train-seconds -1
done

echo "[queue] all seeds done: $(date)"