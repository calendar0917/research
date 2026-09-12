#!/usr/bin/env bash
# 只读探测 res 服务器的完整状态（无 sudo，全部用 /proc、ps、nvidia-smi 等可用手段）。
# 用法: ./scripts/resstatus.sh [host]   主机默认 res（见 ~/.ssh/config）
set -uo pipefail

HOST="${1:-res}"
SSH_OPTS=(-o ConnectTimeout=10 -o BatchMode=yes)

remote() {
  ssh "${SSH_OPTS[@]}" "$HOST" 'bash -s --' <<'RSCRIPT'
set -uo pipefail
H=()

sec() { printf '\n\033[1;33m========== %s ==========\033[0m\n' "$1"; }

sec "01 主机与系统"
echo "主机名: $(hostname)   内核: $(uname -r)"
echo "$(grep PRETTY /etc/os-release | cut -d= -f2- | tr -d '\"')"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)  x$(nproc)"
echo "开机: $(uptime -s)  (已运行 $(uptime -p 2>/dev/null | sed 's/up //'))"
NCPUS=$(nproc)
L1=$(awk '{print $1}' /proc/loadavg); L5=$(awk '{print $2}' /proc/loadavg); L15=$(awk '{print $3}' /proc/loadavg)
echo "负载: ${L1} / ${L5} / ${L15}  (1/5/15min, 核数:${NCPUS})"
if awk -v l="$L15" -v n="$NCPUS" 'BEGIN{exit !(l > n*0.5)}'; then H+=("⚠ 15min负载超过核数50%"); fi
echo "在线用户数: $(who | wc -l)   前4次登录:"; last -n 4 2>/dev/null | head -4

sec "02 内存与交换"
free -h
MAV=$(grep 'MemAvailable' /proc/meminfo | awk '{print $2}')
if awk -v m="$MAV" 'BEGIN{exit !(m < 8*1024*1024)}'; then H+=("⚠ 可用内存不足 8G"); fi

sec "03 磁盘"
df -hT | grep -vE "tmpfs|overlay|efivarfs|udev"
echo "-- inode:"; df -i | grep -E "sda2|sdb1"
echo "-- 关键目录大小(限时20s):"
timeout 20 du -sh ~/enter/envs/* ~/enter/pkgs 2>/dev/null | sort -rh | head -10
echo "-- IO (vmstat 1x2):"; vmstat 1 2 2>/dev/null | tail -2
if df -h /home | awk 'NR==2{u=$5; sub(/%/,"",u); exit !(u>90)}'; then H+=("⚠ /home 使用率过 90%"); fi

sec "04 GPU"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader 2>/dev/null || echo "无 nvidia-smi"
echo "-- 占用 GPU 的进程:"
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader 2>/dev/null | while IFS=, read -r pid mem; do
  ps -p "${pid// /}" -o pid=,user=,comm= 2>/dev/null || echo "  pid=${pid// /} (已退出)"
done
echo "-- 驱动/CUDA:"; nvidia-smi 2>/dev/null | sed -n '1,4p'
if nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | awk -F'[ %]' '{s+=$1} END{exit !(s/NR>50)}'; then
  H+=("⚠ GPU 平均利用率过 50%")
fi

sec "05 进程与负载"
echo "进程总数: $(ps -e --no-headers | wc -l)   僵尸: $(ps -eo state | grep -c Z || true)"
echo "-- 按 CPU 前八:"; ps -eo pid,user,pcpu,pmem,rss,comm --sort=-pcpu --no-headers | head -8
echo "-- 按内存前八:"; ps -eo pid,user,pcpu,pmem,rss,comm --sort=-pmem --no-headers | head -8
echo "-- dmesg 尾部(无 sudo 通常不可读):"
DM=$(dmesg -T 2>/dev/null | tail -3)
[ -n "$DM" ] && echo "$DM" || echo "  无权限或不可用"

sec "06 环境"
~/enter/bin/conda --version 2>/dev/null || echo "conda: 未找到"
for e in ~/enter/envs/*/; do
  [ -x "$e/bin/python" ] || continue
  name=$(basename "$e")
  ver=$("$e/bin/python" -V 2>&1)
  torch=$("$e/bin/python" -c "import torch;print(torch.__version__,'cuda:',torch.cuda.is_available())" 2>/dev/null || echo "无torch")
  printf "  %-18s %-11s torch: %s\n" "$name" "$ver" "$torch"
done
echo "系统 python: $(python3 -V 2>&1)"

sec "07 网络"
ip -brief addr 2>/dev/null | grep -v "LOOPBACK\|DOWN" || true
echo "-- 监听端口(前12):"; ss -tunl 2>/dev/null | awk 'NR>1{print $5}' | head -12 || true

sec "08 健康提示"
if (( ${#H[@]} == 0 )); then echo "  ✓ 未发现明显异常"; else for m in "${H[@]}"; do echo "  $m"; done; fi
RSCRIPT
}

remote
