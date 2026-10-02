#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

printf '== ssh ==\n'
ssh -o BatchMode=yes "$SSH_ALIAS" 'printf "connected: %s@%s\n" "$USER" "$(hostname)"'

printf '\n== remote repo ==\n'
ssh "$SSH_ALIAS" "$REMOTE_ENV && cd $(printf '%q' "$REMOTE_REPO") && git rev-parse HEAD && git status --short && uv --version && uv run python --version"

printf '\n== GPUs ==\n'
ssh "$SSH_ALIAS" "nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader"

printf '\n== PyTorch CUDA ==\n'
ssh "$SSH_ALIAS" "$REMOTE_ENV && cd $(printf '%q' "$REMOTE_REPO") && uv run python - <<'PY'
import torch
print('torch=', torch.__version__)
print('torch_cuda=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
print('gpu_count=', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'gpu[{i}]={p.name}, memory_gib={p.total_memory/1024**3:.1f}')
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit('expected two CUDA GPUs')
PY"

printf '\n== research doctor ==\n'
set +e
ssh "$SSH_ALIAS" "$REMOTE_ENV && cd $(printf '%q' "$REMOTE_REPO") && uv run research doctor"
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo
  echo "preflight incomplete: 'research doctor' failed (commonly data.zinc on a fresh server)." >&2
  exit "$status"
fi

echo
echo "preflight OK"
