#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

light=0
if [[ "${1:-}" == "--light" ]]; then
  light=1
  shift
fi
if [[ $# -ne 1 ]]; then
  echo "usage: $0 [--light] <repo-relative-result-path>" >&2
  exit 2
fi

rel="${1#./}"
if [[ "$rel" == /* || "$rel" == *".."* ]]; then
  echo "error: result path must be a safe repo-relative path" >&2
  exit 2
fi

ROOT="$(local_repo_root)"
dest="$ROOT/$rel"
mkdir -p "$(dirname "$dest")"

remote="${SSH_ALIAS}:${REMOTE_REPO%/}/$rel/"
args=(-avP)
if [[ $light -eq 1 ]]; then
  args+=(--exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.bin')
fi

ssh "$SSH_ALIAS" "test -d $(printf '%q' "${REMOTE_REPO%/}/$rel")" || {
  echo "error: remote result directory does not exist: $rel" >&2
  exit 3
}

rsync "${args[@]}" "$remote" "$dest/"
echo "pulled: $dest"
