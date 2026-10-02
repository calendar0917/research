#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <local-path> <remote-absolute-or-repo-relative-path>" >&2
  exit 2
fi

src="$1"
target="$2"
if [[ ! -e "$src" ]]; then
  echo "error: local source does not exist: $src" >&2
  exit 2
fi

if [[ "$target" == /* ]]; then
  remote_target="$target"
else
  remote_target="${REMOTE_REPO%/}/${target#./}"
fi

ssh "$SSH_ALIAS" "mkdir -p $(printf '%q' "$remote_target")"
if [[ -d "$src" ]]; then
  rsync -avP "$src/" "${SSH_ALIAS}:${remote_target%/}/"
else
  rsync -avP "$src" "${SSH_ALIAS}:${remote_target%/}/"
fi

echo "synced to ${SSH_ALIAS}:$remote_target"
