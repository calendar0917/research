#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

ROOT="$(local_repo_root)"
cd "$ROOT"

branch="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
if [[ -z "$branch" ]]; then
  echo "error: detached HEAD; create/check out a branch before formal deploy" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: local worktree is dirty; commit changes before formal remote deploy" >&2
  git status --short >&2
  exit 2
fi

commit="$(git rev-parse HEAD)"
echo "local branch: $branch"
echo "local commit: $commit"

echo "pushing origin/$branch ..."
git push origin "HEAD:refs/heads/$branch"

ssh "$SSH_ALIAS" bash -s -- "$REMOTE_REPO" "$branch" "$commit" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
repo="$1"
branch="$2"
commit="$3"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'error: remote tracked worktree is dirty; refusing to overwrite' >&2
  git status --short >&2
  exit 3
fi
git fetch origin "$branch"
if git show-ref --verify --quiet "refs/heads/$branch"; then
  git checkout "$branch"
else
  git checkout -b "$branch" --track "origin/$branch"
fi
git merge --ff-only "$commit"
uv sync --frozen
actual="$(git rev-parse HEAD)"
printf 'remote commit: %s\n' "$actual"
[[ "$actual" == "$commit" ]]
REMOTE

echo "deploy OK"
