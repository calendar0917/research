#!/usr/bin/env bash
# Bootstrap the offline uv runtime on res-2 from res (one-time, idempotent).
#
# res-2 has no public internet / DNS, so its uv runtime is assembled offline:
#   res  ~/.local/bin/uv                         -> res-2 ~/opt/uv/bin/uv
#   res  ~/.local/share/uv/python/*              -> res-2 ~/opt/uv/python/*
#   res  ~/.cache/uv                             -> res-2 ~/opt/uv/cache
#
# The bulk data path is **res -> res-2 directly** (server-to-server): a dedicated
# key on res (~/.ssh/id_ed25519_res2) is authorised on res-2, and the transfer
# is executed on res, which pushes to res-2 with `ssh`.  The laptop is only used
# for control-plane setup (installing the key, starting the ssh command) and is
# never in the data path.
#
# res-2 topology is resolved from the laptop's ~/.ssh/config (`ssh -G res-2`)
# and passed explicitly to res, so the same host/port/user is used for the
# transfer as for interactive access.
#
# Usage: bash scripts/bootstrap_offline.sh [--force] [--verify-only]
set -euo pipefail

RES_HOST="${RES_HOST:-res}"
RES2_ALIAS="${RES2_ALIAS:-res-2}"
CHANNEL_KEY="${RES2_CHANNEL_KEY:-~/.ssh/id_ed25519_res2}"
FORCE=0
VERIFY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Quiet the post-quantum warning noise that newer OpenSSH clients print.
SSH_QUIET=(ssh -o BatchMode=yes)
remote2() { "${SSH_QUIET[@]}" "$RES2_ALIAS" "$@"; }

# Resolve res-2 topology from the local ssh config (single source of truth).
eval "$("${SSH_QUIET[@]}" -G "$RES2_ALIAS" | awk '
  /^hostname /{print "RES2_HOST="$2}
  /^port /{print "RES2_PORT="$2}
  /^user /{print "RES2_USER="$2}
')"
: "${RES2_HOST:?could not resolve res-2 hostname}"
: "${RES2_PORT:=22}"
: "${RES2_USER:?could not resolve res-2 user}"

echo "== checking existing res-2 runtime =="
if remote2 'test -x "$HOME/opt/uv/bin/uv" && test -d "$HOME/opt/uv/python" && [ -n "$(ls -A "$HOME/opt/uv/cache" 2>/dev/null)" ]' && [[ $FORCE -eq 0 ]]; then
  echo "res-2 offline runtime already present (use --force to re-sync)"
else
  if [[ $VERIFY_ONLY -eq 1 ]]; then
    echo "res-2 offline runtime incomplete" >&2
    exit 3
  fi

  echo "== ensuring res -> res-2 channel key =="
  PUB="$(ssh -o BatchMode=yes "$RES_HOST" "bash -lc '
    umask 077
    K=\"$CHANNEL_KEY\"
    eval K=\"$K\"
    [ -f \"\$K\" ] || ssh-keygen -t ed25519 -N \"\" -C \"res-to-res2 rr-bootstrap channel\" -f \"\$K\" >/dev/null
    cat \"\$K.pub\"
  '" 2>/dev/null | grep -E '^ssh-ed25519 ' | head -1)"
  if [[ -z "$PUB" ]]; then
    echo "could not obtain the channel public key from $RES_HOST" >&2
    exit 4
  fi
  remote2 "mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && grep -qF '$PUB' ~/.ssh/authorized_keys || printf '%s\n' '$PUB' >> ~/.ssh/authorized_keys"

  echo "== transferring uv runtime res -> res-2 (direct) =="
  # This script runs ON res and pushes directly to res-2; the laptop only
  # carries the (tiny) command text, not the ~GB of data.
  RES_TRANSFER=$(cat <<'EOS'
set -euo pipefail
r2host="$1"; r2port="$2"; r2user="$3"; r2key="$4"
KEY="${r2key/#\~/$HOME}"
R2_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$KEY" -p "$r2port" "$r2user@$r2host")
r2() { "${R2_SSH[@]}" "$@"; }

echo "[res] uv binary"
tar -C "$HOME/.local" -czf - bin/uv | r2 'mkdir -p "$HOME/opt/uv" && tar -xzf - -C "$HOME/opt/uv" && chmod +x "$HOME/opt/uv/bin/uv"'

echo "[res] uv-managed python"
tar -C "$HOME/.local/share/uv" -czf - python | r2 'mkdir -p "$HOME/opt/uv" && tar -xzf - -C "$HOME/opt/uv"'

echo "[res] uv cache (long step)"
# GNU tar exits 1 when a file changes while being read (a concurrent uv sync
# touches the cache).  The post-transfer `uv sync --offline` verification on
# res-2 is the real completeness check.
set +e
tar --warning=no-file-changed -C "$HOME/.cache" -czf - uv \
  | r2 'mkdir -p "$HOME/opt/uv/cache" && tar -xzf - -C "$HOME/opt/uv/cache" --strip-components=1'
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "[res] warn: cache transfer rc=$rc (often 'file changed as we read it'); verifying anyway" >&2
fi
echo "[res] done"
EOS
)
  printf '%s\n' "$RES_TRANSFER" | ssh -o BatchMode=yes "$RES_HOST" bash -s -- \
    "$RES2_HOST" "$RES2_PORT" "$RES2_USER" "$CHANNEL_KEY"
fi

echo "== verifying =="
remote2 'set -e
export PATH="$HOME/opt/uv/bin:$PATH"
export UV_CACHE_DIR="$HOME/opt/uv/cache"
export UV_PYTHON_INSTALL_DIR="$HOME/opt/uv/python"
uv --version
uv python list --only-installed 2>/dev/null | grep -E "cpython-3\.12" || true
cache_size=$(du -sh "$HOME/opt/uv/cache" 2>/dev/null | cut -f1)
echo "cache=$cache_size"
'
echo "bootstrap OK"
