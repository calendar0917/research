#!/usr/bin/env bash
# Sequential unit runner inside one rr job: one physical card, one process.
# Usage: trc_chain.sh <round-relative code dir> <unit spec file>
set -u
CODE="$1"; SPEC="$2"
while read -r run arm sup ds extra; do
  [ -z "${run:-}" ] && continue
  case "$run" in \#*) continue;; esac
  echo "=== chain unit $run arm=$arm sup=$sup ds=$ds extra=$extra $(date -Iseconds)"
  python "$CODE/trc_train.py" --run-id "$run" --arm "$arm" --supervision "$sup" \
      --dataset "$ds" --epochs 160 --device cuda $extra 2>&1 | tail -40
  echo "=== chain unit $run finished exit=$? $(date -Iseconds)"
done < "$SPEC"
echo "=== chain complete $(date -Iseconds)"
