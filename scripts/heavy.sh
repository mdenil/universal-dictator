#!/bin/bash
# Run a memory-heavy command while holding the repo-wide model lock (macOS has no flock).
# Usage: scripts/heavy.sh <command...>
LOCK="$(cd "$(dirname "$0")/.." && pwd)/scratch/heavy.lock.d"
mkdir -p "$(dirname "$LOCK")"
until mkdir "$LOCK" 2>/dev/null; do
  # Break stale locks whose owner process is gone.
  if [ -f "$LOCK/pid" ] && ! kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then rm -rf "$LOCK"; continue; fi
  sleep 5
done
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
"$@"
