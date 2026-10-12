#!/bin/bash
# Recomputes SHA-256 of pinned checkpoint files and compares with docs/PINS.md.
# Usage: scripts/verify_pins.sh [--write]   (--write prints the table rows to paste into PINS.md)
cd "$(dirname "$0")/.."
files=(
  models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo
  models/nemotron-speech-streaming-en-0.6b/model.safetensors
  models/nemotron-speech-streaming-en-0.6b/config.json
  models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b.nemo
  models/Voxtral-Mini-4B-Realtime-2602/model.safetensors
  models/Voxtral-Mini-4B-Realtime-2602/params.json
  models/Voxtral-Mini-4B-Realtime-2602/config.json
  models/Voxtral-Mini-4B-Realtime-2602/tekken.json
)
status=0
for f in "${files[@]}"; do
  # A checkpoint you haven't downloaded (the app needs only Nemotron) is reported, not failed.
  if [ ! -f "$f" ]; then echo "--   $f (not downloaded)"; continue; fi
  h=$(shasum -a 256 "$f" | cut -d' ' -f1)
  if [ "$1" == "--write" ]; then echo "| \`$f\` | \`$h\` |"; continue; fi
  if grep -q "$h" docs/PINS.md; then echo "ok   $f"; else echo "FAIL $f ($h)"; status=1; fi
done
exit $status
