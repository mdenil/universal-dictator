#!/bin/bash
# M6 power draw: samples CPU / GPU / ANE power with powermetrics while each configuration transcribes
# in a loop (30 s each), plus a 20 s idle baseline. Needs root for powermetrics:
#   sudo scripts/power.sh
# The workloads run as the invoking user. Output: artifacts/power/<config>.txt
set -e
cd "$(dirname "$0")/.."
[ "$(id -u)" = 0 ] || { echo "run with sudo: sudo scripts/power.sh"; exit 1; }
U=${SUDO_USER:?run via sudo}
B=.build-dev/release/dictate-cli
C=data/librispeech/LibriSpeech/test-clean/1995/1836/1995-1836-0004.flac
O=artifacts/power; sudo -u "$U" mkdir -p $O

sample() {  # sample <name> <seconds>
  powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n "$2" > "$O/$1.txt" 2>/dev/null
  chown "$U" "$O/$1.txt"
}
run() {  # run <name> <dictate-cli args...>: loop the workload while sampling 30 s
  local name=$1; shift
  sudo -u "$U" bash -c "while true; do $B $* >/dev/null 2>&1; done" & local W=$!
  sleep 3; sample "$name" 30
  kill $W; pkill -u "$U" -f "$B" || true; sleep 2
}

echo "idle baseline (20 s)"; sample idle 20
echo "nemotron-160"; run nemotron-160 transcribe $C --model nemotron --latency 160 --precision f16
echo "nemotron-560"; run nemotron-560 transcribe $C --model nemotron --latency 560 --precision f16
echo "parakeet-h4";  run parakeet-h4  transcribe $C --model parakeet --horizon 4 --precision f16
echo "voxtral-q8h";  run voxtral-q8h  voxtral $C --weights dictator-q8h.safetensors
echo "done: $O"
