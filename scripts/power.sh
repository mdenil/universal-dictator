#!/bin/bash
# M6 power draw: samples CPU / GPU / ANE power with powermetrics while each configuration transcribes
# in a loop (30 s each), plus a 20 s idle baseline. Needs root for powermetrics:
#   sudo scripts/power.sh        all four configurations on Metal
#   sudo scripts/power.sh ane    Nemotron 160 / 560 ms on the Neural Engine and on Metal, same session
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
run() {  # run <name> <env assignments or -> <dictate-cli args...>: loop the workload while sampling 30 s
  local name=$1 envs=$2; shift 2
  [ "$envs" = "-" ] && envs=""
  sudo -u "$U" bash -c "${envs:+export $envs; }while true; do $B $* >/dev/null 2>&1; done" & local W=$!
  sleep 3; sample "$name" 30
  kill $W; pkill -u "$U" -f "$B" || true; sleep 2
}

if [ "$1" = "ane" ]; then
  # Nemotron on the Neural Engine (stateful caches, and the stateless models with caches passed through
  # inputs/outputs) against Metal. A 5-minute file keeps process start-up out of the window, and warm-up
  # runs fill Core ML's Neural Engine compile cache first (a miss costs about 40 s of CPU).
  L=scratch/long_5min.wav
  echo "warming the Neural Engine compile cache"
  for lat in 160 560; do
    sudo -u "$U" $B transcribe $C --model nemotron --latency $lat --precision f16 --backend ane >/dev/null 2>&1
    sudo -u "$U" env DICTATOR_ANE_STATELESS=1 $B transcribe $C --model nemotron --latency $lat --precision f16 --backend ane >/dev/null 2>&1
  done
  echo "idle baseline (20 s)"; sample idle-ane 20
  for lat in 160 560; do
    echo "nemotron-$lat: Neural Engine, stateful";  run nemotron-$lat-ane-stateful - transcribe $L --model nemotron --latency $lat --precision f16 --backend ane
    echo "nemotron-$lat: Neural Engine, stateless"; run nemotron-$lat-ane-stateless DICTATOR_ANE_STATELESS=1 transcribe $L --model nemotron --latency $lat --precision f16 --backend ane
    echo "nemotron-$lat: Metal";                    run nemotron-$lat-metal - transcribe $L --model nemotron --latency $lat --precision f16
  done
  echo "done: $O"; exit 0
fi
echo "idle baseline (20 s)"; sample idle 20
echo "nemotron-160"; run nemotron-160 - transcribe $C --model nemotron --latency 160 --precision f16
echo "nemotron-560"; run nemotron-560 - transcribe $C --model nemotron --latency 560 --precision f16
echo "parakeet-h4";  run parakeet-h4 - transcribe $C --model parakeet --horizon 4 --precision f16
echo "voxtral-q8h";  run voxtral-q8h - voxtral $C --weights dictator-q8h.safetensors
echo "done: $O"
