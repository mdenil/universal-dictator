#!/bin/bash
# L5 on the reference's own random subsets (data/subsets, from reference/nemo/wer.py), fp16.
cd "$(dirname "$0")/.."
B=.build/release/dictate-cli
while pgrep -f eval_queue_parakeet.sh > /dev/null; do sleep 30; done
for set in test-clean test-other; do
  $B eval --model nemotron-160 --precision f16 --manifest data/subsets/${set}_subset500.jsonl --out artifacts/eval/nemotron_160_f16_${set}_ref500.jsonl
  $B eval --model nemotron-560 --precision f16 --manifest data/subsets/${set}_subset500.jsonl --out artifacts/eval/nemotron_560_f16_${set}_ref500.jsonl
  $B eval --model parakeet --precision f16 --manifest data/subsets/${set}_subset200.jsonl --out artifacts/eval/parakeet_exact_f16_${set}_ref200.jsonl
  $B eval --model parakeet-h4 --precision f16 --manifest data/subsets/${set}_subset200.jsonl --out artifacts/eval/parakeet_h4_f16_${set}_ref200.jsonl
done
