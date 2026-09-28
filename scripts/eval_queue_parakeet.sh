#!/bin/bash
# Parakeet Stage A vs Stage B (frozen-prefix horizons) on 200-utterance subsets, fp16 weights.
cd "$(dirname "$0")/.."
B=.build/release/dictate-cli
for set in test-clean test-other; do
  for mode in exact h0 h4 h16; do
    out=artifacts/eval/parakeet_${mode}_f16_${set}_200.jsonl
    if [ "$mode" == "exact" ]; then extra=""; else extra="--horizon ${mode#h}"; fi
    $B eval --model parakeet --precision f16 $extra --manifest data/librispeech/$set.jsonl --out $out --limit 200
  done
done
for set in test-clean test-other; do
  $B eval --model nemotron --latency 160 --precision f16 --manifest data/librispeech/$set.jsonl --out artifacts/eval/nemotron_160_f16_${set}_200.jsonl --limit 200
  $B eval --model nemotron --latency 560 --precision f16 --manifest data/librispeech/$set.jsonl --out artifacts/eval/nemotron_560_f16_${set}_200.jsonl --limit 200
done
reference/nemo/.venv/bin/python eval/score.py artifacts/eval/*_200.jsonl 2>/dev/null
