#!/bin/bash
# M4: Voxtral WER per quantization stage on 100-utterance subsets (first 100 of each set).
cd "$(dirname "$0")/.."
B=.build/release/dictate-cli
for set in test-other test-clean; do
  for w in q8 q4ffn bf16; do
    f=dictator-$w.safetensors; [ "$w" = "bf16" ] && f=model.safetensors
    $B eval --model voxtral --weights $f --manifest data/librispeech/$set.jsonl --out artifacts/eval/voxtral_${w}_${set}_100.jsonl --limit 100
  done
done
