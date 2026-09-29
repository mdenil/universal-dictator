#!/bin/bash
# M6: every shipping configuration on the spontaneous/accented slices (AMI, Earnings-22), Nemotron with
# the D4 end-of-stream tail on the latency subsets, and Voxtral q8h (int8 + int8 head) on LibriSpeech
# and on the reference's own 100-utterance subsets. Uses a copied binary so rebuilds cannot interfere.
cd "$(dirname "$0")/.."
B=scratch/bin/dictate-cli-m6
E=artifacts/eval
H=scripts/heavy.sh
for set in ami earnings22; do
  M=data/subsets/${set}_150.jsonl
  $H $B eval --model nemotron-160 --manifest $M --out $E/nemotron_160d4_f16_${set}_150.jsonl
  $H $B eval --model nemotron-560 --manifest $M --out $E/nemotron_560d4_f16_${set}_150.jsonl
  $H $B eval --model parakeet --manifest $M --out $E/parakeet_exact_f16_${set}_150.jsonl
  $H $B eval --model parakeet-h4 --manifest $M --out $E/parakeet_h4_f16_${set}_150.jsonl
  $H $B eval --model voxtral --weights dictator-q8h.safetensors --manifest $M --out $E/voxtral_q8h_${set}_150.jsonl
done
for set in test-clean test-other; do
  $H $B eval --model nemotron-160 --manifest data/librispeech/$set.jsonl --limit 200 --out $E/nemotron_160d4_f16_${set}_200.jsonl
  $H $B eval --model nemotron-560 --manifest data/librispeech/$set.jsonl --limit 200 --out $E/nemotron_560d4_f16_${set}_200.jsonl
  $H $B eval --model voxtral --weights dictator-q8h.safetensors --manifest data/librispeech/$set.jsonl --limit 100 --out $E/voxtral_q8h_${set}_100.jsonl
  $H $B eval --model voxtral --weights dictator-q8h.safetensors --manifest data/subsets/${set}_voxref100.jsonl --out $E/voxtral_q8h_${set}_voxref100.jsonl
done
echo "m6 queue done $(date)"
