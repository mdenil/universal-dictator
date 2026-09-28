#!/bin/bash
# sherpa-onnx (ONNX Runtime CPU) Nemotron 160 ms, int8 and fp32 exports, on the 200-utterance subsets.
cd "$(dirname "$0")/.."
P=reference/competitors/.venv/bin/python
for set in test-clean test-other; do
  for v in int8 fp32; do
    d=models/competitors/sherpa-nemotron-160ms
    if [ "$v" = "int8" ]; then d=$d-int8; fi
    $P reference/competitors/sherpa_eval.py $d $v data/librispeech/$set.jsonl artifacts/eval/sherpa-nemotron_160_${v}_${set}_200.jsonl --limit 200
  done
done
