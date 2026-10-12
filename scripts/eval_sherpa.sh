#!/bin/bash
# sherpa-onnx (ONNX Runtime CPU) Nemotron 160 ms, int8 and fp32 exports, on the 200-utterance subsets.
# Needs the environment (cd reference/competitors && uv sync) and the exports, at the revisions in docs/PINS.md:
#   hf download csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-2026-04-25 \
#       --revision 9660703a1289d5b5c0c6470ced80eb52096fc740 --local-dir models/competitors/sherpa-nemotron-160ms
#   hf download csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25 \
#       --revision 237e551abd7a411ef92d3595454d9f6ab5fe7d6c --local-dir models/competitors/sherpa-nemotron-160ms-int8
cd "$(dirname "$0")/.."
P=reference/competitors/.venv/bin/python
for set in test-clean test-other; do
  for v in int8 fp32; do
    d=models/competitors/sherpa-nemotron-160ms
    if [ "$v" = "int8" ]; then d=$d-int8; fi
    $P reference/competitors/sherpa_eval.py $d $v data/librispeech/$set.jsonl artifacts/eval/sherpa-nemotron_160_${v}_${set}_200.jsonl --limit 200
  done
done
