# Universal Dictator

Universal Dictator is a local streaming speech-to-text engine for Apple-silicon Macs. It is written in Swift, with hand-written Metal kernels, and runs three open-weight models from their published checkpoints:

| Model | Streaming mode | Default latency |
|---|---|---|
| Nemotron Speech Streaming English 0.6B | cache-aware FastConformer + RNN-T | 160 ms or 560 ms chunks |
| Parakeet Unified English 0.6B | chunked attention: exact buffered mode, or our frozen-prefix caching | 560 ms (160 ms chunk + 400 ms lookahead) |
| Voxtral Mini 4B Realtime | causal encoder + LLM decoder, one token per 80 ms | 480 ms delay |

`PLAN.md` describes the design and milestones. The measurement rules are in `PLAN.md` (methodology section), and results go in `docs/PERF_LOG.md` and `docs/DISCREPANCIES.md`.

## Build

```sh
swift build -c release          # builds DictatorEngine and dictate-cli
swift test                      # kernel and streaming-equivalence tests
```

## Weights

The weights are downloaded into `models/` (gitignored) at the revisions pinned in `docs/PINS.md`. To check them, run `scripts/verify_pins.sh`.

```sh
# NeMo checkpoints -> safetensors with NeMo tensor names
reference/nemo/.venv/bin/python reference/nemo/export_weights.py models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo
reference/nemo/.venv/bin/python reference/nemo/export_weights.py models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b.nemo
# Reduced-precision copies
.build/release/dictate-cli convert nemotron --precision f16
.build/release/dictate-cli convert parakeet --precision f16
.build/release/dictate-cli convert voxtral --quant dec-ffn,dec-attn,enc,adapter --out dictator-q8.safetensors
```

## Use

```sh
.build/release/dictate-cli transcribe audio.wav --model nemotron --latency 160 --precision f16
.build/release/dictate-cli transcribe audio.wav --model parakeet --horizon 4 --precision f16
.build/release/dictate-cli voxtral audio.wav --weights dictator-q8.safetensors
.build/release/dictate-cli stream --model nemotron-160     # live microphone
```

## Correctness and evaluation

```sh
# Parity ladder against the NeMo golden tensors (L0 mel ... L4 tokens, L5 reference streaming)
.build/release/dictate-cli ladder nemotron --golden-root data/golden/nemotron/offline_70_1 \
    --stream-root data/golden/nemotron/stream_70_1 --context 70,1 --precision f16
# WER and latency: stream a manifest through a model, then score it
.build/release/dictate-cli eval --model parakeet-h4 --manifest data/librispeech/test-other.jsonl --out artifacts/eval/run.jsonl
reference/nemo/.venv/bin/python eval/score.py artifacts/eval/run.jsonl
reference/nemo/.venv/bin/python eval/latency.py artifacts/eval/run.jsonl data/align/test-other.jsonl
# Your own dictation test set, recorded on this Mac's microphone
.build/release/dictate-cli record eval/dictation_prompts.txt --out data/dictation
```

## Layout

| Path | Contents |
|---|---|
| `Sources/DictatorEngine` | Engine: `Core/` (Metal runtime, tensors, kernels' Swift side), `Metal/` (MSL kernels), `Models/`, `Frontend/`, `Decoding/` |
| `Sources/dictate-cli` | Development and evaluation CLI |
| `reference/` | Python reference harnesses: NeMo, transformers, CoreML study, competitors (test infrastructure only) |
| `eval/` | Scoring, alignment, latency and report scripts, and the dictation prompts |
| `docs/` | Specs, pins, performance log, discrepancies, research notes |
