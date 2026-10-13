# Developing Universal Dictator

This is for building the app from source, making releases, and working on the speech engine. To just use the app, see the [README](../README.md).

The repository has two products:
- **The Mac app** (`App/`), which runs NVIDIA's Nemotron Speech Streaming model on the Neural Engine.
- **`dictate-cli`** (`Sources/dictate-cli`), the command-line tool the engine was developed with. It runs Nemotron plus two more streaming models, Parakeet and Voxtral, with hand-written Metal kernels.

## Building the app

You need an Apple-silicon Mac with macOS 15 or later, Xcode 16 or later, [uv](https://docs.astral.sh/uv/) for the Python conversion tools, and about 12 GB of disk. Everything is kept inside the repository folder, in gitignored directories (`models/`, `build/`, `.build/`).

```sh
# 1. Nemotron's checkpoint, at the revision pinned in docs/PINS.md (scripts/verify_pins.sh checks the files)
uvx --from huggingface_hub hf download nvidia/nemotron-speech-streaming-en-0.6b \
    --revision ebe59e5a817142986528bbbee5dba8db7b38ed50 --exclude "*.gguf" --local-dir models/nemotron-speech-streaming-en-0.6b

# 2. The conversion tools' Python environment (torch, safetensors, coremltools)
(cd tools && uv sync)

# 3. The checkpoint as safetensors (models/nemotron-speech-streaming-en-0.6b/nemo_extracted/)
tools/.venv/bin/python tools/export_weights.py models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo

# 4. The encoder for the Neural Engine (models/nemotron-speech-streaming-en-0.6b/coreml/nemotron_encoder.mlpackage)
tools/.venv/bin/python tools/nemotron_ane.py

# 5. The app's model folder: the Core ML encoder, the decoder weights and the tokenizer (-> build/app-assets)
swift build -c release && .build/release/dictate-cli convert app-assets

# 6. Build and open the app (or open App/UniversalDictator.xcodeproj in Xcode and press Run)
xcodebuild -project App/UniversalDictator.xcodeproj -scheme UniversalDictator -configuration Release \
    -derivedDataPath build/DerivedData build
open build/DerivedData/Build/Products/Release/UniversalDictator.app
```

The app is signed to run on this Mac only (ad hoc), which needs no Apple account. macOS then treats each rebuild as a new app and asks for the Accessibility permission again. To sign with your own Apple Development certificate instead, which keeps the permissions, see `App/Signing.xcconfig`.

The app links only `DictatorCore` (no Metal); its dictation logic is in `App/Kit`, and its plans are in `plans/`. The design is in Figma ([Universal Dictator — App UI](https://www.figma.com/design/RYBeO9fxHFk9C7WMtVS6Tl)).

Checks without clicking through the app:

```sh
APP=build/DerivedData/Build/Products/Release/UniversalDictator.app/Contents/MacOS/UniversalDictator
$APP --selftest /path/to/audio.flac --model nemotron160    # the app's controller and bundled models on a file; pauses, closes and
                                                            # reopens the panel, edits the paused text and resumes
$APP --snapshot /tmp/panel-states                          # renders every panel state, Settings and History to PNG
$APP --measure-switch                                      # model load and switch times, and memory
$APP --direct-selftest /path/to/audio.flac                  # direct dictation into a test window; PASS if it holds the final text
.build/release/dictate-cli dictate audio.flac --pause-at 5  # the app's code path from the command line
```

The README's screenshot, `docs/images/panel.png`, is `1-listening.png` from `--snapshot`.

## Making a release

Push a version tag:

```sh
git tag -a v0.2.0 -m "Universal Dictator 0.2.0" && git push origin v0.2.0
```

GitHub Actions (`.github/workflows/release.yml`) builds the app from that tag, from nothing as above, and publishes `UniversalDictator-0.2.0.dmg` on the Releases page, in about 5 minutes. The app's version is the tag's; a tag with a suffix, such as `v0.2.0-rc1`, makes a pre-release. Nothing else triggers a build. The plan behind it is `plans/RELEASE_BUILDS_PLAN.md`.

## dictate-cli

`dictate-cli` runs three open-weight streaming models from their published checkpoints, for transcribing files, streaming the microphone, and measuring accuracy and latency:

| Model | Streaming mode | Latency | Runs on |
|---|---|---|---|
| Nemotron Speech Streaming English 0.6B | cache-aware FastConformer + RNN-T | 160 ms or 560 ms chunks | Neural Engine (encoder) or GPU |
| Parakeet Unified English 0.6B | chunked attention: exact buffered mode, or our frozen-prefix caching | 560 ms (160 ms chunk + 400 ms lookahead) | GPU |
| Voxtral Mini 4B Realtime | causal encoder + LLM decoder, one token per 80 ms | 480 ms delay | GPU |

After steps 1–4 of [Building the app](#building-the-app), the other two checkpoints and the reduced-precision weights the tool runs by default (about 30 GB of disk in all):

```sh
uvx --from huggingface_hub hf download nvidia/parakeet-unified-en-0.6b \
    --revision fe53cd885760c96b6a5f51a0bfd362cb4584a98b --local-dir models/parakeet-unified-en-0.6b
uvx --from huggingface_hub hf download mistralai/Voxtral-Mini-4B-Realtime-2602 \
    --revision 2769294da9567371363522aac9bbcfdd19447add --exclude consolidated.safetensors --local-dir models/Voxtral-Mini-4B-Realtime-2602
tools/.venv/bin/python tools/export_weights.py models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b.nemo
swift build -c release
.build/release/dictate-cli convert nemotron --precision f16
.build/release/dictate-cli convert parakeet --precision f16
.build/release/dictate-cli convert voxtral                  # int8 (q8h): dictator-q8h.safetensors
```

Then:

```sh
.build/release/dictate-cli transcribe audio.wav --model nemotron-160     # or nemotron-560, parakeet, voxtral
.build/release/dictate-cli transcribe audio.wav --model parakeet --horizon 4
.build/release/dictate-cli stream --model nemotron-160                   # live microphone
.build/release/dictate-cli                                               # every command and option
```

Nemotron's encoder runs on the Neural Engine by default; `--backend metal` runs it on the GPU. `transcribe` prints the text, with load time, real-time factor and per-step times on stderr.

The engine's design and milestones are in `plans/PLAN.md`. Measurements are in `docs/PERF_LOG.md` and `docs/DISCREPANCIES.md`, and each model's behaviour, as implemented, is specified in `docs/specs/`.

## Tests

```sh
swift test       # the app's logic, the Neural Engine dictation API, kernels and streaming equivalence
```

Tests that need a model or audio skip when it is missing. With all three models and the LibriSpeech test sets below, they all run.

## Correctness and evaluation

The reference side is Python, in `reference/`: NeMo for Nemotron and Parakeet, transformers for Voxtral, and sherpa-onnx for a comparison. Each folder is its own uv project (`uv sync` in it). The NeMo harness also needs two source checkouts in `third_party/`, at the commits in `docs/PINS.md`: a sparse checkout of NeMo with `nemo/collections/asr/modules/conformer_encoder.py`, and the v2.7.3 tag's `examples/asr` (see `reference/nemo/common.py` and `reference/nemo/run_official.py`).

```sh
# LibriSpeech test-clean and test-other (openslr.org/12), extracted into data/librispeech/, then the manifests
#   data/librispeech/{test-clean,test-other}.jsonl and the dev20 subset data/dev/dev20.jsonl
reference/nemo/.venv/bin/python reference/nemo/make_dev_data.py
# Golden tensors from the reference (see each script's header for its options)
scripts/heavy.sh reference/nemo/.venv/bin/python reference/nemo/dump_golden.py --model nemotron --utts data/dev/dev20.jsonl
scripts/heavy.sh reference/nemo/.venv/bin/python reference/nemo/stream_ref.py --model nemotron --config 70_1
# Parity ladder against them (L0 mel ... L4 tokens, L5 reference streaming)
.build/release/dictate-cli ladder nemotron --golden-root data/golden/nemotron/offline_70_1 \
    --stream-root data/golden/nemotron/stream_70_1 --context 70,1 --precision f16
# WER and latency: stream a manifest through a model, then score it
.build/release/dictate-cli eval --model parakeet --horizon 4 --manifest data/librispeech/test-other.jsonl --out artifacts/eval/run.jsonl
reference/nemo/.venv/bin/python eval/score.py artifacts/eval/run.jsonl
reference/nemo/.venv/bin/python eval/align.py data/librispeech/test-other.jsonl test-other    # word alignments, once
reference/nemo/.venv/bin/python eval/latency.py artifacts/eval/run.jsonl data/align/test-other.jsonl
# Your own dictation test set, recorded on this Mac's microphone
.build/release/dictate-cli record eval/dictation_prompts.txt --out data/dictation
```

## Layout

| Path | Contents |
|---|---|
| `App/` | The Mac app: `UniversalDictator.xcodeproj`, its SwiftUI sources, and `Kit/` (the app's dictation logic, tested by `swift test`) |
| `Sources/DictatorCore` | No Metal: audio, mel frontend, tokenizer, RNN-T decoder, Nemotron streamer and Neural Engine encoder, the dictation API the app uses |
| `Sources/DictatorEngine` | Metal engine on top of DictatorCore: `Core/` (Metal runtime, tensors, weights, kernels' Swift side), `Metal/` (MSL kernels), `Models/` |
| `Sources/dictate-cli` | The command-line tool |
| `tools/` | Python: checkpoint export and the Core ML encoder build |
| `reference/` | Python reference harnesses (NeMo, transformers, sherpa-onnx): golden tensors, reference WER and comparisons |
| `eval/` | Scoring, alignment and latency scripts, and the dictation prompts |
| `scripts/` | Pin verification, a lock for heavy reference runs, the sherpa-onnx comparison |
| `docs/` | Model specs, pins, performance log, discrepancies |
| `plans/` | The design and the plans each stage was built from |

## Model licenses

The weights are not part of this repository: the setup downloads them from their publishers, and they keep their own licenses.

| Model | License |
|---|---|
| [Nemotron Speech Streaming English 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| [Parakeet Unified English 0.6B](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Apache License 2.0 |

The app contains Nemotron's weights (as its Core ML encoder and decoder files), so a copy of the app you give to someone else is also covered by the NVIDIA Open Model License. The app and the release DMG carry `LICENSE`, `NOTICE` and the agreement's text (`docs/licenses/`).
