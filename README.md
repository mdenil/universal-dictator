# Universal Dictator

Universal Dictator is local streaming speech-to-text for Apple-silicon Macs. It is written in Swift, with hand-written Metal kernels and a Core ML encoder for the Neural Engine, and runs three open-weight models from their published checkpoints. It has two parts:

- **`dictate-cli`**, a command-line tool that runs all three models: Nemotron at 160 or 560 ms, Parakeet and Voxtral.
- **Universal Dictator**, a menu-bar app for dictation that runs Nemotron on the Neural Engine.

| Model | Streaming mode | Latency | Runs on |
|---|---|---|---|
| Nemotron Speech Streaming English 0.6B | cache-aware FastConformer + RNN-T | 160 ms or 560 ms chunks | Neural Engine (encoder) or GPU |
| Parakeet Unified English 0.6B | chunked attention: exact buffered mode, or our frozen-prefix caching | 560 ms (160 ms chunk + 400 ms lookahead) | GPU |
| Voxtral Mini 4B Realtime | causal encoder + LLM decoder, one token per 80 ms | 480 ms delay | GPU |

The design and milestones are in `plans/PLAN.md`; the app's plans are the other files in `plans/`. Measurements are in `docs/PERF_LOG.md` and `docs/DISCREPANCIES.md`, and each model's behaviour, as implemented, is specified in `docs/specs/`.

## Requirements

- An Apple-silicon Mac with macOS 15 or later.
- Xcode 16 or later (for the app; the command-line tools are enough for `dictate-cli`).
- [uv](https://docs.astral.sh/uv/), for the Python tools that convert the checkpoints.
- About 16 GB of disk for the checkpoints, plus about 14 GB for converted weights.

## Setup

Everything a build needs is kept inside the repository folder, in gitignored directories: `models/` (checkpoints and converted weights), `data/` (datasets), `build/` and `.build/`.

```sh
# 1. The checkpoints, at the revisions pinned in docs/PINS.md (check them with scripts/verify_pins.sh).
#    The app needs only the first.
uvx --from huggingface_hub hf download nvidia/nemotron-speech-streaming-en-0.6b \
    --revision ebe59e5a817142986528bbbee5dba8db7b38ed50 --exclude "*.gguf" --local-dir models/nemotron-speech-streaming-en-0.6b
uvx --from huggingface_hub hf download nvidia/parakeet-unified-en-0.6b \
    --revision fe53cd885760c96b6a5f51a0bfd362cb4584a98b --local-dir models/parakeet-unified-en-0.6b
uvx --from huggingface_hub hf download mistralai/Voxtral-Mini-4B-Realtime-2602 \
    --revision 2769294da9567371363522aac9bbcfdd19447add --exclude consolidated.safetensors --local-dir models/Voxtral-Mini-4B-Realtime-2602

# 2. The conversion tools' Python environment (torch, safetensors, coremltools)
(cd tools && uv sync)

# 3. NeMo checkpoints -> safetensors with NeMo's tensor names (models/<model>/nemo_extracted/)
tools/.venv/bin/python tools/export_weights.py models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo
tools/.venv/bin/python tools/export_weights.py models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b.nemo

# 4. Nemotron's encoder for the Neural Engine (models/nemotron-speech-streaming-en-0.6b/coreml/nemotron_encoder.mlpackage)
tools/.venv/bin/python tools/nemotron_ane.py

# 5. Build dictate-cli, then the reduced-precision weights it runs by default
swift build -c release
.build/release/dictate-cli convert nemotron --precision f16
.build/release/dictate-cli convert parakeet --precision f16
.build/release/dictate-cli convert voxtral                  # int8 (q8h): dictator-q8h.safetensors
```

## dictate-cli

```sh
.build/release/dictate-cli transcribe audio.wav --model nemotron-160     # or nemotron-560, parakeet, voxtral
.build/release/dictate-cli transcribe audio.wav --model parakeet --horizon 4
.build/release/dictate-cli stream --model nemotron-160                   # live microphone
.build/release/dictate-cli                                               # every command and option
```

Nemotron's encoder runs on the Neural Engine by default; `--backend metal` runs it on the GPU. `transcribe` prints the text, with load time, real-time factor and per-step times on stderr.

## The Mac app

Universal Dictator is a menu-bar app: click the waveform icon next to the clock, speak, then copy the text. It runs Nemotron on the Neural Engine only and links `DictatorCore` (no Metal). The design is in Figma ([Universal Dictator — App UI](https://www.figma.com/design/RYBeO9fxHFk9C7WMtVS6Tl)).

```sh
# 1. The app's model folder: the Core ML encoder, the decoder weights and the tokenizer (after setup steps 1-4)
swift build -c release && .build/release/dictate-cli convert app-assets          # -> build/app-assets
# 2. Build and open the app (or open App/UniversalDictator.xcodeproj in Xcode and press Run)
xcodebuild -project App/UniversalDictator.xcodeproj -scheme UniversalDictator -configuration Release \
    -derivedDataPath build/DerivedData build
open build/DerivedData/Build/Products/Release/UniversalDictator.app
```

The app is signed to run on this Mac only (ad hoc), which needs no Apple account. macOS then treats each rebuild as a new app and asks for the Accessibility permission again. To sign with your own Apple Development certificate instead, which keeps the permissions, see `App/Signing.xcconfig`.

Using it:

- **Click the icon:** the panel opens and listening starts; the icon's bars start moving. Words appear about 0.3 s after you say them; the grey word is still being recognised.
- **Copy ⌘C** (the button or the shortcut) stops, copies the whole text and closes the panel: click, talk, ⌘C, then ⌘V wherever you want it.
- **Space** (or Pause) pauses; the icon turns yellow and the text becomes editable. **Resume** keeps your edits and adds new words after them. A paused dictation survives closing the panel: the next click reopens it, still paused.
- **Return** (or Stop) ends the dictation; the text stays editable. **Copy** (⌘Return) copies it and closes the panel; focus returns to the app you were in, ready for ⌘V. **New** starts again.
- **Closing the panel** (clicking away, or the icon) otherwise ends the dictation; the next click starts a new one.
- **Right-click the icon** for the five latest dictations (choosing one copies it), Show All History…, Settings… and Quit.
- **History** keeps the last 50 dictations (10–200 in Settings), saved on this Mac in `~/Library/Application Support/dev.universaldictator.UniversalDictator/history.json`. The History window searches them and has Copy, Edit and Delete for each.
- **Settings** (right-click → Settings…, or ⌘, while the panel is open): the model, **Fast** (Nemotron 160 ms) or **Accurate** (Nemotron 560 ms), the history length, and Clear History….

**Direct dictation** types straight into the focused text field of any app, with no window:

- **Double-tap Left Shift** (the default). In Settings the shortcut can be a double tap of any modifier key (Left or Right ⇧ ⌃ ⌥ ⌘) or a key combination (⌃A, ⌘⇧D, ⇧A…); combinations that would take over ordinary typing or a standard command everywhere show a warning. When the microphone is live, the icon's bars start moving and a short sound plays. Each word is then typed at the cursor as it is recognised, as if you had typed it, so the last word is there as soon as you pause.
- **Toggle** (default): tap Shift once to stop. **Push to talk**: Shift, Shift-hold while talking, release to stop.
- Stopping types the last word and a space. Each direct dictation goes to History, marked "typed". Clicking the icon during one stops it.
- It needs the Accessibility permission (System Settings › Privacy & Security › Accessibility), which the app asks for once. Nothing is typed into password fields or apps using secure keyboard entry.

At launch the app prepares the selected model for the Neural Engine in the background: about 20 seconds the first time after installing or updating (Core ML then caches it per macOS version), under a second after that. Clicking meanwhile shows "Getting things ready for you…" and dictation starts by itself when it is ready. Only the chosen model is kept in memory. Switching prepares the other one the same way the first time it is used on this Mac; after that, switching takes under a second. The app asks for microphone access at launch.

Checks without clicking through the app:

```sh
APP=build/DerivedData/Build/Products/Release/UniversalDictator.app/Contents/MacOS/UniversalDictator
$APP --selftest /path/to/audio.flac --model nemotron160    # the app's controller and bundled models on a file; pauses, closes and
                                                            # reopens the panel, edits the paused text and resumes
$APP --snapshot /tmp/panel-states                          # renders every panel state, Settings and History to PNG
$APP --measure-switch                                      # model load and switch times, and memory
$APP --direct-selftest /path/to/audio.flac                  # direct dictation into a test window; PASS if it holds the final text
.build/release/dictate-cli dictate audio.flac --pause-at 5  # the app's code path from the CLI
```

## Tests

```sh
swift test       # kernels, streaming equivalence, the Neural Engine dictation API and the app's logic
```

Tests that need a model or audio skip when it is missing. With the setup above and the LibriSpeech test sets below, they all run.

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
| `Sources/DictatorCore` | No Metal: audio, mel frontend, tokenizer, RNN-T decoder, Nemotron streamer and Neural Engine encoder, the dictation API |
| `Sources/DictatorEngine` | Metal engine on top of DictatorCore: `Core/` (Metal runtime, tensors, weights, kernels' Swift side), `Metal/` (MSL kernels), `Models/` |
| `Sources/dictate-cli` | The command-line tool |
| `App/` | The Mac app: `UniversalDictator.xcodeproj`, its SwiftUI sources, and `Kit/` (the app's dictation logic, tested by `swift test`) |
| `tools/` | Python: checkpoint export and the Core ML encoder build |
| `reference/` | Python reference harnesses (NeMo, transformers, sherpa-onnx): golden tensors, reference WER and comparisons |
| `eval/` | Scoring, alignment and latency scripts, and the dictation prompts |
| `scripts/` | Pin verification, a lock for heavy reference runs, the sherpa-onnx comparison |
| `docs/` | Model specs, pins, performance log, discrepancies |
| `plans/` | The design and the plans each stage was built from |
