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
# Nemotron's encoder runs on the Neural Engine by default; --backend metal uses the GPU. Build its Core ML models with
#   reference/coreml/.venv/bin/python reference/coreml/nemotron_ane.py              (first-chunk, steady and split models)
#   reference/coreml/.venv/bin/python reference/coreml/nemotron_ane.py --stateful   (stateful steady models, the default)
.build/release/dictate-cli transcribe audio.wav --model nemotron --latency 160 --precision f16
.build/release/dictate-cli transcribe audio.wav --model parakeet --horizon 4 --precision f16
.build/release/dictate-cli voxtral audio.wav --weights dictator-q8.safetensors
.build/release/dictate-cli stream --model nemotron-160     # live microphone
```

## The Mac app

Universal Dictator is a menu-bar app: click the waveform icon next to the clock, speak, then copy the text. It runs Nemotron on the Neural Engine only and links `DictatorCore` (no Metal). The plans are `APP_PLAN.md` and `UX_PLAN.md`; the design is in Figma ([Universal Dictator — App UI](https://www.figma.com/design/RYBeO9fxHFk9C7WMtVS6Tl)).

```sh
# 1. The app's model folder (Core ML models from reference/coreml/nemotron_ane.py --stateful, decoder weights, tokenizer)
swift build -c release && .build/release/dictate-cli convert app-assets          # -> build/app-assets
# 2. Build and open the app (or open App/UniversalDictator.xcodeproj in Xcode and press Run)
xcodebuild -project App/UniversalDictator.xcodeproj -scheme UniversalDictator -configuration Release \
    -derivedDataPath build/DerivedData build
open build/DerivedData/Build/Products/Release/UniversalDictator.app
```

Using it:

- **Click the icon:** the panel opens and listening starts; the icon turns red. Words appear about 0.3 s after you say them; the grey word is still being recognised.
- **Copy ⌘C** (the button or the shortcut) stops, copies the whole text and closes the panel: click, talk, ⌘C, then ⌘V wherever you want it.
- **Space** (or Pause) pauses; the icon turns yellow and the text becomes editable. **Resume** keeps your edits and adds new words after them. A paused dictation survives closing the panel: the next click reopens it, still paused.
- **Return** (or Stop) ends the dictation; the text stays editable. **Copy** (⌘Return) copies it and closes the panel; focus returns to the app you were in, ready for ⌘V. **New** starts again.
- **Closing the panel** (clicking away, or the icon) otherwise ends the dictation; the next click starts a new one.
- **Right-click the icon** for the five latest dictations (choosing one copies it), Show All History…, Settings… and Quit.
- **History** keeps the last 50 dictations (10–200 in Settings), saved on this Mac in `~/Library/Application Support/dev.universaldictator.UniversalDictator/history.json`. The History window searches them and has Copy, Edit and Delete for each.
- **Settings** (right-click → Settings…, or ⌘, while the panel is open): the model, **Fast** (Nemotron 160 ms) or **Accurate** (Nemotron 560 ms), the history length, and Clear History….

**Direct dictation** types straight into the focused text field of any app, with no window:

- **Double-tap Left Shift** (the default; change it in Settings). When the microphone is live, the icon turns red and a short sound plays. Each finished word is then typed at the cursor as if you had typed it. The word still being recognised isn't typed, so nothing is ever erased and retyped.
- **Toggle** (default): tap Shift once to stop. **Hold to talk**: Shift, Shift-hold while talking, release to stop.
- Stopping types the last word. Each direct dictation goes to History, marked "typed". Clicking the icon during one stops it.
- It needs the Accessibility permission (System Settings › Privacy & Security › Accessibility), which the app asks for once. Nothing is typed into password fields or apps using secure keyboard entry.
- The plan is `DIRECT_DICTATION_PLAN.md`.

At launch the app prepares the selected model for the Neural Engine in the background: about half a minute the first time after installing or updating (Core ML then caches it per macOS version), under a second after that. Clicking meanwhile shows "Getting things ready for you…" and dictation starts by itself when it is ready. Only the chosen model is kept in memory. Switching prepares the other one the same way the first time it is used on this Mac; after that, switching takes under a second. The app asks for microphone access at launch.

Checks without clicking through the app:

```sh
APP=build/DerivedData/Build/Products/Release/UniversalDictator.app/Contents/MacOS/UniversalDictator
$APP --selftest /path/to/audio.flac --model nemotron160    # the app's controller and bundled models on a file; pauses, closes and
                                                            # reopens the panel, edits the paused text and resumes
$APP --snapshot /tmp/panel-states                          # renders every panel state, Settings and History to PNG
$APP --measure-switch                                      # model load and switch times, and memory
$APP --typing-test                                         # types into a test window by each route, with keys held (needs Accessibility)
$APP --direct-selftest /path/to/audio.flac                  # direct dictation into a test window; PASS if it holds the final text
$APP --diagnose-start                                      # opens the panel twice for real: start timing, icon position
.build/release/dictate-cli dictate audio.flac --pause-at 5  # the same code path from the CLI
swift test                                                  # engine, core (Neural Engine) and app-logic tests
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
| `Sources/DictatorCore` | No Metal: audio, mel frontend, tokenizer, RNN-T decoder, Nemotron streamer and Neural Engine encoder, the dictation API |
| `Sources/DictatorEngine` | Metal engine on top of DictatorCore: `Core/` (Metal runtime, tensors, kernels' Swift side), `Metal/` (MSL kernels), `Models/` |
| `App/` | The Mac app: `UniversalDictator.xcodeproj`, its SwiftUI sources, and `Kit/` (the app's dictation logic, tested by `swift test`) |
| `Sources/dictate-cli` | Development and evaluation CLI |
| `reference/` | Python reference harnesses: NeMo, transformers, CoreML study, competitors (test infrastructure only) |
| `eval/` | Scoring, alignment, latency and report scripts, and the dictation prompts |
| `docs/` | Specs, pins, performance log, discrepancies, research notes |
