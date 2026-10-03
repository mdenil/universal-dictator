# Universal Dictator: the Mac app

This is the plan for M7 in `PLAN.md`: a menu-bar app that turns speech into text you can edit and copy. It runs Nemotron on the Neural Engine through the engine library built in M0–M6. Those results are in `docs/PERF_LOG.md`, `docs/DISCREPANCIES.md` and the M6 report. The UI design is in Figma: [Universal Dictator — App UI](https://www.figma.com/design/RYBeO9fxHFk9C7WMtVS6Tl).

## What the app does

Universal Dictator lives in the macOS menu bar, the row of status icons next to the clock. It has no Dock icon and no main window.

1. **Click the menu-bar icon.** A small panel drops down under it and the microphone starts listening straight away.
2. **Speak.** The words appear in the panel's text box as you say them, about 0.3 s behind your voice.
3. **Pause and resume** as often as you like. Pause stops listening and keeps the text; Resume continues and appends to it.
4. **⌘C** copies the text so far at any point without stopping the dictation.
5. **Stop** ends the dictation. The text box becomes editable, so you can fix a word or two.
6. **Copy** puts the text on the clipboard and closes the panel.

Other ways to dictate (a global shortcut, pasting straight into the frontmost app, push-to-talk) come later. This first version is: click, talk, edit, copy.

### Models

The app runs only on the Neural Engine. It offers the two Nemotron settings, each measured in M6 and in the Neural Engine study:

| Model | Words appear (median) | WER, test-clean / test-other | Energy |
|---|---|---|---|
| Nemotron 160 ms (default) | ~0.3 s after the word ends | 2.7% / 5.3% | about the same as the GPU |
| Nemotron 560 ms | ~0.4 s | 2.5% / 4.9% | about 3.3× less than the GPU |

A model menu in the panel switches between them, so you can compare how each feels. Parakeet and Voxtral are not in the app: Voxtral cannot keep up in real time, and Parakeet has no Neural Engine version. Parakeet can be added later by giving it a Core ML port the way Nemotron got one.

## How it is built

```
UniversalDictator.xcodeproj      the Mac app (SwiftUI, menu bar): panel, text box, buttons, model menu
        │  links DictatorCore only
DictatorCore (Swift package lib) no Metal: audio capture, mel frontend, tokenizer, RNN-T decoder,
                                 Nemotron on the Neural Engine (Core ML), and the public API
        │  extended by
DictatorEngine (Swift package lib) Metal kernels and the Metal models (Nemotron, Parakeet, Voxtral);
                                 builds on DictatorCore. Used by dictate-cli, tests and benchmarks.
```

**The app talks to the library directly.** It links `DictatorCore` and calls its Swift API; it never runs `dictate-cli` or goes through the CLI's code. The CLI stays as the development and evaluation tool and uses the same API.

**The app contains no Metal code.** The Metal models and kernels stay in the repo and keep working for the CLI, but they live in `DictatorEngine`, which the app does not link.

### A0: split the package and give the core library its API

The current `DictatorEngine` target mixes Metal and Neural Engine code: the safetensors loader maps files into Metal buffers, and `NemotronStream` holds both the streaming logic and the Metal encoder. A0 splits it.

- **`DictatorCore`, a new library target with no Metal dependency:**
  - audio capture (`Microphone`), audio files, the NeMo mel frontend, the SentencePiece detokenizer;
  - the RNN-T decoder, reading its weights through a CPU safetensors reader (memory-mapped, no Metal);
  - the Nemotron streaming logic (chunk schedule, live push, the masked final chunk from D4), written against an **encoder protocol**;
  - the Neural Engine encoder (`NemotronANEEncoder`, stateful Core ML) as one implementation of that protocol;
  - the public API (below).
- **`DictatorEngine`** depends on `DictatorCore` and adds the Metal encoder as the other implementation of the encoder protocol, plus Parakeet, Voxtral, the kernels and the benchmarks. `dictate-cli` and the tests link it as now.
- **The public API in `DictatorCore`:**
  - `ModelChoice`: Nemotron 160 ms or 560 ms, with a display name and the measured latency and accuracy for the model menu.
  - `ModelAssets`: where a model's files live. The app passes its own bundle; the CLI passes `models/`. It reports any missing files.
  - `Engine.load(_ choice:, assets:) async throws -> LoadedModel`: loads the Core ML model and the decoder weights off the main thread, reporting progress.
  - `LoadedModel.startSession() -> DictationSession`: one dictation. A session owns its stream and a serial queue (the engine is not thread-safe). Audio goes in with `push(samples)`, and the session publishes `TranscriptUpdate` events (committed text, the partial word, step timing) as an `AsyncStream`. `pause()` ends the current segment and flushes its last words; `resume()` starts a new segment whose text is appended; `stop()` returns the final text.
- `dictate-cli` moves onto this API. Its Nemotron flags map onto `ModelChoice`. The `DICTATOR_ANE_STATELESS` kill-switch and the stateless Core ML models are deleted, since the stateful models are measured equal or better.
- A converter step writes the app's model files: the stateful Core ML packages for 160 and 560 ms, and a small safetensors file with only the decoder and joint weights.

### A1: the app shell

- `UniversalDictator.xcodeproj`, a standard macOS app project with one app target. It depends on this repo's Swift package as a local package and links `DictatorCore`. It builds in Xcode and from the command line with `xcodebuild`, which is also what a GitHub Actions release build would run later.
- The target's `Info.plist` sets `LSUIElement` (menu bar only, no Dock icon) and the microphone usage text. The app is signed for running on this Mac (a development signature); release signing and notarization come with the release work.
- **The models are bundled.** The Core ML packages are part of the app target, so Xcode compiles them to `.mlmodelc` at build time. The decoder weights and tokenizer go in as resources. The app is about 2.3 GB. Downloading models on first launch replaces bundling in the release version, so `ModelAssets` already treats the model location as data rather than a fixed path.
- SwiftUI's `MenuBarExtra` with the window style gives the drop-down panel. A1 has the icon, an empty panel and Quit, and asks for microphone permission on first use, explaining what to do if it is denied.

### A2: dictation

- **States:**
  - **listening:** live text, a level meter, elapsed time, Pause and Stop;
  - **paused:** text kept, Resume and Stop;
  - **done:** editable text, Copy and New;
  - **preparing:** a model is loading;
  - **error:** no microphone permission, or a model failed to load.
- **Live text:** committed words in the normal colour, the word still being recognised in a lighter shade. No committed word ever changes (M6 measured 0 rewrites), so the text never flickers.
- **⌘C** while listening or paused copies the text so far. **Copy** in the done state copies the edited text and closes the panel. The last dictation stays in the panel until a new one starts.
- **Loading at launch:** the chosen model is loaded when the app launches, so a click starts listening immediately. The first launch after installing or updating the app takes about 40 s longer, while Core ML compiles the model for the Neural Engine (it caches the result). The panel shows the preparing state if it is opened during that time.

### A3: choosing between the two models

- A model menu in the panel lists both Nemotron settings with their latency and accuracy, loads the chosen one in the background, and remembers the choice across launches.
- Only one model is in memory at a time.

### Later

Each of these gets planned when we get to it:

- a global keyboard shortcut and push-to-talk;
- pasting directly into the frontmost app;
- launch at login;
- a history of past dictations;
- release builds on GitHub Actions (`xcodebuild archive`, signing, notarization), with models downloaded on first launch instead of bundled;
- Parakeet on the Neural Engine (a Core ML port), so it can join the model menu.

## Testing

- **Library:** the existing live-equals-buffered tests move onto `DictationSession`, run with the Neural Engine encoder and with the Metal encoder, so both backends of the encoder protocol stay checked. New tests cover `ModelAssets` (missing files), and pause and resume (the text of a paused-and-resumed session equals the concatenation of its segments transcribed separately).
- **App:** the panel's state machine is a plain Swift type, tested with a fake session: start, updates, pause, resume, ⌘C, stop, copy, errors. The SwiftUI views stay thin.
- **By hand:** a short checklist per milestone, run on this Mac: the permission prompt, the first dictation after a fresh install, pause and resume, switching models, copying into another app.

## Risks

- **Disk space:** the Mac has about 6 GB free. Core ML needs a few GB for its compile cache; when it runs out, Neural Engine models fail to compile (seen during the Neural Engine study). The app checks free space and says so if it is too low.
- **App size:** about 2.3 GB, almost all model weights: the 160 and 560 ms models are separate 1.1 GB copies of the same network. A single Core ML file with one function per latency could share the weights, but the stateful-model work has not tested that.
