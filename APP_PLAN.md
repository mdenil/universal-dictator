# Universal Dictator: the Mac app

This is the plan for M7 in `PLAN.md`: a menu-bar app that turns speech into text you can edit and copy, built on the `DictatorEngine` library. The engine work (M0–M6) is done. Its results are in `docs/PERF_LOG.md`, `docs/DISCREPANCIES.md` and the M6 report.

## What the app does

Universal Dictator lives in the macOS menu bar, the row of status icons next to the clock (Wi-Fi, Tailscale, Granola and so on). It has no Dock icon and no main window.

1. Click the menu-bar icon. A small panel drops down under it, like a menu, and the microphone starts listening.
2. Speak. The words appear in the panel's text box as you say them, a fraction of a second behind your voice.
3. Stop (with the Stop button, or by clicking the icon again). The text box becomes editable, so you can fix a word or two.
4. Click **Copy**. The text goes to the clipboard, ready to paste anywhere.

Other ways to start dictating (a global shortcut, pasting straight into the frontmost app, push-to-talk) come later. The first version is deliberately just: click, talk, edit, copy.

The model is a setting. The app starts with the M6 recommendation (Nemotron, 160 ms, on the Neural Engine), and a model menu lets you switch between all the models we ported, so you can judge for yourself which feels best to use.

## How it is built

Three layers, each depending only on the one below it:

```
UniversalDictator.app        SwiftUI menu-bar app: panel, text box, buttons, settings
        │  uses only the public API below
DictatorEngine (library)     models, kernels, Core ML, microphone, and a small typed API:
                             catalog of models, load a model, open a dictation session,
                             receive transcript updates
        │
Weights on disk              models/… (Metal weights, Core ML packages)
```

**The library stays standalone.** The app imports `DictatorEngine` and nothing else from this repo. The library has no UI code and no knowledge of the app. Anything the app needs from the engine goes through the public API described in A0, which `dictate-cli` uses too, so the CLI keeps proving the API works without the app.

### A0: the library's public API (engine side)

Today, choosing and loading a model lives in `dictate-cli` (`transcriberFactory`, `nemotronUsesANE`) and some behaviour is chosen through `DICTATOR_*` environment variables. The original plan promised a typed configuration and transcript events; this milestone delivers them.

- **`ModelChoice`**: a typed value naming a model and its settings: Nemotron 160 / 560 ms, Parakeet h4 / exact, Voxtral q8h; backend (Neural Engine or Metal, where both exist); precision. Each choice also describes itself: display name, latency setting, expected memory, measured WER and step time from M6, so the app's model menu can show them without hard-coding numbers.
- **`ModelStore`**: where weights live. It takes a root folder (this repo's `models/` during development) and reports which choices are installed and which files are missing.
- **`Engine.load(_ choice:) async throws -> LoadedModel`**: loads weights off the main thread. For the Neural Engine it starts on Metal and switches to Core ML when the Core ML model is ready, which avoids the 40-second cold compile measured in M6 freezing the first dictation. Loading reports progress.
- **`LoadedModel.startSession() -> DictationSession`**: one utterance. A session owns its transcriber and a serial queue (the engine is not thread-safe). Audio goes in with `push(samples)`, and the session publishes **`TranscriptUpdate`** events (committed text, partial word, step timing) as an `AsyncStream`. `finish()` returns the final text.
- **`AudioSource`**: the microphone (the existing `Microphone`, moved behind a protocol) or a file, so sessions can be tested without a microphone.
- `dictate-cli` switches to this API, and its model flags map onto `ModelChoice`. `DICTATOR_ANE_STATELESS` and the other measured kill-switches are either deleted or become `ModelChoice` options, as `PLAN.md` requires.

### A1: the app shell

- A new SwiftPM executable target, `UniversalDictator`, in this package, using SwiftUI's `MenuBarExtra` with the window style (the drop-down panel).
- `scripts/build_app.sh` assembles `UniversalDictator.app`: the executable, the engine's resource bundle (Metal sources), an `Info.plist` with `LSUIElement` (menu bar only, no Dock icon) and the microphone usage text, and an ad-hoc code signature, which is enough for this Mac.
- A menu-bar icon, an empty panel, Quit. The app asks for microphone permission on first use and explains what to do if it is denied.

### A2: dictation

- The panel's states: **ready** → **listening** (live text, a level meter, Stop) → **done** (editable text, Copy, New dictation) and **error** (no microphone permission, model not installed, model failed to load).
- Live text: committed words in the normal text colour, the word still being recognised in a lighter shade. No committed word ever changes (M6 measured 0 rewrites for every model), so the text never flickers.
- Copy puts the edited text on the clipboard and confirms it in the panel. The last dictation stays in the panel until a new one starts.
- The model is loaded at launch and kept in memory, so the first click starts listening immediately.

### A3: choosing models

- A model menu in the panel lists every `ModelChoice` with its latency setting, measured accuracy and memory, marks the ones not installed, and loads the chosen one in the background. The choice persists across launches.
- Voxtral is listed but marked "slower than real time on this Mac" (M6), with its 7.8 GB footprint.
- Only one model is in memory at a time.

### A4: learning from real use (opt-in)

So that "what feels nice" can be measured, not just felt:

- An opt-in setting saves each dictation: the audio, the raw transcript, your edited text, and per-step timing, under `~/Library/Application Support/Universal Dictator/sessions/`.
- Your edits are the corrections, so each saved session is a labelled test utterance. `dictate-cli eval` reads that folder as a manifest, so every model can be scored on your own dictation (the set M6 marked as the one that matters most) and compared offline.
- A small debug readout in the panel (toggle in settings) shows the step time and how far the text lags your voice.

### Later

Ideas for when the basic flow feels right; each would be planned when we get to it:

- a global keyboard shortcut and push-to-talk;
- pasting directly into the frontmost app;
- launch at login;
- keeping a history of past dictations;
- signing and notarizing for other Macs, with model download on first launch.

## Testing

- **Library:** unit tests for `ModelChoice`/`ModelStore`, and session tests that stream a file through `DictationSession` with a file `AudioSource` and check that the updates add up to the same transcript as `dictate-cli` (the existing live-equals-buffered tests, moved onto the new API).
- **App:** the panel's state machine is a plain Swift type, tested with a fake session (start, updates, stop, copy, errors). The SwiftUI views stay thin.
- **By hand:** a short checklist for each milestone (permission prompt, first dictation after launch, switching models, copy and paste into another app), run on this Mac.

## Decisions for you

1. **Clicking the icon:** start listening at once (as described above), or open the panel with a **Start** button first?
2. **Stopping:** Stop button and clicking the icon again only, or also stop automatically after a few seconds of silence?
3. **After Copy:** keep the panel open, or close it?
4. **Project format:** SwiftPM plus a small build script (proposed: everything stays buildable from the command line and in one package), or an Xcode project (nicer for SwiftUI previews and Instruments, but a large generated file in the repo)?
5. **Where the app finds models:** this repo's `models/` folder for now (proposed), or a copy under `~/Library/Application Support`?
6. **Session saving (A4):** worth building early, so real use feeds the evaluation from the start?

## Risks

- **Disk space:** the Mac has about 6 GB free. Core ML needs a few GB for its compile cache; when it runs out, Neural Engine models silently fail to compile and fall back to the CPU (seen in M6). The app should check free space and say so.
- **Neural Engine cold start:** a compile-cache miss costs about 40 s of CPU, after every new build of the app. A0's start-on-Metal design hides it; it still costs energy once per build.
- **Contention:** when another app saturates the GPU, Metal steps overrun their budget (M6). The Neural Engine default avoids this for Nemotron; the other models still run on the GPU.
- **Microphone format changes:** plugging in headphones or AirPods changes the input device mid-session; `AudioSource` has to restart cleanly.
