# Universal Dictator: cleaning up for public release

The goal is a repository someone else can clone, build and understand. It holds the two products and only what they need:

- **`dictate-cli`** runs the four models: Nemotron 160 ms, Nemotron 560 ms, Parakeet and Voxtral. Nemotron runs on the Neural Engine or the GPU; the other two run on the GPU.
- **The Mac app** runs Nemotron Fast and Accurate on the Neural Engine.

The audit covered every tracked file, the git history and the built app. Your decisions are recorded in 8 and 9.

## What the audit found

**No private information.**
- There are no paths from your Mac, hostnames, tokens or keys in any tracked file, or anywhere in the git history.
- The things that do identify you:
  - your name and email address as the author of every commit;
  - the Apple team ID in the Xcode project (see 6);
  - a few "this MacBook / M3, 24 GB" remarks in the plans and performance log. These are harmless context and stay.

**The app ships the Nemotron weights twice.**
- The app is 2.2 GB, but only about 1.2 GB of it is distinct.
- The Fast and Accurate Core ML models each contain the whole encoder: 605 identical weight blocks, 1.16 GB.
- The 560 ms subsampling model (9 MB) is also a copy of blocks inside the Fast model.
- Only about 40 MB actually differs between the two: per-layer position tables that depend on the chunk length.
- This is still the case today. See 5.

**A fresh clone can't build the app.**
- The Xcode project signs with your team (`QF8GKSXVE2`, Manual signing).
- The README never says how to download the weights, set up the Python environments, or fetch the reference checkouts.

**Dead and one-off code** (2–4), across the engine, the CLI, the app and the scripts:
- whole files: about 1,500 lines;
- parts of other files: several hundred lines more;
- copied third-party material in `docs/research/`: 8,200 lines.

**Third-party files:**
- `docs/research/parakeet/` holds copies of NVIDIA's NeMo sources, the Parakeet model card, its config and vocabulary, and a sherpa-onnx export script whose copyright notice grants no license.
- None of it is needed.
- There is no LICENSE file.

## 1. Plans go in `plans/`

- **The move:** `PLAN.md`, `APP_PLAN.md`, `UX_PLAN.md`, `DIRECT_DICTATION_PLAN.md`, `ROUND3_PLAN.md` and this plan go into `plans/` with `git mv`, so their history follows them.
- **Links updated:** seven references in README, APP_PLAN, PLAN, `docs/specs/nemotron.md` and `Convert.swift` (that one already points at a wrong path, `docs/PLAN.md`).
- **The plans' content stays as written.** They record what was decided at the time, so for example UX_PLAN still says the icon turns red. There is one exception: the headset note in DIRECT_DICTATION_PLAN says to rerun `--button-probe`, which is removed in 3. It will say where the probe lives in the history: `git show 1351991:App/UniversalDictator/ButtonProbe.swift`.

## 2. Engine and CLI: remove what neither product uses

**CLI subcommands.**

| Fate | Subcommands |
|---|---|
| Kept | `transcribe`, `stream`, `eval`, `record`, `dictate`, `convert`, `ladder` |
| Removed (development probes; their results are in `docs/PERF_LOG.md`) | `peaks` (GPU roofline), `libtime` (Metal compile timing), `probe` (ad-hoc GEMM/GEMV/RNN-T timing), `dump-step` (the finished Core ML study) |

Removing them also removes:
- `Bench/Probe.swift` and `Core/Peaks.swift`;
- the five `bench_*` kernels;
- `NPYArray.save`.

**Experiment switches hard-wired to what ships.** Each is an environment variable that picks a path the performance work rejected:
- `DICTATOR_GEMM_V1`: the old GEMM, all of `10_gemm.metal`. It would crash on int8 Voxtral anyway. `GemmParams` moves to `00_common`.
- `DICTATOR_GEMV`.
- `DICTATOR_ATTN_SPLITS` and `DICTATOR_ATTN_MQ`.
- `DICTATOR_KV_F32`: with it go the f32 attention kernels, which only it and tests reach.
- `DICTATOR_VOX_SNAPSHOT`.
- `DICTATOR_LADDER_FRAMES`.
- `DICTATOR_PROFILE` and `DICTATOR_SPLIT_TIMING`: the step profiler and the `section(...)` calls through Voxtral.
- The documented `DICTATOR_ANE_STATELESS`, which nothing has read since 0402f9c.

**The 4-bit Voxtral experiment, which didn't ship** (q8h is the shipping precision):
- `DType.q4`;
- the q4 GEMV kernel and its branch in `Step.gemm`;
- `Quant.quantizeRowsQ4`;
- `.q4scale` loading;
- `:q4` parsing in `convert voxtral`;
- its test.

**Kernels nothing reaches:**
- `scale_f32`, `cast_*` and `add_scaled_*`;
- the f16-activation variants of `layer_norm`, `rms_norm`, `glu`, `dwconv1d` and `relpos_attention` (every activation is f32);
- the GEMV/GEMM dtype combinations only tests select.

All the Metal sources are compiled together at startup, so this also shortens the CLI's start.

**Swift with no callers:**
- `Step.cast` and `Step.addScaled`;
- `MaskSpec.full`;
- the `forwardOffline(left:right:)` overload;
- one `NemotronLadder.run` overload;
- the function-constants branch of `MetalContext.pipeline`;
- `VoxtralSession.logitsTap` and `ids`, and `VoxtralModel.streamingWord`;
- three never-read `gpuTimes`;
- `SafeTensorsFile.metadata`;
- `Activation.geluTanh`.

**Tests:** the tests of removed code go with it. The f16 GEMM tests now cover only the combinations production uses.

## 3. DictatorCore and the app: remove what's left from investigations

**DictatorCore:**
- **No callers:**
  - `DictationSession.current`, with its `latest` and `lock`;
  - `TranscriptUpdate.isPaused`;
  - the `computeUnits:` parameter of `Engine.load`;
  - `logitsSink` in the RNN-T decoder;
  - `EngineError.shape`.
- **Moves:** the three Metal error cases move to DictatorEngine, which is the only thing that throws them.
- **Visibility:** about ten declarations that are public but used only inside DictatorCore become internal.

**App dev modes:**

| Mode | Fate | Why |
|---|---|---|
| `--selftest`, `--snapshot`, `--direct-selftest`, `--check-accessibility`, `--measure-switch` | kept | ongoing checks; 5 uses `--measure-switch` |
| `--button-probe` (ButtonProbe.swift) | removed | the headset button wasn't built; the note stays in the plan |
| `--typing-test` | removed | its result is recorded in DIRECT_DICTATION_PLAN; with it, `KeyboardTyper` loses its three routes and always uses the one that ships |
| `--diagnose-start`, `--diagnose-open`, `--diagnose-panel` | removed | the notch, empty-panel and microphone-restart investigations are finished |

Removing `--diagnose-start` also removes `Microphone.restartForDiagnosis()` and `blocksDelivered`. The restart path itself stays, but nothing will trigger it on demand any more.

**Diagnostics that still run in every dictation:**
- `OnsetProbe`, which logs leading silence on every capture. The question it answered is closed.
- The menu-bar occlusion observer, which logs the icon position for the notch question.
- `StartSound.outputDescription()`, from the Bluetooth investigation.
- `StartSound.wav(notes:)`, whose parameter only the probe used.

## 4. Scripts, reference harnesses and docs

**Removed:**
- **`docs/research/`**: the NVIDIA and sherpa copies above, plus planning notes the specs superseded. `docs/specs/parakeet.md` will cite upstream NeMo at the pinned commit instead.
- **`reference/coreml/nemotron_step.py` and `voxtral_encoder_step.py`**: the M5 study.
- **`scripts/eval_queue_*.sh`**, the four milestone run queues. They depend on copied binaries and subset manifests that no tracked script makes.
- **`scripts/power.sh`**. It sets the dead `DICTATOR_ANE_STATELESS` and needs local files. The PERF_LOG entry that cites it will name the commit to find it in.
- **`eval/report_html.py` and `report_template.html`**: the one-off M6 report. It hardcodes its run's files, and its prose is now wrong ("the Neural Engine drew nothing").

**Kept:**
- `reference/nemo` and `reference/voxtral`, which make the golden tensors for `ladder` and the reference WER.
- `eval/` scoring, alignment and latency.
- `eval/dictation_prompts.txt`, which is generic text.
- `scripts/heavy.sh` and `verify_pins.sh`. (The audit said `verify_pins.sh` misses the Parakeet weights. It doesn't: Parakeet is published only as the `.nemo`, which is pinned.)
- `scripts/eval_sherpa.sh` with `reference/competitors`. It backs the sherpa-onnx comparison in PERF_LOG, and gets the missing download step.

**The app's build tools get their own small folder.**
- Building the app needs two Python scripts: `export_weights.py` (.nemo → safetensors) and `nemotron_ane.py` (the Core ML models).
- Today the first needs the full NeMo environment, although it only uses torch and safetensors.
- Both move to `tools/`, with one light environment (torch, safetensors, coremltools).
- `reference/` is then what the README already says it is: test infrastructure only.

**Stale lines fixed:**
- README's Voxtral recipe makes q8 where q8h ships.
- README, `docs/PINS.md` and the coreml `pyproject.toml` call the Core ML scripts a study.
- The CLI's usage text lists removed commands and misses `ladder parakeet|voxtral` and `convert voxtral`.
- Two "queued" notes in `docs/specs/voxtral.md`.
- `Whisper`'s English spelling table (`reference/voxtral/whisper_english_spelling.json`, MIT) gets an attribution.

**Tests that always skip in a fresh clone:**
- The three streaming-equivalence tests read `scratch/dbg/clip.wav`, which only exists on your Mac.
- They'll use the LibriSpeech clip that DictatorCore's tests already document.

## 5. One copy of the Nemotron weights

**Today** the app bundles three Core ML models:
- Fast: one stateful step model;
- Accurate: the stateful layers model, plus a separate subsampling model.

Each carries its own copy of the encoder.

**The fix: one Core ML *multifunction* model.**
- Core ML (macOS 15+, our minimum) can store several functions in one model and keep each weight once when they share it. Apple's tool for this (`coremltools.utils.save_multifunction`) deduplicates identical weights.
- The package holds three functions: `fast`, `accurate_sub` and `accurate_layers`.
- `NemotronANEEncoder` loads the function it needs by name (`MLModelConfiguration.functionName`), so only the chosen model is in memory, as now.
- `nemotron_ane.py` makes the package in one run. Its conversion code for the old stateless models goes: nothing loads them since the encoder became stateful-only.
- `convert app-assets` copies the one package into the app.
- The CLI's `--backend ane` uses the same package.

**Expected:** the app goes from 2.2 GB to about 1.25 GB. Building and signing get faster too, since every build copies and hashes the bundle.

**Checked before it's committed:**
- **Same output:** the encoder's output on a set of LibriSpeech clips is compared with the current models, for both latencies. The same graph and weights should give identical results; if they differ at all, I'll report by how much and what it does to the transcripts.
- **The app's own checks:** `--selftest` on both models, and `swift test`.
- **Memory and time:** `--measure-switch` for load, switch time and memory. These must not get worse. The first-launch preparation time is measured too.

If Core ML won't put a multifunction model on the Neural Engine, I'll come back to you before trying anything else.

**Your local `models/` folder** also holds about 5 GB of the old stateless Core ML models, which nothing uses. They're deleted along with the change. They're gitignored, so this affects only your disk.

## 6. Building from a fresh clone

**Signing.**
- The project moves to `App/Signing.xcconfig` (tracked), which signs to run locally (ad hoc) and needs no team.
- It optionally includes `App/Signing.local.xcconfig` (gitignored), which holds your team.
- Your builds keep their stable signature, so macOS keeps the microphone and Accessibility permissions across rebuilds. Someone else's build just works.
- The README notes that with ad hoc signing macOS asks for Accessibility again after each rebuild.

**README:** a setup section from nothing to a running app and CLI:
1. Download the three checkpoints with `hf download` at the revisions in `docs/PINS.md`.
2. `uv sync` for `tools/`.
3. Export and convert.
4. Build.

It covers the reference checkouts and LibriSpeech separately, for the parity and evaluation tools only. It also corrects the layout table.

**The test:** a fresh clone in a scratch folder, built by following the README word for word. To save downloading 14 GB again, it reuses your downloaded checkpoints, copied as clones that take no extra space. Then: `swift build`, `swift test`, the app build, and `--selftest`.

## 7. Duplicated code

| Duplication | Change |
|---|---|
| Two safetensors parsers (DictatorCore's and DictatorEngine's) and three bf16/f16 → f32 conversions | The engine's reader builds on Core's header parsing and conversion, and keeps only its Metal memory mapping |
| Parakeet's buffered-window bookkeeping written three times, with the same constants each time | One implementation. The buffered paths become the live path fed the whole file, which the tests already show is equivalent |
| The golden mel/encoder transpose in three ladder places | One helper |
| `typealias NemotronStream = NemotronStreamer` | Removed |

## 8. The CLI's four models, named one way

Today the subcommands disagree:
- **Model names:** `transcribe --model nemotron --latency 160` versus `stream --model nemotron-160`.
- **Default precision:** `transcribe` uses f32 while the others use f16.
- **Voxtral:** it has its own `voxtral` subcommand, its default weights differ between commands, and `convert voxtral` defaults to a recipe that isn't the shipping one.
- **`--model-dir`:** `eval` and `stream` ignore it.

**Unified** (decided):
- Every command takes `--model nemotron-160|nemotron-560|parakeet|voxtral`, plus `--horizon H` for Parakeet's cached mode.
- `voxtral` folds into `transcribe`.
- The defaults are what ships: f16, and q8h for Voxtral. `convert voxtral` makes q8h.

## 9. License and history

**License: Apache-2.0** (decided). A `LICENSE` file, and the README says so.

The README also gets a "Models" section. It names each model's license (NVIDIA Open Model License for Nemotron and Parakeet, Apache-2.0 for Voxtral) and says that an app built with `convert app-assets` contains NVIDIA's weights under that license.

**History: `docs/research/` is removed from it** (decided).
- Rewritten with `git filter-repo`, then force-pushed to the private repo before you make it public.
- Every commit hash changes. The three places that cite a commit are updated: UX_PLAN, PERF_LOG, and the button-probe pointer added to DIRECT_DICTATION_PLAN in 1.
- Your other clones would need a fresh clone.

**Your email address stays in the commits** (decided).

I won't make the repository public; that's yours to do once you're happy with it.

## Order of work

Each step is committed with `swift build`, `swift test` and the app build passing, and the app is rebuilt and restarted after each.

1. Plans into `plans/` (1).
2. Dead code: engine and CLI (2), then DictatorCore and the app (3), then scripts and docs (4).
3. Duplicated code (7), then the CLI's model names (8).
4. One copy of the weights (5), with the comparisons above.
5. Fresh-clone build: signing, `tools/` and the README (6). It comes last, so the README describes the final layout and the test checks everything above.
6. License, then the history rewrite and force-push (9), after you confirm.
