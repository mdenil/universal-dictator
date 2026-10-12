# Universal Dictator: plan

Goal: build a local, low-latency English dictation engine for Apple-silicon MacBooks. The first target is this machine, an M3 with 4 performance cores, 4 efficiency cores, a 10-core GPU, Metal 4 and 24 GB of unified memory. The engine runs three streaming speech-to-text models from their published weights:

| Model | Hugging Face repo | Latency setting | License |
|---|---|---|---|
| Nemotron Speech Streaming English 0.6B | `nvidia/nemotron-speech-streaming-en-0.6b` | 160 ms and 560 ms | NVIDIA Open Model License |
| Parakeet Unified English 0.6B | `nvidia/parakeet-unified-en-0.6b` | 560 ms (160 ms chunk + 400 ms lookahead) | NVIDIA Open Model License |
| Voxtral Mini 4B Realtime | `mistralai/Voxtral-Mini-4B-Realtime-2602` | 480 ms (6 delay tokens) | Apache 2.0 |

Priority: the best accuracy we can get at low latency. We accept a modest loss of accuracy to avoid delays of about two seconds. The latency settings above are model settings; they are not measured end-to-end latency on the Mac. We will measure that ourselves.

We build the engine first and then evaluate transcription quality. We build the Mac app only after the evaluation.

## Principles

1. **Correctness outranks speed.** Parity comes first and performance second. A faster kernel that changes the output without a measured, accepted reason is reverted and recorded (in `docs/PERF_LOG.md`).
2. **The weights are fixed, but how we run them is ours to choose.** We do not have to copy the reference inference loop. How we cache, chunk, lay out memory and move data is our design space. Section M3 (Parakeet) is the main example.
3. **Two kinds of correctness, kept separate:**
   - *Kernel parity:* our forward pass, given the same inputs and the same context, matches the reference within a tolerance *measured* from the reference's own noise.
   - *Algorithm quality:* when we change how streaming works (caching, context, normalization), we measure the effect on WER and latency against the reference streaming pipeline. Output that is identical token for token is not required.
4. **Build the reference first, then the engine.** Every stage is proven against reference tensors in pipeline order before anything is optimized.
5. **Optimize for this hardware, guided by the profiler.** The base M3 has about 100 GB/s of memory bandwidth and a GPU with a few TFLOPS; we measure the real peaks in M1. Every performance change goes through the performance ritual below. On Apple GPUs, the thing to beat is round trips between the CPU and the GPU, not individual kernels: we fuse work into one command buffer per streaming step and keep data resident on the GPU.
6. **Measure all three accelerators.** These are the GPU (Metal), the Neural Engine (only reachable through CoreML) and the CPU's matrix unit (reachable through Accelerate). Each model part goes on whichever one measures best.
7. **Project material lives in this folder.** Everything the project itself produces or depends on stays inside the repo: weights, datasets, reference source checkouts, the Python environment, benchmark evidence, scratch files and research notes. Large items go in gitignored folders. Ordinary tool caches (uv, pip, Swift, Xcode) stay in their normal system locations.
8. **No ceremony.** We follow the `just-say-no-to-process-porn-and-ceremony` skill. The working engine is the deliverable, and process work earns no credit.
   - **Creation gate:** a process file (ledger, register, report) is created only if it names who or what reads it, what it blocks, the defect it guards against, and when it gets deleted. The process files this plan allows are listed under "Process files" below, and no others are added without passing the same test.
   - **Real code with real tests:** we never weaken tests, regenerate golden data to force a pass, or present mocks as proof.
   - **Honest milestone reports:** before calling a milestone done, I re-run its acceptance checks from scratch and fill out the skill's real-work audit and honesty inventory. The results go into my report to you, not into files. I state plainly what was verified and what was not. A subagent's report is treated as a claim, and I re-run it before relying on it.

## Measurement and correctness methodology

This is adapted from the `ai-model-into-rust-mega-fused-hyper-kernel` skill, with the Rust-specific parts dropped. During implementation I go back to the skill at the points where it matters:

- before each performance pass (its performance-ritual and rejected-ideas references);
- when parity breaks (its parity-triage section);
- before quantizing (its quantization guidance);
- before any speed claim (its evidence standards).

### Truth pack (pinned sources)

- `docs/PINS.md` records:
  - the Hugging Face revision (commit hash) of each checkpoint, with SHA-256 hashes of the weights, configs and tokenizers;
  - pinned commits for NeMo, transformers, mistral-common and each reference port;
  - reference runtime versions taken from the **model cards**, not from `config.json`. For example, the Parakeet card says NeMo 2.7.3.

  A script re-downloads everything and checks it with `--verify`. The reference scripts check these pins when they run.
- **Open questions:** each unresolved question about how a model behaves goes in an "Open questions" section of that model's spec (below). The answer cites the source code line it came from. **No kernel is written against an unresolved question.** Known questions so far:
  - the NeMo mel defaults (mel normalization, log guard value, padding mode);
  - Nemotron's causal subsampling padding;
  - how Dynamic Chunked Convolution behaves in Parakeet training;
  - what Voxtral's `[STREAMING_WORD]` token means;
  - Voxtral's RoPE layout (interleaved vs NeoX) in each weights file;
  - whether the sherpa-onnx exports use the March Nemotron checkpoint.
- Each model gets a short spec, `docs/specs/<model>.md`, with source line citations. The tensor-name map and its traps (fused vs split QKV, RoPE variant, BatchNorm folding) are written down **before** we write any code.

### The reference and its noise floor

- **Before setting any tolerance, measure how much the reference varies from run to run.** We run it twice at different thread counts, and on CPU fp32 versus MPS fp16 where relevant. We record the per-layer max-abs spread, the logit spread and how often tokens differ. Every tolerance is derived from these numbers, never borrowed from another model. The measured envelope is committed.
- The correctness fixtures come from the unmodified reference. We also generate fp32 fixtures ourselves, so the "exact" checks have an fp32 anchor.

### The L0–L5 parity ladder (per model)

| Rung | What is compared | Standard |
|---|---|---|
| L0 | Audio framing, mel features, prompt/token ids, tokenizer round-trip | Exact where deterministic (ids); within measured noise for mel |
| L1 | Each operation, on the reference's exact input tensors | Within the measured tolerance |
| L2 | Each layer (subsampling, each encoder layer, adapter, each decoder layer) | Cosine ≈ 1.0; max-abs difference recorded per layer |
| L3 | Logits (joint-network or language-model head) | Measured budget; argmax exact |
| L4 | Greedy tokens | Exact over the prefix the reference reproduces deterministically |
| L5 | WER over the dev set, and reference streaming output | Within the documented budget, with tail cases reported |

Stages are proven **in pipeline order**. Each stage of ours is fed the reference's exact input tensor, so a bug in one stage cannot hide in another. Every leftover difference is traced to a named cause. Our changes to the streaming algorithm are judged at L5 only, and each is recorded as a discrepancy entry.

The ladder runs with one command, `dictate-cli ladder <model>`, and also from `swift test`. A skipped rung is reported as skipped, never as passed.

### Performance ritual (one change at a time)

1. **Check the performance log first.** Has this idea already been measured and rejected?
2. **Profile again before choosing what to attack.** The engine records timings per stage and per kernel, using Metal GPU timestamps (counter sample buffers) and CPU signposts, so we never work from a stale list of hot spots.
3. **State a hypothesis that can fail, with a gate set in advance.** For example: "−10% per-chunk GPU time, identical output".
4. **Implement one change, behind a temporary kill-switch environment variable** (`DICTATOR_*`). With the switch off, the old code path is unchanged. The switch is removed once the change is kept or reverted.
5. **Prove correctness before measuring speed.** The ladder must still pass, and the output must be identical on real audio. If the change is inherently lossy (quantization), it needs a measured WER cost and a discrepancy entry.
6. **Measure with interleaved paired A/B runs (ABBA order) in the same thermal window**, with warm-up runs discarded. A run is **refused if the coefficient of variation is above 5%**. This MacBook heats up, and sequential runs drift 30% on unchanged code. Formal timings are taken plugged in, with Low Power Mode off, on a quiet machine.
7. **Keep or revert. A neutral result is also a revert.** Either way, it is recorded.
8. **Every timing row gets a roofline column:** the stage's compute floor and memory floor, computed from the M3 peaks we measure in M1. A stage more than 1.3× above its floor is a target, not an excuse.
9. **Commit messages state the correctness level:** `bit-identical`, `argmax-exact`, `WER-neutral (±x)` or `lossy (+x WER)`.

### Comparisons against other implementations

A claim like "faster than X" needs a pinned competitor:

- the PyTorch reference (MPS and CPU);
- sherpa-onnx;
- the MLX ports;
- `antirez/voxtral.c`.

Each competitor is pinned by commit or hash, run at its strongest settings with matching thread counts, checked for identical output on every run, and labelled with its precision in every row. A speedup measured only against our own earlier version is maintenance, not a win.

### Process files (the complete list; each passes the creation gate)

| File | Read by | What it blocks | Guards against | Deleted when |
|---|---|---|---|---|
| `docs/PINS.md` + verify script | the reference scripts (they check the pins at startup) | parity runs against drifted weights or code | silent changes to checkpoints or libraries | never; it is small, and kept while the reference harness exists |
| `docs/specs/<model>.md`, including its open questions | me, while implementing | writing kernels against an unresolved question | porting semantics nobody actually read, such as a wrong RoPE variant | the model's L4 rung passes; the spec is then frozen and its open-questions section removed |
| `docs/PERF_LOG.md` (kept and rejected changes, each rejection with a "retry only if…" condition) | step 1 of the performance ritual | re-spending effort on ideas already measured dead | repeated failed attempts at the same idea | the project ends |
| `docs/DISCREPANCIES.md` (accepted differences from the reference, with kill-switch, measured WER effect) | the M6 evaluation and the default-settings decision | shipping a lossy change nobody measured | quantization or streaming changes that quietly drift | each entry is removed when its change is either made the default or deleted |

Benchmark raw output goes in `artifacts/` (gitignored), with no manifests or dashboards. Nothing else is created without passing the same gate.

### Known hazards

These are lessons carried over from related projects:

- MSL `tanh` returns NaN at fp16 extremes in GELU; clamp its input.
- Near-tied argmax values flip on tiny numerical drift. Decoder attention and the RNN-T joint network need extra care.
- int4 weights without a native packed dot product can be *slower* than int8. Measure it; do not assume.
- A performance claim measured on easy inputs does not hold for hard ones. Quality claims are made per slice (clean / noisy / accented / long-form).

## What we build

There are two Swift products, plus Python tooling that never ships.

1. **`DictatorEngine`, a Swift library.** This is the real product. It has one small streaming API for all three models: create a transcriber for a model and settings, push audio buffers in, and receive partial and final transcript events. The model choice and its settings (model, latency setting, precision) are a typed configuration value passed to the API, not environment variables. The Mac app in M7 links this library directly; it never calls the CLI.
2. **`dictate-cli`, one binary** for development and evaluation. It is a thin wrapper over the library. Its subcommands are:
   - `transcribe`: run a WAV file through a model;
   - `stream`: transcribe live from the microphone;
   - `bench`: timing runs;
   - `ladder`: run the parity checks;
   - `convert`: build our weight files, including quantization;
   - `dump`: write out intermediate tensors.

   Models and behaviours are chosen with **command-line flags**, for example `dictate-cli stream --model nemotron --latency 160 --precision fp16` or `--model voxtral --latency 480 --precision q4`. The flags map one to one onto the library's configuration.
3. **Environment variables are only temporary kill-switches** (`DICTATOR_*`) for changes being tested in the performance ritual. Once a change has been measured, it either becomes the default (the switch is deleted) or becomes a real configuration option if it is a genuine trade-off, such as Parakeet's exact mode against its efficient mode. Otherwise it is removed. No permanent behaviour is controlled by an environment variable.

The Python code in `reference/` and `eval/` produces golden tensors and scores results. It is never part of the engine.

## Technology

- **Engine host:** Swift, as a Swift package named `DictatorEngine`. It records Metal command buffers, manages the streaming state and runs the decode loops.
- **Kernels:** hand-written Metal Shading Language (MSL). They are compiled into a `.metallib` at build time and tuned for the M3 GPU: simdgroup matrix operations, threadgroup memory tiling and fp16/bf16 arithmetic.
- **CPU paths:** Accelerate (vDSP for the mel frontend, BLAS/BNNS for small matrices that may run on the CPU's matrix unit).
- **Neural Engine experiments:** CoreML models converted with coremltools. They are used only where measurement justifies it.
- **Reference harness:** Python in a `uv`-managed environment on Python 3.12, with NeMo, transformers, mistral-common and torch. It is test infrastructure only and never ships.
- **Weights:** a Python step extracts the checkpoints to raw safetensors. **Our Swift code does the quantization and writes our weight format**, using the same quantizer function the runtime uses. That way, a file written by the converter and the same weights quantized at runtime are identical by construction, and a test checks this. The runtime memory-maps the format, so it does not unpickle, parse or copy the weights at load time.

### Repository layout (project material stays in this folder)

```
universal-dictator/
  PLAN.md
  Package.swift
  Sources/
    DictatorEngine/          # Swift: device, buffers, graph runner, streaming sessions
      Metal/                 # MSL kernels (.metal)
      Models/Nemotron/ Models/Parakeet/ Models/Voxtral/
      Frontend/              # mel spectrogram (vDSP + Metal)
      Decoding/              # RNN-T greedy decoder, LLM token loop, tokenizers
      Quant/                 # the one quantizer, shared by the runtime and the converter
    dictate-cli/             # transcribe, stream the mic, benchmark, ladder, convert, dump tensors
  Tests/DictatorEngineTests/ # kernel and layer parity tests against golden tensors
  reference/                 # Python uv project: golden-tensor dumpers, reference streaming runs
  eval/                      # WER/latency harness, dataset manifests, reports
  docs/                      # PINS.md + verify script, specs, research notes, PERF_LOG.md, DISCREPANCIES.md
  --- gitignored ---
  models/                    # downloaded checkpoints and converted weights
  data/                      # datasets, our dictation recordings, golden tensors
  third_party/               # pinned source checkouts: NeMo, sherpa-onnx, voxtral.c, MLX ports
  .venv/  .build/  scratch/  artifacts/   # Python env, Swift build, temp files, benchmark evidence
```

Checkpoints are downloaded straight into `models/` (Hugging Face `local_dir`), so the weights we use are in the repo, not only in a system cache. The research notes from the planning session move into `docs/research/`.

## Milestones

### M0: Scaffolding, truth pack and reference

- Run `git init`. Add `.gitignore`, and set up the Swift package and the `uv` project.
- **Truth pack:**
  - download all three checkpoints (about 14 GB), the sherpa-onnx streaming exports and the pinned reference source checkouts;
  - write `PINS.md` and the verify script;
  - create `PERF_LOG.md` and `DISCREPANCIES.md` (empty);
  - write the per-model specs and tensor-name maps.
- Write reference dump scripts. For each model, they save `.npy` golden tensors at every stage boundary: raw audio, mel features, subsampling output, the output of each encoder layer, decoder states, logits and tokens. There are two modes:
  - **offline:** the full utterance with the model's streaming attention mask, used for the L0–L4 ladder;
  - **reference streaming:** the official streaming loop, whose chunk-by-chunk outputs and final transcripts are the L5 baseline.
- **Measure the reference's noise floor** for each model and commit the envelope.
- Run the reference streaming pipelines on a small development set to confirm we reproduce the published WER: Nemotron and Parakeet on LibriSpeech test-clean/test-other, Voxtral on its card numbers.

**Done when:** pins and hashes are committed and verified, golden tensors and noise envelopes exist for all three models, and the reference WER matches the published figures to within noise.

### M1: The Metal core

Build the runtime and the kernel library that all three models share:

- **Runtime:**
  - device and queue setup;
  - a buffer arena with a planned, preallocated layout, so no allocation happens per chunk;
  - one command buffer per streaming step;
  - GPU timestamps and stage timing that stay in the engine permanently.
- **Weight loader:** memory-maps our format and wraps the mapping in `MTLBuffer`s without copying. This works because Apple-silicon memory is unified.
- **Weight format:** a flat binary file. It holds a sorted JSON header (tensor name, dtype, shape, offset, quantization parameters, a hash of the source checkpoint) followed by page-aligned tensor data.
- **Measure this M3's real peaks first:** memory bandwidth (a copy/stream test), fp16 and fp32 GEMM TFLOPS, and the cost of a command-buffer round trip. These numbers are the roofline for every later timing row.
- **Kernels:**
  - GEMM (fp32, fp16 and bf16 activations; fp16, int8 and int4 group-quantized weights);
  - GEMV for the single-token decode;
  - fused bias, activation (SiLU, GELU with clamped tanh, ReLU, GLU) and residual additions;
  - LayerNorm and RMSNorm;
  - softmax;
  - attention with arbitrary masks: chunked/limited, relative position (Transformer-XL style), and causal sliding-window with rotary position embeddings (RoPE) and grouped-query attention (GQA);
  - depthwise 1-D convolution, causal and symmetric;
  - 2-D convolution for the subsampling stems;
  - BatchNorm folded into the weights at conversion time;
  - LSTM cell;
  - ring-buffer KV-cache writes.
- **Mel frontend:** STFT and log-mel with vDSP on the CPU, with a Metal version as an option. It is incremental: it keeps overlap state between audio callbacks and reproduces the reference framing exactly, including center padding.
- **Tests:** each kernel is tested (L1) against numpy/torch on random inputs and on real layer tensors. Microbenchmarks report each kernel's distance from the measured roofline.
- Correctness comes first: the first version of every model runs in **fp32**. fp16, bf16 and quantization are later changes, made through the performance ritual.

**Done when:** every kernel passes L1, the measured M3 peaks are recorded, and GEMM/GEMV reach a stated fraction of those peaks.

### M2: Nemotron Streaming (160 ms and 560 ms)

This model comes first. It has the cleanest design (cache-aware streaming, causal convolutions, no normalization statistics), and its encoder and decoder are shared with Parakeet.

- **Encoder:**
  - causal `dw_striding` 8× subsampling with a 9-frame pre-encode cache;
  - 24 FastConformer layers (relative-position attention; causal depthwise convolution with LayerNorm).
- **Our cache design:** we do not reproduce NeMo's approach of joining tensors and slicing them every step. Instead:
  - the attention left context (70 frames per layer) lives in a per-layer ring buffer;
  - we cache the **projected K and V**, not the layer inputs, so past frames are never projected again;
  - the convolution state (8 frames per layer) is a small ring buffer;
  - the whole chunk runs as one command buffer.
- **Decoder:** RNN-T greedy decoding with a 2-layer LSTM (640 wide) and the joint network. It runs per token, so we try three options and keep the fastest:
  - (a) on the GPU, inside the same command buffer as the encoder;
  - (b) on the CPU through Accelerate;
  - (c) joint projection on the GPU, LSTM on the CPU.
- **Tokenizer:** a SentencePiece decoder for the 1024-token vocabulary, written in Swift. It must round-trip token ids exactly (L0).
- `dictate-cli` streams a WAV file or the microphone and prints partial transcripts.
- **Performance pass:** switch fp32 to fp16 and fuse kernels, following the ritual. Compare against sherpa-onnx and the MLX port.

**Done when:**

- L0–L4 pass against the offline golden tensors;
- L5 transcripts at both 160 ms and 560 ms match the NeMo reference streaming output (WER within the measured budget);
- per-chunk compute time, RTF and the roofline distance are recorded for both settings;
- the comparisons against sherpa-onnx and MLX are recorded with fairness controls.

### M3: Parakeet Unified (560 ms)

The reference inference is "buffered". Every 160 ms it recomputes normalization statistics and re-runs the full 24-layer encoder over a 6.16 s window, then keeps only the frames for the new chunk. That is roughly 500+ GFLOP/s of steady GPU load. The weights allow a smarter approach, and this milestone explores it.

- Reuse the M2 kernels. The model-specific pieces are:
  - non-causal subsampling;
  - symmetric depthwise convolution with BatchNorm folded into the weights;
  - `xscaling`;
  - the `chunked_limited_with_rc` mask set to [left 70, chunk 2, right 5] frames.
- **Stage A, exact baseline:** implement the buffered algorithm faithfully and pass L0–L5 against the reference. This gives the quality baseline and the "expensive" cost figure.
- **Stage B, efficient variants.** Each variant is a kill-switched discrepancy entry, evaluated at L5 for its WER change against its compute saving:
  1. **Streaming normalization:** replace per-window `per_feature` mean/std with running (EMA) statistics or fixed global statistics. This removes the main reason the whole window has to be recomputed.
  2. **Frozen-prefix caching:** chunk *c* only looks 5 frames to the right, and its right-context frames lie in the region still being recomputed. Frames far enough in the past therefore change very little between steps. We cache the per-layer K/V of frames older than a settling horizon *H* and recompute only the most recent *H* frames. We sweep *H* from 0 (fully cached, like Nemotron) up to the full window (exact).
  3. **Shorter left context:** use fewer than 70 frames if WER holds.
  4. **Right-context reuse:** each step adds only 2 new frames. The 5 lookahead frames overlap with the next step, so partial results can carry forward.
- Choose the default configuration from the Pareto front of WER against GPU time and power. The combined set of lossy changes is gated as a whole, because changes that each pass alone can add up past the budget.

**Done when:** Stage A passes the ladder, and at least one Stage B variant uses substantially less compute with its WER change measured and recorded.

### M4: Voxtral Mini 4B Realtime (480 ms)

On the base M3 this model is limited by memory bandwidth. The 16-bit weights are about 8.9 GB. Each 80 ms step reads roughly 7 GB of decoder weights, plus encoder weights for the new audio. That is about 90 ms per step at 100 GB/s, which is too slow for real time. Quantization and memory traffic are therefore the core of this milestone.

- **Frontend:** 128-bin Slaney log-mel at 100 Hz with Voxtral's fixed normalization. The audio gets 2.56 s of silence prepended, as the reference does.
- **Encoder:**
  - a causal convolution stem (stride 2, giving 50 Hz frames) that keeps a padding cache;
  - 32 causal transformer layers with RoPE and a 750-frame sliding window;
  - an adapter that joins 4 frames into one, giving 12.5 Hz.
- **Our incremental encoder:** it processes only the 4 new frames per step, with a per-layer KV ring buffer of 750 frames. This means a small-batch GEMM rather than re-encoding.
- **Decoder:**
  - 26 layers with GQA (32 query heads, 8 KV heads) and RoPE;
  - a delay-conditioned adaptive norm before each feed-forward block, conditioned on `n_delay = 6`;
  - a tied 131k-token output head;
  - an input at each step made from the previous token's embedding **plus** the audio embedding (the two are added, not interleaved).
- **Tokenizer:** a Tekken v7 decoder in Swift that handles the streaming control tokens (PAD and WORD).
- **Parity first:** L0–L4 pass at fp32/bf16 against transformers before any quantization. On 24 GB this runs offline, not in real time.
- **Quantization by fixed recipe, one stage at a time:**
  1. Decoder feed-forward weights to int8, per output channel, with group-wise scales;
  2. then the decoder attention weights;
  3. then int4 for the decoder feed-forward weights;
  4. then the encoder, as a separate step.

  Norms, embeddings, the adaptive-norm conditioning and (at first) the output head stay in higher precision. Each stage is a discrepancy entry with its measured WER, milliseconds per step and memory use. Quality is claimed per slice; the 16-bit build remains available as the reference. The expected range is Q8 at about 45–50 ms per step down to Q4 at about 25–30 ms. These are estimates to be tested, not results.
- **Memory traffic ideas** (each is a separate change):
  - The output head is about 400M parameters read every step. Options: an int8 head that shortlists the top candidates, then an exact rescoring of just those rows so the chosen token (argmax) is unchanged; or early exit when the model predicts a PAD token.
  - Keep the KV cache in fp16 or int8.
  - Run the encoder step and the decoder step concurrently where their dependencies allow.
- **Long sessions:** a rolling decoder KV cache with a window of 8192 tokens (about 11 minutes), then compaction. Beyond that length we decide on a policy.
- Resolve the RoPE convention once, in the converter, and test it with the ladder.

**Done when:** L0–L4 pass at high precision, a quantized version runs faster than real time on this M3 with its WER cost recorded per slice, and memory use is reported.

### M5: Accelerator placement study

- Convert one encoder chunk step (the Nemotron encoder first, then the Voxtral encoder) to CoreML, running on the Neural Engine. Compare it with our Metal kernels on latency, throughput, power (`powermetrics`) and how awkwardly the streaming caches cross the CoreML boundary.
- Compare the RNN-T LSTM and joint network on the GPU with the same work on the CPU through Accelerate.
- Decide where each part runs. Keep an alternative path only if it wins clearly. A negative result is a valid outcome and goes into `docs/PERF_LOG.md`.

### M6: Quality and latency evaluation (before any UI)

This produces a written report that compares every model and configuration head to head, so we can choose what the app ships.

- **Test sets** (stored in `data/`):
  - LibriSpeech test-clean and test-other (clean read speech);
  - a spontaneous, accented or noisy set (for example TED-LIUM 3, Earnings-21 or AMI), chosen by availability and license;
  - **our own dictation set:** recordings made on this MacBook's built-in microphone, in real rooms, of real dictation (emails, code comments, names and jargon), with hand-corrected transcripts. This set matters most.
- **Metrics:**
  - WER, both normalized and with case and punctuation, reported per slice with the worst-case tail, not just the mean;
  - **latency:**
    - time from a word's end in the audio to its first appearance, measured with forced-alignment timestamps;
    - time until that word stops changing;
    - time to the first word;
  - **stability:** how often partial transcripts rewrite words already shown;
  - RTF, per-step p50 and p99 compute time, GPU and CPU utilization, power draw, and memory footprint;
  - behaviour over long sessions (30+ minutes), after the laptop has heated up, and while other GPU work is running.
- **Configurations:**
  - Nemotron at 160 ms and 560 ms;
  - Parakeet in exact mode and in the best efficient variants;
  - Voxtral at Q8 and Q4, and at 16-bit if it fits;
  - the pinned reference and competitor pipelines as baselines.
- **Deliverable:** a report containing tables, WER-against-latency Pareto plots, listening notes on failure cases and a recommendation for the app's default model and settings.

### M7 (later): The Mac app

This starts only after the M6 review. It will be a SwiftUI menu-bar and dictation app on top of `DictatorEngine`, planned separately.

## Risks and open questions

- **NeMo on Python 3.12:** NeMo may need pinned or older versions. We will pin them in `reference/pyproject.toml`, following the model cards.
- **Open questions:** the open questions in each model spec (above) block the kernels they affect.
- **Voxtral quantization quality:** quality at Q4 with a 480 ms delay is unknown. M4 measures it.
- **Real bandwidth:** this M3's actual memory bandwidth is measured in M1, because it sets Voxtral's budget.
- **Thermals:** this laptop drifts as it heats up, so formal timings use the quiet-window protocol.
- **Disk:** about 100 GB is free. Checkpoints, converted weights, Python environments and datasets need an estimated 40–60 GB, mostly inside this folder. We may need to clean up.

## References

- NeMo:
  - `conformer_encoder.py`, `subsampling.py`, `conformer_modules.py` and `features.py`;
  - `speech_to_text_cache_aware_streaming_infer.py` (Nemotron);
  - `speech_to_text_streaming_infer_rnnt.py` (Parakeet).
- Hugging Face transformers: `NemotronAsrStreamingForRNNT` and `VoxtralRealtimeForConditionalGeneration`.
- Independent ports for cross-checks and competitor comparisons:
  - sherpa-onnx streaming exports at 160 ms and 560 ms;
  - `antirez/voxtral.c` (C and Metal; its `MODEL.md` specification);
  - MLX conversions;
  - FluidInference CoreML conversions.
- Methodology: the `ai-model-into-rust-mega-fused-hyper-kernel` skill (performance ritual, parity ladder, evidence standards), minus the Rust-specific parts.
