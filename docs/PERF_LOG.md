# Performance log

This file lists kept and rejected performance changes. Step 1 of the performance ritual reads it. Timings marked *noisy* were taken while other heavy jobs were running on the machine, and are indicative only.

## Measured peaks (this M3, quiet, 2026-09-29)

| Measurement | Value |
|---|---|
| Read bandwidth | 93.5 GB/s |
| Copy bandwidth | 90.8 GB/s (read + write) |
| FMA, fp32 | 2.95 TFLOPS |
| FMA, fp16 | 3.17 TFLOPS |
| GEMM f32 4096³ (our kernel) | 2.33 TFLOPS |
| GEMM f16 4096³ (our kernel) | 2.06 TFLOPS |
| Empty command-buffer round trip | 151 µs |
| `maxBufferLength` / recommended working set | 13.6 GB / 18.2 GB |

Command: `dictate-cli peaks`. The 2026-09-28 numbers (87 GB/s read) were taken during downloads.

## Step times (quiet, 2026-09-29)

One 33.9 s clip (`test-clean/1995/1836/1995-1836-0004`), 3 runs after a warm-up, wall time per streaming step. Raw output: `artifacts/timing_quiet_2026-09-29.txt`; command: `scratch/ab/timing.sh`.

| Configuration | Step p50 | Step p99 | Step budget |
|---|---|---|---|
| Nemotron 160 ms, f16 | 21.3–22.3 ms | 23–26 ms | 160 ms |
| Nemotron 560 ms, f16 | 36.1–36.7 ms | 39 ms | 560 ms |
| Nemotron 160 ms, f32 | 38.4–38.9 ms | 43–44 ms | 160 ms |
| Parakeet exact buffered, f16 | 83.8–86.8 ms | 96–132 ms | 160 ms |
| Parakeet horizon 4, f16 | 48.1–52.1 ms | 57–61 ms | 160 ms |
| Voxtral q8h (int8 + int8 head) | 88.8–100.5 ms | 105–145 ms | 80 ms |
| Voxtral q8 | 103.7–105.0 ms | 122–145 ms | 80 ms |
| Voxtral q4ffn | 103.8–113.7 ms | 121–168 ms | 80 ms |
| Voxtral bf16 | 156–191 ms | 226–296 ms | 80 ms |

Voxtral is under budget on a 10.4 s clip (q8h and q4ffn both 75–81 ms p50, 2 interleaved pairs) but not on this 33.9 s clip; the cost grows with stream position (not yet profiled; deprioritized 2026-09-29).

GEMV bandwidth at the Voxtral shapes (`dictate-cli probe voxtral`, quiet): M=1 int8 and bf16 reach 80–91 GB/s (the decoder is at the bandwidth limit); M=1 q4 reaches only 44–46 GB/s (unpacking-bound, so q4 decoder FFN is no faster than int8: 0.311 vs 0.331 ms for 9216×3072); M=4 int8 (encoder) reaches 38–46 GB/s.

## Long session, contention and Neural Engine (quiet, 2026-09-29)

Raw output: `artifacts/timing_m6/`; command: `scratch/m6_timing.sh`.

- 30 minutes of test-clean, back to back: Nemotron 160 ms p50 24 ms in both the first and last 5 minutes (p99 45 ms); Parakeet h4 53 → 48 ms (p99 102 ms); Voxtral q8h 99 → 174 ms (p99 224 ms). Voxtral's step grows with stream position and settles at about 180 ms, more than twice its 80 ms budget.
- Memory: the live path (`eval`) goes from 175 MB (10 s clip) to 239 MB (5 min) for Nemotron. The 1.3 GB peak in the 30-minute `transcribe` run is the CLI computing features for the whole file up front (1.18 GB MALLOC_LARGE after 25 s), not engine growth. Wrapping each step in an autorelease pool changed nothing (368 MB both ways on 5 min) and was reverted. Voxtral's live wrapper never dropped consumed audio or mel rows; it now does (commit ade293d).
- Contention, with a Voxtral run saturating the GPU: Nemotron 160 ms p50 80 ms, p99 216–371 ms; Parakeet h4 p50 123–136 ms, p99 204–401 ms.
- Nemotron step through Core ML: Neural Engine 21.2 ms, Core ML GPU 22.5 ms, CPU 27.4 ms; our Metal 22.8 ms in the same session. Outputs within 7e-3 of the engine.
- sherpa-onnx (4 CPU threads), same Nemotron 160 ms checkpoint on 20 test-clean utterances: RTF 0.25 int8, 0.96 fp32; ours 0.16.

## Nemotron on the Neural Engine (default backend)

Nemotron's encoder step runs through Core ML on the Neural Engine by default (`--backend metal` selects the GPU; missing Core ML models fall back to Metal). The frontend and RNN-T decoding stay on the CPU. Models come from `tools/nemotron_ane.py`:

- **Stateful (default):** one steady-state model per latency with the K/V and conv caches held as Core ML state. The first chunk runs through the same model behind 16 leading zero mel rows masked at every subsampling stage (16 → 8 → 4 → 2 rows, the 2 dropped frames), which reproduces its causal zero padding; on the CPU its outputs match the stateless models to 0.004–0.006 (outputs up to about 1.4).
- **Stateless (`DICTATOR_ANE_STATELESS=1`):** first-chunk and steady models with the caches passed through inputs and outputs.
- **560 ms steady step is two models** (subsampling, then layers): the fused graph (65 mel rows → 9 subsampled frames feeding the layers) failed the Neural Engine compile (`ANECCompile() FAILED`) while each half compiled. The disk was nearly full during that bisect, which also breaks Neural Engine compiles, so the fused graph has not been retested with free space.

Measured 2026-09-30, quiet machine:

| | Step p50, 160 ms | Step p50, 560 ms |
|---|---|---|
| Neural Engine, stateful | 18.8 ms | 21.4 ms |
| Neural Engine, stateless | 19.1 ms | 21.4 ms |
| Metal | 22.1 ms | 36.0 ms |

Accuracy, first 200 utterances, against Metal on the same utterances: 160 ms 2.05 / 5.73% (Metal 2.10 / 5.77%), 560 ms test-other 5.02% (Metal 5.02%). 4–10% of transcripts differ (fp16); stateful and stateless transcripts are identical on the 33.9 s clip at both latencies.

Power (`sudo scripts/power.sh ane`; the script was removed in the release cleanup and is in the history before it), looping a 5-minute file with a warm compile cache; idle 0.31 W:

| | CPU | GPU | Neural Engine | Total |
|---|---|---|---|---|
| 160 ms, Neural Engine, stateful | 1.80 W | 0.10 W | 2.76 W | 4.65 W |
| 160 ms, Neural Engine, stateless | 1.95 W | 0.08 W | 2.74 W | 4.77 W |
| 160 ms, Metal | 0.35 W | 3.73 W | 0 W | 4.08 W |
| 560 ms, Neural Engine, stateful | 1.44 W | 0.07 W | 2.48 W | 3.99 W |
| 560 ms, Neural Engine, stateless | 1.20 W | 0.07 W | 2.65 W | 3.91 W |
| 560 ms, Metal | 1.34 W | 6.49 W | 0 W | 7.84 W |

Energy per second of audio (power × step time): 560 ms on the Neural Engine uses about 3.3× less than Metal; 160 ms is within about 3% of Metal. Both leave the GPU free. Stateful caches save about 0.1 W; the Neural Engine path still spends 1.1–1.5 W more CPU than Metal, cause not yet found.

Operational notes: a Neural Engine compile-cache miss costs about 40 s of CPU at model load; the cache is invalidated by a new build of the binary and is evicted when the disk is nearly full (Core ML needs several GB free). An app should load the Core ML model in the background and run on Metal until it is ready.

## Kept

| # | Change | Result | Correctness |
|---|---|---|---|
| K1 | Skinny GEMV kernel for M ≤ 8 (weight rows read once, 4-wide loads, K-blocked activations); kill-switch `DICTATOR_GEMV=0` | M=2 4096×1024 fp32: 1.92 → 0.42 ms isolated, 0.22 ms back to back (≈77 GB/s). Nemotron 160 ms chunk: 228 → 37.7 ms GPU | GEMM tests pass; ladder tokens exact |
| K3 | GEMM v2: 32x64 tiles, BK=32, 4-wide global loads staged through registers with prefetch of the next K tile; kill-switch `DICTATOR_GEMM_V1=1` | f32 4096^3: 0.52 → 2.26 TFLOPS; Parakeet 77x4096x1024: 2.16 → 0.49 ms; Parakeet exact buffered step 417 → 125 ms wall (*noisy*) | GEMM tests pass; Nemotron and Parakeet ladders 20/20 token-exact |
| K4 | GEMV: 16-byte weight loads for 2-byte types (8 bf16/f16 per lane instead of 4) | Voxtral bf16 M=4: ≈30 → 54 GB/s; M=1: ≈60 → 70–74 GB/s; Voxtral step GPU enc 144 → 84 ms, dec 177 → 120 ms (*noisy*) | GEMM tests pass; transcript unchanged |
| K5 | int8 GEMV: rows interleaved (all row loads issued first) and each lane's activation slice read from threadgroup memory once per K chunk, reused across rows | Voxtral encoder shapes (M=4, int8), two interleaved pairs: 1.5–2.5x (e.g. enc down 0.325 → 0.168 ms, 0.455 → 0.179 ms) (*noisy*: reference jobs on MPS) | QuantTests pass |
| K6 | Voxtral: multi-query attention for the 4-frame encoder step (each K/V row read once per head) | ≈5% encoder GPU, won 2/2 interleaved pairs (*noisy*) | matches per-query kernel to 1e-5 |
| K7 | Voxtral silent-prefix snapshot: prefill computes 8 positions instead of 39 | warm prefill 1.3–1.9 s → 0.74–0.93 s (*noisy*) | token stream identical |
| K8 | Voxtral fp16 K/V caches (halves attention bytes and decoder KV memory 1.85 → 0.93 GB); kill-switch `DICTATOR_KV_F32=1` | not timed separately | token stream identical on clip |
| K9 | Split-K multi-query attention (4 key partitions per head + combine); `DICTATOR_ATTN_SPLITS=1` disables | encoder attention section −12% / −26% on 2 interleaved pairs (*noisy*, GPU saturated) | matches per-query kernel to 1e-5 |
| K2 | fp16 encoder weights (D1) | Nemotron 160 ms chunk: 37.7 → 25.7 ms GPU (*noisy*) | tokens exact on 20/20 dev20, 9/9 streaming |

## Rejected

| # | Change | Result | Retry only if |
|---|---|---|---|
| R7 | GEMV pipelines specialized on M (function constant; loops unroll, accumulators sized) | neutral at every Voxtral shape, 2 interleaved pairs | never; the compiler already keeps the accumulators in registers |
| R6 | int8 GEMV: convert weights to float once per chunk instead of once per activation row | neutral at every Voxtral shape (M=4 enc 42 GB/s both), 2 interleaved pairs | never; the compiler hoists it |
| R5 | q4 GEMV with 16 weights per lane (8-byte loads) so a 512-wide K block fills all 32 lanes | neutral (q4 M=1 45 GB/s both; Voxtral step 78 ms both), 2 interleaved pairs | the q4 unpack becomes cheap (then lane occupancy may matter) |
| R4 | Row-interleaved bf16/f16 GEMV (same idea as K5) | neutral to worse at M=4 (2 interleaved pairs) — 2-byte path is load-issue bound, not threadgroup bound | M > 8 per call, or a different register budget |
| R3 | GEMV without threadgroup staging (activations read from device/cache directly, 8 simdgroups) | not better, mostly worse on 2 pairs across bf16/int8 | a GPU with larger L1 or different cache policy |
| R2 | GEMV: load activations into registers once per K step and reuse across the 4 rows of a simdgroup | no gain or slightly worse (M=4 bf16 ≈ 25–31 GB/s) — the limit was bytes per load, fixed by K4 | the kernel becomes threadgroup-memory bound |
| R1 | GEMM v2 with 64x64 tiles (2x2 simdgroups, 4x4 fragments each, BK=32, register prefetch) | slower than v1 at 4096^3 (0.33 vs 0.52 TFLOPS f32), probably register pressure / occupancy; the 32x64 tile (1x4 simdgroups, 4x2 fragments) won at every shape | a different register budget (e.g. no prefetch registers) or new GPU generation |
