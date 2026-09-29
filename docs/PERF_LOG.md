# Performance log

This file lists kept and rejected performance changes. Step 1 of the performance ritual reads it. Timings marked *noisy* were taken while other heavy jobs were running on the machine, and are indicative only.

## Measured peaks (this M3, 2026-09-28)

These are the first measurements, taken while downloads were running, so they are noisy. They will be re-measured in a quiet window.

| Measurement | Value |
|---|---|
| Read bandwidth | 87 GB/s |
| Copy bandwidth | 84 GB/s |
| FMA, fp32 | 2.95 TFLOPS |
| FMA, fp16 | 3.17 TFLOPS |
| Empty command-buffer round trip | 154 µs |
| `maxBufferLength` | 13.6 GB |
| Recommended working set | 18.2 GB |

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
| R4 | Row-interleaved bf16/f16 GEMV (same idea as K5) | neutral to worse at M=4 (2 interleaved pairs) — 2-byte path is load-issue bound, not threadgroup bound | M > 8 per call, or a different register budget |
| R3 | GEMV without threadgroup staging (activations read from device/cache directly, 8 simdgroups) | not better, mostly worse on 2 pairs across bf16/int8 | a GPU with larger L1 or different cache policy |
| R2 | GEMV: load activations into registers once per K step and reuse across the 4 rows of a simdgroup | no gain or slightly worse (M=4 bf16 ≈ 25–31 GB/s) — the limit was bytes per load, fixed by K4 | the kernel becomes threadgroup-memory bound |
| R1 | GEMM v2 with 64x64 tiles (2x2 simdgroups, 4x4 fragments each, BK=32, register prefetch) | slower than v1 at 4096^3 (0.33 vs 0.52 TFLOPS f32), probably register pressure / occupancy; the 32x64 tile (1x4 simdgroups, 4x2 fragments) won at every shape | a different register budget (e.g. no prefetch registers) or new GPU generation |
