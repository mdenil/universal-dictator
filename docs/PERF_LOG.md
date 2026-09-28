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
| K2 | fp16 encoder weights (D1) | Nemotron 160 ms chunk: 37.7 → 25.7 ms GPU (*noisy*) | tokens exact on 20/20 dev20, 9/9 streaming |

## Rejected

| # | Change | Result | Retry only if |
|---|---|---|---|
| R1 | GEMM v2 with 64x64 tiles (2x2 simdgroups, 4x4 fragments each, BK=32, register prefetch) | slower than v1 at 4096^3 (0.33 vs 0.52 TFLOPS f32), probably register pressure / occupancy; the 32x64 tile (1x4 simdgroups, 4x2 fragments) won at every shape | a different register budget (e.g. no prefetch registers) or new GPU generation |
