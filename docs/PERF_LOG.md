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
| K2 | fp16 encoder weights (D1) | Nemotron 160 ms chunk: 37.7 → 25.7 ms GPU (*noisy*) | tokens exact on 20/20 dev20, 9/9 streaming |

## Rejected

None yet.
