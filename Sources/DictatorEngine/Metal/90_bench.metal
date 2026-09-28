// Microbenchmarks for measuring this GPU's real peaks (the roofline for every timing row).

kernel void bench_copy_f4(device const float4* src [[buffer(0)]],
                          device float4* dst [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    dst[i] = src[i];
}

// Read-only bandwidth: each thread reads `per_thread` float4s with a grid-stride and writes one value.
kernel void bench_read_f4(device const float4* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant uint& per_thread [[buffer(2)]],
                          uint i [[thread_position_in_grid]],
                          uint n [[threads_per_grid]]) {
    float4 s = 0;
    for (uint k = 0; k < per_thread; k++) s += src[i + k * n];
    dst[i] = s.x + s.y + s.z + s.w;
}

// Peak FMA throughput: long dependent-free chains of fma per thread.
kernel void bench_fma_f32(device float* dst [[buffer(0)]],
                          constant uint& iters [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    float a0 = float(i) * 1e-7f, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    float a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    const float m = 0.9999f, c = 1e-4f;
    for (uint k = 0; k < iters; k++) {
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
    }
    dst[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

kernel void bench_fma_f16(device half* dst [[buffer(0)]],
                          constant uint& iters [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    half a0 = half(i & 7) * 0.01h, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    half a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    const half m = 0.999h, c = 0.001h;
    for (uint k = 0; k < iters; k++) {
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
    }
    dst[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

kernel void bench_noop(device float* dst [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0xFFFFFFFF) dst[0] = 1;
}
