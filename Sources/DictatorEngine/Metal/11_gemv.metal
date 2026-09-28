// Skinny GEMM for streaming: M <= 8 activation rows against a large weight matrix. This is
// memory-bound on the weights, so each weight row is read exactly once, with 4-wide vector loads,
// and dotted against all M activation rows held in threadgroup memory.
//   C[m, n] = out_scale * act(alpha * sum_k A[m,k] W[n,k] + bias[n]) + res_scale * R[m, n]
// One simdgroup computes GEMV_ROWS output columns n for all m.

constant constexpr int GEMV_MAX_M = 8;
constant constexpr int GEMV_ROWS = 4;      // output columns per simdgroup
constant constexpr int GEMV_SG = 4;        // simdgroups per threadgroup
constant constexpr int GEMV_KB = 512;      // K block staged in threadgroup memory (M*KB floats <= 16 KB)

template <typename T> inline float4 load4(device const T* p) { return float4(*(device const vec<T, 4>*)p); }
template <> inline float4 load4<ushort>(device const ushort* p) {
    ushort4 b = *(device const ushort4*)p;
    return float4(bf16_to_float(b.x), bf16_to_float(b.y), bf16_to_float(b.z), bf16_to_float(b.w));
}

// Requires K % 4 == 0 and rows 16-byte aligned for the vector loads.
template <typename TA, typename TW, typename TC>
kernel void gemv_nt(
    device const TA* A [[buffer(0)]],
    device const TW* W [[buffer(1)]],
    device TC* C [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device const TC* R [[buffer(4)]],
    constant GemmParams& p [[buffer(5)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float4 xs[GEMV_MAX_M * GEMV_KB / 4];
    const int M = p.M, K = p.K;
    const int n0 = (int(tgp) * GEMV_SG + int(sg)) * GEMV_ROWS;

    float acc[GEMV_ROWS][GEMV_MAX_M];
    for (int r = 0; r < GEMV_ROWS; r++)
        for (int m = 0; m < GEMV_MAX_M; m++) acc[r][m] = 0.0f;

    for (int kb = 0; kb < K; kb += GEMV_KB) {
        const int kl = min(GEMV_KB, K - kb);
        const int kl4 = kl / 4;
        for (int i = tid; i < M * kl4; i += GEMV_SG * 32) {
            int m = i / kl4, k4 = i - m * kl4;
            xs[m * (GEMV_KB / 4) + k4] = load4(A + (long)m * p.lda + kb + k4 * 4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < GEMV_ROWS; r++) {
            int n = n0 + r;
            if (n >= p.N) break;
            device const TW* wr = W + (long)n * p.ldb + kb;
            for (int k4 = int(lane); k4 < kl4; k4 += 32) {
                float4 w = load4(wr + k4 * 4);
                for (int m = 0; m < M; m++) acc[r][m] += dot(w, xs[m * (GEMV_KB / 4) + k4]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int r = 0; r < GEMV_ROWS; r++) {
        int n = n0 + r;
        if (n >= p.N) break;
        for (int m = 0; m < M; m++) {
            float v = simd_sum(acc[r][m]);
            if (lane == 0) {
                v = p.alpha * v;
                if (p.has_bias) v += bias[n];
                v = apply_act(v, p.act) * p.out_scale;
                if (p.has_residual) v += p.res_scale * to_f(R[(long)m * p.ldr + n]);
                C[(long)m * p.ldc + n] = from_f<TC>(v);
            }
        }
    }
}

#define INSTANTIATE_GEMV(NAME, TA, TW, TC) \
    template [[host_name(NAME)]] kernel void gemv_nt<TA, TW, TC>( \
        device const TA*, device const TW*, device TC*, device const float*, device const TC*, \
        constant GemmParams&, uint, uint, uint, uint);

INSTANTIATE_GEMV("gemv_nt_f32_f32_f32", float, float, float)
INSTANTIATE_GEMV("gemv_nt_f16_f16_f16", half, half, half)
INSTANTIATE_GEMV("gemv_nt_f32_f16_f32", float, half, float)
INSTANTIATE_GEMV("gemv_nt_f16_f16_f32", half, half, float)
INSTANTIATE_GEMV("gemv_nt_f32_bf16_f32", float, ushort, float)
