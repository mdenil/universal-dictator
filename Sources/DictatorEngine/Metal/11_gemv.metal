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

// 16-byte weight loads: 8 elements for 2-byte types, 4 for float.
template <typename T> struct WVec { static constant constexpr int N = 4; };
template <> struct WVec<half> { static constant constexpr int N = 8; };
template <> struct WVec<ushort> { static constant constexpr int N = 8; };
template <> struct WVec<char> { static constant constexpr int N = 16; };

inline float4 c4(uint u) { return float4(as_type<char4>(u)); }

inline void load8(device const half* p, thread float4& a, thread float4& b) {
    uint4 v = *(device const uint4*)p;
    a = float4(float2(as_type<half2>(v.x)), float2(as_type<half2>(v.y)));
    b = float4(float2(as_type<half2>(v.z)), float2(as_type<half2>(v.w)));
}
inline void load8(device const ushort* p, thread float4& a, thread float4& b) {
    uint4 v = *(device const uint4*)p;
    a = float4(as_type<float>(v.x << 16), as_type<float>(v.x & 0xFFFF0000u), as_type<float>(v.y << 16), as_type<float>(v.y & 0xFFFF0000u));
    b = float4(as_type<float>(v.z << 16), as_type<float>(v.z & 0xFFFF0000u), as_type<float>(v.w << 16), as_type<float>(v.w & 0xFFFF0000u));
}
inline void load8(device const char* p, thread float4& a, thread float4& b) {
    a = float4(*(device const char4*)p); b = float4(*(device const char4*)(p + 4));
}
inline void load8(device const float* p, thread float4& a, thread float4& b) {
    a = *(device const float4*)p; b = *(device const float4*)(p + 4);
}

// Requires K % 4 == 0 and rows 16-byte aligned for the vector loads (K % 8 == 0 for 2-byte weights).
template <typename TA, typename TW, typename TC>
kernel void gemv_nt(
    device const TA* A [[buffer(0)]],
    device const TW* W [[buffer(1)]],
    device TC* C [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device const TC* R [[buffer(4)]],
    constant GemmParams& p [[buffer(5)]],
    device const float* wscale [[buffer(6)]],
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
            if (WVec<TW>::N == 16) {
                for (int k16 = int(lane); k16 < kl4 / 4; k16 += 32) {
                    uint4 v = *(device const uint4*)((device const char*)wr + k16 * 16);
                    float4 w0 = c4(v.x), w1 = c4(v.y), w2 = c4(v.z), w3 = c4(v.w);
                    for (int m = 0; m < M; m++) {
                        threadgroup const float4* x = xs + m * (GEMV_KB / 4) + 4 * k16;
                        acc[r][m] += dot(w0, x[0]) + dot(w1, x[1]) + dot(w2, x[2]) + dot(w3, x[3]);
                    }
                }
            } else if (WVec<TW>::N == 8) {
                for (int k8 = int(lane); k8 < kl4 / 2; k8 += 32) {
                    float4 wa, wb;
                    load8(wr + k8 * 8, wa, wb);
                    for (int m = 0; m < M; m++)
                        acc[r][m] += dot(wa, xs[m * (GEMV_KB / 4) + 2 * k8]) + dot(wb, xs[m * (GEMV_KB / 4) + 2 * k8 + 1]);
                }
            } else {
                for (int k4 = int(lane); k4 < kl4; k4 += 32) {
                    float4 w = load4(wr + k4 * 4);
                    for (int m = 0; m < M; m++) acc[r][m] += dot(w, xs[m * (GEMV_KB / 4) + k4]);
                }
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
                if (p.has_wscale) v *= wscale[n];
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
        constant GemmParams&, device const float*, uint, uint, uint, uint);

INSTANTIATE_GEMV("gemv_nt_f32_f32_f32", float, float, float)
INSTANTIATE_GEMV("gemv_nt_f16_f16_f16", half, half, half)
INSTANTIATE_GEMV("gemv_nt_f32_f16_f32", float, half, float)
INSTANTIATE_GEMV("gemv_nt_f16_f16_f32", half, half, float)
INSTANTIATE_GEMV("gemv_nt_f32_bf16_f32", float, ushort, float)
INSTANTIATE_GEMV("gemv_nt_f32_i8_f32", float, char, float)
