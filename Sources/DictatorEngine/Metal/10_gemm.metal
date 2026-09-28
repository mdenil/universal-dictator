// GEMM: C[M,N] = out_scale * act(alpha * A[M,K]·B[N,K]^T + bias[N]) + res_scale * R[M,N]
// B stays in the checkpoint's native [out, in] layout, so weights never need transposing.
// Accumulation is always fp32. A and B may be f32 / f16 / bf16 (bf16 passed as ushort).

struct GemmParams {
    int M, N, K;
    int lda, ldb, ldc;     // row strides in elements
    int has_bias;
    int act;
    int has_residual;
    int ldr;
    float alpha;
    float res_scale;
    float out_scale;
    int has_wscale;        // int8 weights: multiply the dot product by wscale[n]
};

constant constexpr int GBM = 64;
constant constexpr int GBN = 64;
constant constexpr int GBK = 16;
constant constexpr int GPAD = 4;

template <typename TA, typename TB, typename TC>
kernel void gemm_nt(
    device const TA* A [[buffer(0)]],
    device const TB* B [[buffer(1)]],
    device TC* C [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device const TC* R [[buffer(4)]],
    constant GemmParams& p [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float As[GBM * (GBK + GPAD)];
    threadgroup float Bs[GBN * (GBK + GPAD)];
    threadgroup float Cs[GBM * GBN];

    const int m0 = int(tgp.y) * GBM;
    const int n0 = int(tgp.x) * GBN;
    const int sr = int(sg / 2) * 32;
    const int sc = int(sg % 2) * 32;

    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // Each of the 128 threads loads 8 consecutive K-elements of one row of each tile.
    const int lr = int(tid) / 2;
    const int lc = (int(tid) % 2) * 8;

    for (int k0 = 0; k0 < p.K; k0 += GBK) {
        {
            int gm = m0 + lr;
            threadgroup float* dst = As + lr * (GBK + GPAD) + lc;
            if (gm < p.M) {
                device const TA* src = A + (long)gm * p.lda + k0 + lc;
                for (int e = 0; e < 8; e++) dst[e] = (k0 + lc + e < p.K) ? to_f(src[e]) : 0.0f;
            } else {
                for (int e = 0; e < 8; e++) dst[e] = 0.0f;
            }
            int gn = n0 + lr;
            threadgroup float* dstb = Bs + lr * (GBK + GPAD) + lc;
            if (gn < p.N) {
                device const TB* src = B + (long)gn * p.ldb + k0 + lc;
                for (int e = 0; e < 8; e++) dstb[e] = (k0 + lc + e < p.K) ? to_f(src[e]) : 0.0f;
            } else {
                for (int e = 0; e < 8; e++) dstb[e] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < GBK; kk += 8) {
            simdgroup_float8x8 a[4], b[4];
            for (int i = 0; i < 4; i++)
                simdgroup_load(a[i], As + (sr + i * 8) * (GBK + GPAD) + kk, GBK + GPAD);
            for (int j = 0; j < 4; j++)
                simdgroup_load(b[j], Bs + (sc + j * 8) * (GBK + GPAD) + kk, GBK + GPAD, ulong2(0, 0), true);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            simdgroup_store(acc[i][j], Cs + (sr + i * 8) * GBN + sc + j * 8, GBN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = int(tid); idx < GBM * GBN; idx += 128) {
        int r = idx / GBN, c = idx % GBN;
        int gm = m0 + r, gn = n0 + c;
        if (gm < p.M && gn < p.N) {
            float v = p.alpha * Cs[idx];
            if (p.has_bias) v += bias[gn];
            v = apply_act(v, p.act) * p.out_scale;
            if (p.has_residual) v += p.res_scale * to_f(R[(long)gm * p.ldr + gn]);
            C[(long)gm * p.ldc + gn] = from_f<TC>(v);
        }
    }
}

#define INSTANTIATE_GEMM(NAME, TA, TB, TC) \
    template [[host_name(NAME)]] kernel void gemm_nt<TA, TB, TC>( \
        device const TA*, device const TB*, device TC*, device const float*, device const TC*, \
        constant GemmParams&, uint2, uint, uint);

INSTANTIATE_GEMM("gemm_nt_f32_f32_f32", float, float, float)
INSTANTIATE_GEMM("gemm_nt_f16_f16_f16", half, half, half)
INSTANTIATE_GEMM("gemm_nt_f32_f16_f32", float, half, float)
INSTANTIATE_GEMM("gemm_nt_f16_f16_f32", half, half, float)
INSTANTIATE_GEMM("gemm_nt_f32_bf16_f32", float, ushort, float)
