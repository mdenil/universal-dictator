// GEMM v2: C = out_scale * act(alpha * A·B^T + bias) + res_scale * R   (same contract as gemm_nt)
// - BK = 32 with 4-wide vector global loads staged through registers (next tile is fetched while the
//   current one is multiplied), fp32 simdgroup MMA accumulation.
// - 32x64 tiles (a 64x64 variant with 4x4 fragments per simdgroup was slower on the M3: PERF_LOG R1).
// Threadgroup: 128 threads = 4 simdgroups arranged SGR x SGC; each computes (FM*8) x (FN*8).

template <typename T> inline float4 gload4(device const T* p) { return float4(*(device const vec<T, 4>*)p); }
template <> inline float4 gload4<ushort>(device const ushort* p) {
    ushort4 b = *(device const ushort4*)p;
    return float4(bf16_to_float(b.x), bf16_to_float(b.y), bf16_to_float(b.z), bf16_to_float(b.w));
}

template <typename T, int ROWS, int BK>
struct TileLoader {
    // ROWS x BK elements, 128 threads, each loads PER float4s.
    static constant constexpr int PER = ROWS * BK / 4 / 128;
    float4 regs[PER];

    inline void load(device const T* src, int ld, int row0, int nrows, int k0, int K, bool vec_ok, uint tid) {
        for (int i = 0; i < PER; i++) {
            int idx = int(tid) + i * 128;          // float4 index within the tile
            int r = idx / (BK / 4), c = (idx % (BK / 4)) * 4;
            int gr = row0 + r, gk = k0 + c;
            float4 v = 0;
            if (gr < nrows) {
                device const T* p = src + (long)gr * ld + gk;
                if (vec_ok && gk + 4 <= K) {
                    v = gload4(p);
                } else {
                    for (int e = 0; e < 4; e++) v[e] = (gk + e < K) ? to_f(p[e]) : 0.0f;
                }
            }
            regs[i] = v;
        }
    }

    inline void store(threadgroup float* dst, uint tid) {
        for (int i = 0; i < PER; i++) {
            int idx = int(tid) + i * 128;
            int r = idx / (BK / 4), c = (idx % (BK / 4)) * 4;
            *(threadgroup float4*)(dst + r * (BK + 4) + c) = regs[i];
        }
    }
};

template <typename TA, typename TB, typename TC, int SGR, int SGC, int FM, int FN>
kernel void gemm2_nt(
    device const TA* A [[buffer(0)]],
    device const TB* B [[buffer(1)]],
    device TC* C [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device const TC* R [[buffer(4)]],
    constant GemmParams& p [[buffer(5)]],
    device const float* wscale [[buffer(6)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    constexpr int BM = SGR * FM * 8;
    constexpr int BN = SGC * FN * 8;
    constexpr int BK = 32;
    constexpr int LDS = BK + 4;
    constexpr int SMEM = (BM + BN) * LDS > BM * BN ? (BM + BN) * LDS : BM * BN;
    threadgroup float smem[SMEM];
    threadgroup float* As = smem;
    threadgroup float* Bs = smem + BM * LDS;

    const int m0 = int(tgp.y) * BM;
    const int n0 = int(tgp.x) * BN;
    const int sr = int(sg / SGC) * FM * 8;
    const int sc = int(sg % SGC) * FN * 8;
    const bool vec_ok = (p.K % 4) == 0;

    simdgroup_float8x8 acc[FM][FN];
    for (int i = 0; i < FM; i++)
        for (int j = 0; j < FN; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    TileLoader<TA, BM, BK> la;
    TileLoader<TB, BN, BK> lb;
    la.load(A, p.lda, m0, p.M, 0, p.K, vec_ok, tid);
    lb.load(B, p.ldb, n0, p.N, 0, p.K, vec_ok, tid);

    for (int k0 = 0; k0 < p.K; k0 += BK) {
        la.store(As, tid);
        lb.store(Bs, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < p.K) {
            la.load(A, p.lda, m0, p.M, k0 + BK, p.K, vec_ok, tid);
            lb.load(B, p.ldb, n0, p.N, k0 + BK, p.K, vec_ok, tid);
        }
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[FM], b[FN];
            for (int i = 0; i < FM; i++)
                simdgroup_load(a[i], As + (sr + i * 8) * LDS + kk, LDS);
            for (int j = 0; j < FN; j++)
                simdgroup_load(b[j], Bs + (sc + j * 8) * LDS + kk, LDS, ulong2(0, 0), true);
            for (int i = 0; i < FM; i++)
                for (int j = 0; j < FN; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* Cs = smem;
    for (int i = 0; i < FM; i++)
        for (int j = 0; j < FN; j++)
            simdgroup_store(acc[i][j], Cs + (sr + i * 8) * BN + sc + j * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = int(tid); idx < BM * BN; idx += 128) {
        int r = idx / BN, c = idx % BN;
        int gm = m0 + r, gn = n0 + c;
        if (gm < p.M && gn < p.N) {
            float v = p.alpha * Cs[idx];
            if (p.has_wscale) v *= wscale[gn];
            if (p.has_bias) v += bias[gn];
            v = apply_act(v, p.act) * p.out_scale;
            if (p.has_residual) v += p.res_scale * to_f(R[(long)gm * p.ldr + gn]);
            C[(long)gm * p.ldc + gn] = from_f<TC>(v);
        }
    }
}

#define INSTANTIATE_GEMM2(NAME, TA, TB, TC, SGR, SGC, FM, FN) \
    template [[host_name(NAME)]] kernel void gemm2_nt<TA, TB, TC, SGR, SGC, FM, FN>( \
        device const TA*, device const TB*, device TC*, device const float*, device const TC*, \
        constant GemmParams&, device const float*, uint2, uint, uint);

#define INSTANTIATE_GEMM2_BOTH(T, TA, TB, TC) \
    INSTANTIATE_GEMM2("gemm2_32x64_" T, TA, TB, TC, 1, 4, 4, 2)

INSTANTIATE_GEMM2_BOTH("f32_f32_f32", float, float, float)
INSTANTIATE_GEMM2_BOTH("f16_f16_f16", half, half, half)
INSTANTIATE_GEMM2_BOTH("f32_f16_f32", float, half, float)
INSTANTIATE_GEMM2_BOTH("f16_f16_f32", half, half, float)
INSTANTIATE_GEMM2_BOTH("f32_bf16_f32", float, ushort, float)
INSTANTIATE_GEMM2_BOTH("f32_i8_f32", float, char, float)
