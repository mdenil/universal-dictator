// Kernels for Voxtral Mini 4B Realtime (causal conv stem, RoPE transformer encoder, Mistral decoder).

// ---------------------------------------------------------------- causal conv1d via im2col
// x [Tin, C] time-major; out row t, column c*K + k = x[t*stride + k - pad, c] (zero outside [0, Tin)).
// With the torch weight [Cout, C, K] viewed as [Cout, C*K], conv = GEMM(im2col, W).
struct Im2ColParams { int Tin, C, K, stride, pad, Tout; };

kernel void im2col1d(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                     constant Im2ColParams& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
    int col = g.x, t = g.y;
    if (col >= p.C * p.K || t >= p.Tout) return;
    int c = col / p.K, k = col - c * p.K;
    int ti = t * p.stride + k - p.pad;
    out[(long)t * p.C * p.K + col] = (ti >= 0 && ti < p.Tin) ? x[(long)ti * p.C + c] : 0.0f;
}

// ---------------------------------------------------------------- RoPE (rotate-half / NeoX)
// In place on x [T, H*d]: for pair (i, i + d/2): angle = pos[t] * inv_freq[i]
//   x_i' = x_i cos - x_{i+d/2} sin;  x_{i+d/2}' = x_{i+d/2} cos + x_i sin
struct RopeParams { int T, H, d, ld, pos0; };

template <typename T>
kernel void rope_rotate_half(device T* x [[buffer(0)]], device const float* inv_freq [[buffer(1)]],
                             constant RopeParams& p [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
    int i = g.x, h = g.y, t = g.z;
    int half_d = p.d / 2;
    if (i >= half_d || h >= p.H || t >= p.T) return;
    float angle = float(p.pos0 + t) * inv_freq[i];
    float c = cos(angle), s = sin(angle);
    device T* row = x + (long)t * p.ld + h * p.d;
    float a = float(row[i]), b = float(row[i + half_d]);
    row[i] = T(a * c - b * s);
    row[i + half_d] = T(b * c + a * s);
}
template [[host_name("rope_rotate_half")]] kernel void rope_rotate_half<float>(device float*, device const float*, constant RopeParams&, uint3);
template [[host_name("rope_rotate_half_f16")]] kernel void rope_rotate_half<half>(device half*, device const float*, constant RopeParams&, uint3);

// ---------------------------------------------------------------- GQA attention, online softmax
// q [Tq, Hq*d] (row stride ldq), K/V [Tk, Hkv*d] (row stride ldk); query head h uses kv head h / (Hq/Hkv).
// Query i attends keys [ranges[i].x, ranges[i].y). One threadgroup (d threads) per (query, head).
struct AttnParams { int Tq, Tk, Hq, Hkv, d, ldq, ldk, ldo; float scale; };

constant constexpr int ATTN_BLOCK = 512;

template <typename TK>
kernel void attention_gqa(device const float* Q [[buffer(0)]], device const TK* K [[buffer(1)]],
                          device const TK* V [[buffer(2)]], device const int2* ranges [[buffer(3)]],
                          device float* out [[buffer(4)]], constant AttnParams& p [[buffer(5)]],
                          uint2 tgp [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 ntg [[threads_per_threadgroup]]) {
    threadgroup float qs[128];
    threadgroup float sc[ATTN_BLOCK];
    threadgroup float scratch[32];
    const uint nt = ntg.x;
    const int i = tgp.x, h = tgp.y;
    const int kvh = h / (p.Hq / p.Hkv);
    for (int e = tid; e < p.d; e += nt) qs[e] = Q[(long)i * p.ldq + h * p.d + e];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int2 r = ranges[i];
    const int ks = max(r.x, 0), ke = min(r.y, p.Tk);
    float m = -INFINITY, l = 0.0f, acc = 0.0f;   // acc: output element `tid` (tid < d)
    for (int b0 = ks; b0 < ke; b0 += ATTN_BLOCK) {
        int bn = min(ATTN_BLOCK, ke - b0);
        float bmax = -INFINITY;
        for (int j = tid; j < bn; j += nt) {
            device const TK* kr = K + (long)(b0 + j) * p.ldk + kvh * p.d;
            float s = 0;
            for (int e = 0; e < p.d; e++) s += qs[e] * float(kr[e]);
            s *= p.scale;
            sc[j] = s;
            bmax = max(bmax, s);
        }
        bmax = simd_max(bmax);
        if (tid % 32 == 0) scratch[tid / 32] = bmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bmax = -INFINITY;
        for (uint w = 0; w < (nt + 31) / 32; w++) bmax = max(bmax, scratch[w]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float nm = max(m, bmax);
        float corr = exp(m - nm);
        float lsum = 0;
        for (int j = tid; j < bn; j += nt) { float e = exp(sc[j] - nm); sc[j] = e; lsum += e; }
        lsum = simd_sum(lsum);
        if (tid % 32 == 0) scratch[tid / 32] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float bsum = 0;
        for (uint w = 0; w < (nt + 31) / 32; w++) bsum += scratch[w];
        l = l * corr + bsum;
        if (int(tid) < p.d) {
            float a = acc * corr;
            device const TK* vc = V + (long)b0 * p.ldk + kvh * p.d + tid;
            for (int j = 0; j < bn; j++) a += sc[j] * float(vc[(long)j * p.ldk]);
            acc = a;
        }
        m = nm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (int(tid) < p.d) out[(long)i * p.ldo + h * p.d + tid] = l > 0 ? acc / l : 0.0f;
}
template [[host_name("attention_gqa_f32")]] kernel void attention_gqa<float>(device const float*, device const float*, device const float*, device const int2*, device float*, constant AttnParams&, uint2, uint, uint2);
template [[host_name("attention_gqa_f16")]] kernel void attention_gqa<half>(device const float*, device const half*, device const half*, device const int2*, device float*, constant AttnParams&, uint2, uint, uint2);

// ---------------------------------------------------------------- SwiGLU: out = silu(g) * u
kernel void swiglu(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                   device float* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    float x = g[i];
    out[i] = x / (1.0f + exp(-x)) * u[i];
}

// ---------------------------------------------------------------- token embedding (+ audio embedding)
// out[t, :] = table[ids[t], :] + (add ? audio[t, :] : 0); table is bf16 (ushort).
kernel void embed_add_bf16(device const ushort* table [[buffer(0)]], device const int* ids [[buffer(1)]],
                           device const float* audio [[buffer(2)]], device float* out [[buffer(3)]],
                           constant int& D [[buffer(4)]], constant int& add [[buffer(5)]],
                           uint2 g [[thread_position_in_grid]]) {
    int c = g.x, t = g.y;
    if (c >= D) return;
    float v = bf16_to_float(table[(long)ids[t] * D + c]);
    if (add) v += audio[(long)t * D + c];
    out[(long)t * D + c] = v;
}

// ---------------------------------------------------------------- argmax over a row
// One threadgroup per row; writes (index, value) as int + float.
kernel void argmax_rows(device const float* x [[buffer(0)]], device int* idx [[buffer(1)]],
                        device float* val [[buffer(2)]], constant int& N [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup int si[32];
    device const float* r = x + (long)row * N;
    float best = -INFINITY; int bi = 0;
    for (int j = tid; j < N; j += nt) { float v = r[j]; if (v > best) { best = v; bi = j; } }
    // simdgroup reduction: prefer the lower index on ties (torch.argmax returns the first max).
    for (uint off = 16; off > 0; off /= 2) {
        float ov = simd_shuffle_down(best, off);
        int oi = simd_shuffle_down(bi, off);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (tid % 32 == 0) { sv[tid / 32] = best; si[tid / 32] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b = sv[0]; int bj = si[0];
        for (uint w = 1; w < (nt + 31) / 32; w++) if (sv[w] > b || (sv[w] == b && si[w] < bj)) { b = sv[w]; bj = si[w]; }
        idx[row] = bj; val[row] = b;
    }
}

// ---------------------------------------------------------------- multi-query attention (Tq <= 8)
// One threadgroup (128 threads) per head handles all Tq queries, so each K/V row is read once per
// head instead of once per (query, head). Per-query key ranges; online softmax over key blocks.
constant constexpr int MQ_MAX_Q = 8;
constant constexpr int MQ_BLOCK = 256;

template <typename TK>
kernel void attention_mq(device const float* Q [[buffer(0)]], device const TK* K [[buffer(1)]],
                         device const TK* V [[buffer(2)]], device const int2* ranges [[buffer(3)]],
                         device float* out [[buffer(4)]], constant AttnParams& p [[buffer(5)]],
                         uint h [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[MQ_MAX_Q * 128];
    threadgroup float sc[MQ_MAX_Q * MQ_BLOCK];
    threadgroup float mrow[MQ_MAX_Q], lrow[MQ_MAX_Q], corr[MQ_MAX_Q];
    const int nt = 128;
    const int Tq = p.Tq, d = p.d;
    const int kvh = int(h) / (p.Hq / p.Hkv);
    for (int i = tid; i < Tq * d; i += nt) {
        int q = i / d, e = i - q * d;
        qs[q * d + e] = Q[(long)q * p.ldq + h * d + e];
    }
    if (int(tid) < Tq) { mrow[tid] = -INFINITY; lrow[tid] = 0; }
    int lo = p.Tk, hi = 0;
    for (int q = 0; q < Tq; q++) { lo = min(lo, max(ranges[q].x, 0)); hi = max(hi, min(ranges[q].y, p.Tk)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Output ownership: thread owns dim e = tid % d for queries q = tid / d, tid / d + nt/d, ...
    const int e = int(tid) % d, qstep = nt / d, q0 = int(tid) / d;
    float acc[MQ_MAX_Q];
    for (int q = 0; q < MQ_MAX_Q; q++) acc[q] = 0;

    for (int b0 = lo; b0 < hi; b0 += MQ_BLOCK) {
        int bn = min(MQ_BLOCK, hi - b0);
        // Scores: thread per key, all queries at once (K row read once).
        for (int j = tid; j < bn; j += nt) {
            device const TK* kr = K + (long)(b0 + j) * p.ldk + kvh * d;
            float s[MQ_MAX_Q];
            for (int q = 0; q < Tq; q++) s[q] = 0;
            for (int x = 0; x < d; x++) {
                float kv = float(kr[x]);
                for (int q = 0; q < Tq; q++) s[q] += qs[q * d + x] * kv;
            }
            for (int q = 0; q < Tq; q++) {
                int kj = b0 + j;
                bool ok = kj >= ranges[q].x && kj < ranges[q].y;
                sc[q * MQ_BLOCK + j] = ok ? s[q] * p.scale : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Per-query block max / rescale (one simdgroup per query row, Tq <= 8 rows, 4 simdgroups).
        for (int q = int(tid) / 32; q < Tq; q += nt / 32) {
            float bm = -INFINITY;
            for (int j = int(tid) % 32; j < bn; j += 32) bm = max(bm, sc[q * MQ_BLOCK + j]);
            bm = simd_max(bm);
            float nm = max(mrow[q], bm);
            float bs = 0;
            for (int j = int(tid) % 32; j < bn; j += 32) {
                float v = sc[q * MQ_BLOCK + j];
                float ex = v == -INFINITY ? 0.0f : exp(v - nm);
                sc[q * MQ_BLOCK + j] = ex;
                bs += ex;
            }
            bs = simd_sum(bs);
            if (int(tid) % 32 == 0) {
                float c = mrow[q] == -INFINITY ? 0.0f : exp(mrow[q] - nm);
                corr[q] = c;
                lrow[q] = lrow[q] * c + bs;
                mrow[q] = nm;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int q = q0, qi = 0; q < Tq; q += qstep, qi++) acc[qi] *= corr[q];
        for (int j = 0; j < bn; j++) {
            float v = float(V[(long)(b0 + j) * p.ldk + kvh * d + e]);
            for (int q = q0, qi = 0; q < Tq; q += qstep, qi++) acc[qi] += sc[q * MQ_BLOCK + j] * v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int q = q0, qi = 0; q < Tq; q += qstep, qi++) {
        float l = lrow[q];
        out[(long)q * p.ldo + h * d + e] = l > 0 ? acc[qi] / l : 0.0f;
    }
}
template [[host_name("attention_mq_f32")]] kernel void attention_mq<float>(device const float*, device const float*, device const float*, device const int2*, device float*, constant AttnParams&, uint, uint);
template [[host_name("attention_mq_f16")]] kernel void attention_mq<half>(device const float*, device const half*, device const half*, device const int2*, device float*, constant AttnParams&, uint, uint);

// ---------------------------------------------------------------- split-K multi-query attention
// Same as attention_mq, but each head's key range is split into p.splits partitions handled by
// separate threadgroups (more parallelism for 32-head / few-query steps). Each writes partial
// (max, sum, unnormalized output) to `part`; attention_combine merges them.
struct SplitParams { int splits; };

template <typename TK>
kernel void attention_mq_split(device const float* Q [[buffer(0)]], device const TK* K [[buffer(1)]],
                               device const TK* V [[buffer(2)]], device const int2* ranges [[buffer(3)]],
                               device float* part [[buffer(4)]], constant AttnParams& p [[buffer(5)]],
                               constant SplitParams& sp [[buffer(6)]],
                               uint2 tg2 [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[MQ_MAX_Q * 128];
    threadgroup float sc[MQ_MAX_Q * MQ_BLOCK];
    threadgroup float mrow[MQ_MAX_Q], lrow[MQ_MAX_Q], corr[MQ_MAX_Q];
    const int nt = 128;
    const int h = tg2.x, split = tg2.y;
    const int Tq = p.Tq, d = p.d;
    const int kvh = h / (p.Hq / p.Hkv);
    for (int i = tid; i < Tq * d; i += nt) {
        int q = i / d, e = i - q * d;
        qs[q * d + e] = Q[(long)q * p.ldq + h * d + e];
    }
    if (int(tid) < Tq) { mrow[tid] = -INFINITY; lrow[tid] = 0; }
    int lo = p.Tk, hi = 0;
    for (int q = 0; q < Tq; q++) { lo = min(lo, max(ranges[q].x, 0)); hi = max(hi, min(ranges[q].y, p.Tk)); }
    const int per = (hi - lo + sp.splits - 1) / sp.splits;
    const int slo = lo + split * per, shi = min(hi, slo + per);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int e = int(tid) % d, qstep = nt / d, q0 = int(tid) / d;
    float acc[MQ_MAX_Q];
    for (int q = 0; q < MQ_MAX_Q; q++) acc[q] = 0;
    for (int b0 = slo; b0 < shi; b0 += MQ_BLOCK) {
        int bn = min(MQ_BLOCK, shi - b0);
        for (int j = tid; j < bn; j += nt) {
            device const TK* kr = K + (long)(b0 + j) * p.ldk + kvh * d;
            float s[MQ_MAX_Q];
            for (int q = 0; q < Tq; q++) s[q] = 0;
            for (int x = 0; x < d; x += 4) {
                float4 kv = float4(kr[x], kr[x + 1], kr[x + 2], kr[x + 3]);
                for (int q = 0; q < Tq; q++) s[q] += dot(float4(qs[q * d + x], qs[q * d + x + 1], qs[q * d + x + 2], qs[q * d + x + 3]), kv);
            }
            for (int q = 0; q < Tq; q++) {
                int kj = b0 + j;
                bool ok = kj >= ranges[q].x && kj < ranges[q].y;
                sc[q * MQ_BLOCK + j] = ok ? s[q] * p.scale : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int q = int(tid) / 32; q < Tq; q += nt / 32) {
            float bm = -INFINITY;
            for (int j = int(tid) % 32; j < bn; j += 32) bm = max(bm, sc[q * MQ_BLOCK + j]);
            bm = simd_max(bm);
            float nm = max(mrow[q], bm);
            float bs = 0;
            for (int j = int(tid) % 32; j < bn; j += 32) {
                float v = sc[q * MQ_BLOCK + j];
                float ex = (v == -INFINITY) ? 0.0f : exp(v - nm);
                sc[q * MQ_BLOCK + j] = ex;
                bs += ex;
            }
            bs = simd_sum(bs);
            if (int(tid) % 32 == 0) {
                float c = mrow[q] == -INFINITY ? 0.0f : exp(mrow[q] - nm);
                corr[q] = c; lrow[q] = lrow[q] * c + bs; mrow[q] = nm;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int q = q0, qi = 0; q < Tq; q += qstep, qi++) acc[qi] *= corr[q];
        for (int j = 0; j < bn; j++) {
            float v = float(V[(long)(b0 + j) * p.ldk + kvh * d + e]);
            for (int q = q0, qi = 0; q < Tq; q += qstep, qi++) acc[qi] += sc[q * MQ_BLOCK + j] * v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Partial layout per (head, split): [Tq] max, [Tq] sum, [Tq, d] unnormalized output.
    device float* pp = part + (long)(h * sp.splits + split) * (Tq * (d + 2));
    if (int(tid) < Tq) { pp[tid] = mrow[tid]; pp[Tq + tid] = lrow[tid]; }
    for (int q = q0, qi = 0; q < Tq; q += qstep, qi++) pp[2 * Tq + q * d + e] = acc[qi];
}
template [[host_name("attention_mq_split_f32")]] kernel void attention_mq_split<float>(device const float*, device const float*, device const float*, device const int2*, device float*, constant AttnParams&, constant SplitParams&, uint2, uint);
template [[host_name("attention_mq_split_f16")]] kernel void attention_mq_split<half>(device const float*, device const half*, device const half*, device const int2*, device float*, constant AttnParams&, constant SplitParams&, uint2, uint);

kernel void attention_combine(device const float* part [[buffer(0)]], device float* out [[buffer(1)]],
                              constant AttnParams& p [[buffer(2)]], constant SplitParams& sp [[buffer(3)]],
                              uint3 g [[thread_position_in_grid]]) {
    int e = g.x, q = g.y, h = g.z;
    if (e >= p.d || q >= p.Tq || h >= p.Hq) return;
    const int Tq = p.Tq, d = p.d, stride = Tq * (d + 2);
    float M = -INFINITY;
    for (int s = 0; s < sp.splits; s++) M = max(M, part[(long)(h * sp.splits + s) * stride + q]);
    float num = 0, den = 0;
    for (int s = 0; s < sp.splits; s++) {
        device const float* pp = part + (long)(h * sp.splits + s) * stride;
        float m = pp[q];
        if (m == -INFINITY) continue;
        float w = exp(m - M);
        den += pp[Tq + q] * w;
        num += pp[2 * Tq + q * d + e] * w;
    }
    out[(long)q * p.ldo + h * d + e] = den > 0 ? num / den : 0.0f;
}
