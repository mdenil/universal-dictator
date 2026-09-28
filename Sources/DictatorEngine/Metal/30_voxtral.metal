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

kernel void rope_rotate_half(device float* x [[buffer(0)]], device const float* inv_freq [[buffer(1)]],
                             constant RopeParams& p [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
    int i = g.x, h = g.y, t = g.z;
    int half_d = p.d / 2;
    if (i >= half_d || h >= p.H || t >= p.T) return;
    float angle = float(p.pos0 + t) * inv_freq[i];
    float c = cos(angle), s = sin(angle);
    device float* row = x + (long)t * p.ld + h * p.d;
    float a = row[i], b = row[i + half_d];
    row[i] = a * c - b * s;
    row[i + half_d] = b * c + a * s;
}

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
