// Kernels for FastConformer encoders (Nemotron, Parakeet). Activations are time-major [T, D].

// ---------------------------------------------------------------- LayerNorm / RMSNorm
// One threadgroup per row. out = act(norm(x) * gamma + beta).

struct NormParams {
    int D;
    float eps;
    int act;
    int has_beta;
};

inline float tg_sum(float v, threadgroup float* scratch, uint tid, uint nthreads) {
    v = simd_sum(v);
    uint lane = tid % 32, warp = tid / 32;
    if (lane == 0) scratch[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    uint nw = (nthreads + 31) / 32;
    for (uint w = 0; w < nw; w++) total += scratch[w];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

template <typename T>
kernel void layer_norm(device const T* x [[buffer(0)]],
                       device const float* gamma [[buffer(1)]],
                       device const float* beta [[buffer(2)]],
                       device T* out [[buffer(3)]],
                       constant NormParams& p [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint nt [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const T* xr = x + (long)row * p.D;
    float s = 0;
    for (int i = tid; i < p.D; i += nt) s += to_f(xr[i]);
    float mean = tg_sum(s, scratch, tid, nt) / p.D;
    float v = 0;
    for (int i = tid; i < p.D; i += nt) { float d = to_f(xr[i]) - mean; v += d * d; }
    float var = tg_sum(v, scratch, tid, nt) / p.D;
    float inv = rsqrt(var + p.eps);
    device T* o = out + (long)row * p.D;
    for (int i = tid; i < p.D; i += nt) {
        float y = (to_f(xr[i]) - mean) * inv * gamma[i];
        if (p.has_beta) y += beta[i];
        o[i] = from_f<T>(apply_act(y, p.act));
    }
}

template <typename T>
kernel void rms_norm(device const T* x [[buffer(0)]],
                     device const float* gamma [[buffer(1)]],
                     device const float* beta [[buffer(2)]],
                     device T* out [[buffer(3)]],
                     constant NormParams& p [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const T* xr = x + (long)row * p.D;
    float s = 0;
    for (int i = tid; i < p.D; i += nt) { float v = to_f(xr[i]); s += v * v; }
    float inv = rsqrt(tg_sum(s, scratch, tid, nt) / p.D + p.eps);
    device T* o = out + (long)row * p.D;
    for (int i = tid; i < p.D; i += nt) {
        o[i] = from_f<T>(apply_act(to_f(xr[i]) * inv * gamma[i], p.act));
    }
}

template [[host_name("layer_norm_f32")]] kernel void layer_norm<float>(device const float*, device const float*, device const float*, device float*, constant NormParams&, uint, uint, uint);
template [[host_name("layer_norm_f16")]] kernel void layer_norm<half>(device const half*, device const float*, device const float*, device half*, constant NormParams&, uint, uint, uint);
template [[host_name("rms_norm_f32")]] kernel void rms_norm<float>(device const float*, device const float*, device const float*, device float*, constant NormParams&, uint, uint, uint);
template [[host_name("rms_norm_f16")]] kernel void rms_norm<half>(device const half*, device const float*, device const float*, device half*, constant NormParams&, uint, uint, uint);

// ---------------------------------------------------------------- GLU over channels
// in [T, 2D] -> out[t, d] = in[t, d] * sigmoid(in[t, D + d])
template <typename T>
kernel void glu(device const T* x [[buffer(0)]], device T* out [[buffer(1)]],
                constant int& D [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
    int d = g.x, t = g.y;
    if (d >= D) return;
    float a = to_f(x[(long)t * 2 * D + d]);
    float b = to_f(x[(long)t * 2 * D + D + d]);
    out[(long)t * D + d] = from_f<T>(a / (1.0f + exp(-b)));
}
template [[host_name("glu_f32")]] kernel void glu<float>(device const float*, device float*, constant int&, uint2);
template [[host_name("glu_f16")]] kernel void glu<half>(device const half*, device half*, constant int&, uint2);

// ---------------------------------------------------------------- depthwise conv1d over time
// in [Tin, D] with Tin = Tout + K - 1 (caller supplies left context / padding rows).
// out[t, d] = bias[d] + sum_k w[d, k] * in[t + k, d]
struct DWConvParams { int Tout; int D; int K; int has_bias; int act; };

template <typename T>
kernel void dwconv1d(device const T* x [[buffer(0)]], device const float* w [[buffer(1)]],
                     device const float* bias [[buffer(2)]], device T* out [[buffer(3)]],
                     constant DWConvParams& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    int d = g.x, t = g.y;
    if (d >= p.D || t >= p.Tout) return;
    float acc = p.has_bias ? bias[d] : 0.0f;
    for (int k = 0; k < p.K; k++) acc += w[d * p.K + k] * to_f(x[(long)(t + k) * p.D + d]);
    out[(long)t * p.D + d] = from_f<T>(apply_act(acc, p.act));
}
template [[host_name("dwconv1d_f32")]] kernel void dwconv1d<float>(device const float*, device const float*, device const float*, device float*, constant DWConvParams&, uint2);
template [[host_name("dwconv1d_f16")]] kernel void dwconv1d<half>(device const half*, device const float*, device const float*, device half*, constant DWConvParams&, uint2);

// ---------------------------------------------------------------- conv2d, channel-last, 3x3-ish
// in [Hin, Win, Cin], w [Cout, Cin/groups, KH, KW] (torch layout), out [Hout, Wout, Cout].
// Input coordinates outside [0,Hin)x[0,Win) read as zero (this implements the padding).
// Rows with h >= valid_h are written as zero (NeMo's MaskedConvSequential length mask).
struct Conv2DParams {
    int Hin, Win, Cin, Hout, Wout, Cout;
    int KH, KW, stride, pad_top, pad_left, groups;
    int act;
    int valid_h;
};

kernel void conv2d_cl(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                      device const float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                      constant Conv2DParams& p [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
    int co = g.x, wo = g.y, ho = g.z;
    if (co >= p.Cout || wo >= p.Wout || ho >= p.Hout) return;
    long oidx = ((long)ho * p.Wout + wo) * p.Cout + co;
    if (ho >= p.valid_h) { out[oidx] = 0.0f; return; }
    int cin_per_g = p.Cin / p.groups;
    int cout_per_g = p.Cout / p.groups;
    int g0 = (co / cout_per_g) * cin_per_g;
    float acc = bias[co];
    for (int kh = 0; kh < p.KH; kh++) {
        int hi = ho * p.stride + kh - p.pad_top;
        if (hi < 0 || hi >= p.Hin) continue;
        for (int kw = 0; kw < p.KW; kw++) {
            int wi = wo * p.stride + kw - p.pad_left;
            if (wi < 0 || wi >= p.Win) continue;
            device const float* xp = x + ((long)hi * p.Win + wi) * p.Cin + g0;
            device const float* wp = w + ((long)co * cin_per_g * p.KH + kh) * p.KW + kw;
            for (int ci = 0; ci < cin_per_g; ci++) acc += wp[(long)ci * p.KH * p.KW] * xp[ci];
        }
    }
    out[oidx] = apply_act(acc, p.act);
}

// ---------------------------------------------------------------- relative-position attention
// Transformer-XL style attention used by FastConformer:
//   score[i,j] = ((q_i + u) · k_j + (q_i + v) · P[rel(i,j)]) / sqrt(dk)
//   rel(i,j) = (Tk - Tq) + i - j    (query i sits at absolute key position Tk - Tq + i)
// P is a per-layer table of projected sinusoidal embeddings indexed by rel + rel_offset.
// Keys outside [kstart_i, kend_i) are masked (probability exactly 0, as in NeMo).
// One threadgroup (dk threads) per (query, head).
struct RelAttnParams {
    int Tq, Tk, H, dk;
    int ldq, ldk, ldv, ldo;   // row strides in elements
    int rel_offset;
    int n_rel;
    float scale;
};

constant constexpr int RELATTN_MAX_KEYS = 4096;

template <typename T>
kernel void relpos_attention(device const T* Q [[buffer(0)]], device const T* K [[buffer(1)]],
                             device const T* V [[buffer(2)]], device const T* P [[buffer(3)]],
                             device const float* bias_u [[buffer(4)]], device const float* bias_v [[buffer(5)]],
                             device const int2* ranges [[buffer(6)]], device T* out [[buffer(7)]],
                             constant RelAttnParams& p [[buffer(8)]],
                             uint2 tgp [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint2 ntg [[threads_per_threadgroup]]) {
    const uint nt = ntg.x;
    threadgroup float qu[256];
    threadgroup float qv[256];
    threadgroup float probs[RELATTN_MAX_KEYS];
    threadgroup float scratch[32];

    const int i = tgp.x, h = tgp.y;
    const int hoff = h * p.dk;
    for (int d = tid; d < p.dk; d += nt) {
        float q = to_f(Q[(long)i * p.ldq + hoff + d]);
        qu[d] = q + bias_u[hoff + d];
        qv[d] = q + bias_v[hoff + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int2 r = ranges[i];
    const int ks = max(r.x, 0), ke = min(r.y, p.Tk);
    const int shift = p.Tk - p.Tq + i;

    // Scores: each thread owns keys j = ks + tid, ks + tid + nt, ...
    float local_max = -INFINITY;
    for (int j = ks + tid; j < ke; j += nt) {
        device const T* kr = K + (long)j * p.ldk + hoff;
        int ridx = shift - j + p.rel_offset;
        device const T* pr = P + (long)ridx * (p.H * p.dk) + hoff;
        float s = 0;
        for (int d = 0; d < p.dk; d++) s += qu[d] * to_f(kr[d]) + qv[d] * to_f(pr[d]);
        s *= p.scale;
        probs[j - ks] = s;
        local_max = max(local_max, s);
    }
    // threadgroup max
    float m = simd_max(local_max);
    if (tid % 32 == 0) scratch[tid / 32] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = -INFINITY;
    for (uint w = 0; w < (nt + 31) / 32; w++) m = max(m, scratch[w]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0;
    for (int j = ks + tid; j < ke; j += nt) {
        float e = exp(probs[j - ks] - m);
        probs[j - ks] = e;
        local_sum += e;
    }
    float denom = tg_sum(local_sum, scratch, tid, nt);
    float inv = denom > 0 ? 1.0f / denom : 0.0f;

    for (int d = tid; d < p.dk; d += nt) {
        float acc = 0;
        for (int j = ks; j < ke; j++) acc += probs[j - ks] * to_f(V[(long)j * p.ldv + hoff + d]);
        out[(long)i * p.ldo + hoff + d] = from_f<T>(acc * inv);
    }
}
template [[host_name("relpos_attention_f32")]] kernel void relpos_attention<float>(device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device const int2*, device float*, constant RelAttnParams&, uint2, uint, uint2);
template [[host_name("relpos_attention_f16")]] kernel void relpos_attention<half>(device const half*, device const half*, device const half*, device const half*, device const float*, device const float*, device const int2*, device half*, constant RelAttnParams&, uint2, uint, uint2);

// ---------------------------------------------------------------- misc elementwise
template <typename T>
kernel void add_scaled(device const T* a [[buffer(0)]], device const T* b [[buffer(1)]], device T* out [[buffer(2)]],
                       constant float& sb [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    out[i] = from_f<T>(to_f(a[i]) + sb * to_f(b[i]));
}
template [[host_name("add_scaled_f32")]] kernel void add_scaled<float>(device const float*, device const float*, device float*, constant float&, uint);
template [[host_name("add_scaled_f16")]] kernel void add_scaled<half>(device const half*, device const half*, device half*, constant float&, uint);

kernel void scale_f32(device const float* a [[buffer(0)]], device float* out [[buffer(1)]],
                      constant float& s [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    out[i] = a[i] * s;
}

// Copy `n` elements (in 4-byte words) between buffers; used for cache shifts and gathers.
kernel void copy_words(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    dst[i] = src[i];
}

kernel void cast_f32_f16(device const float* a [[buffer(0)]], device half* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    out[i] = half(a[i]);
}
kernel void cast_f16_f32(device const half* a [[buffer(0)]], device float* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    out[i] = float(a[i]);
}
