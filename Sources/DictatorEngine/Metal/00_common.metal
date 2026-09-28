#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// Activation codes shared with Swift (Kernels.swift: Activation).
constant constexpr int ACT_NONE = 0;
constant constexpr int ACT_RELU = 1;
constant constexpr int ACT_SILU = 2;
constant constexpr int ACT_GELU_TANH = 3;
constant constexpr int ACT_GELU_ERF = 4;

// erf-based GELU is what torch.nn.GELU() computes by default. Metal has no erf, so we use
// the Abramowitz-Stegun 7.1.26 style approximation with max abs error ~1.5e-7.
inline float erf_approx(float x) {
    float s = sign(x);
    float a = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-a * a);
    return s * y;
}

inline float apply_act(float x, int act) {
    switch (act) {
        case ACT_RELU: return max(x, 0.0f);
        case ACT_SILU: return x / (1.0f + exp(-x));
        case ACT_GELU_TANH: {
            // Clamp the tanh argument: tanh is NaN-prone at extreme inputs in reduced precision.
            float u = clamp(0.7978845608f * (x + 0.044715f * x * x * x), -15.0f, 15.0f);
            return 0.5f * x * (1.0f + tanh(u));
        }
        case ACT_GELU_ERF: return 0.5f * x * (1.0f + erf_approx(x * 0.7071067811865476f));
        default: return x;
    }
}

// bfloat16 stored as ushort; convert to float by shifting into the high half.
inline float bf16_to_float(ushort b) { return as_type<float>(uint(b) << 16); }

template <typename T> inline float to_f(T v) { return float(v); }
template <> inline float to_f<ushort>(ushort v) { return bf16_to_float(v); }

template <typename T> inline T from_f(float v) { return T(v); }
