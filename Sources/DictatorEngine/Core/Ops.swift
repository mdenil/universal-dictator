import Foundation
import Metal

struct NormParams { var D: Int32; var eps: Float; var act: Int32; var hasBeta: Int32 }
struct DWConvParams { var Tout: Int32; var D: Int32; var K: Int32; var hasBias: Int32; var act: Int32 }
struct Conv2DParams {
    var Hin: Int32, Win: Int32, Cin: Int32, Hout: Int32, Wout: Int32, Cout: Int32
    var KH: Int32, KW: Int32, stride: Int32, padTop: Int32, padLeft: Int32, groups: Int32
    var act: Int32
    var validH: Int32
}
struct RelAttnParams {
    var Tq: Int32, Tk: Int32, H: Int32, dk: Int32
    var ldq: Int32, ldk: Int32, ldv: Int32, ldo: Int32
    var relOffset: Int32
    var nRel: Int32
    var scale: Float
}

private func tag(_ d: DType) -> String {
    switch d {
    case .f32: return "f32"
    case .f16: return "f16"
    default: fatalError("unsupported activation dtype \(d)")
    }
}

extension Step {
    public func layerNorm(_ x: Tensor, gamma: Tensor, beta: Tensor?, out: Tensor, eps: Float = 1e-5, act: Activation = .none) {
        let D = x.shape.last!, rows = x.count / D
        let p = try! ctx.pipeline("layer_norm_\(tag(x.dtype))")
        bind(x, 0); bind(gamma, 1); bind(beta ?? gamma, 2); bind(out, 3)
        setBytes(NormParams(D: Int32(D), eps: eps, act: act.rawValue, hasBeta: beta == nil ? 0 : 1), 4)
        dispatch(p, threadgroups: MTLSize(width: rows, height: 1, depth: 1), threads: MTLSize(width: 256, height: 1, depth: 1))
    }

    public func rmsNorm(_ x: Tensor, gamma: Tensor, out: Tensor, eps: Float = 1e-5, act: Activation = .none) {
        let D = x.shape.last!, rows = x.count / D
        let p = try! ctx.pipeline("rms_norm_\(tag(x.dtype))")
        bind(x, 0); bind(gamma, 1); bind(gamma, 2); bind(out, 3)
        setBytes(NormParams(D: Int32(D), eps: eps, act: act.rawValue, hasBeta: 0), 4)
        dispatch(p, threadgroups: MTLSize(width: rows, height: 1, depth: 1), threads: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// x [T, 2D] -> out [T, D]
    public func glu(_ x: Tensor, out: Tensor) {
        let D = x.shape.last! / 2, T = x.count / (2 * D)
        let p = try! ctx.pipeline("glu_\(tag(x.dtype))")
        bind(x, 0); bind(out, 1); setBytes(Int32(D), 2)
        enc.setComputePipelineState(p)
        enc.dispatchThreads(MTLSize(width: D, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// x [Tout + K - 1, D], w [D, K] -> out [Tout, D]
    public func dwconv1d(_ x: Tensor, w: Tensor, bias: Tensor?, out: Tensor, act: Activation = .none) {
        let D = w.shape[0], K = w.shape[1], Tout = out.count / D
        precondition(x.count / D == Tout + K - 1, "dwconv1d: in \(x.shape) out \(out.shape) K \(K)")
        let p = try! ctx.pipeline("dwconv1d_\(tag(x.dtype))")
        bind(x, 0); bind(w, 1); bind(bias ?? w, 2); bind(out, 3)
        setBytes(DWConvParams(Tout: Int32(Tout), D: Int32(D), K: Int32(K), hasBias: bias == nil ? 0 : 1, act: act.rawValue), 4)
        enc.setComputePipelineState(p)
        enc.dispatchThreads(MTLSize(width: D, height: Tout, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Channel-last conv2d. x [Hin, Win, Cin], w [Cout, Cin/groups, KH, KW], out [Hout, Wout, Cout].
    public func conv2d(_ x: Tensor, w: Tensor, bias: Tensor, out: Tensor, stride: Int, padTop: Int, padLeft: Int,
                       groups: Int, act: Activation, validH: Int) {
        let (Hin, Win, Cin) = (x.shape[0], x.shape[1], x.shape[2])
        let (Hout, Wout, Cout) = (out.shape[0], out.shape[1], out.shape[2])
        let p = try! ctx.pipeline("conv2d_cl")
        bind(x, 0); bind(w, 1); bind(bias, 2); bind(out, 3)
        setBytes(Conv2DParams(Hin: Int32(Hin), Win: Int32(Win), Cin: Int32(Cin), Hout: Int32(Hout), Wout: Int32(Wout), Cout: Int32(Cout),
                              KH: Int32(w.shape[2]), KW: Int32(w.shape[3]), stride: Int32(stride), padTop: Int32(padTop),
                              padLeft: Int32(padLeft), groups: Int32(groups), act: act.rawValue, validH: Int32(validH)), 4)
        enc.setComputePipelineState(p)
        enc.dispatchThreads(MTLSize(width: Cout, height: Wout, depth: Hout),
                            threadsPerThreadgroup: MTLSize(width: min(Cout, 256), height: 1, depth: 1))
    }

    /// Relative-position attention. q [Tq, H*dk], k/v [Tk, H*dk], posTable [nRel, H*dk] indexed by rel + relOffset,
    /// ranges [Tq] of Int32 pairs (key start, key end), out [Tq, H*dk].
    public func relPosAttention(q: Tensor, k: Tensor, v: Tensor, posTable: Tensor, relOffset: Int,
                                biasU: Tensor, biasV: Tensor, ranges: Tensor, out: Tensor, heads H: Int) {
        let HD = q.shape.last!, dk = HD / H
        let Tq = q.count / HD, Tk = k.count / HD
        let p = try! ctx.pipeline("relpos_attention_\(tag(q.dtype))")
        bind(q, 0); bind(k, 1); bind(v, 2); bind(posTable, 3); bind(biasU, 4); bind(biasV, 5); bind(ranges, 6); bind(out, 7)
        setBytes(RelAttnParams(Tq: Int32(Tq), Tk: Int32(Tk), H: Int32(H), dk: Int32(dk), ldq: Int32(HD), ldk: Int32(HD),
                               ldv: Int32(HD), ldo: Int32(HD), relOffset: Int32(relOffset), nRel: Int32(posTable.shape[0]),
                               scale: 1 / Float(dk).squareRoot()), 8)
        dispatch(p, threadgroups: MTLSize(width: Tq, height: H, depth: 1), threads: MTLSize(width: dk, height: 1, depth: 1))
    }

    /// out = a + s * b (elementwise, same shapes)
    public func addScaled(_ a: Tensor, _ b: Tensor, scale s: Float = 1, out: Tensor) {
        let p = try! ctx.pipeline("add_scaled_\(tag(a.dtype))")
        bind(a, 0); bind(b, 1); bind(out, 2); setBytes(s, 3)
        dispatch1D(p, count: a.count)
    }

    /// Copies all of `src` into `dst` (same byte size, multiple of 4 bytes).
    public func copy(_ src: Tensor, into dst: Tensor) {
        precondition(src.byteCount == dst.byteCount && src.byteCount % 4 == 0, "copy \(src.shape) -> \(dst.shape)")
        let p = try! ctx.pipeline("copy_words")
        bind(src, 0); bind(dst, 1)
        dispatch1D(p, count: src.byteCount / 4)
    }

    /// Zero-fills a tensor (byte size must be a multiple of 4).
    public func zero(_ t: Tensor) {
        precondition(t.byteCount % 4 == 0)
        let p = try! ctx.pipeline("fill_zero_words")
        bind(t, 0)
        dispatch1D(p, count: t.byteCount / 4)
    }

    public func cast(_ src: Tensor, into dst: Tensor) {
        precondition(src.count == dst.count)
        let name: String
        switch (src.dtype, dst.dtype) {
        case (.f32, .f16): name = "cast_f32_f16"
        case (.f16, .f32): name = "cast_f16_f32"
        default: fatalError("cast \(src.dtype)->\(dst.dtype)")
        }
        let p = try! ctx.pipeline(name)
        bind(src, 0); bind(dst, 1)
        dispatch1D(p, count: src.count)
    }
}

/// Kill-switch for the multi-query attention kernel (DICTATOR_ATTN_MQ=0).
let attentionMQEnabled = ProcessInfo.processInfo.environment["DICTATOR_ATTN_MQ"] != "0"

struct Im2ColParams { var Tin: Int32, C: Int32, K: Int32, stride: Int32, pad: Int32, Tout: Int32 }
struct RopeParams { var T: Int32, H: Int32, d: Int32, ld: Int32, pos0: Int32 }
struct AttnParams { var Tq: Int32, Tk: Int32, Hq: Int32, Hkv: Int32, d: Int32, ldq: Int32, ldk: Int32, ldo: Int32; var scale: Float }

extension Step {
    /// Causal conv1d input expansion: x [Tin, C] -> out [Tout, C*K].
    public func im2col1d(_ x: Tensor, out: Tensor, K: Int, stride: Int, pad: Int) {
        let (Tin, C) = (x.shape[0], x.shape[1])
        let Tout = out.shape[0]
        let p = try! ctx.pipeline("im2col1d")
        bind(x, 0); bind(out, 1)
        setBytes(Im2ColParams(Tin: Int32(Tin), C: Int32(C), K: Int32(K), stride: Int32(stride), pad: Int32(pad), Tout: Int32(Tout)), 2)
        enc.setComputePipelineState(p)
        enc.dispatchThreads(MTLSize(width: C * K, height: Tout, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// In-place rotate-half RoPE on x [T, H*d] (row stride = last dim) for positions pos0 ..< pos0 + T.
    public func rope(_ x: Tensor, heads H: Int, headDim d: Int, invFreq: Tensor, pos0: Int) {
        let ld = x.shape.last!, T = x.count / ld
        let p = try! ctx.pipeline("rope_rotate_half")
        bind(x, 0); bind(invFreq, 1)
        setBytes(RopeParams(T: Int32(T), H: Int32(H), d: Int32(d), ld: Int32(ld), pos0: Int32(pos0)), 2)
        enc.setComputePipelineState(p)
        enc.dispatchThreads(MTLSize(width: d / 2, height: H, depth: T), threadsPerThreadgroup: MTLSize(width: d / 2, height: 1, depth: 1))
    }

    /// Grouped-query attention with per-query key ranges. q [Tq, Hq*d], k/v [Tk, Hkv*d], out [Tq, Hq*d].
    public func attention(q: Tensor, k: Tensor, v: Tensor, ranges: Tensor, out: Tensor, heads Hq: Int, kvHeads Hkv: Int, headDim d: Int) {
        let Tq = q.count / (Hq * d), Tk = k.count / (Hkv * d)
        // Few queries: one threadgroup per head reads each K/V row once for all queries.
        if Tq > 1 && Tq <= 8 && d <= 128 && attentionMQEnabled {
            let p = try! ctx.pipeline(k.dtype == .f16 ? "attention_mq_f16" : "attention_mq_f32")
            bind(q, 0); bind(k, 1); bind(v, 2); bind(ranges, 3); bind(out, 4)
            setBytes(AttnParams(Tq: Int32(Tq), Tk: Int32(Tk), Hq: Int32(Hq), Hkv: Int32(Hkv), d: Int32(d), ldq: Int32(Hq * d),
                                ldk: Int32(Hkv * d), ldo: Int32(Hq * d), scale: 1 / Float(d).squareRoot()), 5)
            dispatch(p, threadgroups: MTLSize(width: Hq, height: 1, depth: 1), threads: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        let p = try! ctx.pipeline(k.dtype == .f16 ? "attention_gqa_f16" : "attention_gqa_f32")
        bind(q, 0); bind(k, 1); bind(v, 2); bind(ranges, 3); bind(out, 4)
        setBytes(AttnParams(Tq: Int32(Tq), Tk: Int32(Tk), Hq: Int32(Hq), Hkv: Int32(Hkv), d: Int32(d), ldq: Int32(Hq * d),
                            ldk: Int32(Hkv * d), ldo: Int32(Hq * d), scale: 1 / Float(d).squareRoot()), 5)
        dispatch(p, threadgroups: MTLSize(width: Tq, height: Hq, depth: 1), threads: MTLSize(width: max(d, 32), height: 1, depth: 1))
    }

    public func swiglu(gate: Tensor, up: Tensor, out: Tensor) {
        let p = try! ctx.pipeline("swiglu")
        bind(gate, 0); bind(up, 1); bind(out, 2)
        dispatch1D(p, count: out.count)
    }

    /// out[t] = table[ids[t]] (+ audio[t]); table bf16 [V, D].
    public func embed(table: Tensor, ids: Tensor, audio: Tensor?, out: Tensor) {
        let D = table.shape[1], T = ids.count
        let p = try! ctx.pipeline("embed_add_bf16")
        bind(table, 0); bind(ids, 1); bind(audio ?? out, 2); bind(out, 3)
        setBytes(Int32(D), 4); setBytes(Int32(audio == nil ? 0 : 1), 5)
        enc.setComputePipelineState(p)
        enc.dispatchThreads(MTLSize(width: D, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Row-wise argmax of x [rows, N] into idx (i32) and val (f32).
    public func argmax(_ x: Tensor, idx: Tensor, val: Tensor) {
        let N = x.shape.last!, rows = x.count / N
        let p = try! ctx.pipeline("argmax_rows")
        bind(x, 0); bind(idx, 1); bind(val, 2); setBytes(Int32(N), 3)
        dispatch(p, threadgroups: MTLSize(width: rows, height: 1, depth: 1), threads: MTLSize(width: 1024, height: 1, depth: 1))
    }
}
