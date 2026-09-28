import Foundation
import Metal

/// Static configuration of a NeMo FastConformer encoder.
public struct FastConformerConfig {
    public var layers = 24
    public var dModel = 1024
    public var heads = 8
    public var ffDim = 4096
    public var convKernel = 9
    public var subsamplingChannels = 256
    public var melBins = 128
    /// Causal subsampling (Nemotron) pads (2, 1) on time and frequency; non-causal (Parakeet) pads (1, 1).
    public var causalSubsampling = true
    /// Causal depthwise conv (Nemotron) pads (K-1, 0); symmetric (Parakeet) pads ((K-1)/2, (K-1)/2).
    public var causalConv = true
    /// LayerNorm in the conv module (Nemotron); Parakeet uses BatchNorm folded into the depthwise conv.
    public var convLayerNorm = true
    /// Parakeet multiplies the subsampled input by sqrt(d_model).
    public var xscale = false

    public var headDim: Int { dModel / heads }
    public var subsampledFreq: Int {
        var f = melBins
        for _ in 0..<3 { f = causalSubsampling ? (f + 3 - 3) / 2 + 1 : (f + 2 - 3) / 2 + 1 }
        return f
    }

    public static let nemotron = FastConformerConfig()
    public static var parakeet: FastConformerConfig {
        var c = FastConformerConfig()
        c.causalSubsampling = false
        c.causalConv = false
        c.convLayerNorm = false
        c.xscale = true
        return c
    }
}

/// Weights of one conformer layer, as GPU tensors.
struct ConformerLayerWeights {
    var normFF1W, normFF1B, ff1W1, ff1W2: Tensor
    var normAttW, normAttB, wq, wk, wv, wo, wpos, biasU, biasV: Tensor
    var normConvW, normConvB: Tensor?
    var pw1: Tensor
    var dw: Tensor?
    var dwBias: Tensor?
    var convNormW, convNormB: Tensor?
    var pw2: Tensor
    var normFF2W, normFF2B, ff2W1, ff2W2: Tensor
    var normOutW, normOutB: Tensor
}

/// Subsampling weights (dw_striding, 3 conv stages + linear).
struct SubsamplingWeights {
    var c0W, c0B, c2W, c2B, c3W, c3B, c5W, c5B, c6W, c6B: Tensor
    var outW, outB: Tensor   // outW columns permuted to (freq, channel) order for channel-last input
}

public final class FastConformerEncoder {
    public let config: FastConformerConfig
    let ctx: MetalContext
    let layers: [ConformerLayerWeights]
    let sub: SubsamplingWeights
    /// Per-layer projected relative-position tables: [nRel, dModel], row = rel + relOffset.
    var posTables: [Tensor] = []
    var relOffset = 0
    var relRange = 0...0

    public init(weights: SafeTensorsFile, config: FastConformerConfig, ctx: MetalContext = .shared) throws {
        self.config = config
        self.ctx = ctx
        func t(_ n: String) throws -> Tensor { try weights.tensor(n) }
        func reshaped(_ n: String, _ shape: [Int]) throws -> Tensor { try weights.tensor(n).reshaped(shape) }

        // Subsampling. The final linear expects NeMo's (channel, freq) flatten order; our activations
        // are channel-last (freq, channel), so permute its input columns once here.
        let C = config.subsamplingChannels, F = config.subsampledFreq
        let outWt = try weights.floats("encoder.pre_encode.out.weight")
        let D = config.dModel
        var perm = [Float](repeating: 0, count: outWt.count)
        for o in 0..<D { for c in 0..<C { for f in 0..<F {
            perm[o * C * F + f * C + c] = outWt[o * C * F + c * F + f]
        } } }
        sub = SubsamplingWeights(
            c0W: try t("encoder.pre_encode.conv.0.weight"), c0B: try t("encoder.pre_encode.conv.0.bias"),
            c2W: try t("encoder.pre_encode.conv.2.weight"), c2B: try t("encoder.pre_encode.conv.2.bias"),
            c3W: try reshaped("encoder.pre_encode.conv.3.weight", [C, C]), c3B: try t("encoder.pre_encode.conv.3.bias"),
            c5W: try t("encoder.pre_encode.conv.5.weight"), c5B: try t("encoder.pre_encode.conv.5.bias"),
            c6W: try reshaped("encoder.pre_encode.conv.6.weight", [C, C]), c6B: try t("encoder.pre_encode.conv.6.bias"),
            outW: Tensor.from(perm, shape: [D, C * F], dtype: .f32, ctx), outB: try t("encoder.pre_encode.out.bias"))

        var ls = [ConformerLayerWeights]()
        for i in 0..<config.layers {
            let p = "encoder.layers.\(i)."
            var convNormW: Tensor? = nil, convNormB: Tensor? = nil
            var dwBias: Tensor? = nil
            var dw = try reshaped(p + "conv.depthwise_conv.weight", [D, config.convKernel])
            if config.convLayerNorm {
                convNormW = try t(p + "conv.batch_norm.weight")
                convNormB = try t(p + "conv.batch_norm.bias")
            } else {
                // Fold inference-mode BatchNorm into the depthwise conv: y = (conv(x) - mean) * g / sqrt(var + eps) + b.
                let w = try weights.floats(p + "conv.depthwise_conv.weight")
                let convB = weights.has(p + "conv.depthwise_conv.bias") ? try weights.floats(p + "conv.depthwise_conv.bias") : [Float](repeating: 0, count: D)
                let g = try weights.floats(p + "conv.batch_norm.weight"), b = try weights.floats(p + "conv.batch_norm.bias")
                let mean = try weights.floats(p + "conv.batch_norm.running_mean"), v = try weights.floats(p + "conv.batch_norm.running_var")
                var fw = w, fb = [Float](repeating: 0, count: D)
                for d in 0..<D {
                    let s = g[d] / (v[d] + 1e-5).squareRoot()
                    for k in 0..<config.convKernel { fw[d * config.convKernel + k] = w[d * config.convKernel + k] * s }
                    fb[d] = (convB[d] - mean[d]) * s + b[d]
                }
                dw = Tensor.from(fw, shape: [D, config.convKernel], dtype: .f32, ctx)
                dwBias = Tensor.from(fb, shape: [D], dtype: .f32, ctx)
            }
            ls.append(ConformerLayerWeights(
                normFF1W: try t(p + "norm_feed_forward1.weight"), normFF1B: try t(p + "norm_feed_forward1.bias"),
                ff1W1: try t(p + "feed_forward1.linear1.weight"), ff1W2: try t(p + "feed_forward1.linear2.weight"),
                normAttW: try t(p + "norm_self_att.weight"), normAttB: try t(p + "norm_self_att.bias"),
                wq: try t(p + "self_attn.linear_q.weight"), wk: try t(p + "self_attn.linear_k.weight"),
                wv: try t(p + "self_attn.linear_v.weight"), wo: try t(p + "self_attn.linear_out.weight"),
                wpos: try t(p + "self_attn.linear_pos.weight"),
                biasU: try t(p + "self_attn.pos_bias_u"), biasV: try t(p + "self_attn.pos_bias_v"),
                normConvW: try t(p + "norm_conv.weight"), normConvB: try t(p + "norm_conv.bias"),
                pw1: try reshaped(p + "conv.pointwise_conv1.weight", [2 * D, D]),
                dw: dw, dwBias: dwBias, convNormW: convNormW, convNormB: convNormB,
                pw2: try reshaped(p + "conv.pointwise_conv2.weight", [D, D]),
                normFF2W: try t(p + "norm_feed_forward2.weight"), normFF2B: try t(p + "norm_feed_forward2.bias"),
                ff2W1: try t(p + "feed_forward2.linear1.weight"), ff2W2: try t(p + "feed_forward2.linear2.weight"),
                normOutW: try t(p + "norm_out.weight"), normOutB: try t(p + "norm_out.bias")))
        }
        layers = ls
    }

    /// NeMo RelPositionalEncoding sinusoid for relative position `pos` (float32 arithmetic as in torch).
    func sinusoid(_ pos: Int) -> [Float] {
        let D = config.dModel
        var pe = [Float](repeating: 0, count: D)
        let k = -(Float(log(10000.0)) / Float(D))
        for i in stride(from: 0, to: D, by: 2) {
            let div = Float(exp(Double(Float(i) * k)))
            let angle = Float(pos) * div
            pe[i] = Float(sin(Double(angle)))
            pe[i + 1] = Float(cos(Double(angle)))
        }
        return pe
    }

    /// Builds per-layer projected position tables covering relative positions in `range`.
    public func preparePositionTables(range: ClosedRange<Int>) {
        if range == relRange && !posTables.isEmpty { return }
        let D = config.dModel
        let n = range.count
        var pe = [Float]()
        pe.reserveCapacity(n * D)
        for r in range { pe += sinusoid(r) }
        let peT = Tensor.from(pe, shape: [n, D], dtype: .f32, ctx)
        let s = Step(ctx)
        posTables = layers.map { l in
            let out = Tensor(ctx, shape: [n, D], dtype: .f32)
            s.gemm(peT, l.wpos, out: out)
            return out
        }
        s.run()
        relOffset = -range.lowerBound
        relRange = range
    }

    /// Output frames produced by the subsampling for `melFrames` input frames.
    public func subsampledLength(_ melFrames: Int) -> Int {
        var t = melFrames
        for _ in 0..<3 { t = config.causalSubsampling ? (t + 3 - 3) / 2 + 1 : (t + 2 - 3) / 2 + 1 }
        return t
    }

    /// Encodes the subsampling: mel [T, melBins] -> [T', dModel]. Rows >= validOut are zero-masked
    /// at each stage exactly like NeMo's MaskedConvSequential.
    func encodeSubsampling(_ s: Step, mel: Tensor, validMel: Int, out: Tensor) {
        let cfg = config
        let C = cfg.subsamplingChannels
        let (pt, pl) = cfg.causalSubsampling ? (2, 2) : (1, 1)
        func next(_ t: Int) -> Int { cfg.causalSubsampling ? (t + 3 - 3) / 2 + 1 : (t + 2 - 3) / 2 + 1 }
        let T0 = mel.shape[0], F0 = cfg.melBins
        let T1 = next(T0), F1 = next(F0), T2 = next(T1), F2 = next(F1), T3 = next(T2), F3 = next(F2)
        let v1 = next(validMel), v2 = next(v1), v3 = next(v2)
        let x0 = mel.reshaped([T0, F0, 1])
        let a1 = Tensor(ctx, shape: [T1, F1, C])
        s.conv2d(x0, w: sub.c0W, bias: sub.c0B, out: a1, stride: 2, padTop: pt, padLeft: pl, groups: 1, act: .relu, validH: v1)
        let a2 = Tensor(ctx, shape: [T2, F2, C])
        s.conv2d(a1, w: sub.c2W, bias: sub.c2B, out: a2, stride: 2, padTop: pt, padLeft: pl, groups: C, act: .none, validH: T2)
        let a3 = Tensor(ctx, shape: [T2, F2, C])
        s.gemm(a2.reshaped([T2 * F2, C]), sub.c3W, bias: sub.c3B, out: a3.reshaped([T2 * F2, C]), act: .relu)
        // NeMo masks before the next strided layer using the stride-updated length (v2).
        let a3m = maskRows(s, a3, validRows: v2)
        let a4 = Tensor(ctx, shape: [T3, F3, C])
        s.conv2d(a3m, w: sub.c5W, bias: sub.c5B, out: a4, stride: 2, padTop: pt, padLeft: pl, groups: C, act: .none, validH: T3)
        let a5 = Tensor(ctx, shape: [T3, F3, C])
        s.gemm(a4.reshaped([T3 * F3, C]), sub.c6W, bias: sub.c6B, out: a5.reshaped([T3 * F3, C]), act: .relu)
        let a5m = maskRows(s, a5, validRows: v3)
        s.gemm(a5m.reshaped([T3, F3 * C]), sub.outW, bias: sub.outB, out: out, alpha: 1, outScale: cfg.xscale ? Float(cfg.dModel).squareRoot() : 1)
    }

    /// Zeroes rows >= validRows (first axis). Returns the same tensor.
    func maskRows(_ s: Step, _ x: Tensor, validRows: Int) -> Tensor {
        let T = x.shape[0]
        if validRows >= T { return x }
        let rowElems = x.count / T
        let zero = Tensor(ctx, shape: [(T - validRows) * rowElems], dtype: x.dtype)
        memset(zero.contents, 0, zero.byteCount)
        s.copy(zero, into: x.rows(validRows, T - validRows).reshaped([(T - validRows) * rowElems]))
        return x
    }

    /// Hook for capturing intermediate tensors (golden-tensor comparisons). Called with a name and
    /// a tensor that is valid once the step has run.
    public typealias Tap = (String, Tensor) -> Void

    /// One conformer layer over x [T, D] with optional left context for attention (projected K/V of
    /// previous frames) and for the depthwise conv (previous conv inputs).
    ///  - kvCache: (k, v) tensors [Tc, D] prepended to this chunk's K/V.
    ///  - convCache: [K-1, D] rows prepended to the depthwise conv input (causal) — nil means zeros.
    ///  - ranges: per-query key range over the concatenated [cache ; chunk] keys.
    func encodeLayer(_ s: Step, _ li: Int, x: Tensor, out: Tensor, ranges: Tensor,
                     kCache: Tensor?, vCache: Tensor?, convLeft: Tensor?, validFrames: Int,
                     kOut: Tensor, vOut: Tensor, convIn: Tensor, tap: Tap? = nil) {
        let w = layers[li]
        let cfg = config
        let D = cfg.dModel, T = x.shape[0]
        let tmpD = Tensor(ctx, shape: [T, D])
        let ffH = Tensor(ctx, shape: [T, cfg.ffDim])
        let r1 = Tensor(ctx, shape: [T, D])

        // FF1 (macaron, x0.5)
        s.layerNorm(x, gamma: w.normFF1W, beta: w.normFF1B, out: tmpD)
        s.gemm(tmpD, w.ff1W1, out: ffH, act: .silu)
        s.gemm(ffH, w.ff1W2, out: r1, outScale: 0.5, residual: x)
        tap?("ff1", r1)

        // Self-attention with relative positions. K/V for this chunk land in kOut/vOut, which are
        // the tail of [cache ; chunk] buffers so attention sees one contiguous key range.
        s.layerNorm(r1, gamma: w.normAttW, beta: w.normAttB, out: tmpD)
        let q = Tensor(ctx, shape: [T, D])
        s.gemm(tmpD, w.wq, out: q)
        s.gemm(tmpD, w.wk, out: kOut)
        s.gemm(tmpD, w.wv, out: vOut)
        let keys = kCache ?? kOut, vals = vCache ?? vOut
        let att = Tensor(ctx, shape: [T, D])
        s.relPosAttention(q: q, k: keys, v: vals, posTable: posTables[li], relOffset: relOffset,
                          biasU: w.biasU, biasV: w.biasV, ranges: ranges, out: att, heads: cfg.heads)
        let r2 = Tensor(ctx, shape: [T, D])
        s.gemm(att, w.wo, out: r2, residual: r1)
        tap?("attn", r2)

        // Convolution module. convIn is [left ; T] rows; this chunk's GLU output goes in its tail.
        s.layerNorm(r2, gamma: w.normConvW!, beta: w.normConvB!, out: tmpD)
        let pw = Tensor(ctx, shape: [T, 2 * D])
        s.gemm(tmpD, w.pw1, out: pw)
        let K = cfg.convKernel
        let left = cfg.causalConv ? K - 1 : (K - 1) / 2
        let right = cfg.causalConv ? 0 : (K - 1) / 2
        precondition(convIn.shape[0] == left + T + right)
        let glued = convIn.rows(left, T)
        s.glu(pw, out: glued)
        if validFrames < T { _ = maskRows(s, glued, validRows: validFrames) }
        if let cl = convLeft { s.copy(cl, into: convIn.rows(0, left)) }
        if cfg.convLayerNorm {
            let dwOut = Tensor(ctx, shape: [T, D])
            s.dwconv1d(convIn, w: w.dw!, bias: w.dwBias, out: dwOut)
            s.layerNorm(dwOut, gamma: w.convNormW!, beta: w.convNormB!, out: tmpD, act: .silu)
        } else {
            // BatchNorm is folded into the depthwise weights; SiLU is fused into the conv.
            s.dwconv1d(convIn, w: w.dw!, bias: w.dwBias, out: tmpD, act: .silu)
        }
        let r3 = Tensor(ctx, shape: [T, D])
        s.gemm(tmpD, w.pw2, out: r3, residual: r2)
        tap?("conv", r3)

        // FF2 (x0.5)
        s.layerNorm(r3, gamma: w.normFF2W, beta: w.normFF2B, out: tmpD)
        s.gemm(tmpD, w.ff2W1, out: ffH, act: .silu)
        let r4 = Tensor(ctx, shape: [T, D])
        s.gemm(ffH, w.ff2W2, out: r4, outScale: 0.5, residual: r3)
        tap?("ff2", r4)

        s.layerNorm(r4, gamma: w.normOutW, beta: w.normOutB, out: out)
    }

    /// Key ranges for NeMo's `chunked_limited` mask over a full utterance of T frames.
    public static func chunkedLimitedRanges(T: Int, left: Int, right: Int) -> [Int32] {
        let cs = right + 1
        let leftChunks = left / cs
        var r = [Int32]()
        for i in 0..<T {
            let c = i / cs
            r.append(Int32(max(0, (c - leftChunks) * cs)))
            r.append(Int32(min(T, (c + 1) * cs)))
        }
        return r
    }

    /// Offline forward over a full utterance with the streaming attention mask [left, right]
    /// (chunked_limited). Returns encoder output [T', D]. `tap` receives "pre_encode",
    /// "layer_XX_out" and, for layer 0, the internal seams.
    public func forwardOffline(mel: [Float], melFrames: Int, validMel: Int, left: Int, right: Int, tap: Tap? = nil) -> Tensor {
        let cfg = config
        let D = cfg.dModel
        let T = subsampledLength(melFrames)
        let validT = { var v = validMel; for _ in 0..<3 { v = cfg.causalSubsampling ? (v + 3 - 3) / 2 + 1 : (v + 2 - 3) / 2 + 1 }; return v }()
        preparePositionTables(range: -(T - 1)...(T - 1))
        let melT = Tensor.from(mel, shape: [melFrames, cfg.melBins], dtype: .f32, ctx)
        let s = Step(ctx)
        var x = Tensor(ctx, shape: [T, D])
        encodeSubsampling(s, mel: melT, validMel: validMel, out: x)
        tap?("pre_encode", x)
        let ranges = Tensor(ctx, shape: [T, 2], dtype: .i32)
        var rr = FastConformerEncoder.chunkedLimitedRanges(T: T, left: left, right: right)
        for i in 0..<T { rr[2 * i + 1] = min(rr[2 * i + 1], Int32(validT)) }
        rr.withUnsafeBytes { ranges.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let K = cfg.convKernel
        let convRows = cfg.causalConv ? T + K - 1 : T + K - 1
        for li in 0..<cfg.layers {
            let out = Tensor(ctx, shape: [T, D])
            let kOut = Tensor(ctx, shape: [T, D]), vOut = Tensor(ctx, shape: [T, D])
            let convIn = Tensor(ctx, shape: [convRows, D])
            memset(convIn.contents, 0, convIn.byteCount)
            encodeLayer(s, li, x: x, out: out, ranges: ranges, kCache: nil, vCache: nil, convLeft: nil,
                        validFrames: validT, kOut: kOut, vOut: vOut, convIn: convIn,
                        tap: li == 0 ? { name, t in tap?("layer_00_\(name)", t) } : nil)
            tap?(String(format: "layer_%02d_out", li), out)
            x = out
        }
        s.run()
        return x
    }
}
