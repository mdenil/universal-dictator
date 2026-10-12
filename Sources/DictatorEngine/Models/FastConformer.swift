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
    // Linear/pointwise biases (Parakeet: use_bias=true; Nemotron: none).
    var ff1B1, ff1B2, ff2B1, ff2B2, bq, bk, bv, bo, pw1B, pw2B: Tensor?
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
        let outDtype = weights.entries["encoder.pre_encode.out.weight"]!.dtype
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
            outW: Tensor.from(perm, shape: [D, C * F], dtype: outDtype, ctx), outB: try t("encoder.pre_encode.out.bias"))

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
            func opt(_ n: String) throws -> Tensor? { weights.has(p + n) ? try t(p + n) : nil }
            func optConv(_ n: String) throws -> Tensor? { weights.has(p + n) ? try weights.tensor(p + n).reshaped([weights.entries[p + n]!.shape[0]]) : nil }
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
                normOutW: try t(p + "norm_out.weight"), normOutB: try t(p + "norm_out.bias"),
                ff1B1: try opt("feed_forward1.linear1.bias"), ff1B2: try opt("feed_forward1.linear2.bias"),
                ff2B1: try opt("feed_forward2.linear1.bias"), ff2B2: try opt("feed_forward2.linear2.bias"),
                bq: try opt("self_attn.linear_q.bias"), bk: try opt("self_attn.linear_k.bias"),
                bv: try opt("self_attn.linear_v.bias"), bo: try opt("self_attn.linear_out.bias"),
                pw1B: try optConv("conv.pointwise_conv1.bias"), pw2B: try optConv("conv.pointwise_conv2.bias")))
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

    /// Preallocated scratch for one encoder pass over at most `maxT` frames (post-subsampling) and
    /// `maxMel` mel frames. Streaming steps reuse it, so the steady state allocates nothing.
    public final class Workspace {
        let maxT: Int, maxMel: Int
        let tmpD, ffH, q, att, pw, r1, r2, r3, r4, x, y: Tensor
        let sub1, sub2, sub3, sub4, sub5: Tensor
        let ranges: Tensor

        init(_ cfg: FastConformerConfig, maxT: Int, maxMel: Int, ctx: MetalContext) {
            self.maxT = maxT
            self.maxMel = maxMel
            let D = cfg.dModel, C = cfg.subsamplingChannels
            func mk(_ shape: [Int]) -> Tensor { Tensor(ctx, shape: shape, dtype: .f32) }
            tmpD = mk([maxT, D]); ffH = mk([maxT, cfg.ffDim]); q = mk([maxT, D]); att = mk([maxT, D])
            pw = mk([maxT, 2 * D]); r1 = mk([maxT, D]); r2 = mk([maxT, D]); r3 = mk([maxT, D]); r4 = mk([maxT, D])
            x = mk([maxT, D]); y = mk([maxT, D])
            func nx(_ t: Int) -> Int { cfg.causalSubsampling ? t / 2 + 1 : (t - 1) / 2 + 1 }
            let T1 = nx(maxMel), F1 = nx(cfg.melBins), T2 = nx(T1), F2 = nx(F1), T3 = nx(T2), F3 = nx(F2)
            sub1 = mk([T1 * F1 * C]); sub2 = mk([T2 * F2 * C]); sub3 = mk([T2 * F2 * C])
            sub4 = mk([T3 * F3 * C]); sub5 = mk([T3 * F3 * C])
            ranges = Tensor(ctx, shape: [maxT, 2], dtype: .i32)
        }
    }

    public func makeWorkspace(maxT: Int, maxMel: Int) -> Workspace {
        Workspace(config, maxT: maxT, maxMel: maxMel, ctx: ctx)
    }

    func nextLen(_ t: Int) -> Int { config.causalSubsampling ? (t + 3 - 3) / 2 + 1 : (t + 2 - 3) / 2 + 1 }

    /// Encodes the subsampling: mel [T0, melBins] -> out [T3, dModel]. Rows beyond the valid length are
    /// zeroed at each stage exactly like NeMo's MaskedConvSequential.
    func encodeSubsampling(_ s: Step, _ ws: Workspace, mel: Tensor, validMel: Int, out: Tensor) {
        let cfg = config
        let C = cfg.subsamplingChannels
        let (pt, pl) = cfg.causalSubsampling ? (2, 2) : (1, 1)
        let T0 = mel.shape[0], F0 = cfg.melBins
        let T1 = nextLen(T0), F1 = nextLen(F0), T2 = nextLen(T1), F2 = nextLen(F1), T3 = nextLen(T2), F3 = nextLen(F2)
        let v1 = nextLen(validMel), v2 = nextLen(v1), v3 = nextLen(v2)
        precondition(out.shape[0] == T3)
        let x0 = mel.reshaped([T0, F0, 1])
        if validMel < T0 { maskRows(s, x0, validRows: validMel) }
        let a1 = ws.sub1.prefix([T1, F1, C])
        s.conv2d(x0, w: sub.c0W, bias: sub.c0B, out: a1, stride: 2, padTop: pt, padLeft: pl, groups: 1, act: .relu, validH: v1)
        let a2 = ws.sub2.prefix([T2, F2, C])
        s.conv2d(a1, w: sub.c2W, bias: sub.c2B, out: a2, stride: 2, padTop: pt, padLeft: pl, groups: C, act: .none, validH: T2)
        let a3 = ws.sub3.prefix([T2, F2, C])
        s.gemm(a2.reshaped([T2 * F2, C]), sub.c3W, bias: sub.c3B, out: a3.reshaped([T2 * F2, C]), act: .relu)
        // NeMo masks before the next layer using the stride-updated length.
        maskRows(s, a3, validRows: v2)
        let a4 = ws.sub4.prefix([T3, F3, C])
        s.conv2d(a3, w: sub.c5W, bias: sub.c5B, out: a4, stride: 2, padTop: pt, padLeft: pl, groups: C, act: .none, validH: T3)
        let a5 = ws.sub5.prefix([T3, F3, C])
        s.gemm(a4.reshaped([T3 * F3, C]), sub.c6W, bias: sub.c6B, out: a5.reshaped([T3 * F3, C]), act: .relu)
        maskRows(s, a5, validRows: v3)
        s.gemm(a5.reshaped([T3, F3 * C]), sub.outW, bias: sub.outB, out: out, outScale: cfg.xscale ? Float(cfg.dModel).squareRoot() : 1)
    }

    /// Zeroes rows >= validRows along the first axis.
    func maskRows(_ s: Step, _ x: Tensor, validRows: Int) {
        let T = x.shape[0]
        if validRows >= T { return }
        s.zero(x.rows(max(validRows, 0), T - max(validRows, 0)))
    }

    /// Hook for capturing intermediate tensors (golden-tensor comparisons). Called with a name and
    /// a tensor that is valid once the step has run. Tapped tensors are copies.
    public typealias Tap = (String, Tensor) -> Void

    /// One conformer layer over x [T, D].
    ///  - keys/values: [Tk, D] buffers whose last T rows receive this chunk's projected K/V; earlier rows
    ///    hold cached projections of previous frames (the attention left context).
    ///  - convIn: [left + T + right, D]; rows [left, left+T) receive this chunk's GLU output, the rest must
    ///    already hold the conv left context (cache or zeros) and right padding (zeros).
    ///  - ranges: per-query key range over `keys`.
    func encodeLayer(_ s: Step, _ ws: Workspace, _ li: Int, x: Tensor, out: Tensor, ranges: Tensor,
                     keys: Tensor, values: Tensor, convIn: Tensor, validFrames: Int, tap: Tap? = nil) {
        let w = layers[li]
        let cfg = config
        let T = x.shape[0]
        let tmpD = ws.tmpD.rows(0, T), ffH = ws.ffH.rows(0, T)
        let r1 = ws.r1.rows(0, T), r2 = ws.r2.rows(0, T), r3 = ws.r3.rows(0, T), r4 = ws.r4.rows(0, T)
        func capture(_ name: String, _ t: Tensor) {
            guard let tap = tap else { return }
            let c = Tensor(ctx, shape: t.shape, dtype: t.dtype)
            s.copy(t, into: c)
            tap(name, c)
        }

        // FF1 (macaron, x0.5)
        s.layerNorm(x, gamma: w.normFF1W, beta: w.normFF1B, out: tmpD)
        s.gemm(tmpD, w.ff1W1, bias: w.ff1B1, out: ffH, act: .silu)
        s.gemm(ffH, w.ff1W2, bias: w.ff1B2, out: r1, outScale: 0.5, residual: x)
        capture("ff1", r1)

        // Self-attention with relative positions.
        let Tk = keys.shape[0]
        s.layerNorm(r1, gamma: w.normAttW, beta: w.normAttB, out: tmpD)
        let q = ws.q.rows(0, T)
        s.gemm(tmpD, w.wq, bias: w.bq, out: q)
        s.gemm(tmpD, w.wk, bias: w.bk, out: keys.rows(Tk - T, T))
        s.gemm(tmpD, w.wv, bias: w.bv, out: values.rows(Tk - T, T))
        let att = ws.att.rows(0, T)
        s.relPosAttention(q: q, k: keys, v: values, posTable: posTables[li], relOffset: relOffset,
                          biasU: w.biasU, biasV: w.biasV, ranges: ranges, out: att, heads: cfg.heads)
        s.gemm(att, w.wo, bias: w.bo, out: r2, residual: r1)
        capture("attn", r2)

        // Convolution module.
        s.layerNorm(r2, gamma: w.normConvW!, beta: w.normConvB!, out: tmpD)
        let pw = ws.pw.rows(0, T)
        s.gemm(tmpD, w.pw1, bias: w.pw1B, out: pw)
        let left = cfg.causalConv ? cfg.convKernel - 1 : (cfg.convKernel - 1) / 2
        let glued = convIn.rows(left, T)
        s.glu(pw, out: glued)
        if validFrames < T { maskRows(s, glued, validRows: validFrames) }
        if cfg.convLayerNorm {
            let dwOut = ws.r4.rows(0, T)
            s.dwconv1d(convIn, w: w.dw!, bias: w.dwBias, out: dwOut)
            s.layerNorm(dwOut, gamma: w.convNormW!, beta: w.convNormB!, out: tmpD, act: .silu)
        } else {
            // BatchNorm is folded into the depthwise weights; SiLU is fused into the conv.
            s.dwconv1d(convIn, w: w.dw!, bias: w.dwBias, out: tmpD, act: .silu)
        }
        s.gemm(tmpD, w.pw2, bias: w.pw2B, out: r3, residual: r2)
        capture("conv", r3)

        // FF2 (x0.5)
        s.layerNorm(r3, gamma: w.normFF2W, beta: w.normFF2B, out: tmpD)
        s.gemm(tmpD, w.ff2W1, bias: w.ff2B1, out: ffH, act: .silu)
        s.gemm(ffH, w.ff2W2, bias: w.ff2B2, out: r4, outScale: 0.5, residual: r3)
        capture("ff2", r4)

        s.layerNorm(r4, gamma: w.normOutW, beta: w.normOutB, out: out)
    }

    /// Attention mask styles used by the models we run.
    public enum MaskSpec {
        /// NeMo `chunked_limited` [left, right]: chunks of right+1 frames, left//(right+1) chunks back.
        case chunkedLimited(left: Int, right: Int)
        /// NeMo `chunked_limited_with_rc` [left, chunk, right]: frame i sees
        /// [c*chunk - left, c*chunk + chunk - 1 + right] with c = i / chunk (inclusive, clipped).
        case chunkedLimitedRC(left: Int, chunk: Int, right: Int)

        public func ranges(T: Int) -> [Int32] {
            switch self {
            case .chunkedLimited(let l, let r): return FastConformerEncoder.chunkedLimitedRanges(T: T, left: l, right: r)
            case .chunkedLimitedRC(let l, let c, let r):
                var out = [Int32]()
                for i in 0..<T {
                    let ci = i / c
                    out.append(Int32(max(0, ci * c - l)))
                    out.append(Int32(min(T, ci * c + c + r)))
                }
                return out
            }
        }
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

    /// Offline forward over a full utterance with a streaming attention mask. Returns encoder output
    /// [T', D]. `tap` receives "pre_encode", "layer_XX_out" and, for layer 0, the internal seams.
    public func forwardOffline(mel: [Float], melFrames: Int, validMel: Int, mask: MaskSpec, tap: Tap? = nil) -> Tensor {
        let cfg = config
        let D = cfg.dModel
        let T = subsampledLength(melFrames)
        var validT = validMel
        for _ in 0..<3 { validT = nextLen(validT) }
        preparePositionTables(range: -(T - 1)...(T - 1))
        let ws = makeWorkspace(maxT: T, maxMel: melFrames)
        let melT = Tensor.from(mel, shape: [melFrames, cfg.melBins], dtype: .f32, ctx)
        let s = Step(ctx)
        func capture(_ name: String, _ t: Tensor) {
            guard let tap = tap else { return }
            let c = Tensor(ctx, shape: t.shape, dtype: t.dtype)
            s.copy(t, into: c)
            tap(name, c)
        }
        var x = ws.x.rows(0, T), y = ws.y.rows(0, T)
        encodeSubsampling(s, ws, mel: melT, validMel: validMel, out: x)
        capture("pre_encode", x)
        var rr = mask.ranges(T: T)
        for i in 0..<T { rr[2 * i + 1] = min(rr[2 * i + 1], Int32(validT)) }
        let ranges = ws.ranges.rows(0, T)
        rr.withUnsafeBytes { ranges.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let K = cfg.convKernel
        let keys = Tensor(ctx, shape: [T, D]), values = Tensor(ctx, shape: [T, D])
        let convIn = Tensor(ctx, shape: [T + K - 1, D])
        for li in 0..<cfg.layers {
            s.zero(convIn)
            encodeLayer(s, ws, li, x: x, out: y, ranges: ranges, keys: keys, values: values, convIn: convIn,
                        validFrames: validT, tap: li == 0 ? { name, t in tap?("layer_00_\(name)", t) } : nil)
            capture(String(format: "layer_%02d_out", li), y)
            swap(&x, &y)
        }
        let out = Tensor(ctx, shape: [T, D])
        s.copy(x, into: out)
        s.run()
        return out
    }

    // MARK: - Cache-aware streaming

    /// Per-stream encoder state for NeMo cache-aware streaming (chunked_limited attention).
    /// K/V caches hold *projected* keys and values of the last `leftContext` frames per layer (NeMo caches
    /// the normalized layer inputs and re-projects them every step; projecting once is equivalent).
    /// Caches ping-pong between two buffers so updates never overlap their source.
    public final class StreamState {
        let leftContext: Int
        let maxChunk: Int
        var kv: [[Tensor]]        // [layer][2]: [leftContext + maxChunk, D], K and V interleaved as separate arrays
        var vv: [[Tensor]]
        var conv: [[Tensor]]      // [layer][2]: [K-1 + maxChunk, D]
        var parity = 0
        public internal(set) var cacheLen = 0
        public internal(set) var step = 0

        init(_ cfg: FastConformerConfig, leftContext: Int, maxChunk: Int, ctx: MetalContext) {
            self.leftContext = leftContext
            self.maxChunk = maxChunk
            let D = cfg.dModel
            func mk(_ rows: Int) -> Tensor {
                let t = Tensor(ctx, shape: [rows, D], dtype: .f32)
                memset(t.contents, 0, t.byteCount)
                return t
            }
            kv = (0..<cfg.layers).map { _ in [mk(leftContext + maxChunk), mk(leftContext + maxChunk)] }
            vv = (0..<cfg.layers).map { _ in [mk(leftContext + maxChunk), mk(leftContext + maxChunk)] }
            conv = (0..<cfg.layers).map { _ in [mk(cfg.convKernel - 1 + maxChunk), mk(cfg.convKernel - 1 + maxChunk)] }
        }
    }

    public func makeStreamState(leftContext: Int, maxChunk: Int) -> StreamState {
        precondition(config.causalConv, "cache-aware streaming needs causal convolutions")
        preparePositionTables(range: -(maxChunk - 1)...(leftContext + maxChunk - 1))
        return StreamState(config, leftContext: leftContext, maxChunk: maxChunk, ctx: ctx)
    }

    /// Encodes one cache-aware streaming step into `s`:
    /// melChunk [Tin, melBins] (already including NeMo's pre-encode cache frames), `validMel` valid rows,
    /// `drop` subsampled frames removed from the front (NeMo drop_extra_pre_encoded).
    /// Writes the encoder output rows to `out` ([T, D], T = subsampled - drop) and returns (T, validT).
    func encodeStreamStep(_ s: Step, _ ws: Workspace, state st: StreamState, melChunk: Tensor, validMel: Int, drop: Int,
                          out: Tensor) -> (T: Int, valid: Int) {
        let cfg = config
        let K = cfg.convKernel, L = st.leftContext
        let Tsub = subsampledLength(melChunk.shape[0])
        var validSub = validMel
        for _ in 0..<3 { validSub = nextLen(validSub) }
        let T = Tsub - drop
        let validT = max(0, min(validSub - drop, T))
        precondition(T <= st.maxChunk && T <= ws.maxT, "chunk of \(T) frames exceeds workspace")

        let subOut = ws.y.rows(0, Tsub)
        encodeSubsampling(s, ws, mel: melChunk, validMel: validMel, out: subOut)
        var x = ws.x.rows(0, T)
        s.copy(subOut.rows(drop, T), into: x)
        var y = ws.y.rows(0, T)

        // Keys: [L cached ; T new]. Valid cached keys are the last cacheLen of the L rows.
        var rr = [Int32]()
        for _ in 0..<T { rr += [Int32(L - st.cacheLen), Int32(L + validT)] }
        let ranges = ws.ranges.rows(0, T)
        rr.withUnsafeBytes { ranges.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }

        let p = st.parity, np = 1 - p
        for li in 0..<cfg.layers {
            let keys = st.kv[li][p].rows(0, L + T), values = st.vv[li][p].rows(0, L + T)
            let convIn = st.conv[li][p].rows(0, K - 1 + T)
            encodeLayer(s, ws, li, x: x, out: y, ranges: ranges, keys: keys, values: values, convIn: convIn, validFrames: validT)
            // Next caches: the last L keys/values and the last K-1 conv inputs.
            s.copy(keys.rows(T, L), into: st.kv[li][np].rows(0, L))
            s.copy(values.rows(T, L), into: st.vv[li][np].rows(0, L))
            s.copy(convIn.rows(T, K - 1), into: st.conv[li][np].rows(0, K - 1))
            swap(&x, &y)
        }
        s.copy(x, into: out.rows(0, T))
        st.parity = np
        st.cacheLen = min(st.cacheLen + T, L)
        st.step += 1
        return (T, validT)
    }
}

extension Tensor {
    /// A view of the first elements of this tensor reshaped to `shape`.
    func prefix(_ shape: [Int]) -> Tensor {
        let n = shape.reduce(1, *)
        precondition(n <= count, "prefix \(shape) exceeds \(self.shape)")
        return Tensor(buffer: buffer, offset: offset, shape: shape, dtype: dtype)
    }
}

// MARK: - Region caching (Parakeet Stage B: frozen-prefix caching for chunked_limited_with_rc)

extension FastConformerEncoder {
    /// Per-layer frozen context for region streaming: projected K/V of up to `capacity` frozen frames and
    /// the conv inputs of the last (K-1)/2 frozen frames. Buffers ping-pong so updates never overlap.
    public final class RegionCache {
        let capacity: Int
        let maxRegion: Int
        var kv: [[Tensor]], vv: [[Tensor]], conv: [[Tensor]]
        var parity = 0
        public internal(set) var cached = 0

        init(_ cfg: FastConformerConfig, capacity: Int, maxRegion: Int, ctx: MetalContext) {
            self.capacity = capacity
            self.maxRegion = maxRegion
            let D = cfg.dModel, half = (cfg.convKernel - 1) / 2
            func mk(_ rows: Int) -> Tensor {
                let t = Tensor(ctx, shape: [rows, D], dtype: .f32)
                memset(t.contents, 0, t.byteCount)
                return t
            }
            kv = (0..<cfg.layers).map { _ in [mk(capacity + maxRegion), mk(capacity + maxRegion)] }
            vv = (0..<cfg.layers).map { _ in [mk(capacity + maxRegion), mk(capacity + maxRegion)] }
            conv = (0..<cfg.layers).map { _ in [mk(2 * half + maxRegion), mk(2 * half + maxRegion)] }
        }
    }

    public func makeRegionCache(capacity: Int, maxRegion: Int) -> RegionCache {
        precondition(!config.causalConv, "region caching is for symmetric-conv encoders")
        preparePositionTables(range: -(maxRegion - 1)...(capacity + maxRegion - 1))
        return RegionCache(config, capacity: capacity, maxRegion: maxRegion, ctx: ctx)
    }

    /// Runs all layers over region frames x [R, D] (the newest frames), attending to the frozen cache,
    /// then freezes the first `advance` region frames into the cache. `ranges` index keys laid out as
    /// [cached ; region]. Output rows go to `out` [R, D].
    func encodeRegion(_ s: Step, _ ws: Workspace, cache c: RegionCache, x input: Tensor, ranges: Tensor,
                      advance: Int, out: Tensor) {
        let cfg = config
        let R = input.shape[0], nc = c.cached, half = (cfg.convKernel - 1) / 2
        precondition(R <= c.maxRegion && advance <= R)
        let newNc = min(c.capacity, nc + advance)
        let p = c.parity, np = 1 - p
        var x = ws.x.rows(0, R), y = ws.y.rows(0, R)
        s.copy(input, into: x)
        for li in 0..<cfg.layers {
            let keys = c.kv[li][p].rows(0, nc + R), values = c.vv[li][p].rows(0, nc + R)
            let convIn = c.conv[li][p].rows(0, 2 * half + R)
            s.zero(convIn.rows(half + R, half))   // right edge: zero padding like the buffer end
            encodeLayer(s, ws, li, x: x, out: y, ranges: ranges, keys: keys, values: values, convIn: convIn, validFrames: R)
            if newNc > 0 {
                s.copy(keys.rows(nc + advance - newNc, newNc), into: c.kv[li][np].rows(0, newNc))
                s.copy(values.rows(nc + advance - newNc, newNc), into: c.vv[li][np].rows(0, newNc))
            }
            s.copy(convIn.rows(advance, half), into: c.conv[li][np].rows(0, half))
            swap(&x, &y)
        }
        s.copy(x, into: out.rows(0, R))
        c.parity = np
        c.cached = newNc
    }
}
