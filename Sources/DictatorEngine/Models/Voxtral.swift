import Accelerate
import Foundation

// Voxtral Mini 4B Realtime (docs/specs/voxtral.md). Section numbers below refer to that spec.

/// §2 log-mel frontend: periodic Hann 400, n_fft 400, hop 160, power, Slaney mel (128 bins, 0-8 kHz),
/// log10(max(., 1e-10)), max(., -6.5), (x + 4) / 4. Frames are computed on the padded stream.
public final class VoxtralFrontend {
    public let nMels = 128
    let nFFT = 400, hop = 160, nBins = 201
    let cosT: [Float], sinT: [Float]   // [nBins, nFFT], window folded in
    let filters: [Float]               // [nMels, nBins]

    public init() {
        var c = [Float](repeating: 0, count: nBins * nFFT), s = c
        for k in 0..<nBins { for n in 0..<nFFT {
            let w = 0.5 - 0.5 * cos(2 * Double.pi * Double(n) / Double(nFFT))   // periodic Hann
            let ph = 2 * Double.pi * Double(k) * Double(n) / Double(nFFT)
            c[k * nFFT + n] = Float(cos(ph) * w)
            s[k * nFFT + n] = Float(-sin(ph) * w)
        } }
        cosT = c; sinT = s
        filters = VoxtralFrontend.slaneyFilters(nBins: 201, nMels: 128, fMin: 0, fMax: 8000, sr: 16000)
    }

    static func hzToMel(_ f: Double) -> Double {
        let fSp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / fSp, logstep = log(6.4) / 27
        return f < minLogHz ? f / fSp : minLogMel + log(f / minLogHz) / logstep
    }
    static func melToHz(_ m: Double) -> Double {
        let fSp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / fSp, logstep = log(6.4) / 27
        return m < minLogMel ? fSp * m : minLogHz * exp(logstep * (m - minLogMel))
    }

    /// transformers `mel_filter_bank(..., norm="slaney", mel_scale="slaney")` in float64.
    static func slaneyFilters(nBins: Int, nMels: Int, fMin: Double, fMax: Double, sr: Double) -> [Float] {
        let mMin = hzToMel(fMin), mMax = hzToMel(fMax)
        let edges = (0..<(nMels + 2)).map { melToHz(mMin + (mMax - mMin) * Double($0) / Double(nMels + 1)) }
        let fftFreqs = (0..<nBins).map { Double($0) * (sr / 2) / Double(nBins - 1) }
        var out = [Float](repeating: 0, count: nMels * nBins)
        for m in 0..<nMels {
            let lo = edges[m], ce = edges[m + 1], hi = edges[m + 2]
            let enorm = 2.0 / (hi - lo)
            for b in 0..<nBins {
                let f = fftFreqs[b]
                let down = (f - lo) / (ce - lo), up = (hi - f) / (hi - ce)
                out[m * nBins + b] = Float(max(0, min(down, up)) * enorm)
            }
        }
        return out
    }

    /// Mel frames t0 ..< t0+n of the padded stream `x` (frame t covers samples [160t-200, 160t+200);
    /// out-of-range samples are reflect-padded like torch.stft(center=True)). Returns [n, 128].
    public func frames(_ x: [Float], from t0: Int, count n: Int) -> [Float] {
        let N = x.count
        var fr = [Float](repeating: 0, count: n * nFFT)
        for t in 0..<n {
            let start = (t0 + t) * hop - nFFT / 2
            for i in 0..<nFFT {
                var idx = start + i
                if idx < 0 { idx = -idx }
                if idx >= N { idx = 2 * (N - 1) - idx }
                fr[t * nFFT + i] = (idx >= 0 && idx < N) ? x[idx] : 0
            }
        }
        var re = [Float](repeating: 0, count: n * nBins), im = re
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(n), Int32(nBins), Int32(nFFT), 1, fr, Int32(nFFT), cosT, Int32(nFFT), 0, &re, Int32(nBins))
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(n), Int32(nBins), Int32(nFFT), 1, fr, Int32(nFFT), sinT, Int32(nFFT), 0, &im, Int32(nBins))
        var power = [Float](repeating: 0, count: n * nBins)
        for i in 0..<(n * nBins) { power[i] = re[i] * re[i] + im[i] * im[i] }
        var mel = [Float](repeating: 0, count: n * nMels)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(n), Int32(nMels), Int32(nBins), 1, power, Int32(nBins), filters, Int32(nBins), 0, &mel, Int32(nMels))
        for i in 0..<mel.count { mel[i] = (max(log10(max(mel[i], 1e-10)), -6.5) + 4) / 4 }
        return mel
    }
}

/// Tekken v7 detokenizer: ids < 1000 are control tokens (dropped); id >= 1000 maps to vocab[id - 1000] bytes.
public struct TekkenDecoder {
    let bytes: [[UInt8]]
    public init(url: URL) throws {
        let d = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        let vocab = d["vocab"] as! [[String: Any]]
        bytes = vocab.prefix(131072 - 1000).map { Array(Data(base64Encoded: $0["token_bytes"] as! String) ?? Data()) }
    }
    public func decode(_ ids: [Int]) -> String {
        // Each maximal run of text tokens is decoded separately, as the reference does.
        var out = "", run = [UInt8]()
        for id in ids {
            if id >= 1000 && id - 1000 < bytes.count { run += bytes[id - 1000] }
            else if !run.isEmpty { out += String(decoding: run, as: UTF8.self); run = [] }
        }
        if !run.isEmpty { out += String(decoding: run, as: UTF8.self) }
        return out
    }
}

public final class VoxtralModel {
    // Encoder (§4)
    let encDim = 1280, encHeads = 32, encHD = 64, encFF = 5120, encLayers = 32, encWindow = 750
    // Decoder (§6)
    let decDim = 3072, decHeads = 32, decKV = 8, decHD = 128, decFF = 9216, decLayers = 26, decWindow = 8192
    public let vocab = 131072
    public let nDelay: Int
    public let bos = 1, eos = 2, streamingPad = 32, streamingWord = 33
    public let leftPadTokens = 32

    let w: SafeTensorsFile
    let ctx: MetalContext
    public let frontend = VoxtralFrontend()
    public let tokenizer: TekkenDecoder

    struct EncLayer { let n1, bq, bv, bo, n2, bdown: Tensor; let wq, wk, wv, wo, wgate, wup, wdown: Tensor }
    struct DecLayer { let n1, n2ada: Tensor; let wq, wk, wv, wo, wgate, wup, wdown: Tensor }
    let conv1W, conv1B, conv2W, conv2B, encNorm, adapt1, adapt2: Tensor
    let enc: [EncLayer]
    let dec: [DecLayer]
    let embed, decNorm: Tensor
    let encInvFreq, decInvFreq: Tensor

    public init(modelDir: URL, weights: String = "model.safetensors", delayMs: Int = 480, ctx: MetalContext = .shared) throws {
        self.ctx = ctx
        nDelay = delayMs / 80
        let wf = try SafeTensorsFile(modelDir.appendingPathComponent(weights), ctx)
        w = wf
        tokenizer = try TekkenDecoder(url: modelDir.appendingPathComponent("tekken.json"))
        let cx = ctx
        func t(_ n: String) throws -> Tensor { try wf.tensor(n) }
        func f32(_ n: String) throws -> Tensor { let v = try wf.floats(n); return Tensor.from(v, shape: [v.count], dtype: .f32, cx) }
        conv1W = try t("audio_tower.embedder.conv1.weight").reshaped([1280, 128 * 3])
        conv1B = try f32("audio_tower.embedder.conv1.bias")
        conv2W = try t("audio_tower.embedder.conv2.weight").reshaped([1280, 1280 * 3])
        conv2B = try f32("audio_tower.embedder.conv2.bias")
        encNorm = try f32("audio_tower.norm.weight")
        adapt1 = try t("multi_modal_projector.linear_1.weight")
        adapt2 = try t("multi_modal_projector.linear_2.weight")
        enc = try (0..<32).map { i in
            let p = "audio_tower.layers.\(i)."
            return EncLayer(n1: try f32(p + "self_attn_layer_norm.weight"), bq: try f32(p + "self_attn.q_proj.bias"),
                            bv: try f32(p + "self_attn.v_proj.bias"), bo: try f32(p + "self_attn.o_proj.bias"),
                            n2: try f32(p + "final_layer_norm.weight"), bdown: try f32(p + "mlp.down_proj.bias"),
                            wq: try t(p + "self_attn.q_proj.weight"), wk: try t(p + "self_attn.k_proj.weight"),
                            wv: try t(p + "self_attn.v_proj.weight"), wo: try t(p + "self_attn.o_proj.weight"),
                            wgate: try t(p + "mlp.gate_proj.weight"), wup: try t(p + "mlp.up_proj.weight"),
                            wdown: try t(p + "mlp.down_proj.weight"))
        }
        // §6.2 delay conditioning, folded into the FFN RMSNorm gamma: gamma' = gamma * (1 + g_l).
        let D = 3072
        var tcond = [Float](repeating: 0, count: D)
        for i in 0..<(D / 2) {
            let invf = Float(exp(-log(10000.0) * Double(i) / Double(D / 2)))
            let e = Float(nDelay) * invf
            tcond[i] = cos(e); tcond[D / 2 + i] = sin(e)
        }
        var decs = [DecLayer]()
        for i in 0..<26 {
            let p = "language_model.model.layers.\(i)."
            let l1 = try wf.floats(p + "ada_rms_norm.linear1.weight"), l2 = try wf.floats(p + "ada_rms_norm.linear2.weight")
            var hmid = [Float](repeating: 0, count: 32)
            for r in 0..<32 { var a: Float = 0; for c in 0..<D { a += l1[r * D + c] * tcond[c] }; hmid[r] = 0.5 * a * (1 + Float(erf(Double(a) / 2.0.squareRoot()))) }
            var gamma = try wf.floats(p + "post_attention_layernorm.weight")
            for r in 0..<D { var a: Float = 0; for c in 0..<32 { a += l2[r * 32 + c] * hmid[c] }; gamma[r] *= (1 + a) }
            decs.append(DecLayer(n1: try f32(p + "input_layernorm.weight"), n2ada: Tensor.from(gamma, shape: [D], dtype: .f32, cx),
                                 wq: try t(p + "self_attn.q_proj.weight"), wk: try t(p + "self_attn.k_proj.weight"),
                                 wv: try t(p + "self_attn.v_proj.weight"), wo: try t(p + "self_attn.o_proj.weight"),
                                 wgate: try t(p + "mlp.gate_proj.weight"), wup: try t(p + "mlp.up_proj.weight"),
                                 wdown: try t(p + "mlp.down_proj.weight")))
        }
        dec = decs
        embed = try t("language_model.model.embed_tokens.weight")
        decNorm = try f32("language_model.model.norm.weight")
        func invFreq(_ d: Int) -> Tensor {
            Tensor.from((0..<(d / 2)).map { Float(1.0 / pow(1e6, Double(2 * $0) / Double(d))) }, shape: [d / 2], dtype: .f32, cx)
        }
        encInvFreq = invFreq(64)
        decInvFreq = invFreq(128)
    }

    /// Prompt (§3): BOS + STREAMING_PAD x (32 + n_delay).
    public var prompt: [Int] { [bos] + Array(repeating: streamingPad, count: leftPadTokens + nDelay) }

    private var snapshot: VoxtralSession.Snapshot?
    private let snapshotLock = NSLock()

    /// The silent-prefix snapshot, computed once per model. Decoder positions 0..30 and encoder frames
    /// 0..123 only see mel frames <= 247, which cover left-pad samples only (frame 247 ends at sample
    /// 39720 < 40960), so this state is identical for every stream.
    func silenceSnapshot() -> VoxtralSession.Snapshot? {
        snapshotLock.lock(); defer { snapshotLock.unlock() }
        if snapshot == nil {
            let tmp = VoxtralSession(model: self)
            tmp.useSnapshot = false
            snapshot = tmp.makeSnapshot(positions: 31)
        }
        return snapshot
    }
}

/// Sliding K/V storage for one attention stack: [capacity, width] rows per layer, capacity = window + slack.
/// Rows hold absolute positions base ..< base + count; when full, the last window-1 rows move to the front
/// (once every `slack` positions; the copy never overlaps because slack >= 1... see reserve).
final class SlidingKV {
    let window: Int, capacity: Int, width: Int
    let k: [Tensor], v: [Tensor]
    var base = 0, count = 0

    init(layers: Int, window: Int, slack: Int, width: Int, ctx: MetalContext) {
        self.window = window; capacity = window + slack; self.width = width
        k = (0..<layers).map { _ in Tensor(ctx, shape: [window + slack, width]) }
        v = (0..<layers).map { _ in Tensor(ctx, shape: [window + slack, width]) }
    }

    /// Makes room for n new rows (shifting inside `s` if needed). Returns the slot of the first new row.
    func reserve(_ s: Step, _ n: Int) -> Int {
        precondition(n < window)
        if count + n > capacity {
            let keep = window - 1
            // Source rows [count-keep, count) and destination [0, keep) overlap when slack < window;
            // copy in blocks of (count - keep) rows front to back, which never reads a row already written.
            let shift = count - keep
            var done = 0
            while done < keep {
                let n = min(shift, keep - done)
                for i in 0..<k.count {
                    s.copy(k[i].rows(done + shift, n), into: k[i].rows(done, n))
                    s.copy(v[i].rows(done + shift, n), into: v[i].rows(done, n))
                }
                done += n
            }
            base += count - keep
            count = keep
        }
        return count
    }

    /// Key ranges (slot coordinates) for n queries at absolute positions pos0 ..< pos0 + n, causal + window.
    func ranges(pos0: Int, n: Int) -> [Int32] {
        var r = [Int32]()
        for i in 0..<n {
            let q = pos0 + i
            r += [Int32(max(q - window + 1, base) - base), Int32(q + 1 - base)]
        }
        return r
    }
}

/// One Voxtral transcription stream. `step` consumes 8 mel frames (80 ms) and emits one token.
public final class VoxtralSession {
    public let model: VoxtralModel
    let ctx: MetalContext
    let encKV: SlidingKV, decKV: SlidingKV
    var encPos = 0, decPos = 0
    let convMelCache: Tensor      // last 2 mel frames (conv1 left context)
    let conv1Cache: Tensor        // last conv1 output frame (conv2 left context)
    public private(set) var ids: [Int] = []       // all decoder input ids so far (prompt + emitted)
    public private(set) var emitted: [Int] = []   // tokens produced after the prompt
    public private(set) var stepTimes: [Double] = []
    public var logitsTap: ((Int, [Float]) -> Void)?
    public var tap: ((String, Tensor) -> Void)?

    public init(model: VoxtralModel) {
        self.model = model
        ctx = model.ctx
        encKV = SlidingKV(layers: model.encLayers, window: model.encWindow, slack: 250, width: model.encHeads * model.encHD, ctx: ctx)
        decKV = SlidingKV(layers: model.decLayers, window: model.decWindow, slack: 512, width: model.decKV * model.decHD, ctx: ctx)
        convMelCache = Tensor(ctx, shape: [2, 128]); memset(convMelCache.contents, 0, convMelCache.byteCount)
        conv1Cache = Tensor(ctx, shape: [1, 1280]); memset(conv1Cache.contents, 0, conv1Cache.byteCount)
    }

    public var text: String { model.tokenizer.decode(emitted) }

    /// §4: mel [n*8, 128] -> audio embeddings [n, 3072] (conv stem with caches, encoder with KV cache, adapter).
    func encodeAudio(_ s: Step, mel: Tensor, out: Tensor) {
        let m = model
        let T = mel.shape[0], n = T / 8, E = T / 2
        // conv1: causal k=3 s=1 over [2 cached ; T] mel frames.
        let melX = Tensor(ctx, shape: [T + 2, 128])
        s.copy(convMelCache, into: melX.rows(0, 2))
        s.copy(mel, into: melX.rows(2, T))
        s.copy(melX.rows(T, 2), into: convMelCache)
        let col1 = Tensor(ctx, shape: [T, 128 * 3])
        s.im2col1d(melX, out: col1, K: 3, stride: 1, pad: 0)
        let c1 = Tensor(ctx, shape: [T + 1, 1280])
        s.copy(conv1Cache, into: c1.rows(0, 1))
        s.gemm(col1, m.conv1W, bias: m.conv1B, out: c1.rows(1, T), act: .geluErf)
        s.copy(c1.rows(T, 1), into: conv1Cache)
        // conv2: k=3 s=2, left pad 1 (the cached frame): out j reads c1 rows 2j, 2j+1, 2j+2 (= frames 2j-1..2j+1).
        let col2 = Tensor(ctx, shape: [E, 1280 * 3])
        s.im2col1d(c1, out: col2, K: 3, stride: 2, pad: 0)
        var x = Tensor(ctx, shape: [E, m.encDim])
        s.gemm(col2, m.conv2W, bias: m.conv2B, out: x, act: .geluErf)
        tap?("conv_stem_out", x)

        let slot = encKV.reserve(s, E)
        let rr = encKV.ranges(pos0: encPos, n: E)
        let ranges = Tensor(ctx, shape: [E, 2], dtype: .i32)
        rr.withUnsafeBytes { ranges.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let W = m.encHeads * m.encHD
        let nrm = Tensor(ctx, shape: [E, m.encDim]), q = Tensor(ctx, shape: [E, W]), att = Tensor(ctx, shape: [E, W])
        let g = Tensor(ctx, shape: [E, m.encFF]), u = Tensor(ctx, shape: [E, m.encFF])
        for (li, L) in m.enc.enumerated() {
            s.rmsNorm(x, gamma: L.n1, out: nrm)
            s.gemm(nrm, L.wq, bias: L.bq, out: q)
            let kr = encKV.k[li].rows(slot, E), vr = encKV.v[li].rows(slot, E)
            s.gemm(nrm, L.wk, out: kr)
            s.gemm(nrm, L.wv, bias: L.bv, out: vr)
            s.rope(q, heads: m.encHeads, headDim: m.encHD, invFreq: m.encInvFreq, pos0: encPos)
            s.rope(kr, heads: m.encHeads, headDim: m.encHD, invFreq: m.encInvFreq, pos0: encPos)
            s.attention(q: q, k: encKV.k[li].rows(0, slot + E), v: encKV.v[li].rows(0, slot + E), ranges: ranges, out: att,
                        heads: m.encHeads, kvHeads: m.encHeads, headDim: m.encHD)
            let x2 = Tensor(ctx, shape: [E, m.encDim])
            s.gemm(att, L.wo, bias: L.bo, out: x2, residual: x)
            s.rmsNorm(x2, gamma: L.n2, out: nrm)
            s.gemm(nrm, L.wgate, out: g)
            s.gemm(nrm, L.wup, out: u)
            s.swiglu(gate: g, up: u, out: g)
            let x3 = Tensor(ctx, shape: [E, m.encDim])
            s.gemm(g, L.wdown, bias: L.bdown, out: x3, residual: x2)
            x = x3
            if li == 0 || li == 1 || li == 15 || li == 31 { tap?(String(format: "enc_layer_%02d_out", li), x) }
        }
        encKV.count = slot + E
        encPos += E
        let xn = Tensor(ctx, shape: [E, m.encDim])
        s.rmsNorm(x, gamma: m.encNorm, out: xn)
        // §5 adapter: stack 4 consecutive frames (row-major reshape), 5120 -> 3072 GELU -> 3072.
        let a1 = Tensor(ctx, shape: [n, 3072])
        s.gemm(xn.reshaped([n, 4 * m.encDim]), m.adapt1, out: a1, act: .geluErf)
        s.gemm(a1, m.adapt2, out: out)
        tap?("adapter_out", out)
    }

    /// §6: decoder over M positions with input ids and audio embeddings; returns argmax of the last position.
    func decode(_ s: Step, ids newIds: [Int], audio: Tensor) -> (Tensor, Tensor) {
        let m = model
        let M = newIds.count
        let idsT = Tensor(ctx, shape: [M], dtype: .i32)
        newIds.map { Int32($0) }.withUnsafeBytes { idsT.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        var x = Tensor(ctx, shape: [M, m.decDim])
        s.embed(table: m.embed, ids: idsT, audio: audio, out: x)
        tap?("dec_in", x)
        let slot = decKV.reserve(s, M)
        let rr = decKV.ranges(pos0: decPos, n: M)
        let ranges = Tensor(ctx, shape: [M, 2], dtype: .i32)
        rr.withUnsafeBytes { ranges.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let QW = m.decHeads * m.decHD
        let nrm = Tensor(ctx, shape: [M, m.decDim]), q = Tensor(ctx, shape: [M, QW]), att = Tensor(ctx, shape: [M, QW])
        let g = Tensor(ctx, shape: [M, m.decFF]), u = Tensor(ctx, shape: [M, m.decFF])
        for (li, L) in m.dec.enumerated() {
            s.rmsNorm(x, gamma: L.n1, out: nrm)
            s.gemm(nrm, L.wq, out: q)
            let kr = decKV.k[li].rows(slot, M), vr = decKV.v[li].rows(slot, M)
            s.gemm(nrm, L.wk, out: kr)
            s.gemm(nrm, L.wv, out: vr)
            s.rope(q, heads: m.decHeads, headDim: m.decHD, invFreq: m.decInvFreq, pos0: decPos)
            s.rope(kr, heads: m.decKV, headDim: m.decHD, invFreq: m.decInvFreq, pos0: decPos)
            s.attention(q: q, k: decKV.k[li].rows(0, slot + M), v: decKV.v[li].rows(0, slot + M), ranges: ranges, out: att,
                        heads: m.decHeads, kvHeads: m.decKV, headDim: m.decHD)
            let x2 = Tensor(ctx, shape: [M, m.decDim])
            s.gemm(att, L.wo, out: x2, residual: x)
            s.rmsNorm(x2, gamma: L.n2ada, out: nrm)
            s.gemm(nrm, L.wgate, out: g)
            s.gemm(nrm, L.wup, out: u)
            s.swiglu(gate: g, up: u, out: g)
            let x3 = Tensor(ctx, shape: [M, m.decDim])
            s.gemm(g, L.wdown, out: x3, residual: x2)
            x = x3
            if li == 0 || li == 1 || li == 12 || li == 25 { tap?(String(format: "dec_layer_%02d_out", li), x) }
        }
        decKV.count = slot + M
        decPos += M
        let last = Tensor(ctx, shape: [1, m.decDim])
        s.rmsNorm(x.rows(M - 1, 1), gamma: m.decNorm, out: last)
        let logits = Tensor(ctx, shape: [1, m.vocab])
        s.gemm(last, m.embed, out: logits)
        let idx = Tensor(ctx, shape: [1], dtype: .i32), val = Tensor(ctx, shape: [1])
        s.argmax(logits, idx: idx, val: val)
        return (idx, logits)
    }

    /// Prefill: prompt positions with mel frames 0 ..< 8 * prompt.count. Returns the first emitted token.
    /// Unless disabled (DICTATOR_VOX_SNAPSHOT=0), the silent prefix (decoder positions 0..30, encoder
    /// frames 0..123, which see only the zero left-pad and fixed prompt tokens) is restored from a
    /// per-model snapshot and only the remaining 8 positions are computed.
    @discardableResult
    public func prefill(mel: [Float]) -> Int {
        let P = model.prompt.count
        precondition(mel.count == 8 * P * 128)
        let t0 = DispatchTime.now().uptimeNanoseconds
        let s = Step(ctx)
        var from = 0
        if useSnapshot, let snap = model.silenceSnapshot() {
            restore(snap, s)
            from = snap.positions
        }
        let n = P - from
        let melT = Tensor.from(Array(mel[(8 * from * 128)...]), shape: [8 * n, 128], dtype: .f32, ctx)
        let audio = Tensor(ctx, shape: [n, model.decDim])
        encodeAudio(s, mel: melT, out: audio)
        let (idx, logits) = decode(s, ids: Array(model.prompt[from...]), audio: audio)
        s.run()
        ids = model.prompt
        let tok = Int(idx.contents.assumingMemoryBound(to: Int32.self)[0])
        logitsTap?(0, logits.toFloats())
        emitted.append(tok)
        stepTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        return tok
    }

    public var useSnapshot = ProcessInfo.processInfo.environment["DICTATOR_VOX_SNAPSHOT"] != "0"

    /// State after the constant silent prefix of the stream.
    public struct Snapshot {
        let positions: Int            // decoder positions covered (31)
        let encFrames: Int            // encoder frames covered (124)
        let encK: [Tensor], encV: [Tensor], decK: [Tensor], decV: [Tensor]
        let convMel: Tensor, conv1: Tensor
    }

    func makeSnapshot(positions: Int) -> Snapshot {
        let s = Step(ctx)
        let melFrames = 8 * positions
        let zeros = [Float](repeating: 0, count: 32 * 1280)
        let mel = model.frontend.frames(zeros, from: 0, count: melFrames)
        let melT = Tensor.from(mel, shape: [melFrames, 128], dtype: .f32, ctx)
        let audio = Tensor(ctx, shape: [positions, model.decDim])
        encodeAudio(s, mel: melT, out: audio)
        _ = decode(s, ids: Array(model.prompt[0..<positions]), audio: audio)
        s.run()
        let E = 4 * positions
        func copyRows(_ ts: [Tensor], _ n: Int) -> [Tensor] {
            ts.map { t in let c = Tensor(ctx, shape: [n, t.shape[1]]); memcpy(c.contents, t.contents, n * t.shape[1] * 4); return c }
        }
        let cm = Tensor(ctx, shape: [2, 128]); memcpy(cm.contents, convMelCache.contents, cm.byteCount)
        let c1 = Tensor(ctx, shape: [1, 1280]); memcpy(c1.contents, conv1Cache.contents, c1.byteCount)
        return Snapshot(positions: positions, encFrames: E, encK: copyRows(encKV.k, E), encV: copyRows(encKV.v, E),
                        decK: copyRows(decKV.k, positions), decV: copyRows(decKV.v, positions), convMel: cm, conv1: c1)
    }

    func restore(_ snap: Snapshot, _ s: Step) {
        for i in 0..<snap.encK.count {
            s.copy(snap.encK[i], into: encKV.k[i].rows(0, snap.encFrames))
            s.copy(snap.encV[i], into: encKV.v[i].rows(0, snap.encFrames))
        }
        for i in 0..<snap.decK.count {
            s.copy(snap.decK[i], into: decKV.k[i].rows(0, snap.positions))
            s.copy(snap.decV[i], into: decKV.v[i].rows(0, snap.positions))
        }
        s.copy(snap.convMel, into: convMelCache)
        s.copy(snap.conv1, into: conv1Cache)
        encKV.count = snap.encFrames; encPos = snap.encFrames
        decKV.count = snap.positions; decPos = snap.positions
    }

    /// GPU seconds of the encoder and decoder command buffers (only when `splitTiming` is set).
    public private(set) var splitTimes: [(enc: Double, dec: Double)] = []
    public var splitTiming = ProcessInfo.processInfo.environment["DICTATOR_SPLIT_TIMING"] == "1"

    /// One 80 ms step: 8 new mel frames and the last emitted token. Returns the next token.
    @discardableResult
    public func step(mel: [Float]) -> Int {
        precondition(mel.count == 8 * 128)
        let t0 = DispatchTime.now().uptimeNanoseconds
        var s = Step(ctx)
        let melT = Tensor.from(mel, shape: [8, 128], dtype: .f32, ctx)
        let audio = Tensor(ctx, shape: [1, model.decDim])
        encodeAudio(s, mel: melT, out: audio)
        var encT = 0.0
        if splitTiming { encT = s.run(); s = Step(ctx) }
        let prev = emitted.last!
        let (idx, logits) = decode(s, ids: [prev], audio: audio)
        let decT = s.run()
        if splitTiming { splitTimes.append((encT, decT)) }
        ids.append(prev)
        let tok = Int(idx.contents.assumingMemoryBound(to: Int32.self)[0])
        logitsTap?(emitted.count, logits.toFloats())
        emitted.append(tok)
        stepTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        return tok
    }

    /// §7.1 offline transcription: pad, mel over the whole padded audio, prefill, then one step per
    /// position until EOS or the length cap P.
    public func transcribeOffline(_ audio: [Float]) -> String {
        let padded = VoxtralSession.padOffline(audio, nDelay: model.nDelay)
        let T = padded.count / 160
        let mel = model.frontend.frames(padded, from: 0, count: T)
        let P = T / 8
        let promptLen = model.prompt.count
        var tok = prefill(mel: Array(mel[0..<(8 * promptLen * 128)]))
        var p = promptLen
        while tok != model.eos && p < P - 1 {
            tok = step(mel: Array(mel[(8 * p * 128)..<(8 * (p + 1) * 128)]))
            p += 1
        }
        return text
    }

    /// §3 offline padding: 32*1280 zeros left; right pad to a multiple of 1280 plus 17*1280 zeros.
    public static func padOffline(_ audio: [Float], nDelay: Int) -> [Float] {
        let left = [Float](repeating: 0, count: 32 * 1280)
        let align = (1280 - audio.count % 1280) % 1280
        let right = [Float](repeating: 0, count: align + (nDelay + 1 + 10) * 1280)
        return left + audio + right
    }
}

/// Live Voxtral: prepends the 2.56 s silence pad, computes mel frames as soon as their window is
/// complete, prefills once the 39 prompt positions have audio, then runs one decoder step per 8 new
/// mel frames. `finish()` appends the reference's right padding and runs until EOS or no more frames.
public final class VoxtralLive: StreamingTranscriber {
    public let session: VoxtralSession
    var padded: [Float]
    var mel: [Float] = []
    var melFrames = 0
    var position = 0                 // next decoder position to run
    var realSamples = 0
    var stopped = false
    public private(set) var emissions: [Emission] = []
    public private(set) var stepWallTimes: [Double] = []

    public init(model: VoxtralModel) {
        session = VoxtralSession(model: model)
        padded = [Float](repeating: 0, count: 32 * 1280)
    }

    public var transcript: String { session.text }
    public func text(of tokens: [Int]) -> String { session.model.tokenizer.decode(tokens) }

    public func push(_ samples: [Float]) {
        realSamples += samples.count
        padded += samples
        run()
    }

    public func finish() {
        let align = (1280 - realSamples % 1280) % 1280
        padded += [Float](repeating: 0, count: align + (session.model.nDelay + 1 + 10) * 1280)
        run()
        stopped = true
    }

    func run() {
        // Frame t needs padded samples [160t - 200, 160t + 200).
        let ready = padded.count >= 200 ? (padded.count - 200) / 160 + 1 : 0
        if ready > melFrames {
            mel += session.model.frontend.frames(padded, from: melFrames, count: ready - melFrames)
            melFrames = ready
        }
        let promptLen = session.model.prompt.count
        while !stopped {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let tok: Int
            if position == 0 {
                guard melFrames >= 8 * promptLen else { return }
                tok = session.prefill(mel: Array(mel[0..<(8 * promptLen * 128)]))
                position = promptLen
            } else {
                guard melFrames >= 8 * (position + 1) else { return }
                tok = session.step(mel: Array(mel[(8 * position * 128)..<(8 * (position + 1) * 128)]))
                position += 1
            }
            let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            stepWallTimes.append(dt)
            emissions.append(Emission(token: tok, samplesSeen: realSamples, computeSeconds: dt))
            if tok == session.model.eos { stopped = true }
        }
    }
}
