import Foundation

/// Parakeet's buffer schedule in samples: [left 5.6 s | chunk 160 ms | right 400 ms]; 1280 samples per encoder frame.
enum ParakeetBuffer {
    static let leftSamples = 89600, chunkSamples = 2560, rightSamples = 6400, frameSamples = 1280
}

/// Parakeet Unified 0.6B buffered streaming (NeMo `speech_to_text_streaming_infer_rnnt.py` with
/// left 5.6 s / chunk 0.16 s / right 0.40 s and the [70, 2, 5] chunked_limited_with_rc mask).
///
/// Stage A ("exact"): every step re-runs the preprocessor (with per-buffer normalization) and the
/// full encoder over the whole buffer, then decodes the chunk's encoder frames. This reproduces the
/// reference and is the quality baseline for the cheaper variants. `live()` runs it.
public final class ParakeetStream {
    public let model: NemotronModel
    var decState: RNNTDecoder.State
    public private(set) var tokens: [Int] = []

    public init(model: NemotronModel) {
        precondition(model.kind == .parakeet)
        self.model = model
        decState = model.decoder.initialState()
    }

    public var text: String { model.tokenizer.decode(tokens) }

    /// One buffered step over `buffer` (left context + chunk + right context samples).
    func step(audio buffer: [Float], leftLen: Int, isLast: Bool) {
        let fe = model.frontend
        var mel = fe.features(buffer)
        let frames = mel.count / fe.nMels
        let valid = fe.validFrames(samples: buffer.count)
        fe.normalizePerFeature(&mel, T: frames, valid: valid)
        let enc = model.encoder
        let out = enc.forwardOffline(mel: mel, melFrames: frames, validMel: valid,
                                     mask: .chunkedLimitedRC(left: 70, chunk: 2, right: 5))
        var validT = valid
        for _ in 0..<3 { validT = (validT - 1) / 2 + 1 }
        let first = leftLen / ParakeetBuffer.frameSamples
        let outLen = isLast ? max(0, validT - first) : ParakeetBuffer.chunkSamples / ParakeetBuffer.frameSamples
        guard outLen > 0 else { return }
        let proj = Tensor(shape: [outLen, model.decoder.hidden])
        let s = Step()
        s.gemm(out.rows(first, outLen), model.jointEncW, bias: model.jointEncB, out: proj)
        s.run()
        var newTokens = [Int]()
        model.decoder.greedy(encProj: proj.toFloats(), frames: outLen, state: &decState, tokens: &newTokens)
        tokens += newTokens
    }
}

/// Parakeet Stage B: frozen-prefix caching. Each step re-encodes only the newest region
/// (chunk + right context + `horizon` extra frames); older frames keep the per-layer K/V and conv
/// inputs computed when they left the region. Mel features and normalization for the region come
/// from the same buffer as the reference, so the only difference from Stage A is the frozen context.
/// `live()` runs it.
public final class ParakeetCachedStream {
    public let model: NemotronModel
    public let horizon: Int
    let frameSamples = ParakeetBuffer.frameSamples
    let chunkFrames = 2, rightFrames = 5
    let ws: FastConformerEncoder.Workspace
    let cache: FastConformerEncoder.RegionCache
    var frozen = 0                 // global encoder frames < frozen are in the cache
    var decState: RNNTDecoder.State
    let regionOut: Tensor
    public private(set) var tokens: [Int] = []

    public init(model: NemotronModel, horizon: Int) {
        precondition(model.kind == .parakeet)
        self.model = model
        self.horizon = horizon
        let maxRegion = chunkFrames + rightFrames + horizon + 2
        let maxMel = 8 * (maxRegion + 2) + 16
        ws = model.encoder.makeWorkspace(maxT: maxRegion + 3, maxMel: maxMel)
        cache = model.encoder.makeRegionCache(capacity: 72, maxRegion: maxRegion)
        decState = model.decoder.initialState()
        regionOut = Tensor(shape: [maxRegion, model.encoder.config.dModel])
    }

    public var text: String { model.tokenizer.decode(tokens) }

    func step(buffer: [Float], bufStart: Int, leftLen: Int, isLast: Bool) {
        let fe = model.frontend, enc = model.encoder
        let bins = fe.nMels
        var mel = fe.features(buffer)
        let frames = mel.count / bins
        let valid = fe.validFrames(samples: buffer.count)
        fe.normalizePerFeature(&mel, T: frames, valid: valid)
        let g0 = bufStart / frameSamples                 // global index of buffer encoder frame 0
        var validE = valid
        for _ in 0..<3 { validE = (validE - 1) / 2 + 1 }
        let gE = g0 + validE                             // global end of valid encoder frames
        let F = max(frozen, g0)                          // region start (global)
        let R = gE - F
        guard R > 0 else { return }

        // Subsample a mel window with a 2-frame left margin so region frames see real left context.
        let margin = min(2, F - g0)
        let melStart = 8 * (F - margin - g0)
        let melRows = frames - melStart
        let melT = Tensor(shape: [melRows, bins])
        mel.withUnsafeBufferPointer { m in memcpy(melT.contents, m.baseAddress! + melStart * bins, melRows * bins * 4) }
        let s = Step()
        let Tsub = enc.subsampledLength(melRows)
        let subOut = ws.y.rows(0, Tsub)
        enc.encodeSubsampling(s, ws, mel: melT, validMel: valid - melStart, out: subOut)
        let xRegion = ws.r1.rows(0, R)
        s.copy(subOut.rows(margin, R), into: xRegion)

        // Advance the frozen frontier so the next region is chunk + right + horizon frames.
        let steadyRegion = chunkFrames + rightFrames + horizon
        let nextFrozen = isLast ? F : max(F, min(gE, gE + chunkFrames - steadyRegion))
        let advance = nextFrozen - F

        // Mask ranges over keys [cached ; region] in global coordinates.
        let keyStart = F - cache.cached
        var rr = [Int32]()
        for q in F..<gE {
            let cs = (q / chunkFrames) * chunkFrames
            let lo = max(cs - 70, keyStart), hi = min(cs + chunkFrames + rightFrames, gE)
            rr += [Int32(lo - keyStart), Int32(hi - keyStart)]
        }
        let ranges = ws.ranges.rows(0, R)
        rr.withUnsafeBytes { ranges.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        enc.encodeRegion(s, ws, cache: cache, x: xRegion, ranges: ranges, advance: advance, out: regionOut)
        frozen = nextFrozen

        // Decode the chunk frames exactly as the reference schedule does.
        let first = g0 + leftLen / frameSamples
        let outLen = isLast ? max(0, gE - first) : chunkFrames
        let proj = Tensor(shape: [max(outLen, 1), model.decoder.hidden])
        if outLen > 0 {
            s.gemm(regionOut.rows(first - F, outLen), model.jointEncW, bias: model.jointEncB, out: proj.rows(0, outLen))
        }
        s.run()
        if outLen > 0 {
            var newTokens = [Int]()
            model.decoder.greedy(encProj: proj.rows(0, outLen).toFloats(), frames: outLen, state: &decState, tokens: &newTokens)
            tokens += newTokens
        }
    }
}

/// Live driver for Parakeet's buffered schedule (exact or cached), with the reference's buffer bookkeeping
/// (StreamingBatchedAudioBuffer / ContextSize). A step runs once audio beyond its right edge has arrived (so
/// it is known not to be the last); `finish()` runs the remaining steps with the reference's last-chunk
/// handling. The tokens do not depend on how the audio is split into pushes: pushing a whole file and
/// finishing is the reference's buffered run.
public final class ParakeetLive: StreamingTranscriber {
    public typealias StepFn = (_ buffer: [Float], _ bufStart: Int, _ leftLen: Int, _ isLast: Bool) -> [Int]
    let stepFn: StepFn
    let decode: ([Int]) -> String
    let leftSamples = ParakeetBuffer.leftSamples, chunkSamples = ParakeetBuffer.chunkSamples
    var audio: [Float] = []
    var audioStart = 0            // absolute index of audio[0] (older samples are dropped)
    var leftSample = 0, rightSample = ParakeetBuffer.chunkSamples + ParakeetBuffer.rightSamples
    var bufStart = 0, bufEnd = 0, leftLen = 0, chunkLen = 0, rightLen = 0
    var done = false
    public private(set) var tokens: [Int] = []
    public private(set) var emissions: [Emission] = []
    public private(set) var stepWallTimes: [Double] = []

    public init(step: @escaping StepFn, decode: @escaping ([Int]) -> String) {
        stepFn = step
        self.decode = decode
    }

    public var transcript: String { decode(tokens) }

    /// A whole utterance at once (the reference's buffered run).
    public func run(_ audio: [Float]) {
        push(audio)
        finish()
    }
    public func text(of tokens: [Int]) -> String { decode(tokens) }
    var received: Int { audioStart + audio.count }

    public func push(_ samples: [Float]) {
        audio += samples
        while !done && received > rightSample { runStep(isLast: false, n: rightSample) }
    }

    public func finish() {
        let n = received
        rightSample = min(rightSample, n)
        while !done && leftSample < n { runStep(isLast: rightSample >= n, n: n) }
        done = true
    }

    func runStep(isLast: Bool, n: Int) {
        let added = rightSample - leftSample
        bufEnd += added
        leftLen += chunkLen
        let right = rightLen + added
        if isLast { chunkLen = right; rightLen = 0 } else { chunkLen = min(chunkSamples, right); rightLen = right - chunkLen }
        if leftLen > leftSamples { bufStart += leftLen - leftSamples; leftLen = leftSamples }
        let buffer = Array(audio[(bufStart - audioStart)..<(bufEnd - audioStart)])
        let t0 = DispatchTime.now().uptimeNanoseconds
        let toks = stepFn(buffer, bufStart, leftLen, isLast)
        let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        stepWallTimes.append(dt)
        tokens += toks
        emissions += toks.map { Emission(token: $0, samplesSeen: received, computeSeconds: dt) }
        leftSample = rightSample
        rightSample = min(rightSample + chunkSamples, isLast ? n : Int.max)
        if isLast { done = true }
        // Drop audio that can no longer be part of a buffer.
        if bufStart - audioStart > 16000 { audio.removeFirst(bufStart - audioStart); audioStart = bufStart }
    }
}

extension ParakeetStream {
    /// A live transcriber running the exact (Stage A) algorithm.
    public func live() -> ParakeetLive {
        ParakeetLive(step: { buf, _, left, last in
            let before = self.tokens.count
            self.step(audio: buf, leftLen: left, isLast: last)
            return Array(self.tokens[before...])
        }, decode: { self.model.tokenizer.decode($0) })
    }
}

extension ParakeetCachedStream {
    /// A live transcriber running frozen-prefix caching (Stage B).
    public func live() -> ParakeetLive {
        ParakeetLive(step: { buf, start, left, last in
            let before = self.tokens.count
            self.step(buffer: buf, bufStart: start, leftLen: left, isLast: last)
            return Array(self.tokens[before...])
        }, decode: { self.model.tokenizer.decode($0) })
    }
}
