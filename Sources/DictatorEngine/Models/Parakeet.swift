import Foundation

/// Parakeet Unified 0.6B buffered streaming (NeMo `speech_to_text_streaming_infer_rnnt.py` with
/// left 5.6 s / chunk 0.16 s / right 0.40 s and the [70, 2, 5] chunked_limited_with_rc mask).
///
/// Stage A ("exact"): every step re-runs the preprocessor (with per-buffer normalization) and the
/// full encoder over the whole buffer, then decodes the chunk's encoder frames. This reproduces the
/// reference and is the quality baseline for the cheaper variants.
public final class ParakeetStream {
    public let model: NemotronModel
    public let leftSamples = 89600     // 70 encoder frames
    public let chunkSamples = 2560     // 2 encoder frames (160 ms)
    public let rightSamples = 6400     // 5 encoder frames (400 ms)
    let frameSamples = 1280
    var decState: RNNTDecoder.State
    public private(set) var tokens: [Int] = []
    public private(set) var gpuTimes: [Double] = []
    public private(set) var wallTimes: [Double] = []

    public init(model: NemotronModel) {
        precondition(model.kind == .parakeet)
        self.model = model
        decState = model.decoder.initialState()
    }

    public var text: String { model.tokenizer.decode(tokens) }

    /// Runs the reference buffered loop over a complete utterance.
    public func runBuffered(audio: [Float]) {
        let n = audio.count
        var leftSample = 0
        var rightSample = min(chunkSamples + rightSamples, n)
        // Buffer bookkeeping (StreamingBatchedAudioBuffer / ContextSize): [left | chunk | right].
        var bufStart = 0            // absolute sample index of buffer[0]
        var bufEnd = 0              // absolute end (exclusive)
        var leftLen = 0, chunkLen = 0, rightLen = 0
        while leftSample < n {
            let isLast = rightSample >= n
            let added = rightSample - leftSample
            bufEnd += added
            if bufEnd - bufStart == added && leftLen == 0 && chunkLen == 0 {
                // First step: the chunk comes from the new samples, the rest is right context.
                if isLast { chunkLen = added; rightLen = 0 } else { chunkLen = min(chunkSamples, added); rightLen = added - chunkLen }
            } else {
                leftLen += chunkLen
                let right = rightLen + added
                if isLast { chunkLen = right; rightLen = 0 } else { chunkLen = min(chunkSamples, right); rightLen = right - chunkLen }
            }
            if leftLen > leftSamples {
                let trim = leftLen - leftSamples
                bufStart += trim
                leftLen = leftSamples
            }
            step(audio: Array(audio[bufStart..<bufEnd]), leftLen: leftLen, isLast: isLast)
            leftSample = rightSample
            rightSample = min(rightSample + chunkSamples, n)
        }
    }

    /// One buffered step over `buffer` (left context + chunk + right context samples).
    func step(audio buffer: [Float], leftLen: Int, isLast: Bool) {
        let t0 = DispatchTime.now().uptimeNanoseconds
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
        let first = leftLen / frameSamples
        let outLen = isLast ? max(0, validT - first) : chunkSamples / frameSamples
        guard outLen > 0 else { return }
        let proj = Tensor(shape: [outLen, model.decoder.hidden])
        let s = Step()
        s.gemm(out.rows(first, outLen), model.jointEncW, bias: model.jointEncB, out: proj)
        gpuTimes.append(s.run())
        var newTokens = [Int]()
        model.decoder.greedy(encProj: proj.toFloats(), frames: outLen, state: &decState, tokens: &newTokens)
        tokens += newTokens
        wallTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
    }
}

/// Parakeet Stage B: frozen-prefix caching. Each step re-encodes only the newest region
/// (chunk + right context + `horizon` extra frames); older frames keep the per-layer K/V and conv
/// inputs computed when they left the region. Mel features and normalization for the region come
/// from the same buffer as the reference, so the only difference from Stage A is the frozen context.
public final class ParakeetCachedStream {
    public let model: NemotronModel
    public let horizon: Int
    let leftSamples = 89600, chunkSamples = 2560, rightSamples = 6400, frameSamples = 1280
    let chunkFrames = 2, rightFrames = 5
    let ws: FastConformerEncoder.Workspace
    let cache: FastConformerEncoder.RegionCache
    var frozen = 0                 // global encoder frames < frozen are in the cache
    var decState: RNNTDecoder.State
    let regionOut: Tensor
    public private(set) var tokens: [Int] = []
    public private(set) var wallTimes: [Double] = []
    public private(set) var gpuTimes: [Double] = []

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

    /// Same outer loop (buffer bookkeeping and decode schedule) as `ParakeetStream.runBuffered`.
    public func runBuffered(audio: [Float]) {
        let n = audio.count
        var leftSample = 0
        var rightSample = min(chunkSamples + rightSamples, n)
        var bufStart = 0, bufEnd = 0, leftLen = 0, chunkLen = 0, rightLen = 0
        while leftSample < n {
            let isLast = rightSample >= n
            let added = rightSample - leftSample
            bufEnd += added
            leftLen += chunkLen
            let right = rightLen + added
            if isLast { chunkLen = right; rightLen = 0 } else { chunkLen = min(chunkSamples, right); rightLen = right - chunkLen }
            if leftLen > leftSamples { bufStart += leftLen - leftSamples; leftLen = leftSamples }
            step(buffer: Array(audio[bufStart..<bufEnd]), bufStart: bufStart, leftLen: leftLen, isLast: isLast)
            leftSample = rightSample
            rightSample = min(rightSample + chunkSamples, n)
        }
    }

    func step(buffer: [Float], bufStart: Int, leftLen: Int, isLast: Bool) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let fe = model.frontend, enc = model.encoder
        let bins = fe.nMels
        var mel = fe.features(buffer)
        let frames = mel.count / bins
        let valid = fe.validFrames(samples: buffer.count)
        fe.normalizePerFeature(&mel, T: frames, valid: valid)
        let g0 = bufStart / frameSamples                 // global index of buffer encoder frame 0
        let E = enc.subsampledLength(frames)
        var validE = valid
        for _ in 0..<3 { validE = (validE - 1) / 2 + 1 }
        let gE = g0 + validE                             // global end of valid encoder frames
        let F = max(frozen, g0)                          // region start (global)
        let R = gE - F
        guard R > 0 else { return }
        _ = E

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
        let outLen = isLast ? max(0, gE - first) : chunkSamples / frameSamples
        let proj = Tensor(shape: [max(outLen, 1), model.decoder.hidden])
        if outLen > 0 {
            s.gemm(regionOut.rows(first - F, outLen), model.jointEncW, bias: model.jointEncB, out: proj.rows(0, outLen))
        }
        gpuTimes.append(s.run())
        if outLen > 0 {
            var newTokens = [Int]()
            model.decoder.greedy(encProj: proj.rows(0, outLen).toFloats(), frames: outLen, state: &decState, tokens: &newTokens)
            tokens += newTokens
        }
        wallTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
    }
}
