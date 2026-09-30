import Foundation

/// The two NeMo FastConformer + RNN-T models we run.
public enum NemoModelKind: String {
    case nemotron, parakeet

    public var dirName: String {
        switch self {
        case .nemotron: return "nemotron-speech-streaming-en-0.6b"
        case .parakeet: return "parakeet-unified-en-0.6b"
        }
    }
    public var config: FastConformerConfig { self == .nemotron ? .nemotron : .parakeet }
    /// Parakeet normalizes mel features per bin (`per_feature`); Nemotron does not.
    public var perFeatureNormalize: Bool { self == .parakeet }
}

/// A NeMo FastConformer + RNN-T model (Nemotron Speech Streaming or Parakeet Unified).
public final class NemotronModel {
    public let kind: NemoModelKind
    public let encoder: FastConformerEncoder
    public let decoder: RNNTDecoder
    public let frontend: NemoMelFrontend
    public let tokenizer: SentencePieceDecoder
    let jointEncW: Tensor, jointEncB: Tensor
    let weights: SafeTensorsFile

    /// Weight file for a precision: fp32 is the extracted NeMo checkpoint; others are produced by
    /// `dictate-cli convert`.
    public static func weightsURL(modelDir: URL, precision: String) -> URL {
        precision == "f32" ? modelDir.appendingPathComponent("nemo_extracted/weights.safetensors")
                           : modelDir.appendingPathComponent("dictator-\(precision).safetensors")
    }

    public init(modelDir: URL, kind: NemoModelKind = .nemotron, precision: String = "f32") throws {
        self.kind = kind
        let dir = modelDir.appendingPathComponent("nemo_extracted")
        weights = try SafeTensorsFile(NemotronModel.weightsURL(modelDir: modelDir, precision: precision))
        encoder = try FastConformerEncoder(weights: weights, config: kind.config)
        decoder = try RNNTDecoder(weights: weights)
        frontend = NemoMelFrontend(filterbank: try weights.floats("preprocessor.featurizer.fb"),
                                   window: try weights.floats("preprocessor.featurizer.window"))
        tokenizer = try SentencePieceDecoder(vocabFile: dir.appendingPathComponent("tokenizer.vocab"))
        jointEncW = try weights.tensor("joint.enc.weight")
        jointEncB = try weights.tensor("joint.enc.bias")
    }
}

/// NeMo cache-aware streaming schedule for a causal dw_striding x8 encoder with `lookahead` frames
/// of right context ([70, lookahead] attention): the first chunk is 1 + 8*lookahead mel frames, later
/// chunks 8 + 8*lookahead frames, each prefixed with the previous 9 mel frames (pre-encode cache)
/// from which the first 2 subsampled frames are dropped.
public struct NemotronSchedule {
    public let lookahead: Int
    public let leftContext = 70
    public let preEncodeCache = 9
    public let dropExtra = 2
    public var firstChunk: Int { 1 + 8 * lookahead }
    public var chunk: Int { 8 + 8 * lookahead }
    public var outFramesPerChunk: Int { lookahead + 1 }
    public var chunkMs: Int { 80 * (lookahead + 1) }
    public init(lookahead: Int) { self.lookahead = lookahead }
}

/// One streaming transcription session. Feed mel frames (or audio) in order; emits RNN-T tokens
/// per chunk. All GPU work for a chunk is one command buffer.
public final class NemotronStream {
    public let model: NemotronModel
    public let schedule: NemotronSchedule
    let ws: FastConformerEncoder.Workspace
    let state: FastConformerEncoder.StreamState
    var decState: RNNTDecoder.State
    public private(set) var tokens: [Int] = []
    let melIn: Tensor
    let encOut: Tensor
    let proj: Tensor
    /// GPU seconds per step (for benchmarking).
    public private(set) var gpuTimes: [Double] = []
    /// Wall seconds per step including CPU decoding.
    public private(set) var wallTimes: [Double] = []
    // Live-input state (see `push`).
    lazy var liveMel = NemoMelStream(model.frontend)
    var liveBuffer: [Float] = []
    var liveBufferStart = 0
    var liveNextFrame = 0
    var liveSamples = 0
    public var endPadding = ProcessInfo.processInfo.environment["DICTATOR_END_PAD"] != "0"
    var liveEmissions: [Emission] = []

    /// When set, the encoder step runs on the Neural Engine through Core ML instead of Metal.
    public var ane: NemotronANEEncoder?

    /// Receives the kept encoder output rows of each chunk (parity testing).
    public var encTap: (([Float]) -> Void)?

    public init(model: NemotronModel, lookahead: Int) {
        self.model = model
        schedule = NemotronSchedule(lookahead: lookahead)
        let maxMel = schedule.preEncodeCache + schedule.chunk
        let maxT = model.encoder.subsampledLength(maxMel)
        ws = model.encoder.makeWorkspace(maxT: maxT, maxMel: maxMel)
        state = model.encoder.makeStreamState(leftContext: schedule.leftContext, maxChunk: maxT)
        decState = model.decoder.initialState()
        melIn = Tensor(shape: [maxMel, model.encoder.config.melBins])
        encOut = Tensor(shape: [maxT, model.encoder.config.dModel])
        proj = Tensor(shape: [maxT, model.decoder.hidden])
    }

    public var text: String { model.tokenizer.decode(tokens) }

    /// One streaming step over mel rows [nRows, melBins] (pre-encode cache frames included),
    /// `validRows` of which are real. Returns the tokens emitted by this chunk.
    @discardableResult
    public func step(melRows: UnsafeBufferPointer<Float>, nRows: Int, validRows: Int, keepAll: Bool) -> [Int] {
        let t0 = DispatchTime.now().uptimeNanoseconds
        if let ane = ane {
            let (p, validT) = try! ane.step(mel: melRows, nRows: nRows, validRows: validRows)
            let keep = keepAll ? validT : min(validT, schedule.outFramesPerChunk)
            var newTokens = [Int]()
            if keep > 0 { model.decoder.greedy(encProj: Array(p.prefix(keep * model.decoder.hidden)), frames: keep, state: &decState, tokens: &newTokens) }
            tokens += newTokens
            wallTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            return newTokens
        }
        let bins = model.encoder.config.melBins
        let chunkMel = melIn.rows(0, nRows)
        memcpy(chunkMel.contents, melRows.baseAddress!, nRows * bins * 4)
        let s = Step()
        let drop = state.step == 0 ? 0 : schedule.dropExtra
        let (T, validT) = model.encoder.encodeStreamStep(s, ws, state: state, melChunk: chunkMel, validMel: validRows,
                                                         drop: drop, out: encOut)
        // NeMo keeps only valid_out_len frames except on the final chunk.
        let keep = keepAll ? validT : min(validT, schedule.outFramesPerChunk)
        if keep > 0 {
            s.gemm(encOut.rows(0, keep), model.jointEncW, bias: model.jointEncB, out: proj.rows(0, keep))
        }
        gpuTimes.append(s.run())
        _ = T
        if keep > 0, let tap = encTap { tap(encOut.rows(0, keep).toFloats()) }
        var newTokens = [Int]()
        if keep > 0 {
            let p = proj.rows(0, keep).toFloats()
            model.decoder.greedy(encProj: p, frames: keep, state: &decState, tokens: &newTokens)
        }
        tokens += newTokens
        wallTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        return newTokens
    }

    /// Replays NeMo's CacheAwareStreamingAudioBuffer over a complete mel buffer [T, bins]
    /// (full-utterance features, `validFrames` = NeMo's processed length). Used for parity with
    /// the reference streaming pipeline.
    public func runBuffered(mel: [Float], frames: Int, validFrames: Int) {
        let bins = model.encoder.config.melBins
        var idx = 0
        mel.withUnsafeBufferPointer { m in
            while idx < frames {
                let first = idx == 0
                let size = first ? schedule.firstChunk : schedule.chunk
                let end = min(idx + size, frames)
                let chunkLen = end - idx
                if chunkLen < (first ? 1 : 8) { return }
                let cacheStart = first ? idx : max(0, idx - schedule.preEncodeCache)
                let added = idx - cacheStart
                // (Zero-padding of a short pre-encode cache never happens after the first chunk,
                // since the first chunk is already >= 9 frames.)
                let rows = added + chunkLen
                let valid = max(0, min(validFrames - idx + added, rows))
                let slice = UnsafeBufferPointer(rebasing: m[(cacheStart * bins)..<((cacheStart + rows) * bins)])
                idx += size
                step(melRows: slice, nRows: rows, validRows: valid, keepAll: idx >= frames)
            }
        }
    }
}

extension NemotronStream {
    /// Live input: push audio samples as they arrive. Runs every chunk whose mel frames are complete
    /// and returns the tokens emitted. Equivalent to `runBuffered` over the whole utterance.
    public func push(samples: [Float]) -> [Int] {
        liveSamples += samples.count
        let mel = liveMel.push(samples)
        liveBuffer += mel
        return drainChunks(final: false)
    }

    /// Ends the stream (flushes the final partial chunk); returns the tokens it emitted.
    /// With `endPadding` (default; DICTATOR_END_PAD=0 disables) a final tail shorter than 8 mel frames is
    /// zero-extended with the extra rows masked as invalid (as NeMo's batched mode does through lengths)
    /// instead of being dropped. NeMo's batch-1 streaming drops it, losing the last word on ~2-3% of
    /// utterances at 160 ms (see D4).
    public func flush() -> [Int] {
        let out = [Int]()
        liveBuffer += liveMel.finish()
        return out + drainChunks(final: true)
    }

    func drainChunks(final: Bool) -> [Int] {
        let bins = model.encoder.config.melBins
        let have = liveBuffer.count / bins + liveBufferStart
        var out = [Int]()
        while true {
            let idx = liveNextFrame
            let first = idx == 0
            let size = first ? schedule.firstChunk : schedule.chunk
            let end = idx + size
            if end > have && !final { break }
            let chunkEnd = min(end, have)
            let chunkLen = chunkEnd - idx
            let minLen = first ? 1 : 8
            if chunkLen <= 0 || (chunkLen < minLen && !(final && endPadding)) { break }
            let cacheStart = first ? idx : max(0, idx - schedule.preEncodeCache)
            let rows = chunkEnd - cacheStart
            let padRows = max(0, minLen - chunkLen)          // only for a short final tail
            let isLast = final && end >= have
            let a = (cacheStart - liveBufferStart) * bins
            let t0 = DispatchTime.now().uptimeNanoseconds
            var toks = [Int]()
            if padRows > 0 {
                var m = Array(liveBuffer[a..<(a + rows * bins)])
                m += [Float](repeating: 0, count: padRows * bins)
                m.withUnsafeBufferPointer { toks = step(melRows: $0, nRows: rows + padRows, validRows: rows, keepAll: true) }
            } else {
                liveBuffer.withUnsafeBufferPointer { m in
                    let slice = UnsafeBufferPointer(rebasing: m[a..<(a + rows * bins)])
                    toks = step(melRows: slice, nRows: rows, validRows: rows, keepAll: isLast)
                }
            }
            let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            liveEmissions += toks.map { Emission(token: $0, samplesSeen: liveSamples, computeSeconds: dt) }
            out += toks
            liveNextFrame = end
            // Keep only the pre-encode cache window.
            let keepFrom = liveNextFrame - schedule.preEncodeCache
            if keepFrom > liveBufferStart {
                liveBuffer.removeFirst(min(liveBuffer.count, (keepFrom - liveBufferStart) * bins))
                liveBufferStart = keepFrom
            }
            if isLast { break }
        }
        return out
    }
}

extension NemotronStream: StreamingTranscriber {
    public func push(_ samples: [Float]) { _ = push(samples: samples) }
    public func finish() { _ = flush() }
    public var transcript: String { text }
    public var emissions: [Emission] { liveEmissions }
    public var stepWallTimes: [Double] { wallTimes }
    public func text(of tokens: [Int]) -> String { model.tokenizer.decode(tokens) }
}
