import Foundation

/// One streaming step of Nemotron's encoder, ending in the joint's encoder projection. Implemented on
/// the Neural Engine (`NemotronANEEncoder`, Core ML) and on the GPU (`MetalNemotronEncoder` in
/// DictatorEngine). An encoder carries the streaming caches of one stream.
public protocol NemotronEncoder: AnyObject {
    var schedule: NemotronSchedule { get }
    /// One chunk: mel rows [nRows, 128] (pre-encode cache rows included), the first `validRows` real.
    /// Returns the kept frames' joint encoder projection [frames, 640]: the valid frames, at most
    /// `schedule.outFramesPerChunk` of them unless `keepAll` (the final chunk).
    func step(mel: UnsafeBufferPointer<Float>, nRows: Int, validRows: Int, keepAll: Bool) throws -> (proj: [Float], frames: Int)
}

/// Nemotron cache-aware streaming transcription: feeds audio (or mel frames) through an encoder in
/// NeMo's chunk schedule and decodes with the RNN-T decoder on the CPU.
public final class NemotronStreamer {
    public let encoder: NemotronEncoder
    public let decoder: RNNTDecoder
    public let tokenizer: SentencePieceDecoder
    public let frontend: NemoMelFrontend
    public var schedule: NemotronSchedule { encoder.schedule }
    var decState: RNNTDecoder.State
    public private(set) var tokens: [Int] = []
    /// Wall seconds per step, encoder and decoding.
    public private(set) var wallTimes: [Double] = []
    /// The first encoder error, if a step failed (that step emits nothing).
    public private(set) var error: Error?
    /// A short final chunk is zero-extended to 8 mel frames and masked (as NeMo's batched mode does)
    /// instead of dropped as NeMo's batch-1 streaming does (D4). `false` gives the batch-1 schedule.
    public var endPadding = true

    // Live input (see `push`).
    lazy var liveMel = NemoMelStream(frontend)
    var liveBuffer: [Float] = []
    var liveBufferStart = 0
    var liveNextFrame = 0
    var liveSamples = 0
    var liveEmissions: [Emission] = []

    public init(encoder: NemotronEncoder, decoder: RNNTDecoder, tokenizer: SentencePieceDecoder, frontend: NemoMelFrontend) {
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.frontend = frontend
        decState = decoder.initialState()
    }

    public var text: String { tokenizer.decode(tokens) }

    /// One streaming step over mel rows [nRows, melBins] (pre-encode cache frames included),
    /// `validRows` of which are real. Returns the tokens emitted by this chunk.
    @discardableResult
    public func step(melRows: UnsafeBufferPointer<Float>, nRows: Int, validRows: Int, keepAll: Bool) -> [Int] {
        let t0 = DispatchTime.now().uptimeNanoseconds
        var newTokens = [Int]()
        do {
            let (p, frames) = try encoder.step(mel: melRows, nRows: nRows, validRows: validRows, keepAll: keepAll)
            if frames > 0 { decoder.greedy(encProj: p, frames: frames, state: &decState, tokens: &newTokens) }
        } catch {
            if self.error == nil { self.error = error }
        }
        tokens += newTokens
        wallTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        return newTokens
    }

    /// Replays NeMo's CacheAwareStreamingAudioBuffer over a complete mel buffer [T, bins]
    /// (full-utterance features, `validFrames` = NeMo's processed length). Used for parity with
    /// the reference streaming pipeline.
    public func runBuffered(mel: [Float], frames: Int, validFrames: Int) {
        let bins = frontend.nMels
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

    /// Live input: push audio samples as they arrive. Runs every chunk whose mel frames are complete
    /// and returns the tokens emitted. Equivalent to `runBuffered` over the whole utterance.
    @discardableResult
    public func push(samples: [Float]) -> [Int] {
        liveSamples += samples.count
        liveBuffer += liveMel.push(samples)
        return drainChunks(final: false)
    }

    /// Ends the stream (flushes the final partial chunk); returns the tokens it emitted.
    @discardableResult
    public func flush() -> [Int] {
        liveBuffer += liveMel.finish()
        return drainChunks(final: true)
    }

    func drainChunks(final: Bool) -> [Int] {
        let bins = frontend.nMels
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

extension NemotronStreamer: StreamingTranscriber {
    public func push(_ samples: [Float]) { push(samples: samples) }
    public func finish() { flush() }
    public var transcript: String { text }
    public var emissions: [Emission] { liveEmissions }
    public var stepWallTimes: [Double] { wallTimes }
    public func text(of tokens: [Int]) -> String { tokenizer.decode(tokens) }
}
