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
