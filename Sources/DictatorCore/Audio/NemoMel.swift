import Accelerate
import Foundation

/// NeMo `FilterbankFeatures` log-mel frontend, reproduced for inference (dither off):
///   preemphasis (0.97, first sample kept) -> centered STFT (n_fft 512, hop 160, 400-sample window
///   zero-padded to 512, constant zero padding of n_fft/2 on both sides) -> power spectrum ->
///   mel filterbank (taken from the checkpoint) -> log(x + 2^-24).
/// The optional per-feature normalization is applied by the caller when a model needs it.
///
/// The DFT is a pair of matrix products with precomputed cos/sin tables, run through Accelerate.
public final class NemoMelFrontend {
    public let nFFT = 512
    public let hop = 160
    public let winLength = 400
    public let nMels: Int
    public let preemph: Float = 0.97
    public let logGuard: Float = Float(pow(2.0, -24.0))
    let nBins = 257
    let fb: [Float]          // [nMels, nBins]
    let cosT: [Float]        // [nBins, winLength] (window folded in)
    let sinT: [Float]

    public init(filterbank: [Float], window: [Float]) {
        precondition(window.count == winLength)
        nMels = filterbank.count / nBins
        fb = filterbank
        // torch.stft pads the win_length window with zeros to n_fft, centered: offset (512-400)/2.
        let off = (nFFT - winLength) / 2
        var c = [Float](repeating: 0, count: nBins * winLength)
        var s = [Float](repeating: 0, count: nBins * winLength)
        for k in 0..<nBins {
            for n in 0..<winLength {
                let phase = 2 * Double.pi * Double(k) * Double(n + off) / Double(nFFT)
                c[k * winLength + n] = Float(cos(phase)) * window[n]
                s[k * winLength + n] = Float(-sin(phase)) * window[n]
            }
        }
        cosT = c
        sinT = s
    }

    /// Number of valid frames NeMo reports for `samples` input samples (the extra final frame is zeroed).
    public func validFrames(samples: Int) -> Int { samples / hop }

    /// Full-utterance features, time-major [T, nMels] with T = samples / hop + 1 (last frame zero).
    public func features(_ audio: [Float]) -> [Float] {
        let n = audio.count
        var x = [Float](repeating: 0, count: n)
        if n > 0 {
            x[0] = audio[0]
            for i in 1..<n { x[i] = audio[i] - preemph * audio[i - 1] }
        }
        let T = n / hop + 1
        let valid = validFrames(samples: n)
        // Frame t covers padded samples [t*hop, t*hop + nFFT); the window occupies [off, off+400).
        let pad = nFFT / 2
        let off = (nFFT - winLength) / 2
        var frames = [Float](repeating: 0, count: T * winLength)
        for t in 0..<T {
            let start = t * hop - pad + off
            for i in 0..<winLength {
                let idx = start + i
                if idx >= 0 && idx < n { frames[t * winLength + i] = x[idx] }
            }
        }
        return featuresFromFrames(frames, T: T, valid: valid)
    }

    /// NeMo `per_feature` normalization in place over time-major mel [T, nMels]: per bin, mean and
    /// unbiased std (+1e-5 on the std) over the first `valid` frames; frames >= valid are set to 0.
    public func normalizePerFeature(_ mel: inout [Float], T: Int, valid: Int) {
        for m in 0..<nMels {
            var sum = 0.0
            for t in 0..<valid { sum += Double(mel[t * nMels + m]) }
            let mean = valid > 0 ? sum / Double(valid) : 0
            var ss = 0.0
            for t in 0..<valid { let d = Double(mel[t * nMels + m]) - mean; ss += d * d }
            var std = valid > 1 ? (ss / Double(valid - 1)).squareRoot() : 0
            if std.isNaN { std = 0 }
            std += 1e-5
            for t in 0..<T { mel[t * nMels + m] = t < valid ? Float((Double(mel[t * nMels + m]) - mean) / std) : 0 }
        }
    }

    /// frames [T, winLength] of preemphasized samples -> log-mel [T, nMels]; frames >= valid are zero.
    func featuresFromFrames(_ frames: [Float], T: Int, valid: Int) -> [Float] {
        var re = [Float](repeating: 0, count: T * nBins)
        var im = [Float](repeating: 0, count: T * nBins)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(T), Int32(nBins), Int32(winLength), 1,
                    frames, Int32(winLength), cosT, Int32(winLength), 0, &re, Int32(nBins))
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(T), Int32(nBins), Int32(winLength), 1,
                    frames, Int32(winLength), sinT, Int32(winLength), 0, &im, Int32(nBins))
        var power = [Float](repeating: 0, count: T * nBins)
        for i in 0..<(T * nBins) {
            let mag = (re[i] * re[i] + im[i] * im[i]).squareRoot()
            power[i] = mag * mag
        }
        var mel = [Float](repeating: 0, count: T * nMels)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(T), Int32(nMels), Int32(nBins), 1,
                    power, Int32(nBins), fb, Int32(nBins), 0, &mel, Int32(nMels))
        for t in 0..<T {
            for m in 0..<nMels {
                mel[t * nMels + m] = t < valid ? log(mel[t * nMels + m] + logGuard) : 0
            }
        }
        return mel
    }
}

/// Incremental version of `NemoMelFrontend.features`: push audio in any sizes; frames are produced as
/// soon as their window is complete, and `finish()` emits the tail exactly like the full-utterance
/// computation (zero padding on the right, last frame zeroed). Output is identical to `features`.
public final class NemoMelStream {
    public let fe: NemoMelFrontend
    var audio: [Float] = []          // raw samples received so far
    public private(set) var frames = 0   // mel frames emitted
    public private(set) var finished = false

    public init(_ fe: NemoMelFrontend) { self.fe = fe }

    /// Frame t uses preemphasized samples [160t - 200, 160t + 200) (window of 400 centered in the
    /// zero-padded 512-sample frame), so it is ready once sample 160t + 200 - 1 has arrived.
    func readyFrames() -> Int {
        let n = audio.count
        return n >= 200 ? (n - 200) / fe.hop + 1 : 0
    }

    /// Appends samples; returns newly completed mel frames [k, nMels].
    public func push(_ samples: [Float]) -> [Float] {
        audio += samples
        return emit(upTo: readyFrames(), valid: Int.max)
    }

    /// Ends the stream: emits the remaining frames (T = n/160 + 1 in total, the last one zero).
    public func finish() -> [Float] {
        finished = true
        let n = audio.count
        return emit(upTo: n / fe.hop + 1, valid: n / fe.hop)
    }

    func emit(upTo end: Int, valid: Int) -> [Float] {
        guard end > frames else { return [] }
        let W = fe.winLength, off = (fe.nFFT - W) / 2, pad = fe.nFFT / 2
        let k = end - frames
        var fr = [Float](repeating: 0, count: k * W)
        let n = audio.count
        for t in 0..<k {
            let start = (frames + t) * fe.hop - pad + off
            for i in 0..<W {
                let idx = start + i
                guard idx >= 0 && idx < n else { continue }
                fr[t * W + i] = idx == 0 ? audio[0] : audio[idx] - fe.preemph * audio[idx - 1]
            }
        }
        // Rows at or beyond `valid` are zeroed (only the very last frame, at finish).
        let validRows = max(0, min(k, valid - frames))
        let out = fe.featuresFromFrames(fr, T: k, valid: validRows)
        frames = end
        return out
    }
}
