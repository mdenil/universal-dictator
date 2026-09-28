import Foundation

/// One emitted token and how much input audio (samples at 16 kHz) had been received when it was
/// produced. `samplesSeen / 16000` is the token's algorithmic emission time; add compute time for
/// the wall-clock emission time.
public struct Emission {
    public let token: Int
    public let samplesSeen: Int
    public let computeSeconds: Double
}

/// Common interface for every streaming transcriber: push audio as it arrives, finish at the end.
public protocol StreamingTranscriber: AnyObject {
    /// Feeds audio samples (16 kHz mono, any chunk size).
    func push(_ samples: [Float])
    /// Ends the stream.
    func finish()
    /// Current transcript.
    var transcript: String { get }
    /// Every emitted token with its emission time.
    var emissions: [Emission] { get }
    /// Wall seconds of each processing step.
    var stepWallTimes: [Double] { get }
    /// Detokenizes a token prefix (for partial display and word-level latency).
    func text(of tokens: [Int]) -> String
}
