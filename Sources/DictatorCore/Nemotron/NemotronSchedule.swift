import Foundation

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
