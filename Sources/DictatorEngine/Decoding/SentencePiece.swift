import Foundation

/// SentencePiece detokenizer: maps token ids to pieces (from `tokenizer.vocab`, one "piece\tscore"
/// per line, id = line number) and joins them, turning the "▁" word marker into spaces.
public struct SentencePieceDecoder {
    public let pieces: [String]

    public init(vocabFile: URL) throws {
        let text = try String(contentsOf: vocabFile, encoding: .utf8)
        pieces = text.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.isEmpty }
            .map { String($0.split(separator: "\t", maxSplits: 1).first ?? "") }
    }

    public var count: Int { pieces.count }

    public func decode(_ ids: [Int]) -> String {
        var bytes = [UInt8]()
        for id in ids where id >= 0 && id < pieces.count {
            let p = pieces[id]
            if p.hasPrefix("<0x"), p.hasSuffix(">"), p.count == 6, let b = UInt8(p.dropFirst(3).dropLast(), radix: 16) {
                bytes.append(b)  // byte-fallback piece
            } else if p == "<unk>" || p == "<s>" || p == "</s>" || p == "<pad>" {
                continue
            } else {
                bytes += Array(p.replacingOccurrences(of: "\u{2581}", with: " ").utf8)
            }
        }
        let s = String(decoding: bytes, as: UTF8.self)
        return s.hasPrefix(" ") ? String(s.dropFirst()) : s
    }
}
