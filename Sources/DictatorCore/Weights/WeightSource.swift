import Foundation

/// Named float tensors, as read by the CPU-side model parts (RNN-T decoder, mel frontend).
public protocol WeightSource {
    func has(_ name: String) -> Bool
    func shape(_ name: String) -> [Int]?
    /// The tensor widened to f32.
    func floats(_ name: String) throws -> [Float]
}

/// A safetensors file read on the CPU: memory-mapped, header parsed once, tensors widened to f32 on
/// request (F32, F16 and BF16).
public final class SafeTensorsReader: WeightSource {
    public struct Entry {
        public let dtype: String
        public let shape: [Int]
        let begin: Int, end: Int
    }

    public let url: URL
    public let entries: [String: Entry]
    let data: Data
    let dataStart: Int

    public init(_ url: URL) throws {
        self.url = url
        guard let d = try? Data(contentsOf: url, options: .alwaysMapped) else { throw EngineError.io("open \(url.path)") }
        data = d
        guard d.count >= 8 else { throw EngineError.io("bad safetensors file \(url.lastPathComponent)") }
        let n = Int(d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt64.self) }.littleEndian)
        guard 8 + n <= d.count, let json = try JSONSerialization.jsonObject(with: d.subdata(in: 8..<(8 + n))) as? [String: Any] else {
            throw EngineError.io("bad safetensors header in \(url.lastPathComponent)")
        }
        dataStart = 8 + n
        var e = [String: Entry]()
        for (k, v) in json where k != "__metadata__" {
            guard let m = v as? [String: Any], let dt = m["dtype"] as? String, let sh = m["shape"] as? [Int],
                  let off = m["data_offsets"] as? [Int], off.count == 2 else { continue }
            e[k] = Entry(dtype: dt, shape: sh, begin: off[0], end: off[1])
        }
        entries = e
    }

    public func has(_ name: String) -> Bool { entries[name] != nil }
    public func shape(_ name: String) -> [Int]? { entries[name]?.shape }

    public func floats(_ name: String) throws -> [Float] {
        guard let e = entries[name] else { throw EngineError.io("tensor \(name) not in \(url.lastPathComponent)") }
        let count = e.shape.reduce(1, *)
        return try data.withUnsafeBytes { raw -> [Float] in
            let p = raw.baseAddress!.advanced(by: dataStart + e.begin)
            switch e.dtype {
            case "F32":
                var out = [Float](repeating: 0, count: count)
                out.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(start: p, count: count * 4)) }
                return out
            case "F16":
                let h = p.assumingMemoryBound(to: Float16.self)
                return (0..<count).map { Float(h[$0]) }
            case "BF16":
                let b = p.assumingMemoryBound(to: UInt16.self)
                return (0..<count).map { Float(bitPattern: UInt32(b[$0]) << 16) }
            default:
                throw EngineError.io("floats(\(name)) dtype \(e.dtype)")
            }
        }
    }
}
