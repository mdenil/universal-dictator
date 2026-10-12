import Foundation

/// Named float tensors, as read by the CPU-side model parts (RNN-T decoder, mel frontend).
public protocol WeightSource {
    func has(_ name: String) -> Bool
    func shape(_ name: String) -> [Int]?
    /// The tensor widened to f32.
    func floats(_ name: String) throws -> [Float]
}

/// The header of a safetensors file (8-byte little-endian length, then JSON), parsed once: each tensor's
/// dtype ("F32", "BF16", ...), shape and byte range within the data region.
public struct SafeTensorsHeader {
    public struct Entry {
        public let dtype: String
        public let shape: [Int]
        public let begin: Int, end: Int
    }

    public let entries: [String: Entry]
    /// Where the data region starts in the file.
    public let dataStart: Int

    /// `bytes`: the start of the file (at least the whole header).
    public init(_ bytes: UnsafeRawBufferPointer, file: String) throws {
        guard bytes.count >= 8 else { throw EngineError.io("bad safetensors file \(file)") }
        let n = Int(UInt64(littleEndian: bytes.loadUnaligned(fromByteOffset: 0, as: UInt64.self)))
        guard 8 + n <= bytes.count,
              let json = try JSONSerialization.jsonObject(with: Data(bytes[8..<(8 + n)])) as? [String: Any] else {
            throw EngineError.io("bad safetensors header in \(file)")
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
}

/// `count` values of safetensors dtype F32, F16 or BF16 at `p`, widened to f32; nil for other dtypes.
public func widenToFloats(_ p: UnsafeRawPointer, dtype: String, count: Int) -> [Float]? {
    switch dtype {
    case "F32": return Array(UnsafeBufferPointer(start: p.assumingMemoryBound(to: Float.self), count: count))
    case "F16": return UnsafeBufferPointer(start: p.assumingMemoryBound(to: UInt16.self), count: count).map(halfToFloat)
    case "BF16": return UnsafeBufferPointer(start: p.assumingMemoryBound(to: UInt16.self), count: count).map { Float(bitPattern: UInt32($0) << 16) }
    default: return nil
    }
}

/// A safetensors file read on the CPU: memory-mapped, header parsed once, tensors widened to f32 on
/// request (F32, F16 and BF16).
public final class SafeTensorsReader: WeightSource {
    public let url: URL
    let header: SafeTensorsHeader
    let data: Data

    public var entries: [String: SafeTensorsHeader.Entry] { header.entries }

    public init(_ url: URL) throws {
        self.url = url
        guard let d = try? Data(contentsOf: url, options: .alwaysMapped) else { throw EngineError.io("open \(url.path)") }
        data = d
        header = try d.withUnsafeBytes { try SafeTensorsHeader($0, file: url.lastPathComponent) }
    }

    public func has(_ name: String) -> Bool { entries[name] != nil }
    public func shape(_ name: String) -> [Int]? { entries[name]?.shape }

    public func floats(_ name: String) throws -> [Float] {
        guard let e = entries[name] else { throw EngineError.io("tensor \(name) not in \(url.lastPathComponent)") }
        let f = data.withUnsafeBytes { widenToFloats($0.baseAddress! + header.dataStart + e.begin, dtype: e.dtype, count: e.shape.reduce(1, *)) }
        guard let f else { throw EngineError.io("floats(\(name)) dtype \(e.dtype)") }
        return f
    }
}

/// IEEE 754 half -> float (no Float16, which Intel builds lack).
@inline(__always) func halfToFloat(_ h: UInt16) -> Float {
    let sign = UInt32(h & 0x8000) << 16
    let exp = Int((h >> 10) & 0x1F), man = UInt32(h & 0x3FF)
    if exp == 0 {
        if man == 0 { return Float(bitPattern: sign) }
        let v = Float(man) * pow(2, -24)                 // subnormal
        return sign != 0 ? -v : v
    }
    if exp == 31 { return Float(bitPattern: sign | 0x7F80_0000 | (man << 13)) }
    return Float(bitPattern: sign | (UInt32(exp + 112) << 23) | (man << 13))
}
