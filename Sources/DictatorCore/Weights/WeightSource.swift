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
        guard ["F32", "F16", "BF16"].contains(e.dtype) else { throw EngineError.io("floats(\(name)) dtype \(e.dtype)") }
        let count = e.shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        let offset = dataStart + e.begin
        let dtype = e.dtype
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let p = raw.baseAddress!.advanced(by: offset)
            out.withUnsafeMutableBufferPointer { o in
                if dtype == "F32" {
                    memcpy(o.baseAddress!, p, count * 4)
                } else if dtype == "F16" {
                    let h = p.assumingMemoryBound(to: UInt16.self)
                    for i in 0..<count { o[i] = halfToFloat(h[i]) }
                } else {
                    let b = p.assumingMemoryBound(to: UInt16.self)
                    for i in 0..<count { o[i] = Float(bitPattern: UInt32(b[i]) << 16) }
                }
            }
        }
        return out
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
