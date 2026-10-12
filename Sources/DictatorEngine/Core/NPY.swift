import Foundation

/// Minimal reader and writer for NumPy `.npy` files (C order, little endian).
/// The golden tensors from the Python reference arrive in this format.
public struct NPYArray {
    public var shape: [Int]
    public var dtype: String   // numpy descr, e.g. "<f4"
    public var data: Data

    public var count: Int { shape.reduce(1, *) }

    /// The floats of a 2-D tensor laid out time-major, [T, width]: one stored channel-first, [width, T], is
    /// transposed.
    public func timeMajor(width: Int) throws -> (floats: [Float], shape: [Int]) {
        let f = try floats()
        guard shape.count == 2 && shape[0] == width && shape[1] != width else { return (f, shape) }
        let (W, T) = (shape[0], shape[1])
        var tm = [Float](repeating: 0, count: W * T)
        for w in 0..<W { for t in 0..<T { tm[t * W + w] = f[w * T + t] } }
        return (tm, [T, W])
    }

    public func floats() throws -> [Float] {
        switch dtype {
        case "<f4":
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)) }
        case "<f8":
            return data.withUnsafeBytes { $0.bindMemory(to: Double.self).prefix(count).map { Float($0) } }
        case "<f2":
            return data.withUnsafeBytes { $0.bindMemory(to: Float16.self).prefix(count).map { Float($0) } }
        case "<i4":
            return data.withUnsafeBytes { $0.bindMemory(to: Int32.self).prefix(count).map { Float($0) } }
        case "<i8":
            return data.withUnsafeBytes { $0.bindMemory(to: Int64.self).prefix(count).map { Float($0) } }
        default:
            throw EngineError.io("npy dtype \(dtype) not supported as float")
        }
    }

    public func ints() throws -> [Int] {
        switch dtype {
        case "<i4": return data.withUnsafeBytes { $0.bindMemory(to: Int32.self).prefix(count).map { Int($0) } }
        case "<i8": return data.withUnsafeBytes { $0.bindMemory(to: Int64.self).prefix(count).map { Int($0) } }
        case "|u1": return data.map { Int($0) }
        default: throw EngineError.io("npy dtype \(dtype) not supported as int")
        }
    }

    public static func load(_ url: URL) throws -> NPYArray {
        let d = try Data(contentsOf: url, options: .alwaysMapped)
        guard d.count >= 10, d[0] == 0x93, String(data: d[1..<6], encoding: .ascii) == "NUMPY" else {
            throw EngineError.io("not an npy file: \(url.path)")
        }
        let major = d[6]
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            headerLen = Int(d[8]) | (Int(d[9]) << 8)
            headerStart = 10
        } else {
            headerLen = Int(d[8]) | (Int(d[9]) << 8) | (Int(d[10]) << 16) | (Int(d[11]) << 24)
            headerStart = 12
        }
        guard let header = String(data: d[headerStart..<(headerStart + headerLen)], encoding: .ascii) else {
            throw EngineError.io("bad npy header")
        }
        let fortran = header.contains("'fortran_order': True")
        func field(_ key: String) -> String? {
            guard let r = header.range(of: "'\(key)':") else { return nil }
            return String(header[r.upperBound...]).trimmingCharacters(in: .whitespaces)
        }
        guard let descrPart = field("descr"), let q1 = descrPart.firstIndex(of: "'") else { throw EngineError.io("npy descr") }
        let afterQ = descrPart[descrPart.index(after: q1)...]
        guard let q2 = afterQ.firstIndex(of: "'") else { throw EngineError.io("npy descr") }
        let descr = String(afterQ[..<q2])
        guard let shapePart = field("shape"), let open = shapePart.firstIndex(of: "("), let close = shapePart.firstIndex(of: ")") else {
            throw EngineError.io("npy shape")
        }
        let dims = shapePart[shapePart.index(after: open)..<close]
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }.compactMap { Int($0) }
        var payload = d.subdata(in: (headerStart + headerLen)..<d.count)
        if fortran && dims.count > 1 {
            payload = fortranToC(payload, shape: dims, elemSize: Int(descr.dropFirst(2)) ?? 4)
        }
        return NPYArray(shape: dims, dtype: descr, data: payload)
    }

    /// Reorders column-major (Fortran) element data into row-major (C) order.
    static func fortranToC(_ data: Data, shape: [Int], elemSize: Int) -> Data {
        let n = shape.reduce(1, *)
        var out = Data(count: n * elemSize)
        var fStrides = [Int](repeating: 1, count: shape.count)
        for i in 1..<shape.count { fStrides[i] = fStrides[i - 1] * shape[i - 1] }
        data.withUnsafeBytes { src in
            out.withUnsafeMutableBytes { dst in
                var idx = [Int](repeating: 0, count: shape.count)
                for c in 0..<n {
                    var f = 0
                    for k in 0..<shape.count { f += idx[k] * fStrides[k] }
                    memcpy(dst.baseAddress! + c * elemSize, src.baseAddress! + f * elemSize, elemSize)
                    var k = shape.count - 1
                    while k >= 0 { idx[k] += 1; if idx[k] < shape[k] { break }; idx[k] = 0; k -= 1 }
                }
            }
        }
        return out
    }
}
