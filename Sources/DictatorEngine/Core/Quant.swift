import Foundation

/// The one int8 quantizer: symmetric, per output row (channel). Used by the weight converter and
/// by anything that quantizes at runtime, so both produce byte-identical weights by construction.
public enum Quant {
    public static let q4Group = 64

    /// rows x cols floats -> (packed nibbles [rows, cols/2], f16 scales [rows, cols/64]); symmetric per
    /// group of 64 along the row: q = clamp(round(x / s), -8, 7) with s = maxabs / 7, stored as q + 8,
    /// element 2i in the low nibble of byte i.
    public static func quantizeRowsQ4(_ x: UnsafeBufferPointer<Float>, rows: Int, cols: Int) -> (packed: [UInt8], scale: [Float16]) {
        precondition(cols % q4Group == 0)
        let groups = cols / q4Group
        var packed = [UInt8](repeating: 0, count: rows * cols / 2)
        var scale = [Float16](repeating: 0, count: rows * groups)
        packed.withUnsafeMutableBufferPointer { pp in
            scale.withUnsafeMutableBufferPointer { sp in
                let pb = pp.baseAddress!, sb = sp.baseAddress!
                DispatchQueue.concurrentPerform(iterations: rows) { r in
                    for g in 0..<groups {
                        let base = r * cols + g * q4Group
                        var m: Float = 0
                        for c in 0..<q4Group { m = max(m, abs(x[base + c])) }
                        let s16 = Float16(m > 0 ? m / 7 : 1)
                        let s = Float(s16) > 0 ? Float(s16) : 1
                        sb[r * groups + g] = Float16(s)
                        let inv = 1 / s
                        for c in stride(from: 0, to: q4Group, by: 2) {
                            let a = Int(max(-8, min(7, (x[base + c] * inv).rounded()))) + 8
                            let b = Int(max(-8, min(7, (x[base + c + 1] * inv).rounded()))) + 8
                            pb[(base + c) / 2] = UInt8(a | (b << 4))
                        }
                    }
                }
            }
        }
        return (packed, scale)
    }

    /// rows x cols floats -> (int8 codes, per-row scales) with w ≈ q * scale, q in [-127, 127].
    public static func quantizeRowsInt8(_ x: UnsafeBufferPointer<Float>, rows: Int, cols: Int) -> (q: [Int8], scale: [Float]) {
        var q = [Int8](repeating: 0, count: rows * cols)
        var scale = [Float](repeating: 0, count: rows)
        DispatchQueue.concurrentPerform(iterations: rows) { r in
            var m: Float = 0
            for c in 0..<cols { m = max(m, abs(x[r * cols + c])) }
            let s = m > 0 ? m / 127 : 1
            scale[r] = s
            let inv = 1 / s
            q.withUnsafeMutableBufferPointer { qp in
                for c in 0..<cols { qp[r * cols + c] = Int8(max(-127, min(127, (x[r * cols + c] * inv).rounded()))) }
            }
        }
        return (q, scale)
    }
}
