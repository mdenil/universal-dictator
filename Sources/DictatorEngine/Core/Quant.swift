import Foundation

/// The one int8 quantizer: symmetric, per output row (channel). Used by the weight converter and
/// by anything that quantizes at runtime, so both produce byte-identical weights by construction.
public enum Quant {
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
