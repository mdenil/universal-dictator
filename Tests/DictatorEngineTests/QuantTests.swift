import DictatorCore
import XCTest
@testable import DictatorEngine

final class QuantTests: XCTestCase {
    func testQ4GemvMatchesDequantized() {
        for (M, N, K) in [(1, 300, 1280), (4, 257, 3072), (8, 64, 512), (13, 70, 1024)] {
            let a = randomFloats(M * K, seed: 93), w = randomFloats(N * K, seed: 94)
            let (pk, sc) = w.withUnsafeBufferPointer { Quant.quantizeRowsQ4($0, rows: N, cols: K) }
            var deq = [Float](repeating: 0, count: N * K)
            for r in 0..<N { for c in 0..<K {
                let byte = pk[(r * K + c) / 2]
                let nib = c % 2 == 0 ? Int(byte & 0xF) : Int(byte >> 4)
                deq[r * K + c] = Float(nib - 8) * Float(sc[r * (K / 64) + c / 64])
            } }
            let wt = Tensor(shape: [N, K], dtype: .q4)
            pk.withUnsafeBytes { wt.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            let st = Tensor(shape: [N, K / 64], dtype: .f16)
            sc.withUnsafeBytes { st.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            wt.qscale = st
            let out = Tensor(shape: [M, N])
            let s = Step(); s.gemm(Tensor.from(a, shape: [M, K]), wt, out: out); s.run()
            let ref = refGemmNT(a, deq, M: M, N: N, K: K)
            XCTAssertLessThan(maxAbsDiff(out.toFloats(), ref), 2e-4 * Float(K).squareRoot(), "\(M)x\(N)x\(K)")
            // Quantization error itself stays bounded (sanity): half a step of the group scale.
            let maxErr = zip(w, deq).map { abs($0 - $1) }.max()!
            XCTAssertLessThan(maxErr, 1.0 / 7 / 2 + 1e-3)
        }
    }

    /// int8 GEMV (M <= 8) and GEMM (larger M) against a CPU product with the dequantized weights.
    func testInt8GemmMatchesDequantized() {
        for (M, N, K) in [(1, 300, 1280), (4, 257, 5120), (7, 64, 384), (39, 130, 3072)] {
            let a = randomFloats(M * K, seed: 91), w = randomFloats(N * K, seed: 92)
            let (q, sc) = w.withUnsafeBufferPointer { Quant.quantizeRowsInt8($0, rows: N, cols: K) }
            let deq = (0..<(N * K)).map { Float(q[$0]) * sc[$0 / K] }
            let wt = Tensor(shape: [N, K], dtype: .i8)
            q.withUnsafeBytes { wt.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            wt.qscale = Tensor.from(sc, shape: [N])
            let out = Tensor(shape: [M, N])
            let s = Step(); s.gemm(Tensor.from(a, shape: [M, K]), wt, out: out); s.run()
            let ref = refGemmNT(a, deq, M: M, N: N, K: K)
            XCTAssertLessThan(maxAbsDiff(out.toFloats(), ref), 2e-4 * Float(K).squareRoot(), "\(M)x\(N)x\(K)")
        }
    }
}
