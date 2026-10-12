import DictatorCore
import XCTest
@testable import DictatorEngine

final class QuantTests: XCTestCase {
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
