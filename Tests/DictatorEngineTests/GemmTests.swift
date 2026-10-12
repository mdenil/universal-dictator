import DictatorCore
import Accelerate
import XCTest
@testable import DictatorEngine

/// Deterministic pseudo-random floats in [-1, 1).
func randomFloats(_ n: Int, seed: UInt64) -> [Float] {
    var s = seed &* 6364136223846793005 &+ 1442695040888963407
    return (0..<n).map { _ in
        s = s &* 6364136223846793005 &+ 1442695040888963407
        return Float(Int64(bitPattern: s >> 11) % (1 << 24)) / Float(1 << 23) - 1
    }
}

/// CPU reference: C = A·B^T in fp64-accumulated float.
func refGemmNT(_ a: [Float], _ b: [Float], M: Int, N: Int, K: Int) -> [Float] {
    var c = [Float](repeating: 0, count: M * N)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(M), Int32(N), Int32(K), 1, a, Int32(K), b, Int32(K), 0, &c, Int32(N))
    return c
}

func maxAbsDiff(_ x: [Float], _ y: [Float]) -> Float {
    precondition(x.count == y.count)
    var m: Float = 0
    for i in 0..<x.count { m = max(m, abs(x[i] - y[i])) }
    return m
}

func roundTrip(_ x: [Float], _ d: DType) -> [Float] {
    Tensor.from(x, shape: [x.count], dtype: d).toFloats()
}

final class GemmTests: XCTestCase {
    let shapes: [(Int, Int, Int)] = [(1, 1, 1), (1, 1024, 1024), (7, 33, 65), (64, 64, 64), (100, 257, 130), (129, 640, 1024), (3, 4096, 1024)]

    func testF32() {
        for (M, N, K) in shapes {
            let a = randomFloats(M * K, seed: 1), b = randomFloats(N * K, seed: 2)
            let ta = Tensor.from(a, shape: [M, K]), tb = Tensor.from(b, shape: [N, K])
            let tc = Tensor(shape: [M, N])
            let s = Step(); s.gemm(ta, tb, out: tc); s.run()
            let err = maxAbsDiff(tc.toFloats(), refGemmNT(a, b, M: M, N: N, K: K))
            XCTAssertLessThan(err, 1e-4 * Float(K).squareRoot() + 1e-5, "f32 \(M)x\(N)x\(K) err \(err)")
        }
    }

    func testMixedDtypes() {
        // The combinations the models use: f32 activations with f32, f16 or bf16 weights.
        let combos: [(DType, DType, DType)] = [(.f32, .f16, .f32), (.f32, .bf16, .f32), (.f32, .bf16, .f16)]
        for (da, db, dc) in combos {
            for (M, N, K) in shapes {
                // Reference uses the same rounded inputs, so only accumulation/output rounding differs.
                let a = roundTrip(randomFloats(M * K, seed: 3), da), b = roundTrip(randomFloats(N * K, seed: 4), db)
                let ta = Tensor.from(a, shape: [M, K], dtype: da), tb = Tensor.from(b, shape: [N, K], dtype: db)
                let tc = Tensor(shape: [M, N], dtype: dc)
                let s = Step(); s.gemm(ta, tb, out: tc); s.run()
                let ref = refGemmNT(a, b, M: M, N: N, K: K)
                let err = maxAbsDiff(tc.toFloats(), ref)
                let refMax = ref.map(abs).max() ?? 1
                let tol: Float = dc == .f16 ? refMax * 1e-3 + 1e-3 : 1e-4 * Float(K).squareRoot() + 1e-5
                XCTAssertLessThan(err, tol, "\(da)/\(db)/\(dc) \(M)x\(N)x\(K) err \(err)")
            }
        }
    }

    func testEpilogue() {
        let (M, N, K) = (37, 70, 50)
        let a = randomFloats(M * K, seed: 5), b = randomFloats(N * K, seed: 6)
        let bias = randomFloats(N, seed: 7), res = randomFloats(M * N, seed: 8)
        let ta = Tensor.from(a, shape: [M, K]), tb = Tensor.from(b, shape: [N, K])
        let tbias = Tensor.from(bias, shape: [N]), tres = Tensor.from(res, shape: [M, N])
        for act in [Activation.none, .relu, .silu, .geluErf] {
            let tc = Tensor(shape: [M, N])
            let s = Step()
            s.gemm(ta, tb, bias: tbias, out: tc, act: act, alpha: 0.5, outScale: 0.25, residual: tres, resScale: 2)
            s.run()
            let ab = refGemmNT(a, b, M: M, N: N, K: K)
            var ref = [Float](repeating: 0, count: M * N)
            for i in 0..<M { for j in 0..<N {
                var v = 0.5 * ab[i * N + j] + bias[j]
                switch act {
                case .none: break
                case .relu: v = max(v, 0)
                case .silu: v = v / (1 + exp(-v))
                case .geluErf: v = 0.5 * v * (1 + erf(v / 2.0.squareRoot().float))
                }
                ref[i * N + j] = 0.25 * v + 2 * res[i * N + j]
            } }
            XCTAssertLessThan(maxAbsDiff(tc.toFloats(), ref), 2e-5, "act \(act)")
        }
    }
}

extension Double { var float: Float { Float(self) } }
