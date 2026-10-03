import DictatorCore
import XCTest
@testable import DictatorEngine

final class VoxtralOpTests: XCTestCase {
    func testAttentionGQAMultiBlock() {
        // 1200 keys spans three online-softmax blocks; 8 query heads share 2 kv heads.
        let (Tq, Tk, Hq, Hkv, d) = (3, 1200, 8, 2, 64)
        let q = randomFloats(Tq * Hq * d, seed: 51), k = randomFloats(Tk * Hkv * d, seed: 52), v = randomFloats(Tk * Hkv * d, seed: 53)
        let ranges: [Int32] = [0, 1200, 450, 1100, 700, 701]
        let rt = Tensor(shape: [Tq, 2], dtype: .i32)
        ranges.withUnsafeBytes { rt.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let out = Tensor(shape: [Tq, Hq * d])
        let s = Step()
        s.attention(q: Tensor.from(q, shape: [Tq, Hq * d]), k: Tensor.from(k, shape: [Tk, Hkv * d]), v: Tensor.from(v, shape: [Tk, Hkv * d]),
                    ranges: rt, out: out, heads: Hq, kvHeads: Hkv, headDim: d)
        s.run()
        var ref = [Float](repeating: 0, count: Tq * Hq * d)
        let scale = 1 / Float(d).squareRoot()
        for i in 0..<Tq { for h in 0..<Hq {
            let kvh = h / (Hq / Hkv)
            let lo = Int(ranges[2 * i]), hi = Int(ranges[2 * i + 1])
            var sc = [Float]()
            for j in lo..<hi {
                var a: Float = 0
                for e in 0..<d { a += q[i * Hq * d + h * d + e] * k[j * Hkv * d + kvh * d + e] }
                sc.append(a * scale)
            }
            let m = sc.max()!
            let ex = sc.map { exp($0 - m) }
            let den = ex.reduce(0, +)
            for e in 0..<d {
                var a: Float = 0
                for (jj, j) in (lo..<hi).enumerated() { a += ex[jj] / den * v[j * Hkv * d + kvh * d + e] }
                ref[i * Hq * d + h * d + e] = a
            }
        } }
        XCTAssertLessThan(maxAbsDiff(out.toFloats(), ref), 2e-5)
    }

    func testMultiQueryAttentionMatchesPerQuery() {
        // Encoder-like: 4 queries, 32 heads x 64, causal-window ranges over 700 keys (3 blocks).
        let (Tq, Tk, H, d) = (4, 700, 32, 64)
        let q = randomFloats(Tq * H * d, seed: 55), k = randomFloats(Tk * H * d, seed: 56), v = randomFloats(Tk * H * d, seed: 57)
        let ranges: [Int32] = [0, 697, 1, 698, 2, 699, 3, 700]
        let rt = Tensor(shape: [Tq, 2], dtype: .i32)
        ranges.withUnsafeBytes { rt.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let qt = Tensor.from(q, shape: [Tq, H * d]), kt = Tensor.from(k, shape: [Tk, H * d]), vt = Tensor.from(v, shape: [Tk, H * d])
        let a = Tensor(shape: [Tq, H * d]), b = Tensor(shape: [Tq, H * d])
        var s = Step(); s.attention(q: qt, k: kt, v: vt, ranges: rt, out: a, heads: H, kvHeads: H, headDim: d); s.run()
        // Reference: one query at a time (single-query kernel path).
        s = Step()
        for i in 0..<Tq {
            s.attention(q: qt.rows(i, 1), k: kt, v: vt, ranges: rt.rows(i, 1), out: b.rows(i, 1), heads: H, kvHeads: H, headDim: d)
        }
        s.run()
        XCTAssertLessThan(maxAbsDiff(a.toFloats(), b.toFloats()), 1e-5)
    }

    func testRopeRotateHalf() {
        let (T, H, d, pos0) = (5, 3, 16, 1234)
        let x = randomFloats(T * H * d, seed: 61)
        let invFreq = (0..<(d / 2)).map { Float(1.0 / pow(1e6, Double(2 * $0) / Double(d))) }
        let t = Tensor.from(x, shape: [T, H * d])
        let s = Step(); s.rope(t, heads: H, headDim: d, invFreq: Tensor.from(invFreq, shape: [d / 2]), pos0: pos0); s.run()
        var ref = x
        for ti in 0..<T { for h in 0..<H { for i in 0..<(d / 2) {
            let ang = Float(pos0 + ti) * invFreq[i]
            let a = x[ti * H * d + h * d + i], b = x[ti * H * d + h * d + i + d / 2]
            ref[ti * H * d + h * d + i] = a * cos(ang) - b * sin(ang)
            ref[ti * H * d + h * d + i + d / 2] = b * cos(ang) + a * sin(ang)
        } } }
        XCTAssertLessThan(maxAbsDiff(t.toFloats(), ref), 1e-5)
    }

    func testArgmaxTiesPickFirst() {
        var x = randomFloats(2 * 131072, seed: 71)
        x[500] = 5; x[90000] = 5          // tie in row 0: first wins
        x[131072 + 7] = 9
        let idx = Tensor(shape: [2], dtype: .i32), val = Tensor(shape: [2])
        let s = Step(); s.argmax(Tensor.from(x, shape: [2, 131072]), idx: idx, val: val); s.run()
        let p = idx.contents.assumingMemoryBound(to: Int32.self)
        XCTAssertEqual(p[0], 500)
        XCTAssertEqual(p[1], 7)
    }

    func testIm2Col() {
        let (Tin, C, K) = (6, 3, 3)
        let x = randomFloats(Tin * C, seed: 81)
        for (stride, pad) in [(1, 2), (2, 1)] {
            let Tout = stride == 1 ? Tin : Tin / 2
            let out = Tensor(shape: [Tout, C * K])
            let s = Step(); s.im2col1d(Tensor.from(x, shape: [Tin, C]), out: out, K: K, stride: stride, pad: pad); s.run()
            var ref = [Float]()
            for t in 0..<Tout { for c in 0..<C { for k in 0..<K {
                let ti = t * stride + k - pad
                ref.append(ti >= 0 && ti < Tin ? x[ti * C + c] : 0)
            } } }
            XCTAssertEqual(out.toFloats(), ref)
        }
    }
}
