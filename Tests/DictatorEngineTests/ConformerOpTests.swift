import XCTest
@testable import DictatorEngine

final class ConformerOpTests: XCTestCase {
    func testLayerNorm() {
        let (T, D) = (5, 1024)
        let x = randomFloats(T * D, seed: 11).map { $0 * 3 + 0.5 }
        let g = randomFloats(D, seed: 12), b = randomFloats(D, seed: 13)
        let out = Tensor(shape: [T, D])
        let s = Step()
        s.layerNorm(Tensor.from(x, shape: [T, D]), gamma: Tensor.from(g, shape: [D]), beta: Tensor.from(b, shape: [D]), out: out, eps: 1e-5, act: .silu)
        s.run()
        var ref = [Float]()
        for t in 0..<T {
            let row = Array(x[(t * D)..<((t + 1) * D)])
            let mean = row.reduce(0, +) / Float(D)
            let v = row.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(D)
            for d in 0..<D {
                let y = (row[d] - mean) / (v + 1e-5).squareRoot() * g[d] + b[d]
                ref.append(y / (1 + exp(-y)))
            }
        }
        XCTAssertLessThan(maxAbsDiff(out.toFloats(), ref), 1e-5)
    }

    func testGLUAndDWConv() {
        let (T, D, K) = (6, 64, 9)
        let x = randomFloats(T * 2 * D, seed: 21)
        let out = Tensor(shape: [T, D])
        var s = Step(); s.glu(Tensor.from(x, shape: [T, 2 * D]), out: out); s.run()
        var ref = [Float]()
        for t in 0..<T { for d in 0..<D { ref.append(x[t * 2 * D + d] / (1 + exp(-x[t * 2 * D + D + d]))) } }
        XCTAssertLessThan(maxAbsDiff(out.toFloats(), ref), 1e-6)

        let Tin = T + K - 1
        let xin = randomFloats(Tin * D, seed: 22), w = randomFloats(D * K, seed: 23), bias = randomFloats(D, seed: 24)
        let o2 = Tensor(shape: [T, D])
        s = Step(); s.dwconv1d(Tensor.from(xin, shape: [Tin, D]), w: Tensor.from(w, shape: [D, K]), bias: Tensor.from(bias, shape: [D]), out: o2); s.run()
        var ref2 = [Float]()
        for t in 0..<T { for d in 0..<D {
            var a = bias[d]
            for k in 0..<K { a += w[d * K + k] * xin[(t + k) * D + d] }
            ref2.append(a)
        } }
        XCTAssertLessThan(maxAbsDiff(o2.toFloats(), ref2), 1e-5)
    }

    func testConv2DGroupsAndPadding() {
        let (Hin, Win, Cin, Cout, stride) = (9, 13, 4, 4, 2)
        for (groups, padTop, padLeft) in [(1, 2, 2), (4, 1, 1)] {
            let Hout = (Hin + padTop + (padTop == 2 ? 1 : 1) - 3) / stride + 1
            let Wout = (Win + padLeft + (padLeft == 2 ? 1 : 1) - 3) / stride + 1
            let x = randomFloats(Hin * Win * Cin, seed: 31)
            let w = randomFloats(Cout * (Cin / groups) * 9, seed: 32), b = randomFloats(Cout, seed: 33)
            let out = Tensor(shape: [Hout, Wout, Cout])
            let s = Step()
            s.conv2d(Tensor.from(x, shape: [Hin, Win, Cin]), w: Tensor.from(w, shape: [Cout, Cin / groups, 3, 3]), bias: Tensor.from(b, shape: [Cout]),
                     out: out, stride: stride, padTop: padTop, padLeft: padLeft, groups: groups, act: .relu, validH: Hout - 1)
            s.run()
            var ref = [Float](repeating: 0, count: Hout * Wout * Cout)
            let cpg = Cin / groups, opg = Cout / groups
            for ho in 0..<(Hout - 1) { for wo in 0..<Wout { for co in 0..<Cout {
                var a = b[co]
                for kh in 0..<3 { for kw in 0..<3 {
                    let hi = ho * stride + kh - padTop, wi = wo * stride + kw - padLeft
                    if hi < 0 || hi >= Hin || wi < 0 || wi >= Win { continue }
                    for ci in 0..<cpg {
                        let cin = (co / opg) * cpg + ci
                        a += w[((co * cpg + ci) * 3 + kh) * 3 + kw] * x[(hi * Win + wi) * Cin + cin]
                    }
                } }
                ref[(ho * Wout + wo) * Cout + co] = max(a, 0)
            } } }
            XCTAssertLessThan(maxAbsDiff(out.toFloats(), ref), 1e-5, "groups \(groups)")
        }
    }

    /// Checks the kernel against a literal transcription of NeMo's RelPositionMultiHeadAttention,
    /// including the pad/view rel_shift trick, with a key cache (Tk > Tq) and a chunked mask.
    func testRelPosAttentionMatchesNeMoRelShift() {
        let (Tq, cache, H, dk) = (7, 5, 2, 16)
        let Tk = cache + Tq
        let HD = H * dk
        let q = randomFloats(Tq * HD, seed: 41), k = randomFloats(Tk * HD, seed: 42), v = randomFloats(Tk * HD, seed: 43)
        let u = randomFloats(HD, seed: 44), vb = randomFloats(HD, seed: 45)
        // NeMo's projected pos_emb has 2*Tk-1 rows for relative positions Tk-1 ... -(Tk-1).
        let nPos = 2 * Tk - 1
        let pe = randomFloats(nPos * HD, seed: 46)
        // Our table is indexed by rel + offset with rel ascending: row(rel) = pe row (Tk-1-rel).
        let offset = Tk - 1
        var table = [Float](repeating: 0, count: nPos * HD)
        for rel in -(Tk - 1)...(Tk - 1) {
            let src = (Tk - 1) - rel, dst = rel + offset
            for c in 0..<HD { table[dst * HD + c] = pe[src * HD + c] }
        }
        // Mask: query i sees keys [lo_i, hi_i).
        var ranges = [Int32]()
        var lo = [Int](), hi = [Int]()
        for i in 0..<Tq { let l = max(0, i - 3), h = min(Tk, cache + i + 2); lo.append(l); hi.append(h); ranges += [Int32(l), Int32(h)] }

        let out = Tensor(shape: [Tq, HD])
        let rt = Tensor(shape: [Tq, 2], dtype: .i32)
        ranges.withUnsafeBytes { rt.contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let s = Step()
        s.relPosAttention(q: Tensor.from(q, shape: [Tq, HD]), k: Tensor.from(k, shape: [Tk, HD]), v: Tensor.from(v, shape: [Tk, HD]),
                          posTable: Tensor.from(table, shape: [nPos, HD]), relOffset: offset,
                          biasU: Tensor.from(u, shape: [H, dk]), biasV: Tensor.from(vb, shape: [H, dk]), ranges: rt, out: out, heads: H)
        s.run()

        var ref = [Float](repeating: 0, count: Tq * HD)
        for h in 0..<H {
            // matrix_bd [Tq, nPos] then NeMo rel_shift.
            var bd = [Float](repeating: 0, count: Tq * nPos)
            for i in 0..<Tq { for n in 0..<nPos {
                var a: Float = 0
                for d in 0..<dk { a += (q[i * HD + h * dk + d] + vb[h * dk + d]) * pe[n * HD + h * dk + d] }
                bd[i * nPos + n] = a
            } }
            var padded = [Float](repeating: 0, count: Tq * (nPos + 1))
            for i in 0..<Tq { for n in 0..<nPos { padded[i * (nPos + 1) + n + 1] = bd[i * nPos + n] } }
            let shifted = Array(padded[Tq...])   // view(-1, Tq)[1:] flattened == drop first Tq elements
            for i in 0..<Tq {
                var scores = [Float](repeating: 0, count: Tk)
                for j in 0..<Tk {
                    var ac: Float = 0
                    for d in 0..<dk { ac += (q[i * HD + h * dk + d] + u[h * dk + d]) * k[j * HD + h * dk + d] }
                    scores[j] = (ac + shifted[i * nPos + j]) / Float(dk).squareRoot()
                }
                let m = (lo[i]..<hi[i]).map { scores[$0] }.max()!
                var den: Float = 0
                var p = [Float](repeating: 0, count: Tk)
                for j in lo[i]..<hi[i] { p[j] = exp(scores[j] - m); den += p[j] }
                for d in 0..<dk {
                    var a: Float = 0
                    for j in 0..<Tk { a += p[j] / den * v[j * HD + h * dk + d] }
                    ref[i * HD + h * dk + d] = a
                }
            }
        }
        XCTAssertLessThan(maxAbsDiff(out.toFloats(), ref), 1e-5)
    }
}
