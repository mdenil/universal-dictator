import Accelerate
import Foundation

/// RNN-T prediction network (embedding + stacked LSTM) and joint network, run on the CPU with
/// Accelerate. The encoder side of the joint (`joint.enc`) is applied by the encoder (Metal or Core ML),
/// so this only sees 640-wide projected encoder frames.
public final class RNNTDecoder {
    public let hidden: Int          // prediction net / joint hidden size (640)
    public let vocab: Int           // number of outputs including blank (1025)
    let blank: Int
    public let layers: Int
    let maxSymbolsPerFrame: Int

    let embed: [Float]              // [vocab, hidden]; blank row is the padding row
    let wIH: [[Float]], wHH: [[Float]], bias: [[Float]]   // per layer: [4H, H], [4H, H], b_ih + b_hh
    let wCat: [[Float]]                                   // per layer: [4H, 2H] = [W_ih | W_hh]
    let predW: [Float], predB: [Float]                    // joint.pred [H, H], [H]
    let outW: [Float], outB: [Float]                      // joint_net.2 [vocab, H], [vocab]

    /// Decoding state carried across chunks.
    public struct State {
        /// Committed LSTM state: after consuming every emitted token.
        public var h: [[Float]]
        public var c: [[Float]]
        public var lastToken: Int
        /// Prediction output for `lastToken` and the LSTM state it produces. Cached so frames that
        /// end in blank do not rerun the LSTM; committed only when a non-blank token is emitted.
        var pending: (proj: [Float], h: [[Float]], c: [[Float]])?
    }

    public init(weights: WeightSource, prefix: String = "", maxSymbolsPerFrame: Int = 10) throws {
        let embedE = try weights.floats(prefix + "decoder.prediction.embed.weight")
        guard let embedShape = weights.shape(prefix + "decoder.prediction.embed.weight") else { throw EngineError.io("no RNN-T embedding") }
        vocab = embedShape[0]
        hidden = embedShape[1]
        blank = vocab - 1
        embed = embedE
        var l = 0
        var ih = [[Float]](), hh = [[Float]](), bb = [[Float]]()
        while weights.has(prefix + "decoder.prediction.dec_rnn.lstm.weight_ih_l\(l)") {
            ih.append(try weights.floats(prefix + "decoder.prediction.dec_rnn.lstm.weight_ih_l\(l)"))
            hh.append(try weights.floats(prefix + "decoder.prediction.dec_rnn.lstm.weight_hh_l\(l)"))
            let bi = try weights.floats(prefix + "decoder.prediction.dec_rnn.lstm.bias_ih_l\(l)")
            let bh = try weights.floats(prefix + "decoder.prediction.dec_rnn.lstm.bias_hh_l\(l)")
            bb.append(zip(bi, bh).map { $0 + $1 })
            l += 1
        }
        layers = l
        wIH = ih; wHH = hh; bias = bb
        let Hh = hidden
        wCat = (0..<l).map { li in
            var w = [Float](repeating: 0, count: 4 * Hh * 2 * Hh)
            for r in 0..<(4 * Hh) {
                for c in 0..<Hh { w[r * 2 * Hh + c] = ih[li][r * Hh + c]; w[r * 2 * Hh + Hh + c] = hh[li][r * Hh + c] }
            }
            return w
        }
        predW = try weights.floats(prefix + "joint.pred.weight")
        predB = try weights.floats(prefix + "joint.pred.bias")
        outW = try weights.floats(prefix + "joint.joint_net.2.weight")
        outB = try weights.floats(prefix + "joint.joint_net.2.bias")
        self.maxSymbolsPerFrame = maxSymbolsPerFrame
    }

    public func initialState() -> State {
        State(h: Array(repeating: [Float](repeating: 0, count: hidden), count: layers),
              c: Array(repeating: [Float](repeating: 0, count: hidden), count: layers),
              lastToken: blank, pending: nil)
    }

    /// One prediction-network step: returns the joint's pred projection (joint.pred applied) and the new state.
    /// Per layer: gates = [W_ih W_hh] · [x; h] + b in a single sgemv, activations via vForce.
    public func predict(token: Int, h: [[Float]], c: [[Float]]) -> (proj: [Float], h: [[Float]], c: [[Float]]) {
        let H = hidden
        var xh = [Float](repeating: 0, count: 2 * H)
        if token != blank { xh.withUnsafeMutableBufferPointer { p in embed.withUnsafeBufferPointer { e in
            p.baseAddress!.update(from: e.baseAddress! + token * H, count: H) } } }
        var nh = h, nc = c
        var gates = [Float](repeating: 0, count: 4 * H)
        var n32 = Int32(H)
        for l in 0..<layers {
            xh.withUnsafeMutableBufferPointer { p in h[l].withUnsafeBufferPointer { hp in (p.baseAddress! + H).update(from: hp.baseAddress!, count: H) } }
            gates = bias[l]
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(4 * H), Int32(2 * H), 1, wCat[l], Int32(2 * H), xh, 1, 1, &gates, 1)
            gates.withUnsafeMutableBufferPointer { g in
                let gp = g.baseAddress!
                // sigmoid(i, f, o) = 1 / (1 + exp(-x)); tanh(g)
                var neg: Float = -1
                vDSP_vsmul(gp, 1, &neg, gp, 1, vDSP_Length(2 * H))
                vDSP_vsmul(gp + 3 * H, 1, &neg, gp + 3 * H, 1, vDSP_Length(H))
                vvexpf(gp, gp, &n32); vvexpf(gp + H, gp + H, &n32); vvexpf(gp + 3 * H, gp + 3 * H, &n32)
                vvtanhf(gp + 2 * H, gp + 2 * H, &n32)
                var one: Float = 1
                for off in [0, H, 3 * H] {
                    vDSP_vsadd(gp + off, 1, &one, gp + off, 1, vDSP_Length(H))
                    vvrecf(gp + off, gp + off, &n32)
                }
                nc[l].withUnsafeMutableBufferPointer { cp in
                    nh[l].withUnsafeMutableBufferPointer { hp in
                        let cc = cp.baseAddress!, hh = hp.baseAddress!
                        // c' = f*c + i*g ; h' = o * tanh(c')
                        vDSP_vmul(gp + H, 1, cc, 1, cc, 1, vDSP_Length(H))
                        vDSP_vma(gp, 1, gp + 2 * H, 1, cc, 1, cc, 1, vDSP_Length(H))
                        vvtanhf(hh, cc, &n32)
                        vDSP_vmul(hh, 1, gp + 3 * H, 1, hh, 1, vDSP_Length(H))
                    }
                }
            }
            xh.withUnsafeMutableBufferPointer { p in nh[l].withUnsafeBufferPointer { hp in p.baseAddress!.update(from: hp.baseAddress!, count: H) } }
        }
        var proj = predB
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(H), Int32(H), 1, predW, Int32(H), xh, 1, 1, &proj, 1)
        return (proj, nh, nc)
    }

    /// Joint logits for one projected encoder frame and one prediction projection.
    public func jointLogits(encProj: UnsafePointer<Float>, predProj: [Float]) -> [Float] {
        var hsum = [Float](repeating: 0, count: hidden)
        for j in 0..<hidden { hsum[j] = max(encProj[j] + predProj[j], 0) }
        var logits = outB
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(vocab), Int32(hidden), 1, outW, Int32(hidden), hsum, 1, 1, &logits, 1)
        return logits
    }

    /// Greedy RNN-T over `frames` projected encoder frames ([frames, hidden], row-major).
    /// Appends emitted token ids to `tokens`; `state` carries across calls (chunks).
    public func greedy(encProj: [Float], frames: Int, state: inout State, tokens: inout [Int]) {
        encProj.withUnsafeBufferPointer { ep in
            for t in 0..<frames {
                var emitted = 0
                while emitted < maxSymbolsPerFrame {
                    if state.pending == nil {
                        let r = predict(token: state.lastToken, h: state.h, c: state.c)
                        state.pending = (r.proj, r.h, r.c)
                    }
                    let pending = state.pending!
                    let logits = jointLogits(encProj: ep.baseAddress! + t * hidden, predProj: pending.proj)
                    var best = 0
                    var bestV = logits[0]
                    for k in 1..<logits.count where logits[k] > bestV { bestV = logits[k]; best = k }
                    if best == blank { break }
                    tokens.append(best)
                    emitted += 1
                    state.h = pending.h
                    state.c = pending.c
                    state.lastToken = best
                    state.pending = nil
                }
            }
        }
    }
}
