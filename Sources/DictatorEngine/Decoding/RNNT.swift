import Accelerate
import Foundation

/// RNN-T prediction network (embedding + stacked LSTM) and joint network, run on the CPU with
/// Accelerate. The encoder side of the joint (`joint.enc`) is applied on the GPU beforehand, so
/// this only sees 640-wide projected encoder frames.
public final class RNNTDecoder {
    public let hidden: Int          // prediction net / joint hidden size (640)
    public let vocab: Int           // number of outputs including blank (1025)
    public let blank: Int
    public let layers: Int
    public let maxSymbolsPerFrame: Int

    let embed: [Float]              // [vocab, hidden]; blank row is the padding row
    let wIH: [[Float]], wHH: [[Float]], bias: [[Float]]   // per layer: [4H, H], [4H, H], b_ih + b_hh
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

    public init(weights: SafeTensorsFile, prefix: String = "", maxSymbolsPerFrame: Int = 10) throws {
        let embedE = try weights.floats(prefix + "decoder.prediction.embed.weight")
        vocab = weights.entries[prefix + "decoder.prediction.embed.weight"]!.shape[0]
        hidden = weights.entries[prefix + "decoder.prediction.embed.weight"]!.shape[1]
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
    func predict(token: Int, h: [[Float]], c: [[Float]]) -> (proj: [Float], h: [[Float]], c: [[Float]]) {
        let H = hidden
        var x = token == blank ? [Float](repeating: 0, count: H) : Array(embed[(token * H)..<((token + 1) * H)])
        var nh = h, nc = c
        var gates = [Float](repeating: 0, count: 4 * H)
        for l in 0..<layers {
            gates = bias[l]
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(4 * H), Int32(H), 1, wIH[l], Int32(H), x, 1, 1, &gates, 1)
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(4 * H), Int32(H), 1, wHH[l], Int32(H), h[l], 1, 1, &gates, 1)
            var hl = [Float](repeating: 0, count: H), cl = [Float](repeating: 0, count: H)
            for j in 0..<H {
                let i = 1 / (1 + exp(-gates[j]))
                let f = 1 / (1 + exp(-gates[H + j]))
                let g = tanh(gates[2 * H + j])
                let o = 1 / (1 + exp(-gates[3 * H + j]))
                cl[j] = f * c[l][j] + i * g
                hl[j] = o * tanh(cl[j])
            }
            nh[l] = hl; nc[l] = cl
            x = hl
        }
        var proj = predB
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(H), Int32(H), 1, predW, Int32(H), x, 1, 1, &proj, 1)
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
    public func greedy(encProj: [Float], frames: Int, state: inout State, tokens: inout [Int],
                       logitsSink: ((Int, [Float]) -> Void)? = nil) {
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
                    logitsSink?(t, logits)
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
