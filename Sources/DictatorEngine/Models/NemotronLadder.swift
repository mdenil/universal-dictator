import Foundation

/// Stage-by-stage comparison against golden tensors from the NeMo reference.
/// Each stage is fed the reference's exact input so a bug cannot hide behind an earlier stage.
public struct LadderRow: CustomStringConvertible {
    public var name: String
    public var shape: [Int]
    public var cosine: Double
    public var maxAbs: Float
    public var refMaxAbs: Float
    public var skipped: String?

    public var description: String {
        if let s = skipped { return String(format: "%-22@ SKIPPED (%@)", name as NSString, s as NSString) }
        return String(format: "%-22@ %-14@ cos %.8f  maxabs %.3e  (ref max %.3e)", name as NSString,
                      shape.map(String.init).joined(separator: "x") as NSString, cosine, maxAbs, refMaxAbs)
    }
}

public func compareTensors(_ name: String, ours: [Float], ref: [Float], shape: [Int]) -> LadderRow {
    guard ours.count == ref.count else {
        return LadderRow(name: name, shape: shape, cosine: .nan, maxAbs: .nan, refMaxAbs: .nan, skipped: "size \(ours.count) vs ref \(ref.count)")
    }
    var dot = 0.0, na = 0.0, nb = 0.0
    var m: Float = 0, rm: Float = 0
    for i in 0..<ours.count {
        let a = Double(ours[i]), b = Double(ref[i])
        dot += a * b; na += a * a; nb += b * b
        m = max(m, abs(ours[i] - ref[i])); rm = max(rm, abs(ref[i]))
    }
    return LadderRow(name: name, shape: shape, cosine: dot / max((na * nb).squareRoot(), 1e-30), maxAbs: m, refMaxAbs: rm, skipped: nil)
}

public enum NemotronLadder {
    /// Runs the offline ladder for one utterance directory of golden tensors.
    /// Expects: audio.npy, mel.npy [T,128], pre_encode.npy, layer_XX_out.npy, enc_out.npy, tokens.npy.
    public static func run(modelDir: URL, goldenDir: URL, left: Int, right: Int, config: FastConformerConfig = .nemotron) throws -> (rows: [LadderRow], text: String, refText: String) {
        let weights = try SafeTensorsFile(modelDir.appendingPathComponent("nemo_extracted/weights.safetensors"))
        var rows = [LadderRow]()
        func golden(_ n: String) -> NPYArray? {
            do { return try NPYArray.load(goldenDir.appendingPathComponent(n + ".npy")) } catch { print("golden \(n): \(error)"); return nil }
        }

        // L0: mel frontend from raw audio.
        let audio = try golden("audio")!.floats()
        let fe = NemoMelFrontend(filterbank: try weights.floats("preprocessor.featurizer.fb"),
                                 window: try weights.floats("preprocessor.featurizer.window"))
        let ourMel = fe.features(audio)
        let refMelArr = golden("mel")!
        let refMel = try refMelArr.floats()
        rows.append(compareTensors("L0 mel", ours: ourMel, ref: refMel, shape: refMelArr.shape))

        // L2: encoder, fed the reference mel.
        let enc = try FastConformerEncoder(weights: weights, config: config)
        var taps: [String: Tensor] = [:]
        let melFrames = refMelArr.shape[0]
        let out = enc.forwardOffline(mel: refMel, melFrames: melFrames, validMel: fe.validFrames(samples: audio.count),
                                     left: left, right: right) { name, t in taps[name] = t }
        for name in ["pre_encode"] + (0..<config.layers).map { String(format: "layer_%02d_out", $0) } {
            guard let g = golden(name), let t = taps[name] else {
                rows.append(LadderRow(name: name, shape: [], cosine: .nan, maxAbs: .nan, refMaxAbs: .nan, skipped: "missing"))
                continue
            }
            rows.append(compareTensors(name, ours: t.toFloats(), ref: try g.floats(), shape: g.shape))
        }

        // L4: greedy tokens from our encoder output.
        let D = config.dModel
        let T = out.shape[0]
        let jointEncW = try weights.tensor("joint.enc.weight"), jointEncB = try weights.tensor("joint.enc.bias")
        let proj = Tensor(shape: [T, jointEncW.shape[0]])
        let s = Step(); s.gemm(out.reshaped([T, D]), jointEncW, bias: jointEncB, out: proj); s.run()
        let dec = try RNNTDecoder(weights: weights)
        var state = dec.initialState()
        var tokens = [Int]()
        let validT = min(T, golden("enc_out")?.shape[0] ?? T)
        dec.greedy(encProj: proj.toFloats(), frames: validT, state: &state, tokens: &tokens)
        let sp = try SentencePieceDecoder(vocabFile: modelDir.appendingPathComponent("nemo_extracted/tokenizer.vocab"))
        let refTokens = (try? golden("tokens")?.ints()) ?? []
        rows.append(LadderRow(name: "L4 tokens", shape: [tokens.count], cosine: tokens == refTokens ? 1 : 0,
                              maxAbs: Float(zip(tokens, refTokens).prefix { $0 == $1 }.count), refMaxAbs: Float(refTokens.count),
                              skipped: nil))
        let refText = (try? String(contentsOf: goldenDir.appendingPathComponent("transcript.txt"), encoding: .utf8)) ?? ""
        return (rows, sp.decode(tokens), refText)
    }
}
