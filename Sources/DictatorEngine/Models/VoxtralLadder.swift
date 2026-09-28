import Foundation

/// Voxtral parity ladder against the transformers golden tensors (docs/specs/voxtral.md, golden layout
/// data/golden/voxtral/offline/<utt>/). Runs our offline schedule on the golden (padded) audio with taps
/// on every stage and compares in pipeline order: mel, conv stem, encoder layers, adapter, decoder
/// input and layers (first positions), then the token sequence.
public enum VoxtralLadder {
    public static func run(model: VoxtralModel, goldenDir: URL) throws -> (rows: [LadderRow], text: String, refText: String) {
        func golden(_ n: String) -> NPYArray? { try? NPYArray.load(goldenDir.appendingPathComponent(n + ".npy")) }
        var rows = [LadderRow]()
        guard let audioArr = golden("audio") else { throw EngineError.io("no audio.npy in \(goldenDir.path)") }
        let padded = try audioArr.floats()          // already includes the silence padding (spec §3)
        let T = padded.count / 160
        let mel = model.frontend.frames(padded, from: 0, count: T)
        if let g = golden("mel") {
            var ref = try g.floats(), shape = g.shape
            if shape.count == 2 && shape[0] == 128 && shape[1] != 128 {   // [128, T] -> [T, 128]
                let (B, N) = (shape[0], shape[1])
                var tm = [Float](repeating: 0, count: B * N)
                for b in 0..<B { for t in 0..<N { tm[t * B + b] = ref[b * N + t] } }
                ref = tm; shape = [N, B]
            }
            rows.append(compareTensors("L0 mel", ours: mel, ref: ref, shape: shape))
        }

        // Stage taps: our tensors per name, concatenated over calls (prefill + steps) so positions line up.
        var taps: [String: [Float]] = [:]
        let sess = VoxtralSession(model: model)
        sess.useSnapshot = false          // compute every position so all encoder frames are tapped
        sess.tap = { name, t in taps[name, default: []] += t.toFloats() }
        let P = T / 8
        let promptLen = model.prompt.count
        var tok = sess.prefill(mel: Array(mel[0..<(8 * promptLen * 128)]))
        var p = promptLen
        while tok != model.eos && p < P - 1 {
            tok = sess.step(mel: Array(mel[(8 * p * 128)..<(8 * (p + 1) * 128)]))
            p += 1
        }
        let encNames = ["conv_stem_out", "enc_layer_00_out", "enc_layer_01_out", "enc_layer_15_out", "enc_layer_31_out", "adapter_out"]
        let decNames = ["dec_in", "dec_layer_00_out", "dec_layer_01_out", "dec_layer_12_out", "dec_layer_25_out"]
        for name in encNames + decNames {
            guard let g = golden(name), let ours = taps[name] else {
                rows.append(LadderRow(name: name, shape: [], cosine: .nan, maxAbs: .nan, refMaxAbs: .nan, skipped: "missing"))
                continue
            }
            let ref = try g.floats()
            // Golden decoder tensors may cover only the first positions; compare the common prefix.
            let n = min(ref.count, ours.count)
            rows.append(compareTensors(name, ours: Array(ours.prefix(n)), ref: Array(ref.prefix(n)), shape: g.shape))
        }
        let refTokens = (try? golden("tokens")?.ints()) ?? []
        // The golden token list may include the prompt; align on the emitted part.
        let refEmitted = refTokens.count > promptLen && Array(refTokens.prefix(promptLen)) == model.prompt ? Array(refTokens.dropFirst(promptLen)) : refTokens
        let ours = sess.emitted
        rows.append(LadderRow(name: "L4 tokens", shape: [ours.count], cosine: ours == refEmitted ? 1 : 0,
                              maxAbs: Float(zip(ours, refEmitted).prefix { $0 == $1 }.count), refMaxAbs: Float(refEmitted.count), skipped: nil))
        let refText = (try? String(contentsOf: goldenDir.appendingPathComponent("transcript.txt"), encoding: .utf8)) ?? ""
        return (rows, sess.text, refText)
    }
}
