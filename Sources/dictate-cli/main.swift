import DictatorEngine
import Foundation

let usage = """
usage: dictate-cli <command> [options]

commands:
  peaks        measure this machine's GPU bandwidth / FLOPS / dispatch overhead
  transcribe   transcribe a WAV/FLAC file with streaming inference:
               transcribe <audio> [--model nemotron] [--latency 160|560]
  convert      write converted weights: convert nemotron|parakeet --precision f16
  ladder       compare against golden tensors:
               ladder nemotron --golden <utt-dir> [--context 70,1] [--model-dir models/...]
"""

func option(_ name: String, _ def: String? = nil) -> String? {
    if let i = args.firstIndex(of: name), i + 1 < args.count { return args[i + 1] }
    return def
}

let args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else {
    print(usage)
    exit(2)
}

do {
    switch command {
    case "peaks":
        let ctx = MetalContext.shared
        print("device: \(ctx.device.name)  maxBufferLength: \(ctx.device.maxBufferLength / (1 << 20)) MB  recommendedMaxWorkingSet: \(ctx.device.recommendedMaxWorkingSetSize / (1 << 20)) MB")
        print(try Peaks.measure(ctx))
    case "transcribe":
        guard args.count > 1 else { throw EngineError.io("transcribe: missing audio path") }
        let audio = try AudioFile.readMono16k(URL(fileURLWithPath: args[1]))
        guard let kind = NemoModelKind(rawValue: option("--model", "nemotron")!) else { throw EngineError.io("unknown --model") }
        let modelDir = URL(fileURLWithPath: option("--model-dir", "models/\(kind.dirName)")!)
        let t0 = Date()
        let model = try NemotronModel(modelDir: modelDir, kind: kind, precision: option("--precision", "f32")!)
        let load = Date().timeIntervalSince(t0)
        let t1 = Date()
        let text: String
        var wallTimes: [Double]
        var chunkMs: Int
        switch kind {
        case .nemotron:
            let latency = Int(option("--latency", "160")!)!
            let stream = NemotronStream(model: model, lookahead: latency / 80 - 1)
            let mel = model.frontend.features(audio)
            // NeMo's streaming buffer treats every mel frame (including the zeroed last one) as valid.
            stream.runBuffered(mel: mel, frames: audio.count / 160 + 1, validFrames: audio.count / 160 + 1)
            text = stream.text; wallTimes = stream.wallTimes; chunkMs = stream.schedule.chunkMs
        case .parakeet:
            let stream = ParakeetStream(model: model)
            stream.runBuffered(audio: audio)
            text = stream.text; wallTimes = stream.wallTimes; chunkMs = 160
        }
        let wall = Date().timeIntervalSince(t1)
        print(text)
        let w = wallTimes.sorted()
        FileHandle.standardError.write(String(format: "load %.2fs | %d steps of %d ms | audio %.2fs, wall %.3fs (RTF %.4f) | per-step wall p50 %.2f ms p99 %.2f ms\n",
            load, w.count, chunkMs, Double(audio.count) / 16000, wall, wall / (Double(audio.count) / 16000),
            w[w.count / 2] * 1000, w[min(w.count - 1, w.count * 99 / 100)] * 1000).data(using: .utf8)!)
    case "convert":
        let model = args.count > 1 ? args[1] : "nemotron"
        let precision = WeightConverter.Precision(rawValue: option("--precision", "f16")!)!
        let dirName = ["nemotron": "nemotron-speech-streaming-en-0.6b", "parakeet": "parakeet-unified-en-0.6b"][model]!
        let modelDir = URL(fileURLWithPath: option("--model-dir", "models/\(dirName)")!)
        let dst = NemotronModel.weightsURL(modelDir: modelDir, precision: precision.rawValue)
        let r = try WeightConverter.convert(src: NemotronModel.weightsURL(modelDir: modelDir, precision: "f32"), dst: dst, precision: precision)
        print("wrote \(dst.path): \(r.tensors) tensors, \(r.bytes / (1 << 20)) MB")
    case "probe":
        let w = try? SafeTensorsFile(URL(fileURLWithPath: "models/nemotron-speech-streaming-en-0.6b/nemo_extracted/weights.safetensors"))
        if args.contains("gemm") { Probe.gemmShapes() } else { Probe.gemmSmallM(weights: w) }
    case "ladder":
        guard let kind = NemoModelKind(rawValue: args.count > 1 ? args[1] : "nemotron") else { throw EngineError.io("ladder: unknown model") }
        let ctxParts = option("--context", kind == .nemotron ? "70,1" : "70,2,5")!.split(separator: ",").map { Int($0)! }
        let mask: FastConformerEncoder.MaskSpec = ctxParts.count == 3 ? .chunkedLimitedRC(left: ctxParts[0], chunk: ctxParts[1], right: ctxParts[2])
                                                                      : .chunkedLimited(left: ctxParts[0], right: ctxParts[1])
        let modelDir = URL(fileURLWithPath: option("--model-dir", "models/\(kind.dirName)")!)
        // --golden <utt-dir> for one utterance, or --golden-root <dir> for every utterance under it.
        var dirs = [URL]()
        if let g = option("--golden") { dirs = [URL(fileURLWithPath: g)] }
        if let root = option("--golden-root") {
            let r = URL(fileURLWithPath: root)
            dirs = try FileManager.default.contentsOfDirectory(at: r, includingPropertiesForKeys: nil)
                .filter { $0.hasDirectoryPath }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        }
        let streamRoot = option("--stream-root").map { URL(fileURLWithPath: $0) }
        let verbose = args.contains("-v") || dirs.count == 1
        var worst: [String: LadderRow] = [:]
        var order = [String]()
        var exact = 0, streamExact = 0, streamCount = 0
        let precision = option("--precision", "f32")!
        let streamModel = streamRoot == nil ? nil : try NemotronModel(modelDir: modelDir, kind: kind, precision: precision)
        for d in dirs {
            let r = try NemotronLadder.run(modelDir: modelDir, goldenDir: d, kind: kind, mask: mask, precision: precision)
            var rows = r.rows
            if let sr = streamRoot, let m = streamModel, let (mel, fr, valid) = NemotronLadder.lastMel {
                let sd = sr.appendingPathComponent(d.lastPathComponent)
                if FileManager.default.fileExists(atPath: sd.path) {
                    let srows = try NemotronLadder.streamParity(model: m, mel: mel, frames: fr, validFrames: fr, // NeMo streaming counts every frame valid (valid=\(valid) unused)
                                                                lookahead: ctxParts[1], streamDir: sd)
                    rows += srows
                    streamCount += 1
                    if srows.last?.cosine == 1 { streamExact += 1 }
                }
            }
            if verbose { print("== \(d.lastPathComponent)"); for row in rows { print(row) }; print("ours: \(r.text)\nref:  \(r.refText)") }
            if rows.first(where: { $0.name == "L4 tokens" })?.cosine == 1 { exact += 1 }
            for row in rows where row.skipped == nil {
                if worst[row.name] == nil { order.append(row.name) }
                if let w = worst[row.name], w.cosine <= row.cosine && w.maxAbs >= row.maxAbs { continue }
                var m = worst[row.name] ?? row
                m.cosine = min(m.cosine, row.cosine); m.maxAbs = max(m.maxAbs, row.maxAbs); m.refMaxAbs = max(m.refMaxAbs, row.refMaxAbs)
                worst[row.name] = m
            }
        }
        if dirs.count > 1 {
            print("worst over \(dirs.count) utterances (min cosine, max abs diff):")
            for n in order { print(worst[n]!) }
        }
        print("L4 offline tokens exact: \(exact)/\(dirs.count)")
        if streamRoot != nil { print("L5 streaming tokens exact vs NeMo stream: \(streamExact)/\(streamCount)") }
    default:
        print(usage)
        exit(2)
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
