import DictatorEngine
import Foundation

let usage = """
usage: dictate-cli <command> [options]

commands:
  peaks        measure this machine's GPU bandwidth / FLOPS / dispatch overhead
  transcribe   transcribe a WAV/FLAC file with streaming inference:
               transcribe <audio> [--model nemotron] [--latency 160|560]
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
        let latency = Int(option("--latency", "160")!)!
        let modelDir = URL(fileURLWithPath: option("--model-dir", "models/nemotron-speech-streaming-en-0.6b")!)
        let t0 = Date()
        let model = try NemotronModel(modelDir: modelDir)
        let load = Date().timeIntervalSince(t0)
        let stream = NemotronStream(model: model, lookahead: latency / 80 - 1)
        let mel = model.frontend.features(audio)
        let t1 = Date()
        // NeMo's streaming buffer treats every mel frame (including the zeroed last one) as valid.
        stream.runBuffered(mel: mel, frames: audio.count / 160 + 1, validFrames: audio.count / 160 + 1)
        let wall = Date().timeIntervalSince(t1)
        print(stream.text)
        let g = stream.gpuTimes.sorted(), w = stream.wallTimes.sorted()
        FileHandle.standardError.write(String(format: "load %.2fs | %d chunks of %d ms | audio %.2fs, wall %.3fs (RTF %.4f) | per-chunk GPU p50 %.2f ms p99 %.2f ms | wall p50 %.2f ms p99 %.2f ms\n",
            load, g.count, stream.schedule.chunkMs, Double(audio.count) / 16000, wall, wall / (Double(audio.count) / 16000),
            g[g.count / 2] * 1000, g[min(g.count - 1, g.count * 99 / 100)] * 1000, w[w.count / 2] * 1000, w[min(w.count - 1, w.count * 99 / 100)] * 1000).data(using: .utf8)!)
    case "probe":
        let w = try? SafeTensorsFile(URL(fileURLWithPath: "models/nemotron-speech-streaming-en-0.6b/nemo_extracted/weights.safetensors"))
        Probe.gemmSmallM(weights: w)
    case "ladder":
        let model = args.count > 1 ? args[1] : "nemotron"
        guard model == "nemotron" else { throw EngineError.io("ladder: unknown model \(model)") }
        let ctxParts = option("--context", "70,1")!.split(separator: ",").map { Int($0)! }
        let modelDir = URL(fileURLWithPath: option("--model-dir", "models/nemotron-speech-streaming-en-0.6b")!)
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
        let streamModel = streamRoot == nil ? nil : try NemotronModel(modelDir: modelDir)
        for d in dirs {
            let r = try NemotronLadder.run(modelDir: modelDir, goldenDir: d, left: ctxParts[0], right: ctxParts[1])
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
