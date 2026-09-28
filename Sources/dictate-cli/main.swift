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
        stream.runBuffered(mel: mel, frames: audio.count / 160 + 1, validFrames: model.frontend.validFrames(samples: audio.count))
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
        let golden = URL(fileURLWithPath: option("--golden")!)
        let r = try NemotronLadder.run(modelDir: modelDir, goldenDir: golden, left: ctxParts[0], right: ctxParts[1])
        for row in r.rows { print(row) }
        print("ours: \(r.text)")
        print("ref:  \(r.refText)")
    default:
        print(usage)
        exit(2)
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
