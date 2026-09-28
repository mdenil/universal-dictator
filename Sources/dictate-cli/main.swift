import DictatorEngine
import Foundation

let usage = """
usage: dictate-cli <command> [options]

commands:
  peaks        measure this machine's GPU bandwidth / FLOPS / dispatch overhead
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
