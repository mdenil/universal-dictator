import DictatorEngine
import Foundation

let usage = """
usage: dictate-cli <command> [options]

commands:
  peaks        measure this machine's GPU bandwidth / FLOPS / dispatch overhead
"""

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
    default:
        print(usage)
        exit(2)
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
