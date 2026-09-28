import DictatorEngine
import Foundation

let usage = """
usage: dictate-cli <command> [options]

commands:
  peaks        measure this machine's GPU bandwidth / FLOPS / dispatch overhead
  transcribe   transcribe a WAV/FLAC file with streaming inference:
               transcribe <audio> [--model nemotron|parakeet] [--latency 160|560] [--precision f32|f16]
               parakeet: default exact buffered (Stage A); --horizon H uses frozen-prefix caching
  eval         live-stream a manifest to jsonl (score with eval/score.py, eval/latency.py):
               eval --model nemotron-160|nemotron-560|parakeet|parakeet-h<H>|voxtral --manifest <jsonl> --out <jsonl>
                    [--precision f16|f32] [--weights dictator-q8.safetensors] [--limit N]
  stream       live microphone transcription: stream --model nemotron-160|nemotron-560|parakeet-h4|voxtral
  record       record the dictation test set: record eval/dictation_prompts.txt [--out data/dictation]
  convert      write converted weights: convert nemotron|parakeet --precision f16
  ladder       compare against golden tensors:
               ladder nemotron --golden <utt-dir> [--context 70,1] [--model-dir models/...]
"""

/// Builds a transcriber factory from a model spec:
///   nemotron[-160|-560]  parakeet (exact) | parakeet-h<H> (cached)  voxtral
/// with --precision f32|f16 (NeMo models) and --weights <file> (Voxtral).
func transcriberFactory(spec: String) throws -> () -> StreamingTranscriber {
    let precision = option("--precision", "f16")!
    if spec.hasPrefix("nemotron") {
        let latency = spec.hasSuffix("560") ? 560 : Int(option("--latency", "160")!)!
        let model = try NemotronModel(modelDir: URL(fileURLWithPath: "models/\(NemoModelKind.nemotron.dirName)"), kind: .nemotron, precision: precision)
        return { NemotronStream(model: model, lookahead: latency / 80 - 1) }
    }
    if spec.hasPrefix("parakeet") {
        let model = try NemotronModel(modelDir: URL(fileURLWithPath: "models/\(NemoModelKind.parakeet.dirName)"), kind: .parakeet, precision: precision)
        if let r = spec.range(of: "-h"), let h = Int(spec[r.upperBound...]) {
            return { ParakeetCachedStream(model: model, horizon: h).live() }
        }
        return { ParakeetStream(model: model).live() }
    }
    if spec == "voxtral" {
        let model = try VoxtralModel(modelDir: URL(fileURLWithPath: "models/Voxtral-Mini-4B-Realtime-2602"),
                                     weights: option("--weights", "dictator-q8.safetensors")!)
        return { VoxtralLive(model: model) }
    }
    throw EngineError.io("unknown model spec \(spec)")
}

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
    case "libtime":
        let t0 = Date()
        _ = try MetalContext()
        print(String(format: "Metal library load/compile: %.0f ms", Date().timeIntervalSince(t0) * 1000))
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
            if let h = option("--horizon").flatMap(Int.init) {
                let stream = ParakeetCachedStream(model: model, horizon: h)
                stream.runBuffered(audio: audio)
                text = stream.text; wallTimes = stream.wallTimes; chunkMs = 160
            } else {
                let stream = ParakeetStream(model: model)
                stream.runBuffered(audio: audio)
                text = stream.text; wallTimes = stream.wallTimes; chunkMs = 160
            }
        }
        let wall = Date().timeIntervalSince(t1)
        print(text)
        let w = wallTimes.sorted()
        FileHandle.standardError.write(String(format: "load %.2fs | %d steps of %d ms | audio %.2fs, wall %.3fs (RTF %.4f) | per-step wall p50 %.2f ms p99 %.2f ms\n",
            load, w.count, chunkMs, Double(audio.count) / 16000, wall, wall / (Double(audio.count) / 16000),
            w[w.count / 2] * 1000, w[min(w.count - 1, w.count * 99 / 100)] * 1000).data(using: .utf8)!)
    case "voxtral":
        // voxtral <audio>: offline (reference-schedule) transcription with the full-precision weights.
        let audio = try AudioFile.readMono16k(URL(fileURLWithPath: args[1]))
        let t0 = Date()
        let model = try VoxtralModel(modelDir: URL(fileURLWithPath: option("--model-dir", "models/Voxtral-Mini-4B-Realtime-2602")!),
                                     weights: option("--weights", "model.safetensors")!)
        let load = Date().timeIntervalSince(t0)
        let sess = VoxtralSession(model: model)
        let t1 = Date()
        let text = sess.transcribeOffline(audio)
        let wall = Date().timeIntervalSince(t1)
        print(text)
        let st = sess.stepTimes.dropFirst().sorted()
        FileHandle.standardError.write(String(format: "load %.2fs | prefill %.1f ms | %d steps, p50 %.1f ms p99 %.1f ms | audio %.2fs wall %.2fs | tokens %@\n",
            load, sess.stepTimes[0] * 1000, st.count, st.isEmpty ? 0 : st[st.count / 2] * 1000, st.isEmpty ? 0 : st[st.count * 99 / 100] * 1000,
            Double(audio.count) / 16000, wall, sess.emitted.map(String.init).joined(separator: ",")).data(using: .utf8)!)
        if !sess.splitTimes.isEmpty {
            let e = sess.splitTimes.map { $0.enc }.sorted(), d = sess.splitTimes.map { $0.dec }.sorted()
            FileHandle.standardError.write(String(format: "GPU per step: encoder p50 %.1f ms, decoder p50 %.1f ms\n", e[e.count / 2] * 1000, d[d.count / 2] * 1000).data(using: .utf8)!)
        }
    case "eval":
        // Transcribes every manifest line with a live streaming transcriber (audio pushed in 20 ms
        // pieces); resumable (skips done lines). Records per-step emission data for latency analysis.
        let spec = option("--model", "nemotron")!
        let manifest = option("--manifest")!
        let outPath = option("--out")!
        let limit = Int(option("--limit", "1000000")!)!
        let makeTranscriber = try transcriberFactory(spec: spec)
        var done = Set<String>()
        if let existing = try? String(contentsOfFile: outPath, encoding: .utf8) {
            for line in existing.split(separator: "\n") {
                if let d = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any], let a = d["audio"] as? String { done.insert(a) }
            }
        } else {
            try FileManager.default.createDirectory(at: URL(fileURLWithPath: outPath).deletingLastPathComponent(), withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: outPath, contents: nil)
        }
        let out = try FileHandle(forWritingTo: URL(fileURLWithPath: outPath))
        out.seekToEndOfFile()
        let lines = try String(contentsOfFile: manifest, encoding: .utf8).split(separator: "\n").prefix(limit)
        var n = 0
        for line in lines {
            guard let d = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                  let path = d["audio"] as? String, !done.contains(path) else { continue }
            let audio = try AudioFile.readMono16k(URL(fileURLWithPath: path))
            let t0 = Date()
            let tr = makeTranscriber()
            // Steps: [samples received, compute seconds, transcript so far] whenever a step emitted tokens.
            var steps = [[Any]]()
            var seen = 0
            func record() {
                let em = tr.emissions
                while seen < em.count {
                    let e = em[seen]
                    var j = seen
                    while j + 1 < em.count && em[j + 1].samplesSeen == e.samplesSeen && em[j + 1].computeSeconds == e.computeSeconds { j += 1 }
                    steps.append([e.samplesSeen, e.computeSeconds, tr.text(of: em[0...j].map { $0.token })])
                    seen = j + 1
                }
            }
            var i = 0
            while i < audio.count {
                let k = min(320, audio.count - i)
                tr.push(Array(audio[i..<(i + k)])); i += k
                record()
            }
            tr.finish(); record()
            let st = tr.stepWallTimes
            let rec: [String: Any] = ["audio": path, "ref": d["text"] ?? "", "hyp": tr.transcript,
                                      "duration": Double(audio.count) / 16000, "wall": Date().timeIntervalSince(t0),
                                      "steps": steps, "step_compute_total": st.reduce(0, +), "n_steps": st.count]
            out.write(try JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys]))
            out.write("\n".data(using: .utf8)!)
            n += 1
            if n % 100 == 0 { FileHandle.standardError.write("\(n) done\n".data(using: .utf8)!) }
        }
        try out.close()
    case "convert" where args.count > 1 && args[1] == "voxtral":
        // convert voxtral --quant dec-ffn,dec-attn,enc,adapter,stem --out dictator-q8.safetensors
        let dir = URL(fileURLWithPath: option("--model-dir", "models/Voxtral-Mini-4B-Realtime-2602")!)
        let groups = Set(option("--quant", "dec-ffn")!.split(separator: ",").map(String.init))
        let dst = dir.appendingPathComponent(option("--out", "dictator-q8.safetensors")!)
        let r = try WeightConverter.quantizeVoxtral(src: dir.appendingPathComponent("model.safetensors"), dst: dst, groups: groups)
        print("wrote \(dst.path): \(r.quantized) int8 tensors, \(r.bytes / (1 << 20)) MB")
    case "convert":
        let model = args.count > 1 ? args[1] : "nemotron"
        let precision = WeightConverter.Precision(rawValue: option("--precision", "f16")!)!
        let dirName = ["nemotron": "nemotron-speech-streaming-en-0.6b", "parakeet": "parakeet-unified-en-0.6b"][model]!
        let modelDir = URL(fileURLWithPath: option("--model-dir", "models/\(dirName)")!)
        let dst = NemotronModel.weightsURL(modelDir: modelDir, precision: precision.rawValue)
        let r = try WeightConverter.convert(src: NemotronModel.weightsURL(modelDir: modelDir, precision: "f32"), dst: dst, precision: precision)
        print("wrote \(dst.path): \(r.tensors) tensors, \(r.bytes / (1 << 20)) MB")
    case "dump-step":
        // dump-step <audio> --chunk K --latency 160 --out <dir>
        let audio = try AudioFile.readMono16k(URL(fileURLWithPath: args[1]))
        try Probe.dumpNemotronStep(modelDir: URL(fileURLWithPath: "models/nemotron-speech-streaming-en-0.6b"), audio: audio,
                                   chunk: Int(option("--chunk", "50")!)!, lookahead: Int(option("--latency", "160")!)! / 80 - 1,
                                   out: URL(fileURLWithPath: option("--out", "scratch/stepdump")!))
    case "stream":
        // Live microphone transcription: stream --model nemotron-160|nemotron-560|parakeet-h4|voxtral
        let make = try transcriberFactory(spec: option("--model", "nemotron-160")!)
        let tr = make()
        let q = DispatchQueue(label: "asr")
        let mic = Microphone()
        var last = ""
        try mic.start { samples in
            q.async {
                tr.push(samples)
                let t = tr.transcript
                if t != last { last = t; print("\r\u{1B}[K" + t.suffix(150), terminator: ""); fflush(stdout) }
            }
        }
        FileHandle.standardError.write("listening... press Return to stop\n".data(using: .utf8)!)
        _ = readLine()
        mic.stop()
        q.sync { tr.finish() }
        print("\n" + tr.transcript)
    case "record":
        // record <prompts.txt> --out data/dictation: shows each prompt, records until Return, saves
        // <id>.wav and appends {"audio","text","duration"} to <out>/manifest.jsonl (resumable).
        let prompts = try String(contentsOfFile: args[1], encoding: .utf8).split(separator: "\n").map(String.init).filter { !$0.isEmpty }
        let outDir = URL(fileURLWithPath: option("--out", "data/dictation")!)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let manifest = outDir.appendingPathComponent("manifest.jsonl")
        if !FileManager.default.fileExists(atPath: manifest.path) { FileManager.default.createFile(atPath: manifest.path, contents: nil) }
        let done = (try? String(contentsOf: manifest, encoding: .utf8))?.split(separator: "\n").count ?? 0
        for (i, prompt) in prompts.enumerated() where i >= done {
            print("\n[\(i + 1)/\(prompts.count)] Read aloud, then press Return:\n  \(prompt)")
            var buf = [Float]()
            let lock = NSLock()
            let mic = Microphone()
            try mic.start { s in lock.lock(); buf += s; lock.unlock() }
            _ = readLine()
            mic.stop()
            let wav = outDir.appendingPathComponent(String(format: "dict_%03d.wav", i))
            try WAVWriter.write(buf, to: wav)
            let rec: [String: Any] = ["audio": wav.path, "text": prompt, "duration": Double(buf.count) / 16000]
            let fh = try FileHandle(forWritingTo: manifest); fh.seekToEndOfFile()
            fh.write(try JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys])); fh.write("\n".data(using: .utf8)!)
            try fh.close()
            print(String(format: "  saved %.1fs", Double(buf.count) / 16000))
        }
    case "probe":
        let w = try? SafeTensorsFile(URL(fileURLWithPath: "models/nemotron-speech-streaming-en-0.6b/nemo_extracted/weights.safetensors"))
        if args.contains("rnnt") { try Probe.rnntDecoderCPU(modelDir: URL(fileURLWithPath: "models/nemotron-speech-streaming-en-0.6b")) } else if args.contains("gemm") { Probe.gemmShapes() } else if args.contains("voxtral") { Probe.voxtralShapes() } else { Probe.gemmSmallM(weights: w) }
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
