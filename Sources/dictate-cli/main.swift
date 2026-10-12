import DictatorCore
import DictatorEngine
import Foundation

let usage = """
usage: dictate-cli <command> [options]

commands:
  transcribe   transcribe a WAV/FLAC file with streaming inference:
               transcribe <audio> [--model nemotron|parakeet] [--latency 160|560] [--precision f32|f16]
               parakeet: default exact buffered (Stage A); --horizon H uses frozen-prefix caching
               [--step-log <file>] writes each step's wall time (also for voxtral)
               nemotron: the encoder runs on the Neural Engine (Core ML) by default; --backend metal uses the GPU
                 (also for eval and stream).
  voxtral      voxtral <audio> [--weights model.safetensors|dictator-q8h.safetensors] [--step-log <file>]
  eval         live-stream a manifest to jsonl (score with eval/score.py, eval/latency.py):
               eval --model nemotron-160|nemotron-560|parakeet|parakeet-h<H>|voxtral --manifest <jsonl> --out <jsonl>
                    [--precision f16|f32] [--weights dictator-q8.safetensors] [--limit N]
  stream       live microphone transcription: stream --model nemotron-160|nemotron-560|parakeet-h4|voxtral
  record       record the dictation test set: record eval/dictation_prompts.txt [--out data/dictation]
  dictate      the app's code path on a file: dictate <audio> [--assets build/app-assets] [--model nemotron160|nemotron560] [--pause-at S]
  convert      write converted weights: convert nemotron|parakeet --precision f16
               convert app-assets [--out build/app-assets]: the Mac app's model folder
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
        let dir = URL(fileURLWithPath: "models/\(NemoModelKind.nemotron.dirName)")
        let schedule = NemotronSchedule(lookahead: latency / 80 - 1)
        if nemotronUsesANE(modelDir: dir, schedule: schedule) {
            let assets = ModelAssets(directory: dir)
            return {
                NemotronStreamer(encoder: try! NemotronANEEncoder(assets: assets, schedule: schedule),
                                 decoder: model.decoder, tokenizer: model.tokenizer, frontend: model.frontend)
            }
        }
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

/// --step-log <path>: one step wall time (seconds) per line, for drift over long sessions.
func writeStepLog(_ times: [Double]) throws {
    guard let path = option("--step-log") else { return }
    try times.map { String(format: "%.6f", $0) }.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
}

/// Nemotron's encoder backend: the Neural Engine (Core ML) by default, Metal with --backend metal or when
/// the Core ML models have not been built.
func nemotronUsesANE(modelDir: URL, schedule: NemotronSchedule) -> Bool {
    guard option("--backend", "ane") == "ane" else { return false }
    if NemotronANEEncoder.available(assets: ModelAssets(directory: modelDir), schedule: schedule) { return true }
    FileHandle.standardError.write("Neural Engine models not found in \(modelDir.path)/coreml; using Metal (build them with tools/nemotron_ane.py --stateful)\n".data(using: .utf8)!)
    return false
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
            let schedule = NemotronSchedule(lookahead: latency / 80 - 1)
            let stream = nemotronUsesANE(modelDir: modelDir, schedule: schedule)
                ? NemotronStreamer(encoder: try NemotronANEEncoder(assets: ModelAssets(directory: modelDir), schedule: schedule),
                                   decoder: model.decoder, tokenizer: model.tokenizer, frontend: model.frontend)
                : NemotronStream(model: model, lookahead: schedule.lookahead)
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
        try writeStepLog(wallTimes)
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
        try writeStepLog(sess.stepTimes)
        let st = sess.stepTimes.dropFirst().sorted()
        FileHandle.standardError.write(String(format: "load %.2fs | prefill %.1f ms | %d steps, p50 %.1f ms p99 %.1f ms | audio %.2fs wall %.2fs | tokens %@\n",
            load, sess.stepTimes[0] * 1000, st.count, st.isEmpty ? 0 : st[st.count / 2] * 1000, st.isEmpty ? 0 : st[st.count * 99 / 100] * 1000,
            Double(audio.count) / 16000, wall, sess.emitted.map(String.init).joined(separator: ",")).data(using: .utf8)!)
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
    case "dictate":
        // dictate <audio> [--assets build/app-assets] [--model nemotron160|nemotron560] [--pause-at SECONDS]:
        // the Mac app's code path (DictatorCore only): load, live session in 20 ms pieces, optional pause/resume.
        let audio = try AudioFile.readMono16k(URL(fileURLWithPath: args[1]))
        let assets = ModelAssets(directory: URL(fileURLWithPath: option("--assets", "build/app-assets")!))
        guard let choice = ModelChoice(rawValue: option("--model", "nemotron160")!) else { throw EngineError.io("unknown --model") }
        let pauseAt = option("--pause-at").flatMap(Double.init).map { Int($0 * 16000) }
        let sem = DispatchSemaphore(value: 0)
        var result: Result<String, Error> = .failure(EngineError.io("not run"))
        let t0 = Date()
        Task {
            do {
                let model = try await Engine.load(choice, assets: assets)
                let load = Date().timeIntervalSince(t0)
                let session = try model.startSession()
                var i = 0
                while i < audio.count {
                    let k = min(320, audio.count - i)
                    if let p = pauseAt, i <= p, p < i + k { session.pause(); session.resume() }
                    session.push(Array(audio[i..<(i + k)])); i += k
                }
                let text = await session.stop()
                if let e = session.error { throw e }
                FileHandle.standardError.write(String(format: "load %.2fs (includes Neural Engine compile on first use)\n", load).data(using: .utf8)!)
                result = .success(text)
            } catch { result = .failure(error) }
            sem.signal()
        }
        sem.wait()
        print(try result.get())
    case "convert" where args.count > 1 && args[1] == "app-assets":
        // convert app-assets --out build/app-assets: the Mac app's model folder. Compiled stateful Core ML
        // models for 160 and 560 ms (cloned, so no extra disk on APFS), the RNN-T decoder, joint and mel
        // frontend weights as one small f32 file, and the tokenizer.
        let src = URL(fileURLWithPath: "models/\(NemoModelKind.nemotron.dirName)")
        let out = URL(fileURLWithPath: option("--out", "build/app-assets")!)
        let fm = FileManager.default
        if fm.fileExists(atPath: out.path) { try fm.removeItem(at: out) }
        try fm.createDirectory(at: out, withIntermediateDirectories: true)
        let assets = ModelAssets(directory: src)
        for choice in ModelChoice.allCases {
            let names = NemotronANEEncoder.modelNames(choice.schedule)
            for name in [names.sub, names.state].compactMap({ $0 }) {
                guard let compiled = try assets.compiledCoreMLModel(name) else {
                    throw EngineError.io("\(name) not found in \(src.path)/coreml (tools/nemotron_ane.py --stateful)")
                }
                try fm.copyItem(at: compiled, to: out.appendingPathComponent(name + ".mlmodelc"))
            }
        }
        let w = try SafeTensorsReader(src.appendingPathComponent("nemo_extracted/weights.safetensors"))
        let keep = w.entries.keys.filter { $0.hasPrefix("decoder.") || $0.hasPrefix("joint.pred.") || $0.hasPrefix("joint.joint_net.") || $0.hasPrefix("preprocessor.featurizer.") }.sorted()
        let writer = SafeTensorsWriter()
        for name in keep {
            let f = try w.floats(name)
            writer.add(name, dtype: .f32, shape: w.entries[name]!.shape, data: f.withUnsafeBytes { Data($0) })
        }
        try writer.write(to: out.appendingPathComponent("nemotron-decoder.safetensors"))
        try fm.copyItem(at: src.appendingPathComponent("nemo_extracted/tokenizer.vocab"), to: out.appendingPathComponent("tokenizer.vocab"))
        let missing = ModelChoice.allCases.flatMap { ModelAssets(directory: out).missingFiles(for: $0) }
        guard missing.isEmpty else { throw EngineError.io("app assets incomplete: \(missing)") }
        print("wrote \(out.path): \(keep.count) decoder tensors, Core ML models for \(ModelChoice.allCases.map(\.displayName).joined(separator: ", "))")
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
    case "ladder" where args.count > 1 && args[1] == "voxtral":
        // ladder voxtral --golden-root data/golden/voxtral/offline [--weights model.safetensors]
        let model = try VoxtralModel(modelDir: URL(fileURLWithPath: "models/Voxtral-Mini-4B-Realtime-2602"),
                                     weights: option("--weights", "model.safetensors")!)
        let root = URL(fileURLWithPath: option("--golden-root", "data/golden/voxtral/offline")!)
        let dirs = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
            .filter { $0.hasDirectoryPath }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        var exact = 0
        for d in dirs {
            let r = try VoxtralLadder.run(model: model, goldenDir: d)
            print("== \(d.lastPathComponent)"); for row in r.rows { print(row) }
            print("ours: \(r.text)\nref:  \(r.refText)")
            if r.rows.last?.cosine == 1 { exact += 1 }
        }
        print("L4 Voxtral tokens exact: \(exact)/\(dirs.count)")
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
        if kind == .parakeet, let sr = streamRoot {
            // L5 for Parakeet: our exact buffered algorithm (Stage A) on the golden audio vs NeMo's buffered run.
            let model = try NemotronModel(modelDir: modelDir, kind: .parakeet, precision: option("--precision", "f32")!)
            var exact = 0, total = 0
            for d in dirs {
                let sd = sr.appendingPathComponent(d.lastPathComponent)
                guard let refText = try? String(contentsOf: sd.appendingPathComponent("transcript.txt"), encoding: .utf8) else { continue }
                let audio = try NPYArray.load(d.appendingPathComponent("audio.npy")).floats()
                let st = ParakeetStream(model: model); st.runBuffered(audio: audio)
                total += 1
                let ref = refText.trimmingCharacters(in: .whitespacesAndNewlines)
                if st.text == ref { exact += 1 } else { print("\(d.lastPathComponent):\n  ours: \(st.text)\n  ref:  \(ref)") }
            }
            print("L5 Parakeet exact buffered vs NeMo buffered: transcript exact \(exact)/\(total)")
            break
        }
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
