import CoreML
import Foundation

/// A model and its settings, as offered to users. Measured numbers come from the M6 evaluation and the
/// Neural Engine study (docs/PERF_LOG.md).
public enum ModelChoice: String, CaseIterable, Codable, Sendable, Identifiable {
    case nemotron160
    case nemotron560

    public var id: String { rawValue }

    public var schedule: NemotronSchedule {
        switch self {
        case .nemotron160: return NemotronSchedule(lookahead: 1)
        case .nemotron560: return NemotronSchedule(lookahead: 6)
        }
    }

    public var displayName: String {
        switch self {
        case .nemotron160: return "Nemotron · 160 ms"
        case .nemotron560: return "Nemotron · 560 ms"
        }
    }

    /// One line for a model menu: when words appear, accuracy, power.
    public var summary: String {
        switch self {
        case .nemotron160: return "Words in ~0.3 s · 2.7% / 5.3% word errors"
        case .nemotron560: return "Words in ~0.4 s · 2.5% / 4.9% word errors · lowest power"
        }
    }
}

/// Where a model's files live: a folder holding the compiled Core ML models, the decoder weights and the
/// tokenizer. The app passes a folder in its bundle; development tools pass a model folder under `models/`,
/// where the files may sit in `coreml/` and `nemo_extracted/`.
public struct ModelAssets: Sendable {
    public let directory: URL

    public init(directory: URL) { self.directory = directory }

    func existing(_ candidates: [String]) -> URL? {
        candidates.map { directory.appendingPathComponent($0) }.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// `<name>.mlmodelc` or `<name>.mlpackage`, here or in `coreml/`.
    public func coreMLModel(_ name: String) -> URL? {
        existing([name + ".mlmodelc", "coreml/" + name + ".mlmodelc", name + ".mlpackage", "coreml/" + name + ".mlpackage"])
    }

    /// The compiled model, compiling a `.mlpackage` once into `.mlmodelc` next to it if needed.
    public func compiledCoreMLModel(_ name: String) throws -> URL? {
        guard let url = coreMLModel(name) else { return nil }
        if url.pathExtension == "mlmodelc" { return url }
        let out = url.deletingPathExtension().appendingPathExtension("mlmodelc")
        let tmp = try MLModel.compileModel(at: url)
        try FileManager.default.moveItem(at: tmp, to: out)
        return out
    }

    /// RNN-T decoder, joint and mel frontend weights.
    public var decoderWeights: URL? { existing(["nemotron-decoder.safetensors", "nemo_extracted/weights.safetensors"]) }
    public var tokenizerVocab: URL? { existing(["tokenizer.vocab", "nemo_extracted/tokenizer.vocab"]) }

    /// The files `choice` needs that are not here.
    public func missingFiles(for choice: ModelChoice) -> [String] {
        let n = NemotronANEEncoder.modelNames(choice.schedule)
        var missing = [String]()
        for name in [n.sub, n.state].compactMap({ $0 }) where coreMLModel(name) == nil { missing.append(name + ".mlmodelc") }
        if decoderWeights == nil { missing.append("nemotron-decoder.safetensors") }
        if tokenizerVocab == nil { missing.append("tokenizer.vocab") }
        return missing
    }
}

public enum Engine {
    /// Loads a model off the calling thread: decoder weights, tokenizer, and the Core ML models, which
    /// Core ML compiles for the Neural Engine on first use (about 40 s after an install or update, then
    /// cached). A warm-up step runs before returning, so the first dictation starts immediately.
    public static func load(_ choice: ModelChoice, assets: ModelAssets,
                            computeUnits: MLComputeUnits = .cpuAndNeuralEngine) async throws -> LoadedModel {
        try await Task.detached(priority: .userInitiated) {
            let missing = assets.missingFiles(for: choice)
            guard missing.isEmpty else { throw EngineError.io("missing model files in \(assets.directory.path): \(missing.joined(separator: ", "))") }
            let w = try SafeTensorsReader(assets.decoderWeights!)
            let decoder = try RNNTDecoder(weights: w)
            let frontend = NemoMelFrontend(filterbank: try w.floats("preprocessor.featurizer.fb"),
                                           window: try w.floats("preprocessor.featurizer.window"))
            let tokenizer = try SentencePieceDecoder(vocabFile: assets.tokenizerVocab!)
            let loaded = LoadedModel(choice: choice, assets: assets, computeUnits: computeUnits,
                                     decoder: decoder, tokenizer: tokenizer, frontend: frontend)
            // Warm-up: one silent chunk through a throwaway stream.
            let s = try loaded.makeStreamer()
            s.push(samples: [Float](repeating: 0, count: 16000))
            s.flush()
            if let e = s.error { throw e }
            return loaded
        }.value
    }
}

/// A loaded model, ready to start dictation sessions. Sessions may run one after another; the model's
/// weights are shared.
public final class LoadedModel: @unchecked Sendable {
    public let choice: ModelChoice
    public let assets: ModelAssets
    let computeUnits: MLComputeUnits
    public let decoder: RNNTDecoder
    public let tokenizer: SentencePieceDecoder
    public let frontend: NemoMelFrontend

    init(choice: ModelChoice, assets: ModelAssets, computeUnits: MLComputeUnits,
         decoder: RNNTDecoder, tokenizer: SentencePieceDecoder, frontend: NemoMelFrontend) {
        self.choice = choice; self.assets = assets; self.computeUnits = computeUnits
        self.decoder = decoder; self.tokenizer = tokenizer; self.frontend = frontend
    }

    /// A fresh stream (new encoder state) on the shared weights.
    public func makeStreamer() throws -> NemotronStreamer {
        let enc = try NemotronANEEncoder(assets: assets, schedule: choice.schedule, computeUnits: computeUnits)
        return NemotronStreamer(encoder: enc, decoder: decoder, tokenizer: tokenizer, frontend: frontend)
    }

    public func startSession() throws -> DictationSession { try DictationSession(model: self) }
}

/// The transcript as of one moment of a dictation.
public struct TranscriptUpdate: Sendable, Equatable {
    /// Text that will not change: earlier segments and the finished words of the current one.
    public let committed: String
    /// The word still being recognised (may grow); empty when paused or stopped.
    public let partial: String
    public let isPaused: Bool
    public let isFinished: Bool
    public var text: String { committed + partial }

    public init(committed: String, partial: String, isPaused: Bool, isFinished: Bool) {
        self.committed = committed; self.partial = partial; self.isPaused = isPaused; self.isFinished = isFinished
    }
}

/// One dictation: audio in, transcript updates out. It can be paused and resumed; each resume starts a
/// new segment whose text is appended. Thread-safe: audio may be pushed from any thread (work runs on a
/// private serial queue), and updates arrive on `updates`.
public final class DictationSession: @unchecked Sendable {
    public let model: LoadedModel
    public let updates: AsyncStream<TranscriptUpdate>
    let continuation: AsyncStream<TranscriptUpdate>.Continuation
    let queue = DispatchQueue(label: "DictationSession")
    var streamer: NemotronStreamer?          // nil while paused or stopped
    var segments: [String] = []
    var finished = false
    let lock = NSLock()
    var latest: TranscriptUpdate
    /// The first error from the engine, if any.
    public private(set) var error: Error?

    init(model: LoadedModel) throws {
        self.model = model
        (updates, continuation) = AsyncStream<TranscriptUpdate>.makeStream(bufferingPolicy: .bufferingNewest(8))
        latest = TranscriptUpdate(committed: "", partial: "", isPaused: false, isFinished: false)
        streamer = try model.makeStreamer()
    }

    /// The current transcript.
    public var current: TranscriptUpdate { lock.lock(); defer { lock.unlock() }; return latest }

    /// Feeds 16 kHz mono samples. Ignored while paused or after `stop`.
    public func push(_ samples: [Float]) {
        queue.async {
            guard let s = self.streamer else { return }
            let before = s.tokens.count
            s.push(samples: samples)
            if s.tokens.count != before { self.publish() }
        }
    }

    /// Stops listening: flushes the current segment's last words and keeps the text.
    public func pause() {
        queue.async {
            self.endSegment()
            self.publish()
        }
    }

    /// Starts a new segment after `pause`.
    public func resume() {
        queue.async {
            guard self.streamer == nil, !self.finished else { return }
            do { self.streamer = try self.model.makeStreamer() } catch { self.error = self.error ?? error }
            self.publish()
        }
    }

    /// Ends the dictation and returns the final text.
    public func stop() async -> String {
        await withCheckedContinuation { (c: CheckedContinuation<String, Never>) in
            queue.async {
                self.endSegment()
                self.finished = true
                self.publish()
                self.continuation.finish()
                c.resume(returning: self.joined(self.segments))
            }
        }
    }

    func endSegment() {
        guard let s = streamer else { return }
        s.flush()
        if let e = s.error, error == nil { error = e }
        let t = s.text.trimmingCharacters(in: .whitespaces)
        if !t.isEmpty { segments.append(t) }
        streamer = nil
    }

    func joined(_ parts: [String]) -> String { parts.joined(separator: " ") }

    func publish() {
        var committed = joined(segments)
        var partial = ""
        if let s = streamer {
            let t = s.text.trimmingCharacters(in: .whitespaces)
            if !t.isEmpty {
                // The last word may still grow until the next word starts.
                if let sp = t.lastIndex(of: " ") {
                    let done = String(t[...sp])
                    committed += (committed.isEmpty ? "" : " ") + done
                    partial = String(t[t.index(after: sp)...])
                } else {
                    if !committed.isEmpty { committed += " " }
                    partial = t
                }
            }
        }
        let u = TranscriptUpdate(committed: committed, partial: partial, isPaused: streamer == nil && !finished, isFinished: finished)
        lock.lock(); latest = u; lock.unlock()
        continuation.yield(u)
    }
}
