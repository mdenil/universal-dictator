import XCTest
import DictatorCore

/// The dictation API on the Neural Engine, with no Metal in the process. Needs the Nemotron model folder
/// (Core ML models from reference/coreml/nemotron_ane.py --stateful); skipped without it.
final class DictationTests: XCTestCase {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    var assets: ModelAssets { ModelAssets(directory: root.appendingPathComponent("models/nemotron-speech-streaming-en-0.6b")) }
    var clip: URL { root.appendingPathComponent("data/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac") }

    func load(_ choice: ModelChoice) async throws -> LoadedModel {
        try XCTSkipUnless(assets.missingFiles(for: choice).isEmpty, "model files missing: \(assets.missingFiles(for: choice))")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "test clip missing")
        return try await Engine.load(choice, assets: assets)
    }

    func pushAll(_ push: ([Float]) -> Void, _ audio: [Float]) {
        var i = 0
        while i < audio.count { let k = min(320, audio.count - i); push(Array(audio[i..<(i + k)])); i += k }
    }

    func testMissingFilesAreReported() {
        let empty = ModelAssets(directory: FileManager.default.temporaryDirectory.appendingPathComponent("no-such-model-\(UUID())"))
        XCTAssertEqual(Set(empty.missingFiles(for: .nemotron160)), ["nemotron_160_state.mlmodelc", "nemotron-decoder.safetensors", "tokenizer.vocab"])
        XCTAssertTrue(empty.missingFiles(for: .nemotron560).contains("nemotron_560_steady_sub.mlmodelc"))
    }

    /// Live push in 20 ms pieces gives the same tokens as the buffered NeMo schedule (batch-1 tail).
    func testNeuralEngineLiveEqualsBuffered() async throws {
        for choice in ModelChoice.allCases {
            let model = try await load(choice)
            let audio = try AudioFile.readMono16k(clip)
            let a = try model.makeStreamer(); a.endPadding = false
            a.runBuffered(mel: model.frontend.features(audio), frames: audio.count / 160 + 1, validFrames: audio.count / 160 + 1)
            let b = try model.makeStreamer(); b.endPadding = false
            pushAll({ b.push(samples: $0) }, audio); b.flush()
            XCTAssertNil(a.error); XCTAssertNil(b.error)
            XCTAssertFalse(a.tokens.isEmpty)
            XCTAssertEqual(a.tokens, b.tokens, "\(choice)")
        }
    }

    /// A session's final text equals a streamer's over the same audio.
    func testSessionMatchesStreamer() async throws {
        let model = try await load(.nemotron160)
        let audio = try AudioFile.readMono16k(clip)
        let ref = try model.makeStreamer()
        pushAll({ ref.push(samples: $0) }, audio); ref.flush()
        let session = try model.startSession()
        pushAll({ session.push($0) }, audio)
        let text = await session.stop()
        XCTAssertNil(session.error)
        XCTAssertEqual(text, ref.text.trimmingCharacters(in: .whitespaces))
        XCTAssertTrue(text.lowercased().contains("stew for dinner"), text)
    }

    /// Pause/resume: the text is the two segments' texts joined, each transcribed as its own stream; audio
    /// pushed while paused is ignored.
    func testPauseResumeJoinsSegments() async throws {
        let model = try await load(.nemotron160)
        let audio = try AudioFile.readMono16k(clip)
        let half = audio.count / 2
        let first = Array(audio[..<half]), second = Array(audio[half...])
        func alone(_ x: [Float]) throws -> String {
            let s = try model.makeStreamer(); pushAll({ s.push(samples: $0) }, x); s.flush()
            return s.text.trimmingCharacters(in: .whitespaces)
        }
        let session = try model.startSession()
        pushAll({ session.push($0) }, first)
        session.pause()
        pushAll({ session.push($0) }, [Float](repeating: 0.1, count: 16000))      // ignored while paused
        session.resume()
        pushAll({ session.push($0) }, second)
        let text = await session.stop()
        XCTAssertEqual(text, try alone(first) + " " + alone(second))
        var last: TranscriptUpdate?
        for await u in session.updates { last = u }
        XCTAssertEqual(last?.isFinished, true)
        XCTAssertEqual(last?.text, text)
    }

    /// Editing while paused: the edited text replaces everything so far; resumed words are appended to it.
    func testEditWhilePaused() async throws {
        let model = try await load(.nemotron160)
        let audio = try AudioFile.readMono16k(clip)
        let half = audio.count / 2
        let first = Array(audio[..<half]), second = Array(audio[half...])
        func alone(_ x: [Float]) throws -> String {
            let s = try model.makeStreamer(); pushAll({ s.push(samples: $0) }, x); s.flush()
            return s.text.trimmingCharacters(in: .whitespaces)
        }
        let session = try model.startSession()
        pushAll({ session.push($0) }, first)
        session.pause()
        let paused = await session.settledText()
        XCTAssertEqual(paused, try alone(first))                                  // the last words are in
        session.replaceText("Edited first half.\n")
        session.resume()
        pushAll({ session.push($0) }, second)
        let text = await session.stop()
        XCTAssertEqual(text, "Edited first half.\n" + (try alone(second)))         // no extra space after a newline
    }

    /// Deleting everything while paused: only the resumed words remain.
    func testClearWhilePaused() async throws {
        let model = try await load(.nemotron160)
        let audio = try AudioFile.readMono16k(clip)
        let half = audio.count / 2
        let session = try model.startSession()
        pushAll({ session.push($0) }, Array(audio[..<half]))
        session.pause()
        session.replaceText("")
        session.resume()
        pushAll({ session.push($0) }, Array(audio[half...]))
        let s = try model.makeStreamer(); pushAll({ s.push(samples: $0) }, Array(audio[half...])); s.flush()
        let text = await session.stop()
        XCTAssertEqual(text, s.text.trimmingCharacters(in: .whitespaces))
    }
}
