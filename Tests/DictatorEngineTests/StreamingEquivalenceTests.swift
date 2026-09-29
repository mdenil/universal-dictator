import XCTest
@testable import DictatorEngine

/// Live (push) streaming must equal the parity-tested buffered replay exactly.
final class StreamingEquivalenceTests: XCTestCase {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    func testNemoMelStreamEqualsFullUtterance() throws {
        let fe = NemoMelFrontend(filterbank: [Float](repeating: 0.01, count: 128 * 257), window: (0..<400).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / 399)) })
        let audio = randomFloats(16000 + 123, seed: 5)
        let full = fe.features(audio)
        let st = NemoMelStream(fe)
        var got = [Float]()
        var i = 0, sizes = [1, 37, 160, 999, 3000]
        while i < audio.count { let n = min(sizes[0], audio.count - i); sizes.append(sizes.removeFirst()); got += st.push(Array(audio[i..<(i + n)])); i += n }
        got += st.finish()
        XCTAssertEqual(got.count, full.count)
        XCTAssertLessThan(maxAbsDiff(got, full), 1e-4)
    }

    func testNemotronLiveEqualsBuffered() throws {
        let dir = root.appendingPathComponent("models/nemotron-speech-streaming-en-0.6b")
        let clip = root.appendingPathComponent("scratch/dbg/clip.wav")
        guard FileManager.default.fileExists(atPath: clip.path), FileManager.default.fileExists(atPath: dir.path) else {
            throw XCTSkip("model or clip missing")
        }
        let model = try NemotronModel(modelDir: dir, precision: "f16")
        let audio = try AudioFile.readMono16k(clip)
        for lookahead in [1, 6] {
            let a = NemotronStream(model: model, lookahead: lookahead)
            a.runBuffered(mel: model.frontend.features(audio), frames: audio.count / 160 + 1, validFrames: audio.count / 160 + 1)
            let b = NemotronStream(model: model, lookahead: lookahead)
            b.endPadding = false   // exact replay of NeMo's batch-1 schedule
            var i = 0
            while i < audio.count { let n = min(1600, audio.count - i); _ = b.push(samples: Array(audio[i..<(i + n)])); i += n }
            _ = b.flush()
            XCTAssertEqual(a.tokens, b.tokens, "lookahead \(lookahead)")
        }
    }
}

extension StreamingEquivalenceTests {
    func pushAll(_ t: StreamingTranscriber, _ audio: [Float], chunk: Int) {
        var i = 0
        while i < audio.count { let n = min(chunk, audio.count - i); t.push(Array(audio[i..<(i + n)])); i += n }
        t.finish()
    }

    func testParakeetLiveEqualsBuffered() throws {
        let dir = root.appendingPathComponent("models/parakeet-unified-en-0.6b")
        let clip = root.appendingPathComponent("scratch/dbg/clip.wav")
        guard FileManager.default.fileExists(atPath: clip.path) else { throw XCTSkip("clip missing") }
        let model = try NemotronModel(modelDir: dir, kind: .parakeet, precision: "f16")
        let audio = Array(try AudioFile.readMono16k(clip).prefix(16000 * 4 + 777))
        let a = ParakeetStream(model: model); a.runBuffered(audio: audio)
        let b = ParakeetStream(model: model).live(); pushAll(b, audio, chunk: 1000)
        XCTAssertEqual(a.tokens, b.tokens)
        let c = ParakeetCachedStream(model: model, horizon: 4); c.runBuffered(audio: audio)
        let d = ParakeetCachedStream(model: model, horizon: 4).live(); pushAll(d, audio, chunk: 3333)
        XCTAssertEqual(c.tokens, d.tokens)
    }

    func testVoxtralLiveEqualsOffline() throws {
        let dir = root.appendingPathComponent("models/Voxtral-Mini-4B-Realtime-2602")
        let clip = root.appendingPathComponent("scratch/dbg/clip.wav")
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("dictator-q8.safetensors").path) else { throw XCTSkip("q8 weights missing") }
        let model = try VoxtralModel(modelDir: dir, weights: "dictator-q8.safetensors")
        let audio = Array(try AudioFile.readMono16k(clip).prefix(16000 * 4))
        let a = VoxtralSession(model: model); _ = a.transcribeOffline(audio)
        let b = VoxtralLive(model: model); pushAll(b, audio, chunk: 2000)
        XCTAssertEqual(a.emitted, b.session.emitted)
    }
}
