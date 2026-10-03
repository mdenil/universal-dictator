import DictatorCore
import XCTest
@testable import UniversalDictatorKit

final class FakeSession: DictationSessionType, @unchecked Sendable {
    let updates: AsyncStream<TranscriptUpdate>
    let cont: AsyncStream<TranscriptUpdate>.Continuation
    var pushed = 0, pauses = 0, resumes = 0
    var final = "hello world"
    init() { (updates, cont) = AsyncStream<TranscriptUpdate>.makeStream() }
    func push(_ samples: [Float]) { pushed += samples.count }
    func pause() { pauses += 1 }
    func resume() { resumes += 1 }
    func stop() async -> String { cont.yield(TranscriptUpdate(committed: final, partial: "", isPaused: false, isFinished: true)); cont.finish(); return final }
    func say(_ committed: String, _ partial: String) { cont.yield(TranscriptUpdate(committed: committed, partial: partial, isPaused: false, isFinished: false)) }
}

final class FakeFactory: SessionFactory, @unchecked Sendable {
    var sessions: [FakeSession] = []
    func makeSession() throws -> DictationSessionType { let s = FakeSession(); sessions.append(s); return s }
}

struct FakeLoader: ModelLoader {
    let factory: FakeFactory
    var fail = false
    func load(_ choice: ModelChoice) async throws -> SessionFactory {
        if fail { throw EngineError.io("no model") }
        return factory
    }
}

final class FakeMic: AudioInput {
    var running = false
    var callback: (([Float]) -> Void)?
    func start(_ onSamples: @escaping ([Float]) -> Void) throws { running = true; callback = onSamples }
    func stop() { running = false; callback = nil }
}

struct FakePermission: MicrophonePermission {
    var granted = true
    var isGranted: Bool { granted }
    var isDenied: Bool { !granted }
    func request() async -> Bool { granted }
}

/// Not yet asked; granting takes a moment (the system prompt), during which the panel may close.
final class PromptPermission: MicrophonePermission, @unchecked Sendable {
    var asked = false
    var onPrompt: (@MainActor () -> Void)?
    var isGranted: Bool { asked }
    var isDenied: Bool { false }
    func request() async -> Bool { await onPrompt?(); asked = true; return true }
}

final class FakeClipboard: Clipboard { var text: String?; func copy(_ t: String) { text = t } }

@MainActor
final class DictationControllerTests: XCTestCase {
    var factory = FakeFactory()
    var mic = FakeMic()
    var clip = FakeClipboard()

    func makeController(granted: Bool = true, failLoad: Bool = false) async -> DictationController {
        let c = DictationController(model: .nemotron160, loader: FakeLoader(factory: factory, fail: failLoad), microphone: mic,
                                    permission: FakePermission(granted: granted), clipboard: clip)
        c.loadModel()
        for _ in 0..<50 where c.phase == .preparing { await Task.yield(); try? await Task.sleep(for: .milliseconds(5)) }
        return c
    }

    func settle() async { for _ in 0..<20 { await Task.yield() }; try? await Task.sleep(for: .milliseconds(20)) }

    func testOpeningThePanelStartsListening() async {
        let c = await makeController()
        XCTAssertEqual(c.phase, .idle)
        await c.panelOpened()
        XCTAssertEqual(c.phase, .listening)
        XCTAssertTrue(mic.running)
        mic.callback?([Float](repeating: 0, count: 320))
        XCTAssertEqual(factory.sessions.first?.pushed, 320)
    }

    func testLiveTextFollowsUpdates() async {
        let c = await makeController()
        await c.panelOpened()
        factory.sessions[0].say("Hi Sarah, thanks ", "for")
        await settle()
        XCTAssertEqual(c.committed, "Hi Sarah, thanks ")
        XCTAssertEqual(c.partial, "for")
    }

    func testPauseResumeAndClosingThePanel() async {
        let c = await makeController()
        await c.panelOpened()
        c.pause()
        XCTAssertEqual(c.phase, .paused); XCTAssertFalse(mic.running); XCTAssertEqual(factory.sessions[0].pauses, 1)
        c.resume()
        XCTAssertEqual(c.phase, .listening); XCTAssertTrue(mic.running); XCTAssertEqual(factory.sessions[0].resumes, 1)
        c.panelClosed()                                    // closing the panel pauses
        XCTAssertEqual(c.phase, .paused); XCTAssertFalse(mic.running)
        await c.panelOpened()                              // reopening resumes the same dictation
        XCTAssertEqual(c.phase, .listening); XCTAssertEqual(factory.sessions.count, 1)
    }

    func testCopySoFarKeepsListening() async {
        let c = await makeController()
        await c.panelOpened()
        factory.sessions[0].say("so far ", "words")
        await settle()
        c.copySoFar()
        XCTAssertEqual(clip.text, "so far words")
        XCTAssertEqual(c.phase, .listening)
    }

    func testStopEditCopyAndNew() async {
        let c = await makeController()
        await c.panelOpened()
        await c.stop()
        XCTAssertEqual(c.phase, .done); XCTAssertFalse(mic.running)
        XCTAssertEqual(c.editedText, "hello world")
        c.editedText = "Hello, world."
        c.copyAndFinish()
        XCTAssertEqual(clip.text, "Hello, world.")
        XCTAssertEqual(c.phase, .idle); XCTAssertEqual(c.text, "")
        await c.panelOpened()                              // the next click starts a new dictation
        XCTAssertEqual(c.phase, .listening); XCTAssertEqual(factory.sessions.count, 2)
    }

    func testDoneTextSurvivesReopening() async {
        let c = await makeController()
        await c.panelOpened()
        await c.stop()
        c.panelClosed()
        await c.panelOpened()
        XCTAssertEqual(c.phase, .done)
        XCTAssertEqual(c.editedText, "hello world")
    }

    /// The panel closes while the permission prompt is up: granting must not start the microphone.
    func testPanelClosedDuringPermissionPromptKeepsMicOff() async {
        let perm = PromptPermission()
        let c = DictationController(model: .nemotron160, loader: FakeLoader(factory: factory), microphone: mic, permission: perm, clipboard: clip)
        c.loadModel()
        for _ in 0..<50 where c.phase == .preparing { try? await Task.sleep(for: .milliseconds(5)) }
        perm.onPrompt = { c.panelClosed() }
        await c.panelOpened()
        XCTAssertFalse(mic.running)
        XCTAssertNotEqual(c.phase, .listening)
        await c.panelOpened()                              // the next click starts normally
        XCTAssertEqual(c.phase, .listening); XCTAssertTrue(mic.running)
    }

    /// Opening while the model loads, then closing before it is ready: the microphone stays off.
    func testClosedBeforeModelReadyKeepsMicOff() async {
        let c = DictationController(model: .nemotron160, loader: FakeLoader(factory: factory), microphone: mic,
                                    permission: FakePermission(), clipboard: clip)
        c.loadModel()
        await c.panelOpened()
        c.panelClosed()
        for _ in 0..<20 { try? await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(mic.running)
        XCTAssertNotEqual(c.phase, .listening)
    }

    func testMicrophoneDenied() async {
        let c = await makeController(granted: false)
        await c.panelOpened()
        XCTAssertEqual(c.phase, .microphoneDenied)
        XCTAssertFalse(mic.running)
    }

    func testLoadFailureIsShown() async {
        let c = await makeController(failLoad: true)
        guard case .failed = c.phase else { return XCTFail("phase \(c.phase)") }
    }

    func testOpeningWhilePreparingStartsWhenReady() async {
        let c = DictationController(model: .nemotron160, loader: FakeLoader(factory: factory), microphone: mic,
                                    permission: FakePermission(), clipboard: clip)
        c.loadModel()
        XCTAssertEqual(c.phase, .preparing)
        await c.panelOpened()
        for _ in 0..<50 where c.phase != .listening { await Task.yield(); try? await Task.sleep(for: .milliseconds(5)) }
        XCTAssertEqual(c.phase, .listening)
    }
}
