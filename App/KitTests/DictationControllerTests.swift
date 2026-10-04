import DictatorCore
import XCTest
@testable import UniversalDictatorKit

final class FakeSession: DictationSessionType, @unchecked Sendable {
    let updates: AsyncStream<TranscriptUpdate>
    let cont: AsyncStream<TranscriptUpdate>.Continuation
    var pushed = 0, pauses = 0, resumes = 0
    /// What `stop` returns; nil returns the text said (or handed over) last.
    var final: String? = "hello world"
    var said = ""
    /// What a pause finishes the text to (the last words); nil: the text said last.
    var flushed: String?
    var replaced: [String] = []
    init() { (updates, cont) = AsyncStream<TranscriptUpdate>.makeStream() }
    func push(_ samples: [Float]) { pushed += samples.count }
    func pause() { pauses += 1 }
    func settledText() async -> String { flushed ?? said }
    func replaceText(_ text: String) { replaced.append(text); said = text }
    func resume() { resumes += 1 }
    func stop() async -> String {
        let f = final ?? said
        cont.yield(TranscriptUpdate(committed: f, partial: "", isPaused: false, isFinished: true)); cont.finish(); return f
    }
    func say(_ committed: String, _ partial: String) {
        said = committed + partial
        cont.yield(TranscriptUpdate(committed: committed, partial: partial, isPaused: false, isFinished: false))
    }
}

final class FakeFactory: SessionFactory, @unchecked Sendable {
    var sessions: [FakeSession] = []
    func makeSession() throws -> DictationSessionType { let s = FakeSession(); sessions.append(s); return s }
}

final class LoadCounter: @unchecked Sendable { var loads: [ModelChoice] = [] }

struct FakeLoader: ModelLoader {
    let factory: FakeFactory
    var fail = false
    var counter = LoadCounter()
    func load(_ choice: ModelChoice) async throws -> SessionFactory {
        counter.loads.append(choice)
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

    func testPauseAndResumeInsideThePanel() async {
        let c = await makeController()
        await c.panelOpened()
        c.pause()
        XCTAssertEqual(c.phase, .paused); XCTAssertFalse(mic.running); XCTAssertEqual(factory.sessions[0].pauses, 1)
        c.resume()
        XCTAssertEqual(c.phase, .listening); XCTAssertTrue(mic.running); XCTAssertEqual(factory.sessions[0].resumes, 1)
    }

    /// Closing the panel ends the dictation (mic off at once, text to history); the next click starts a new one.
    func testClosingEndsTheDictationAndReopeningStartsANewOne() async {
        let c = await makeController()
        await c.panelOpened()
        factory.sessions[0].say("first ", "one")
        await settle()
        c.panelClosed()
        XCTAssertFalse(mic.running)
        XCTAssertEqual(c.phase, .idle)
        await settle()
        XCTAssertEqual(c.history.map(\.text), ["hello world"])      // the session's final text
        XCTAssertEqual(c.history.first?.copied, false)
        await c.panelOpened()
        XCTAssertEqual(c.phase, .listening); XCTAssertEqual(factory.sessions.count, 2)
        XCTAssertEqual(c.text, "")
    }

    /// Closing the panel while paused keeps the dictation: the next opening shows it still paused, and Resume
    /// carries on in the same session.
    func testClosingWhilePausedKeepsTheDictation() async {
        let c = await makeController()
        await c.panelOpened()
        factory.sessions[0].say("first ", "part")
        await settle()
        c.pause()
        await settle()
        c.panelClosed()
        await settle()
        XCTAssertEqual(c.phase, .paused); XCTAssertFalse(mic.running); XCTAssertTrue(c.history.isEmpty)
        await c.panelOpened()
        XCTAssertEqual(c.phase, .paused); XCTAssertEqual(c.text, "first part"); XCTAssertFalse(mic.running)
        XCTAssertEqual(factory.sessions.count, 1)
        c.resume()
        XCTAssertEqual(c.phase, .listening); XCTAssertTrue(mic.running)
        mic.callback?([Float](repeating: 0, count: 320))
        XCTAssertEqual(factory.sessions[0].pushed, 320)
    }

    func testReopenedPausedDictationStops() async {
        let c = await makeController()
        await c.panelOpened()
        c.pause(); c.panelClosed(); await c.panelOpened()
        await c.stop()
        XCTAssertEqual(c.phase, .done); XCTAssertEqual(c.editedText, "hello world")
    }

    /// Resume can't happen with the panel closed (the microphone only runs while it is open).
    func testResumeNeedsTheOpenPanel() async {
        let c = await makeController()
        await c.panelOpened()
        c.pause(); c.panelClosed()
        c.resume()
        XCTAssertEqual(c.phase, .paused); XCTAssertFalse(mic.running)
    }

    /// Pause shows the text at once, then adds the last words the session finishes.
    func testPauseAddsTheLastWords() async {
        let c = await makeController()
        await c.panelOpened()
        let s = factory.sessions[0]
        s.say("so far ", "wor"); s.flushed = "so far words."
        await settle()
        c.pause()
        XCTAssertEqual(c.text, "so far wor")
        await settle()
        XCTAssertEqual(c.text, "so far words.")
    }

    /// Typing before the last words arrive keeps what was typed.
    func testTypingDuringPauseIsKept() async {
        let c = await makeController()
        await c.panelOpened()
        let s = factory.sessions[0]
        s.say("so far ", "wor"); s.flushed = "so far words."
        await settle()
        c.pause()
        c.editedText = "typed"
        await settle()
        XCTAssertEqual(c.editedText, "typed")
    }

    /// An edit made while paused goes to the session on Resume; new words follow it; Stop gives the lot.
    func testEditWhilePausedThenResume() async {
        let c = await makeController()
        await c.panelOpened()
        let s = factory.sessions[0]; s.final = nil
        s.say("frist draft ", "")
        await settle()
        c.pause(); await settle()
        c.editedText = "First draft."
        c.resume()
        XCTAssertEqual(s.replaced, ["First draft."])
        XCTAssertEqual(c.phase, .listening); XCTAssertEqual(c.text, "First draft.")
        s.say("First draft. And more", "")
        await settle()
        await c.stop()
        XCTAssertEqual(c.editedText, "First draft. And more")
    }

    func testEditWhilePausedThenStop() async {
        let c = await makeController()
        await c.panelOpened()
        let s = factory.sessions[0]; s.final = nil
        s.say("frist ", ""); await settle()
        c.pause(); await settle()
        c.editedText = "First."
        await c.stopAndCopy()
        XCTAssertEqual(s.replaced, ["First."]); XCTAssertEqual(clip.text, "First.")
    }

    func testNoEditHandsNothingOver() async {
        let c = await makeController()
        await c.panelOpened()
        c.pause(); await settle()
        c.resume(); c.pause(); await settle(); await c.stop()
        XCTAssertEqual(factory.sessions[0].replaced, [])
    }

    func testQuitWhilePausedKeepsTheTextInHistory() async {
        let c = await makeController()
        await c.panelOpened()
        c.pause(); await settle()
        c.editedText = "Keep me."
        c.panelClosed()
        c.finishForQuit()
        XCTAssertEqual(c.history.map(\.text), ["Keep me."])
    }

    /// ⌘C while listening: the microphone stops, the whole (final) text is copied, the dictation clears.
    func testStopAndCopy() async {
        let c = await makeController()
        await c.panelOpened()
        factory.sessions[0].say("so far ", "words")
        await settle()
        await c.stopAndCopy()
        XCTAssertFalse(mic.running)
        XCTAssertEqual(clip.text, "hello world")          // the session's final text, not the partial
        XCTAssertEqual(c.phase, .idle)
        XCTAssertEqual(c.text, "")
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

    /// Stopped (not copied) text goes to the history when the panel closes, with any edits.
    func testStoppedTextGoesToHistoryOnClose() async {
        let c = await makeController()
        await c.panelOpened()
        await c.stop()
        c.editedText = "Edited."
        c.panelClosed()
        XCTAssertEqual(c.phase, .idle)
        XCTAssertEqual(c.history.map(\.text), ["Edited."])
        await c.panelOpened()
        XCTAssertEqual(c.phase, .listening)
    }

    func testCopiesGoToHistoryNewestFirst() async {
        let c = await makeController()
        await c.panelOpened(); await c.stopAndCopy()
        await c.panelOpened(); await c.stop(); c.editedText = "Second."; c.copyAndFinish()
        XCTAssertEqual(c.history.map(\.text), ["Second.", "hello world"])
        XCTAssertTrue(c.history.allSatisfy(\.copied))
        c.copyFromHistory(c.history[1])
        XCTAssertEqual(clip.text, "hello world")
        c.clearHistory()
        XCTAssertTrue(c.history.isEmpty)
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

    /// A slow microphone start runs in the background (the panel stays responsive); closing the panel before
    /// it finishes still leaves the microphone off.
    func testSlowMicrophoneStartThenCloseLeavesMicOff() async {
        final class SlowMic: AudioInput, @unchecked Sendable {
            let lock = NSLock(); var _running = false
            var running: Bool { lock.lock(); defer { lock.unlock() }; return _running }
            func start(_ onSamples: @escaping ([Float]) -> Void) throws { usleep(200_000); lock.lock(); _running = true; lock.unlock() }
            func stop() { lock.lock(); _running = false; lock.unlock() }
        }
        let slow = SlowMic()
        let c = DictationController(model: .nemotron160, loader: FakeLoader(factory: factory), microphone: slow, permission: FakePermission(),
                                    clipboard: clip, microphoneQueue: DispatchQueue(label: "test-mic"))
        c.loadModel()
        for _ in 0..<50 where c.phase == .preparing { try? await Task.sleep(for: .milliseconds(5)) }
        let t0 = Date()
        await c.panelOpened()
        XCTAssertLessThan(Date().timeIntervalSince(t0), 0.1)   // did not wait for the microphone
        XCTAssertEqual(c.phase, .listening)
        c.panelClosed()
        try? await Task.sleep(for: .milliseconds(400))
        XCTAssertFalse(slow.running)
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

    // MARK: History kept between launches

    func entry(_ text: String, _ minutesAgo: Double) -> HistoryEntry {
        HistoryEntry(text: text, date: Date(timeIntervalSinceNow: -60 * minutesAgo), model: .nemotron160, copied: false)
    }

    func controller(history: HistoryStore, limit: Int = 50) -> DictationController {
        DictationController(model: .nemotron160, loader: FakeLoader(factory: factory), microphone: mic, permission: FakePermission(),
                            clipboard: clip, history: history, historyLimit: limit)
    }

    func testHistorySurvivesARelaunch() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("history-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let first = controller(history: FileHistoryStore(url: url))
        first.archive("older", model: .nemotron560, copied: true)
        first.archive("newer", model: .nemotron160, copied: false)
        let second = controller(history: FileHistoryStore(url: url))
        // The file keeps dates to the second (ISO 8601), everything else exactly.
        XCTAssertEqual(second.history.map(\.id), first.history.map(\.id))
        XCTAssertEqual(second.history.map(\.copied), [false, true])
        for (a, b) in zip(second.history, first.history) { XCTAssertEqual(a.date.timeIntervalSince1970, b.date.timeIntervalSince1970, accuracy: 1) }
        XCTAssertEqual(second.history.map(\.text), ["newer", "older"])
        XCTAssertEqual(second.history.last?.model, .nemotron560)
    }

    func testHistoryKeepsTheNewestUpToTheLimit() {
        let store = MemoryHistoryStore()
        let c = controller(history: store, limit: 3)
        for i in 1...5 { c.archive("dictation \(i)", model: .nemotron160, copied: false) }
        XCTAssertEqual(c.history.map(\.text), ["dictation 5", "dictation 4", "dictation 3"])
        XCTAssertEqual(store.entries.count, 3)
    }

    func testLoweringTheLimitTrimsAtOnce() {
        let store = MemoryHistoryStore((1...5).map { entry("e\($0)", Double($0)) })
        let c = controller(history: store)
        c.historyLimit = 2
        XCTAssertEqual(c.history.map(\.text), ["e1", "e2"])
        XCTAssertEqual(store.entries.map(\.text), ["e1", "e2"])
    }

    func testClearHistoryEmptiesTheFile() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("history-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let c = controller(history: FileHistoryStore(url: url))
        c.archive("gone soon", model: .nemotron160, copied: false)
        c.clearHistory()
        XCTAssertTrue(c.history.isEmpty)
        XCTAssertTrue(FileHistoryStore(url: url).load().isEmpty)
    }

    func testDamagedHistoryFileGivesAnEmptyHistory() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("history-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("{ not json".utf8).write(to: url)
        let c = controller(history: FileHistoryStore(url: url))
        XCTAssertTrue(c.history.isEmpty)
        c.archive("fresh", model: .nemotron160, copied: false)
        XCTAssertEqual(FileHistoryStore(url: url).load().map(\.text), ["fresh"])
    }

    func testEditingAnEntrySavesItInPlace() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("history-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let c = controller(history: FileHistoryStore(url: url))
        c.archive("first", model: .nemotron160, copied: false)
        c.archive("secnod", model: .nemotron160, copied: true)
        c.archive("third", model: .nemotron160, copied: false)
        let target = c.history[1], date = target.date
        c.editHistory(target.id, text: "second ")
        let reloaded = controller(history: FileHistoryStore(url: url)).history
        XCTAssertEqual(reloaded.map(\.text), ["third", "second", "first"])
        XCTAssertEqual(reloaded[1].edited, true); XCTAssertEqual(reloaded[1].copied, true)
        XCTAssertEqual(reloaded[1].date.timeIntervalSince1970, date.timeIntervalSince1970, accuracy: 1)
        XCTAssertEqual(reloaded[0].edited, false)
        c.copyFromHistory(c.history[1])
        XCTAssertEqual(clip.text, "second")
    }

    func testEmptyOrUnchangedEditIsIgnored() {
        let c = controller(history: MemoryHistoryStore())
        c.archive("keep", model: .nemotron160, copied: false)
        c.editHistory(c.history[0].id, text: "  \n")
        c.editHistory(c.history[0].id, text: "keep")
        XCTAssertEqual(c.history[0].text, "keep"); XCTAssertFalse(c.history[0].edited)
    }

    func testDeleteOneEntry() {
        let c = controller(history: MemoryHistoryStore())
        c.archive("a", model: .nemotron160, copied: false); c.archive("b", model: .nemotron160, copied: false)
        c.deleteHistory(c.history[0].id)
        XCTAssertEqual(c.history.map(\.text), ["a"])
    }

    // MARK: Models

    /// Switching models loads the new one and lets go of the old: switching back loads it again.
    func testSwitchingLoadsOnlyTheChosenModel() async {
        let counter = LoadCounter()
        let c = DictationController(model: .nemotron160, loader: FakeLoader(factory: factory, counter: counter), microphone: mic,
                                    permission: FakePermission(), clipboard: clip)
        func ready() async { for _ in 0..<50 where c.phase == .preparing { try? await Task.sleep(for: .milliseconds(5)) } }
        c.loadModel(); await ready()
        c.selectModel(.nemotron560); await ready()
        c.selectModel(.nemotron160); await ready()
        XCTAssertEqual(c.phase, .idle); XCTAssertTrue(c.isModelReady)
        XCTAssertEqual(counter.loads, [.nemotron160, .nemotron560, .nemotron160])
    }

    func testModelCantChangeWhilePaused() async {
        let c = await makeController()
        await c.panelOpened()
        c.pause()
        c.selectModel(.nemotron560)
        XCTAssertEqual(c.model, .nemotron160)
    }
}
