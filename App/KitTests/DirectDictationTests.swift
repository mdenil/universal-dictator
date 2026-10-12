import DictatorCore
import XCTest
@testable import UniversalDictatorKit

final class FakeTyper: TextTyper, @unchecked Sendable {
    var typed: [String] = []
    /// The field's contents.
    var all = ""
    func type(_ text: String) { typed.append(text); all += text }
    func deleteBackward(_ count: Int) { typed.append("⌫\(count)"); all.removeLast(count) }
}

final class FakeAccessibility: AccessibilityPermission { var isGranted = true }

final class DoubleTapDetectorTests: XCTestCase {
    /// The double-tap outputs (single taps left out; `taps` counts them).
    func run(_ events: [DoubleTapDetector.Event], key: ModifierKey = .leftShift) -> [DoubleTapDetector.Output] {
        var d = DoubleTapDetector(key: key)
        return events.compactMap { d.handle($0) }.filter { $0 != .tap }
    }

    func taps(_ events: [DoubleTapDetector.Event]) -> Int {
        var d = DoubleTapDetector(key: .leftShift)
        return events.compactMap { d.handle($0) }.filter { $0 == .tap }.count
    }

    /// Toggle stops with one tap: a quick press and release, with nothing in between.
    func testSingleTap() {
        XCTAssertEqual(taps([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.1)]), 1)
        XCTAssertEqual(taps([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.9)]), 0)       // held: not a tap
        XCTAssertEqual(taps([.modifierDown(.leftShift, at: 0), .other, .modifierUp(.leftShift, at: 0.1)]), 0)   // typing a capital
        XCTAssertEqual(taps([.modifierDown(.rightShift, at: 0), .modifierUp(.rightShift, at: 0.1)]), 0)
    }

    /// After a tap is used (to stop), a quick second press doesn't count as a double tap.
    func testResetAfterATap() {
        var d = DoubleTapDetector(key: .leftShift)
        _ = d.handle(.modifierDown(.leftShift, at: 0))
        XCTAssertEqual(d.handle(.modifierUp(.leftShift, at: 0.1)), .tap)
        d.reset()
        XCTAssertNil(d.handle(.modifierDown(.leftShift, at: 0.2)))
    }

    func testTwoQuickTapsFire() {
        XCTAssertEqual(run([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.08),
                            .modifierDown(.leftShift, at: 0.25), .modifierUp(.leftShift, at: 0.33)]),
                       [.doubleTapDown, .doubleTapUp])
    }

    /// Push to talk: Shift, Shift-hold while talking, release.
    func testSecondPressCanBeHeld() {
        XCTAssertEqual(run([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.1),
                            .modifierDown(.leftShift, at: 0.3)]), [.doubleTapDown])
        XCTAssertEqual(run([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.1),
                            .modifierDown(.leftShift, at: 0.3), .modifierUp(.leftShift, at: 9.0)]), [.doubleTapDown, .doubleTapUp])
    }

    func testSlowTapsDontFire() {
        XCTAssertEqual(run([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.1),
                            .modifierDown(.leftShift, at: 0.7), .modifierUp(.leftShift, at: 0.8)]), [])
    }

    /// Shift held to type a capital is not a tap.
    func testLongFirstPressDoesntCount() {
        XCTAssertEqual(run([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.9),
                            .modifierDown(.leftShift, at: 1.0)]), [])
    }

    func testRightShiftDoesntCount() {
        XCTAssertEqual(run([.modifierDown(.rightShift, at: 0), .modifierUp(.rightShift, at: 0.1),
                            .modifierDown(.rightShift, at: 0.2)]), [])
        XCTAssertEqual(run([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.1),
                            .modifierDown(.rightShift, at: 0.2)]), [])
    }

    /// Typing capitals ("AB": Shift+A, Shift+B) never fires.
    func testKeyOrClickBetweenCancels() {
        XCTAssertEqual(run([.modifierDown(.leftShift, at: 0), .other, .modifierUp(.leftShift, at: 0.1),
                            .modifierDown(.leftShift, at: 0.2), .other, .modifierUp(.leftShift, at: 0.3)]), [])
        XCTAssertEqual(run([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.1), .other,
                            .modifierDown(.leftShift, at: 0.2)]), [])
    }

    /// Push to talk: keys arriving while Shift is held (the dictation's own typing) don't stop the release
    /// from ending it.
    func testKeysDuringTheHeldSecondPressDontCancelTheRelease() {
        XCTAssertEqual(run([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.1),
                            .modifierDown(.leftShift, at: 0.3), .other, .other, .modifierUp(.leftShift, at: 5)]), [.doubleTapDown, .doubleTapUp])
    }

    func testAnotherModifierCancels() {
        XCTAssertEqual(run([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.1),
                            .modifierDown(.leftCommand, at: 0.15), .modifierDown(.leftShift, at: 0.2)]), [])
    }

    func testOtherKeysCanBeChosen() {
        XCTAssertEqual(run([.modifierDown(.rightOption, at: 0), .modifierUp(.rightOption, at: 0.1),
                            .modifierDown(.rightOption, at: 0.2)], key: .rightOption), [.doubleTapDown])
    }

    func testThreeTapsFireOnce() {
        XCTAssertEqual(run([.modifierDown(.leftShift, at: 0), .modifierUp(.leftShift, at: 0.1),
                            .modifierDown(.leftShift, at: 0.2), .modifierUp(.leftShift, at: 0.3),
                            .modifierDown(.leftShift, at: 0.4), .modifierUp(.leftShift, at: 0.5)]), [.doubleTapDown, .doubleTapUp])
    }

    func combo(_ mods: UInt32, _ display: String) -> DictationShortcut { .combination(keyCode: 0, carbonModifiers: mods, display: display) }

    /// Combinations that take over typing or a standard command everywhere get a warning; the rest don't.
    func testCombinationWarnings() {
        XCTAssertEqual(combo(0x0200, "⇧A").warning, "⇧A replaces typing a capital A everywhere")
        XCTAssertEqual(combo(0x0100, "⌘A").warning, "⌘A replaces Select All everywhere")
        XCTAssertEqual(combo(0x0100, "⌘C").warning, "⌘C replaces Copy everywhere")
        XCTAssertEqual(combo(0x0100, "⌘Space").warning, "⌘Space replaces Spotlight everywhere")
        XCTAssertEqual(combo(0x0800, "⌥E").warning, "⌥E replaces typing a special character everywhere")
        XCTAssertNil(combo(0x1000, "⌃A").warning)
        XCTAssertNil(combo(0x1000 | 0x0800, "⌃⌥D").warning)
        XCTAssertNil(combo(0x0100 | 0x0200, "⇧⌘D").warning)
        XCTAssertNil(combo(0, "F5").warning)
        XCTAssertNil(DictationShortcut.doubleTap(.leftCommand).warning)
        XCTAssertEqual(ModifierKey.leftCommand.display, "Left ⌘")
    }

    func testShortcutDisplay() {
        XCTAssertEqual(DictationShortcut.default.display, "Double Right ⇧")
        let saved = try! JSONEncoder().encode(DictationShortcut.combination(keyCode: 2, carbonModifiers: 6144, display: "⌃⌥D"))
        XCTAssertEqual(try JSONDecoder().decode(DictationShortcut.self, from: saved).display, "⌃⌥D")
    }
}

final class TypingEditTests: XCTestCase {
    func testTextThatGrowsIsTypedOnceWithItsSpaces() {
        XCTAssert(typingEdit(to: "Hi Sarah, ", typed: "") == (0, "Hi Sarah,"))
        XCTAssert(typingEdit(to: "Hi Sarah, thanks ", typed: "Hi Sarah,") == (0, " thanks"))
        XCTAssert(typingEdit(to: "Hi Sarah, thanks", typed: "Hi Sarah, thanks") == (0, ""))
        XCTAssert(typingEdit(to: "Hi Sarah, thanks", typed: "Hi Sarah, than") == (0, "ks"))     // a word that grows
    }

    /// A revision (not expected from the model) is erased back to where it differs and retyped.
    func testRevisedTextIsErasedAndRetyped() {
        XCTAssert(typingEdit(to: "Hello there", typed: "Help") == (1, "lo there"))
        XCTAssert(typingEdit(to: "Yes", typed: "No") == (2, "Yes"))
    }
}

@MainActor
final class DirectDictationTests: XCTestCase {
    var factory = FakeFactory()
    var mic = FakeMic()
    var clip = FakeClipboard()
    var typer = FakeTyper()
    var access = FakeAccessibility()

    func ready(granted: Bool = true) async -> DictationController {
        let c = DictationController(model: .nemotron160, loader: FakeLoader(factory: factory), microphone: mic,
                                    permission: FakePermission(granted: granted), clipboard: clip)
        c.enableDirectDictation(typer: typer, accessibility: access)
        c.loadModel()
        for _ in 0..<50 where c.phase == .preparing { try? await Task.sleep(for: .milliseconds(5)) }
        return c
    }

    func settle() async { for _ in 0..<20 { await Task.yield() }; try? await Task.sleep(for: .milliseconds(20)) }

    /// Only finished words are typed, in order, each once; Stop types the last word and keeps the text in History.
    func testTypesFinishedWordsThenTheLastOnStop() async {
        let c = await ready()
        let r = await c.startDirectDictation()
        XCTAssertEqual(r, .started)
        XCTAssertTrue(c.isDirectDictating); XCTAssertTrue(mic.running)
        XCTAssertEqual(c.phase, .idle)                                  // no panel dictation
        let s = factory.sessions[0]
        s.say("", "Hi")                                                 // the word in progress is typed at once
        await settle()
        XCTAssertEqual(typer.all, "Hi")
        s.say("Hi ", "Sar")
        await settle()
        XCTAssertEqual(typer.all, "Hi Sar")
        s.say("Hi Sarah, thanks ", "for")                               // a pause here: "for" is already there
        await settle()
        XCTAssertEqual(typer.all, "Hi Sarah, thanks for")
        s.final = "Hi Sarah, thanks for the draft."
        await c.stopDirectDictation()
        XCTAssertEqual(typer.all, "Hi Sarah, thanks for the draft. ")                 // a space after the last word
        XCTAssertEqual(typer.typed, ["Hi", " Sar", "ah, thanks for", " the draft.", " "])
        XCTAssertFalse(c.isDirectDictating); XCTAssertFalse(mic.running)
        XCTAssertEqual(c.history.first?.text, "Hi Sarah, thanks for the draft.")
        XCTAssertEqual(c.history.first?.typed, true)
    }

    func testNeedsAccessibility() async {
        let c = await ready()
        access.isGranted = false
        let r = await c.startDirectDictation()
        XCTAssertEqual(r, .needsAccessibility)
        XCTAssertFalse(mic.running)
        c.showAccessibilityNeeded()
        XCTAssertEqual(c.phase, .accessibilityNeeded)
        c.panelClosed()
        XCTAssertEqual(c.phase, .idle)
    }

    func testNeedsThePanelWhileGettingReadyOrWithoutMicrophone() async {
        let c = DictationController(model: .nemotron160, loader: FakeLoader(factory: factory), microphone: mic,
                                    permission: FakePermission(), clipboard: clip)
        c.enableDirectDictation(typer: typer, accessibility: access)
        c.loadModel()                                                   // still preparing
        let r1 = await c.startDirectDictation()
        XCTAssertEqual(r1, .needsPanel)
        let d = await ready(granted: false)
        let r2 = await d.startDirectDictation()
        XCTAssertEqual(r2, .needsPanel)
        XCTAssertFalse(mic.running)
    }

    /// A paused panel dictation is untouched by a direct dictation: same text, same session, elapsed time unchanged.
    func testPausedPanelDictationSurvives() async {
        let c = await ready()
        await c.panelOpened()
        factory.sessions[0].say("panel words ", "")
        await settle()
        c.pause(); await settle()
        c.panelClosed()
        let before = c.elapsed
        _ = await c.startDirectDictation()
        factory.sessions[1].say("typed ", "")
        await settle()
        await c.stopDirectDictation()
        XCTAssertEqual(c.phase, .paused); XCTAssertEqual(c.text, "panel words "); XCTAssertEqual(c.elapsed, before)
        await c.panelOpened()
        c.resume()
        XCTAssertEqual(factory.sessions[0].resumes, 1)
    }

    func testIgnoredWhileThePanelIsOpen() async {
        let c = await ready()
        await c.panelOpened()
        let r = await c.startDirectDictation()
        XCTAssertEqual(r, .ignored)
        XCTAssertEqual(factory.sessions.count, 1)
    }

    func testModelCantChangeDuringDirectDictation() async {
        let c = await ready()
        _ = await c.startDirectDictation()
        c.selectModel(.nemotron560)
        XCTAssertEqual(c.model, .nemotron160)
    }

    func testQuitDuringDirectDictationKeepsWhatWasTyped() async {
        let c = await ready()
        _ = await c.startDirectDictation()
        factory.sessions[0].say("so far ", "x")
        await settle()
        c.finishForQuit()
        XCTAssertEqual(c.history.first?.text, "so far x"); XCTAssertEqual(c.history.first?.typed, true)
        XCTAssertFalse(mic.running)
    }

    func testOldHistoryWithoutTheTypedFieldStillLoads() throws {
        let json = #"[{"id":"8081B11B-60BB-4D5A-9460-3AB9C268EAF6","text":"old","date":"2026-10-04T09:18:10Z","model":"nemotron160","copied":true}]"#
        let d = JSONDecoder(); d.dateDecodingStrategy = .iso8601
        let entries = try d.decode([HistoryEntry].self, from: Data(json.utf8))
        XCTAssertEqual(entries.first?.typed, false); XCTAssertEqual(entries.first?.edited, false)
    }

    /// Nothing said: nothing typed, not even the space.
    func testNothingSaidTypesNothing() async {
        let c = await ready()
        _ = await c.startDirectDictation()
        factory.sessions[0].final = ""
        await c.stopDirectDictation()
        XCTAssertEqual(typer.typed, [])
    }

    /// The microphone can't start (a headset mid-switch): the direct dictation ends, the failure shows, and
    /// the next one can start.
    func testMicrophoneFailureEndsTheDirectDictation() async {
        final class BrokenMic: AudioInput, @unchecked Sendable {
            var fail = true
            func start(_ onSamples: @escaping ([Float]) -> Void) throws { if fail { throw EngineError.io("formats don't match") } }
            func stop() {}
        }
        let broken = BrokenMic()
        let c = DictationController(model: .nemotron160, loader: FakeLoader(factory: factory), microphone: broken, permission: FakePermission(),
                                    clipboard: clip, microphoneQueue: DispatchQueue(label: "test-mic"))
        c.enableDirectDictation(typer: typer, accessibility: access)
        c.loadModel()
        for _ in 0..<50 where c.phase == .preparing { try? await Task.sleep(for: .milliseconds(5)) }
        _ = await c.startDirectDictation()
        try? await Task.sleep(for: .milliseconds(100))
        XCTAssertFalse(c.isDirectDictating); XCTAssertTrue(c.directDictationFailed)
        broken.fail = false
        let r = await c.startDirectDictation()
        XCTAssertEqual(r, .started)
    }

    /// After the microphone failed in the panel, opening it again tries again.
    func testPanelRetriesAfterAMicrophoneFailure() async {
        final class FlakyMic: AudioInput, @unchecked Sendable {
            var fail = true
            func start(_ onSamples: @escaping ([Float]) -> Void) throws { if fail { throw EngineError.io("formats don't match") } }
            func stop() {}
        }
        let flaky = FlakyMic()
        let c = DictationController(model: .nemotron160, loader: FakeLoader(factory: factory), microphone: flaky, permission: FakePermission(),
                                    clipboard: clip)
        c.loadModel()
        for _ in 0..<50 where c.phase == .preparing { try? await Task.sleep(for: .milliseconds(5)) }
        await c.panelOpened()
        guard case .failed = c.phase else { return XCTFail("phase \(c.phase)") }
        c.panelClosed()
        flaky.fail = false
        await c.panelOpened()
        XCTAssertEqual(c.phase, .listening)
    }

    /// If the text is ever revised (not just extended), the field is corrected with Backspace instead of
    /// typing stopping.
    func testRevisionIsCorrected() async {
        let c = await ready()
        _ = await c.startDirectDictation()
        let s = factory.sessions[0]
        s.say("Help", "")
        await settle()
        s.say("Hello there", "")
        await settle()
        XCTAssertEqual(typer.all, "Hello there")
        XCTAssertEqual(typer.typed, ["Help", "⌫1", "lo there"])
    }
}
