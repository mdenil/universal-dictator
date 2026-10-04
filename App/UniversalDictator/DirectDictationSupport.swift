import AppKit
import ApplicationServices
import Carbon
import Observation
import UniversalDictatorKit

// The app's side of direct dictation: typing with keyboard events, the Accessibility permission, the shortcut
// settings, and listening for the shortcut.

/// Types text into whatever has keyboard focus by posting keyboard events that carry the characters, one
/// character per key press, on a serial queue (in order, without holding up the caller).
final class KeyboardTyper: TextTyper, @unchecked Sendable {
    enum Route: String, CaseIterable { case hidTap, sessionTap, frontmostApp }

    let queue = DispatchQueue(label: "typing", qos: .userInitiated)
    var route: Route = .hidTap

    func type(_ text: String) {
        let route = self.route
        queue.async { Self.post(text, route: route) }
    }

    static func post(_ text: String, route: Route) {
        let source = CGEventSource(stateID: .privateState)
        let pid = NSWorkspace.shared.frontmostApplication?.processIdentifier
        for ch in text {
            let units = Array(String(ch).utf16)
            for down in [true, false] {
                guard let e = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: down) else { continue }
                e.flags = []                                    // no modifiers, whatever keys are held
                e.keyboardSetUnicodeString(stringLength: units.count, unicodeString: units)
                switch route {
                case .hidTap: e.post(tap: .cghidEventTap)
                case .sessionTap: e.post(tap: .cgSessionEventTap)
                case .frontmostApp: if let pid { e.postToPid(pid) }
                }
            }
        }
    }

    /// Whether macOS is blocking typed events (a password field, or "secure keyboard entry").
    static var secureInputIsOn: Bool { IsSecureEventInputEnabled() }
}

struct SystemAccessibility: AccessibilityPermission {
    var isGranted: Bool { AXIsProcessTrusted() }

    /// macOS's own prompt, which leads to System Settings › Privacy & Security › Accessibility.
    static func prompt() {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        _ = AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
    }
}

/// The direct-dictation settings, saved in UserDefaults, and whether Accessibility is allowed (checked every
/// two seconds while it isn't, so the Settings row and the shortcut come alive as soon as it is).
@MainActor @Observable
final class DirectSettings {
    var shortcut: DictationShortcut {
        didSet { if let d = try? JSONEncoder().encode(shortcut) { UserDefaults.standard.set(d, forKey: "directShortcut") } }
    }
    var mode: DirectMode {
        didSet { UserDefaults.standard.set(mode.rawValue, forKey: "directMode") }
    }
    private(set) var accessibilityGranted = AXIsProcessTrusted()
    /// While the Settings shortcut field records a new shortcut, the current one does nothing.
    var isRecording = false
    @ObservationIgnored var poll: Timer?

    init() {
        let d = UserDefaults.standard
        shortcut = d.data(forKey: "directShortcut").flatMap { try? JSONDecoder().decode(DictationShortcut.self, from: $0) } ?? .default
        mode = d.string(forKey: "directMode").flatMap(DirectMode.init(rawValue:)) ?? .toggle
        poll = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.refreshAccessibility() }
        }
    }

    func refreshAccessibility() {
        let now = AXIsProcessTrusted()
        if now != accessibilityGranted { accessibilityGranted = now }
    }
}

/// Listens for the shortcut anywhere and reports `down` and `up`: for a double tap, the second press and its
/// release; for a key combination, its press and release.
@MainActor
final class HotkeyCenter {
    let settings: DirectSettings
    var onDown: () -> Void = {}
    var onUp: () -> Void = {}

    var detector: DoubleTapDetector?
    var monitors: [Any] = []
    var hotKey: EventHotKeyRef?
    var handler: EventHandlerRef?
    var installedFor: (DictationShortcut, Bool)?

    init(settings: DirectSettings) {
        self.settings = settings
        observe()
    }

    /// Re-installs whenever the shortcut or the permission changes (a global event monitor installed before
    /// the permission was granted never receives events).
    func observe() {
        withObservationTracking {
            install(settings.shortcut, granted: settings.accessibilityGranted)
        } onChange: {
            Task { @MainActor [weak self] in self?.observe() }
        }
    }

    func install(_ shortcut: DictationShortcut, granted: Bool) {
        if let i = installedFor, i.0 == shortcut, i.1 == granted { return }
        uninstall()
        installedFor = (shortcut, granted)
        switch shortcut {
        case .doubleTap(let key):
            detector = DoubleTapDetector(key: key)
            let handle: (NSEvent) -> Void = { [weak self] e in self?.feed(e) }
            if let g = NSEvent.addGlobalMonitorForEvents(matching: [.flagsChanged, .keyDown, .leftMouseDown, .rightMouseDown], handler: handle) {
                monitors.append(g)
            }
            if let l = NSEvent.addLocalMonitorForEvents(matching: [.flagsChanged, .keyDown, .leftMouseDown, .rightMouseDown],
                                                        handler: { handle($0); return $0 }) {
                monitors.append(l)
            }
        case .combination(let keyCode, let mods, _):
            HotkeyCenter.current = self
            var spec = [EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
                        EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased))]
            InstallEventHandler(GetEventDispatcherTarget(), { _, event, _ in
                let kind = GetEventKind(event)
                Task { @MainActor in
                    guard let c = HotkeyCenter.current, !c.settings.isRecording else { return }
                    if kind == UInt32(kEventHotKeyPressed) { c.onDown() } else { c.onUp() }
                }
                return noErr
            }, 2, &spec, nil, &handler)
            let id = EventHotKeyID(signature: OSType(0x5544_4454), id: 1)       // 'UDDT'
            RegisterEventHotKey(keyCode, mods, id, GetEventDispatcherTarget(), 0, &hotKey)
        }
    }

    func uninstall() {
        monitors.forEach { NSEvent.removeMonitor($0) }
        monitors = []
        detector = nil
        if let h = hotKey { UnregisterEventHotKey(h); hotKey = nil }
        if let h = handler { RemoveEventHandler(h); handler = nil }
    }

    static var current: HotkeyCenter?

    /// Device-dependent modifier bits (which side is down).
    static let sideMask: [ModifierKey: UInt] = [.leftControl: 0x01, .leftShift: 0x02, .rightShift: 0x04, .leftCommand: 0x08,
                                               .rightCommand: 0x10, .leftOption: 0x20, .rightOption: 0x40, .rightControl: 0x2000]

    /// One key event as the detector sees it.
    static func detectorEvent(_ e: NSEvent) -> DoubleTapDetector.Event {
        guard e.type == .flagsChanged, let key = ModifierKey(keyCode: e.keyCode) else { return .other }
        let down = e.modifierFlags.rawValue & (sideMask[key] ?? 0) != 0
        let t = e.timestamp
        return down ? .modifierDown(key, at: t) : .modifierUp(key, at: t)
    }

    func feed(_ e: NSEvent) {
        guard !settings.isRecording, var d = detector else { return }
        // Keys this app types itself (direct dictation) aren't the user's.
        if let src = e.cgEvent?.getIntegerValueField(.eventSourceUnixProcessID), src == Int64(ProcessInfo.processInfo.processIdentifier) { return }
        let out = d.handle(Self.detectorEvent(e))
        detector = d
        switch out {
        case .doubleTapDown: onDown()
        case .doubleTapUp: onUp()
        case nil: break
        }
    }
}
