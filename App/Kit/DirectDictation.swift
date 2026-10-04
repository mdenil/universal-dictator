import Foundation

// Direct dictation: a global shortcut that dictates straight into the focused text field of any app, by typing
// each finished word. The panel's behaviour is unchanged; this is a second way in.

/// A modifier key, told apart by side.
public enum ModifierKey: String, CaseIterable, Codable, Sendable {
    case leftShift, rightShift, leftControl, rightControl, leftOption, rightOption, leftCommand, rightCommand

    /// The macOS virtual key code.
    public var keyCode: UInt16 {
        switch self {
        case .leftShift: return 56
        case .rightShift: return 60
        case .leftControl: return 59
        case .rightControl: return 62
        case .leftOption: return 58
        case .rightOption: return 61
        case .leftCommand: return 55
        case .rightCommand: return 54
        }
    }

    public init?(keyCode: UInt16) {
        guard let k = Self.allCases.first(where: { $0.keyCode == keyCode }) else { return nil }
        self = k
    }

    public var symbol: String {
        switch self {
        case .leftShift, .rightShift: return "⇧"
        case .leftControl, .rightControl: return "⌃"
        case .leftOption, .rightOption: return "⌥"
        case .leftCommand, .rightCommand: return "⌘"
        }
    }

    public var side: String { rawValue.hasPrefix("left") ? "Left" : "Right" }
}

/// The direct-dictation shortcut: a double tap of one modifier key, or a key combination.
public enum DictationShortcut: Codable, Equatable, Sendable {
    case doubleTap(ModifierKey)
    /// A key with modifiers: the virtual key code, and the modifiers as Carbon flags (cmdKey, optionKey, ...).
    case combination(keyCode: UInt32, carbonModifiers: UInt32, display: String)

    public static let `default` = DictationShortcut.doubleTap(.leftShift)

    /// "Double Left ⇧", or the combination as macOS writes it ("⌃⌥D").
    public var display: String {
        switch self {
        case .doubleTap(let k): return "Double \(k.side) \(k.symbol)"
        case .combination(_, _, let d): return d
        }
    }
}

/// How the shortcut starts and stops a direct dictation.
public enum DirectMode: String, CaseIterable, Codable, Sendable, Identifiable {
    /// Use the shortcut to start and again to stop.
    case toggle
    /// Dictate while the shortcut is held (for a double tap: Shift, Shift-hold while talking, release).
    case pushToTalk = "holdToTalk"            // the saved setting keeps its old name

    public var id: String { rawValue }
    public var displayName: String { self == .toggle ? "Toggle" : "Push to talk" }
}

/// Recognises a double tap of one modifier key from key events: press, release, then a second press within
/// `interval` of the release. Any other key, any other modifier, or a mouse click in between cancels it, as
/// does holding the first press longer than `interval` (Shift held to type a capital is not a tap). Once the
/// second press is down, other keys no longer cancel it, so its release always arrives.
public struct DoubleTapDetector {
    public enum Event: Equatable {
        case modifierDown(ModifierKey, at: TimeInterval)
        case modifierUp(ModifierKey, at: TimeInterval)
        /// Any non-modifier key, or a mouse click.
        case other
    }

    public enum Output: Equatable {
        /// A single quick press and release (Toggle stops a running dictation with it).
        case tap
        /// The second press: a double tap (Toggle starts with it; Push to talk starts).
        case doubleTapDown
        /// The release of that second press (Push to talk stops).
        case doubleTapUp
    }

    public let key: ModifierKey
    public var interval: TimeInterval = 0.4

    enum State { case idle, firstDown(TimeInterval), firstUp(TimeInterval), secondDown }
    var state = State.idle

    public init(key: ModifierKey) { self.key = key }

    /// Forgets any tap in progress (after a tap has been used, so it can't start a double tap).
    public mutating func reset() { state = .idle }

    public mutating func handle(_ event: Event) -> Output? {
        switch event {
        case .other:
            // While the second press is held (Push to talk), other keys don't matter: the release still stops.
            if case .secondDown = state { return nil }
            state = .idle
            return nil
        case .modifierDown(let k, let t):
            guard k == key else { state = .idle; return nil }
            if case .firstUp(let up) = state, t - up <= interval {
                state = .secondDown
                return .doubleTapDown
            }
            state = .firstDown(t)
            return nil
        case .modifierUp(let k, let t):
            guard k == key else { state = .idle; return nil }
            switch state {
            case .firstDown(let down):
                guard t - down <= interval else { state = .idle; return nil }
                state = .firstUp(t)
                return .tap
            case .secondDown:
                state = .idle
                return .doubleTapUp
            default:
                state = .idle
                return nil
            }
        }
    }
}

/// Types text into whatever has keyboard focus (in the app: keyboard events carrying the text).
public protocol TextTyper {
    func type(_ text: String)
    /// Presses Backspace `count` times.
    func deleteBackward(_ count: Int)
}

/// The Accessibility permission, which typing into other apps and noticing a double tap need.
public protocol AccessibilityPermission {
    var isGranted: Bool { get }
}

/// What starting a direct dictation did.
public enum DirectStartResult: Equatable {
    case started
    /// The model is still getting ready, or the microphone isn't allowed: the panel shows why.
    case needsPanel
    /// Typing into other apps isn't allowed yet.
    case needsAccessibility
    /// Already running, or the panel is open.
    case ignored
}

/// What to type so the field, which holds `typed`, shows `text`: Backspace `delete` times, then type `insert`.
/// The model's text only grows, so normally nothing is deleted and `insert` is the new end (trailing spaces
/// wait for the next word, so the text ends exactly where the dictation does). If the text was revised (not
/// expected), the differing end is erased and retyped.
public func typingEdit(to text: String, typed: String) -> (delete: Int, insert: String) {
    var t = text
    while let last = t.last, last.isWhitespace { t.removeLast() }
    if t.hasPrefix(typed) { return (0, String(t.dropFirst(typed.count))) }
    let common = zip(t, typed).prefix { $0 == $1 }.count
    return (typed.count - common, String(t.dropFirst(common)))
}
