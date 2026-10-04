import DictatorCore
import Foundation

/// A finished dictation, kept in the history.
public struct HistoryEntry: Identifiable, Equatable, Codable, Sendable {
    public let id: UUID
    public var text: String
    public let date: Date
    public let model: ModelChoice
    /// Whether the text was copied (Copy or ⌘C) rather than left when the panel closed.
    public let copied: Bool
    /// Whether the text was changed in the History window.
    public var edited: Bool
    /// Whether it was a direct dictation, typed into another app.
    public let typed: Bool

    public init(id: UUID = UUID(), text: String, date: Date, model: ModelChoice, copied: Bool, edited: Bool = false, typed: Bool = false) {
        self.id = id
        self.text = text
        self.date = date
        self.model = model
        self.copied = copied
        self.edited = edited
        self.typed = typed
    }

    enum CodingKeys: String, CodingKey { case id, text, date, model, copied, edited, typed }

    /// Reads entries saved before a field existed (it takes its default).
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        text = try c.decode(String.self, forKey: .text)
        date = try c.decode(Date.self, forKey: .date)
        model = try c.decode(ModelChoice.self, forKey: .model)
        copied = try c.decode(Bool.self, forKey: .copied)
        edited = try c.decodeIfPresent(Bool.self, forKey: .edited) ?? false
        typed = try c.decodeIfPresent(Bool.self, forKey: .typed) ?? false
    }
}

/// Where the history is kept between launches.
public protocol HistoryStore {
    /// The saved entries, newest first; empty if there are none or they can't be read.
    func load() -> [HistoryEntry]
    func save(_ entries: [HistoryEntry])
}

/// The history as a JSON file (in the app: Application Support, on this Mac only).
public struct FileHistoryStore: HistoryStore {
    public let url: URL

    public init(url: URL) { self.url = url }

    /// `~/Library/Application Support/<bundle id>/history.json`.
    public static func standard(bundleID: String) -> FileHistoryStore {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return FileHistoryStore(url: support.appendingPathComponent(bundleID).appendingPathComponent("history.json"))
    }

    public func load() -> [HistoryEntry] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode([HistoryEntry].self, from: data)) ?? []
    }

    public func save(_ entries: [HistoryEntry]) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(entries) else { return }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }
}

/// The history in memory only (tests and development modes).
public final class MemoryHistoryStore: HistoryStore {
    public var entries: [HistoryEntry]
    public init(_ entries: [HistoryEntry] = []) { self.entries = entries }
    public func load() -> [HistoryEntry] { entries }
    public func save(_ entries: [HistoryEntry]) { self.entries = entries }
}
