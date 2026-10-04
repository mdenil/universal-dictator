import DictatorCore
import SwiftUI
import UniversalDictatorKit

/// The History window (design: Figma "History (draft)", window 3): every saved dictation, newest first,
/// with search, and Copy, Edit and Delete per entry.
struct HistoryView: View {
    @Bindable var controller: DictationController
    @State var search = ""
    /// The entry being edited and its text so far.
    @State var editingID: UUID?
    @State var draft = ""
    @FocusState var editorFocused: Bool

    var shown: [HistoryEntry] {
        search.isEmpty ? controller.history : controller.history.filter { $0.text.localizedCaseInsensitiveContains(search) }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Spacer()
                TextField("Search dictations", text: $search)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 180)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            Divider()
            if shown.isEmpty {
                Text(controller.history.isEmpty ? "No dictations yet" : "No matches")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(shown) { entry in
                            row(entry)
                            Divider()
                        }
                    }
                }
            }
            Divider()
            Text("Saved on this Mac · keeps the last \(controller.historyLimit) · change in Settings")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
        }
        .frame(minWidth: 480, idealWidth: 520, minHeight: 300, idealHeight: 420)
    }

    func row(_ entry: HistoryEntry) -> some View {
        let editing = editingID == entry.id
        return HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(meta(entry)).font(.system(size: 11)).foregroundStyle(.secondary)
                if editing {
                    TextEditor(text: $draft)
                        .font(.system(size: 13))
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: 54)
                        .padding(4)
                        .background(RoundedRectangle(cornerRadius: 6).fill(Color(nsColor: .textBackgroundColor)))
                        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.accentColor, lineWidth: 2))
                        .focused($editorFocused)
                } else {
                    Text(entry.text).font(.system(size: 13)).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            HStack(spacing: 8) {
                if editing {
                    Button("Done") {
                        controller.editHistory(entry.id, text: draft)
                        editingID = nil
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    Button("Cancel") { editingID = nil }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                        .keyboardShortcut(.cancelAction)
                } else {
                    Button("Copy") { controller.copyFromHistory(entry) }
                    Button("Edit") {
                        draft = entry.text
                        editingID = entry.id
                        editorFocused = true
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    Button("Delete") { controller.deleteHistory(entry.id) }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.leading, 16)
        .padding(.trailing, 12)
        .padding(.vertical, 10)
        .background(editing ? Color.accentColor.opacity(0.05) : Color.clear)
    }

    /// "Today at 09:10 · Fast · copied · edited" (or "· typed" for a direct dictation).
    func meta(_ entry: HistoryEntry) -> String {
        var parts = [Self.dateFormatter.string(from: entry.date), entry.model.displayName]
        if entry.copied { parts.append("copied") }
        if entry.typed { parts.append("typed") }
        if entry.edited { parts.append("edited") }
        return parts.joined(separator: " · ")
    }

    static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        f.doesRelativeDateFormatting = true
        return f
    }()
}
