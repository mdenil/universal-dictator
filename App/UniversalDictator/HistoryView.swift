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
        }
        .frame(minWidth: 480, idealWidth: 520, minHeight: 300, idealHeight: 420)
    }

    /// One entry: its details and text, and Copy, Edit and Delete stacked on the right. Editing makes the text
    /// itself editable in place (same font, position and line breaks, growing with the text) with a thin outline
    /// drawn around it, and Done and Cancel take the buttons' places, so nothing moves.
    func row(_ entry: HistoryEntry) -> some View {
        let editing = editingID == entry.id
        return HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(meta(entry)).font(.system(size: 11)).foregroundStyle(.secondary)
                InPlaceText(text: editing ? $draft : .constant(entry.text), editing: editing,
                            onCommit: { finishEditing(entry) }, onCancel: { editingID = nil })
                    .frame(maxWidth: .infinity, alignment: .leading)
                .overlay {
                    if editing { RoundedRectangle(cornerRadius: 3).stroke(Color.accentColor, lineWidth: 1.5).padding(-3) }
                }
            }
            VStack(alignment: .trailing, spacing: 4) {
                if editing {
                    Button("Done") { finishEditing(entry) }
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    Button("Cancel") { editingID = nil }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                        .keyboardShortcut(.cancelAction)
                    Text("Delete").hidden()                       // keeps the column's height
                } else {
                    Button("Copy") { controller.copyFromHistory(entry) }
                    Button("Edit") {
                        draft = entry.text
                        editingID = entry.id
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    Button("Delete") { controller.deleteHistory(entry.id) }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                }
            }
            .fixedSize()
        }
        .padding(.leading, 16)
        .padding(.trailing, 12)
        .padding(.vertical, 10)
    }

    func finishEditing(_ entry: HistoryEntry) {
        guard !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        controller.editHistory(entry.id, text: draft)
        editingID = nil
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

/// A history entry's text: one wrapping text field used both for reading (selectable, read-only) and for
/// editing, so starting to edit changes nothing on screen but the outline drawn around it. Return or Done
/// keeps the edit, Esc or Cancel drops it.
struct InPlaceText: NSViewRepresentable {
    @Binding var text: String
    let editing: Bool
    let onCommit: () -> Void
    let onCancel: () -> Void

    func makeNSView(context: Context) -> NSTextField {
        let f = NSTextField(wrappingLabelWithString: text)
        f.font = .systemFont(ofSize: 13)
        f.isBordered = false
        f.drawsBackground = false
        f.focusRingType = .none
        f.isSelectable = true
        f.delegate = context.coordinator
        f.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        return f
    }

    func updateNSView(_ f: NSTextField, context: Context) {
        context.coordinator.parent = self
        if f.currentEditor() == nil, f.stringValue != text { f.stringValue = text }
        if f.isEditable != editing {
            f.isEditable = editing
            if editing { DispatchQueue.main.async { f.window?.makeFirstResponder(f); f.currentEditor()?.moveToEndOfDocument(nil) } }
            else if f.currentEditor() != nil { f.window?.makeFirstResponder(nil) }
        }
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView f: NSTextField, context: Context) -> CGSize? {
        let width = proposal.width ?? 400
        let h = f.cell?.cellSize(forBounds: NSRect(x: 0, y: 0, width: width, height: .greatestFiniteMagnitude)).height ?? 16
        return CGSize(width: width, height: ceil(h))
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, NSTextFieldDelegate {
        var parent: InPlaceText
        init(_ parent: InPlaceText) { self.parent = parent }

        func controlTextDidChange(_ n: Notification) {
            if let f = n.object as? NSTextField { parent.text = f.stringValue }
        }

        func control(_ control: NSControl, textView: NSTextView, doCommandBy selector: Selector) -> Bool {
            if selector == #selector(NSResponder.insertNewline(_:)) { parent.onCommit(); return true }
            if selector == #selector(NSResponder.cancelOperation(_:)) { parent.onCancel(); return true }
            return false
        }
    }
}
