import DictatorCore
import SwiftUI
import UniversalDictatorKit

/// The Settings window (design: Figma "Settings (draft)"): labels on the left, controls on the right.
struct SettingsView: View {
    @Bindable var controller: DictationController
    @State var confirmingClear = false

    var body: some View {
        VStack(spacing: 12) {
            row("Model") {
                Picker("Model", selection: Binding(get: { controller.model }, set: { choice in
                    controller.selectModel(choice)
                    UserDefaults.standard.set(choice.rawValue, forKey: "model")
                })) {
                    ForEach(ModelChoice.allCases) { Text($0.displayName).tag($0) }
                }
                .labelsHidden()
                .fixedSize()
                .disabled(!controller.canChangeModel)
                .help(modelHelp)
            }
            row("History") {
                Button("Clear History…") { confirmingClear = true }
                    .disabled(controller.history.isEmpty)
                Picker("History", selection: Binding(get: { controller.historyLimit }, set: { limit in
                    controller.historyLimit = limit
                    UserDefaults.standard.set(limit, forKey: "historyLimit")
                })) {
                    ForEach(DictationController.historyLimits, id: \.self) { Text("\($0)").tag($0) }
                }
                .labelsHidden()
                .fixedSize()
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 22)
        .frame(width: 360)
        .alert("Delete all \(controller.history.count) dictations?", isPresented: $confirmingClear) {
            Button("Delete", role: .destructive) { controller.clearHistory() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This can’t be undone.")
        }
    }

    func row<Controls: View>(_ label: String, @ViewBuilder controls: () -> Controls) -> some View {
        HStack(spacing: 8) {
            Text(label)
            Spacer()
            controls()
        }
    }

    /// Why the Model menu is disabled; nothing otherwise (the choices are just Fast and Accurate).
    var modelHelp: String {
        switch controller.phase {
        case .paused: return "A dictation is paused. Stop it or copy it to change models."
        case .listening: return "Stop the dictation to change models."
        case .preparing: return "The model is getting ready."
        default: return ""
        }
    }
}
