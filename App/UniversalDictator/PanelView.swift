import DictatorCore
import SwiftUI
import UniversalDictatorKit

/// The drop-down panel (design: Figma "Universal Dictator — App UI"). One fixed size for every state, so
/// the panel never jumps under the menu bar.
struct PanelView: View {
    @Bindable var controller: DictationController
    let close: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            content
                .frame(height: 150)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .textBackgroundColor)))
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(controller.phase == .done ? Color.accentColor : Color(nsColor: .separatorColor),
                                                                       lineWidth: controller.phase == .done ? 2 : 1))
            footer.frame(height: 26)
        }
        .padding(.horizontal, 14)
        .padding(.top, 12)
        .padding(.bottom, 14)
        .frame(width: 360)
    }

    // MARK: Header

    var header: some View {
        HStack(spacing: 8) {
            Image(systemName: controller.phase == .listening ? "waveform.badge.mic" : "waveform")
                .foregroundStyle(controller.phase == .listening ? Color.red : Color.primary)
            Text("Universal Dictator").font(.system(size: 13, weight: .semibold))
            Spacer()
            Menu {
                ForEach(ModelChoice.allCases) { choice in
                    Button {
                        controller.selectModel(choice)
                        UserDefaults.standard.set(choice.rawValue, forKey: "model")
                    } label: {
                        Text(choice == controller.model ? "✓ \(choice.displayName)" : choice.displayName)
                        Text(choice.summary)
                    }
                    .disabled(!controller.canChangeModel)
                }
                Divider()
                Button("Quit Universal Dictator") { NSApp.terminate(nil) }
            } label: {
                Text(controller.model.displayName).font(.system(size: 11, weight: .medium))
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help(controller.canChangeModel ? "Choose a model" : "Stop the dictation to change models")
        }
    }

    // MARK: Text area

    @ViewBuilder var content: some View {
        switch controller.phase {
        case .idle, .listening, .paused:
            ScrollViewReader { proxy in
                ScrollView {
                    (Text(controller.committed) + Text(controller.partial).foregroundStyle(.tertiary))
                        .font(.system(size: 14))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(10)
                        .id("end")
                }
                .onChange(of: controller.text) { proxy.scrollTo("end", anchor: .bottom) }
            }
            .overlay {
                if controller.text.isEmpty {
                    Text(controller.phase == .paused ? "Paused" : "Listening…")
                        .font(.system(size: 13)).foregroundStyle(.tertiary)
                }
            }
        case .done:
            TextEditor(text: $controller.editedText)
                .font(.system(size: 14))
                .scrollContentBackground(.hidden)
                .padding(6)
        case .preparing:
            VStack(spacing: 10) {
                Text("Loading \(controller.model.displayName)…").font(.system(size: 13, weight: .medium))
                ProgressView().controlSize(.small)
                Text("The first launch after installing compiles the model for the Neural Engine, which takes about a minute.")
                    .font(.system(size: 11)).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            .padding(14)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .microphoneDenied:
            VStack(spacing: 8) {
                Image(systemName: "mic.slash").font(.system(size: 22)).foregroundStyle(.secondary)
                Text("Microphone access is off").font(.system(size: 13, weight: .semibold))
                Text("Turn on Universal Dictator in System Settings › Privacy & Security › Microphone. Audio never leaves this Mac.")
                    .font(.system(size: 12)).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            .padding(14)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .failed(let message):
            VStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle").font(.system(size: 20)).foregroundStyle(.secondary)
                Text(message).font(.system(size: 12)).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            .padding(14)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    // MARK: Footer

    @ViewBuilder var footer: some View {
        HStack(spacing: 8) {
            switch controller.phase {
            case .listening, .paused:
                if controller.phase == .listening { LevelMeter(level: controller.level) }
                Text(status)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                Spacer()
                Button(controller.copiedSoFar ? "Copied ✓" : "Copy so far") { controller.copySoFar() }
                    .buttonStyle(.borderless)
                    .font(.system(size: 12))
                    .foregroundStyle(controller.copiedSoFar ? Color.green : Color.secondary)
                    .disabled(controller.text.isEmpty)
                if controller.phase == .listening {
                    Button("Pause") { controller.pause() }.keyboardShortcut(.space, modifiers: [])
                } else {
                    Button("Resume") { controller.resume() }.keyboardShortcut(.space, modifiers: [])
                }
                Button("Stop") { Task { await controller.stop() } }
                    .tint(.red)
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [])
            case .done:
                Text("\(controller.wordCount) words · \(clock(controller.elapsed))")
                    .font(.system(size: 12, weight: .medium)).foregroundStyle(.secondary).monospacedDigit()
                Spacer()
                Button("New") { Task { await controller.newDictation() } }
                Button("Copy") {
                    controller.copyAndFinish()
                    close()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: .command)
            case .microphoneDenied:
                Spacer()
                Button("Open System Settings") { controller.openMicrophoneSettings() }.buttonStyle(.borderedProminent)
            case .idle, .preparing, .failed:
                Spacer()
                Button("Quit") { NSApp.terminate(nil) }.buttonStyle(.borderless).font(.system(size: 12))
            }
        }
        .controlSize(.regular)
        .background {
            // ⌘C while dictating: stop, copy everything and close the panel.
            Button("") {
                Task {
                    await controller.stopAndCopy()
                    close()
                }
            }
            .keyboardShortcut("c", modifiers: .command)
            .disabled(controller.phase != .listening && controller.phase != .paused)
            .opacity(0)
            .allowsHitTesting(false)
        }
    }

    var status: String { controller.phase == .paused ? "Paused · \(clock(controller.elapsed))" : clock(controller.elapsed) }

    func clock(_ t: TimeInterval) -> String { String(format: "%d:%02d", Int(t) / 60, Int(t) % 60) }
}

/// Eight bars that rise with the microphone level.
struct LevelMeter: View {
    let level: Float
    let shape: [Float] = [0.35, 0.6, 0.9, 0.7, 1.0, 0.55, 0.8, 0.4]

    var body: some View {
        HStack(spacing: 2) {
            ForEach(shape.indices, id: \.self) { i in
                Capsule()
                    .fill(Color.red)
                    .frame(width: 2.5, height: CGFloat(3 + 13 * shape[i] * max(0.15, level)))
            }
        }
        .frame(width: 34, height: 16)
        .animation(.easeOut(duration: 0.08), value: level)
    }
}
