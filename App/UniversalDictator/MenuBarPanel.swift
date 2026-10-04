import AppKit
import OSLog
import SwiftUI
import UniversalDictatorKit

let log = Logger(subsystem: "dev.universaldictator.UniversalDictator", category: "panel")

/// The menu-bar item and the panel that drops down under it. Clicking the item toggles the panel; the
/// panel also closes when it loses focus, and after Copy.
@MainActor
final class MenuBarPanel: NSObject, NSWindowDelegate {
    let controller: DictationController
    let direct: DirectSettings
    let item: NSStatusItem
    let panel: DropDownPanel
    var host: NSHostingView<PanelView>!

    init(controller: DictationController, direct: DirectSettings) {
        self.controller = controller
        self.direct = direct
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        panel = DropDownPanel()
        super.init()
        host = NSHostingView(rootView: PanelView(controller: controller, close: { [weak self] in self?.close() },
                                                 openSettings: { [weak self] in self?.showSettings() }))
        panel.contentView = PanelBackground(content: host)
        panel.delegate = self
        item.button?.target = self
        item.button?.action = #selector(clicked)
        // Right clicks on the icon open the app menu. Caught by a monitor so the button keeps its default
        // left-click behaviour (making it act on mouse-up instead broke opening the panel).
        rightClickMonitor = NSEvent.addLocalMonitorForEvents(matching: [.rightMouseDown]) { [weak self] event in
            guard let self, let w = self.item.button?.window, event.window === w else { return event }
            self.showAppMenu()
            return nil
        }
        item.button?.toolTip = "Universal Dictator"
        updateIcon()
        // When the icon is hidden or shown (for example pushed behind the notch), log where it is and which
        // app is in front, to find out why it disappears while dictating.
        if let w = item.button?.window {
            occlusionObserver = NotificationCenter.default.addObserver(forName: NSWindow.didChangeOcclusionStateNotification, object: w,
                                                                       queue: .main) { [weak self] _ in
                MainActor.assumeIsolated {
                    guard let self else { return }
                    let front = NSWorkspace.shared.frontmostApplication?.localizedName ?? "?"
                    log.notice("icon changed: \(self.describeIcon(), privacy: .public); phase \(String(describing: self.controller.phase), privacy: .public); front app \(front, privacy: .public)")
                }
            }
        }
    }

    /// When the panel last closed: clicking the menu-bar item while the panel is open first takes focus
    /// from the panel (closing it), and that same click must not reopen it.
    var lastClose = Date.distantPast

    var rightClickMonitor: Any?
    var occlusionObserver: Any?

    /// Left click toggles the panel; Control-click shows the app menu with Quit (right click: see the monitor).
    @objc func clicked() {
        let e = NSApp.currentEvent
        log.notice("icon clicked (event \(e.map { String(describing: $0.type) } ?? "none", privacy: .public))")
        if e?.modifierFlags.contains(.control) == true { showAppMenu() } else { toggle() }
    }

    /// The right-click menu (design: Figma "History (draft)"): the five latest dictations (choosing one
    /// copies it), Show All History…, Settings…, Quit.
    func showAppMenu() {
        close()
        let menu = NSMenu()
        menu.addItem(.sectionHeader(title: "Recent dictations"))
        if controller.history.isEmpty {
            let none = NSMenuItem(title: "No dictations yet", action: nil, keyEquivalent: "")
            none.isEnabled = false
            menu.addItem(none)
        }
        for entry in controller.history.prefix(5) {
            let i = NSMenuItem(title: Self.firstLine(entry.text), action: #selector(copyRecent(_:)), keyEquivalent: "")
            i.target = self
            i.representedObject = entry.id
            i.badge = NSMenuItemBadge(string: Self.age(entry.date))
            menu.addItem(i)
        }
        menu.addItem(.separator())
        let all = NSMenuItem(title: "Show All History…", action: #selector(showHistory), keyEquivalent: "y")
        all.target = self
        all.isEnabled = !controller.history.isEmpty
        menu.addItem(all)
        let settings = NSMenuItem(title: "Settings…", action: #selector(showSettings), keyEquivalent: ",")
        settings.target = self
        menu.addItem(settings)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Universal Dictator", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        quit.target = NSApp
        menu.addItem(quit)
        menu.autoenablesItems = false
        item.menu = menu                     // shown under the icon like any menu-bar menu, then detached so
        item.button?.performClick(nil)       // left clicks keep opening the panel
        item.menu = nil
    }

    /// A recent dictation chosen in the menu: copied, and the icon shows a check mark for a moment.
    @objc func copyRecent(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? UUID, let entry = controller.history.first(where: { $0.id == id }) else { return }
        controller.copyFromHistory(entry)
        item.button?.image = Self.symbol("checkmark", color: nil)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in self?.updateIcon() }
    }

    static func firstLine(_ text: String) -> String {
        let line = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? text
        return line.count > 40 ? String(line.prefix(39)).trimmingCharacters(in: .whitespaces) + "…" : line
    }

    /// "now", "2 min", "3 h", "Yesterday", or the date.
    static func age(_ date: Date) -> String {
        let s = Date().timeIntervalSince(date)
        if s < 60 { return "now" }
        if s < 3600 { return "\(Int(s / 60)) min" }
        if Calendar.current.isDateInToday(date) { return "\(Int(s / 3600)) h" }
        if Calendar.current.isDateInYesterday(date) { return "Yesterday" }
        return date.formatted(.dateTime.day().month(.abbreviated))
    }

    // MARK: Windows

    var settingsWindow: NSPanel?
    var historyWindow: NSPanel?

    @objc func showSettings() {
        if settingsWindow == nil {
            settingsWindow = Self.window(title: "Settings", content: SettingsView(controller: controller, direct: direct), resizable: false)
        }
        Self.present(settingsWindow!)
    }

    @objc func showHistory() {
        if historyWindow == nil {
            historyWindow = Self.window(title: "History", content: HistoryView(controller: controller), resizable: true)
        }
        Self.present(historyWindow!)
    }

    /// A titled window that takes the keyboard without activating the app, like the drop-down panel (an
    /// active menu-bar app can push status icons behind the notch).
    static func window<Content: View>(title: String, content: Content, resizable: Bool) -> NSPanel {
        var style: NSWindow.StyleMask = [.titled, .closable, .nonactivatingPanel]
        if resizable { style.insert(.resizable) }
        let w = NSPanel(contentRect: .zero, styleMask: style, backing: .buffered, defer: true)
        w.title = title
        w.contentViewController = NSHostingController(rootView: content)
        w.isReleasedWhenClosed = false
        w.hidesOnDeactivate = false
        w.becomesKeyOnlyIfNeeded = false
        w.center()
        return w
    }

    static func present(_ w: NSPanel) {
        w.orderFrontRegardless()
        w.makeKey()
    }

    @objc func toggle() {
        // A click during a direct dictation stops it (typing the last words) and opens nothing.
        if controller.isDirectDictating { Task { await controller.stopDirectDictation() }; return }
        if panel.isVisible { close() } else if Date().timeIntervalSince(lastClose) > 0.3 { open() }
    }

    // MARK: Direct dictation

    /// Whether the shortcut is held down (Hold to talk stops when it's released, even mid-start).
    var shortcutHeld = false

    func shortcutDown() {
        shortcutHeld = true
        switch direct.mode {
        case .toggle:
            if controller.isDirectDictating { Task { await controller.stopDirectDictation() } } else { startDirect() }
        case .holdToTalk:
            if !controller.isDirectDictating { startDirect() }
        }
    }

    /// A single tap of the double-tap key: in Toggle mode it stops a running direct dictation. True if used.
    func shortcutTap() -> Bool {
        guard direct.mode == .toggle, controller.isDirectDictating else { return false }
        Task { await controller.stopDirectDictation() }
        return true
    }

    func shortcutUp() {
        shortcutHeld = false
        if direct.mode == .holdToTalk, controller.isDirectDictating { Task { await controller.stopDirectDictation() } }
    }

    func startDirect() {
        if KeyboardTyper.secureInputIsOn {
            log.notice("direct dictation: secure input is on, not starting")
            item.button?.image = Self.symbol("exclamationmark.triangle", color: nil)
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in self?.updateIcon() }
            return
        }
        if panel.isVisible { close() }
        Task {
            let result = await controller.startDirectDictation()
            log.notice("direct dictation start: \(String(describing: result), privacy: .public)")
            switch result {
            case .started:
                if direct.mode == .holdToTalk, !shortcutHeld { await controller.stopDirectDictation() }
            case .needsPanel:
                open()
            case .needsAccessibility:
                controller.showAccessibilityNeeded()
                open(startDictation: false)
            case .ignored:
                break
            }
        }
    }

    /// Where the panel goes: under the menu-bar icon, inside the screen.
    func panelFrame() -> NSRect? {
        guard let button = item.button, let window = button.window else { return nil }
        let iconFrame = window.convertToScreen(button.convert(button.bounds, to: nil))
        // The SwiftUI content's own size (the material view around it reports zero).
        var size = host.fittingSize
        if size.width < 100 || size.height < 100 { size = NSSize(width: 360, height: 238) }
        let screen = window.screen?.visibleFrame ?? NSScreen.main!.visibleFrame
        var x = iconFrame.midX - size.width / 2
        x = min(max(x, screen.minX + 8), screen.maxX - size.width - 8)
        return NSRect(x: x, y: iconFrame.minY - size.height - 6, width: size.width, height: size.height)
    }

    /// For diagnosing placement without showing anything: the icon, screen and panel frames.
    func describeGeometry() -> String {
        let button = item.button
        let iconFrame = button?.window.map { $0.convertToScreen(button!.convert(button!.bounds, to: nil)) }
        return "icon \(String(describing: iconFrame)), screen \(String(describing: button?.window?.screen?.visibleFrame)), " +
            "content fitting \(String(describing: host.fittingSize)), panel \(String(describing: panelFrame()))"
    }

    /// For diagnosing a hidden icon: where the status item's window is, and whether any of it is on screen.
    func describeIcon() -> String {
        guard let w = item.button?.window else { return "icon: no window" }
        let screen = w.screen ?? NSScreen.main
        let notch = screen.map { s -> String in
            let l = s.auxiliaryTopLeftArea?.maxX ?? 0, r = s.auxiliaryTopRightArea?.minX ?? 0
            return "notch x \(Int(l))–\(Int(r))"
        } ?? "no screen"
        return "icon frame x \(Int(w.frame.minX))–\(Int(w.frame.maxX)) y \(Int(w.frame.minY)), visible \(w.isVisible), " +
            "occlusion-visible \(w.occlusionState.contains(.visible)), \(notch)"
    }

    func open(startDictation: Bool = true) {
        guard let frame = panelFrame(), frame.width > 0, frame.height > 0 else { log.error("open: no usable frame"); return }
        let t0 = Date()
        log.notice("open at \(String(describing: frame), privacy: .public)")
        panel.setFrame(frame, display: true)
        // The panel takes keyboard focus without activating the app, like Spotlight: the app's menus never
        // replace the frontmost app's (which on a notched MacBook pushes status icons, ours included, behind
        // the notch), and closing the panel leaves the previous app in front, ready for ⌘V.
        panel.makeKeyAndOrderFront(nil)
        log.notice("open: shown after \(Int(Date().timeIntervalSince(t0) * 1000)) ms")
        item.button?.highlight(true)
        guard startDictation else { return }
        Task { [weak self] in
            guard let self, self.panel.isVisible else { log.notice("open: panel already gone, not starting"); return }
            log.notice("open: starting after \(Int(Date().timeIntervalSince(t0) * 1000)) ms")
            await self.controller.panelOpened()
        }
    }

    func close() {
        guard panel.isVisible else { return }
        log.notice("close (phase \(String(describing: self.controller.phase), privacy: .public))")
        panel.orderOut(nil)
        lastClose = Date()
        item.button?.highlight(false)
        controller.panelClosed()
    }

    func windowDidResignKey(_ notification: Notification) {
        log.notice("panel lost focus")
        close()
    }

    /// A menu-bar symbol: a template (drawn by macOS to suit the menu bar) or in a fixed colour.
    static func symbol(_ name: String, color: NSColor?) -> NSImage? {
        guard let base = NSImage(systemSymbolName: name, accessibilityDescription: "Universal Dictator") else { return nil }
        guard let color else { base.isTemplate = true; return base }
        let image = base.withSymbolConfiguration(NSImage.SymbolConfiguration(paletteColors: [color]))
        image?.isTemplate = false
        return image
    }

    /// Whether the microphone was live at the last icon update (a direct dictation plays "Tink" when it goes live).
    var wasLive = false

    /// Keeps the menu-bar icon in step with the dictation (listening, paused, preparing).
    func updateIcon() {
        withObservationTracking {
            // Safety net: the microphone never runs while the panel is hidden.
            if controller.phase == .listening && !panel.isVisible {
                Task { @MainActor [weak self] in self?.controller.panelClosed() }
            }
            log.notice("phase \(String(describing: self.controller.phase), privacy: .public)")
            // One shape; the colour tells the state: as the menu bar (idle, getting ready), red (listening),
            // yellow (paused). Problems that need the user show a warning triangle.
            let image: NSImage?
            // Red only once the microphone is actually live (it takes a moment to wake), so red means "speak".
            let live = controller.isMicrophoneLive
            if live, !wasLive, controller.isDirectDictating {                          // no window to look at
                let played = NSSound(named: "Tink")?.play() ?? false
                log.notice("direct dictation live: sound \(played ? "played" : "failed", privacy: .public)")
            }
            wasLive = live
            switch controller.isDirectDictating ? .listening : controller.phase {
            case .listening: image = Self.symbol("waveform", color: live ? .systemRed : nil)
            case .paused: image = Self.symbol("waveform", color: .systemYellow)
            case .microphoneDenied, .accessibilityNeeded, .failed: image = Self.symbol("exclamationmark.triangle", color: nil)
            default: image = Self.symbol("waveform", color: nil)
            }
            item.button?.image = image
        } onChange: {
            Task { @MainActor [weak self] in self?.updateIcon() }
        }
    }
}

/// A borderless panel that can take keyboard focus (for editing and shortcuts).
final class DropDownPanel: NSPanel {
    init() {
        super.init(contentRect: NSRect(x: 0, y: 0, width: 360, height: 260), styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
                   backing: .buffered, defer: true)
        becomesKeyOnlyIfNeeded = false
        isFloatingPanel = true
        level = .popUpMenu
        hasShadow = true
        backgroundColor = .clear
        isOpaque = false
        isMovable = false
        hidesOnDeactivate = false
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
    }
    override var canBecomeKey: Bool { true }
}

/// The menu material with rounded corners behind the SwiftUI content, resizing with it.
final class PanelBackground: NSVisualEffectView {
    init(content: NSView) {
        super.init(frame: .zero)
        material = .popover
        blendingMode = .behindWindow
        state = .active
        wantsLayer = true
        layer?.cornerRadius = 12
        layer?.masksToBounds = true
        content.translatesAutoresizingMaskIntoConstraints = false
        addSubview(content)
        NSLayoutConstraint.activate([
            content.leadingAnchor.constraint(equalTo: leadingAnchor), content.trailingAnchor.constraint(equalTo: trailingAnchor),
            content.topAnchor.constraint(equalTo: topAnchor), content.bottomAnchor.constraint(equalTo: bottomAnchor),
        ])
    }
    required init?(coder: NSCoder) { fatalError() }
}
