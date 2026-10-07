import AppKit
import AVFoundation
import CoreAudio
import DictatorCore
import MediaPlayer

/// `--button-probe`: reports everything a headset button press could arrive as, in three phases, each announced
/// by chimes (one, two, three); a falling chime ends it.
///   1. microphone off, no audio playing (30 s)
///   2. microphone off, silence playing, as if the app were a media player (30 s)
///   3. microphone on, so a Bluetooth headset is in its call mode (40 s)
/// Watched: media-key events (an event monitor and an event tap), the media remote commands macOS sends to the
/// app that is "now playing", the microphone mute gesture (AVAudioApplication), and audio device changes.
@MainActor
enum ButtonProbe {
    nonisolated static let start = Date()
    nonisolated static func log(_ s: String) {
        let t = String(format: "%6.2f", Date().timeIntervalSince(start))
        FileHandle.standardError.write("[\(t)] \(s)\n".data(using: .utf8)!)
    }

    static var keep: [Any] = []
    static var playing = true

    static func run() {
        log("button probe: \(ProcessInfo.processInfo.processIdentifier)")
        watchMediaKeys()
        watchRemoteCommands()
        watchMute()
        watchDevices()
        Task {
            try? await Task.sleep(for: .seconds(3))
            await phase(1, "microphone off, nothing playing") {}
            try? await Task.sleep(for: .seconds(30))
            var player: AVAudioPlayer?
            await phase(2, "microphone off, silence playing") { player = playSilence() }
            try? await Task.sleep(for: .seconds(30))
            player?.stop()
            let mic = Microphone()
            mic.onRestart = { log("microphone restarted, input rate \($0.map { "\(Int($0))" } ?? "failed")") }
            var heard = false
            await phase(3, "microphone on") {
                DispatchQueue.global().async {
                    do {
                        try mic.start { x in
                            if !heard, x.contains(where: { $0 != 0 }) { heard = true; log("microphone live (first sound)") }
                        }
                        log("microphone started")
                    } catch { log("microphone failed: \(error)") }
                }
            }
            try? await Task.sleep(for: .seconds(40))
            mic.stop()
            log("microphone stopped; done")
            try? await Task.sleep(for: .seconds(1.5))
            play(StartSound.wav(notes: [(880.0, 110.0), (587.33, 140.0)]))
            try? await Task.sleep(for: .seconds(1))
            exit(0)
        }
    }

    /// Announces a phase with `n` chimes (after `setup`, so in phase 3 the chimes come once the headset is in
    /// call mode and you hear them through it).
    static func phase(_ n: Int, _ what: String, setup: () -> Void) async {
        log("— phase \(n): \(what) — \(devices())")
        setup()
        if n == 3 { try? await Task.sleep(for: .seconds(2.5)) }
        for _ in 0..<n {
            play(StartSound.wav())
            try? await Task.sleep(for: .seconds(0.6))
        }
    }

    static var sounds: [NSSound] = []
    static func play(_ wav: Data) {
        guard let s = NSSound(data: wav) else { return }
        sounds.append(s)
        s.play()
    }

    static func playSilence() -> AVAudioPlayer? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("probe-silence.wav")
        try? WAVWriter.write([Float](repeating: 0, count: 16000), to: url)
        let p = try? AVAudioPlayer(contentsOf: url)
        p?.numberOfLoops = -1
        p?.play()
        log("silence playing: \(p?.isPlaying ?? false)")
        return p
    }

    // MARK: Media keys

    static let keyNames: [Int: String] = [0: "volume up", 1: "volume down", 7: "mute", 16: "play/pause", 17: "next",
                                          18: "previous", 19: "fast", 20: "rewind"]

    static func describe(data1: Int) -> String {
        let code = (data1 & 0xFFFF_0000) >> 16, state = (data1 & 0xFF00) >> 8
        return "\(keyNames[code] ?? "key \(code)") \(state == 0xA ? "down" : state == 0xB ? "up" : "state \(state)")"
    }

    static func watchMediaKeys() {
        let handle: (NSEvent, String) -> Void = { e, how in
            log("media key (\(how) monitor): subtype \(e.subtype.rawValue), \(e.subtype.rawValue == 8 ? describe(data1: e.data1) : "data1 \(e.data1)")")
        }
        if let g = NSEvent.addGlobalMonitorForEvents(matching: .systemDefined, handler: { handle($0, "global") }) { keep.append(g) }
        if let l = NSEvent.addLocalMonitorForEvents(matching: .systemDefined, handler: { handle($0, "local"); return $0 }) { keep.append(l) }
        let mask = CGEventMask(1 << 14)                                   // NX_SYSDEFINED
        let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .listenOnly,
                                    eventsOfInterest: mask, callback: { _, type, event, _ in
            if type.rawValue == 14, let e = NSEvent(cgEvent: event) {
                ButtonProbe.log("media key (event tap): subtype \(e.subtype.rawValue), \(e.subtype.rawValue == 8 ? ButtonProbe.describe(data1: e.data1) : "data1 \(e.data1)")")
            }
            return Unmanaged.passUnretained(event)
        }, userInfo: nil)
        if let tap {
            let src = CFMachPortCreateRunLoopSource(nil, tap, 0)
            CFRunLoopAddSource(CFRunLoopGetMain(), src, .commonModes)
            CGEvent.tapEnable(tap: tap, enable: true)
            keep.append(tap)
            log("event tap on")
        } else {
            log("event tap refused")
        }
    }

    // MARK: Remote commands (what macOS sends the "now playing" app)

    static func watchRemoteCommands() {
        let c = MPRemoteCommandCenter.shared()
        let commands: [(String, MPRemoteCommand)] = [
            ("toggle play/pause", c.togglePlayPauseCommand), ("play", c.playCommand), ("pause", c.pauseCommand),
            ("stop", c.stopCommand), ("next track", c.nextTrackCommand), ("previous track", c.previousTrackCommand),
            ("skip forward", c.skipForwardCommand), ("skip backward", c.skipBackwardCommand),
            ("seek forward", c.seekForwardCommand), ("seek backward", c.seekBackwardCommand),
        ]
        for (name, cmd) in commands {
            cmd.isEnabled = true
            cmd.addTarget { _ in
                log("remote command: \(name)")
                switch name {
                case "toggle play/pause": playing.toggle()
                case "play": playing = true
                case "pause", "stop": playing = false
                default: break
                }
                MPNowPlayingInfoCenter.default().playbackState = playing ? .playing : .paused
                return .success
            }
        }
        MPNowPlayingInfoCenter.default().nowPlayingInfo = [MPMediaItemPropertyTitle: "Button probe",
                                                           MPNowPlayingInfoPropertyPlaybackRate: 1.0]
        MPNowPlayingInfoCenter.default().playbackState = .playing
        log("now playing: claimed")
    }

    // MARK: The microphone mute gesture

    static func watchMute() {
        do {
            try AVAudioApplication.shared.setInputMuteStateChangeHandler { muted in
                log("mute gesture: \(muted ? "muted" : "unmuted")")
                return true
            }
            log("mute gesture handler on")
        } catch {
            log("mute gesture handler refused: \(error)")
        }
        keep.append(NotificationCenter.default.addObserver(forName: AVAudioApplication.inputMuteStateChangeNotification,
                                                           object: nil, queue: nil) { n in
            log("mute notification: \(n.userInfo ?? [:])")
        })
    }

    // MARK: Devices

    static func watchDevices() {
        for selector in [kAudioHardwarePropertyDefaultInputDevice, kAudioHardwarePropertyDefaultOutputDevice] {
            var addr = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                                  mElement: kAudioObjectPropertyElementMain)
            AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &addr, .main) { _, _ in
                log("default device changed: \(devices())")
            }
        }
    }

    nonisolated static func devices() -> String {
        func name(_ selector: AudioObjectPropertySelector) -> String {
            var device = AudioObjectID(0)
            var size = UInt32(MemoryLayout<AudioObjectID>.size)
            var addr = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                                  mElement: kAudioObjectPropertyElementMain)
            guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &device) == noErr else { return "?" }
            var n: Unmanaged<CFString>?
            size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
            addr.mSelector = kAudioObjectPropertyName
            _ = AudioObjectGetPropertyData(device, &addr, 0, nil, &size, &n)
            var rate = Float64(0)
            size = UInt32(MemoryLayout<Float64>.size)
            addr.mSelector = kAudioDevicePropertyNominalSampleRate
            _ = AudioObjectGetPropertyData(device, &addr, 0, nil, &size, &rate)
            return "\(n?.takeRetainedValue() as String? ?? "?") at \(Int(rate)) Hz"
        }
        return "in: \(name(kAudioHardwarePropertyDefaultInputDevice)), out: \(name(kAudioHardwarePropertyDefaultOutputDevice))"
    }
}
