import AVFoundation
import Foundation

/// Captures the default input device as 16 kHz mono float samples and delivers them in small blocks
/// on a background queue. `stop` delivers the last samples before returning.
///
/// When the input's format changes while capturing (a Bluetooth headset switching to its call mode when its
/// microphone turns on, or the default device changing), macOS stops the engine; the microphone then
/// restarts itself with the new format and reports it through `onRestart`.
public final class Microphone {
    let engine = AVAudioEngine()
    var converter: AVAudioConverter?
    var onSamples: (([Float]) -> Void)?
    var configObserver: NSObjectProtocol?
    let lock = NSLock()
    let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    /// Called (on an arbitrary thread) after a format change restarted capture, with the input's new sample
    /// rate, or nil if the restart failed.
    public var onRestart: ((Double?) -> Void)?

    public init() {}

    public func start(_ onSamples: @escaping ([Float]) -> Void) throws {
        lock.lock(); defer { lock.unlock() }
        self.onSamples = onSamples
        try installAndStart()
        configObserver = NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine,
                                                                queue: nil) { [weak self] _ in self?.restart() }
    }

    /// The tap and converter for the input's current format, then the engine.
    func installAndStart() throws {
        let input = engine.inputNode
        let fmt = input.outputFormat(forBus: 0)
        guard fmt.sampleRate > 0, let conv = AVAudioConverter(from: fmt, to: target) else { throw EngineError.io("no converter from \(fmt)") }
        converter = conv
        let deliver = onSamples
        input.installTap(onBus: 0, bufferSize: 1024, format: fmt) { [weak self] buf, _ in
            guard let self = self else { return }
            let cap = AVAudioFrameCount(Double(buf.frameLength) * 16000 / fmt.sampleRate) + 16
            guard let out = AVAudioPCMBuffer(pcmFormat: self.target, frameCapacity: cap) else { return }
            var fed = false
            var err: NSError?
            conv.convert(to: out, error: &err) { _, status in
                if fed { status.pointee = .noDataNow; return nil }
                fed = true
                status.pointee = .haveData
                return buf
            }
            if out.frameLength > 0 {
                deliver?(Array(UnsafeBufferPointer(start: out.floatChannelData![0], count: Int(out.frameLength))))
            }
        }
        try engine.start()
    }

    /// The input's format changed and the engine stopped: capture again with the new format.
    func restart() {
        lock.lock()
        guard onSamples != nil else { lock.unlock(); return }                 // stopped meanwhile
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        let rate: Double?
        do {
            try installAndStart()
            rate = engine.inputNode.outputFormat(forBus: 0).sampleRate
        } catch {
            rate = nil
        }
        lock.unlock()
        onRestart?(rate)
    }

    /// Stops capturing. The resampler still holds a few milliseconds of the last audio: they are delivered
    /// (on the calling thread) before this returns, so nothing said before the stop is lost.
    public func stop() {
        lock.lock(); defer { lock.unlock() }
        if let o = configObserver { NotificationCenter.default.removeObserver(o); configObserver = nil }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        if let conv = converter, let deliver = onSamples, let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: 4096) {
            var err: NSError?
            conv.convert(to: out, error: &err) { _, status in
                status.pointee = .endOfStream
                return nil
            }
            if out.frameLength > 0 {
                deliver(Array(UnsafeBufferPointer(start: out.floatChannelData![0], count: Int(out.frameLength))))
            }
        }
        converter = nil
        onSamples = nil
    }
}

public enum WAVWriter {
    /// Writes 16 kHz mono float samples as a 16-bit PCM WAV file.
    public static func write(_ samples: [Float], to url: URL) throws {
        var d = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: UInt16) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 2)) }
        d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + samples.count * 2)); d.append("WAVEfmt ".data(using: .ascii)!)
        u32(16); u16(1); u16(1); u32(16000); u32(32000); u16(2); u16(16)
        d.append("data".data(using: .ascii)!); u32(UInt32(samples.count * 2))
        for s in samples { u16(UInt16(bitPattern: Int16(max(-1, min(1, s)) * 32767))) }
        try d.write(to: url)
    }
}
