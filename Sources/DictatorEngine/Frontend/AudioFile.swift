import AVFoundation
import Foundation

public enum AudioFile {
    /// Reads an audio file as mono float samples at 16 kHz (resampling with AVAudioConverter if needed).
    public static func readMono16k(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        let src = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: frames) else { throw EngineError.io("buffer") }
        try file.read(into: inBuf)
        if src.sampleRate == 16000 && src.channelCount == 1 && src.commonFormat == .pcmFormatFloat32 {
            return Array(UnsafeBufferPointer(start: inBuf.floatChannelData![0], count: Int(inBuf.frameLength)))
        }
        guard let conv = AVAudioConverter(from: src, to: target) else { throw EngineError.io("converter") }
        let outCap = AVAudioFrameCount(Double(frames) * 16000 / src.sampleRate) + 1024
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: outCap) else { throw EngineError.io("buffer") }
        var done = false
        var err: NSError?
        conv.convert(to: outBuf, error: &err) { _, status in
            if done { status.pointee = .endOfStream; return nil }
            done = true
            status.pointee = .haveData
            return inBuf
        }
        if let e = err { throw e }
        return Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: Int(outBuf.frameLength)))
    }
}
