import CoreML
import Foundation

/// Nemotron's cache-aware streaming encoder step on the Neural Engine, through Core ML models built by
/// `reference/coreml/nemotron_ane.py` (one fixed shape for the first chunk, one for the rest). Same schedule
/// and masking as `FastConformerEncoder.encodeStreamStep`; a short final chunk is zero-padded to the model's
/// rows and masked. Outputs the joint's encoder projection, ready for the CPU RNN-T decoder.
public final class NemotronANEEncoder {
    let first: MLModel, steady: MLModel
    let schedule: NemotronSchedule
    let firstRows: Int, steadyRows: Int
    public let T: Int
    let L = 70, layers = 24, D = 1024, convCache = 8, projDim = 640, melBins = 128
    var kcache: MLMultiArray, vcache: MLMultiArray, ccache: MLMultiArray
    var cacheLen = 0
    public private(set) var steps = 0

    public init(modelDir: URL, schedule: NemotronSchedule, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        self.schedule = schedule
        let lat = schedule.chunkMs
        first = try Self.load(modelDir, "nemotron_\(lat)_first", computeUnits)
        steady = try Self.load(modelDir, "nemotron_\(lat)_steady", computeUnits)
        firstRows = schedule.firstChunk
        steadyRows = schedule.preEncodeCache + schedule.chunk
        T = schedule.outFramesPerChunk
        kcache = try MLMultiArray(shape: [24, 70, 1024], dataType: .float16)
        vcache = try MLMultiArray(shape: [24, 70, 1024], dataType: .float16)
        ccache = try MLMultiArray(shape: [24, 8, 1024], dataType: .float16)
        reset()
    }

    /// Loaded models are shared by every stream (loading one takes about a second).
    static var loaded: [String: MLModel] = [:]
    static let lock = NSLock()

    static func load(_ modelDir: URL, _ name: String, _ units: MLComputeUnits) throws -> MLModel {
        lock.lock(); defer { lock.unlock() }
        let key = "\(modelDir.path)|\(name)|\(units.rawValue)"
        if let m = loaded[key] { return m }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        let m = try MLModel(contentsOf: try compiled(modelDir, name), configuration: cfg)
        loaded[key] = m
        return m
    }

    /// Compiles `<name>.mlpackage` to `<name>.mlmodelc` next to it once, and returns the compiled URL.
    static func compiled(_ modelDir: URL, _ name: String) throws -> URL {
        let dir = modelDir.appendingPathComponent("coreml")
        let out = dir.appendingPathComponent(name + ".mlmodelc")
        if FileManager.default.fileExists(atPath: out.path) { return out }
        let tmp = try MLModel.compileModel(at: dir.appendingPathComponent(name + ".mlpackage"))
        try FileManager.default.moveItem(at: tmp, to: out)
        return out
    }

    public func reset() {
        for a in [kcache, vcache, ccache] { memset(a.dataPointer, 0, a.count * 2) }
        cacheLen = 0
        steps = 0
    }

    static func nextLen(_ t: Int) -> Int { t / 2 + 1 }

    /// One chunk: mel rows [nRows, 128] (pre-encode cache rows included), the first `validRows` real.
    /// Returns the joint encoder projection [T, 640] and the number of valid output frames.
    public func step(mel: UnsafeBufferPointer<Float>, nRows: Int, validRows: Int) throws -> (proj: [Float], validT: Int) {
        let isFirst = steps == 0
        let R = isFirst ? firstRows : steadyRows
        let drop = isFirst ? 0 : schedule.dropExtra
        precondition(nRows <= R, "chunk of \(nRows) mel rows exceeds the model's \(R)")
        let v1 = Self.nextLen(validRows), v2 = Self.nextLen(v1), v3 = Self.nextLen(v2)
        let r1 = Self.nextLen(R), r2 = Self.nextLen(r1), r3 = Self.nextLen(r2)
        let validT = max(0, min(v3 - drop, T))

        func array(_ n: Int, _ fill: (Int) -> Float) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: [NSNumber(value: n)], dataType: .float32)
            let p = a.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<n { p[i] = fill(i) }
            return a
        }
        let melA = try MLMultiArray(shape: [NSNumber(value: R), 128], dataType: .float32)
        let mp = melA.dataPointer.assumingMemoryBound(to: Float.self)
        memset(mp, 0, R * melBins * 4)
        memcpy(mp, mel.baseAddress!, nRows * melBins * 4)
        let inputs: [String: MLFeatureValue] = [
            "mel": MLFeatureValue(multiArray: melA),
            "m0": MLFeatureValue(multiArray: try array(R) { $0 < validRows ? 1 : 0 }),
            "m1": MLFeatureValue(multiArray: try array(r1) { $0 < v1 ? 1 : 0 }),
            "m2": MLFeatureValue(multiArray: try array(r2) { $0 < v2 ? 1 : 0 }),
            "m3": MLFeatureValue(multiArray: try array(r3) { $0 < v3 ? 1 : 0 }),
            "fmask": MLFeatureValue(multiArray: try array(T) { $0 < validT ? 1 : 0 }),
            "kbias": MLFeatureValue(multiArray: try array(L + T) { j in (j >= L - cacheLen && j < L + validT) ? 0 : -1e4 }),
            "kcache": MLFeatureValue(multiArray: kcache),
            "vcache": MLFeatureValue(multiArray: vcache),
            "convcache": MLFeatureValue(multiArray: ccache),
        ]
        let out = try (isFirst ? first : steady).prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
        kcache = out.featureValue(for: "knext")!.multiArrayValue!
        vcache = out.featureValue(for: "vnext")!.multiArrayValue!
        ccache = out.featureValue(for: "cnext")!.multiArrayValue!
        let pa = out.featureValue(for: "enc_proj")!.multiArrayValue!
        var proj = [Float](repeating: 0, count: T * projDim)
        if pa.dataType == .float32 {
            // Rows may be padded (strides); copy row by row.
            let s0 = pa.strides[0].intValue
            let p = pa.dataPointer.assumingMemoryBound(to: Float.self)
            for t in 0..<T { for c in 0..<projDim { proj[t * projDim + c] = p[t * s0 + c] } }
        } else {
            for i in 0..<(T * projDim) { proj[i] = pa[[NSNumber(value: i / projDim), NSNumber(value: i % projDim)]].floatValue }
        }
        cacheLen = min(cacheLen + T, L)
        steps += 1
        return (proj, validT)
    }
}
