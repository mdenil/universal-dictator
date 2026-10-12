import CoreML
import Foundation

/// Nemotron's cache-aware streaming encoder step on the Neural Engine, through the Core ML package built by
/// `tools/nemotron_ane.py`: one function per latency (the 560 ms step is split into a subsampling function and
/// a layers function, as the fused graph fails the Neural Engine compile), sharing one copy of the weights.
/// The K/V and conv caches live in Core ML state and are updated in place. Same schedule
/// and masking as `FastConformerEncoder.encodeStreamStep`; a short final chunk is zero-padded to the
/// model's rows and masked. Outputs the joint's encoder projection for the CPU RNN-T decoder.
public final class NemotronANEEncoder: NemotronEncoder {
    public let schedule: NemotronSchedule
    let model: MLModel
    let sub: MLModel?
    var state: MLState
    let firstRows: Int, steadyRows: Int
    let T: Int
    let L = 70, projDim = 640, melBins = 128
    var cacheLen = 0
    private(set) var steps = 0

    /// The Core ML package's name.
    public static let modelName = "nemotron_encoder"

    /// The package's functions for a latency: (subsampling function or nil, stateful function).
    static func functions(_ schedule: NemotronSchedule) -> (sub: String?, state: String) {
        schedule.chunkMs == 560 ? ("accurate_sub", "accurate_layers") : (nil, "fast")
    }

    /// Loads (or reuses) the functions for a latency from `assets` and starts a fresh stream.
    public init(assets: ModelAssets, schedule: NemotronSchedule) throws {
        self.schedule = schedule
        let f = Self.functions(schedule)
        model = try Self.load(assets, f.state)
        sub = try f.sub.map { try Self.load(assets, $0) }
        firstRows = schedule.firstChunk
        steadyRows = schedule.preEncodeCache + schedule.chunk
        T = schedule.outFramesPerChunk
        state = model.makeState()
    }

    /// Loaded functions, by package path and function name, are shared by every stream (loading one compiles
    /// it for the Neural Engine).
    static var loaded: [String: MLModel] = [:]
    static let lock = NSLock()

    static func load(_ assets: ModelAssets, _ function: String) throws -> MLModel {
        lock.lock(); defer { lock.unlock() }
        guard let url = try assets.compiledCoreMLModel(modelName) else {
            throw EngineError.io("Core ML model \(modelName) not found in \(assets.directory.path)")
        }
        let key = "\(url.path)|\(function)"
        if let m = loaded[key] { return m }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        cfg.functionName = function
        let m = try MLModel(contentsOf: url, configuration: cfg)
        loaded[key] = m
        return m
    }

    /// Releases every loaded function except this latency's, so only one model stays in memory. Streams
    /// already running keep theirs until they end.
    static func keepOnly(_ schedule: NemotronSchedule) {
        let f = functions(schedule)
        let keep = Set([f.sub, f.state].compactMap { $0 })
        lock.lock(); defer { lock.unlock() }
        loaded = loaded.filter { key, _ in keep.contains(String(key.split(separator: "|").last ?? "")) }
    }

    /// Whether the Core ML package is present.
    public static func available(assets: ModelAssets) -> Bool { assets.coreMLModel(modelName) != nil }

    static func nextLen(_ t: Int) -> Int { t / 2 + 1 }

    static func floats(_ n: Int, _ fill: (Int) -> Float) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [NSNumber(value: n)], dataType: .float32)
        let p = a.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<n { p[i] = fill(i) }
        return a
    }

    /// Every chunk runs through the steady model. The first chunk's rows sit after
    /// `lead` = steadyRows - firstRows zero rows that are masked at each subsampling stage (lead, lead/2,
    /// lead/4, lead/8 rows), which reproduces its causal zero padding; the lead/8 = 2 frames are the
    /// steady model's dropped frames.
    public func step(mel: UnsafeBufferPointer<Float>, nRows: Int, validRows: Int, keepAll: Bool) throws -> (proj: [Float], frames: Int) {
        let isFirst = steps == 0
        let R = steadyRows
        let lead = isFirst ? steadyRows - firstRows : 0
        precondition(nRows <= (isFirst ? firstRows : steadyRows), "chunk of \(nRows) mel rows exceeds the model")
        let v1 = Self.nextLen(validRows), v2 = Self.nextLen(v1), v3 = Self.nextLen(v2)
        let r1 = Self.nextLen(R), r2 = Self.nextLen(r1), r3 = Self.nextLen(r2)
        let validT = isFirst ? max(0, min(v3, T)) : max(0, min(v3 - schedule.dropExtra, T))
        let melA = try MLMultiArray(shape: [NSNumber(value: R), NSNumber(value: melBins)], dataType: .float32)
        let mp = melA.dataPointer.assumingMemoryBound(to: Float.self)
        memset(mp, 0, R * melBins * 4)
        memcpy(mp + lead * melBins, mel.baseAddress!, nRows * melBins * 4)
        let (l0, l1, l2, l3) = (lead, lead / 2, lead / 4, lead / 8)
        let subIn: [String: MLFeatureValue] = [
            "mel": MLFeatureValue(multiArray: melA),
            "m0": MLFeatureValue(multiArray: try Self.floats(R) { $0 >= l0 && $0 < l0 + validRows ? 1 : 0 }),
            "m1": MLFeatureValue(multiArray: try Self.floats(r1) { $0 >= l1 && $0 < l1 + v1 ? 1 : 0 }),
            "m2": MLFeatureValue(multiArray: try Self.floats(r2) { $0 >= l2 && $0 < l2 + v2 ? 1 : 0 }),
            "m3": MLFeatureValue(multiArray: try Self.floats(r3) { $0 >= l3 && $0 < l3 + v3 ? 1 : 0 }),
        ]
        var inputs: [String: MLFeatureValue] = [
            "fmask": MLFeatureValue(multiArray: try Self.floats(T) { $0 < validT ? 1 : 0 }),
            "kbias": MLFeatureValue(multiArray: try Self.floats(L + T) { j in (j >= L - cacheLen && j < L + validT) ? 0 : -1e4 }),
        ]
        if let sub = sub {
            inputs["x"] = try sub.prediction(from: MLDictionaryFeatureProvider(dictionary: subIn)).featureValue(for: "x")!
        } else {
            inputs.merge(subIn) { a, _ in a }
        }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs), using: state)
        cacheLen = min(cacheLen + T, L)
        steps += 1
        let keep = keepAll ? validT : min(validT, schedule.outFramesPerChunk)
        guard keep > 0, let pa = out.featureValue(for: "enc_proj")?.multiArrayValue else { return ([], 0) }
        var proj = [Float](repeating: 0, count: keep * projDim)
        let s0 = pa.strides[0].intValue
        let p = pa.dataPointer.assumingMemoryBound(to: Float.self)
        for t in 0..<keep { for c in 0..<projDim { proj[t * projDim + c] = p[t * s0 + c] } }
        return (proj, keep)
    }
}
