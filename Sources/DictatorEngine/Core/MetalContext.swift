import Foundation
import Metal

public enum EngineError: Error, CustomStringConvertible {
    case noDevice
    case library(String)
    case missingFunction(String)
    case io(String)
    case shape(String)

    public var description: String {
        switch self {
        case .noDevice: return "no Metal device"
        case .library(let s): return "Metal library: \(s)"
        case .missingFunction(let s): return "missing Metal function \(s)"
        case .io(let s): return "io: \(s)"
        case .shape(let s): return "shape: \(s)"
        }
    }
}

/// Owns the Metal device, the command queue, the compiled kernel library and a pipeline cache.
public final class MetalContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    public static let shared: MetalContext = {
        do { return try MetalContext() } catch { fatalError("MetalContext init failed: \(error)") }
    }()

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError.noDevice }
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw EngineError.noDevice }
        self.queue = queue
        self.library = try MetalContext.loadLibrary(device: device)
    }

    /// Kernels ship as `.metal` sources in the bundle. They are concatenated in file-name order
    /// (so `00_common.metal` comes first) and compiled once at startup.
    static func loadLibrary(device: MTLDevice) throws -> MTLLibrary {
        guard let dir = Bundle.module.url(forResource: "Metal", withExtension: nil) else {
            throw EngineError.library("Metal resource directory not found in bundle")
        }
        let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "metal" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var source = ""
        for f in files {
            source += "// ---- \(f.lastPathComponent)\n"
            source += try String(contentsOf: f, encoding: .utf8)
            source += "\n"
        }
        let opts = MTLCompileOptions()
        opts.mathMode = .safe
        opts.languageVersion = .version3_2
        do {
            return try device.makeLibrary(source: source, options: opts)
        } catch {
            throw EngineError.library("\(error)")
        }
    }

    /// Returns a cached pipeline for `name`, specialised with integer function constants
    /// (index -> value) when given.
    public func pipeline(_ name: String, constants: [Int: Int32] = [:]) throws -> MTLComputePipelineState {
        let key = constants.isEmpty ? name : name + "|" + constants.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ",")
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[key] { return p }
        let fn: MTLFunction
        if constants.isEmpty {
            guard let f = library.makeFunction(name: name) else { throw EngineError.missingFunction(name) }
            fn = f
        } else {
            let values = MTLFunctionConstantValues()
            for (idx, v) in constants {
                var vv = v
                values.setConstantValue(&vv, type: .int, index: idx)
            }
            fn = try library.makeFunction(name: name, constantValues: values)
        }
        let p = try device.makeComputePipelineState(function: fn)
        pipelines[key] = p
        return p
    }

    public func makeBuffer(bytes: Int) -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
            fatalError("makeBuffer(\(bytes)) failed")
        }
        return b
    }
}
