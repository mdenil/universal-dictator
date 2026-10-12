import Foundation
import Metal

public enum DType: String {
    case f32, f16, bf16, i32, i8, u8

    /// The dtype's name in safetensors headers.
    public var safetensors: String {
        switch self {
        case .f32: return "F32"
        case .f16: return "F16"
        case .bf16: return "BF16"
        case .i32: return "I32"
        case .i8: return "I8"
        case .u8: return "U8"
        }
    }

    public init?(safetensors name: String) {
        guard let d = [DType.f32, .f16, .bf16, .i32, .i8, .u8].first(where: { $0.safetensors == name }) else { return nil }
        self = d
    }

    public var size: Int {
        switch self {
        case .f32, .i32: return 4
        case .f16, .bf16: return 2
        case .i8, .u8: return 1
        }
    }
}

/// A contiguous row-major tensor living in a Metal buffer (shared storage, so the CPU can read it).
public final class Tensor {
    public let buffer: MTLBuffer
    public let offset: Int
    public private(set) var shape: [Int]
    public let dtype: DType
    /// Per-output-row dequantization scales (f32 [rows]) for int8 weights: w = q * qscale[row].
    public var qscale: Tensor?

    public init(buffer: MTLBuffer, offset: Int = 0, shape: [Int], dtype: DType) {
        self.buffer = buffer
        self.offset = offset
        self.shape = shape
        self.dtype = dtype
    }

    public convenience init(_ ctx: MetalContext = .shared, shape: [Int], dtype: DType = .f32) {
        let n = shape.reduce(1, *)
        self.init(buffer: ctx.makeBuffer(bytes: n * dtype.size), shape: shape, dtype: dtype)
    }

    public var count: Int { shape.reduce(1, *) }
    public var byteCount: Int { count * dtype.size }

    /// Same storage, new shape (element count must match).
    public func reshaped(_ newShape: [Int]) -> Tensor {
        precondition(newShape.reduce(1, *) == count, "reshape \(shape) -> \(newShape)")
        let t = Tensor(buffer: buffer, offset: offset, shape: newShape, dtype: dtype)
        t.qscale = qscale
        return t
    }

    /// A view of rows [start, start+n) along the first axis.
    public func rows(_ start: Int, _ n: Int) -> Tensor {
        let rowElems = shape.dropFirst().reduce(1, *)
        var s = shape
        s[0] = n
        return Tensor(buffer: buffer, offset: offset + start * rowElems * dtype.size, shape: s, dtype: dtype)
    }

    public var contents: UnsafeMutableRawPointer { buffer.contents() + offset }

    public func toFloats() -> [Float] {
        let n = count
        switch dtype {
        case .f32, .f16, .bf16:
            return widenToFloats(contents, dtype: dtype.safetensors, count: n)!
        case .i32:
            let p = contents.assumingMemoryBound(to: Int32.self)
            return (0..<n).map { Float(p[$0]) }
        case .i8:
            let p = contents.assumingMemoryBound(to: Int8.self)
            return (0..<n).map { Float(p[$0]) }
        case .u8:
            let p = contents.assumingMemoryBound(to: UInt8.self)
            return (0..<n).map { Float(p[$0]) }
        }
    }

    public func copy(from floats: [Float]) {
        precondition(floats.count == count, "copy \(floats.count) into \(shape)")
        switch dtype {
        case .f32:
            floats.withUnsafeBytes { contents.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        case .f16:
            let p = contents.assumingMemoryBound(to: Float16.self)
            for i in 0..<floats.count { p[i] = Float16(floats[i]) }
        case .bf16:
            let p = contents.assumingMemoryBound(to: UInt16.self)
            for i in 0..<floats.count { p[i] = bf16Bits(floats[i]) }
        default:
            fatalError("copy(from:) unsupported for \(dtype)")
        }
    }

    public static func from(_ floats: [Float], shape: [Int], dtype: DType = .f32, _ ctx: MetalContext = .shared) -> Tensor {
        let t = Tensor(ctx, shape: shape, dtype: dtype)
        t.copy(from: floats)
        return t
    }
}

/// Round-to-nearest-even float -> bfloat16 bits.
@inline(__always)
public func bf16Bits(_ f: Float) -> UInt16 {
    let u = f.bitPattern
    if f.isNaN { return 0x7FC0 }
    let rounding = ((u >> 16) & 1) &+ 0x7FFF
    return UInt16((u &+ rounding) >> 16)
}

/// Named scratch buffers reused across steps (grown when a larger size is requested). Views of these
/// are only valid until the same name is requested again, so a step must not hold one across steps.
public final class Scratch {
    let ctx: MetalContext
    private var buffers: [String: (MTLBuffer, Int)] = [:]
    public init(_ ctx: MetalContext = .shared) { self.ctx = ctx }

    public func get(_ name: String, _ shape: [Int], _ dtype: DType = .f32) -> Tensor {
        let bytes = shape.reduce(1, *) * dtype.size
        if let (b, cap) = buffers[name], cap >= bytes { return Tensor(buffer: b, shape: shape, dtype: dtype) }
        let b = ctx.makeBuffer(bytes: bytes)
        buffers[name] = (b, bytes)
        return Tensor(buffer: b, shape: shape, dtype: dtype)
    }
}
