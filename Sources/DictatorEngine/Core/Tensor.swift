import Foundation
import Metal

public enum DType: String {
    case f32, f16, bf16, i32, i8, u8
    /// Packed 4-bit weights, two per byte; logical shape [N, K], storage N*K/2 bytes, with
    /// `qscale` holding f16 scales [N, K/64] (symmetric, value = (nibble - 8) * scale).
    case q4

    public var size: Int {
        switch self {
        case .f32, .i32: return 4
        case .f16, .bf16: return 2
        case .i8, .u8, .q4: return 1
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
    public var byteCount: Int { dtype == .q4 ? count / 2 : count * dtype.size }

    /// Same storage, new shape (element count must match).
    public func reshaped(_ newShape: [Int]) -> Tensor {
        precondition(newShape.reduce(1, *) == count, "reshape \(shape) -> \(newShape)")
        precondition(dtype != .q4 || newShape.last == shape.last, "q4 tensors keep their row length")
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
        case .f32:
            return Array(UnsafeBufferPointer(start: contents.assumingMemoryBound(to: Float.self), count: n))
        case .f16:
            let p = contents.assumingMemoryBound(to: Float16.self)
            return (0..<n).map { Float(p[$0]) }
        case .bf16:
            let p = contents.assumingMemoryBound(to: UInt16.self)
            return (0..<n).map { Float(bitPattern: UInt32(p[$0]) << 16) }
        case .i32:
            let p = contents.assumingMemoryBound(to: Int32.self)
            return (0..<n).map { Float(p[$0]) }
        case .i8:
            let p = contents.assumingMemoryBound(to: Int8.self)
            return (0..<n).map { Float(p[$0]) }
        case .u8:
            let p = contents.assumingMemoryBound(to: UInt8.self)
            return (0..<n).map { Float(p[$0]) }
        case .q4:
            fatalError("toFloats on packed q4")
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
