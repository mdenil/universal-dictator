import Foundation
import Metal

/// Memory-mapped safetensors file. The whole file is mapped once; the data region is exposed
/// to Metal without copying, and individual tensors are views (buffer + offset) into it.
public final class SafeTensorsFile {
    public struct Entry {
        public let dtype: DType
        public let shape: [Int]
        public let begin: Int   // byte offset within the data region
        public let end: Int
    }

    public let url: URL
    public let entries: [String: Entry]
    public let metadata: [String: String]
    private let base: UnsafeMutableRawPointer
    private let mappedLength: Int
    private let dataStart: Int
    private var buffers: [(start: Int, length: Int, buffer: MTLBuffer)] = []
    private let ctx: MetalContext

    public init(_ url: URL, _ ctx: MetalContext = .shared) throws {
        self.url = url
        self.ctx = ctx
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw EngineError.io("open \(url.path)") }
        defer { close(fd) }
        var st = stat()
        fstat(fd, &st)
        let size = Int(st.st_size)
        let page = Int(getpagesize())
        let mapLen = (size + page - 1) / page * page
        guard let p = mmap(nil, mapLen, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
            throw EngineError.io("mmap \(url.path)")
        }
        base = p
        mappedLength = mapLen
        let headerLen = Int(p.loadUnaligned(as: UInt64.self))
        let headerData = Data(bytes: p + 8, count: headerLen)
        dataStart = 8 + headerLen
        guard let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw EngineError.io("bad safetensors header")
        }
        var entries: [String: Entry] = [:]
        var meta: [String: String] = [:]
        for (k, v) in json {
            if k == "__metadata__" { meta = (v as? [String: String]) ?? [:]; continue }
            guard let d = v as? [String: Any], let dt = d["dtype"] as? String, let shape = d["shape"] as? [Int],
                  let offs = d["data_offsets"] as? [Int] else { continue }
            let dtype: DType
            switch dt {
            case "F32": dtype = .f32
            case "F16": dtype = .f16
            case "BF16": dtype = .bf16
            case "I32": dtype = .i32
            case "I8": dtype = .i8
            case "U8": dtype = .u8
            default: continue
            }
            entries[k] = Entry(dtype: dtype, shape: shape, begin: offs[0], end: offs[1])
        }
        self.entries = entries
        self.metadata = meta
    }

    deinit { munmap(base, mappedLength) }

    public func has(_ name: String) -> Bool { entries[name] != nil }

    /// Raw pointer to a tensor's bytes (CPU access).
    public func pointer(_ name: String) throws -> (UnsafeRawPointer, Entry) {
        guard let e = entries[name] else { throw EngineError.io("tensor \(name) not in \(url.lastPathComponent)") }
        return (UnsafeRawPointer(base + dataStart + e.begin), e)
    }

    public func floats(_ name: String) throws -> [Float] {
        let (p, e) = try pointer(name)
        let n = e.shape.reduce(1, *)
        switch e.dtype {
        case .f32: return Array(UnsafeBufferPointer(start: p.assumingMemoryBound(to: Float.self), count: n))
        case .f16: return UnsafeBufferPointer(start: p.assumingMemoryBound(to: Float16.self), count: n).map { Float($0) }
        case .bf16: return UnsafeBufferPointer(start: p.assumingMemoryBound(to: UInt16.self), count: n).map { Float(bitPattern: UInt32($0) << 16) }
        default: throw EngineError.io("floats(\(name)) dtype \(e.dtype)")
        }
    }

    /// A zero-copy GPU view of a tensor. Maps the page-aligned region around it into a Metal buffer
    /// (Metal caps a single buffer's length, so large files are split into windows).
    public func tensor(_ name: String) throws -> Tensor {
        guard let e = entries[name] else { throw EngineError.io("tensor \(name) not in \(url.lastPathComponent)") }
        let abs = dataStart + e.begin
        let len = e.end - e.begin
        let (buf, off) = try window(containing: abs, length: len)
        return Tensor(buffer: buf, offset: off, shape: e.shape, dtype: e.dtype)
    }

    private func window(containing abs: Int, length: Int) throws -> (MTLBuffer, Int) {
        for w in buffers where abs >= w.start && abs + length <= w.start + w.length {
            return (w.buffer, abs - w.start)
        }
        let page = Int(getpagesize())
        let windowSize = min(ctx.device.maxBufferLength / page * page, 1 << 30)
        let start = abs / page * page
        let end = min(max(start + windowSize, (abs + length + page - 1) / page * page), mappedLength)
        guard let b = ctx.device.makeBuffer(bytesNoCopy: base + start, length: end - start, options: .storageModeShared, deallocator: nil) else {
            throw EngineError.io("makeBuffer(bytesNoCopy) failed for window at \(start)")
        }
        buffers.append((start, end - start, b))
        return (b, abs - start)
    }
}
