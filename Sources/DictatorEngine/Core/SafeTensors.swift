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
        let header = try SafeTensorsHeader(UnsafeRawBufferPointer(start: p, count: size), file: url.lastPathComponent)
        dataStart = header.dataStart
        entries = header.entries.compactMapValues { e in
            DType(safetensors: e.dtype).map { Entry(dtype: $0, shape: e.shape, begin: e.begin, end: e.end) }
        }
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
        guard let f = widenToFloats(p, dtype: e.dtype.safetensors, count: e.shape.reduce(1, *)) else {
            throw EngineError.io("floats(\(name)) dtype \(e.dtype)")
        }
        return f
    }

    /// A zero-copy GPU view of a tensor. Maps the page-aligned region around it into a Metal buffer
    /// (Metal caps a single buffer's length, so large files are split into windows).
    public func tensor(_ name: String) throws -> Tensor {
        guard let e = entries[name] else { throw EngineError.io("tensor \(name) not in \(url.lastPathComponent)") }
        let abs = dataStart + e.begin
        let len = e.end - e.begin
        let (buf, off) = try window(containing: abs, length: len)
        let t = Tensor(buffer: buf, offset: off, shape: e.shape, dtype: e.dtype)
        if e.dtype == .i8, entries[name + ".qscale"] != nil { t.qscale = try tensor(name + ".qscale") }
        return t
    }

    private func window(containing abs: Int, length: Int) throws -> (MTLBuffer, Int) {
        if buffers.isEmpty { try buildWindows() }
        for w in buffers where abs >= w.start && abs + length <= w.start + w.length {
            return (w.buffer, abs - w.start)
        }
        throw EngineError.io("no window covers [\(abs), \(abs + length))")
    }

    /// Partitions the file into non-overlapping page-aligned Metal buffers, greedily in file order,
    /// each at most `maxBufferLength` long and never splitting a tensor. (Overlapping windows made
    /// the GPU account for the same pages several times and run out of memory.)
    private func buildWindows() throws {
        let page = Int(getpagesize())
        let limit = ctx.device.maxBufferLength / page * page
        let spans = entries.values.map { (dataStart + $0.begin, dataStart + $0.end) }.sorted { $0.0 < $1.0 }
        var start = 0
        var end = 0
        func flush() throws {
            let s0 = start / page * page
            let e0 = min((end + page - 1) / page * page, mappedLength)
            guard e0 > s0 else { return }
            guard let b = ctx.device.makeBuffer(bytesNoCopy: base + s0, length: e0 - s0, options: .storageModeShared, deallocator: nil) else {
                throw EngineError.io("makeBuffer(bytesNoCopy) failed for window at \(s0) length \(e0 - s0)")
            }
            buffers.append((s0, e0 - s0, b))
        }
        for (b, e) in spans {
            let s0 = start / page * page
            if e - s0 > limit && end > start {
                try flush()
                start = (end + page - 1) / page * page
                if b < start { start = b / page * page }  // shared page: window starts at the tensor's page
            }
            end = max(end, e)
        }
        try flush()
    }
}

extension SafeTensorsFile: WeightSource {
    public func shape(_ name: String) -> [Int]? { entries[name]?.shape }
}
