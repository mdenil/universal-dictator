import Foundation

/// Writes safetensors files. Every tensor starts on a 64-byte boundary (gaps are filled with
/// `__pad_N` u8 tensors, which keeps the file valid safetensors) so vector loads are aligned.
public final class SafeTensorsWriter {
    struct Item {
        let name: String; let dtype: String; let shape: [Int]; let bytes: Int
        let produce: () throws -> Data
    }
    private var items: [Item] = []
    public var metadata: [String: String] = [:]

    public init() {}

    public func add(_ name: String, dtype: DType, shape: [Int], data: Data) {
        addLazy(name, dtype: dtype, shape: shape, bytes: data.count) { data }
    }

    /// Adds a tensor whose bytes are produced only when the file is written (keeps memory flat).
    public func addLazy(_ name: String, dtype: DType, shape: [Int], bytes: Int, produce: @escaping () throws -> Data) {
        let tag: String
        switch dtype {
        case .f32: tag = "F32"
        case .f16: tag = "F16"
        case .bf16: tag = "BF16"
        case .i32: tag = "I32"
        case .i8: tag = "I8"
        case .u8: tag = "U8"
        }
        items.append(Item(name: name, dtype: tag, shape: shape, bytes: bytes, produce: produce))
    }

    public func write(to url: URL) throws {
        // Lay out data with 64-byte alignment relative to the data section start; the header is
        // padded so the data section itself starts 64-byte aligned in the file.
        var layout: [(Item, Int)] = []
        var offset = 0
        var padIndex = 0
        var all: [Item] = []
        for item in items.sorted(by: { $0.name < $1.name }) {
            let aligned = (offset + 63) / 64 * 64
            if aligned > offset {
                let n = aligned - offset
                all.append(Item(name: "__pad_\(padIndex)", dtype: "U8", shape: [n], bytes: n, produce: { Data(count: n) }))
                padIndex += 1
                offset = aligned
            }
            all.append(item)
            offset += item.bytes
        }
        offset = 0
        var header: [String: Any] = [:]
        for item in all {
            header[item.name] = ["dtype": item.dtype, "shape": item.shape, "data_offsets": [offset, offset + item.bytes]]
            layout.append((item, offset))
            offset += item.bytes
        }
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        let pad = (64 - (8 + json.count) % 64) % 64
        json.append(Data(repeating: 0x20, count: pad))
        var out = Data()
        var len = UInt64(json.count)
        out.append(Data(bytes: &len, count: 8))
        out.append(json)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let fh = try FileHandle(forWritingTo: url)
        defer { try? fh.close() }
        try fh.write(contentsOf: out)
        for (item, _) in layout {
            let d = try item.produce()
            precondition(d.count == item.bytes, "\(item.name): produced \(d.count) bytes, declared \(item.bytes)")
            try fh.write(contentsOf: d)
        }
    }
}

/// Weight conversion: casts selected tensors to a lower precision (the runtime reads any dtype the
/// kernels support). Names and shapes are unchanged so one loader handles every precision.
public enum WeightConverter {
    public enum Precision: String { case f32, f16 }

    /// Tensors that stay fp32 regardless: small vectors, norms, positional biases, subsampling convs,
    /// the mel frontend and everything the CPU-side RNN-T decoder reads.
    static func keepF32(_ name: String, shape: [Int]) -> Bool {
        if shape.count < 2 { return true }
        if name.hasPrefix("preprocessor.") || name.hasPrefix("decoder.") { return true }
        if name.hasPrefix("joint.") && !name.hasPrefix("joint.enc.") { return true }
        if name.contains("pos_bias") || name.contains("depthwise_conv") || name.contains("batch_norm") { return true }
        if name.hasPrefix("encoder.pre_encode.conv.") { return true }
        return false
    }

    public static func convert(src: URL, dst: URL, precision: Precision) throws -> (tensors: Int, bytes: Int) {
        let f = try SafeTensorsFile(src)
        let w = SafeTensorsWriter()
        w.metadata["source"] = src.lastPathComponent
        w.metadata["precision"] = precision.rawValue
        var bytes = 0
        for name in f.entries.keys.sorted() where !name.hasPrefix("__pad_") {
            let e = f.entries[name]!
            let (ptr, _) = try f.pointer(name)
            let n = e.shape.reduce(1, *)
            if precision == .f16 && e.dtype == .f32 && !keepF32(name, shape: e.shape) {
                let src = ptr.assumingMemoryBound(to: Float.self)
                var d = Data(count: n * 2)
                d.withUnsafeMutableBytes { raw in
                    let p = raw.bindMemory(to: Float16.self)
                    for i in 0..<n { p[i] = Float16(src[i]) }
                }
                w.add(name, dtype: .f16, shape: e.shape, data: d)
                bytes += d.count
            } else {
                let d = Data(bytes: ptr, count: e.end - e.begin)
                w.add(name, dtype: e.dtype, shape: e.shape, data: d)
                bytes += d.count
            }
        }
        try w.write(to: dst)
        return (f.entries.count, bytes)
    }
}

extension WeightConverter {
    /// Voxtral int8 conversion by stage (docs/PLAN.md M4): `groups` picks which weight families become
    /// per-row int8 (+ "<name>.qscale" f32); everything else is copied unchanged (bf16).
    ///   dec-ffn: decoder mlp gate/up/down   dec-attn: decoder q/k/v/o   enc: encoder layer linears
    ///   adapter: multi_modal_projector      stem: conv1/conv2
    public static func quantizeVoxtral(src: URL, dst: URL, groups: Set<String>) throws -> (quantized: Int, bytes: Int) {
        let f = try SafeTensorsFile(src)
        let w = SafeTensorsWriter()
        w.metadata["source"] = src.lastPathComponent
        w.metadata["quantized_groups"] = groups.sorted().joined(separator: ",")
        func group(_ n: String) -> String? {
            if n.hasPrefix("language_model.model.layers.") && n.contains(".mlp.") { return "dec-ffn" }
            if n.hasPrefix("language_model.model.layers.") && n.contains(".self_attn.") && n.hasSuffix("_proj.weight") { return "dec-attn" }
            if n.hasPrefix("audio_tower.layers.") && n.hasSuffix("_proj.weight") { return "enc" }
            if n.hasPrefix("multi_modal_projector.") { return "adapter" }
            if n.hasPrefix("audio_tower.embedder.conv") && n.hasSuffix(".weight") { return "stem" }
            return nil
        }
        var nq = 0, bytes = 0
        for name in f.entries.keys.sorted() where !name.hasPrefix("__pad_") {
            let e = f.entries[name]!
            if let g = group(name), groups.contains(g) {
                let rows = e.shape[0], cols = e.shape.dropFirst().reduce(1, *)
                var cache: (q: [Int8], scale: [Float])?
                func get() throws -> (q: [Int8], scale: [Float]) {
                    if let c = cache { return c }
                    let x = try f.floats(name)
                    let r = x.withUnsafeBufferPointer { Quant.quantizeRowsInt8($0, rows: rows, cols: cols) }
                    cache = r
                    return r
                }
                // Tensors are written in sorted order, so "<name>" is produced just before "<name>.qscale".
                w.addLazy(name, dtype: .i8, shape: e.shape, bytes: rows * cols) { try get().q.withUnsafeBytes { Data($0) } }
                w.addLazy(name + ".qscale", dtype: .f32, shape: [rows], bytes: rows * 4) {
                    let d = try get().scale.withUnsafeBytes { Data($0) }
                    cache = nil
                    return d
                }
                nq += 1
                bytes += rows * cols + rows * 4
            } else {
                let n = e.end - e.begin
                w.addLazy(name, dtype: e.dtype, shape: e.shape, bytes: n) { let (p, _) = try f.pointer(name); return Data(bytes: p, count: n) }
                bytes += n
            }
        }
        try w.write(to: dst)
        return (nq, bytes)
    }
}
