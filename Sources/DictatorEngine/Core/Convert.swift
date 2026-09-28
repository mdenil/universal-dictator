import Foundation

/// Writes safetensors files. Every tensor starts on a 64-byte boundary (gaps are filled with
/// `__pad_N` u8 tensors, which keeps the file valid safetensors) so vector loads are aligned.
public final class SafeTensorsWriter {
    struct Item { let name: String; let dtype: String; let shape: [Int]; let data: Data }
    private var items: [Item] = []
    public var metadata: [String: String] = [:]

    public init() {}

    public func add(_ name: String, dtype: DType, shape: [Int], data: Data) {
        let tag: String
        switch dtype {
        case .f32: tag = "F32"
        case .f16: tag = "F16"
        case .bf16: tag = "BF16"
        case .i32: tag = "I32"
        case .i8: tag = "I8"
        case .u8: tag = "U8"
        }
        items.append(Item(name: name, dtype: tag, shape: shape, data: data))
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
                all.append(Item(name: "__pad_\(padIndex)", dtype: "U8", shape: [aligned - offset], data: Data(count: aligned - offset)))
                padIndex += 1
                offset = aligned
            }
            all.append(item)
            offset += item.data.count
        }
        offset = 0
        var header: [String: Any] = [:]
        for item in all {
            header[item.name] = ["dtype": item.dtype, "shape": item.shape, "data_offsets": [offset, offset + item.data.count]]
            layout.append((item, offset))
            offset += item.data.count
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
        for (item, _) in layout { try fh.write(contentsOf: item.data) }
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
