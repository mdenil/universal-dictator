import Foundation

/// Ad-hoc timing probes used while profiling (dictate-cli probe).
public enum Probe {
    public static func gemmSmallM(weights: SafeTensorsFile?) {
        let ctx = MetalContext.shared
        func time(_ label: String, _ body: (Step) -> Void) {
            var best = Double.infinity
            for _ in 0..<5 { let s = Step(ctx); body(s); best = min(best, s.run()) }
            print(String(format: "%-40@ %8.3f ms", label as NSString, best * 1000))
        }
        let a = Tensor(shape: [2, 1024]), out = Tensor(shape: [2, 4096])
        let wAnon = Tensor(shape: [4096, 1024])
        time("gemm M=2 4096x1024 anon f32") { s in s.gemm(a, wAnon, out: out) }
        time("gemm M=2 4096x1024 anon f32 x96") { s in for _ in 0..<96 { s.gemm(a, wAnon, out: out) } }
        if let w = weights, let wf = try? w.tensor("encoder.layers.0.feed_forward1.linear1.weight") {
            time("gemm M=2 4096x1024 mmap f32") { s in s.gemm(a, wf, out: out) }
            time("gemm M=2 4096x1024 mmap f32 x96") { s in for _ in 0..<96 { s.gemm(a, wf, out: out) } }
        }
        let q = Tensor(shape: [2, 1024]), k = Tensor(shape: [72, 1024]), att = Tensor(shape: [2, 1024])
        let pt = Tensor(shape: [80, 1024]), bu = Tensor(shape: [8, 128]), rg = Tensor(shape: [2, 2], dtype: .i32)
        let r: [Int32] = [0, 72, 0, 72]
        r.withUnsafeBytes { rg.contents.copyMemory(from: $0.baseAddress!, byteCount: 16) }
        time("relpos attention Tq=2 Tk=72") { s in s.relPosAttention(q: q, k: k, v: k, posTable: pt, relOffset: 1, biasU: bu, biasV: bu, ranges: rg, out: att, heads: 8) }
        let mel = Tensor(shape: [25, 128, 1]), c0w = Tensor(shape: [256, 1, 3, 3]), c0b = Tensor(shape: [256]), a1 = Tensor(shape: [13, 65, 256])
        time("conv2d first 25x128 -> 13x65x256") { s in s.conv2d(mel, w: c0w, bias: c0b, out: a1, stride: 2, padTop: 2, padLeft: 2, groups: 1, act: .relu, validH: 13) }
        let a2 = Tensor(shape: [7, 33, 256]), dw = Tensor(shape: [256, 1, 3, 3])
        time("conv2d dw 13x65 -> 7x33x256") { s in s.conv2d(a1, w: dw, bias: c0b, out: a2, stride: 2, padTop: 2, padLeft: 2, groups: 256, act: .none, validH: 7) }
        let x = Tensor(shape: [2, 1024]), g = Tensor(shape: [1024])
        time("layernorm 2x1024 x 120") { s in for _ in 0..<120 { s.layerNorm(x, gamma: g, beta: g, out: x) } }
    }
}
