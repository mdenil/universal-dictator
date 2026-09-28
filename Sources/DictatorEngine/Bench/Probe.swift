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

extension Probe {
    /// GEMM throughput at the shapes the models use (GPU time, best of 5, back-to-back x8).
    public static func gemmShapes() {
        let ctx = MetalContext.shared
        let shapes: [(String, Int, Int, Int)] = [
            ("4096^3", 4096, 4096, 4096),
            ("parakeet ff1 77x4096x1024", 77, 4096, 1024),
            ("parakeet ff2 77x1024x4096", 77, 1024, 4096),
            ("parakeet att 77x1024x1024", 77, 1024, 1024),
            ("offline 616x4096x1024", 616, 4096, 1024),
            ("subsample pw 2112x256x256", 2112, 256, 256),
        ]
        for (dt, wdt) in [(DType.f32, DType.f32), (DType.f32, DType.f16), (DType.f16, DType.f16)] {
            for (name, M, N, K) in shapes {
                let a = Tensor(ctx, shape: [M, K], dtype: dt), w = Tensor(ctx, shape: [N, K], dtype: wdt)
                let c = Tensor(ctx, shape: [M, N], dtype: dt)
                var best = Double.infinity
                for _ in 0..<5 { let s = Step(ctx); for _ in 0..<8 { s.gemm(a, w, out: c) }; best = min(best, s.run() / 8) }
                let tf = 2.0 * Double(M * N * K) / best / 1e12
                print(String(format: "%@x%@  %-28@ %8.3f ms  %6.2f TFLOPS", dt.rawValue as NSString, wdt.rawValue as NSString, name as NSString, best * 1000, tf))
            }
        }
    }
}

extension Probe {
    /// Voxtral step shapes: bf16 weights, small M (GPU time per call, back to back x16).
    public static func voxtralShapes() {
        let ctx = MetalContext.shared
        let shapes: [(String, Int, Int, Int)] = [
            ("enc q 4x2048x1280", 4, 2048, 1280), ("enc gate 4x5120x1280", 4, 5120, 1280), ("enc down 4x1280x5120", 4, 1280, 5120),
            ("conv2 4x1280x3840", 4, 1280, 3840), ("dec q 1x4096x3072", 1, 4096, 3072), ("dec gate 1x9216x3072", 1, 9216, 3072),
            ("dec down 1x3072x9216", 1, 3072, 9216), ("head 1x131072x3072", 1, 131072, 3072)]
        for (name, M, N, K) in shapes {
            for wdt in [DType.bf16, DType.f16] {
                let a = Tensor(ctx, shape: [M, K]), w = Tensor(ctx, shape: [N, K], dtype: wdt), c = Tensor(ctx, shape: [M, N])
                var best = Double.infinity
                for _ in 0..<5 { let s = Step(ctx); for _ in 0..<16 { s.gemm(a, w, out: c) }; best = min(best, s.run() / 16) }
                let gbs = Double(N * K * wdt.size) / best / 1e9
                print(String(format: "%-26@ %@  %8.3f ms  %6.1f GB/s", name as NSString, wdt.rawValue as NSString, best * 1000, gbs))
            }
        }
    }
}
