import Foundation
import Metal

/// Measures this machine's real GPU peaks: the roofline every timing row is compared against.
public enum Peaks {
    public struct Result: CustomStringConvertible {
        public var copyGBs: Double
        public var readGBs: Double
        public var fmaF32TFLOPS: Double
        public var fmaF16TFLOPS: Double
        public var emptyCommandBufferUs: Double
        public var gemmF32TFLOPS: Double
        public var gemmF16TFLOPS: Double

        public var description: String {
            String(format: """
            copy bandwidth      %7.1f GB/s (read+write counted)
            read bandwidth      %7.1f GB/s
            fma f32             %7.2f TFLOPS
            fma f16             %7.2f TFLOPS
            gemm f32 4096^3     %7.2f TFLOPS (current kernel)
            gemm f16 4096^3     %7.2f TFLOPS (current kernel)
            empty cmd buffer    %7.1f us round trip
            """, copyGBs, readGBs, fmaF32TFLOPS, fmaF16TFLOPS, gemmF32TFLOPS, gemmF16TFLOPS, emptyCommandBufferUs)
        }
    }

    /// Best-of-`reps` GPU time for the work encoded by `body`.
    static func best(_ reps: Int, _ ctx: MetalContext, _ body: (Step) -> Void) -> Double {
        var best = Double.infinity
        for _ in 0..<reps {
            let s = Step(ctx)
            body(s)
            best = min(best, s.run())
        }
        return best
    }

    public static func measure(_ ctx: MetalContext = .shared) throws -> Result {
        let n4 = 64 * 1024 * 1024 / 4  // 256 MB of float4 per buffer
        let src = ctx.makeBuffer(bytes: n4 * 16)
        let dst = ctx.makeBuffer(bytes: n4 * 16)
        memset(src.contents(), 1, n4 * 16)

        let copyP = try ctx.pipeline("bench_copy_f4")
        let tCopy = best(10, ctx) { s in
            s.enc.setBuffer(src, offset: 0, index: 0); s.enc.setBuffer(dst, offset: 0, index: 1)
            s.dispatch1D(copyP, count: n4)
        }

        let readP = try ctx.pipeline("bench_read_f4")
        let perThread: UInt32 = 64
        let threads = n4 / Int(perThread)
        let tRead = best(10, ctx) { s in
            s.enc.setBuffer(src, offset: 0, index: 0); s.enc.setBuffer(dst, offset: 0, index: 1)
            s.setBytes(perThread, 2)
            s.dispatch1D(readP, count: threads)
        }

        let fmaThreads = 1 << 20
        let iters: UInt32 = 4096
        let f32P = try ctx.pipeline("bench_fma_f32")
        let tF32 = best(5, ctx) { s in
            s.enc.setBuffer(dst, offset: 0, index: 0); s.setBytes(iters, 1)
            s.dispatch1D(f32P, count: fmaThreads)
        }
        let f16P = try ctx.pipeline("bench_fma_f16")
        let tF16 = best(5, ctx) { s in
            s.enc.setBuffer(dst, offset: 0, index: 0); s.setBytes(iters, 1)
            s.dispatch1D(f16P, count: fmaThreads)
        }
        let flops = Double(fmaThreads) * Double(iters) * 8 * 2

        // Empty command buffer wall-clock round trip (CPU submit -> GPU -> CPU wake).
        let noopP = try ctx.pipeline("bench_noop")
        var rt = [Double]()
        for _ in 0..<50 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let s = Step(ctx)
            s.enc.setBuffer(dst, offset: 0, index: 0)
            s.dispatch1D(noopP, count: 32)
            s.run()
            rt.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000)
        }
        rt.sort()

        let n = 4096
        let a32 = Tensor(ctx, shape: [n, n], dtype: .f32)
        let b32 = Tensor(ctx, shape: [n, n], dtype: .f32)
        let c32 = Tensor(ctx, shape: [n, n], dtype: .f32)
        let a16 = Tensor(ctx, shape: [n, n], dtype: .f16)
        let b16 = Tensor(ctx, shape: [n, n], dtype: .f16)
        let c16 = Tensor(ctx, shape: [n, n], dtype: .f16)
        let tG32 = best(3, ctx) { s in s.gemm(a32, b32, out: c32) }
        let tG16 = best(3, ctx) { s in s.gemm(a16, b16, out: c16) }
        let gflop = 2.0 * Double(n * n * n)

        return Result(
            copyGBs: Double(n4 * 16 * 2) / tCopy / 1e9,
            readGBs: Double(n4 * 16) / tRead / 1e9,
            fmaF32TFLOPS: flops / tF32 / 1e12,
            fmaF16TFLOPS: flops / tF16 / 1e12,
            emptyCommandBufferUs: rt[rt.count / 2],
            gemmF32TFLOPS: gflop / tG32 / 1e12,
            gemmF16TFLOPS: gflop / tG16 / 1e12)
    }
}
