import Foundation
import Metal

public enum Activation: Int32 {
    case none = 0, relu = 1, silu = 2, geluTanh = 3, geluErf = 4
}

struct GemmParams {
    var M: Int32, N: Int32, K: Int32
    var lda: Int32, ldb: Int32, ldc: Int32
    var hasBias: Int32
    var act: Int32
    var hasResidual: Int32
    var ldr: Int32
    var alpha: Float
    var resScale: Float
    var outScale: Float
    var hasWScale: Int32 = 0
}

/// One unit of GPU work: a command buffer with a single serial compute encoder.
/// A whole streaming step is recorded into one `Step` and committed once.
public final class Step {
    public let ctx: MetalContext
    public let cb: MTLCommandBuffer
    public let enc: MTLComputeCommandEncoder

    public init(_ ctx: MetalContext = .shared) {
        self.ctx = ctx
        self.cb = ctx.queue.makeCommandBuffer()!
        self.enc = cb.makeComputeCommandEncoder()!
    }

    /// Ends encoding, commits and waits. Returns GPU execution time in seconds.
    @discardableResult
    public func run() -> Double {
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { fatalError("GPU error: \(e)") }
        return cb.gpuEndTime - cb.gpuStartTime
    }

    func bind(_ t: Tensor, _ index: Int) {
        enc.setBuffer(t.buffer, offset: t.offset, index: index)
    }

    func setBytes<T>(_ v: T, _ index: Int) {
        var vv = v
        enc.setBytes(&vv, length: MemoryLayout<T>.stride, index: index)
    }

    func dispatch(_ p: MTLComputePipelineState, threadgroups: MTLSize, threads: MTLSize) {
        enc.setComputePipelineState(p)
        enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)
    }

    func dispatch1D(_ p: MTLComputePipelineState, count: Int) {
        enc.setComputePipelineState(p)
        let w = min(p.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }
}

/// Kill-switch for the skinny GEMV path (DICTATOR_GEMV=0 disables it).
let gemvEnabled = ProcessInfo.processInfo.environment["DICTATOR_GEMV"] != "0"
/// Kill-switch for GEMM v2 (DICTATOR_GEMM_V1=1 uses the original tiled kernel).
let gemmV1 = ProcessInfo.processInfo.environment["DICTATOR_GEMM_V1"] == "1"

private func dtypeTag(_ d: DType) -> String {
    switch d {
    case .f32: return "f32"
    case .f16: return "f16"
    case .bf16: return "bf16"
    case .i8: return "i8"
    default: fatalError("no GEMM for \(d)")
    }
}

extension Step {
    /// out[M,N] = outScale * act(alpha * a[M,K] · w[N,K]^T + bias) + resScale * residual.
    /// `a` may be any 2-D (or flattened) tensor whose last dim is K.
    public func gemm(_ a: Tensor, _ w: Tensor, bias: Tensor? = nil, out: Tensor,
                     act: Activation = .none, alpha: Float = 1, outScale: Float = 1,
                     residual: Tensor? = nil, resScale: Float = 1) {
        let K = a.shape.last!
        let M = a.count / K
        let N = w.shape[0]
        precondition(w.shape.count == 2 && w.shape[1] == K, "gemm: a \(a.shape) w \(w.shape)")
        precondition(out.count == M * N, "gemm: out \(out.shape) expected \(M)x\(N)")
        let types = "\(dtypeTag(a.dtype))_\(dtypeTag(w.dtype))_\(dtypeTag(out.dtype))"
        // Skinny inputs are weight-bandwidth bound: use the GEMV kernel (reads each weight row once).
        let kAlign = w.dtype == .i8 ? 16 : (w.dtype.size == 2 ? 8 : 4)
        let skinny = M <= 8 && K % kAlign == 0 && gemvEnabled
        // GEMM v2 uses 32x64 tiles (a 64x64 v2 tile measured slower; see PERF_LOG R1).
        // DICTATOR_GEMM_V1=1 restores the v1 kernel.
        let name = skinny ? "gemv_nt_" + types : (gemmV1 ? "gemm_nt_" + types : "gemm2_32x64_" + types)
        let p = try! ctx.pipeline(name)
        var params = GemmParams(M: Int32(M), N: Int32(N), K: Int32(K), lda: Int32(K), ldb: Int32(K), ldc: Int32(N),
                                hasBias: bias == nil ? 0 : 1, act: act.rawValue,
                                hasResidual: residual == nil ? 0 : 1, ldr: Int32(N),
                                alpha: alpha, resScale: resScale, outScale: outScale,
                                hasWScale: w.qscale == nil ? 0 : 1)
        precondition(w.dtype != .i8 || w.qscale != nil, "int8 weight without scales")
        if let q = w.qscale { bind(q, 6) } else { bind(out, 6) }
        bind(a, 0); bind(w, 1); bind(out, 2)
        if let b = bias { precondition(b.dtype == .f32); bind(b, 3) } else { bind(out, 3) }
        if let r = residual { precondition(r.dtype == out.dtype); bind(r, 4) } else { bind(out, 4) }
        enc.setBytes(&params, length: MemoryLayout<GemmParams>.stride, index: 5)
        if skinny {
            dispatch(p, threadgroups: MTLSize(width: (N + 15) / 16, height: 1, depth: 1),
                     threads: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            let bm = gemmV1 ? 64 : 32
            dispatch(p, threadgroups: MTLSize(width: (N + 63) / 64, height: (M + bm - 1) / bm, depth: 1),
                     threads: MTLSize(width: 128, height: 1, depth: 1))
        }
    }
}
