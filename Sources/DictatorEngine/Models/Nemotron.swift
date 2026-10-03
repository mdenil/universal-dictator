import Foundation

/// The two NeMo FastConformer + RNN-T models we run.
public enum NemoModelKind: String {
    case nemotron, parakeet

    public var dirName: String {
        switch self {
        case .nemotron: return "nemotron-speech-streaming-en-0.6b"
        case .parakeet: return "parakeet-unified-en-0.6b"
        }
    }
    public var config: FastConformerConfig { self == .nemotron ? .nemotron : .parakeet }
    /// Parakeet normalizes mel features per bin (`per_feature`); Nemotron does not.
    public var perFeatureNormalize: Bool { self == .parakeet }
}

/// A NeMo FastConformer + RNN-T model (Nemotron Speech Streaming or Parakeet Unified).
public final class NemotronModel {
    public let kind: NemoModelKind
    public let encoder: FastConformerEncoder
    public let decoder: RNNTDecoder
    public let frontend: NemoMelFrontend
    public let tokenizer: SentencePieceDecoder
    let jointEncW: Tensor, jointEncB: Tensor
    let weights: SafeTensorsFile

    /// Weight file for a precision: fp32 is the extracted NeMo checkpoint; others are produced by
    /// `dictate-cli convert`.
    public static func weightsURL(modelDir: URL, precision: String) -> URL {
        precision == "f32" ? modelDir.appendingPathComponent("nemo_extracted/weights.safetensors")
                           : modelDir.appendingPathComponent("dictator-\(precision).safetensors")
    }

    public init(modelDir: URL, kind: NemoModelKind = .nemotron, precision: String = "f32") throws {
        self.kind = kind
        let dir = modelDir.appendingPathComponent("nemo_extracted")
        weights = try SafeTensorsFile(NemotronModel.weightsURL(modelDir: modelDir, precision: precision))
        encoder = try FastConformerEncoder(weights: weights, config: kind.config)
        decoder = try RNNTDecoder(weights: weights)
        frontend = NemoMelFrontend(filterbank: try weights.floats("preprocessor.featurizer.fb"),
                                   window: try weights.floats("preprocessor.featurizer.window"))
        tokenizer = try SentencePieceDecoder(vocabFile: dir.appendingPathComponent("tokenizer.vocab"))
        jointEncW = try weights.tensor("joint.enc.weight")
        jointEncB = try weights.tensor("joint.enc.bias")
    }
}

/// Nemotron's streaming encoder step on the GPU (Metal), as one implementation of `NemotronEncoder`.
/// All GPU work for a chunk is one command buffer.
public final class MetalNemotronEncoder: NemotronEncoder {
    public let model: NemotronModel
    public let schedule: NemotronSchedule
    let ws: FastConformerEncoder.Workspace
    let state: FastConformerEncoder.StreamState
    let melIn: Tensor
    let encOut: Tensor
    let proj: Tensor
    /// GPU seconds per step (for benchmarking).
    public private(set) var gpuTimes: [Double] = []
    /// Receives the kept encoder output rows of each chunk (parity testing).
    public var encTap: (([Float]) -> Void)?

    public init(model: NemotronModel, schedule: NemotronSchedule) {
        self.model = model
        self.schedule = schedule
        let maxMel = schedule.preEncodeCache + schedule.chunk
        let maxT = model.encoder.subsampledLength(maxMel)
        ws = model.encoder.makeWorkspace(maxT: maxT, maxMel: maxMel)
        state = model.encoder.makeStreamState(leftContext: schedule.leftContext, maxChunk: maxT)
        melIn = Tensor(shape: [maxMel, model.encoder.config.melBins])
        encOut = Tensor(shape: [maxT, model.encoder.config.dModel])
        proj = Tensor(shape: [maxT, model.decoder.hidden])
    }

    public func step(mel melRows: UnsafeBufferPointer<Float>, nRows: Int, validRows: Int, keepAll: Bool) throws -> (proj: [Float], frames: Int) {
        let bins = model.encoder.config.melBins
        let chunkMel = melIn.rows(0, nRows)
        memcpy(chunkMel.contents, melRows.baseAddress!, nRows * bins * 4)
        let s = Step()
        let drop = state.step == 0 ? 0 : schedule.dropExtra
        let (_, validT) = model.encoder.encodeStreamStep(s, ws, state: state, melChunk: chunkMel, validMel: validRows,
                                                         drop: drop, out: encOut)
        // NeMo keeps only valid_out_len frames except on the final chunk.
        let keep = keepAll ? validT : min(validT, schedule.outFramesPerChunk)
        if keep > 0 {
            s.gemm(encOut.rows(0, keep), model.jointEncW, bias: model.jointEncB, out: proj.rows(0, keep))
        }
        gpuTimes.append(s.run())
        guard keep > 0 else { return ([], 0) }
        if let tap = encTap { tap(encOut.rows(0, keep).toFloats()) }
        return (proj.rows(0, keep).toFloats(), keep)
    }
}

/// Nemotron streaming on the GPU: the shared streamer driven by the Metal encoder.
public typealias NemotronStream = NemotronStreamer

extension NemotronStreamer {
    /// A Metal stream for `model` with `lookahead` frames of right context (160 ms: 1, 560 ms: 6).
    public convenience init(model: NemotronModel, lookahead: Int) {
        self.init(encoder: MetalNemotronEncoder(model: model, schedule: NemotronSchedule(lookahead: lookahead)),
                  decoder: model.decoder, tokenizer: model.tokenizer, frontend: model.frontend)
    }

    /// The Metal encoder's parity tap (Metal streams only).
    public var encTap: (([Float]) -> Void)? {
        get { (encoder as? MetalNemotronEncoder)?.encTap }
        set { (encoder as? MetalNemotronEncoder)?.encTap = newValue }
    }
}
