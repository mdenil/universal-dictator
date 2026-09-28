# Discrepancies from the reference

This file lists every accepted difference between our engine and the reference implementation. It is read by the M6 evaluation and by the default-settings decision. An entry is removed when its change becomes the default or is deleted.

## D1: Nemotron fp16 weights

- **Change:** encoder linear and pointwise-conv weights are stored in fp16. Activations, accumulation, norms, biases, depthwise convs, the subsampling convs, the frontend and the RNN-T decoder stay fp32.
- **How to select it:** `--precision f16`, which reads `dictator-f16.safetensors`. `--precision f32` reads the extracted NeMo weights.
- **Measured (2026-09-28):**
  - dev20 offline at [70,1] and [70,6]: tokens exact on 20/20, worst per-layer cosine 0.9999984;
  - NeMo streaming at 160 ms: tokens exact on 9/9.
  - Command: `dictate-cli ladder nemotron --golden-root data/golden/nemotron/offline_70_{1,6} --stream-root data/golden/nemotron/stream_70_{1,6} --context 70,{1,6} --precision f16`.
- **Not yet measured:** WER on the full LibriSpeech test sets.
- **Review:** at M6.

## D2: Parakeet exact buffered mode, trailing period on the final chunk

- **What:** our Stage A buffered streaming reproduces NeMo's official buffered transcripts exactly on 12 of the 20 dev utterances. On the other 8, the **only** difference is a final "." that we emit at the last chunk and NeMo does not. Every other token matches.
- **Cause:** not yet identified. Our loop follows `speech_to_text_streaming_infer_rnnt.py` (c42e444) line by line: the last-chunk flag, `out_len = enc_len - left` on the last chunk, and the left-frame offset. The offline ladder is token-exact on 20/20, so the encoder itself agrees.
- **Effect:** none on normalized WER (the normalizer strips punctuation). A sentence-final period is arguably the better dictation output.
- **Measured:** `dictate-cli ladder parakeet --golden-root data/golden/parakeet/offline_70_2_5 --stream-root data/golden/parakeet/stream_70_2_5`.
- **Review:** at M6. Instrument the reference's last step (encoder length and emitted tokens) if exact end-of-stream parity is needed.
