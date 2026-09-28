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
