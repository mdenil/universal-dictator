# Pinned sources

`scripts/verify_pins.sh` recomputes the hashes below and fails if any has changed. The reference scripts and parity results are only valid against these pins.

## Checkpoints (Hugging Face revision, SHA-256)

| Model | Repo @ revision |
|---|---|
| Nemotron Speech Streaming 0.6B | `nvidia/nemotron-speech-streaming-en-0.6b` @ `ebe59e5a817142986528bbbee5dba8db7b38ed50` |
| Parakeet Unified 0.6B | `nvidia/parakeet-unified-en-0.6b` @ `fe53cd885760c96b6a5f51a0bfd362cb4584a98b` |
| Voxtral Mini 4B Realtime | `mistralai/Voxtral-Mini-4B-Realtime-2602` @ `2769294da9567371363522aac9bbcfdd19447add` |

| File | SHA-256 |
|---|---|
| `models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo` | `283638054c44f6794e74fe9af9048d78a6d9d6c058c12131856c7859a62ac9cd` |
| `models/nemotron-speech-streaming-en-0.6b/model.safetensors` | `bddd8a7300826efd19cf7e01f1c7db8402bed6786fc4c7739632894f69c71473` |
| `models/nemotron-speech-streaming-en-0.6b/config.json` | `dffe850bc79ad2b0f8117804502b24d2c4a445aafbed4c1e40f8d78e0cb44065` |
| `models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b.nemo` | `ec23ed9150c8fde49072c3e2d61678ab903dbcef389d658db833420cbc1da35b` |
| `models/Voxtral-Mini-4B-Realtime-2602/model.safetensors` | `e745e4902df6a4c48f29f2f8dc1f6d0fb4cc73c7156bc45923451a5bcdfcd1d6` |
| `models/Voxtral-Mini-4B-Realtime-2602/params.json` | `2ace010ebf7f0b62c60747d91c6d140e3c7238632d3e9c63d60a2bd2065ea301` |
| `models/Voxtral-Mini-4B-Realtime-2602/config.json` | `96582f5809ad179531d47f9f89d76c696058fbdeff608d6611dcb7e076ea7e93` |
| `models/Voxtral-Mini-4B-Realtime-2602/tekken.json` | `8434af1d39eba99f0ef46cf1450bf1a63fa941a26933a1ef5dbbf4adf0d00e44` |

## Reference runtimes

| Environment | Pins | Why |
|---|---|---|
| `reference/nemo` | nemo_toolkit 2.7.3, torch 2.14.0, transformers 4.57.6 (locked in `uv.lock`) | NeMo 2.7.3 is the runtime named on the Parakeet model card. It lacks Parakeet's `chunked_limited_with_rc` mask, so `reference/nemo/common.py` applies the upstream encoder methods verbatim from NeMo `c42e4445127650492f96a7b47a39a4f449fdcf50` (sparse checkout in `third_party/NeMo`). See `docs/specs/parakeet.md` §0. |
| `reference/voxtral` | transformers 5.17.0, mistral-common 1.12.0, torch 2.14.0 | These are the transformers and mistral-common versions with `VoxtralRealtime` support. |
| `reference/coreml` | coremltools 9.0, torch 2.7.0 | Used only by the M5 Neural Engine study. |

## Reference ports

- `third_party/voxtral.c` @ `134d366c24d20c64b614a3dcc8bda2a6922d077d`.
