# Parakeet Unified English 0.6B: behavioural spec

Checkpoint: `models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b.nemo` (2,474,055,680 bytes). Class `EncDecRNNTBPEModel`. The config is identical to `docs/research/parakeet/model_config.yaml`. The tokenizer is byte-identical to Nemotron's (SHA-256 `07d4e5a6…b0a9`).

This model shares its layer structure, positional encoding, decoder, joint and tokenizer with Nemotron. Where a section says "as Nemotron", `docs/specs/nemotron.md` gives the exact semantics and citations. This file states every difference in full.

## 0. Runtime: NeMo 2.7.3 cannot load this model

The model card says "Runtime Engine: NeMo 2.7.3", but the **2.7.3 wheel cannot instantiate this checkpoint**:

```
InstantiationException ... ConformerEncoder.__init__() got an unexpected keyword argument 'att_chunk_context_size'
```

(This is reproduced by instantiating `cfg.encoder` with the 2.7.3 wheel.) The encoder config uses `att_chunk_context_size`, `att_context_style: chunked_limited_with_rc` and `conv_context_style: dcc`. These were added to NeMo `main` by commit `c42e444` ("Add streaming inference support for Unified RNNT model", PR #15522, 2026-04-07), which is not an ancestor of the `v2.7.3` tag. No released NeMo (latest on PyPI: 2.7.3) contains it.

What the reference does instead (`reference/nemo/common.py: install_unified_shim`): it keeps the pinned 2.7.3 wheel and patches `ConformerEncoder` with upstream code objects taken from `c42e444`'s `conformer_encoder.py` (sparse checkout in `third_party/NeMo`). The patched pieces are `_create_masks`, `_calc_context_sizes`, `setup_streaming_params`, and a constructor wrapper that accepts and stores the two new arguments. `git diff v2.7.3 c42e444` over the asr modules used here (features, subsampling, conformer_modules, multi_head_attention, rnnt, rnnt_greedy_decoding, label looping, lstm_decoder, the buffered streaming script) shows no other behavioural change. The only other differences are docstrings, deleted unused classes, and a `keep_hypotheses` option in `RNNTJoint.forward` that is not used here. The streaming script is the `c42e444` version of `speech_to_text_streaming_infer_rnnt.py`. That is the v2.7.3 script plus the 12-line `att_context_size_as_chunk` block. Line citations for the `c42` file use the prefix `c42:conformer_encoder.py`. Everything else cites the 2.7.3 wheel with the same short names as in `nemotron.md`, plus:

| Short | Path |
|---|---|
| `c42:conformer_encoder.py` | `third_party/NeMo/nemo/collections/asr/modules/conformer_encoder.py` @ `c42e444` |
| `buf_infer.py` | `third_party/NeMo/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py` @ `c42e444` |

## 1. Frontend

As Nemotron §1 (same class, window, fb construction, preemphasis, `center=True` zero padding, magnitude-then-square, `log(x + 2**-24)`, last frame zeroed, `pad_to 0`, dither 0). The difference is **`normalize: per_feature`** (`features.py:50-84, 476-477`). After the log, for each of the 128 mel bins, over the **valid** frames `t < seq_len = N//160`:

```
mean[m] = Σ_t x[m,t] / seq_len
std[m]  = sqrt( Σ_t (x[m,t] - mean[m])² / (seq_len - 1) ) + 1e-5      # unbiased; 1e-5 added to the std, not the variance
x[m,t]  = (x[m,t] - mean[m]) / std[m]      for all t
```

Frames `t >= seq_len` are then set to 0.0 (`features.py:479-483`). If `seq_len == 1`, `std` becomes NaN → 0 before adding 1e-5 (`features.py:81`). The statistics come from the whole input the preprocessor sees. In the official buffered streaming (§6) that input is **the current 6.16 s buffer**, recomputed every 160 ms. In the offline goldens it is the whole utterance. **Trap:** as with Nemotron, use the checkpoint `window` and `fb` buffers.

## 2. Subsampling (`dw_striding`, non-causal, 8×)

The layer list is the same as Nemotron §2, but `causal_downsampling: false`. So the three 3×3 convs are plain `torch.nn.Conv2d(..., padding=1)`: symmetric 1-sample padding on both time and frequency (`subsampling.py:152-154, 167-202`).

- Output size per stage: `floor((L + 2 - 3)/2) + 1 = floor((L-1)/2) + 1`. Frequency 128 → 64 → 32 → **16**, so `pre_encode.out` is Linear(**4096** → 1024).
- Time masking before every layer, as Nemotron (`MaskedConvSequential`). The length update uses `padding = (1, 1)`: `(L + 2 - 3)//2 + 1`.
- Flatten is channel-major (`c*16 + f`), then `out` with bias.
- Receptive field: encoder frame `k` depends on mel frames `8k-7 … 8k+7`.

**xscaling: true.** `RelPositionalEncoding.forward` multiplies the subsampling output by `xscale = sqrt(1024) = 32` before layer 0 (`conformer_encoder.py:363-366`, `mha.py:1087-1088`). The golden `pre_encode.npy` is before this scaling. `enc_in.npy` is after it (= layer-0 input). [verified]

## 3. Encoder layer (×24)

Same op order as Nemotron §3: `x + ½FF1`, `+ MHSA`, `+ Conv`, `+ ½FF2`, then `LN_out`. Same relative-position attention formula, the same `-10000` masking, and the same positional table. The differences:

- **Biases everywhere** (`use_bias` defaults to True): `linear1/linear2`, `linear_q/k/v/out`, `pointwise_conv1/2` and `depthwise_conv` all have biases. **`linear_pos` never has a bias** (`mha.py:245`).
- **Depthwise conv is symmetric**: `conv_context_size: null` → `[(9-1)//2, (9-1)//2] = [4, 4]` (`conformer_encoder.py:907`). `CausalConv1D` with `padding=[4,4]` pads 4 zeros on each side (`causal_convs.py:105-110, 130-133`). Output `t` = `b[c] + Σ_{k=0..8} w[c,k]·in[t-4+k]`.
- **`conv_norm_type: batch_norm`**: `nn.BatchNorm1d(1024, eps=1e-5)` in eval mode, using the running statistics: `y = (x - running_mean)/sqrt(running_var + 1e-5) * weight + bias`. It can be folded into the depthwise conv: `w' = w·s`, `b' = (b - μ)·s + β`, with `s = γ/sqrt(σ² + 1e-5)`.
- `pos_bias_u/v` are per layer (`untie_biases: true`), [8,128] each.

**`conv_context_style: dcc`** (Dynamic Chunked Convolution) has no effect at inference in this code. `c42:conformer_encoder.py:358-359` validates and stores the value, and nothing reads it. Current `main` (`1f73a91`, the copy in `docs/research/parakeet/enc.py:373-374`) is the same, and `conformer_modules.py` has no DCC branch. So the depthwise conv is a plain symmetric k=9 conv over the whole encoder input. It is **not** limited to chunk boundaries. See Open questions.

## 4. Attention mask: `chunked_limited_with_rc` [70, 2, 5]

With the shim, `_calc_context_sizes` forces `att_context_size_all = [[-1,-1,-1]]` for this style (`c42:conformer_encoder.py:930-931`). So the model **loads in full-context (offline) mode**: `sum == -3`, so no chunk mask at all (`c42:conformer_encoder.py:843`). Streaming at 560 ms requires `set_default_att_context_size([70, 2, 5])`. `buf_infer.py:305-309` does this when `att_context_size_as_chunk=true`.

For `[left, chunk, right]` over `n` frames (`c42:conformer_encoder.py:843-868`):

```
c(i)       = i // chunk                                   # chunks aligned to frame 0 of the input
start(i)   = max(0, c(i)*chunk - left)
end(i)     = min(n-1, c(i)*chunk + chunk - 1 + right)
visible(i, j) = start(i) <= j <= end(i)
```

With [70, 2, 5], frame `i` sees from 70 frames before the **start of its chunk** to 5 frames past the **end of its chunk**. That is 70 or 71 frames back, and 5 or 6 frames ahead. The padding mask is combined with it, as in Nemotron. [verified]

`setup_streaming_params` has a `chunked_limited_with_rc` branch (`c42:conformer_encoder.py:1012-1014`), but the buffered pipeline never uses `streaming_cfg` or caches.

## 5. Offline golden mode

The whole utterance runs through the preprocessor (whole-utterance normalization) and one encoder call with the [70,2,5] mask. **This is not what the buffered streaming computes**, for three reasons:

- normalization statistics are per utterance, not per buffer;
- the symmetric conv (4 frames per layer, 24 layers) and the 7-frame subsampling receptive field let information from beyond the 5-frame attention right context reach a frame; in the buffered pipeline the buffer ends 5 encoder frames after the chunk, so those frames see zero padding instead;
- the left history is not truncated to the buffer.

Offline goldens are for L0–L4 kernel parity. The buffered transcripts are the L5 baseline.

## 6. Buffered streaming (official path, `buf_infer.py`)

Run with `left_context_secs=5.6 chunk_secs=0.16 right_context_secs=0.40 att_context_size_as_chunk=true` (`compute_dtype=float32`; the default `None` would pick bf16 only on CUDA, see `get_auto_inference_dtype`).

**Sizes** (`buf_infer.py:283-303`):
- `features_frame2audio_samples = make_divisible_by(int(16000*0.01), 8) = 160`
- `encoder_frame2audio_samples = 160*8 = 1280` (80 ms)
- encoder frames: `left = int(5.6*100/8) = 70`, `chunk = int(0.16*100/8) = 2`, `right = int(0.40*100/8) = 5`
- samples: `left = 70*1280 = 89600`, `chunk = 2560`, `right = 6400`

**Loop** (`buf_infer.py:382-447`, buffer logic in `StreamingBatchedAudioBuffer` / `ContextSize`, `streaming_utils.py:2167-2329`):

```
left_sample = 0; right_sample = min(chunk + right, len)       # first step: 8960 samples (0.56 s)
buffer = empty; decoder state = None
while left_sample < len:
    append audio[left_sample:right_sample] to the buffer
    context bookkeeping: previous chunk → left; the new samples go to the right part; if this is not
      the last chunk, the first `chunk` samples of the right part become the new chunk; else everything
      becomes the chunk and right = 0; left is trimmed from the front to at most 89600 samples
    enc, enc_len = model(input_signal=buffer, length=buffer_len)   # FULL preprocessor (per-buffer normalization)
                                                                   # + full 24-layer encoder with the [70,2,5] mask
    enc = enc[:, left_samples // 1280 :]                           # drop the left-context frames
    out_len = chunk_samples // 1280 (= 2)  or, on the last chunk, enc_len - left_samples // 1280
    hyps, state = label-looping greedy(enc, out_len, prev_state=state)   # decoder state carried (as Nemotron §7)
    left_sample = right_sample; right_sample = min(right_sample + 2560, len)
```

Consequences for an engine:
- Every 160 ms, the preprocessor runs over the whole buffer (up to 89600 + 2560 + 6400 = 98560 samples, 6.16 s). Preemphasis restarts at the buffer start (`y[0] = x[0]`). The buffer is zero-padded by 256 samples at both ends. The mel statistics are recomputed over the buffer's `len//160` valid frames. The buffer start is always a multiple of 1280 samples, so mel frames stay on the global 160-sample grid. Only the edge frames and the normalization differ from a whole-utterance mel.
- Up to 77 encoder frames go through all 24 layers each step. Only 2 frames are decoded.
- In the attention mask, chunk index 0 is the first buffer frame. The left part always holds an even number of encoder frames (it grows by 2 up to 70), so the decoded chunk is always a mask chunk, with exactly its 5 right frames present.
- Audio is loaded with `librosa.load(path, sr=16000)` (`streaming_utils.py:2332-2335`). This gives the same float32 samples as soundfile for 16 kHz FLAC.
- With `batch_size > 1`, utterances are zero-padded and handled through per-utterance lengths. The dev goldens use `batch_size=1`.

## 7. Decoder, joint, greedy, detokenization

As Nemotron §7: same shapes, same `max_symbols 10` label-looping greedy decoding, blank = SOS = 1024 with a zero embedding row, a 2-layer LSTM (640), the joint `W_out·relu(enc(f)+pred(g))`, and the same SentencePiece BPE model (cased, punctuated output). Checked: `dump_golden.py`'s plain greedy loop matched NeMo's decoder on all 20 dev utterances.

## 8. Tensor-name map

NeMo `state_dict` from the `.nemo` (989 tensors = 29 non-layer + 24 × 40 per layer). There is no HF safetensors for this repo. Differences from Nemotron's map (`nemotron.md` §8):

| NeMo name | Shape | Notes |
|---|---|---|
| `encoder.pre_encode.out.{weight,bias}` | [1024,**4096**], [1024] | input index `c*16+f` |
| `encoder.pre_encode.conv.{0,2,5}.*` | as Nemotron | **non-causal**, padding 1 |
| `encoder.layers.{L}.feed_forward{1,2}.linear1.bias` | [4096] | biases present |
| `encoder.layers.{L}.feed_forward{1,2}.linear2.bias` | [1024] | |
| `encoder.layers.{L}.self_attn.linear_{q,k,v,out}.bias` | [1024] | `linear_pos` has **no** bias |
| `encoder.layers.{L}.conv.pointwise_conv1.bias` | [2048] | |
| `encoder.layers.{L}.conv.depthwise_conv.bias` | [1024] | symmetric k=9 |
| `encoder.layers.{L}.conv.batch_norm.{weight,bias,running_mean,running_var}` | [1024] | **real BatchNorm1d**, eps 1e-5; fold into the depthwise conv |
| `encoder.layers.{L}.conv.batch_norm.num_batches_tracked` | [] int64 | ignore |
| `encoder.layers.{L}.conv.pointwise_conv2.bias` | [1024] | |

Everything else (norm names, `pos_bias_u/v`, LSTM, embed, joint, `preprocessor.featurizer.{window,fb}`) has the same names and shapes as Nemotron. **Trap:** a converter written for Nemotron silently drops the biases and misreads `conv.batch_norm` (a LayerNorm there, a BatchNorm here).

## 9. Open questions (resolved)

1. **Mel defaults.** As Nemotron (Slaney scale and norm, `log(x + 2**-24)`, `center=True` zero padding, symmetric Hann in 512, power 2), plus `per_feature` normalization with the unbiased std and `+1e-5` on the std, over valid frames only (`features.py:50-84`). [verified]
2. **Parakeet DCC behaviour.** At inference, `conv_context_style: dcc` does nothing. It is stored and never read (`c42:conformer_encoder.py:358-359`; the same holds on `main` 1f73a91). The depthwise conv is a symmetric k=9 conv over the whole input (buffer or utterance), not masked at chunk boundaries. DCC was a training-time technique (arXiv 2304.09325, cited by the card). The training code is not released ("We will release the unified training pipeline soon", model card). So there is nothing to port. The effect is that the effective lookahead in offline-with-mask mode exceeds the 5-frame attention right context, while the buffered pipeline truncates it at the buffer end (§5).
3. **Which NeMo can run it?** Not 2.7.3 as released (§0). The reference uses 2.7.3 plus the verbatim `c42e444` encoder methods.
4. **Default attention context.** Full context `[-1,-1,-1]` (§4). The [70,2,5] mask must be set explicitly.

## 10. Noise floor (measured)

Envelope file: `data/golden/parakeet/noise_floor.json`. Every number is the worst case over the 20 dev utterances, on CPU in fp32, for `offline_70_2_5`.

**Thread-count spread** (threads=1 vs threads=8):

| Tensor | max-abs | relative (max-abs / max\|tensor\|) |
|---|---|---|
| mel (normalized), pre_encode | 0 (bit-identical) | 0 |
| layer_00_out | 3.8e-6 | 7.0e-8 |
| worst layer output (layer 21; values reach ±150) | 6.2e-3 | 4.3e-5 |
| enc_out | 3.5e-6 | 1.2e-6 |
| raw joint logits (first 20 emissions) | 4.2e-5 | 9.4e-7 |

Tokens: identical on 20/20 utterances. The argmax of the first 20 joint steps was identical everywhere.

**Spec re-implementation vs reference** (`check_spec.py`, 3 utterances): the normalized mel matches to 2.4e-6 max-abs (the per-bin mean/std reduction order differs). pre_encode, enc_in (×32), pos_emb and att_mask are bit-identical. Layer outputs match to ≤ 3.1e-5 max-abs (≤ 3.7e-7 relative), and tokens are identical.

**Offline-with-mask vs official buffered streaming** (different algorithms, §5): 8/20 dev transcripts are identical. The rest differ mostly in punctuation, casing and spelling variants ("Grey"/"Gray"). Normalized WER on dev20 is 4.11% offline_70_2_5 vs 5.90% buffered. The buffered dev20 golden (`data/golden/parakeet/stream_70_2_5/`) was produced on **MPS fp32** with `batch_size=1`. On this machine's CPU the buffered pipeline ran at about 1/20 of real time. The same run at `batch_size=20` on MPS gave identical transcripts on 20/20 utterances.
