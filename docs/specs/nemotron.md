# Nemotron Speech Streaming English 0.6B: behavioural spec

Checkpoint: `models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo`, SHA-256 `283638054c44…c9cd`. This is the March 2026 checkpoint: on the pinned HF revision `ebe59e5`, the `.nemo` was last changed by commit `2ea504f` "update the model checkpoint" (2026-03-13). Class `EncDecRNNTBPEModel`; the config says it was saved with `nemo_version: 2.4.0rc0`.

Reference runtime: the NeMo 2.7.3 wheel in `reference/nemo/.venv`, run on CPU in fp32 with dither 0. Citations are `file:line` in that wheel. The path prefixes are:

| Short | Path |
|---|---|
| `features.py` | `nemo/collections/asr/parts/preprocessing/features.py` |
| `audio_preprocessing.py` | `nemo/collections/asr/modules/audio_preprocessing.py` |
| `subsampling.py` | `nemo/collections/asr/parts/submodules/subsampling.py` |
| `causal_convs.py` | `nemo/collections/asr/parts/submodules/causal_convs.py` |
| `conformer_encoder.py` | `nemo/collections/asr/modules/conformer_encoder.py` |
| `conformer_modules.py` | `nemo/collections/asr/parts/submodules/conformer_modules.py` |
| `mha.py` | `nemo/collections/asr/parts/submodules/multi_head_attention.py` |
| `streaming_utils.py` | `nemo/collections/asr/parts/utils/streaming_utils.py` |
| `streaming.py` | `nemo/collections/asr/parts/mixins/streaming.py` |
| `mixins.py` | `nemo/collections/asr/parts/mixins/mixins.py` |
| `rnnt.py` | `nemo/collections/asr/modules/rnnt.py` |
| `label_looping.py` | `nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py` |
| `rnn.py` | `nemo/collections/common/parts/rnn.py` |
| `sp_tok.py` | `nemo/collections/common/tokenizers/sentencepiece_tokenizer.py` |
| `ca_infer.py` | `third_party/nemo-v2.7.3-examples/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py` (from the `v2.7.3` git tag; the wheel does not ship examples) |

Every statement below was checked against the source. The ones marked **[verified]** were also reproduced numerically by `reference/nemo/check_spec.py`, which re-implements the model from this spec with plain torch ops on the raw `state_dict` and compares each stage with the golden tensors (see §10).

Notation: `T_mel` = mel frames, `T` = encoder frames (after 8× subsampling), `D` = 1024, `H` = 8 heads, `d_k` = 128. Row-major, time-major unless noted.

## 1. Frontend (log-mel)

Config (`model_config.yaml` → `AudioToMelSpectrogramPreprocessor`): `sample_rate 16000, normalize NA, window_size 0.025, window_stride 0.01, window hann, features 128, n_fft 512, log true, frame_splicing 1, dither 1e-5, pad_to 0, pad_value 0.0`. Everything else takes the class defaults (`audio_preprocessing.py:202-231`): `preemph 0.97, lowfreq 0, highfreq None, log_zero_guard_type "add", log_zero_guard_value 2**-24, exact_pad False, mag_power 2.0, mel_norm "slaney"`.

Given mono float32 audio `x[0..N-1]` (16 kHz, in [-1, 1)):

1. **Window and hop.** `win_length = int(0.025*16000) = 400`, `hop = int(0.01*16000) = 160` (`audio_preprocessing.py:240-243`). `n_fft = 512`.
2. **Dither** is applied only when `self.training` (`features.py:424-426`); at inference it is 0. The reference scripts also set `dither = 0.0` and `pad_to = 0` explicitly (`streaming_utils.py:1716-1726`).
3. **Preemphasis** over the whole signal (`features.py:428-432`): `y[0] = x[0]`, `y[n] = x[n] - 0.97*x[n-1]`. Samples at or beyond the length are zeroed (only matters for batches).
4. **STFT** (`features.py:376-386`): `torch.stft(y, n_fft=512, hop_length=160, win_length=400, center=True, pad_mode="constant", window=<checkpoint buffer>, return_complex=True)`. Because `exact_pad` is False, `center=True`: `y` is zero-padded by 256 samples on both sides. Frame `t` covers padded samples `[160t, 160t+512)`, i.e. original samples `[160t-256, 160t+256)`. The 400-tap window is zero-padded to 512 **centred**: taps 56..455 hold the window, the rest are 0. Frames produced: `T_mel = N // 160 + 1`.
5. **Window.** A **symmetric** Hann (`torch.hann_window(400, periodic=False)`, `features.py:316`), so `w[0] = w[399] = 0`. **Trap:** use the checkpoint buffer `preprocessor.featurizer.window` [400]. It differs from today's `torch.hann_window` in 59 taps by up to 6e-8, and the checkpoint buffer is what `stft` uses (`features.py:383`).
6. **Magnitude, then power** (`features.py:441-451`): `mag = sqrt(re² + im²)`, then `p = mag ** 2.0`. (Computing `re²+im²` directly differs by rounding only.)
7. **Mel** (`features.py:328-334, 459-461`): `mel = fb @ p`, with `fb` [128, 257] built by `librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm="slaney")`, which uses the Slaney mel scale (`htk=False`) and Slaney area normalization. **Trap:** use the checkpoint buffer `preprocessor.featurizer.fb` [1,128,257]. It differs from librosa 1.0's output by up to 1.9e-9 (it was built with an older librosa).
8. **Log** (`features.py:463-465`): `log(mel + 2**-24)` (natural log, additive guard).
9. **Normalization:** none. `normalize: NA` is a truthy string, so `normalize_batch` is called, but `"NA"` matches no branch and the input is returned unchanged (`features.py:50-103`, else-branch at 102-103).
10. **Length and last frame** (`features.py:403-407, 479-483`): `seq_len = (N + 2*256 - 512) // 160 = N // 160`. Frames `t >= seq_len` are set to `pad_value = 0.0`. So the tensor has `N//160 + 1` frames, and the **last one is always 0.0** (not log(2^-24)). `pad_to = 0`, so no further padding.

Output: `mel` [128, T_mel], float32, plus `mel_len = N // 160`. [verified: `check_spec.py` reproduces `mel.npy`]

**Chunk edges.** The official streaming path never computes mel per chunk: `CacheAwareStreamingAudioBuffer.append_audio_file` runs the preprocessor once on the whole file and then slices mel frames (`streaming_utils.py:1728-1738, 1778-1786`). An incremental frontend matches it exactly if it (a) carries the last input sample for preemphasis (only sample 0 of the stream passes through unchanged); (b) treats the stream start as 256 zeros of left padding; and (c) emits frame `t` once samples up to `160t+199` exist (the last non-zero window tap). At end of stream, the right padding is 256 zeros, and the final frame `N//160` is 0.0.

**Streaming-buffer trap:** `append_processed_signal` records the stream length as `processed_signal.size(-1)` = `N//160 + 1`, not `mel_len` (`streaming_utils.py:1741, 1765`). So in the official streaming path the zeroed last frame counts as a **valid** frame. The offline path (`encoder(mel, mel_len)`) masks it out. See §6.4.

## 2. Subsampling (`dw_striding`, causal, 8×)

Config: `subsampling dw_striding, subsampling_factor 8, subsampling_conv_channels 256, causal_downsampling true`. Module: `ConvSubsampling` (`subsampling.py:62-436`) with `activation = nn.ReLU(True)` (`conformer_encoder.py:381-390`).

Layer list (`subsampling.py:142-215`), as `pre_encode.conv.{i}`:

| i | op | weight |
|---|---|---|
| 0 | `CausalConv2D(1→256, k=3, s=2)` | `[256,1,3,3]` + bias |
| 1 | ReLU | — |
| 2 | `CausalConv2D(256→256, k=3, s=2, groups=256)` (depthwise) | `[256,1,3,3]` + bias |
| 3 | `Conv2d(256→256, k=1)` (pointwise) | `[256,256,1,1]` + bias |
| 4 | ReLU | — |
| 5 | `CausalConv2D(256→256, k=3, s=2, groups=256)` | `[256,1,3,3]` + bias |
| 6 | `Conv2d(256→256, k=1)` | `[256,256,1,1]` + bias |
| 7 | ReLU | — |

Input: mel transposed to [T_mel, 128], viewed as an image `[B, 1, T_mel, 128]`: height = time, width = frequency (`conformer_encoder.py:628`, `subsampling.py:728`).

**Causal padding** (`causal_convs.py:24-69`): `CausalConv2D` sets `_left_padding = k-1 = 2`, `_right_padding = s-1 = 1` and runs `F.pad(x, (2, 1, 2, 1))`, then a conv with `padding=0`. The padding is applied to **both the time axis and the frequency axis** (top 2 / bottom 1 in time, left 2 / right 1 in frequency). Output sizes per stride-2 stage: `floor((L + 3 - 3)/2) + 1 = floor(L/2) + 1`. Frequency: 128 → 65 → 33 → **17**. Time: `T_mel → ⌊T_mel/2⌋+1 → … ` three times.

**Time masking** (`MaskedConvSequential`, `subsampling.py:725-759`): before **every** layer in the list (convs and ReLUs alike), and once more after the last one, time rows `>= current_length` are multiplied by 0. `current_length` starts at the input length. It is updated after each stride-2 layer by `(L + 2 + 1 - 3) // 2 + 1` (`calculate_conv_output_size` with the layer's `(_left_padding, _right_padding)`, `subsampling.py:720-722, 741-749`). Frequency is never masked. Because the pointwise convs have a bias, masked rows are non-zero right after them, but the next mask zeros them again.

**Output length** (`calc_length`, `subsampling.py:576-586`, all_paddings = 3): `L → floor(L/2) + 1`, three times.

**Flatten order** (`subsampling.py:429-431`): `x` [B, C=256, T, F=17] → `transpose(1,2)` → [B, T, 256, 17] → reshape [B, T, 4352]. So the feature index is `c*17 + f` (channel-major). Then `pre_encode.out`: Linear(4352 → 1024) with bias. The ReLU after the last conv comes **before** the flatten. There is no activation after `out`.

**Receptive field:** encoder frame `k` depends on mel frames `8k-14 … 8k`. [verified: `pre_encode.npy`]

Then `pos_enc` (`mha.py:1077-1100`): `xscaling: false`, so no `sqrt(d_model)` scaling. Dropout is identity at inference. The layer-0 input equals the subsampling output.

## 3. Encoder layer (×24)

`ConformerLayer.forward` (`conformer_modules.py:160-233`). The config has `use_bias: false`, so **no Linear or Conv1d inside the layers has a bias**; LayerNorms do. `fc_factor = 0.5` (`conformer_modules.py:91`). All norms are `torch.nn.LayerNorm(1024)` with eps 1e-5. Dropout is identity.

```
r  = x
r  = r + 0.5 * FF1(LN_ff1(r))                        # conformer_modules.py:174-177
r  = r + MHSA(LN_att(r), pos_emb, att_mask)          # :179-192
r  = r + Conv(LN_conv(r), pad_mask)                  # :205-209
r  = r + 0.5 * FF2(LN_ff2(r))                        # :211-213
out = LN_out(r)                                      # :215
```

**Feed-forward** (`conformer_modules.py:366-387`): `linear2(swish(linear1(z)))`, 1024 → 4096 → 1024. `Swish` = `nn.SiLU` = `x·sigmoid(x)`.

**Convolution module** (`conformer_modules.py:236-350`; config `conv_kernel_size 9, conv_norm_type layer_norm, conv_context_size causal`):

1. `pointwise_conv1`: Conv1d 1024 → 2048, k=1, no bias.
2. `glu(dim=channels)`: `a[:1024] * sigmoid(a[1024:])` (first half is the value, second half the gate).
3. Padded time steps are set to 0 (`pad_mask`, `:330-331`).
4. `depthwise_conv`: `CausalConv1D(1024, 1024, k=9, groups=1024, padding=[8, 0])`, no bias. `conv_context_size: causal` → `[kernel-1, 0] = [8, 0]` (`conformer_encoder.py:901-902`). So there are 8 zeros of left padding and none on the right (`causal_convs.py:105-110, 130-133`). Output `t` = Σ_{k=0..8} `w[c,k] · in[t-8+k]`.
5. **`batch_norm` is a LayerNorm** over the 1024 channels, per time step (`:300-301, 337-340`), eps 1e-5. **Trap:** the parameter is named `conv.batch_norm.{weight,bias}`, and there are no running statistics.
6. Swish.
7. `pointwise_conv2`: Conv1d 1024 → 1024, k=1, no bias.

**Relative-position multi-head self-attention** (`RelPositionMultiHeadAttention`, `mha.py:212-354`; `untie_biases: true`, so each layer has its own `pos_bias_u/v` [8,128]):

Inputs: `z = LN_att(r)` [T, 1024], `pos_emb` [2T-1, 1024] (below), `mask` [T, T] (True = hidden).

```
q = z Wq^T,  k = z Wk^T,  v = z Wv^T                → [T, H, d_k]    (mha.py:104-123)
p = pos_emb Wpos^T                                   → [2T-1, H, d_k] (linear_pos, never has a bias, mha.py:245)
ac[h,i,j] = (q[i,h] + u[h]) · k[j,h]                  (mha.py:302, 346)
bd_raw[h,i,r] = (q[i,h] + v_b[h]) · p[r,h]            (mha.py:304, 313)
bd = rel_shift(bd_raw)[:, :, :, :T]                  (mha.py:259-270, 347)
    ⇒ bd[h,i,j] = bd_raw[h, i, (T-1) - i + j]         (row r of pos_emb is relative position (T-1) - r, so this is position i - j)
s = (ac + bd) / sqrt(d_k) = (ac + bd) / sqrt(128)    (mha.py:94, 348)
s[mask] = -10000.0                                   (INF_VAL, mha.py:52, 137)
a = softmax(s, dim=-1); a[mask] = 0.0                (mha.py:138)  → fully masked rows give 0
o = concat_h(a @ v) → linear_out                      (mha.py:143-146)
```

`use_pytorch_sdpa` is False (the default), so the explicit path above runs. **[verified]** The `rel_shift` index formula was checked against NeMo's pad/view trick.

**Relative positional table** (`RelPositionalEncoding`, `mha.py:1015-1100`): the table is built once for positions `L-1 … -(L-1)` with `L = max(5000, input length)` (`pos_emb_max_len 5000`, `conformer_encoder.py:480, 781-792`). For an input of `n` frames (plus `cache_len` in streaming), the slice used holds positions `n-1, n-2, …, -(n-1)`, i.e. row `r` ↔ position `n-1-r`, with 2n-1 rows. Values (fp32, `create_pe`):

```
div[i]   = exp( (2i) * -(ln(10000)/1024) ),  i = 0..511     (fp32 arange × python-double scalar → fp32)
pe[r,2i]   = sin(pos_r * div[i])
pe[r,2i+1] = cos(pos_r * div[i])
```

The same `pos_emb` is passed to all 24 layers. It is **not** added to `x` (Transformer-XL style). **[verified]** The table is bit-identical to `pos_emb.npy`.

## 4. Attention masks (streaming configs)

`att_context_style: chunked_limited`. The config lists four `att_context_size` values, `[[70,13],[70,6],[70,1],[70,0]]`. **Trap:** at inference the first one, `[70,13]` (1120 ms), is the default (`conformer_encoder.py:620-625, 908`). We must call `set_default_att_context_size([70,1])` (160 ms) or `([70,6])` (560 ms).

For `[L, R]` over `n` frames (`conformer_encoder.py:803-822`):

```
chunk = R + 1                              # 2 for [70,1], 7 for [70,6]
left_chunks = L // chunk                   # 35 for [70,1], 10 for [70,6]   (L % chunk must be 0)
visible(i,j) = 0 <= (i//chunk - j//chunk) <= left_chunks
```

Chunks are aligned to frame 0. A frame sees its whole chunk (up to R frames ahead) and exactly the previous `left_chunks` whole chunks (70 frames). The padding mask is combined with it (`:826-845`): frame pairs where either side is padding are hidden. **[verified]** The mask matches `att_mask.npy` with zero mismatches.

## 5. Offline golden mode

The whole utterance runs in one `encoder(audio_signal=mel, length=mel_len)` call with the target `att_context_size`. No caches are used. The conv module is causal anyway, so this equals the streaming computation (see §6.4 for the measured difference).

## 6. Cache-aware streaming (official path)

Driver: `perform_streaming` (`ca_infer.py:212-299`) with `CacheAwareStreamingAudioBuffer` (`streaming_utils.py:1557-1786`) and `conformer_stream_step` (`mixins.py:592-729`) → `cache_aware_stream_step` (`streaming.py:40-76`) → `ConformerEncoder.forward_internal` with caches (`conformer_encoder.py:593-759`) → `streaming_post_process` (`:523-546`).

### 6.1 Streaming parameters

These come from `setup_streaming_params` (`conformer_encoder.py:927-1032`), with `get_sampling_frames() = [1, 8]` and `get_streaming_cache_size() = [0, 9]` (`subsampling.py:379-383`). Printed from the loaded model:

| | [70,1] (160 ms) | [70,6] (560 ms) |
|---|---|---|
| `chunk_size` (mel frames, first / later) | [9, 16] | [49, 56] |
| `shift_size` | [9, 16] | [49, 56] |
| `pre_encode_cache_size` | [0, 9] | [0, 9] |
| `drop_extra_pre_encoded` | 2 | 2 |
| `valid_out_len` (encoder frames per chunk) | 2 | 7 |
| `last_channel_cache_size` | 70 | 70 |
| `cache_drop_size` | 0 | 0 |

The formulas are: `lookahead = R`; `chunk_size = [1 + 8R, 8 + 8R]`; `valid_out_len = (shift[1] - 8)//8 + 1 = R + 1`; `drop_extra_pre_encoded = 1 + (9-1)//8 = 2`.

### 6.2 Per-step input (`streaming_utils.py:1597-1695`)

- **Step 0:** the mel frames `[0, chunk_size[0])` (9 or 49). The pre-encode cache has size `pre_encode_cache_size[0] = 0`, and `drop_extra_pre_encoded = 0` at step 0 (`ca_infer.py:204-209`). Subsampling of 9 frames gives 9 → 5 → 3 → **2** encoder frames (49 → 25 → 13 → **7**).
- **Step k ≥ 1:** the input is the **9 mel frames just before the chunk** (the pre-encode cache; zero-padded on the left if fewer exist) followed by the next `chunk_size[1]` frames (16 or 56). Its length is `min(remaining + 9, 25)` or `(…, 65)`. Subsampling gives 25 → 13 → 7 → 4 (65 → 33 → 17 → 9) frames, and the **first 2 are dropped** (`conformer_encoder.py:636-638`). That leaves 2 (or 7) frames. These equal the offline frames, because encoder frame `k` only needs mel `8k-14 … 8k` (§2) and the 2 dropped frames are exactly the ones whose receptive field reaches the causal zero padding.
- The **last** step (`keep_all_outputs = streaming_buffer.is_buffer_empty()`, `ca_infer.py:271`) is not cropped to `valid_out_len`. Earlier steps are cropped to `valid_out_len` (`conformer_encoder.py:542-544`).

### 6.3 Caches

Initial state (`get_initial_cache_state`, `conformer_encoder.py:1034-1072`): all zeros.

- `cache_last_channel` [24, B, 70, 1024]: per layer, the last 70 rows of the **attention input `LN_att(r)`**, i.e. the normalized hidden states, not K or V. In `update_cache` (`mha.py:204-209`): `key = value = cat([cache, z])` (70 + t rows), and `next_cache = cat([cache[:, t:], z[:, :t]])[-70:]` (since `cache_drop_size = 0`, `q_keep_size = t`). K and V are recomputed from the cached rows every step. (Caching projected K/V instead, as PLAN.md M2 proposes, is equivalent.)
- `cache_last_time` [24, B, 1024, 8]: per layer, the last 8 time steps of the **depthwise-conv input**, i.e. the GLU output after the pad-mask fill (`causal_convs.py:130-142`): `new_x = cat([cache, x])`, `next_cache = new_x[:, :, -8:]`.
- `cache_last_channel_len` [B]: `min(prev_len + t_kept, 70)` (`conformer_encoder.py:645-647, 756`).

Inside a step with `t` new frames (`conformer_encoder.py:643-673`):

- `pos_enc(x, cache_len=70)` builds positions `70+t-1 … -(70+t-1)`, so query `i` (absolute index `70+i` in `[cache | new]`) and key `j` get relative position `70+i-j`.
- The mask is built over `70 + t` frames with chunk alignment from index 0 (70 is a multiple of the chunk, 2 or 7, so the alignment matches offline). Then `offset = 70 - cache_last_channel_len` hides the not-yet-filled cache slots `j < offset` (`:649, 831-835`). Only the last `t` query rows are kept (`:667-670`).
- The conv `pad_mask` also keeps only the new frames.

The output rows sent to the decoder are `valid_out_len` frames per step (the last step keeps everything).

### 6.4 Offline vs streaming equivalence and the end-of-stream frame

Every streaming frame outside the final chunk equals the offline frame up to fp32 noise. On dev20 the worst difference is ≤ 1e-5 max-abs, at both [70,1] and [70,6] (§10).

The final chunk differs. Two source behaviours cause this:

1. The buffer counts mel frame `N//160` as valid (§1), and that frame is always 0.0. The frames of the last chunk that see it differ from offline, where the frame is masked. The difference is up to 0.28 max-abs in the last 2 frames at [70,1], and in the last 6 frames at [70,6].
2. `__iter__` **stops** when the remaining piece is shorter than `sampling_frames[1] = 8` mel frames (`streaming_utils.py:1628-1638`). So up to 7 trailing mel frames are never encoded.

So the streaming output can have one more or one fewer encoder frame than offline. Offline has `calc_length(N//160)` frames. On dev20 at [70,1], 6 of 20 utterances lose their final token(s) because of this. Five lose the final `.`. One (`6432-63722-0000`) ends "…givin up fis" instead of "…givin up fishing." (§10).

This is reference behaviour. It is what the L5 baseline contains, and it is not something the engine has to copy.

## 7. RNN-T decoder, joint and greedy decoding

**Prediction network** (`RNNTDecoder`, `rnnt.py:552-897`; config `pred_hidden 640, pred_rnn_layers 2, blank_as_pad true, vocab_size 1024`):

- `embed`: `Embedding(1025, 640, padding_idx=1024)` (`rnnt.py:844-845`). Row 1024 (blank) is all zeros in the checkpoint (checked). Blank is also the start symbol: `_SOS = blank = 1024` (`label_looping.py:229`).
- `dec_rnn`: `torch.nn.LSTM(640, 640, num_layers=2)` (`rnn.py:78-89, 193-195`, `norm=None`, `proj_size=0`). PyTorch gate order is `i, f, g, o`; both biases are added: `gates = W_ih x + b_ih + W_hh h + b_hh`, `c' = σ(f)⊙c + σ(i)⊙tanh(g)`, `h' = σ(o)⊙tanh(c')`. Layer 1's input is layer 0's `h'`. Dropout is identity.
- Initial state: `h = c = 0` [2, B, 640] (`rnnt.py:868-897`).

**Joint** (`RNNTJoint`, `rnnt.py:1280-1758`; `joint_hidden 640, activation relu`):

```
f = enc  W_enc^T  + b_enc     # joint.enc:  1024 → 640, computed once per encoder frame (label_looping.py:298)
g = h_top W_pred^T + b_pred   # joint.pred:  640 → 640, computed once per emitted token
logits = W_out · relu(f + g) + b_out   # joint.joint_net.2: 640 → 1025
```

On CPU NeMo then applies `log_softmax` (`rnnt.py:1710-1716`: `log_softmax is None` and the tensor is not on CUDA). This does not change the argmax. The golden `joint_logits_first20.npy` holds the **raw** logits, before log_softmax.

**Greedy decoding** (`greedy_batch`, `max_symbols 10`, label looping, `label_looping.py:275-547`; `loop_labels=True` by default, `rnnt_greedy_decoding.py:614`). For one stream, this is equivalent to:

```
g, state = pred(SOS=blank, zero state)
t = 0; n = 0
while t < T:
    k = argmax(joint(f[t], g))
    if k == blank: t += 1; n = 0; continue
    emit k (frame t); g, state = pred(k, state); n += 1
    if n >= 10: t += 1; n = 0          # force advance after 10 symbols on one frame (label_looping.py:501-519)
```

`dump_golden.py` implements exactly this loop, and it checks the result against NeMo's own decoder on every golden utterance. They were identical on all 40 (utterance, config) pairs.

**State carried across streaming steps** (`label_looping.py:521-547`, `rnnt_greedy_decoding.py:780-830`): the LSTM `(h, c)`, the last projected prediction output `g`, the last label, and the decoded length. The next step continues from them. Tokens found in one step are never revised: the transcript is the concatenation.

**Detokenization** (`sp_tok.py:266-291`): non-legacy SentencePiece `decode_ids(ids)`. The model is BPE with 1024 pieces (id 0 = `<unk>`, the rest normal pieces), `add_dummy_prefix` true, `nmt_nfkc`, no byte fallback. Decoding joins the pieces, maps `▁` to a space and drops the leading space. `<unk>` decodes to ` ⁇ ` (unk_surface). The text is **cased with punctuation**. The same tokenizer file is used by Parakeet Unified (SHA-256 `07d4e5a6…b0a9`).

## 8. Tensor-name map

NeMo `state_dict` names (`models/nemotron-speech-streaming-en-0.6b/nemo_extracted/weights.safetensors` holds exactly these 653 tensors). The right-hand column gives the name in the HF `model.safetensors` (651 tensors, transformers naming; it has no window or fb). All tensors are F32.

| NeMo name | Shape | HF `model.safetensors` name | Notes |
|---|---|---|---|
| `preprocessor.featurizer.window` | [400] | — | symmetric Hann; use this buffer |
| `preprocessor.featurizer.fb` | [1,128,257] | — | Slaney mel; use this buffer |
| `encoder.pre_encode.conv.0.{weight,bias}` | [256,1,3,3], [256] | `encoder.subsampling.conv_in.*` | causal pad (2,1) in both axes |
| `encoder.pre_encode.conv.2.{weight,bias}` | [256,1,3,3], [256] | `encoder.subsampling.layers.0.depthwise_conv.*` | groups=256 |
| `encoder.pre_encode.conv.3.{weight,bias}` | [256,256,1,1], [256] | `encoder.subsampling.layers.0.pointwise_conv.*` | |
| `encoder.pre_encode.conv.5.{weight,bias}` | [256,1,3,3], [256] | `encoder.subsampling.layers.1.depthwise_conv.*` | |
| `encoder.pre_encode.conv.6.{weight,bias}` | [256,256,1,1], [256] | `encoder.subsampling.layers.1.pointwise_conv.*` | |
| `encoder.pre_encode.out.{weight,bias}` | [1024,4352], [1024] | `encoder.subsampling.linear.*` | input index `c*17+f` |
| `encoder.layers.{L}.norm_feed_forward1.{weight,bias}` | [1024] | same | |
| `encoder.layers.{L}.feed_forward1.linear1.weight` | [4096,1024] | same | no bias |
| `encoder.layers.{L}.feed_forward1.linear2.weight` | [1024,4096] | same | no bias |
| `encoder.layers.{L}.norm_self_att.{weight,bias}` | [1024] | same | |
| `encoder.layers.{L}.self_attn.linear_{q,k,v}.weight` | [1024,1024] | `…self_attn.{q,k,v}_proj.weight` | separate Q/K/V, no bias; output rows grouped by head, `h*128+d` |
| `encoder.layers.{L}.self_attn.linear_out.weight` | [1024,1024] | `…self_attn.o_proj.weight` | |
| `encoder.layers.{L}.self_attn.linear_pos.weight` | [1024,1024] | `…self_attn.relative_k_proj.weight` | never has a bias |
| `encoder.layers.{L}.self_attn.pos_bias_u` | [8,128] | `…self_attn.bias_u` | per layer (untied) |
| `encoder.layers.{L}.self_attn.pos_bias_v` | [8,128] | `…self_attn.bias_v` | |
| `encoder.layers.{L}.norm_conv.{weight,bias}` | [1024] | same | |
| `encoder.layers.{L}.conv.pointwise_conv1.weight` | [2048,1024,1] | same | squeeze the k=1 axis; rows 0..1023 = value, 1024..2047 = gate |
| `encoder.layers.{L}.conv.depthwise_conv.weight` | [1024,1,9] | same | causal, tap 8 = current frame |
| `encoder.layers.{L}.conv.batch_norm.{weight,bias}` | [1024] | `…conv.norm.*` | **a LayerNorm**, despite the name |
| `encoder.layers.{L}.conv.pointwise_conv2.weight` | [1024,1024,1] | same | |
| `encoder.layers.{L}.norm_feed_forward2.*`, `feed_forward2.*` | as FF1 | same | |
| `encoder.layers.{L}.norm_out.{weight,bias}` | [1024] | same | |
| `decoder.prediction.embed.weight` | [1025,640] | `decoder.embedding.weight` | row 1024 = blank = 0 |
| `decoder.prediction.dec_rnn.lstm.weight_ih_l{0,1}` | [2560,640] | `decoder.lstm.weight_ih_l{0,1}` | gates i,f,g,o |
| `decoder.prediction.dec_rnn.lstm.weight_hh_l{0,1}` | [2560,640] | `decoder.lstm.weight_hh_l{0,1}` | |
| `decoder.prediction.dec_rnn.lstm.bias_{ih,hh}_l{0,1}` | [2560] | `decoder.lstm.bias_{ih,hh}_l{0,1}` | add both |
| `joint.pred.{weight,bias}` | [640,640], [640] | `decoder.decoder_projector.*` | **trap:** HF puts it under `decoder.` |
| `joint.enc.{weight,bias}` | [640,1024], [640] | `encoder_projector.*` | **trap:** HF top level |
| `joint.joint_net.2.{weight,bias}` | [1025,640], [1025] | `joint.head.*` | index 2 because 0=ReLU and 1=Dropout |

Other traps:
- There are 26 tensors per encoder layer: no biases in the layer Linears or Conv1ds (`use_bias: false`), but all LayerNorms have a bias.
- There is no BatchNorm to fold (unlike Parakeet).
- `nemo_extracted/weights.safetensors` and the HF `model.safetensors` were written by other tools. The goldens here come from the `.nemo` loaded by NeMo. Checked: with the mapping in the table above, all 651 HF tensors exist, and each is **bit-identical** to its `nemo_extracted` counterpart. No HF tensor is left unmapped.

## 9. Open questions (resolved)

1. **Mel defaults.** The mel scale and filter norm are both Slaney (`librosa.filters.mel(..., norm="slaney")`, `htk=False`; `audio_preprocessing.py:228`, `features.py:328-333`). The log guard is `log(x + 2**-24)` (`audio_preprocessing.py:217-218`, `features.py:463-465`). Padding is `center=True` with zero (`"constant"`) padding of 256 samples each side, because `exact_pad=False` (`features.py:376-386`). The window is a symmetric Hann(400), centred in 512 (`features.py:316`). The power is `|X|` then squared (`features.py:441-451`). There is no normalization (`normalize: NA` falls through `normalize_batch`, `features.py:102-103`). Dither is training-only (`features.py:424-426`). The last frame is zeroed (`features.py:479-483`). Checkpoint `window` and `fb` buffers differ from freshly computed ones at the 1e-8 level, so use the buffers. [verified]
2. **Causal subsampling padding.** `CausalConv2D` pads `(left=2, right=1)` on **both** time and frequency (`F.pad(x, (2,1,2,1))`, `causal_convs.py:46-47, 67`), so frequency goes 128 → 65 → 33 → 17 (Linear in = 4352). Only the three 3×3 convs are causal; the 1×1 convs are plain. Time rows beyond the running length are zeroed before every layer (`subsampling.py:725-759`). The streaming step keeps 9 mel frames of pre-encode cache and drops the first 2 subsampled frames (`conformer_encoder.py:636-638, 1015-1023`). [verified]
3. **Whether the sherpa-onnx exports use the March checkpoint.** Very probably yes. The sherpa-onnx export script loads `ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")` with no revision (`scripts/nemo/nemotron-speech-streaming-en-0.6b/export_onnx.py:55-57`). The export repos were created on 2026-04-25, and HF `main`'s `.nemo` has been the March checkpoint since 2026-03-13 (commit `2ea504f`, unchanged through the pinned `ebe59e5`). This is inferred from dates and has not been checked by comparing ONNX weights.
4. **Which lookahead is the default?** `[70,13]`, not 160 or 560 ms (`conformer_encoder.py:908`). The reference scripts must set it.

## 10. Noise floor (measured)

Envelope file: `data/golden/nemotron/noise_floor.json` (written by `reference/nemo/noise_floor.py`). Every number is the worst case over the 20 dev utterances in `data/dev/dev20.jsonl`, on CPU in fp32.

**Thread-count spread** (threads=1 vs threads=8, same code and input). "Relative" means max-abs / max|tensor|.

| Tensor | [70,1] max-abs | [70,1] relative | [70,6] max-abs | [70,6] relative |
|---|---|---|---|---|
| mel, pre_encode | 0 (bit-identical) | 0 | 0 | 0 |
| layer_00_out | 7.6e-6 | 2.1e-8 | 3.8e-6 | 1.1e-8 |
| worst layer output (layer 21 / 22; values reach ±183) | 2.8e-3 | 1.6e-5 | 2.0e-4 | 1.1e-6 |
| enc_out (layer 23, after LN) | 3.9e-6 | 1.1e-5 | 2.0e-6 | 7.4e-6 |
| raw joint logits (first 20 emissions) | 2.1e-4 | 1.3e-6 | 2.4e-4 | 1.8e-6 |

Tokens: identical on 20/20 utterances at both configs. The argmax of the first 20 joint steps was identical everywhere. Cosine similarity ≥ 0.999999999997 for every tensor.

**Spec re-implementation vs reference** (`check_spec.py`, first 3 utterances per config, each stage fed the golden input): mel, pre_encode, pos_emb and att_mask are bit-identical. Layer outputs match to ≤ 1.4e-4 max-abs (≤ 8.7e-7 relative), and tokens are identical. That is inside the thread spread, so the spec in §1–§7 is complete at fp32 resolution.

**Offline vs official cache-aware streaming** (`stream_<cfg>/…/stream_enc_out.npy` vs `offline_<cfg>/…/enc_out.npy`):

- Every frame before the final chunk: ≤ 9.6e-6 max-abs ([70,1]) and ≤ 5.6e-6 ([70,6]). That is thread-spread level.
- Final chunk: up to 0.28 max-abs, confined to the last 2 frames at [70,1] and the last 6 frames at [70,6]. There are frame-count differences of ±1 (7 utterances at [70,1], 2 at [70,6]). The causes are in §6.4.
- Tokens equal: 14/20 at [70,1] (the other 6 lose their final `.` or, in one case, "hing."), and 20/20 at [70,6].

A tolerance for kernel parity (L1–L3) should be derived from the thread-spread row: about 1e-5 relative on layer outputs and 2e-6 relative on logits. The streaming-vs-offline envelope holds only away from the final chunk.
