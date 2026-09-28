# Voxtral Mini 4B Realtime: behavioural spec

Status: M0 reference spec. Written for whoever writes the Metal kernels and the Swift decode loop. Every statement below is either cited to the pinned source or was measured with the scripts in `reference/voxtral/`. The open-questions section is at the end.

## Pins and citation conventions

| Item | Pin |
|---|---|
| Checkpoint | `mistralai/Voxtral-Mini-4B-Realtime-2602` @ `2769294da9567371363522aac9bbcfdd19447add`, `models/Voxtral-Mini-4B-Realtime-2602/model.safetensors` (8,859,446,848 bytes, 712 tensors incl. no `lm_head`) |
| transformers | 5.17.0 (installed in `reference/voxtral/.venv`) |
| mistral-common | 1.12.0 |
| torch | 2.14.0 (CPU BLAS = Accelerate; MPS available) |
| voxtral.c | `third_party/voxtral.c` @ `134d366c24d20c64b614a3dcc8bda2a6922d077d` |
| Tech report | arXiv 2602.11298 (HTML version, fetched 2026-09-28) |

Citation prefixes (all relative to `reference/voxtral/.venv/lib/python3.12/site-packages/`):

- `MOD` = `transformers/models/voxtral_realtime/modeling_voxtral_realtime.py`
- `FE` = `transformers/models/voxtral_realtime/feature_extraction_voxtral_realtime.py`
- `PROC` = `transformers/models/voxtral_realtime/processing_voxtral_realtime.py`
- `CFG` = `transformers/models/voxtral_realtime/configuration_voxtral_realtime.py`
- `AU` = `transformers/audio_utils.py`
- `MASK` = `transformers/masking_utils.py`, `CACHE` = `transformers/cache_utils.py`
- `MCA` = `mistral_common/tokens/tokenizers/audio.py`, `MCI` = `mistral_common/tokens/tokenizers/instruct.py`, `MCT` = `mistral_common/tokens/tokenizers/tekken.py`
- `VC` = `third_party/voxtral.c/MODEL.md`
- HF docs/tests at tag v5.17.0 were fetched into `third_party/transformers_v5.17.0/` (`voxtral_realtime.md`, `test_modeling_voxtral_realtime.py`).

## 1. Pipeline and rates

```
16 kHz PCM ─► [left pad 40960 zeros] ─► log-mel 128 × 100 Hz ─► causal conv stem (÷2) ─► 1280-d @ 50 Hz
  ─► 32 causal RoPE transformer layers (sliding window 750) ─► RMSNorm ─► stack 4 frames (5120-d) @ 12.5 Hz
  ─► adapter MLP 5120→3072→3072 ─► audio_embed[p]
decoder input[p] = tok_embed(id[p]) + audio_embed[p] ─► 26 Mistral layers (GQA 32/8, ada-RMSNorm on FFN) ─► RMSNorm ─► tied head ─► argmax = id[p+1]
```

One decoder position = 8 mel frames = 4 encoder frames = 1280 samples = 80 ms (`MCA:525-538`: `raw_audio_length_per_tok = 16000 // 12.5 = 1280`, `audio_length_per_tok = 1280 / 160 = 8`; `CFG:169-171`).

## 2. Frontend (log-mel)

Source: `FE:59-108`, called from `PROC:205-209`.

1. Window: `torch.hann_window(400)`, which is the **periodic** Hann window (torch default), `w[n] = 0.5 − 0.5·cos(2πn/400)`, n = 0..399 (`FE:87`). `n_fft = win_length = 400` (processor_config.json), so there is no zero-padding of the frame.
2. STFT: `torch.stft(x, n_fft=400, hop_length=160, window=w, center=center, return_complex=True)` (`FE:88`). With `center=True` torch pads 200 samples on each side in **reflect** mode (torch default `pad_mode="reflect"`); frame t covers padded samples [160t, 160t+400), i.e. original samples [160t−200, 160t+200). Output is 201 bins × (1 + N//160) frames.
3. **Drop the last frame**: `stft[..., :-1]` (`FE:89`). The last axis of the STFT is time, so this drops the last *frame*, not a frequency bin (voxtral.c `MODEL.md:37` says "drops last bin"; that wording is wrong, its code drops the frame). Result: N//160 frames for N samples.
4. Power: `|X|²` (`FE:89`).
5. Mel: `mel = filters.T @ power`, filters = `mel_filter_bank(201, 128, 0 Hz, 8000 Hz, 16000, norm="slaney", mel_scale="slaney")` (`FE:75-83`, `AU:736-822`): Slaney mel scale (linear below 1 kHz: `3f/200`; log above: `15 + ln(f/1000)·27/ln 6.4`, `AU:546-580`), 130 edge points linearly spaced in mel, triangular filters over the 201 bin centre frequencies `linspace(0, 8000, 201)` (`AU:639-655`), area-normalised by `2/(f[i+2]−f[i])` (`AU:815-817`). Computed in float64 numpy, cast to float32 at use (`FE:91`). The exact float32 matrix is saved as `data/golden/voxtral/offline/mel_filters.npy` [201,128] and the window as `hann_window.npy`.
6. `log10(max(mel, 1e-10))` (`FE:94`).
7. Fixed floor: `max(x, 1.5 − 8.0) = max(x, −6.5)`; `global_log_mel_max = 1.5` comes from processor_config.json, so **there is no per-utterance max** (`FE:95-104`). The `else: log_spec.max()` branch is dead for this checkpoint.
8. Scale: `(x + 4) / 4` (`FE:105`). Output float32 `[128, T]`.

No pre-emphasis, no dither, no mean/variance normalisation (`do_normalize` is ignored).

**Streaming frontend.** The first chunk is computed with `center=True`; every later chunk with `center=False` (`PROC:205-209`). Later chunks are 1680 samples = 400 + 8·160, starting at `mel_frame·160 − 200` (docs recipe, `third_party/transformers_v5.17.0/voxtral_realtime.md:119-140`), so each yields 9 frames, of which the last is dropped: 8 new frames. Because the left edge of the stream is 40960 zeros (reflect-padding zeros gives zeros) and every later frame is computed from real samples, **streaming mel frames equal offline mel frames**; only the final offline frame (which reaches 40 samples past the padded end, reflect-padded) has no streaming counterpart. For an incremental implementation: frame t needs samples [160t−200, 160t+200), so frame t can be computed once sample 160t+200 has arrived.

## 3. Silence padding and the prompt

Source: tekken.json `audio` block: `transcription_delay_ms = 480`, `streaming_n_left_pad_tokens = 32`, `frame_rate = 12.5`, `transcription_format = "streaming"`.

- **n_delay** = `num_audio_tokens(delay_len(480))` = `ceil((int(0.480·16000) / 160) / 8)` = `ceil(48/8)` = **6** (`MCA:477-507`, `MCA:448-463`). Other delays: n_delay = delay_ms / 80 for multiples of 80 ms (the config asserts the delay is a multiple of 80 ms, `MCA:431-436`).
- **Left pad:** `32 × 1280 = 40960` zero samples = 2.56 s (`MCA:566-584`, `MCA:713-732`). Offline and streaming both prepend it (`MCA:670-673`; `PROC:176-178`).
- **Right pad (offline only):** `(1280 − N mod 1280) mod 1280` samples to reach a multiple of 1280, plus `n_right_pad_tokens × 1280` with `n_right_pad_tokens = (n_delay + 1) + OFFLINE_STREAMING_BUFFER_TOKENS = 6 + 1 + 10 = 17` (`MCA:313`, `MCA:540-563`, `MCA:713-722`), i.e. 21760 samples = 1.36 s. The padded length is therefore a multiple of 1280, and the number of decoder positions is `P = padded_len / 1280 = 32 + ceil(N/1280) + 17`.
- **Prompt** (identical offline and streaming): `[BOS] + [STREAMING_PAD] × (n_left_pad + n_delay)` = `[1] + [32]×38`, length **39** (`MCI:89-95` `start()` = `[bos_id]`; `MCA:753-766`; `MCI:1092-1105`). No `[BEGIN_AUDIO]`/`[AUDIO]` tokens, no language tag, no `[TRANSCRIBE]`.

Measured (say1 test clip, N = 82856): padded 145920 = 40960 + 82856 + 344 + 21760; mel 912 frames; P = 114.

## 4. Audio encoder

Config: `config.json` `audio_config`; classes `MOD:212-609`.

### 4.1 Causal conv stem (`MOD:212-244`, `MOD:415-430`)

- conv1: Conv1d(128 → 1280, k=3, stride 1, bias), then exact GELU (erf form, `nn.functional.gelu` default).
- conv2: Conv1d(1280 → 1280, k=3, stride 2, bias), then exact GELU.
- Causal padding: `left_pad = (k−1)·dilation + 1 − stride` (`MOD:229-232`) → **conv1: 2 zeros on the left, conv2: 1 zero on the left**, no right padding (`MOD:242`). Output length for T mel frames (T even): conv1 T, conv2 T/2. Output frame j of conv2 reads conv1 frames 2j−1, 2j, 2j+1 (frame −1 = 0).
- Streaming cache (`MOD:55-112`): per conv, the last `left_pad` *input* frames of the previous chunk (conv1: last 2 mel frames; conv2: last 1 conv1-output frame, post-GELU). Initial cache is zeros, which equals the offline zero padding. Each streaming step feeds 8 mel frames → 4 encoder frames.
- Output transposed to `[T/2, 1280]` (`MOD:429`).

### 4.2 Encoder layer (`MOD:433-479`, attention `MOD:338-396`, MLP `MOD:399-412`)

Pre-norm; 32 layers; dim 1280; 32 heads × head_dim 64 (so q/k/v are **2048-wide**, larger than the 1280 model dim); MHA (no GQA).

```
h = x + o_proj(Attn(RoPE(q_proj(n)), RoPE(k_proj(n)), v_proj(n)))     n = RMSNorm(x; self_attn_layer_norm)
y = h + down_proj(silu(gate_proj(m)) * up_proj(m))                     m = RMSNorm(h; final_layer_norm)
```

- Biases: q_proj **yes**, k_proj **no**, v_proj yes, o_proj yes (`MOD:351-355`); MLP gate/up no bias, down_proj **yes** (`MOD:405-407`). MLP hidden 5120, activation SiLU (`hidden_act: silu`; `activation_function: gelu` in the config is unused by the layer, `MOD:438`).
- RMSNorm (`MOD:247-262`): computed in fp32: `x · rsqrt(mean(x²) + 1e-5)`, cast back to the input dtype, then `weight *`. eps = 1e-5.
- Attention scale `64^-0.5` (`MOD:348`); softmax in fp32 (`MOD:330`, eager path; SDPA is the default implementation and computes the same function).
- Final `audio_tower.norm` RMSNorm after the last layer (`MOD:603`).

### 4.3 RoPE (both encoder and decoder)

- `inv_freq[i] = θ^(−2i/d)`, i = 0..d/2−1, θ = 1e6 for both towers, d = head_dim (64 encoder, 128 decoder) (`MOD:173-191`, `config.json`).
- `angle[p, i] = p · inv_freq[i]`; `cos/sin` tables are `cat(freqs, freqs)` over the last axis (`MOD:204-207`), computed in fp32 with autocast disabled, then cast to the activation dtype (`MOD:209`).
- Applied **NeoX / rotate-half style**: `x' = x·cos + rotate_half(x)·sin`, `rotate_half(x) = cat(−x[d/2:], x[:d/2])` (`MOD:268-298`). Pairs are (i, i + d/2).
- **The q/k weights in `model.safetensors` are already permuted for this layout.** Verified bit-exactly on encoder layer 0 `q_proj.weight` and `q_proj.bias` and decoder layer 0 `k_proj.weight`, by fetching the same tensors from `consolidated.safetensors` (Mistral names, interleaved layout) with HTTP range requests: `hf[h·d + j] = consolidated[h·d + 2j]` and `hf[h·d + d/2 + j] = consolidated[h·d + 2j + 1]` for j < d/2 (i.e. `w.view(H, d/2, 2, in).transpose(1, 2)`). So: with `model.safetensors`, use rotate-half RoPE and **do not permute**. With `consolidated.safetensors`, use interleaved (GPT-J) RoPE, as voxtral.c does (`VC:223-227`). v/o are not permuted. The encoder q bias is permuted together with its weight.
- Positions are absolute stream indices starting at 0, including the silence pad and the prompt: encoder position = 50 Hz frame index since stream start; decoder position = token index (prompt BOS is position 0). With the KV cache they continue from `cache.get_seq_length()`, which is the **cumulative** length even when the sliding-window cache has been trimmed (`MOD:576-579`, `MOD:804-807`, `CACHE:264-285`), so positions do not saturate at the window size.

### 4.4 Sliding window 750 (encoder) / 8192 (decoder)

Mask: query q may attend key k iff `k ≤ q` and `k > q − W` (`MASK:93-102`, `MASK:135-139`), i.e. **W keys including itself** (self + 749 past frames = 15 s for the encoder; self + 8191 past tokens for the decoder). The dynamic cache keeps the last W−1 keys between calls (`CACHE:250-256`).

## 5. Adapter (`MOD:854-867`, `MOD:924-928`)

- Frame stacking: `last_hidden_state.reshape(B, T/4, 1280·4)`, a row-major reshape, so adapter row p is the concatenation **[enc[4p], enc[4p+1], enc[4p+2], enc[4p+3]]** (time-major; channel c of frame 4p+r lands at index 1280·r + c). Stacking happens after the final encoder RMSNorm.
- `linear_1` 5120→3072 (no bias) → exact GELU → `linear_2` 3072→3072 (no bias).
- Result: `audio_embed[p]` for decoder position p, covering mel frames 8p..8p+7.

## 6. Decoder

Config: `text_config` (26 layers, dim 3072, 32 q heads, 8 kv heads, head_dim 128, SwiGLU hidden 9216, RMSNorm eps 1e-5, RoPE θ = 1e6, sliding window 8192, vocab 131072). Classes `MOD:611-835`.

### 6.1 Per-step input (`MOD:966-981`)

`input[p] = embed_tokens[id[p]] + audio_embed[p]` (elementwise add, same position; not interleaved). `id[p]` is the prompt token for p < 39 and the token emitted at position p−1 for p ≥ 39. No embedding scaling.

### 6.2 Delay conditioning `t_cond` (`MOD:838-851`, `MOD:991-998`)

```
inv_freq[i] = exp(−ln(10000) · i / 1536),  i = 0..1535         (theta = 10000, dim = 3072)
e = n_delay · inv_freq                                          (n_delay = 6.0)
t_cond = [cos(e_0..e_1535), sin(e_0..e_1535)]                   (3072, cos first)
g_l = linear2_l( gelu( linear1_l(t_cond) ) )                    (3072→32→3072, no bias, exact GELU)
```

`t_cond` and every `g_l` are constants for a given delay: precompute them once. Trap: `time_tensor` is created in the **activation dtype** and `inv_freq` is cast to it (`MOD:849-850`, `MOD:991-996`), so the official bf16 path computes `t_cond` from a bf16 `inv_freq` and a bf16 product. The fp32 golden uses fp32 throughout. (`MOD:499-504` re-initialises `inv_freq`; see the precision section for what the loaded buffer dtype actually is.)

### 6.3 Decoder layer (`MOD:700-742`)

```
h = x + o_proj(Attn_GQA(RoPE(q_proj(n)), RoPE(k_proj(n)), v_proj(n)))     n = RMSNorm(x; input_layernorm)
m = RMSNorm(h; post_attention_layernorm) * (1 + g_l)                        (ada-norm on the FFN branch only)
y = h + down_proj(silu(gate_proj(m)) * up_proj(m))
```

- No biases anywhere in the decoder (`MOD:637-640`, `MOD:690-692`).
- GQA: q head h uses kv head `h // 4` (`repeat_kv` = repeat-interleave, `MOD:301-310`).
- Scale `128^-0.5`; causal, sliding window 8192 (`config.json`).
- The attention branch is not conditioned (paper §2.3 agrees).

### 6.4 Head (`MOD:1023-1030`, `MOD:1100-1102`)

Final RMSNorm (`language_model.model.norm`), then logits = `h · embed_tokens.weightᵀ` (tied, `tie_word_embeddings: true`; `model.safetensors` has no `lm_head` tensor). No logit soft-capping, no bias. During `generate()` only the last position's logits are computed (`logits_to_keep = 1`, `transformers/generation/utils.py:2746-2750`).

## 7. Decode loop

### 7.1 Offline (the golden path)

`VoxtralRealtimeProcessor(audio)` → `model.generate(...)`:

1. Pad (section 3) and compute the mel over the whole padded waveform (`center=True`).
2. The conv stem runs once over the whole mel (`MOD:1156-1160`).
3. `max_length = ceil(T_mel / 8) = P` total tokens including the 39-token prompt (`MOD:1266-1277`, clamped `MOD:1312-1317`); greedy argmax (`generation/utils.py:3073`).
4. Prefill: positions 0..38 in one forward. The encoder transformer is run on conv frames 0..155 with a fresh encoder KV cache (`MOD:1139-1146`), the adapter produces `audio_embed[0..38]`, and the decoder consumes `tok_embed(prompt) + audio_embed[0..38]`. The logits at position 38 give the first emitted token.
5. Each later step p = 39, 40, …: the encoder transformer runs on the 4 new conv frames 4p..4p+3 with its cache, the adapter gives `audio_embed[p]`, the decoder consumes `tok_embed(id[p]) + audio_embed[p]`.
6. Stops when EOS (id 2) is emitted or when the sequence reaches P tokens (then P − 39 tokens have been generated; the audio of the last position P−1 is never consumed).

So the encoder is causal and incremental even offline; an implementation that encodes the whole utterance in one batched pass computes the same function.

### 7.2 Streaming (reference recipe, `stream_ref.py`)

Same prompt and left pad. The first chunk is `audio[:9000]` (`num_samples_first_audio_chunk = (8·7 − 1)·160 + 200 = 9000`, `PROC:109-124`) after the processor prepends the 40960 zeros: 312 mel frames = 39 positions, the whole prefill. Then one 1680-sample chunk (`PROC:126-128`) per decoder step, with the conv padding cache and the encoder KV cache carried over (`MOD:1178-1202`). Generation stops when the chunk generator is exhausted (`MOD:1172-1176`, `MOD:1281-1285`), not at a length limit. The official test right-pads the audio with 17 × 1280 zeros first (`test_modeling_voxtral_realtime.py`, `test_modeling_integration_streaming`); the docs example computes that padded array but then slices the unpadded one (a bug in `voxtral_realtime.md:109-129`). `stream_ref.py` follows the test.

### 7.3 EOS and end of stream

- EOS = `</s>` = 2 (`generation_config.json`). In offline mode the 17 right-pad tokens give the model room to finish the last word (delay 6 + BOS 1 + 10 buffer, `MCA:308-313`). See "Measured behaviour" for how often EOS is actually reached before the length cap.
- `generation_config.pad_token_id = 11` is irrelevant for batch size 1.
- A long utterance can be truncated by the length cap (the upstream long-form test output ends mid-sentence, "…and the hope of the", `test_modeling_voxtral_realtime.py`).

## 8. Tokens and text assembly

- Tekken v7: 1000 special ids (0..999), then `vocab[id − 1000]` for ids 1000..131071 (only the first 130072 of the 150000 BPE entries are reachable, `default_vocab_size = 131072`) (`MCT:155-222`, `VC:123-131`). Ids 1000..1255 are the 256 raw bytes (`MCT:520-530`).
- Special ids used here: BOS `<s>` = 1, EOS `</s>` = 2, `[STREAMING_PAD]` = 32, `[STREAMING_WORD]` = 33 (tekken.json `special_tokens`; names in `mistral_common/tokens/tokenizers/base.py:105-106`).
- **`[STREAMING_PAD]` ([P])** is the non-emitting "wait" token: the model emits it for every 80 ms frame in which no text is due (paper §2.3, §3.1).
- **`[STREAMING_WORD]` ([W])** is the word-boundary token: "Once a word has been fully observed and the specified target delay has elapsed, a [W] token is emitted to mark the onset of text generation, followed by the subword tokens corresponding to the word itself. When consecutive words share the same emission frame, no additional [W] token is inserted; the subword tokens of the next word follow directly" (paper §3.1). It carries no text. The spaces between words are inside the subword tokens (Tekken pieces carry their leading space), not implied by [W].
- **Text assembly** (`processor.batch_decode(ids, skip_special_tokens=True)` → `MistralCommonBackend._decode`, `transformers/tokenization_mistral_common.py:433-461` → `Tekkenizer.decode`, `MCT:492-518`, `MCT:582-610`): drop every id < 1000 (BOS, EOS, PAD, WORD, …), and decode each maximal run of consecutive non-special ids as one byte string (UTF-8, invalid sequences replaced with U+FFFD by tiktoken), then concatenate the runs. `clean_up_tokenization_spaces` is False for this tokenizer (`tokenization_mistral_common.py:230`); a leading `lang:xx` would be stripped (`tokenization_mistral_common.py:166-179`). The transcript therefore usually starts with a space (the first word's leading space). Trap: a multi-byte UTF-8 character split by a [W]/[P] between its byte tokens decodes to replacement characters in the reference; a Swift decoder that decodes the concatenated bytes of all text tokens differs from the reference in exactly that case.
- Streaming display: each emitted text token can be shown immediately; a [W] means "a new group of words starts here".

## 9. Tensor-name map (`model.safetensors`)

All 712 tensors are **BF16**, stored in lexicographic key order; total 4,429,679,360 parameters (encoder 970,395,392; adapter 25,165,824; decoder incl. embedding 3,434,118,144). transformers renames on load: `audio_tower.*` → `model.audio_tower.*`, `multi_modal_projector.*` → `model.multi_modal_projector.*`, `language_model.model.*` → `model.language_model.*` (`transformers/conversion_mapping.py:97`, `:673-678`); `lm_head.weight` is tied to `model.language_model.embed_tokens.weight` (`MOD:1024`). All Linear weights are `[out, in]` (y = x·Wᵀ + b). Conv weights are `[out, in, k]`.

| Key (N = layer index) | Count | Shape | Notes | voxtral.c / consolidated name |
|---|---|---|---|---|
| `audio_tower.embedder.conv1.weight` / `.bias` | 1 | [1280,128,3] / [1280] | causal, left pad 2 | `…whisper_encoder.conv_layers.0.conv.*` |
| `audio_tower.embedder.conv2.weight` / `.bias` | 1 | [1280,1280,3] / [1280] | stride 2, left pad 1 | `…conv_layers.1.conv.*` |
| `audio_tower.layers.N.self_attn_layer_norm.weight` | 32 | [1280] | RMSNorm | `…transformer.layers.N.attention_norm.weight` |
| `audio_tower.layers.N.self_attn.q_proj.weight` / `.bias` | 32 | [2048,1280] / [2048] | **rows permuted for rotate-half** | `…attention.wq.*` (interleaved) |
| `audio_tower.layers.N.self_attn.k_proj.weight` | 32 | [2048,1280] | **no bias**; rows permuted | `…attention.wk.weight` |
| `audio_tower.layers.N.self_attn.v_proj.weight` / `.bias` | 32 | [2048,1280] / [2048] | | `…attention.wv.*` |
| `audio_tower.layers.N.self_attn.o_proj.weight` / `.bias` | 32 | [1280,2048] / [1280] | | `…attention.wo.*` |
| `audio_tower.layers.N.final_layer_norm.weight` | 32 | [1280] | RMSNorm | `…ffn_norm.weight` |
| `audio_tower.layers.N.mlp.gate_proj.weight` | 32 | [5120,1280] | SiLU branch, no bias | `…feed_forward.w1.weight` |
| `audio_tower.layers.N.mlp.up_proj.weight` | 32 | [5120,1280] | no bias | `…feed_forward.w3.weight` |
| `audio_tower.layers.N.mlp.down_proj.weight` / `.bias` | 32 | [1280,5120] / [1280] | **has bias** | `…feed_forward.w2.*` |
| `audio_tower.norm.weight` | 1 | [1280] | final encoder RMSNorm | `…transformer.norm.weight` |
| `multi_modal_projector.linear_1.weight` | 1 | [3072,5120] | no bias | `…audio_language_projection.0.weight` |
| `multi_modal_projector.linear_2.weight` | 1 | [3072,3072] | no bias | `…audio_language_projection.2.weight` |
| `language_model.model.embed_tokens.weight` | 1 | [131072,3072] | input embedding **and** output head | `…tok_embeddings.weight` |
| `language_model.model.layers.N.input_layernorm.weight` | 26 | [3072] | | `layers.N.attention_norm.weight` |
| `language_model.model.layers.N.self_attn.q_proj.weight` | 26 | [4096,3072] | rows permuted for rotate-half | `layers.N.attention.wq.weight` |
| `language_model.model.layers.N.self_attn.k_proj.weight` | 26 | [1024,3072] | rows permuted | `layers.N.attention.wk.weight` |
| `language_model.model.layers.N.self_attn.v_proj.weight` | 26 | [1024,3072] | | `layers.N.attention.wv.weight` |
| `language_model.model.layers.N.self_attn.o_proj.weight` | 26 | [3072,4096] | | `layers.N.attention.wo.weight` |
| `language_model.model.layers.N.post_attention_layernorm.weight` | 26 | [3072] | | `layers.N.ffn_norm.weight` |
| `language_model.model.layers.N.ada_rms_norm.linear1.weight` | 26 | [32,3072] | t_cond → 32 | `layers.N.ada_rms_norm_t_cond.0.weight` |
| `language_model.model.layers.N.ada_rms_norm.linear2.weight` | 26 | [3072,32] | 32 → 3072 | `layers.N.ada_rms_norm_t_cond.2.weight` |
| `language_model.model.layers.N.mlp.gate_proj.weight` | 26 | [9216,3072] | | `layers.N.feed_forward.w1.weight` |
| `language_model.model.layers.N.mlp.up_proj.weight` | 26 | [9216,3072] | | `layers.N.feed_forward.w3.weight` |
| `language_model.model.layers.N.mlp.down_proj.weight` | 26 | [3072,9216] | | `layers.N.feed_forward.w2.weight` |
| `language_model.model.norm.weight` | 1 | [3072] | final decoder RMSNorm | `norm.weight` |

Consolidated prefix `…` = `mm_streams_embeddings.embedding_module.` (+ `whisper_encoder.` for the encoder), from `VC:156-197`. The consolidated file has 711 tensors (header fetched); the one-tensor difference was not investigated because we do not use that file.

### Traps

1. **RoPE layout differs by file**: `model.safetensors` = rotate-half with pre-permuted q/k (+ encoder q bias); `consolidated.safetensors` = interleaved. Never permute `model.safetensors`.
2. Encoder attention width 2048 ≠ model width 1280; k_proj has no bias while q/v/o do; encoder down_proj has a bias. Decoder: no biases at all.
3. Encoder config says `activation_function: gelu` and `hidden_act: silu`: the MLP uses SiLU (SwiGLU); GELU is only in the conv stem and the adapter.
4. The adapter stacks 4 **consecutive frames** (frame-major), not channels-interleaved.
5. There is no `lm_head` tensor: the head is `embed_tokens`.
6. `t_cond` is `[cos, sin]` (cos first), θ = 10000 (not the RoPE θ), dim 3072, and the ada factor is `1 + g`, not `g`.
7. The STFT drops its last *frame*; the Hann window is periodic; `global_log_mel_max` is a constant 1.5.
8. Positions (encoder and decoder) count from the start of the padded stream and keep increasing past the sliding window.
9. The sliding window admits W keys including the query itself (750, not 751).
10. BF16 checkpoint: the "fp32 reference" is fp32 arithmetic on bf16-valued weights.
