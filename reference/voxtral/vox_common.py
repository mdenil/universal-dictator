"""Shared helpers for the Voxtral Mini 4B Realtime reference scripts.

Test infrastructure only. Loads the pinned transformers implementation
(`VoxtralRealtimeForConditionalGeneration`, transformers 5.17.0) from the
local checkpoint in `models/Voxtral-Mini-4B-Realtime-2602/`.

Precision modes (see docs/specs/voxtral.md, "Precision and memory"):
  fp32  weights stay bf16 in memory (8.86 GB) and every weight is upcast to
        fp32 at the point of use (nn.Linear, nn.Conv1d, nn.Embedding). bf16 ->
        fp32 is exact, so this computes exactly what a full-fp32 model would,
        at about half the resident memory. All activations are fp32.
  bf16  the official path: weights, activations and mel in bf16.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from types import MethodType

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models" / "Voxtral-Mini-4B-Realtime-2602"
GOLDEN_ROOT = REPO / "data" / "golden" / "voxtral"
SAFETENSORS_BYTES = 8_859_446_848

TOKEN_BOS = 1
TOKEN_EOS = 2
TOKEN_STREAMING_PAD = 32
TOKEN_STREAMING_WORD = 33


def check_weights() -> None:
    st = MODEL_DIR / "model.safetensors"
    if not st.exists() or st.stat().st_size != SAFETENSORS_BYTES:
        size = st.stat().st_size if st.exists() else None
        raise SystemExit(f"{st} missing or incomplete (size={size}, want {SAFETENSORS_BYTES})")


def set_threads(n: int | None) -> None:
    if n:
        torch.set_num_threads(n)


def load_audio(path: str | Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(-1)
    if sr != 16000:
        raise ValueError(f"{path}: sampling rate {sr}, expected 16000")
    return audio.astype(np.float32)


def read_utts(jsonl: str | Path) -> list[dict]:
    utts = []
    for line in Path(jsonl).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d.setdefault("id", Path(d["audio"]).name.split(".")[0])
        if not os.path.isabs(d["audio"]):
            d["audio"] = str(REPO / d["audio"])
        utts.append(d)
    return utts


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

_SCRATCH: dict = {}


def _up(w, dtype):
    """Copy a bf16 weight into a reused fp32 scratch buffer (avoids a fresh allocation, and the
    page faults that come with it, for every matmul). Execution is stream-ordered on CPU and MPS,
    so reusing one buffer between consecutive ops is safe."""
    if w.dtype == dtype:
        return w
    key = (w.device, dtype)
    buf = _SCRATCH.get(key)
    if buf is None or buf.numel() < w.numel():
        buf = torch.empty(max(w.numel(), 16384 * 3072), dtype=dtype, device=w.device)
        _SCRATCH[key] = buf
    out = buf[: w.numel()].view(w.shape)
    out.copy_(w)
    return out


def _linear_upcast(self, x):
    b = None if self.bias is None else self.bias.to(x.dtype)
    return F.linear(x, _up(self.weight, x.dtype), b)


def _lm_head_upcast_chunked(self, x, chunk=16384):
    # Tied 131072x3072 head: upcast in row chunks so the temporary is 200 MB, not 1.6 GB.
    w = self.weight
    outs = [F.linear(x, _up(w[i:i + chunk], x.dtype)) for i in range(0, w.shape[0], chunk)]
    return torch.cat(outs, dim=-1)


def _embedding_upcast(self, ids):
    return F.embedding(ids, self.weight, self.padding_idx).to(torch.float32)


def enable_fp32_compute(model) -> None:
    """Upcast every bf16 weight to fp32 at the point of use (exact)."""
    for _name, m in model.named_modules():
        if m is model.lm_head:
            m.forward = MethodType(_lm_head_upcast_chunked, m)
        elif isinstance(m, nn.Linear):
            m.forward = MethodType(_linear_upcast, m)
        elif isinstance(m, nn.Conv1d):
            def _conv(inp, w, b, _m=m):
                return nn.Conv1d._conv_forward(_m, inp, _up(w, inp.dtype), None if b is None else b.to(inp.dtype))
            m._conv_forward = _conv
        elif isinstance(m, nn.Embedding):
            m.forward = MethodType(_embedding_upcast, m)
    # Non-persistent fp32 buffers: make sure they are fp32 (recomputed exactly as __init__ does).
    te = model.model.time_embedding
    te.inv_freq = torch.exp(
        -math.log(te.theta) * torch.arange(te.dim // 2, dtype=torch.float32) / (te.dim // 2)
    ).to(te.inv_freq.device)
    for rot in (model.model.audio_tower.rotary_emb, model.model.language_model.rotary_emb):
        inv, _ = rot.compute_default_rope_parameters(rot.config)
        rot.inv_freq = inv.to(torch.float32).to(rot.inv_freq.device)
        rot.original_inv_freq = rot.inv_freq.clone()


def load_model(device: str = "cpu", precision: str = "fp32", attn: str = "sdpa"):
    from transformers import VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeProcessor

    check_weights()
    t0 = time.time()
    processor = VoxtralRealtimeProcessor.from_pretrained(str(MODEL_DIR))
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map=device, attn_implementation=attn
    )
    model.eval()
    if device == "cpu":
        # from_pretrained leaves CPU weights as views of the mmap'd safetensors file. Under memory
        # pressure macOS drops those clean file pages and re-reads them from disk on every step
        # (measured: the per-step weight upcast ran ~5x slower than on resident memory). Copy them
        # into anonymous memory once (same 8.86 GB, bf16).
        with torch.no_grad():
            for prm in model.parameters():
                prm.data = prm.data.clone()
    buffers_before = {
        "time_embedding.inv_freq": str(model.model.time_embedding.inv_freq.dtype),
        "audio_rotary.inv_freq": str(model.model.audio_tower.rotary_emb.inv_freq.dtype),
        "text_rotary.inv_freq": str(model.model.language_model.rotary_emb.inv_freq.dtype),
    }
    if precision == "fp32":
        enable_fp32_compute(model)
        compute_dtype = torch.float32
    elif precision == "bf16":
        compute_dtype = torch.bfloat16
    else:
        raise ValueError(precision)
    info = {
        "device": device,
        "precision": precision,
        "attn_implementation": attn,
        "torch_threads": torch.get_num_threads(),
        "load_seconds": round(time.time() - t0, 1),
        "buffer_dtypes_at_load": buffers_before,
        "torch": torch.__version__,
    }
    return model, processor, compute_dtype, info


def rss_gb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


# ----------------------------------------------------------------------------
# Offline input preparation (identical to VoxtralRealtimeProcessor.__call__ with
# is_streaming=False, but also returns the padded waveform).
# ----------------------------------------------------------------------------

def prepare_offline(processor, audio: np.ndarray):
    from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
    from mistral_common.tokens.tokenizers.audio import Audio

    audio_obj = Audio(audio_array=audio, sampling_rate=16000, format="wav")
    req = TranscriptionRequest(audio=audio_obj.to_base64(audio_obj.format), streaming=StreamingMode.OFFLINE, language=None)
    tok = processor.tokenizer.tokenizer.encode_transcription(req)
    padded = np.asarray(tok.audios[0].audio_array, dtype=np.float32)
    inputs = processor(audio, return_tensors="pt")
    # Cross-check: the processor's own tokens and padded audio are what we recorded.
    assert inputs["input_ids"][0].tolist() == list(tok.tokens), "prompt mismatch"
    feat2 = processor.feature_extractor([padded], sampling_rate=16000, return_tensors="pt", center=True)["input_features"]
    assert torch.equal(feat2, inputs["input_features"]), "mel mismatch vs padded audio"
    return inputs, padded


# ----------------------------------------------------------------------------
# Tensor capture
# ----------------------------------------------------------------------------

class Capture:
    """Forward hooks that record stage outputs across all generate() forward calls."""

    def __init__(self, model, dec_layers, n_logits_full=30, topk=10):
        self.model = model
        self.h = []
        self.embedder = []
        self.enc_layers = {i: [] for i in range(len(model.model.audio_tower.layers))}
        self.enc_norm = []
        self.adapter = []
        self.dec_inputs = []
        self.dec_layers = {i: [] for i in dec_layers}
        self.t_cond = None
        self.ada = {}
        self.logits_full = []
        self.logits_topv = []
        self.logits_topi = []
        self.n_logits_full = n_logits_full
        self.topk = topk
        m = model.model

        def f32(t):
            return t.detach().to("cpu", torch.float32)

        def rec(lst):
            return lambda mod, i, o: lst.append(f32(o[0] if isinstance(o, tuple) else o)[0])

        self.h.append(m.audio_tower.embedder.register_forward_hook(rec(self.embedder)))
        for i, layer in enumerate(m.audio_tower.layers):
            self.h.append(layer.register_forward_hook(rec(self.enc_layers[i])))
        self.h.append(m.audio_tower.norm.register_forward_hook(rec(self.enc_norm)))
        self.h.append(m.multi_modal_projector.register_forward_hook(rec(self.adapter)))

        def pre_lm(mod, args, kwargs):
            self.dec_inputs.append(f32(kwargs["inputs_embeds"])[0])

        self.h.append(m.language_model.register_forward_pre_hook(pre_lm, with_kwargs=True))
        for i in dec_layers:
            self.h.append(m.language_model.layers[i].register_forward_hook(rec(self.dec_layers[i])))

        def tc(mod, i, o):
            if self.t_cond is None:
                self.t_cond = f32(o)

        self.h.append(m.time_embedding.register_forward_hook(tc))
        for i, layer in enumerate(m.language_model.layers):
            def ada(mod, inp, o, _i=i):
                if _i not in self.ada:
                    self.ada[_i] = f32(o).reshape(-1)

            self.h.append(layer.ada_rms_norm.register_forward_hook(ada))

        def lm(mod, i, o):
            x = f32(o)[0, -1]
            if len(self.logits_full) < self.n_logits_full:
                self.logits_full.append(x)
            v, idx = torch.topk(x, self.topk)
            self.logits_topv.append(v)
            self.logits_topi.append(idx)

        self.h.append(model.lm_head.register_forward_hook(lm))

    def remove(self):
        for h in self.h:
            h.remove()


def cat(lst):
    return torch.cat(lst, dim=0).numpy() if lst else None


def save(d: Path, name: str, arr) -> None:
    np.save(d / f"{name}.npy", np.ascontiguousarray(arr))
