"""Independent re-implementation of docs/specs/voxtral.md, checked against the golden dump.

Uses only torch primitives and the raw tensors from model.safetensors (no transformers
model code), so it tests the spec text, not the reference against itself. fp32 on CPU.
Loads only the tensors it needs (<1.5 GB).

  reference/voxtral/.venv/bin/python reference/voxtral/spec_check.py --utt data/golden/voxtral/offline/1221-135766-0015
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

import vox_common as vc

ST = vc.MODEL_DIR / "model.safetensors"


class W:
    def __init__(self):
        self.f = safe_open(str(ST), framework="pt")

    def __call__(self, k):
        return self.f.get_tensor(k).to(torch.float32)


def rms(x, w, eps=1e-5):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rope(x, pos, theta=1e6):
    # x [T, H, d]; rotate-half (NeoX) on model.safetensors weights
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    ang = pos[:, None].float() * inv[None]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None]
    rot = torch.cat([-x[..., d // 2:], x[..., : d // 2]], -1)
    return x * cos + rot * sin


def attn(q, k, v, window):
    # q [T,H,d], k/v [T,Hk,d]; causal sliding window: key j visible to query i iff i-window < j <= i
    T, H, d = q.shape
    rep = H // k.shape[1]
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    s = torch.einsum("ihd,jhd->hij", q, k) / math.sqrt(d)
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None]
    s = s.masked_fill(~((j <= i) & (j > i - window)), float("-inf"))
    return torch.einsum("hij,jhd->ihd", s.softmax(-1), v).reshape(T, H * d)


def report(name, ours, ref):
    ref = torch.as_tensor(ref, dtype=torch.float32)
    ours = ours[: ref.shape[0]]  # generate() never runs the encoder on the last position's 4 frames
    d = (ours - ref).abs().max().item()
    r = d / ref.abs().max().item()
    print(f"{name:34s} max_abs={d:.3e} rel={r:.3e}")
    return {"max_abs": d, "rel_max_abs": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utt", required=True)
    args = ap.parse_args()
    g = Path(args.utt)
    L = lambda n: torch.from_numpy(np.load(g / f"{n}.npy"))  # noqa: E731
    w = W()
    out = {}
    torch.set_grad_enabled(False)

    # Frontend
    audio = L("audio").double()
    win = torch.hann_window(400, dtype=torch.float64)
    X = torch.stft(audio, 400, 160, window=win, center=True, return_complex=True)[..., :-1]
    fb = torch.from_numpy(np.load(g.parent / "mel_filters.npy")).double()
    mel = torch.clamp(fb.T @ X.abs() ** 2, min=1e-10).log10()
    mel = (torch.maximum(mel, torch.tensor(-6.5, dtype=torch.float64)) + 4) / 4
    out["mel_fp64"] = report("mel (fp64 re-impl)", mel.float(), L("mel"))

    # Conv stem
    x = L("mel")[None]
    x = F.gelu(F.conv1d(F.pad(x, (2, 0)), w("audio_tower.embedder.conv1.weight"), w("audio_tower.embedder.conv1.bias")))
    x = F.gelu(F.conv1d(F.pad(x, (1, 0)), w("audio_tower.embedder.conv2.weight"), w("audio_tower.embedder.conv2.bias"), stride=2))
    out["conv_stem"] = report("conv stem", x[0].T, L("conv_stem_out"))

    # Encoder layer 0 on the reference conv output
    h = L("conv_stem_out")
    T = h.shape[0]
    p = "audio_tower.layers.0."
    n = rms(h, w(p + "self_attn_layer_norm.weight"))
    q = (n @ w(p + "self_attn.q_proj.weight").T + w(p + "self_attn.q_proj.bias")).view(T, 32, 64)
    k = (n @ w(p + "self_attn.k_proj.weight").T).view(T, 32, 64)
    v = (n @ w(p + "self_attn.v_proj.weight").T + w(p + "self_attn.v_proj.bias")).view(T, 32, 64)
    pos = torch.arange(T)
    a = attn(rope(q, pos), rope(k, pos), v, 750)
    h = h + a @ w(p + "self_attn.o_proj.weight").T + w(p + "self_attn.o_proj.bias")
    m = rms(h, w(p + "final_layer_norm.weight"))
    h = h + (F.silu(m @ w(p + "mlp.gate_proj.weight").T) * (m @ w(p + "mlp.up_proj.weight").T)) @ w(p + "mlp.down_proj.weight").T + w(p + "mlp.down_proj.bias")
    out["enc_layer_00"] = report("encoder layer 0", h, L("enc_layer_00_out"))

    # Final encoder norm + adapter (frame stacking) on reference tensors
    e = rms(L("enc_layer_31_out"), w("audio_tower.norm.weight"))
    out["enc_norm"] = report("encoder final norm", e, L("enc_norm_out"))
    s = L("enc_norm_out").reshape(-1, 4 * 1280)
    ad = F.gelu(s @ w("multi_modal_projector.linear_1.weight").T) @ w("multi_modal_projector.linear_2.weight").T
    out["adapter"] = report("adapter (4-frame stack)", ad, L("adapter_out"))

    # Decoder input, t_cond, ada scales
    ids = L("dec_input_ids")
    emb = w("language_model.model.embed_tokens.weight")
    di = emb[ids] + L("adapter_out")[: len(ids)]
    out["dec_input"] = report("decoder input embeds", di, L("dec_input_embeds"))
    inv = torch.exp(-math.log(10000.0) * torch.arange(1536, dtype=torch.float32) / 1536)
    tc = torch.cat([(6.0 * inv).cos(), (6.0 * inv).sin()])
    out["t_cond"] = report("t_cond", tc, L("t_cond"))
    g_all = torch.stack([F.gelu(tc @ w(f"language_model.model.layers.{i}.ada_rms_norm.linear1.weight").T)
                         @ w(f"language_model.model.layers.{i}.ada_rms_norm.linear2.weight").T for i in range(26)])
    out["ada_scale"] = report("ada scales (26 layers)", g_all, L("ada_scale"))

    # Decoder layer 0 on the reference input embeddings
    h = L("dec_input_embeds")
    T = h.shape[0]
    p = "language_model.model.layers.0."
    n = rms(h, w(p + "input_layernorm.weight"))
    q = (n @ w(p + "self_attn.q_proj.weight").T).view(T, 32, 128)
    k = (n @ w(p + "self_attn.k_proj.weight").T).view(T, 8, 128)
    v = (n @ w(p + "self_attn.v_proj.weight").T).view(T, 8, 128)
    pos = torch.arange(T)
    h = h + attn(rope(q, pos), rope(k, pos), v, 8192) @ w(p + "self_attn.o_proj.weight").T
    m = rms(h, w(p + "post_attention_layernorm.weight")) * (1 + g_all[0])
    h = h + (F.silu(m @ w(p + "mlp.gate_proj.weight").T) * (m @ w(p + "mlp.up_proj.weight").T)) @ w(p + "mlp.down_proj.weight").T
    out["dec_layer_00"] = report("decoder layer 0", h, L("dec_layer_00_out"))

    # Head: final norm + tied embedding on layer-25 output, decisions at positions 38..67
    hl = L("dec_layer_25_out")[38:68]
    logits = rms(hl, w("language_model.model.norm.weight")) @ emb.T
    ref = L("logits_first30")[: len(hl)]
    out["logits"] = report("logits (tied head)", logits, ref)
    out["argmax_equal"] = bool(torch.equal(logits.argmax(-1), ref.argmax(-1)))
    print("argmax equal:", out["argmax_equal"])
    (g / "spec_check.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
