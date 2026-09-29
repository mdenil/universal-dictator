"""M5 study: Voxtral encoder streaming step (8 mel frames -> 4 encoder frames -> 1 adapter embedding)
as a stateful CoreML model, timed on the Neural Engine / GPU / CPU. The sliding K/V window lives in
CoreML state (752 slots, a multiple of 4 so a step never wraps; the mask admits the latest 750
positions, as the reference's causal sliding window does).

Validated against the Metal engine by running both on the same audio and comparing the adapter
output of a steady-state step (dumped with DICTATOR tap from `dictate-cli`) is left for integration;
here we check numerics against a pure-torch run of the same module and time CoreML.

Usage: reference/coreml/.venv/bin/python reference/coreml/voxtral_encoder_step.py [--layers 32]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
WFILE = ROOT / "models/Voxtral-Mini-4B-Realtime-2602/model.safetensors"
D, H, HD, FF, SLOTS, WIN = 1280, 32, 64, 5120, 752, 750


def load(names):
    out = {}
    with safe_open(str(WFILE), framework="pt") as f:
        for n in names:
            out[n] = f.get_tensor(n).to(torch.float32)
    return out


class EncoderStep(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.n = layers
        names = ["audio_tower.embedder.conv1.weight", "audio_tower.embedder.conv1.bias",
                 "audio_tower.embedder.conv2.weight", "audio_tower.embedder.conv2.bias", "audio_tower.norm.weight",
                 "multi_modal_projector.linear_1.weight", "multi_modal_projector.linear_2.weight"]
        for i in range(layers):
            p = f"audio_tower.layers.{i}."
            names += [p + s for s in ["self_attn_layer_norm.weight", "self_attn.q_proj.weight", "self_attn.q_proj.bias",
                                      "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.v_proj.bias",
                                      "self_attn.o_proj.weight", "self_attn.o_proj.bias", "final_layer_norm.weight",
                                      "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "mlp.down_proj.bias"]]
        w = load(names)
        self.keys = {}
        for i, (k, v) in enumerate(w.items()):
            self.register_buffer(f"p{i}", v)
            self.keys[k] = f"p{i}"
        inv = 1.0 / (1e6 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
        self.register_buffer("inv_freq", inv)
        # CoreML state: K/V windows and conv caches.
        self.register_buffer("kcache", torch.zeros(layers, SLOTS, H * HD, dtype=torch.float16))
        self.register_buffer("vcache", torch.zeros(layers, SLOTS, H * HD, dtype=torch.float16))
        self.register_buffer("melcache", torch.zeros(2, 128, dtype=torch.float16))
        self.register_buffer("c1cache", torch.zeros(1, D, dtype=torch.float16))

    def w(self, n):
        return getattr(self, self.keys[n])

    def rms(self, x, n):
        x32 = x.float()
        return x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + 1e-5) * self.w(n)

    def rope(self, x, pos):  # x [4, H*HD], pos [4]
        ang = pos[:, None].float() * self.inv_freq[None, :]
        c, s = torch.cos(ang), torch.sin(ang)
        xh = x.reshape(4, H, HD)
        a, b = xh[..., : HD // 2], xh[..., HD // 2:]
        return torch.cat([a * c[:, None] - b * s[:, None], b * c[:, None] + a * s[:, None]], -1).reshape(4, H * HD)

    def forward(self, mel, pos0):
        # mel [8,128] fp16; pos0 [1] int32 = encoder position of the first new frame (multiple of 4).
        melx = torch.cat([self.melcache.float(), mel.float()], 0)       # [10,128]
        self.melcache[:] = melx[8:].to(torch.float16)
        c1 = F.gelu(F.conv1d(melx.T[None], self.w("audio_tower.embedder.conv1.weight"), self.w("audio_tower.embedder.conv1.bias")))[0].T  # [8,1280]
        c1x = torch.cat([self.c1cache.float(), c1], 0)                  # [9,1280]
        self.c1cache[:] = c1x[8:].to(torch.float16)
        x = F.gelu(F.conv1d(c1x.T[None], self.w("audio_tower.embedder.conv2.weight"), self.w("audio_tower.embedder.conv2.bias"), stride=2))[0].T  # [4,1280]
        pos = pos0.float() + torch.arange(4, dtype=torch.float32)
        slot = torch.remainder(pos0, SLOTS)
        # Slot s holds the latest position <= pos0+3 congruent to s (slots fill in order), so the
        # causal sliding-window mask follows from pos0 alone (no position state needed).
        last = pos0.float() + 3.0
        s_idx = torch.arange(SLOTS, dtype=torch.float32)
        sp = last - torch.remainder(last - s_idx, float(SLOTS))
        ok = (sp[None, :] >= 0) & (sp[None, :] <= pos[:, None]) & (sp[None, :] > pos[:, None] - WIN)
        bias = torch.where(ok, 0.0, -1e4)                               # [4, SLOTS]
        for i in range(self.n):
            p = f"audio_tower.layers.{i}."
            h = self.rms(x, p + "self_attn_layer_norm.weight")
            q = self.rope(F.linear(h, self.w(p + "self_attn.q_proj.weight"), self.w(p + "self_attn.q_proj.bias")), pos)
            k = self.rope(F.linear(h, self.w(p + "self_attn.k_proj.weight")), pos)
            v = F.linear(h, self.w(p + "self_attn.v_proj.weight"), self.w(p + "self_attn.v_proj.bias"))
            self.kcache[i, slot:slot + 4] = k.to(torch.float16)
            self.vcache[i, slot:slot + 4] = v.to(torch.float16)
            K = self.kcache[i].float().reshape(SLOTS, H, HD).transpose(0, 1)       # [H, S, HD]
            V = self.vcache[i].float().reshape(SLOTS, H, HD).transpose(0, 1)
            Q = q.reshape(4, H, HD).transpose(0, 1)                          # [H, 4, HD]
            att = torch.softmax((Q @ K.transpose(1, 2)) / 8.0 + bias, -1) @ V
            x = x + F.linear(att.transpose(0, 1).reshape(4, H * HD), self.w(p + "self_attn.o_proj.weight"), self.w(p + "self_attn.o_proj.bias"))
            h = self.rms(x, p + "final_layer_norm.weight")
            x = x + F.linear(F.silu(F.linear(h, self.w(p + "mlp.gate_proj.weight"))) * F.linear(h, self.w(p + "mlp.up_proj.weight")),
                             self.w(p + "mlp.down_proj.weight"), self.w(p + "mlp.down_proj.bias"))
        x = self.rms(x, "audio_tower.norm.weight").reshape(1, 4 * D)
        return F.linear(F.gelu(F.linear(x, self.w("multi_modal_projector.linear_1.weight"))), self.w("multi_modal_projector.linear_2.weight"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "artifacts/m5"))
    a = ap.parse_args()
    import coremltools as ct
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = EncoderStep(a.layers).eval()
    mel = (torch.randn(8, 128) * 0.3 - 0.5).to(torch.float16)
    pos0 = torch.tensor([400], dtype=torch.int32)
    traced = torch.jit.trace(m, (mel, pos0))
    states = [ct.StateType(wrapped_type=ct.TensorType(shape=getattr(m, n).shape, dtype=np.float16), name=n)
              for n in ["kcache", "vcache", "melcache", "c1cache"]]
    mlm = ct.convert(traced, inputs=[ct.TensorType(name="mel", shape=(8, 128), dtype=np.float16),
                                     ct.TensorType(name="pos0", shape=(1,), dtype=np.int32)],
                     outputs=[ct.TensorType(name="emb", dtype=np.float16)], states=states,
                     convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                     minimum_deployment_target=ct.target.macOS15)
    path = out / f"voxtral_encoder_step_{a.layers}l.mlpackage"
    mlm.save(str(path))
    res = {}
    for units in ["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"]:
        model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, units))
        st = model.make_state()
        feed = {"mel": mel.numpy(), "pos0": np.array([0], dtype=np.int32)}
        times = []
        for step in range(40):
            feed["pos0"] = np.array([4 * step], dtype=np.int32)
            t0 = time.perf_counter(); model.predict(feed, state=st); times.append(time.perf_counter() - t0)
        times = sorted(times[5:])
        print(f"CoreML {units:12s} ({a.layers} layers): p50 {times[len(times) // 2] * 1000:7.2f} ms  min {times[0] * 1000:7.2f} ms")
        res[units] = {"p50_ms": times[len(times) // 2] * 1000, "min_ms": times[0] * 1000}
    (out / f"voxtral_encoder_step_{a.layers}l.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
