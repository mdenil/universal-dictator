"""M5 study: Nemotron cache-aware streaming step (steady state, fixed shapes) as a traceable PyTorch
module, validated against the Swift/Metal engine's dump, converted to CoreML and timed on
CPU+GPU, CPU+Neural Engine and CPU only.

Steady state for lookahead=1 (160 ms): 25 mel rows in (9 pre-encode cache + 16 new), 4 subsampled
frames, the first 2 dropped, T=2 output frames; attention keys = 70 cached projected K/V + 2 new;
conv cache 8 rows. Same cache representation as the engine (projected K/V).

Usage:
  reference/coreml/.venv/bin/python reference/coreml/nemotron_step.py --dump scratch/stepdump
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "models/nemotron-speech-streaming-en-0.6b/nemo_extracted/weights.safetensors"
L, D, H, DK, K = 70, 1024, 8, 128, 9


def sinusoid(pos, d=D):
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
    pe = torch.zeros(d)
    pe[0::2] = torch.sin(pos * div)
    pe[1::2] = torch.cos(pos * div)
    return pe


class NemotronStep(torch.nn.Module):
    def __init__(self, sd, T=2, nlayers=24):
        super().__init__()
        self.T, self.n = T, nlayers
        self.p = {k: v.float() for k, v in sd.items() if k.startswith("encoder.")}
        for i, (k, v) in enumerate(self.p.items()):
            self.register_buffer(f"w{i}", v)
        self.names = {k: f"w{i}" for i, k in enumerate(self.p)}
        # Relative-position gather: rel(i, j) = (Tk - Tq) + i - j, Tk = L + T.
        Tk = L + T
        rel = torch.tensor([[(Tk - T) + i - j for j in range(Tk)] for i in range(T)])
        pe = torch.stack([sinusoid(float(r)) for r in range(-(T - 1), Tk)])      # rows for rel in [-(T-1), Tk-1]
        idx = rel + (T - 1)
        for l in range(nlayers):
            P = pe @ self.w(f"encoder.layers.{l}.self_attn.linear_pos.weight").T  # [nrel, D]
            self.register_buffer(f"pg{l}", P[idx].reshape(T, Tk, H, DK).permute(2, 0, 1, 3).contiguous())  # [H, T, Tk, DK]

    def w(self, name):
        return getattr(self, self.names[name])

    def forward(self, mel, kcache, vcache, convcache):
        # Subsampling: causal dw_striding x8 (pad (2,1) on time and freq), ReLU.
        x = mel[None, None]
        x = F.relu(F.conv2d(F.pad(x, (2, 1, 2, 1)), self.w("encoder.pre_encode.conv.0.weight"), self.w("encoder.pre_encode.conv.0.bias"), stride=2))
        x = F.conv2d(F.pad(x, (2, 1, 2, 1)), self.w("encoder.pre_encode.conv.2.weight"), self.w("encoder.pre_encode.conv.2.bias"), stride=2, groups=256)
        x = F.relu(F.conv2d(x, self.w("encoder.pre_encode.conv.3.weight"), self.w("encoder.pre_encode.conv.3.bias")))
        x = F.conv2d(F.pad(x, (2, 1, 2, 1)), self.w("encoder.pre_encode.conv.5.weight"), self.w("encoder.pre_encode.conv.5.bias"), stride=2, groups=256)
        x = F.relu(F.conv2d(x, self.w("encoder.pre_encode.conv.6.weight"), self.w("encoder.pre_encode.conv.6.bias")))
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(t, c * f)
        x = F.linear(x, self.w("encoder.pre_encode.out.weight"), self.w("encoder.pre_encode.out.bias"))[2:]   # drop 2
        T = self.T
        newk, newv, newc = [], [], []
        for l in range(self.n):
            p = f"encoder.layers.{l}."
            ln = lambda z, n: F.layer_norm(z, (D,), self.w(p + n + ".weight"), self.w(p + n + ".bias"))
            h = ln(x, "norm_feed_forward1")
            x = x + 0.5 * F.linear(F.silu(F.linear(h, self.w(p + "feed_forward1.linear1.weight"))), self.w(p + "feed_forward1.linear2.weight"))
            h = ln(x, "norm_self_att")
            q = F.linear(h, self.w(p + "self_attn.linear_q.weight"))
            k = torch.cat([kcache[l], F.linear(h, self.w(p + "self_attn.linear_k.weight"))], 0)
            v = torch.cat([vcache[l], F.linear(h, self.w(p + "self_attn.linear_v.weight"))], 0)
            qh = q.reshape(T, H, DK).transpose(0, 1)                       # [H, T, DK]
            kh = k.reshape(L + T, H, DK).transpose(0, 1)                   # [H, Tk, DK]
            vh = v.reshape(L + T, H, DK).transpose(0, 1)
            u = self.w(p + "self_attn.pos_bias_u")[:, None, :]
            bv = self.w(p + "self_attn.pos_bias_v")[:, None, :]
            ac = (qh + u) @ kh.transpose(1, 2)                              # [H, T, Tk]
            bd = ((qh + bv)[:, :, None, :] * getattr(self, f"pg{l}")).sum(-1)
            att = torch.softmax((ac + bd) / math.sqrt(DK), dim=-1) @ vh    # [H, T, DK]
            x = x + F.linear(att.transpose(0, 1).reshape(T, D), self.w(p + "self_attn.linear_out.weight"))
            h = ln(x, "norm_conv")
            g = F.glu(F.linear(h, self.w(p + "conv.pointwise_conv1.weight")[:, :, 0]), dim=-1)
            ci = torch.cat([convcache[l], g], 0)                           # [8 + T, D]
            dw = self.w(p + "conv.depthwise_conv.weight")[:, 0, :]         # [D, 9]
            y = torch.stack([(ci[t:t + K] * dw.T).sum(0) for t in range(T)], 0)
            y = F.silu(F.layer_norm(y, (D,), self.w(p + "conv.batch_norm.weight"), self.w(p + "conv.batch_norm.bias")))
            x = x + F.linear(y, self.w(p + "conv.pointwise_conv2.weight")[:, :, 0])
            h = ln(x, "norm_feed_forward2")
            x = x + 0.5 * F.linear(F.silu(F.linear(h, self.w(p + "feed_forward2.linear1.weight"))), self.w(p + "feed_forward2.linear2.weight"))
            x = ln(x, "norm_out")
            newk.append(k[T:]); newv.append(v[T:]); newc.append(ci[T:])
        return x, torch.stack(newk), torch.stack(newv), torch.stack(newc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ROOT / "scratch/stepdump"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/m5"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = {n: torch.from_numpy(np.load(f"{a.dump}/{n}.npy")) for n in ["mel", "kcache", "vcache", "convcache", "enc"]}
    m = NemotronStep(load_file(str(W))).eval()
    with torch.no_grad():
        enc, *_ = m(d["mel"], d["kcache"], d["vcache"], d["convcache"])
    err = (enc - d["enc"]).abs().max().item()
    print(f"torch step vs Metal engine: max abs diff {err:.3e} (ref max {d['enc'].abs().max():.3f})")
    results = {"torch_vs_engine_maxabs": err}

    import coremltools as ct
    inputs = (d["mel"], d["kcache"], d["vcache"], d["convcache"])
    traced = torch.jit.trace(m, inputs)
    mlm = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=t.shape) for n, t in zip(["mel", "kcache", "vcache", "convcache"], inputs)],
                     outputs=[ct.TensorType(name=n) for n in ["enc", "knext", "vnext", "cnext"]],
                     convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
    path = out / "nemotron_step160.mlpackage"
    mlm.save(str(path))
    feed = {n: t.numpy() for n, t in zip(["mel", "kcache", "vcache", "convcache"], inputs)}
    for units in ["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"]:
        model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, units))
        r = model.predict(feed)
        e = np.abs(r["enc"] - d["enc"].numpy()).max()
        times = []
        for _ in range(30):
            t0 = time.perf_counter(); model.predict(feed); times.append(time.perf_counter() - t0)
        times.sort()
        print(f"CoreML {units:12s}: p50 {times[15] * 1000:7.2f} ms  min {times[0] * 1000:7.2f} ms  maxabs vs engine {e:.3e}")
        results[units] = {"p50_ms": times[15] * 1000, "min_ms": times[0] * 1000, "maxabs_vs_engine": float(e)}
    (out / "nemotron_step160.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
