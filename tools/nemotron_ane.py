"""Builds the Core ML (Neural Engine) versions of Nemotron's cache-aware streaming encoder step, for the
Swift engine's `--backend ane`. One fixed-shape model per (latency, first/steady chunk):

  160 ms: first 9 mel rows (drop 0) / steady 25 rows = 9 pre-encode cache + 16 new (drop 2)  -> T = 2
  560 ms: first 49 rows (drop 0)    / steady 65 rows = 9 + 56 (drop 2)                        -> T = 7

The masking is the engine's (FastConformer.encodeStreamStep / encodeSubsampling / encodeLayer), passed in
as float masks so one model serves full, short-final (zero-padded) and cache-filling chunks:
  m0..m3  valid rows of the mel input and of the three subsampling stages (1 valid, 0 masked)
  fmask   valid frames of this chunk (the GLU output is zeroed on invalid frames)
  kbias   additive key bias over [70 cached ; T new] keys (0 inside [70 - cacheLen, 70 + validT), else -1e4)
Outputs: enc_proj [T, 640] (encoder output through the joint's encoder projection), and the next K/V
caches [24, 70, 1024] and conv cache [24, 8, 1024] (fp16), fed back as the next step's inputs.

Usage: tools/.venv/bin/python tools/nemotron_ane.py
Writes models/nemotron-speech-streaming-en-0.6b/coreml/nemotron_{160,560}_{first,steady}.mlpackage, except
the 560 ms steady step, which is split into nemotron_560_steady_{sub,layers}.mlpackage (see convert_split).
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/nemotron-speech-streaming-en-0.6b"
L, D, H, DK, K, NL, MEL = 70, 1024, 8, 128, 9, 24, 128


def nxt(n):
    return n // 2 + 1          # causal dw_striding conv: k=3, s=2, pad (2, 1)


def sinusoid(pos, d=D):
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
    pe = torch.zeros(d)
    pe[0::2] = torch.sin(pos * div)
    pe[1::2] = torch.cos(pos * div)
    return pe


class Step(torch.nn.Module):
    def __init__(self, sd, rows, drop, nlayers=NL):
        super().__init__()
        self.R, self.drop, self.nl = rows, drop, nlayers
        self.T1, self.T2 = nxt(rows), nxt(nxt(rows))
        self.T3 = nxt(self.T2)
        self.T = self.T3 - drop
        T, Tk = self.T, L + self.T
        keep = {k: v.float() for k, v in sd.items() if k.startswith("encoder.") or k.startswith("joint.enc")}
        self.names = {}
        for i, (k, v) in enumerate(keep.items()):
            self.register_buffer(f"w{i}", v)
            self.names[k] = f"w{i}"
        # Relative positions: rel(i, j) = (Tk - T) + i - j (Transformer-XL), projected per layer.
        rel = torch.tensor([[(Tk - T) + i - j for j in range(Tk)] for i in range(T)])
        pe = torch.stack([sinusoid(float(r)) for r in range(-(T - 1), Tk)])
        idx = rel + (T - 1)
        for l in range(nlayers):
            P = pe @ self.w(f"encoder.layers.{l}.self_attn.linear_pos.weight").T
            self.register_buffer(f"pg{l}", P[idx].reshape(T, Tk, H, DK).permute(2, 0, 1, 3).contiguous())

    def w(self, name):
        return getattr(self, self.names[name])

    def forward(self, mel, m0, m1, m2, m3, fmask, kbias, kcache, vcache, convcache):
        return self.layers(self.subsample(mel, m0, m1, m2, m3), fmask, kbias, kcache, vcache, convcache)

    def subsample(self, mel, m0, m1, m2, m3):
        mt = lambda x, m: x * m[None, None, :, None]
        x = mt(mel[None, None], m0)
        x = mt(F.relu(F.conv2d(F.pad(x, (2, 1, 2, 1)), self.w("encoder.pre_encode.conv.0.weight"), self.w("encoder.pre_encode.conv.0.bias"), stride=2)), m1)
        x = F.conv2d(F.pad(x, (2, 1, 2, 1)), self.w("encoder.pre_encode.conv.2.weight"), self.w("encoder.pre_encode.conv.2.bias"), stride=2, groups=256)
        x = mt(F.relu(F.conv2d(x, self.w("encoder.pre_encode.conv.3.weight"), self.w("encoder.pre_encode.conv.3.bias"))), m2)
        x = F.conv2d(F.pad(x, (2, 1, 2, 1)), self.w("encoder.pre_encode.conv.5.weight"), self.w("encoder.pre_encode.conv.5.bias"), stride=2, groups=256)
        x = mt(F.relu(F.conv2d(x, self.w("encoder.pre_encode.conv.6.weight"), self.w("encoder.pre_encode.conv.6.bias"))), m3)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(t, c * f)
        return F.linear(x, self.w("encoder.pre_encode.out.weight"), self.w("encoder.pre_encode.out.bias"))[self.drop:]

    def layers(self, x, fmask, kbias, kcache, vcache, convcache):
        T = self.T
        kc, vc, cc = kcache.float(), vcache.float(), convcache.float()
        newk, newv, newc = [], [], []
        for l in range(self.nl):
            p = f"encoder.layers.{l}."
            ln = lambda z, n: F.layer_norm(z, (D,), self.w(p + n + ".weight"), self.w(p + n + ".bias"))
            h = ln(x, "norm_feed_forward1")
            x = x + 0.5 * F.linear(F.silu(F.linear(h, self.w(p + "feed_forward1.linear1.weight"))), self.w(p + "feed_forward1.linear2.weight"))
            h = ln(x, "norm_self_att")
            q = F.linear(h, self.w(p + "self_attn.linear_q.weight"))
            k = torch.cat([kc[l], F.linear(h, self.w(p + "self_attn.linear_k.weight"))], 0)
            v = torch.cat([vc[l], F.linear(h, self.w(p + "self_attn.linear_v.weight"))], 0)
            qh = q.reshape(T, H, DK).transpose(0, 1)
            kh = k.reshape(L + T, H, DK).transpose(0, 1)
            vh = v.reshape(L + T, H, DK).transpose(0, 1)
            u = self.w(p + "self_attn.pos_bias_u")[:, None, :]
            bv = self.w(p + "self_attn.pos_bias_v")[:, None, :]
            ac = (qh + u) @ kh.transpose(1, 2)
            bd = ((qh + bv)[:, :, None, :] * getattr(self, f"pg{l}")).sum(-1)
            att = torch.softmax((ac + bd) / math.sqrt(DK) + kbias[None, None, :], dim=-1) @ vh
            x = x + F.linear(att.transpose(0, 1).reshape(T, D), self.w(p + "self_attn.linear_out.weight"))
            h = ln(x, "norm_conv")
            g = F.glu(F.linear(h, self.w(p + "conv.pointwise_conv1.weight")[:, :, 0]), dim=-1) * fmask[:, None]
            ci = torch.cat([cc[l], g], 0)
            dw = self.w(p + "conv.depthwise_conv.weight")[:, 0, :]
            y = torch.stack([(ci[t:t + K] * dw.T).sum(0) for t in range(T)], 0)
            y = F.silu(F.layer_norm(y, (D,), self.w(p + "conv.batch_norm.weight"), self.w(p + "conv.batch_norm.bias")))
            x = x + F.linear(y, self.w(p + "conv.pointwise_conv2.weight")[:, :, 0])
            h = ln(x, "norm_feed_forward2")
            x = x + 0.5 * F.linear(F.silu(F.linear(h, self.w(p + "feed_forward2.linear1.weight"))), self.w(p + "feed_forward2.linear2.weight"))
            x = ln(x, "norm_out")
            newk.append(k[T:]); newv.append(v[T:]); newc.append(ci[T:])
        proj = F.linear(x, self.w("joint.enc.weight"), self.w("joint.enc.bias"))
        return proj, torch.stack(newk).half(), torch.stack(newv).half(), torch.stack(newc).half()


def convert(sd, lat, kind, rows, drop, path, nlayers=NL):
    import coremltools as ct
    m = Step(sd, rows, drop, nlayers).eval()
    T, Tk = m.T, L + m.T
    ex = (torch.zeros(rows, MEL), torch.ones(rows), torch.ones(m.T1), torch.ones(m.T2), torch.ones(m.T3), torch.ones(T),
          torch.zeros(Tk), torch.zeros(nlayers, L, D), torch.zeros(nlayers, L, D), torch.zeros(nlayers, K - 1, D))
    names = ["mel", "m0", "m1", "m2", "m3", "fmask", "kbias", "kcache", "vcache", "convcache"]
    with torch.no_grad():
        traced = torch.jit.trace(m, ex)
    f16 = {"kcache", "vcache", "convcache"}
    mlm = ct.convert(traced,
                     inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float16 if n in f16 else np.float32) for n, t in zip(names, ex)],
                     outputs=[ct.TensorType(name="enc_proj", dtype=np.float32)] + [ct.TensorType(name=n, dtype=np.float16) for n in ["knext", "vnext", "cnext"]],
                     convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
    mlm.save(str(path))
    print(f"wrote {path} (rows {rows}, T {T}, layers {nlayers})", flush=True)


class SubOnly(torch.nn.Module):
    def __init__(self, step):
        super().__init__(); self.s = step

    def forward(self, mel, m0, m1, m2, m3):
        return self.s.subsample(mel, m0, m1, m2, m3)


class LayersOnly(torch.nn.Module):
    def __init__(self, step):
        super().__init__(); self.s = step

    def forward(self, x, fmask, kbias, kcache, vcache, convcache):
        return self.s.layers(x, fmask, kbias, kcache, vcache, convcache)


def convert_split(sd, rows, drop, sub_path, layers_path):
    """The step as two models (subsampling, then the conformer layers): the Neural Engine compiler rejects
    the fused 560 ms steady graph (65 mel rows -> 9 subsampled frames feeding the layers), while each
    half compiles."""
    import coremltools as ct
    m = Step(sd, rows, drop).eval()
    T = m.T
    ex = (torch.zeros(rows, MEL), torch.ones(rows), torch.ones(m.T1), torch.ones(m.T2), torch.ones(m.T3))
    with torch.no_grad():
        tr = torch.jit.trace(SubOnly(m).eval(), ex)
    ct.convert(tr, inputs=[ct.TensorType(name=n, shape=t.shape) for n, t in zip(["mel", "m0", "m1", "m2", "m3"], ex)],
               outputs=[ct.TensorType(name="x", dtype=np.float32)],
               convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15).save(str(sub_path))
    ex = (torch.zeros(T, D), torch.ones(T), torch.zeros(L + T), torch.zeros(NL, L, D), torch.zeros(NL, L, D), torch.zeros(NL, K - 1, D))
    names = ["x", "fmask", "kbias", "kcache", "vcache", "convcache"]
    f16 = {"kcache", "vcache", "convcache"}
    with torch.no_grad():
        tr = torch.jit.trace(LayersOnly(m).eval(), ex)
    ct.convert(tr, inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float16 if n in f16 else np.float32) for n, t in zip(names, ex)],
               outputs=[ct.TensorType(name="enc_proj", dtype=np.float32)] + [ct.TensorType(name=n, dtype=np.float16) for n in ["knext", "vnext", "cnext"]],
               convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15).save(str(layers_path))
    print(f"wrote {sub_path} and {layers_path} (rows {rows}, T {T})", flush=True)


class Stateful(torch.nn.Module):
    """The steady step with the K/V and conv caches as Core ML state, updated in place each step (no cache
    copies through inputs/outputs). The first chunk runs through the same model with 16 leading zero mel
    rows masked at every subsampling stage (16 -> 8 -> 4 -> 2 masked rows, the 2 dropped frames), which
    reproduces the first chunk's causal zero padding. With `sub=False` it takes the subsampled frames x as
    input (the layers half of a split step)."""

    def __init__(self, step, sub=True):
        super().__init__()
        self.s, self.sub = step, sub
        self.register_buffer("kstate", torch.zeros(NL, L, D, dtype=torch.float16))
        self.register_buffer("vstate", torch.zeros(NL, L, D, dtype=torch.float16))
        self.register_buffer("cstate", torch.zeros(NL, K - 1, D, dtype=torch.float16))

    def forward(self, *a):
        if self.sub:
            mel, m0, m1, m2, m3, fmask, kbias = a
            x = self.s.subsample(mel, m0, m1, m2, m3)
        else:
            x, fmask, kbias = a
        proj, k, v, c = self.s.layers(x, fmask, kbias, self.kstate, self.vstate, self.cstate)
        self.kstate[:] = k
        self.vstate[:] = v
        self.cstate[:] = c
        return proj


def convert_stateful(sd, rows, drop, path, sub=True):
    import coremltools as ct
    m = Step(sd, rows, drop).eval()
    T = m.T
    st = Stateful(m, sub).eval()
    if sub:
        ex = (torch.zeros(rows, MEL), torch.ones(rows), torch.ones(m.T1), torch.ones(m.T2), torch.ones(m.T3), torch.ones(T), torch.zeros(L + T))
        names = ["mel", "m0", "m1", "m2", "m3", "fmask", "kbias"]
    else:
        ex = (torch.zeros(T, D), torch.ones(T), torch.zeros(L + T))
        names = ["x", "fmask", "kbias"]
    with torch.no_grad():
        tr = torch.jit.trace(st, ex)
    states = [ct.StateType(wrapped_type=ct.TensorType(shape=getattr(st, n).shape, dtype=np.float16), name=n) for n in ["kstate", "vstate", "cstate"]]
    ct.convert(tr, inputs=[ct.TensorType(name=n, shape=t.shape) for n, t in zip(names, ex)],
               outputs=[ct.TensorType(name="enc_proj", dtype=np.float32)], states=states,
               convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15).save(str(path))
    print(f"wrote {path} (stateful, rows {rows}, T {T})", flush=True)


SHAPES = {(160, "first"): (9, 0), (160, "steady"): (25, 2), (560, "first"): (49, 0), (560, "steady"): (65, 2)}
SPLIT = {(560, "steady")}      # converted as two models (see convert_split)


def convert(sd, rows, drop, path, nlayers=NL):
    import coremltools as ct
    m = Step(sd, rows, drop, nlayers).eval()
    T, Tk = m.T, L + m.T
    ex = (torch.zeros(rows, MEL), torch.ones(rows), torch.ones(m.T1), torch.ones(m.T2), torch.ones(m.T3), torch.ones(T),
          torch.zeros(Tk), torch.zeros(nlayers, L, D), torch.zeros(nlayers, L, D), torch.zeros(nlayers, K - 1, D))
    names = ["mel", "m0", "m1", "m2", "m3", "fmask", "kbias", "kcache", "vcache", "convcache"]
    with torch.no_grad():
        traced = torch.jit.trace(m, ex)
    f16 = {"kcache", "vcache", "convcache"}
    mlm = ct.convert(traced,
                     inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float16 if n in f16 else np.float32) for n, t in zip(names, ex)],
                     outputs=[ct.TensorType(name="enc_proj", dtype=np.float32)] + [ct.TensorType(name=n, dtype=np.float16) for n in ["knext", "vnext", "cnext"]],
                     convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
    mlm.save(str(path))
    print(f"wrote {path} (rows {rows}, T {T}, layers {nlayers})", flush=True)


SHAPES = {(160, "first"): (9, 0), (160, "steady"): (25, 2), (560, "first"): (49, 0), (560, "steady"): (65, 2)}
SPLIT = {(560, "steady")}      # converted as two models (see convert_split)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="lat,kind (e.g. 560,steady); default: all four")
    ap.add_argument("--layers", type=int, default=NL)
    ap.add_argument("--out", help="output .mlpackage (with --only)")
    ap.add_argument("--stateful", action="store_true", help="build the stateful steady models (nemotron_<lat>_state*.mlpackage)")
    a = ap.parse_args()
    sd = load_file(str(MODEL / "nemo_extracted/weights.safetensors"))
    out = MODEL / "coreml"
    out.mkdir(exist_ok=True)
    if a.stateful:
        for lat in [160, 560]:
            rows, drop = SHAPES[(lat, "steady")]
            if (lat, "steady") in SPLIT:
                # The subsampling half is the stateless nemotron_<lat>_steady_sub model.
                convert_stateful(sd, rows, drop, out / f"nemotron_{lat}_state_layers.mlpackage", sub=False)
            else:
                convert_stateful(sd, rows, drop, out / f"nemotron_{lat}_state.mlpackage")
        return
    todo = [(int(a.only.split(",")[0]), a.only.split(",")[1])] if a.only else list(SHAPES)
    for lat, kind in todo:
        rows, drop = SHAPES[(lat, kind)]
        if (lat, kind) in SPLIT and not a.out:
            convert_split(sd, rows, drop, out / f"nemotron_{lat}_{kind}_sub.mlpackage", out / f"nemotron_{lat}_{kind}_layers.mlpackage")
            continue
        convert(sd, rows, drop, Path(a.out) if a.out else out / f"nemotron_{lat}_{kind}.mlpackage", a.layers)


if __name__ == "__main__":
    main()
