"""Executable check of docs/specs/{nemotron,parakeet}.md.

Re-implements the forward pass from the spec using plain torch ops on the raw state_dict (no NeMo module
forward is called), then compares every stage with the offline golden tensors written by dump_golden.py.
If the spec is wrong about an op order, a padding or a formula, this shows it.

Usage:
  scripts/heavy.sh reference/nemo/.venv/bin/python reference/nemo/check_spec.py --model nemotron
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_model, set_threads  # noqa: E402

CFG = {
    "nemotron": dict(dirs={"offline_70_1": [70, 1], "offline_70_6": [70, 6]}, normalize=False, causal=True,
                     xscale=None, conv_norm="layer", mask="chunked_limited"),
    "parakeet": dict(dirs={"offline_70_2_5": [70, 2, 5]}, normalize=True, causal=False,
                     xscale=math.sqrt(1024), conv_norm="batch", mask="chunked_limited_with_rc"),
}


def mel_spec(audio: torch.Tensor, fb: torch.Tensor, normalize: bool):
    n = audio.shape[0]
    y = torch.cat([audio[:1], audio[1:] - 0.97 * audio[:-1]])
    y = F.pad(y, (256, 256))
    n_frames = n // 160 + 1
    idx = torch.arange(n_frames)[:, None] * 160 + torch.arange(512)[None, :]
    w = torch.zeros(512)
    w[56:456] = torch.hann_window(400, periodic=False)
    X = torch.fft.rfft(y[idx] * w, dim=-1)  # [T, 257]
    mag = torch.sqrt(X.real**2 + X.imag**2)
    p = mag.pow(2.0)
    m = torch.log(p @ fb.T + 2.0**-24).T  # [128, T]
    t_valid = n // 160
    if normalize:
        v = m[:, :t_valid]
        mean = v.mean(1, keepdim=True)
        std = torch.sqrt(((v - mean) ** 2).sum(1, keepdim=True) / (t_valid - 1)) + 1e-5
        m = (m - mean) / std
    m[:, t_valid:] = 0.0
    return m, t_valid


def subsample(mel, length, sd, causal):
    x = mel.T[None, None]  # [1,1,T,128]
    L = length

    def tmask(x, L):
        x = x.clone()
        x[:, :, L:, :] = 0
        return x

    def conv_s2(x, w, b, groups, L):
        if causal:
            x = F.pad(x, (2, 1, 2, 1))
            L = L // 2 + 1
        else:
            x = F.pad(x, (1, 1, 1, 1))
            L = (L - 1) // 2 + 1
        return F.conv2d(x, w, b, stride=2, groups=groups), L

    p = "encoder.pre_encode.conv."
    x = tmask(x, L)
    x, L = conv_s2(x, sd[p + "0.weight"], sd[p + "0.bias"], 1, L)
    x = F.relu(tmask(x, L))
    for dw, pw in [("2", "3"), ("5", "6")]:
        x, L = conv_s2(tmask(x, L), sd[p + dw + ".weight"], sd[p + dw + ".bias"], 256, L)
        x = F.conv2d(tmask(x, L), sd[p + pw + ".weight"], sd[p + pw + ".bias"])
        x = F.relu(tmask(x, L))
    x = tmask(x, L)
    b, c, t, f = x.shape
    x = x.transpose(1, 2).reshape(t, c * f)  # channel-major flatten
    return F.linear(x, sd["encoder.pre_encode.out.weight"], sd["encoder.pre_encode.out.bias"])[:L], L


def rel_pos_table(T):
    pos = torch.arange(T - 1, -T, -1, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, 1024, 2, dtype=torch.float32) * -(math.log(10000.0) / 1024))
    pe = torch.zeros(2 * T - 1, 1024)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def att_mask(T, acs, style):
    i = torch.arange(T)
    if style == "chunked_limited":
        cs = acs[1] + 1
        left_chunks = acs[0] // cs
        d = (i[:, None] // cs) - (i[None, :] // cs)
        visible = (d >= 0) & (d <= left_chunks)
    else:
        left, chunk, right = acs
        c = i // chunk
        start = torch.clamp(c * chunk - left, min=0)
        end = torch.clamp(c * chunk + chunk - 1 + right, max=T - 1)
        visible = (i[None, :] >= start[:, None]) & (i[None, :] <= end[:, None])
    return ~visible  # True = masked


def lin(x, sd, name):
    return F.linear(x, sd[name + ".weight"], sd.get(name + ".bias"))


def ln(x, sd, name):
    return F.layer_norm(x, (x.shape[-1],), sd[name + ".weight"], sd[name + ".bias"], 1e-5)


def swish(x):
    return x * torch.sigmoid(x)


def layer(x, pe, mask, sd, i, cfg, seams=None):
    p = f"encoder.layers.{i}."
    h, dk = 8, 128
    T = x.shape[0]

    def ff(z, n):
        return lin(swish(lin(ln(z, sd, p + f"norm_feed_forward{n}"), sd, p + f"feed_forward{n}.linear1")), sd,
                   p + f"feed_forward{n}.linear2")

    r = x + 0.5 * ff(x, 1)
    if seams is not None:
        seams["after_ff1"] = r
    z = ln(r, sd, p + "norm_self_att")
    a = p + "self_attn."
    q = lin(z, sd, a + "linear_q").view(T, h, dk)
    k = lin(z, sd, a + "linear_k").view(T, h, dk)
    v = lin(z, sd, a + "linear_v").view(T, h, dk)
    pp = F.linear(pe, sd[a + "linear_pos.weight"]).view(2 * T - 1, h, dk)
    qu = q + sd[a + "pos_bias_u"]
    qv = q + sd[a + "pos_bias_v"]
    ac = torch.einsum("ihd,jhd->hij", qu, k)
    bd_raw = torch.einsum("ihd,rhd->hir", qv, pp)  # r indexes position (T-1-r)
    ii = torch.arange(T)[:, None]
    jj = torch.arange(T)[None, :]
    bd = bd_raw[:, ii, (T - 1) - ii + jj]  # position i - j
    s = (ac + bd) / math.sqrt(dk)
    s = s.masked_fill(mask[None], -10000.0)
    att = torch.softmax(s, -1).masked_fill(mask[None], 0.0)
    o = torch.einsum("hij,jhd->ihd", att, v).reshape(T, h * dk)
    r = r + lin(o, sd, a + "linear_out")
    if seams is not None:
        seams["after_att"] = r
    c = p + "conv."
    z = ln(r, sd, p + "norm_conv")
    z = F.conv1d(z.T[None], sd[c + "pointwise_conv1.weight"], sd.get(c + "pointwise_conv1.bias"))
    z = F.glu(z, dim=1)
    z = F.pad(z, (8, 0) if cfg["causal"] else (4, 4))
    z = F.conv1d(z, sd[c + "depthwise_conv.weight"], sd.get(c + "depthwise_conv.bias"), groups=1024)
    if cfg["conv_norm"] == "layer":
        z = F.layer_norm(z.transpose(1, 2), (1024,), sd[c + "batch_norm.weight"], sd[c + "batch_norm.bias"],
                         1e-5).transpose(1, 2)
    else:
        z = F.batch_norm(z, sd[c + "batch_norm.running_mean"], sd[c + "batch_norm.running_var"],
                         sd[c + "batch_norm.weight"], sd[c + "batch_norm.bias"], False, 0.0, 1e-5)
    z = F.conv1d(swish(z), sd[c + "pointwise_conv2.weight"], sd.get(c + "pointwise_conv2.bias"))[0].T
    r = r + z
    if seams is not None:
        seams["after_conv"] = r
    r = r + 0.5 * ff(r, 2)
    if seams is not None:
        seams["after_ff2"] = r
    return ln(r, sd, p + "norm_out")


def greedy(enc, sd, max_symbols=10, blank=1024):
    f_all = F.linear(enc, sd["joint.enc.weight"], sd["joint.enc.bias"])
    W = [(sd[f"decoder.prediction.dec_rnn.lstm.weight_ih_l{l}"], sd[f"decoder.prediction.dec_rnn.lstm.weight_hh_l{l}"],
          sd[f"decoder.prediction.dec_rnn.lstm.bias_ih_l{l}"] + sd[f"decoder.prediction.dec_rnn.lstm.bias_hh_l{l}"])
         for l in range(2)]
    hs = [torch.zeros(640) for _ in range(2)]
    cs = [torch.zeros(640) for _ in range(2)]

    def step(tok):
        x = sd["decoder.prediction.embed.weight"][tok]
        for l, (wi, wh, b) in enumerate(W):
            gates = wi @ x + wh @ hs[l] + b
            i_, f_, g_, o_ = gates.chunk(4)
            cs[l] = torch.sigmoid(f_) * cs[l] + torch.sigmoid(i_) * torch.tanh(g_)
            hs[l] = torch.sigmoid(o_) * torch.tanh(cs[l])
            x = hs[l]
        return F.linear(x, sd["joint.pred.weight"], sd["joint.pred.bias"])

    g = step(blank)
    toks, t, n = [], 0, 0
    while t < f_all.shape[0]:
        logits = F.linear(F.relu(f_all[t] + g), sd["joint.joint_net.2.weight"], sd["joint.joint_net.2.bias"])
        k = int(torch.argmax(logits))
        if k == blank:
            t, n = t + 1, 0
            continue
        toks.append(k)
        g = step(k)
        n += 1
        if n >= max_symbols:
            t, n = t + 1, 0
    return toks


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max()), float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CFG))
    ap.add_argument("--n-utts", type=int, default=3)
    args = ap.parse_args()
    set_threads(8)
    cfg = CFG[args.model]
    model = load_model(args.model)
    sd = {k: v.float() for k, v in model.state_dict().items()}
    fb = sd["preprocessor.featurizer.fb"][0]
    worst = {}
    for dname, acs in cfg["dirs"].items():
        udirs = sorted((DATA / "golden" / args.model / dname).iterdir())[: args.n_utts]
        for ud in udirs:
            g = lambda n: np.load(ud / f"{n}.npy")  # noqa: E731
            audio = torch.from_numpy(g("audio"))
            mel, t_valid = mel_spec(audio, fb, cfg["normalize"])
            res = {"mel": rel(mel, g("mel"))}
            # From here on feed the golden input of each stage (pipeline-order isolation).
            gm = torch.from_numpy(g("mel"))
            pre, T = subsample(gm, int(g("mel_len")), sd, cfg["causal"])
            res["pre_encode"] = rel(pre, g("pre_encode"))
            x = torch.from_numpy(g("pre_encode"))
            if cfg["xscale"]:
                x = x * cfg["xscale"]
            res["enc_in"] = rel(x, g("enc_in"))
            pe = rel_pos_table(T)
            res["pos_emb"] = rel(pe, g("pos_emb"))
            m = att_mask(T, acs, cfg["mask"])
            res["att_mask_mismatches"] = (int((m.numpy().astype(np.uint8) != g("att_mask")).sum()), 0.0)
            seams = {}
            out0 = layer(torch.from_numpy(g("enc_in")), pe, m, sd, 0, cfg, seams)
            for s in ["after_ff1", "after_att", "after_conv", "after_ff2"]:
                res[f"layer_00_{s}"] = rel(seams[s], g(f"layer_00_{s}"))
            res["layer_00_out"] = rel(out0, g("layer_00_out"))
            for i in range(1, 24):
                xin = torch.from_numpy(g(f"layer_{i - 1:02d}_out"))
                res[f"layer_{i:02d}_out"] = rel(layer(xin, pe, m, sd, i, cfg), g(f"layer_{i:02d}_out"))
            toks = greedy(torch.from_numpy(g("enc_out")), sd)
            res["tokens_mismatch"] = (float(toks != list(g("tokens"))), 0.0)
            for k, v in res.items():
                worst[k] = max(worst.get(k, (0, 0)), v)
            print(dname, ud.name, {k: f"{v[0]:.3g}/{v[1]:.2g}" for k, v in res.items()
                                   if k in ("mel", "pre_encode", "layer_00_out", "layer_23_out", "tokens_mismatch",
                                            "att_mask_mismatches", "pos_emb")}, flush=True)
    print("WORST (max-abs / max-abs relative to max|golden|):")
    for k, v in worst.items():
        print(f"  {k:28s} {v[0]:.3e}  {v[1]:.3e}")


if __name__ == "__main__":
    main()
