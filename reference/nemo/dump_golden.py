"""Dump offline golden tensors for Nemotron / Parakeet Unified (NeMo reference, CPU fp32, dither 0).

"Offline" = the whole utterance in one encoder call, with the streaming attention mask of the target
config (Nemotron: chunked_limited [70,1] / [70,6]; Parakeet: chunked_limited_with_rc [70,2,5]).

Layout: data/golden/<model>/offline_<cfg>/<utt_id>/*.npy   (all float32 unless noted; time-major)
  audio.npy            [N]            16 kHz float32 samples
  mel.npy              [128, T_mel]   preprocessor output (normalized for Parakeet; includes the zeroed last frame)
  mel_len.npy          int64 scalar   valid mel frames (= N // 160)
  pre_encode.npy       [T, 1024]      ConvSubsampling output (before xscale / pos_enc)
  enc_in.npy           [T, 1024]      layer-0 input (pre_encode * xscale for Parakeet; identical to pre_encode for Nemotron)
  pos_emb.npy          [2T-1, 1024]   relative positional embedding table passed to every layer (positions T-1 .. -(T-1))
  att_mask.npy         uint8 [T, T]   1 = masked (not visible)
  layer_00_ff1_out.npy / _att_out / _conv_out / _ff2_out    raw sub-module outputs of layer 0
  layer_00_after_ff1.npy   residual after x + 0.5*FF1(LN(x))
  layer_00_after_att.npy   residual after + MHSA(LN(.))
  layer_00_after_conv.npy  residual after + Conv(LN(.))
  layer_00_after_ff2.npy   residual after + 0.5*FF2(LN(.))
  layer_00_after_norm_out.npy  norm_out(.) (= layer_00_out)
  layer_00_out.npy ... layer_23_out.npy   [T, 1024]
  enc_out.npy          [T, 1024]
  tokens.npy           int64 [U]      greedy RNN-T token ids (blank = 1024 excluded)
  token_frames.npy     int64 [U]      encoder frame index at which each token was emitted
  joint_logits_first20.npy [min(U,20), 1025]  RAW joint logits (before log_softmax) at the steps that
                                     emitted the first 20 non-blank tokens
  transcript.txt
  meta.json

Usage:
  scripts/heavy.sh reference/nemo/.venv/bin/python reference/nemo/dump_golden.py --model nemotron \
      --utts data/dev/dev20.jsonl [--threads 8] [--out-root data/golden/nemotron]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, load_audio, load_model, read_manifest, set_threads, utt_id  # noqa: E402

CONFIGS = {
    "nemotron": {"offline_70_1": [70, 1], "offline_70_6": [70, 6]},
    "parakeet": {"offline_70_2_5": [70, 2, 5]},
}


def np32(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float32).numpy()


@torch.inference_mode()
def greedy_rnnt(model, enc_out_TD: torch.Tensor, max_symbols: int):
    """Plain frame-synchronous greedy RNN-T, written to mirror NeMo's label-looping semantics.

    Returns tokens, frame indices and raw joint logits for each emitting step.
    """
    dec, joint = model.decoder, model.joint
    blank = dec.blank_idx
    f_all = joint.project_encoder(enc_out_TD.unsqueeze(0))[0]  # [T, 640]
    state = dec.initialize_state(f_all.unsqueeze(0))  # zeros [2, 1, 640] x 2
    g, state = dec.predict(torch.tensor([[blank]]), state, add_sos=False, batch_size=1)
    g = joint.project_prednet(g)[0, 0]  # [640]
    tokens, frames, logits_emit = [], [], []
    T = f_all.shape[0]
    t = 0
    n_at_t = 0
    while t < T:
        logits = joint.joint_net(f_all[t] + g)  # raw logits [1025]
        k = int(torch.argmax(logits))
        if k == blank:
            t += 1
            n_at_t = 0
            continue
        tokens.append(k)
        frames.append(t)
        if len(logits_emit) < 20:
            logits_emit.append(np32(logits))
        g, state = dec.predict(torch.tensor([[k]]), state, add_sos=False, batch_size=1)
        g = joint.project_prednet(g)[0, 0]
        n_at_t += 1
        if n_at_t >= max_symbols:
            t += 1
            n_at_t = 0
    return tokens, frames, logits_emit


def dump_one(model, rec, out_dir: Path, max_symbols: int):
    enc = model.encoder
    audio = load_audio(str(ROOT / rec["audio"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    save = lambda name, arr: np.save(out_dir / f"{name}.npy", arr)  # noqa: E731

    with torch.inference_mode():
        sig = torch.from_numpy(audio).unsqueeze(0)
        sig_len = torch.tensor([audio.shape[0]], dtype=torch.int64)
        mel, mel_len = model.preprocessor(input_signal=sig, length=sig_len)

        cap = {}
        hooks = []

        def keep(name, fn=lambda o: o):
            def h(mod, inp, out):
                cap[name] = fn(out)

            return h

        hooks.append(enc.pre_encode.register_forward_hook(keep("pre_encode", lambda o: o[0])))

        def layer0_pre(mod, args, kwargs):
            cap["enc_in"] = kwargs["x"]
            cap["pos_emb"] = kwargs["pos_emb"]
            cap["att_mask"] = kwargs["att_mask"]
            cap["pad_mask"] = kwargs["pad_mask"]

        l0 = enc.layers[0]
        hooks.append(l0.register_forward_pre_hook(layer0_pre, with_kwargs=True))
        hooks.append(l0.feed_forward1.register_forward_hook(keep("layer_00_ff1_out")))
        hooks.append(l0.self_attn.register_forward_hook(keep("layer_00_att_out")))
        hooks.append(l0.conv.register_forward_hook(keep("layer_00_conv_out")))
        hooks.append(l0.feed_forward2.register_forward_hook(keep("layer_00_ff2_out")))
        for mod, nm in [
            (l0.norm_self_att, "layer_00_after_ff1"),
            (l0.norm_conv, "layer_00_after_att"),
            (l0.norm_feed_forward2, "layer_00_after_conv"),
            (l0.norm_out, "layer_00_after_ff2"),
        ]:
            hooks.append(mod.register_forward_pre_hook(lambda m, a, nm=nm: cap.__setitem__(nm, a[0])))
        hooks.append(l0.norm_out.register_forward_hook(keep("layer_00_after_norm_out")))
        for i, layer in enumerate(enc.layers):
            hooks.append(layer.register_forward_hook(keep(f"layer_{i:02d}_out")))
        try:
            enc_out, enc_len = enc(audio_signal=mel, length=mel_len)
        finally:
            for h in hooks:
                h.remove()

        T = int(enc_len[0])
        assert enc_out.shape[-1] == T, (enc_out.shape, T)
        save("audio", audio)
        save("mel", np32(mel[0]))
        save("mel_len", np.array(int(mel_len[0]), dtype=np.int64))
        for k in ["pre_encode", "enc_in"] + [k for k in cap if k.startswith("layer_")]:
            save(k, np32(cap[k][0]))
        save("pos_emb", np32(cap["pos_emb"][0]))
        save("att_mask", cap["att_mask"][0].cpu().numpy().astype(np.uint8))
        assert not bool(cap["pad_mask"].any()), "unexpected padding in batch-1 offline run"
        enc_TD = enc_out[0].transpose(0, 1).contiguous()
        save("enc_out", np32(enc_TD))

        tokens, frames, logits20 = greedy_rnnt(model, enc_TD, max_symbols)
        # Cross-check against NeMo's own decoder (label-looping greedy_batch) on the same encoder output.
        hyps = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=enc_out, encoded_lengths=enc_len, return_hypotheses=True
        )
        nemo_tokens = [int(x) for x in hyps[0].y_sequence]
        if nemo_tokens != tokens:
            raise RuntimeError(f"own greedy != NeMo greedy for {out_dir.name}: {tokens[:20]} vs {nemo_tokens[:20]}")
        save("tokens", np.array(tokens, dtype=np.int64))
        save("token_frames", np.array(frames, dtype=np.int64))
        save("joint_logits_first20", np.stack(logits20) if logits20 else np.zeros((0, 1025), np.float32))
        text = model.tokenizer.ids_to_text(tokens)
        (out_dir / "transcript.txt").write_text(text + "\n")
        meta = {
            "audio": rec["audio"],
            "ref": rec["text"],
            "duration": rec["duration"],
            "n_samples": int(audio.shape[0]),
            "mel_frames": int(mel.shape[-1]),
            "mel_len": int(mel_len[0]),
            "enc_frames": T,
            "att_context_size": list(enc.att_context_size),
            "threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "transcript": text,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["nemotron", "parakeet"])
    ap.add_argument("--utts", default=str(DATA / "dev" / "dev20.jsonl"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--configs", nargs="*", default=None, help="subset of config dir names")
    args = ap.parse_args()

    set_threads(args.threads)
    torch.manual_seed(0)
    out_root = Path(args.out_root) if args.out_root else DATA / "golden" / args.model
    model = load_model(args.model)
    max_symbols = int(model.cfg.decoding.greedy.max_symbols)
    recs = read_manifest(args.utts)
    for cfg_name, acs in CONFIGS[args.model].items():
        if args.configs and cfg_name not in args.configs:
            continue
        model.encoder.set_default_att_context_size(acs)
        assert list(model.encoder.att_context_size) == acs
        t0 = time.time()
        for rec in recs:
            uid = utt_id(rec)
            text = dump_one(model, rec, out_root / cfg_name / uid, max_symbols)
            print(f"[{cfg_name}] {uid} ({rec['duration']:.1f}s): {text}", flush=True)
        print(f"[{cfg_name}] {len(recs)} utts in {time.time() - t0:.1f}s, threads={args.threads}", flush=True)


if __name__ == "__main__":
    main()
