"""Dump offline golden tensors for Voxtral Mini 4B Realtime (480 ms, n_delay = 6).

Runs the unmodified transformers offline path: VoxtralRealtimeProcessor (offline
streaming request: 32 left-pad tokens of silence, right pad to a multiple of 1280
samples plus 17 tokens) and model.generate(), greedy, one token per 80 ms frame.
Forward hooks record every stage boundary.

Usage (always under the heavy-model lock):
  scripts/heavy.sh reference/voxtral/.venv/bin/python reference/voxtral/dump_golden.py \
      --utts data/dev/dev20.jsonl [--device cpu --precision fp32] [--out data/golden/voxtral/offline]

Per utterance, writes <out>/<utt_id>/:
  audio.npy            padded waveform fed to the frontend (float32, 16 kHz)
  mel.npy              [128, T_mel] log-mel (after (x+4)/4)
  prompt_ids.npy       the 39 prompt ids: BOS + [STREAMING_PAD]*38
  conv_stem_out.npy    [T_enc, 1280] after conv1+GELU, conv2+GELU (50 Hz)
  enc_layer_XX_out.npy [T_enc, 1280] residual stream after encoder layer XX
  enc_norm_out.npy     [T_enc, 1280] after the final encoder RMSNorm
  adapter_out.npy      [T_enc/4, 3072] audio embeddings (12.5 Hz)
  dec_input_embeds.npy [N_pos, 3072] decoder input = tok_embed(id_p) + adapter_out[p]
  dec_input_ids.npy    [N_pos] the token id fed at each of those positions
  dec_layer_XX_out.npy [N_pos, 3072] residual stream after decoder layer XX
  t_cond.npy           [3072] sinusoidal delay embedding for n_delay = 6
  ada_scale.npy        [26, 3072] g = ada_rms_norm(t_cond) per layer (the FFN-norm output is scaled by 1+g)
  logits_first30.npy   [30, 131072] logits of the first 30 decoding decisions (positions 38..67)
  logits_top10_values.npy / logits_top10_ids.npy  [n_steps, 10] for every decision
  tokens.npy           all generated ids (after the prompt), control tokens included
  transcript.txt       decode(tokens, skip_special_tokens=True)
  meta.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import vox_common as vc


def run_one(model, processor, dtype, utt, out_root: Path, dec_layers, n_dec_pos, enc_layers):
    audio = vc.load_audio(utt["audio"])
    inputs, padded = vc.prepare_offline(processor, audio)
    input_ids = inputs["input_ids"].to(model.device)
    attn = inputs["attention_mask"].to(model.device)
    feats = inputs["input_features"].to(model.device, dtype=dtype)
    n_delay = int(inputs["num_delay_tokens"])

    cap = vc.Capture(model, dec_layers)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            input_features=feats,
            num_delay_tokens=n_delay,
            do_sample=False,
        )
    secs = time.time() - t0
    cap.remove()

    seq = out[0].tolist()
    prompt = input_ids[0].tolist()
    assert seq[: len(prompt)] == prompt
    gen = seq[len(prompt):]
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    text_keep = processor.batch_decode(out[:, len(prompt):], skip_special_tokens=False)[0]

    d = out_root / utt["id"]
    d.mkdir(parents=True, exist_ok=True)
    vc.save(d, "audio", padded)
    vc.save(d, "mel", inputs["input_features"][0].numpy())
    vc.save(d, "prompt_ids", np.array(prompt, dtype=np.int64))
    vc.save(d, "conv_stem_out", vc.cat(cap.embedder))
    for i in enc_layers:
        vc.save(d, f"enc_layer_{i:02d}_out", vc.cat(cap.enc_layers[i]))
    vc.save(d, "enc_norm_out", vc.cat(cap.enc_norm))
    adapter = vc.cat(cap.adapter)
    vc.save(d, "adapter_out", adapter)
    dec_in = vc.cat(cap.dec_inputs)
    vc.save(d, "dec_input_embeds", dec_in[:n_dec_pos])
    vc.save(d, "dec_input_ids", np.array(seq[: min(n_dec_pos, len(seq))], dtype=np.int64))
    for i in dec_layers:
        vc.save(d, f"dec_layer_{i:02d}_out", vc.cat(cap.dec_layers[i])[:n_dec_pos])
    vc.save(d, "t_cond", cap.t_cond.numpy())
    vc.save(d, "ada_scale", torch.stack([cap.ada[i] for i in sorted(cap.ada)]).numpy())
    vc.save(d, "logits_first30", torch.stack(cap.logits_full).numpy())
    vc.save(d, "logits_top10_values", torch.stack(cap.logits_topv).numpy())
    vc.save(d, "logits_top10_ids", torch.stack(cap.logits_topi).numpy())
    vc.save(d, "tokens", np.array(gen, dtype=np.int64))
    (d / "transcript.txt").write_text(text + "\n")

    # Self-consistency checks on what was captured (the spec relies on these).
    tok_emb = model.model.language_model.embed_tokens.weight
    n_chk = min(n_dec_pos, dec_in.shape[0], len(seq))
    ids = torch.tensor(seq[:n_chk], device=tok_emb.device)
    emb = tok_emb[ids].to("cpu", torch.float32).numpy()
    if dtype == torch.bfloat16:
        recon = (torch.from_numpy(emb).bfloat16() + torch.from_numpy(adapter[:n_chk]).bfloat16()).float().numpy()
    else:
        recon = emb + adapter[:n_chk]
    input_sum_maxabs = float(np.abs(recon - dec_in[:n_chk]).max())
    steps_argmax = torch.stack(cap.logits_topi)[:, 0].numpy()
    argmax_matches_tokens = bool(np.array_equal(steps_argmax[: len(gen)], np.array(gen)))

    meta = {
        "utt": utt["id"],
        "audio_path": utt["audio"],
        "ref_text": utt.get("text"),
        "duration_s": len(audio) / 16000,
        "n_samples_raw": int(len(audio)),
        "n_samples_padded": int(len(padded)),
        "left_pad_samples": 32 * 1280,
        "n_mel_frames": int(inputs["input_features"].shape[-1]),
        "n_enc_frames": int(adapter.shape[0] * 4),
        "n_positions_total": int(inputs["input_features"].shape[-1] // 8),
        "n_prompt": len(prompt),
        "n_generated": len(gen),
        "ended_with_eos": bool(gen and gen[-1] == vc.TOKEN_EOS),
        "n_delay_tokens": n_delay,
        "transcript": text,
        "transcript_with_specials": text_keep,
        "check_dec_input_equals_tok_embed_plus_adapter_maxabs": input_sum_maxabs,
        "check_argmax_equals_tokens": argmax_matches_tokens,
        "generate_seconds": round(secs, 2),
        "seconds_per_step": round(secs / max(1, len(gen)), 3),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utts", required=True)
    ap.add_argument("--out", default=str(vc.GOLDEN_ROOT / "offline"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dec-layers", default="0,1,12,25")
    ap.add_argument("--enc-layers", default="all", help="'all' or comma list")
    ap.add_argument("--n-dec-pos", type=int, default=69, help="decoder positions to save (39 prompt + 30 decode)")
    args = ap.parse_args()

    vc.set_threads(args.threads)
    torch.manual_seed(0)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    model, processor, dtype, info = vc.load_model(args.device, args.precision)
    print("loaded", info, flush=True)
    dec_layers = [int(x) for x in args.dec_layers.split(",")]
    enc_layers = list(range(32)) if args.enc_layers == "all" else [int(x) for x in args.enc_layers.split(",")]
    utts = vc.read_utts(args.utts)[: args.limit]
    # Frontend constants exactly as the feature extractor uses them.
    fe = processor.feature_extractor
    vc.save(out_root, "mel_filters", torch.from_numpy(fe.mel_filters).to(torch.float32).numpy())  # [201, 128]
    vc.save(out_root, "hann_window", torch.hann_window(fe.n_fft).numpy())  # periodic, 400
    metas = []
    for u in utts:
        m = run_one(model, processor, dtype, u, out_root, dec_layers, args.n_dec_pos, enc_layers)
        metas.append(m)
        print(json.dumps({k: m[k] for k in ("utt", "n_generated", "ended_with_eos", "generate_seconds", "seconds_per_step",
                                            "check_argmax_equals_tokens", "check_dec_input_equals_tok_embed_plus_adapter_maxabs")}),
              "|", m["transcript"], flush=True)
    info["max_rss_gb"] = round(vc.rss_gb(), 2)
    info["args"] = vars(args)
    (out_root / "run_info.json").write_text(json.dumps({"info": info, "utts": [m["utt"] for m in metas]}, indent=1) + "\n")
    print("done", info, flush=True)


if __name__ == "__main__":
    main()
