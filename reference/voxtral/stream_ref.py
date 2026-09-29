"""Official transformers streaming path for Voxtral Mini 4B Realtime at 480 ms.

Reproduces the streaming recipe from the transformers 5.17.0 docs
(docs/source/en/model_doc/voxtral_realtime.md) and its integration test
(tests/models/voxtral_realtime/test_modeling_voxtral_realtime.py,
test_modeling_integration_streaming): the audio is right-padded with
num_right_pad_tokens * 1280 zero samples (the test does this; the doc example
computes the padded array but then forgets to use it), the first chunk is
audio[:num_samples_first_audio_chunk] with the 2.56 s silence prepended by the
processor, and each later chunk is 1680 samples (8 mel frames + 400-sample
window overlap) processed with center=False. generate() consumes one chunk per
decoder step and stops when the generator is exhausted.

Usage:
  scripts/heavy.sh reference/voxtral/.venv/bin/python reference/voxtral/stream_ref.py \
      --utts data/dev/dev20.jsonl [--compare data/golden/voxtral/offline]

Writes data/golden/voxtral/stream_480/<utt_id>/{tokens.npy, transcript.txt,
adapter_out.npy, conv_stem_out.npy, logits_top10_*.npy, emissions.json, meta.json}
and <out>/summary.json (streaming vs offline token comparison when --compare is given).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers.generation.streamers import BaseStreamer

import vox_common as vc


class Recorder(BaseStreamer):
    def __init__(self):
        self.calls = []

    def put(self, value):
        self.calls.append(value.detach().cpu().reshape(-1).tolist())

    def end(self):
        pass


def run_one(model, processor, dtype, utt, out_root: Path):
    audio = vc.load_audio(utt["audio"])
    audio_p = np.pad(audio, (0, processor.num_right_pad_tokens * processor.raw_audio_length_per_tok))

    first = processor(audio_p[: processor.num_samples_first_audio_chunk], is_streaming=True,
                      is_first_audio_chunk=True, return_tensors="pt")
    first_feats = first["input_features"].to(model.device, dtype=dtype)
    chunk_spans = []

    def gen():
        yield first_feats
        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        hop = processor.feature_extractor.hop_length
        win = processor.feature_extractor.win_length
        start = mel_frame_idx * hop - win // 2
        while (end := start + processor.num_samples_per_audio_chunk) < audio_p.shape[0]:
            x = processor(audio_p[start:end], is_streaming=True, is_first_audio_chunk=False, return_tensors="pt")
            chunk_spans.append((start, end))
            yield x["input_features"].to(model.device, dtype=dtype)
            mel_frame_idx += processor.audio_length_per_tok
            start = mel_frame_idx * hop - win // 2

    cap = vc.Capture(model, dec_layers=[], n_logits_full=0)
    rec = Recorder()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            input_ids=first["input_ids"].to(model.device),
            input_features=gen(),
            num_delay_tokens=first["num_delay_tokens"],
            streamer=rec,
            do_sample=False,
        )
    secs = time.time() - t0
    cap.remove()

    prompt = first["input_ids"][0].tolist()
    seq = out[0].tolist()
    gen_ids = seq[len(prompt):]
    text = processor.batch_decode(out, skip_special_tokens=True)[0]

    d = out_root / utt["id"]
    d.mkdir(parents=True, exist_ok=True)
    vc.save(d, "tokens", np.array(gen_ids, dtype=np.int64))
    vc.save(d, "conv_stem_out", vc.cat(cap.embedder))
    vc.save(d, "adapter_out", vc.cat(cap.adapter))
    vc.save(d, "logits_top10_values", torch.stack(cap.logits_topv).cpu().numpy())
    vc.save(d, "logits_top10_ids", torch.stack(cap.logits_topi).cpu().numpy())
    (d / "transcript.txt").write_text(text + "\n")
    # Emission log: step k is the k-th decoder step (position 38 + k); the text is what a
    # streaming client would print after that step (decode of the new token alone).
    emissions = []
    for k, tid in enumerate(gen_ids):
        piece = processor.tokenizer.decode([tid], skip_special_tokens=False)
        emissions.append({"step": k, "position": len(prompt) - 1 + k, "audio_end_s_in_stream": round((len(prompt) + k) * 0.08, 2), "id": tid, "piece": piece})
    (d / "emissions.json").write_text(json.dumps(emissions) + "\n")
    meta = {
        "utt": utt["id"],
        "duration_s": len(audio) / 16000,
        "n_chunks_after_first": len(chunk_spans),
        "first_chunk_samples": int(processor.num_samples_first_audio_chunk),
        "chunk_samples": int(processor.num_samples_per_audio_chunk),
        "n_generated": len(gen_ids),
        "ended_with_eos": bool(gen_ids and gen_ids[-1] == vc.TOKEN_EOS),
        "transcript": text,
        "generate_seconds": round(secs, 2),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta, gen_ids


def compare(stream_ids, off_dir: Path):
    f = off_dir / "tokens.npy"
    if not f.exists():
        return None
    off = np.load(f).tolist()
    n = min(len(off), len(stream_ids))
    first_diff = next((i for i in range(n) if off[i] != stream_ids[i]), None)
    return {
        "offline_len": len(off),
        "stream_len": len(stream_ids),
        "identical": off == stream_ids,
        "common_prefix_identical": first_diff is None,
        "first_diff_step": first_diff,
        "offline_transcript": (off_dir / "transcript.txt").read_text().strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utts", required=True)
    ap.add_argument("--out", default=str(vc.GOLDEN_ROOT / "stream_480"))
    ap.add_argument("--compare", default=None, help="offline golden dir to compare token sequences against")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    vc.set_threads(args.threads)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    model, processor, dtype, info = vc.load_model(args.device, args.precision)
    print("loaded", info, flush=True)
    rows = []
    for u in vc.read_utts(args.utts)[: args.limit]:
        meta, ids = run_one(model, processor, dtype, u, out_root)
        cmp = compare(ids, Path(args.compare) / u["id"]) if args.compare else None
        row = {**meta, "vs_offline": cmp}
        rows.append(row)
        print(json.dumps({"utt": u["id"], "n": meta["n_generated"], "secs": meta["generate_seconds"],
                          "vs_offline": None if cmp is None else {k: cmp[k] for k in ("identical", "first_diff_step", "offline_len", "stream_len")}}),
              "|", meta["transcript"], flush=True)
    info["max_rss_gb"] = round(vc.rss_gb(), 2)
    info["args"] = vars(args)
    (out_root / "summary.json").write_text(json.dumps({"info": info, "utts": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
