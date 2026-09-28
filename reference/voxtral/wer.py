"""Reference WER for Voxtral Mini 4B Realtime (offline transformers path, 480 ms).

Transcribes with the unmodified processor + model.generate() (greedy, batch size 1)
and scores with jiwer after the Whisper English normalizer
(transformers.models.whisper.english_normalizer.EnglishTextNormalizer with
OpenAI's english spelling map, reference/voxtral/whisper_english_spelling.json).

Input: either --utts <jsonl with audio,text> or --librispeech <split dir> --n N
(deterministic subset: all utterances sorted by id, evenly strided to N).
Output: --out <jsonl> with one row per utterance (resumable; existing rows are kept).

  scripts/heavy.sh reference/voxtral/.venv/bin/python reference/voxtral/wer.py \
      --librispeech data/librispeech/LibriSpeech/test-clean --n 200 --device mps --precision bf16 \
      --out data/golden/voxtral/wer/test-clean-200_mps_bf16.jsonl

Score only (no model): --score-only, or --from-golden <dir> to score transcript.txt files.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jiwer
import torch

import vox_common as vc

HERE = Path(__file__).resolve().parent


def normalizer():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer(json.loads((HERE / "whisper_english_spelling.json").read_text()))


def librispeech_utts(split_dir: Path, n: int | None):
    rows = []
    for tf in sorted(split_dir.glob("*/*/*.trans.txt")):
        for line in tf.read_text().splitlines():
            uid, text = line.split(" ", 1)
            rows.append({"id": uid, "audio": str(tf.parent / f"{uid}.flac"), "text": text})
    rows.sort(key=lambda r: r["id"])
    if n and n < len(rows):
        idx = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
        rows = [rows[i] for i in idx]
    return rows


def score(rows, norm):
    refs = [norm(r["text"]) for r in rows]
    hyps = [norm(r["hyp"]) for r in rows]
    keep = [i for i, r in enumerate(refs) if r.strip()]
    refs = [refs[i] for i in keep]
    hyps = [hyps[i] for i in keep]
    o = jiwer.process_words(refs, hyps)
    n_words = sum(len(r.split()) for r in refs)
    return {"n_utts": len(refs), "n_ref_words": n_words, "wer": o.wer, "substitutions": o.substitutions,
            "deletions": o.deletions, "insertions": o.insertions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utts")
    ap.add_argument("--librispeech")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--from-golden", default=None)
    args = ap.parse_args()
    norm = normalizer()

    if args.utts:
        utts = vc.read_utts(args.utts)
    else:
        utts = librispeech_utts(Path(args.librispeech), args.n)

    if args.from_golden:
        rows = []
        for u in utts:
            t = Path(args.from_golden) / u["id"] / "transcript.txt"
            if t.exists():
                rows.append({**u, "hyp": t.read_text().strip()})
        print(json.dumps({"from_golden": args.from_golden, **score(rows, norm)}))
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    todo = [u for u in utts if u["id"] not in done]
    if todo and not args.score_only:
        vc.set_threads(args.threads)
        model, processor, dtype, info = vc.load_model(args.device, args.precision)
        print("loaded", info, flush=True)
        t_all = time.time()
        audio_s = 0.0
        with out.open("a") as fh:
            for i, u in enumerate(todo):
                audio = vc.load_audio(u["audio"])
                inputs = processor(audio, return_tensors="pt")
                t0 = time.time()
                with torch.inference_mode():
                    seq = model.generate(
                        input_ids=inputs["input_ids"].to(model.device),
                        attention_mask=inputs["attention_mask"].to(model.device),
                        input_features=inputs["input_features"].to(model.device, dtype=dtype),
                        num_delay_tokens=int(inputs["num_delay_tokens"]),
                        do_sample=False,
                    )
                dt = time.time() - t0
                gen = seq[0, inputs["input_ids"].shape[1]:].tolist()
                hyp = processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
                r = {"id": u["id"], "audio": u["audio"], "text": u["text"], "hyp": hyp, "duration_s": len(audio) / 16000,
                     "seconds": round(dt, 2), "n_generated": len(gen), "ended_with_eos": bool(gen and gen[-1] == vc.TOKEN_EOS),
                     "device": args.device, "precision": args.precision}
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                done[u["id"]] = r
                audio_s += r["duration_s"]
                if i % 10 == 0:
                    print(f"[{i + 1}/{len(todo)}] {dt:.1f}s rtf={dt / r['duration_s']:.2f} | {hyp}", flush=True)
        print(json.dumps({"wall_seconds": round(time.time() - t_all, 1), "audio_seconds": round(audio_s, 1),
                          "max_rss_gb": round(vc.rss_gb(), 2)}), flush=True)
    rows = [done[u["id"]] for u in utts if u["id"] in done]
    res = score(rows, norm)
    res["n_hit_max_length_without_eos"] = sum(1 for r in rows if not r["ended_with_eos"])
    print(json.dumps({"out": str(out), **res}))


if __name__ == "__main__":
    main()
