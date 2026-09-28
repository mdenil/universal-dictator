"""Reference WER on LibriSpeech through the official NeMo streaming scripts.

Runs the official script (via run_official.py) on a manifest, then scores it with the Whisper English
text normalizer (whisper_normalizer.english.EnglishTextNormalizer) and jiwer (corpus WER = total word
errors / total reference words).

  nemotron: speech_to_text_cache_aware_streaming_infer.py (v2.7.3), att_context_size=[70,1] or [70,6]
  parakeet: speech_to_text_streaming_infer_rnnt.py (c42e444) with 5.6 / 0.16 / 0.40 s, att_context_size_as_chunk

The NeMo manifest is written sorted by duration (longest first) so batches hold similar lengths; the
official scripts batch consecutive manifest lines, and the per-utterance results are matched back by path.

Outputs: data/wer/<model>_<cfg>/<set>/{manifest.nemo.jsonl, official output, result.json}

Usage:
  scripts/heavy.sh reference/nemo/.venv/bin/python reference/nemo/wer.py --model nemotron --config 70_1 \
      --set test-clean [--device cpu|mps] [--batch-size 32] [--limit N]
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODEL_PATHS, ROOT, read_manifest  # noqa: E402


def score(pairs):
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    norm = EnglishTextNormalizer()
    refs, hyps = [], []
    for ref, hyp in pairs:
        r, h = norm(ref), norm(hyp)
        if not r.strip():
            continue
        refs.append(r)
        hyps.append(h)
    out = jiwer.process_words(refs, hyps)
    n_ref = sum(len(r.split()) for r in refs)
    return {
        "wer": out.wer,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "ref_words": n_ref,
        "n_scored": len(refs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["nemotron", "parakeet"])
    ap.add_argument("--config", required=True, help="nemotron: 70_1 | 70_6; parakeet: 70_2_5")
    ap.add_argument("--set", required=True, help="test-clean | test-other | path to a .jsonl manifest")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="fixed random subset of this many utterances (seed 0)")
    ap.add_argument("--rescore", action="store_true", help="skip the run; score the existing official output")
    args = ap.parse_args()

    src = Path(args.set) if args.set.endswith(".jsonl") else DATA / "librispeech" / f"{args.set}.jsonl"
    recs = read_manifest(src)
    tag = src.stem
    if args.limit:
        recs = random.Random(0).sample(recs, args.limit)
        tag += f"_subset{args.limit}"
    recs = sorted(recs, key=lambda r: -r["duration"])
    out_dir = DATA / "wer" / f"{args.model}_{args.config}_{args.device}" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    man = out_dir / "manifest.nemo.jsonl"
    with open(man, "w") as f:
        for r in recs:
            f.write(json.dumps({"audio_filepath": str(ROOT / r["audio"]), "text": r["text"],
                                "duration": r["duration"]}) + "\n")

    runner = [sys.executable, str(Path(__file__).resolve().parent / "run_official.py")]
    mps = "true" if args.device == "mps" else "false"
    if args.model == "nemotron":
        acs = "[" + ",".join(args.config.split("_")) + "]"
        cmd = runner + [
            "cache_aware", f"model_path={MODEL_PATHS['nemotron']}", f"dataset_manifest={man}",
            f"att_context_size={acs}", f"batch_size={args.batch_size}", "cuda=-1", f"allow_mps={mps}",
            "compute_dtype=float32", f"output_path={out_dir}",
        ]
        # ca_infer.py:474 names it "streaming_out_" + splitext(model_name)[0] + "_" + <manifest stem> + ".json";
        # model_name is already the .nemo stem, so splitext cuts it at ".6b" -> "...-en-0". Glob instead.
        pred_file = None
    else:
        assert args.config == "70_2_5"
        pred_file = out_dir / "official_output.jsonl"
        cmd = runner + [
            "buffered_rnnt", f"model_path={MODEL_PATHS['parakeet']}", f"dataset_manifest={man}",
            f"output_filename={pred_file}", "left_context_secs=5.6", "chunk_secs=0.16",
            "right_context_secs=0.40", "att_context_size_as_chunk=true", f"batch_size={args.batch_size}",
            "compute_dtype=float32", "cuda=-1", f"allow_mps={mps}",
        ]
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    if not args.rescore:
        subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    if args.rescore:
        prev = json.loads((out_dir / "result.json").read_text()) if (out_dir / "result.json").exists() else {}
        elapsed = prev.get("elapsed_s", float("nan"))
    if pred_file is None:
        (pred_file,) = list(out_dir.glob("streaming_out_*.json"))

    preds = read_manifest(pred_file)
    if args.model == "nemotron":
        # The cache-aware script writes records in manifest order with keys pred_text / text.
        assert len(preds) == len(recs), (len(preds), len(recs))
        pairs = [(r["text"], p["pred_text"]) for r, p in zip(recs, preds)]
        for r, p in zip(recs, preds):
            assert p["text"] == r["text"]
    else:
        by_path = {p["audio_filepath"]: p for p in preds}
        pairs = [(r["text"], by_path[str(ROOT / r["audio"])]["pred_text"]) for r in recs]
    res = score(pairs)
    audio_h = sum(r["duration"] for r in recs) / 3600
    res.update({
        "model": args.model, "config": args.config, "set": tag, "device": args.device,
        "batch_size": args.batch_size, "n_utts": len(recs), "audio_hours": audio_h,
        "elapsed_s": elapsed, "rtfx": audio_h * 3600 / elapsed, "command": " ".join(cmd),
        "normalizer": "whisper_normalizer.english.EnglishTextNormalizer",
    })
    (out_dir / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
