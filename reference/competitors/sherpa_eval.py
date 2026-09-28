"""Runs the sherpa-onnx streaming export of Nemotron (ONNX Runtime, CPU) over a manifest, writing the
same jsonl as `dictate-cli eval` (ref, hyp, duration, wall) so eval/score.py scores both identically.
Audio is fed in 20 ms pieces like our eval. Speed numbers are only admissible from a quiet machine.

Usage: reference/competitors/.venv/bin/python reference/competitors/sherpa_eval.py <model_dir> <int8|fp32>
         <manifest> <out.jsonl> [--limit N] [--threads 4]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("variant"); ap.add_argument("manifest"); ap.add_argument("out")
ap.add_argument("--limit", type=int, default=10**9); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
d = ROOT / a.model_dir
sfx = ".int8.onnx" if a.variant == "int8" else ".onnx"
rec = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=str(d / "tokens.txt"), encoder=str(d / f"encoder{sfx}"), decoder=str(d / f"decoder{sfx}"),
    joiner=str(d / f"joiner{sfx}"), num_threads=a.threads, sample_rate=16000, feature_dim=128,
    decoding_method="greedy_search", provider="cpu")
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["audio"] for l in open(out)} if out.exists() else set()
rows = [json.loads(l) for l in open(ROOT / a.manifest) if l.strip()][: a.limit]
with open(out, "a") as f:
    for r in rows:
        if r["audio"] in done:
            continue
        wav, sr = sf.read(ROOT / r["audio"], dtype="float32")
        t0 = time.perf_counter()
        s = rec.create_stream()
        for i in range(0, len(wav), 320):
            s.accept_waveform(16000, wav[i:i + 320])
            while rec.is_ready(s):
                rec.decode_stream(s)
        s.accept_waveform(16000, np.zeros(16000 // 2, dtype=np.float32))   # tail padding as in sherpa examples
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        hyp = rec.get_result(s)
        f.write(json.dumps({"audio": r["audio"], "ref": r["text"], "hyp": hyp, "duration": len(wav) / 16000,
                            "wall": time.perf_counter() - t0}, sort_keys=True) + "\n")
        f.flush()
