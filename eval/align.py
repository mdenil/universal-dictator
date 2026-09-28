"""Forced-aligns reference transcripts to get word start/end times (MMS_FA wav2vec2 CTC aligner).

Writes data/align/<name>.jsonl with {"audio", "words": [[word, start_s, end_s], ...]} per utterance,
resumable. The model is downloaded once into the torch hub cache.

Usage: reference/nemo/.venv/bin/python eval/align.py <manifest.jsonl> <name> [--limit N]
"""

import argparse
import json
import re
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("name")
    ap.add_argument("--limit", type=int, default=10**9)
    a = ap.parse_args()
    out_path = ROOT / "data" / "align" / f"{a.name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["audio"] for l in open(out_path) if l.strip()}
    torch.set_num_threads(4)
    model = bundle.get_model(with_star=False).eval()
    dictionary = bundle.get_dict(star=None)
    aligner = bundle.get_aligner()
    rows = [json.loads(l) for l in open(ROOT / a.manifest if not Path(a.manifest).is_absolute() else a.manifest) if l.strip()][: a.limit]
    with open(out_path, "a") as out:
        for r in rows:
            if r["audio"] in done:
                continue
            wav, sr = sf.read(ROOT / r["audio"], dtype="float32")
            assert sr == 16000
            words = [w for w in re.sub(r"[^a-z' ]", " ", r["text"].lower()).split() if w]
            tokens = [[dictionary[c] for c in w if c in dictionary] for w in words]
            with torch.inference_mode():
                emission, _ = model(torch.from_numpy(wav)[None])
                spans = aligner(emission[0], tokens)
            ratio = wav.shape[0] / emission.shape[1] / sr   # seconds per emission frame
            res = [[w, s[0].start * ratio, s[-1].end * ratio] for w, s in zip(words, spans)]
            out.write(json.dumps({"audio": r["audio"], "words": res}) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
