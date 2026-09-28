"""Build LibriSpeech test-clean/test-other manifests and the fixed dev20 subset.

Input:  data/librispeech/LibriSpeech/{test-clean,test-other}/**.flac (extracted openslr tarballs)
Output: data/librispeech/{test-clean,test-other}.jsonl  and  data/dev/dev20.jsonl
Each line: {"audio": <repo-relative path to the 16 kHz mono FLAC>, "text": <reference>, "duration": <s>}

The FLAC files are kept as-is: they are 16 kHz mono 16-bit, and soundfile (which NeMo's get_samples and
librosa.load both use) decodes them to float32 in [-1, 1).

Usage: reference/nemo/.venv/bin/python reference/nemo/make_dev_data.py
"""

import json
import random
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
LS = ROOT / "data" / "librispeech"
SEED = 20260928
# Per set: 3 short (< 5 s), 4 medium (5-15 s), 3 long (> 15 s).
STRATA = [(0.0, 5.0, 3), (5.0, 15.0, 4), (15.0, 1e9, 3)]


def build(split: str) -> list[dict]:
    base = LS / "LibriSpeech" / split
    recs = []
    for trans in sorted(base.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            flac = trans.parent / f"{uid}.flac"
            info = sf.info(str(flac))
            assert info.samplerate == 16000 and info.channels == 1, (flac, info)
            recs.append(
                {
                    "audio": str(flac.relative_to(ROOT)),
                    "text": text,
                    "duration": info.frames / info.samplerate,
                }
            )
    out = LS / f"{split}.jsonl"
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{out}: {len(recs)} utterances, {sum(r['duration'] for r in recs) / 3600:.3f} h")
    return recs


def main():
    rng = random.Random(SEED)
    dev = []
    for split in ["test-clean", "test-other"]:
        recs = build(split)
        for lo, hi, n in STRATA:
            pool = [r for r in recs if lo <= r["duration"] < hi]
            dev.extend(rng.sample(pool, n))
    assert any(r["duration"] > 15 for r in dev)
    out = ROOT / "data" / "dev" / "dev20.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in dev:
            f.write(json.dumps(r) + "\n")
    print(f"{out}: {len(dev)} utterances, durations {[round(r['duration'], 2) for r in dev]}")


if __name__ == "__main__":
    main()
