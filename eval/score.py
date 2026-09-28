"""Scores engine hypotheses (jsonl lines with "ref" and "hyp") with the same normalizer and corpus
WER as reference/nemo/wer.py, so our numbers and the reference numbers are directly comparable.

Usage: reference/nemo/.venv/bin/python eval/score.py artifacts/eval/<run>.jsonl [...]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference" / "nemo"))
from wer import score  # noqa: E402

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    r = score([(x["ref"], x["hyp"]) for x in rows])
    audio = sum(x.get("duration", 0) for x in rows)
    wall = sum(x.get("wall", 0) for x in rows)
    print(f"{Path(path).name}: WER {100 * r['wer']:.2f}% over {r['n_scored']} utts / {r['ref_words']} words "
          f"(S {r['substitutions']} D {r['deletions']} I {r['insertions']})"
          + (f", RTF {wall / audio:.4f}" if audio else ""))
