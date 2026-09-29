"""Collects every eval run in artifacts/eval into one table: WER per test set, RTF, and word latency
(when the run recorded per-step emissions and alignments exist). Writes artifacts/eval/summary.json
and prints a markdown table.

Usage: reference/nemo/.venv/bin/python eval/report.py [--pattern '*_200.jsonl']
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "nemo"))
sys.path.insert(0, str(ROOT / "eval"))
from wer import score  # noqa: E402
from latency import edit_pairs, words  # noqa: E402


def latency(rows, align):
    fin = []
    for r in rows:
        if "steps" not in r or r["audio"] not in align:
            continue
        aw = align[r["audio"]]
        ref = [w[0] for w in aw]
        steps = [(s[0] / 16000.0 + s[1], words(s[2])) for s in r["steps"]]
        final = words(r["hyp"])
        for i, j in edit_pairs(ref, final):
            t_final = None
            for t, ws in steps:
                ok = len(ws) > j and ws[j] == final[j] and (len(ws) > j + 1 or j == len(final) - 1)
                if ok and t_final is None:
                    t_final = t
                elif not ok:
                    t_final = None
            if t_final is not None:
                fin.append(t_final - aw[i][2])
    return fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*.jsonl")
    a = ap.parse_args()
    aligns = {}
    for f in glob.glob(str(ROOT / "data/align/*.jsonl")):
        for l in open(f):
            d = json.loads(l)
            aligns[d["audio"]] = d["words"]
    table = {}
    for path in sorted(glob.glob(str(ROOT / "artifacts/eval" / a.pattern))):
        name = Path(path).stem
        m = re.match(r"(.+?)_(f16|f32|q8|q4\w*|int8|fp32)_(test-clean|test-other|dictation)(?:_(?:ref)?(\d+))?$", name)
        if not m:
            continue
        cfg, prec, tset = m.group(1), m.group(2), m.group(3)
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if not rows:
            continue
        s = score([(r["ref"], r["hyp"]) for r in rows])
        audio = sum(r["duration"] for r in rows)
        comp = sum(r.get("step_compute_total", r.get("wall", 0)) for r in rows)
        lat = latency(rows, aligns)
        e = table.setdefault(f"{cfg} ({prec})", {})
        e[tset] = {"wer": 100 * s["wer"], "n": len(rows), "rtf": comp / audio,
                   "lat_p50_ms": float(np.percentile(lat, 50) * 1000) if lat else None,
                   "lat_p90_ms": float(np.percentile(lat, 90) * 1000) if lat else None}
    (ROOT / "artifacts/eval/summary.json").write_text(json.dumps(table, indent=1))
    sets = ["test-clean", "test-other", "dictation"]
    print("| config | " + " | ".join(f"{t} WER" for t in sets) + " | RTF | final-word latency p50 / p90 |")
    print("|---|" + "---|" * (len(sets) + 2))
    for cfg, e in table.items():
        cells = [f"{e[t]['wer']:.2f}% (n={e[t]['n']})" if t in e else "" for t in sets]
        rtf = np.mean([v["rtf"] for v in e.values()])
        lats = [v for v in e.values() if v["lat_p50_ms"] is not None]
        lat = f"{lats[0]['lat_p50_ms']:.0f} / {lats[0]['lat_p90_ms']:.0f} ms" if lats else ""
        print(f"| {cfg} | " + " | ".join(cells) + f" | {rtf:.3f} | {lat} |")


if __name__ == "__main__":
    main()
