"""Collects every eval run in artifacts/eval into one long-form table, one row per
(config, precision, test set, subset). Writes artifacts/eval/summary.json and prints markdown.

Run file names: <config>_<precision>_<set>_<subset>.jsonl, where subset is `200` / `100` (the first
N utterances of the set), `ref500` / `ref200` (the NeMo reference's subsets), `voxref100` (the
Voxtral reference's utterances) or `150` (the AMI / Earnings-22 slices). Rows on different subsets
are not comparable.

Metrics:
  wer       normalized WER (Whisper normalizer + jiwer), corpus level
  utt p90   90th percentile of per-utterance normalized WER (the bad tail)
  wer_pc    case- and punctuation-sensitive WER (words and . , ? ! as tokens), only where the
            reference is cased and punctuated (AMI, Earnings-22)
  latency   per matched word, relative to its forced-aligned end: first appearance p50, final form
            p50 / p90 (alignments exist for the first 200 LibriSpeech utterances of each set)
  ttfw      time to first word: first non-empty partial minus the first aligned word's end, p50
  rewrites  partial-transcript revisions per 100 final words: a shown complete word changes, or the
            last (partial) word changes other than by growing
  rtf       compute / audio, under whatever load the run had (use the timing session for speed)

Usage: reference/nemo/.venv/bin/python eval/report.py [--pattern '*.jsonl']
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

NAME = re.compile(r"(.+?)_(f16|f32|q8h?|q4\w*|bf16|int8|fp32)_(test-clean|test-other|ami|earnings22|dictation)_((?:ref|voxref)?\d+)$")
CASED = {"ami", "earnings22", "dictation"}


_norm = None


def utt_wer(ref, hyp):
    """Normalized WER of one utterance (None when the normalized reference is empty)."""
    global _norm
    if _norm is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _norm = EnglishTextNormalizer()
    r, h = _norm(ref).split(), _norm(hyp).split()
    if not r:
        return None
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


def pc_tokens(text):
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[.,?!]", text)


def wer_pc(rows):
    import jiwer
    refs = [" ".join(pc_tokens(r["ref"])) for r in rows]
    hyps = [" ".join(pc_tokens(r["hyp"])) for r in rows]
    keep = [i for i, r in enumerate(refs) if r.strip()]
    return jiwer.wer([refs[i] for i in keep], [hyps[i] for i in keep])


def stream_metrics(rows, align):
    first, fin, ttfw = [], [], []
    rewrites = total_words = 0
    for r in rows:
        if "steps" not in r:
            continue
        steps = [(s[0] / 16000.0, words(s[2])) for s in r["steps"]]
        final = words(r["hyp"])
        total_words += len(final)
        prev = []
        for _, ws in steps:
            for j, w in enumerate(prev):
                if j >= len(ws):
                    rewrites += 1; break
                if j < len(prev) - 1 and ws[j] != w:
                    rewrites += 1; break
                if j == len(prev) - 1 and not ws[j].startswith(w):
                    rewrites += 1; break
            prev = ws
        if r["audio"] not in align:
            continue
        aw = align[r["audio"]]
        if aw and steps:
            t0 = next((t for t, ws in steps if ws), None)
            if t0 is not None:
                ttfw.append(t0 - aw[0][2])
        for i, j in edit_pairs([w[0] for w in aw], final):
            t_first = next((t for t, ws in steps if len(ws) > j), None)
            t_final = None
            for t, ws in steps:
                ok = len(ws) > j and ws[j] == final[j] and (len(ws) > j + 1 or j == len(final) - 1)
                if ok and t_final is None:
                    t_final = t
                elif not ok:
                    t_final = None
            if t_final is not None and t_first is not None:
                first.append(t_first - aw[i][2])
                fin.append(t_final - aw[i][2])
    pct = lambda x, q: float(np.percentile(x, q) * 1000) if x else None
    return {"first_p50_ms": pct(first, 50), "final_p50_ms": pct(fin, 50), "final_p90_ms": pct(fin, 90),
            "ttfw_p50_ms": pct(ttfw, 50), "n_latency_words": len(fin),
            "rewrites_per_100w": 100.0 * rewrites / total_words if total_words else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*.jsonl")
    a = ap.parse_args()
    aligns = {}
    for f in glob.glob(str(ROOT / "data/align/*.jsonl")):
        for l in open(f):
            d = json.loads(l)
            aligns[d["audio"]] = d["words"]
    table = []
    for path in sorted(glob.glob(str(ROOT / "artifacts/eval" / a.pattern))):
        m = NAME.match(Path(path).stem)
        if not m:
            continue
        cfg, prec, tset, subset = m.groups()
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if not rows:
            continue
        s = score([(r["ref"], r["hyp"]) for r in rows])
        utt = [u for u in (utt_wer(r["ref"], r["hyp"]) for r in rows) if u is not None]
        audio = sum(r["duration"] for r in rows)
        comp = sum(r.get("step_compute_total", r.get("wall", 0)) for r in rows)
        # Alignments cover the first 200 LibriSpeech utterances, so latency is reported on those runs only.
        sm = stream_metrics(rows, aligns if subset in ("200", "100") else {})
        table.append({"config": cfg, "precision": prec, "set": tset, "subset": subset, "n": len(rows),
                      "wer": 100 * s["wer"], "subs": s["substitutions"], "dels": s["deletions"], "ins": s["insertions"],
                      "ref_words": s["ref_words"], "utt_wer_p90": 100 * float(np.percentile(utt, 90)),
                      "wer_pc": 100 * wer_pc(rows) if tset in CASED else None, "rtf": comp / audio, **sm})
    (ROOT / "artifacts/eval/summary.json").write_text(json.dumps(table, indent=1))
    f = lambda v, fmt: "" if v is None else fmt.format(v)
    print("| config | set | subset | n | WER | utt p90 | WER-PC | first / final p50 / p90 (ms) | TTFW p50 | rewrites /100w | RTF (loaded) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in table:
        lat = "" if e["final_p50_ms"] is None else f"{e['first_p50_ms']:.0f} / {e['final_p50_ms']:.0f} / {e['final_p90_ms']:.0f}"
        print(f"| {e['config']} ({e['precision']}) | {e['set']} | {e['subset']} | {e['n']} | {e['wer']:.2f}% | {e['utt_wer_p90']:.1f}% | "
              f"{f(e['wer_pc'], '{:.1f}%')} | {lat} | {f(e['ttfw_p50_ms'], '{:.0f}')} | {f(e['rewrites_per_100w'], '{:.2f}')} | {e['rtf']:.3f} |")


if __name__ == "__main__":
    main()
