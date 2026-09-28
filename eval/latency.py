"""Word-level streaming latency from `dictate-cli eval` step records and forced alignments.

For each hypothesis word that matches a reference word (Levenshtein alignment on simply normalized
words), we measure, relative to the reference word's aligned end time:
  first  - the first step at which the word appears in the partial transcript (possibly incomplete)
  final  - the first step after which the word has its final form (a later word has started or the
           stream has ended); this is when a dictation UI could commit the word
Step times are "algorithmic" (audio received / 16 kHz) and "+compute" (plus that step's compute time).
All three decoders are append-only (no rewrites of committed text); partial words only grow.

Usage: reference/nemo/.venv/bin/python eval/latency.py <run.jsonl> <align.jsonl> [...more runs]
"""

import json
import re
import sys

import numpy as np


def words(text):
    return [w for w in re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("-", " ")).split() if w]


def edit_pairs(ref, hyp):
    """Indices (i, j) of ref/hyp words aligned as exact matches by minimum edit distance."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif d[i, j] == d[i - 1, j - 1] + 1:
            i -= 1; j -= 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def analyse(run_path, align_path):
    align = {json.loads(l)["audio"]: json.loads(l)["words"] for l in open(align_path) if l.strip()}
    first_alg, final_alg, final_cmp = [], [], []
    for line in open(run_path):
        r = json.loads(line)
        if "steps" not in r or r["audio"] not in align:
            continue
        aw = align[r["audio"]]
        ref = [w[0] for w in aw]
        steps = [(s[0] / 16000.0, s[1], words(s[2])) for s in r["steps"]]
        final = words(r["hyp"])
        if not steps:
            continue
        for i, j in edit_pairs(ref, final):
            end = aw[i][2]
            t_first = next((t for t, c, ws in steps if len(ws) > j), None)
            # Final: word j equals its final form and a later word exists (or it is the last word and
            # the transcript is final); take the earliest step from which that holds for good.
            t_final = c_final = None
            for t, c, ws in steps:
                ok = len(ws) > j and ws[j] == final[j] and (len(ws) > j + 1 or j == len(final) - 1)
                if ok and t_final is None:
                    t_final, c_final = t, c
                elif not ok:
                    t_final = c_final = None
            if t_first is None or t_final is None:
                continue
            first_alg.append(t_first - end)
            final_alg.append(t_final - end)
            final_cmp.append(t_final + c_final - end)
    def q(x):
        x = np.array(x)
        return f"mean {x.mean() * 1000:6.0f}  p50 {np.percentile(x, 50) * 1000:6.0f}  p90 {np.percentile(x, 90) * 1000:6.0f}  p99 {np.percentile(x, 99) * 1000:6.0f} ms"
    print(f"{run_path.split('/')[-1]}: {len(final_alg)} matched words")
    print(f"  first appearance (algorithmic)  {q(first_alg)}")
    print(f"  final form       (algorithmic)  {q(final_alg)}")
    print(f"  final form       (+compute)     {q(final_cmp)}")


if __name__ == "__main__":
    align_path = sys.argv[2]
    for run in [sys.argv[1]] + sys.argv[3:]:
        analyse(run, align_path)
