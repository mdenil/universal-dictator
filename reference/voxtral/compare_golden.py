"""Compare two golden dumps (noise floor of the reference).

Usage:
  reference/voxtral/.venv/bin/python reference/voxtral/compare_golden.py \
      --a data/golden/voxtral/offline --b data/golden/voxtral/offline_mps_fp32 \
      --label cpu_fp32_vs_mps_fp32 --json data/golden/voxtral/noise_floor.json

For every utterance present in both dumps and every float tensor present in both,
records max-abs difference, relative max-abs (max|a-b| / max|a|), relative RMS
(||a-b|| / ||a||) and cosine similarity, then aggregates the worst case per
tensor over utterances. Token sequences are compared exactly. Decoder tensors
after the first position where the fed token ids differ are excluded (they no
longer have the same input) and the exclusion is reported.
The result is merged into --json under key --label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INT_TENSORS = {"tokens", "prompt_ids", "dec_input_ids", "logits_top10_ids"}


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = a - b
    na = np.linalg.norm(a)
    return {
        "max_abs": float(np.abs(d).max()),
        "ref_max_abs": float(np.abs(a).max()),
        "rel_max_abs": float(np.abs(d).max() / max(np.abs(a).max(), 1e-30)),
        "rel_rms": float(np.linalg.norm(d) / max(na, 1e-30)),
        "cosine": float((a.ravel() @ b.ravel()) / max(na * np.linalg.norm(b), 1e-30)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    A, B = Path(args.a), Path(args.b)
    utts = sorted(p.name for p in A.iterdir() if p.is_dir() and (B / p.name).is_dir())
    per_utt = {}
    worst: dict[str, dict] = {}
    tok_equal = 0
    for u in utts:
        ua, ub = A / u, B / u
        ta, tb = np.load(ua / "tokens.npy"), np.load(ub / "tokens.npy")
        n = min(len(ta), len(tb))
        diff = np.nonzero(ta[:n] != tb[:n])[0]
        first_diff = int(diff[0]) if len(diff) else (None if len(ta) == len(tb) else n)
        equal = bool(len(ta) == len(tb) and first_diff is None)
        tok_equal += equal
        # decoder positions whose fed ids are identical in both runs: prompt (39) + first_diff steps
        n_prompt = len(np.load(ua / "prompt_ids.npy"))
        dec_ok = None if first_diff is None else n_prompt + first_diff  # positions < dec_ok are valid
        rec = {"tokens_equal": equal, "first_token_diff_step": first_diff, "len_a": int(len(ta)), "len_b": int(len(tb)),
               "transcript_equal": (ua / "transcript.txt").read_text() == (ub / "transcript.txt").read_text(),
               "tensors": {}}
        for f in sorted(ua.glob("*.npy")):
            name = f.stem
            if name in INT_TENSORS or not (ub / f.name).exists():
                continue
            a, b = np.load(f), np.load(ub / f.name)
            if name.startswith("dec_") and dec_ok is not None:
                a, b = a[:dec_ok], b[:dec_ok]
            if name.startswith("logits_first30") and first_diff is not None:
                a, b = a[: first_diff + 1], b[: first_diff + 1]  # decision at first_diff shares its input
            if name == "logits_top10_values":
                m = min(len(a), len(b)) if first_diff is None else first_diff + 1
                a, b = a[:m], b[:m]
            if a.shape != b.shape or a.size == 0:
                rec["tensors"][name] = {"skipped": f"shape {a.shape} vs {b.shape}"}
                continue
            s = stats(a, b)
            rec["tensors"][name] = s
            w = worst.setdefault(name, {"max_abs": 0.0, "rel_max_abs": 0.0, "rel_rms": 0.0, "min_cosine": 1.0})
            w["max_abs"] = max(w["max_abs"], s["max_abs"])
            w["rel_max_abs"] = max(w["rel_max_abs"], s["rel_max_abs"])
            w["rel_rms"] = max(w["rel_rms"], s["rel_rms"])
            w["min_cosine"] = min(w["min_cosine"], s["cosine"])
        # argmax agreement + top-1 margin on the shared decisions
        ia, ib = np.load(ua / "logits_top10_ids.npy"), np.load(ub / "logits_top10_ids.npy")
        va = np.load(ua / "logits_top10_values.npy")
        m = min(len(ia), len(ib)) if first_diff is None else first_diff + 1
        rec["argmax_agree"] = int((ia[:m, 0] == ib[:m, 0]).sum())
        rec["argmax_compared"] = int(m)
        rec["min_top1_margin_a"] = float((va[:m, 0] - va[:m, 1]).min()) if m else None
        per_utt[u] = rec
    result = {
        "a": str(A), "b": str(B),
        "n_utts": len(utts),
        "n_token_sequences_equal": tok_equal,
        "worst_over_utts": worst,
        "per_utt": per_utt,
    }
    print(json.dumps({"label": args.label, "n_utts": len(utts), "tokens_equal": tok_equal}, indent=None))
    for k, v in worst.items():
        print(f"  {k:28s} max_abs={v['max_abs']:.3e} rel_max={v['rel_max_abs']:.3e} rel_rms={v['rel_rms']:.3e} min_cos={v['min_cosine']:.8f}")
    for u, r in per_utt.items():
        if not r["tokens_equal"]:
            print(f"  tokens differ: {u} first_diff_step={r['first_token_diff_step']} len {r['len_a']} vs {r['len_b']}")
    if args.json:
        p = Path(args.json)
        allres = json.loads(p.read_text()) if p.exists() else {}
        allres[args.label] = result
        p.write_text(json.dumps(allres, indent=1) + "\n")


if __name__ == "__main__":
    main()
