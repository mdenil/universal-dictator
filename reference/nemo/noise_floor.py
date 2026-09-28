"""Measure the reference's own numerical spread and write data/golden/<model>/noise_floor.json.

1. Thread-count spread: offline goldens dumped with torch threads=8 (data/golden/<model>/offline_*) versus
   threads=1 (data/golden/<model>/noise_t1/offline_*). Per saved tensor name: max-abs difference and
   max-abs difference relative to max|threads-8 tensor|, worst case over the dev20 utterances; token
   sequence equality per utterance.
2. Nemotron only: offline (full utterance, same mask) vs official cache-aware streaming encoder output
   (data/golden/nemotron/stream_<cfg>/<utt>/stream_enc_out.npy), plus token / transcript equality.
3. Parakeet only: offline-with-mask transcript vs official buffered streaming transcript (these are
   different algorithms, so only equality counts are reported).

Usage: reference/nemo/.venv/bin/python reference/nemo/noise_floor.py --model nemotron
(no model is loaded)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402

INT_TENSORS = {"tokens", "token_frames", "mel_len", "att_mask"}


def diff(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = np.abs(a - b)
    denom = np.abs(b).max()
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    return float(d.max()), float(d.max() / denom) if denom > 0 else 0.0, cos


def thread_spread(model: str):
    root = DATA / "golden" / model
    out = {}
    for cfg_dir in sorted(root.glob("offline_*")):
        t1_dir = root / "noise_t1" / cfg_dir.name
        per_tensor = {}
        tokens_equal = {}
        argmax_equal_first20 = {}
        for ud in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            u1 = t1_dir / ud.name
            for f in sorted(ud.glob("*.npy")):
                name = f.stem
                a8, a1 = np.load(f), np.load(u1 / f.name)
                if name in INT_TENSORS:
                    if name == "tokens":
                        tokens_equal[ud.name] = bool(a8.shape == a1.shape and (a8 == a1).all())
                    elif not (a8.shape == a1.shape and (a8 == a1).all()):
                        per_tensor.setdefault(name + " (int, mismatch)", []).append(ud.name)
                    continue
                if a8.shape != a1.shape:
                    per_tensor.setdefault(name + " (shape mismatch)", []).append(ud.name)
                    continue
                if a8.size == 0:
                    continue
                mx, rl, cos = diff(a1, a8)
                w = per_tensor.setdefault(name, {"max_abs": 0.0, "max_rel": 0.0, "min_cos": 1.0, "max_ref_abs": 0.0})
                w["max_abs"] = max(w["max_abs"], mx)
                w["max_rel"] = max(w["max_rel"], rl)
                w["min_cos"] = min(w["min_cos"], cos)
                w["max_ref_abs"] = max(w["max_ref_abs"], float(np.abs(a8).max()))
                if name == "joint_logits_first20":
                    argmax_equal_first20[ud.name] = bool((a8.argmax(-1) == a1.argmax(-1)).all())
        out[cfg_dir.name] = {
            "compare": "threads=1 vs threads=8 (reference = threads 8)",
            "per_tensor_worst_over_utts": per_tensor,
            "tokens_equal": tokens_equal,
            "all_tokens_equal": all(tokens_equal.values()),
            "joint_argmax_equal_first20": all(argmax_equal_first20.values()) if argmax_equal_first20 else None,
        }
    return out


def nemotron_stream_vs_offline():
    root = DATA / "golden" / "nemotron"
    res = {}
    for cfg in ["70_1", "70_6"]:
        off_dir, st_dir = root / f"offline_{cfg}", root / f"stream_{cfg}"
        if not st_dir.exists():
            continue
        per_utt = {}
        worst = {"max_abs": 0.0, "max_rel": 0.0, "min_cos": 1.0}
        for ud in sorted(p for p in st_dir.iterdir() if p.is_dir()):
            so = np.load(ud / "stream_enc_out.npy")
            oo = np.load(off_dir / ud.name / "enc_out.npy")
            n = min(so.shape[0], oo.shape[0])
            mx, rl, cos = diff(so[:n], oo[:n])
            tok_s = np.load(ud / "tokens.npy")
            tok_o = np.load(off_dir / ud.name / "tokens.npy")
            txt_s = (ud / "transcript.txt").read_text().strip()
            txt_o = (off_dir / ud.name / "transcript.txt").read_text().strip()
            # Per-frame diff profile: where does the difference live?
            fr = np.abs(so[:n].astype(np.float64) - oo[:n]).max(axis=1)
            per_utt[ud.name] = {
                "stream_frames": int(so.shape[0]),
                "offline_frames": int(oo.shape[0]),
                "compared_frames": n,
                "max_abs": mx,
                "max_rel": rl,
                "cos": cos,
                "worst_frame": int(fr.argmax()),
                "max_abs_excl_last_frame": float(fr[:-1].max()) if n > 1 else None,
                "tokens_equal": bool(tok_s.shape == tok_o.shape and (tok_s == tok_o).all()),
                "transcript_equal": txt_s == txt_o,
            }
            worst["max_abs"] = max(worst["max_abs"], mx)
            worst["max_rel"] = max(worst["max_rel"], rl)
            worst["min_cos"] = min(worst["min_cos"], cos)
        res[cfg] = {
            "compare": f"official cache-aware streaming (stream_{cfg}) vs offline_{cfg} enc_out, common frames",
            "worst": worst,
            "n_tokens_equal": sum(v["tokens_equal"] for v in per_utt.values()),
            "n_transcripts_equal": sum(v["transcript_equal"] for v in per_utt.values()),
            "n_utts": len(per_utt),
            "per_utt": per_utt,
        }
    return res


def parakeet_stream_vs_offline():
    root = DATA / "golden" / "parakeet"
    st = root / "stream_70_2_5" / "transcripts.jsonl"
    if not st.exists():
        return None
    per = {}
    for line in st.read_text().splitlines():
        r = json.loads(line)
        off = (root / "offline_70_2_5" / r["utt_id"] / "transcript.txt").read_text().strip()
        per[r["utt_id"]] = {"equal": off == r["pred_text"].strip(), "offline": off, "buffered": r["pred_text"]}
    return {
        "compare": "offline_70_2_5 (whole utterance, [70,2,5] mask, whole-utterance normalization) vs official "
        "buffered streaming (5.6/0.16/0.40 s): different algorithms, transcript equality only",
        "n_equal": sum(v["equal"] for v in per.values()),
        "n_utts": len(per),
        "differing": {k: v for k, v in per.items() if not v["equal"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["nemotron", "parakeet"])
    args = ap.parse_args()
    out = {"thread_spread": thread_spread(args.model)}
    if args.model == "nemotron":
        out["stream_vs_offline"] = nemotron_stream_vs_offline()
    else:
        out["buffered_vs_offline"] = parakeet_stream_vs_offline()
    path = DATA / "golden" / args.model / "noise_floor.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")
    for cfg, v in out["thread_spread"].items():
        print(cfg, "all_tokens_equal", v["all_tokens_equal"], "argmax20", v["joint_argmax_equal_first20"])
        for k in ["mel", "pre_encode", "layer_00_out", "layer_11_out", "layer_23_out", "enc_out",
                  "joint_logits_first20"]:
            w = v["per_tensor_worst_over_utts"].get(k)
            if w:
                print(f"  {k:22s} max_abs {w['max_abs']:.3e}  max_rel {w['max_rel']:.3e}  min_cos {w['min_cos']:.9f}")
    if args.model == "nemotron":
        for cfg, v in out["stream_vs_offline"].items():
            print("stream vs offline", cfg, v["worst"], "tokens_equal", v["n_tokens_equal"], "/", v["n_utts"],
                  "transcripts_equal", v["n_transcripts_equal"])
    elif out["buffered_vs_offline"]:
        v = out["buffered_vs_offline"]
        print("buffered vs offline transcripts equal:", v["n_equal"], "/", v["n_utts"])


if __name__ == "__main__":
    main()
