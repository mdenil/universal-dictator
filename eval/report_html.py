"""Builds the M6 evaluation report (artifacts/report/m6.html) from the eval runs in artifacts/eval,
the quiet-window timings in artifacts/timing_quiet_2026-09-29.txt and artifacts/timing_m6/.

Head-to-head numbers are computed on matched utterances: the first 100 of each LibriSpeech set
(covered by the forced alignments, so latency is comparable), and the 150-utterance AMI and
Earnings-22 slices. Reference comparisons use each reference's own subset.

Usage: reference/nemo/.venv/bin/python eval/report_html.py
"""

import glob
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "nemo"))
sys.path.insert(0, str(ROOT / "eval"))
from wer import score  # noqa: E402
from report import stream_metrics, utt_wer, wer_pc  # noqa: E402

E = ROOT / "artifacts/eval"
OUT = ROOT / "artifacts/report/m6.html"

# The shipping candidates: (label, file stem prefix, colour token, latency setting in ms, step budget in ms)
CANDIDATES = [
    ("Nemotron 160 ms", "nemotron_160d4_f16", "nem", 160, 160),
    ("Nemotron 560 ms", "nemotron_560d4_f16", "nem2", 560, 560),
    ("Parakeet exact", "parakeet_exact_f16", "par", 560, 160),
    ("Parakeet h4", "parakeet_h4_f16", "par2", 560, 160),
    ("Voxtral q8h", "voxtral_q8h", "vox", 480, 80),
]
TIMING_NAMES = {"Nemotron 160 ms": "nemotron-160-f16", "Nemotron 560 ms": "nemotron-560-f16", "Parakeet exact": "parakeet-exact-f16",
                "Parakeet h4": "parakeet-h4-f16", "Voxtral q8h": "voxtral-q8h"}


def load(path):
    p = Path(path)
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def first_n(set_name, n):
    return [json.loads(l)["audio"] for l in open(ROOT / f"data/librispeech/{set_name}.jsonl")][:n]


def find_run(prefix, set_name, subsets):
    for s in subsets:
        p = E / f"{prefix}_{set_name}_{s}.jsonl"
        if p.exists():
            return load(p)
    return []


def aligns():
    a = {}
    for f in glob.glob(str(ROOT / "data/align/*.jsonl")):
        for l in open(f):
            d = json.loads(l)
            a[d["audio"]] = d["words"]
    return a


def matched(rows, keys):
    by = {r["audio"]: r for r in rows}
    return [by[k] for k in keys if k in by] if all(k in by for k in keys) else None


def metrics(rows, al=None, cased=False):
    s = score([(r["ref"], r["hyp"]) for r in rows])
    utt = [u for u in (utt_wer(r["ref"], r["hyp"]) for r in rows) if u is not None]
    m = {"wer": 100 * s["wer"], "n": len(rows), "p90": 100 * float(np.percentile(utt, 90)),
         "empty": sum(1 for r in rows if not r["hyp"].strip())}
    if cased:
        m["wer_pc"] = 100 * wer_pc(rows)
    if al is not None:
        m.update(stream_metrics(rows, al))
    return m


def parse_timing():
    t = {}
    p = ROOT / "artifacts/timing_quiet_2026-09-29.txt"
    for line in open(p):
        m = re.match(r"(\S+) \|.*?p50 ([\d.]+) ms p99 ([\d.]+) ms", line)
        if m:
            t.setdefault(m.group(1), []).append((float(m.group(2)), float(m.group(3))))
    return {k: (float(np.median([a for a, _ in v])), float(np.median([b for _, b in v]))) for k, v in t.items()}


def long_sessions():
    out = {}
    for name, f in [("Nemotron 160 ms", "long_nemotron160"), ("Parakeet h4", "long_parakeet_h4"), ("Voxtral q8h", "long_voxtral_q8h")]:
        steps = ROOT / f"artifacts/timing_m6/{f}.steps"
        err = ROOT / f"artifacts/timing_m6/{f}.err"
        if not steps.exists():
            continue
        x = np.array([float(v) for v in open(steps).read().split()]) * 1000
        step_s = {"Voxtral q8h": 0.08}.get(name, 0.16)
        per_min = int(round(60 / step_s))
        mins = [float(np.median(x[i:i + per_min])) for i in range(0, len(x) - per_min + 1, per_min)]
        rss = fp = None
        if err.exists():
            e = open(err).read()
            m = re.search(r"(\d+)\s+peak memory footprint", e)
            fp = int(m.group(1)) / 1e9 if m else None
        out[name] = {"per_min_p50": mins, "p50": float(np.median(x)), "p99": float(np.percentile(x, 99)),
                     "first5": float(np.median(x[: 5 * per_min])), "last5": float(np.median(x[-5 * per_min:])),
                     "footprint_gb": fp, "steps": len(x)}
    return out


def svg_pareto(points, w=640, h=380):
    """points: (label, colour token, latency_first, latency_final, wer). x = latency (ms), y = WER (%)."""
    pad_l, pad_r, pad_t, pad_b = 56, 24, 20, 48
    xs = [p[2] for p in points] + [p[3] for p in points]
    ys = [p[4] for p in points]
    x0, x1 = 0, max(1400, (int(max(xs) / 200) + 1) * 200)
    y0, y1 = (int(min(ys)) - 1) if min(ys) > 1 else 0, int(max(ys)) + 2
    X = lambda v: pad_l + (v - x0) / (x1 - x0) * (w - pad_l - pad_r)
    Y = lambda v: h - pad_b - (v - y0) / (y1 - y0) * (h - pad_t - pad_b)
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Word error rate against word latency" class="chart">']
    for v in range(x0, x1 + 1, 200):
        g.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{pad_t}" y2="{h - pad_b}" class="grid"/>')
        g.append(f'<text x="{X(v):.1f}" y="{h - pad_b + 18}" class="tick" text-anchor="middle">{v}</text>')
    for v in range(y0, y1 + 1):
        g.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        g.append(f'<text x="{pad_l - 8}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v}%</text>')
    g.append(f'<text x="{(pad_l + w - pad_r) / 2}" y="{h - 8}" class="axis" text-anchor="middle">word latency after the word ends (ms, median) — open circle: first shown, dot: final</text>')
    for label, col, lf, ll, wv in points:
        g.append(f'<line x1="{X(lf):.1f}" x2="{X(ll):.1f}" y1="{Y(wv):.1f}" y2="{Y(wv):.1f}" style="stroke:var(--{col})" class="span"/>')
        g.append(f'<circle cx="{X(lf):.1f}" cy="{Y(wv):.1f}" r="5" class="hollow" style="stroke:var(--{col})"/>')
        g.append(f'<circle cx="{X(ll):.1f}" cy="{Y(wv):.1f}" r="6" style="fill:var(--{col})"/>')
        g.append(f'<text x="{X(ll) + 10:.1f}" y="{Y(wv) + 4:.1f}" class="lbl">{html.escape(label)}</text>')
    g.append("</svg>")
    return "\n".join(g)


def svg_load(rows, w=640):
    """rows: (label, colour, p50 ms, p99 ms, budget ms). Bars show p50 as % of the step budget, tick at p99."""
    bar_h, gap, pad_l, pad_r, pad_t = 22, 14, 130, 60, 10
    h = pad_t + len(rows) * (bar_h + gap) + 34
    xmax = max(200, int(max(r[3] / r[4] * 100 for r in rows) / 50 + 1) * 50)
    X = lambda v: pad_l + v / xmax * (w - pad_l - pad_r)
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Step time as a share of the real-time budget" class="chart">']
    for v in range(0, xmax + 1, 50):
        g.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{pad_t}" y2="{h - 30}" class="grid"/>')
        g.append(f'<text x="{X(v):.1f}" y="{h - 14}" class="tick" text-anchor="middle">{v}%</text>')
    g.append(f'<line x1="{X(100):.1f}" x2="{X(100):.1f}" y1="{pad_t - 4}" y2="{h - 30}" class="budget"/>')
    for i, (label, col, p50, p99, budget) in enumerate(rows):
        y = pad_t + i * (bar_h + gap)
        g.append(f'<text x="{pad_l - 10}" y="{y + bar_h * 0.7:.1f}" class="lbl" text-anchor="end">{html.escape(label)}</text>')
        g.append(f'<rect x="{pad_l}" y="{y}" width="{X(p50 / budget * 100) - pad_l:.1f}" height="{bar_h}" rx="3" style="fill:var(--{col})"/>')
        g.append(f'<line x1="{X(p99 / budget * 100):.1f}" x2="{X(p99 / budget * 100):.1f}" y1="{y - 3}" y2="{y + bar_h + 3}" class="p99"/>')
        g.append(f'<text x="{X(max(p50, p99) / budget * 100) + 6:.1f}" y="{y + bar_h * 0.7:.1f}" class="tick">{p50:.0f} / {budget} ms</text>')
    g.append("</svg>")
    return "\n".join(g)


def svg_drift(series, w=640, h=260):
    """series: (label, colour, per-minute p50 list, budget ms). y = % of budget."""
    pad_l, pad_r, pad_t, pad_b = 48, 20, 14, 40
    n = max(len(s[2]) for s in series)
    ymax = max(120, int(max(max(v / s[3] * 100 for v in s[2]) for s in series) / 20 + 1) * 20)
    X = lambda i: pad_l + i / max(1, n - 1) * (w - pad_l - pad_r)
    Y = lambda v: h - pad_b - v / ymax * (h - pad_t - pad_b)
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Step time over a 30 minute session" class="chart">']
    for v in range(0, ymax + 1, 20):
        g.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        g.append(f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v}%</text>')
    g.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(100):.1f}" y2="{Y(100):.1f}" class="budget"/>')
    for i in range(0, n, 5):
        g.append(f'<text x="{X(i):.1f}" y="{h - pad_b + 16}" class="tick" text-anchor="middle">{i}</text>')
    g.append(f'<text x="{(pad_l + w - pad_r) / 2}" y="{h - 6}" class="axis" text-anchor="middle">minute of the session (median step time as % of budget)</text>')
    for label, col, vals, budget in series:
        pts = " ".join(f"{X(i):.1f},{Y(v / budget * 100):.1f}" for i, v in enumerate(vals))
        g.append(f'<polyline points="{pts}" class="line" style="stroke:var(--{col})"/>')
        g.append(f'<text x="{X(len(vals) - 1) - 4:.1f}" y="{Y(vals[-1] / budget * 100) - 8:.1f}" class="lbl" text-anchor="end">{html.escape(label)}</text>')
    g.append("</svg>")
    return "\n".join(g)


def fmt(v, f="{:.2f}%", none="—"):
    return none if v is None else f.format(v)


def table(head, rows, num_from=1):
    h = "".join(f"<th{' class=\"num\"' if i >= num_from else ''}>{html.escape(c)}</th>" for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f"<td{' class=\"num\"' if i >= num_from else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def tables(d):
    cand, t = d["cand"], {}
    sw = lambda l: f'<span class="sw" style="background:var(--{cand[l]["col"]})"></span>{html.escape(l)}'
    t["accuracy"] = table(["Configuration", "LS clean", "LS other", "AMI meetings", "Earnings-22 calls", "Earnings-22, case + punct."],
        [[sw(l), fmt(c["test-clean"] and c["test-clean"]["wer"]), fmt(c["test-other"] and c["test-other"]["wer"]),
          fmt(c["ami"] and c["ami"]["wer"]), fmt(c["earnings22"] and c["earnings22"]["wer"]),
          fmt(c["earnings22"] and c["earnings22"]["wer_pc"], "{:.1f}%")] for l, c in cand.items()])
    t["tail"] = table(["Configuration", "LS other, worst 10% of utterances ≥", "AMI, worst 10% ≥", "AMI, empty output", "Earnings-22, empty output"],
        [[sw(l), fmt(c["test-other"] and c["test-other"]["p90"], "{:.0f}%"), fmt(c["ami"] and c["ami"]["p90"], "{:.0f}%"),
          fmt(c["ami"] and c["ami"]["empty"], "{} / 150"), fmt(c["earnings22"] and c["earnings22"]["empty"], "{} / 150")] for l, c in cand.items()])
    t["latency"] = table(["Configuration", "Setting", "Word first shown", "Word final", "Word final, p90", "First word", "Rewrites"],
        [[sw(l), f'{c["lat"]} ms'] + ([fmt(o["first_p50_ms"], "{:.0f} ms"), fmt(o["final_p50_ms"], "{:.0f} ms"), fmt(o["final_p90_ms"], "{:.0f} ms"),
          fmt(o["ttfw_p50_ms"], "{:.0f} ms"), fmt(o["rewrites_per_100w"], "{:.0f}")] if (o := c["test-other"]) else ["—"] * 5) for l, c in cand.items()])
    t["reference"] = table(["Configuration (reference subset)", "Ours, clean", "Reference, clean", "Ours, other", "Reference, other"],
        [[html.escape(n), fmt(v[0]), fmt(r[0]), fmt(v[1]), fmt(r[1])] for n, v, r in d["refcmp"]])
    t["variants"] = table(["Configuration", "LS clean", "LS other"], [[html.escape(n), fmt(r[0]), fmt(r[1])] for n, r in d["variants"]])
    tm = d["timing"]
    t["speed"] = table(["Configuration", "Step", "Step p50", "Step p99", "Share of budget (p50)"],
        [[sw(l), f'{cand[l]["budget"]} ms'] + ([fmt(tm[k][0], "{:.0f} ms"), fmt(tm[k][1], "{:.0f} ms"), fmt(tm[k][0] / cand[l]["budget"] * 100, "{:.0f}%")]
          if (k := TIMING_NAMES[l]) in tm else ["—"] * 3) for l in cand])
    lo = d["longs"]
    t["long"] = table(["Configuration", "Steps", "p50, first 5 min", "p50, last 5 min", "p99, whole session", "Peak memory (excl. mapped weights)"],
        [[sw(l), str(v["steps"]), fmt(v["first5"], "{:.0f} ms"), fmt(v["last5"], "{:.0f} ms"), fmt(v["p99"], "{:.0f} ms"), fmt(v["footprint_gb"], "{:.2f} GB")]
         for l, v in lo.items()])
    return t


def main():
    al = aligns()
    keys = {s: first_n(s, 100) for s in ["test-clean", "test-other"]}
    cand = {}
    for label, prefix, col, lat, budget in CANDIDATES:
        c = {"col": col, "lat": lat, "budget": budget}
        for s in ["test-clean", "test-other"]:
            rows = matched(find_run(prefix, s, ["200", "100"]), keys[s])
            c[s] = metrics(rows, al) if rows else None
        for s in ["ami", "earnings22"]:
            rows = load(E / f"{prefix}_{s}_150.jsonl")
            c[s] = metrics(rows, cased=True) if len(rows) == 150 else None
        cand[label] = c

    # Reference comparisons on each reference's own subset (ours vs theirs).
    refcmp = []
    def ref_row(name, ours_files, theirs):
        vals = []
        for f in ours_files:
            rows = load(E / f)
            vals.append(100 * score([(r["ref"], r["hyp"]) for r in rows])["wer"] if rows else None)
        refcmp.append((name, vals, theirs))
    ref_row("Nemotron 160 ms (NeMo, 500 each)", ["nemotron_160d4_f16_test-clean_ref500.jsonl", "nemotron_160d4_f16_test-other_ref500.jsonl"], (2.70, 5.37))
    ref_row("Nemotron 560 ms (NeMo, 500 each)", ["nemotron_560_f16_test-clean_ref500.jsonl", "nemotron_560_f16_test-other_ref500.jsonl"], (2.45, 4.83))
    ref_row("Parakeet exact (NeMo, 200 each)", ["parakeet_exact_f16_test-clean_ref200.jsonl", "parakeet_exact_f16_test-other_ref200.jsonl"], (1.81, 4.04))
    ref_row("Voxtral (transformers bf16, 100 each)", ["voxtral_q8h_test-clean_voxref100.jsonl", "voxtral_q8h_test-other_voxref100.jsonl"], (1.72, 5.28))

    # Competitor and precision variants on the first 100 of test-other.
    variants = []
    for label, prefix in [("sherpa-onnx Nemotron 160 ms, fp32", "sherpa-nemotron_160_fp32"), ("sherpa-onnx Nemotron 160 ms, int8", "sherpa-nemotron_160_int8"),
                          ("Voxtral bf16", "voxtral_bf16"), ("Voxtral q8", "voxtral_q8"), ("Voxtral q4ffn", "voxtral_q4ffn"), ("Voxtral q8h", "voxtral_q8h"),
                          ("Nemotron 160 ms before D4", "nemotron_160_f16"), ("Parakeet h0", "parakeet_h0_f16"), ("Parakeet h16", "parakeet_h16_f16")]:
        r = []
        for s in ["test-clean", "test-other"]:
            rows = matched(find_run(prefix, s, ["200", "100"]), keys[s])
            r.append(100 * score([(x["ref"], x["hyp"]) for x in rows])["wer"] if rows else None)
        variants.append((label, r))

    timing = parse_timing()
    longs = long_sessions()
    tm6 = ROOT / "artifacts/timing_m6"
    contention = open(tm6 / "contention.txt").read() if (tm6 / "contention.txt").exists() else ""
    coreml = open(tm6 / "coreml_nemotron.txt").read() if (tm6 / "coreml_nemotron.txt").exists() else ""
    sherpa_speed = {}
    for v in ["int8", "fp32"]:
        rows = load(tm6 / f"sherpa_{v}.jsonl")
        if rows:
            sherpa_speed[v] = sum(r["wall"] for r in rows) / sum(r["duration"] for r in rows)

    data = {"cand": cand, "refcmp": refcmp, "variants": variants, "timing": timing, "longs": longs,
            "contention": contention, "coreml": coreml, "sherpa_speed": sherpa_speed}
    (ROOT / "artifacts/report").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts/report/m6_data.json").write_text(json.dumps(data, indent=1, default=str))

    pareto = [(l, c["col"], c["test-other"]["first_p50_ms"], c["test-other"]["final_p50_ms"], c["test-other"]["wer"])
              for l, c in cand.items() if c["test-other"] and c["test-other"].get("final_p50_ms")]
    load_rows = [(l, cand[l]["col"], *timing[TIMING_NAMES[l]], cand[l]["budget"]) for l in cand if TIMING_NAMES[l] in timing]
    drift = [(l, cand[l]["col"], v["per_min_p50"], cand[l]["budget"]) for l, v in longs.items() if v["per_min_p50"]]
    charts = {"pareto": svg_pareto(pareto) if pareto else "", "load": svg_load(load_rows) if load_rows else "",
              "drift": svg_drift(drift) if drift else ""}
    page = open(ROOT / "eval/report_template.html").read()
    for k, v in {**charts, **tables(data)}.items():
        page = page.replace(f"<!--{k}-->", v)
    OUT.write_text(page)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
