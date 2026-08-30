#!/usr/bin/env python3
"""Dependency-free SVG charts from results/summary.csv (summarize.py output).
Okabe-Ito palette; every chart labels points, shows n, and marks the Pareto
frontier. Usage: python3 charts.py [summary.csv] [outdir]"""
import csv
import json
import math
import sys
from pathlib import Path

# Okabe-Ito
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "verm": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}

def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;")

def scatter(rows, xkey, ykey, xlabel, ylabel, title, out, logx=False, ymax=None):
    W, H, M = 720, 480, 70
    xs = [r[xkey] for r in rows if r.get(xkey) is not None and r.get(ykey) is not None]
    ys = [r[ykey] for r in rows if r.get(xkey) is not None and r.get(ykey) is not None]
    if not xs:
        return
    xmin, xmax = min(xs), max(xs)
    if logx:
        xmin, xmax = math.log10(max(xmin, 1e-9)), math.log10(max(xmax, 1e-8))
    ymin, ymax = 0, ymax or max(ys) * 1.1
    def px(v): return M + (logx and (math.log10(max(v,1e-9))-xmin) or (v-xmin)) / max(xmax-xmin, 1e-12) * (W-M-30)
    def py(v): return H-M - v / max(ymax-ymin, 1e-12) * (H-M-40) - 10
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" role="img" aria-label="{esc(title)}">',
             f'<title>{esc(title)}</title>', '<rect width="100%" height="100%" fill="white"/>']
    for gv in range(6):
        gy = py(ymax*gv/5)
        parts.append(f'<line x1="{M}" y1="{gy:.0f}" x2="{W-20}" y2="{gy:.0f}" stroke="#ddd"/>'
                     f'<text x="{M-8}" y="{gy+4:.0f}" font-size="11" text-anchor="end" fill="#444">{ymax*gv/5:.3g}</text>')
    parts.append(f'<line x1="{M}" y1="{H-M}" x2="{W-20}" y2="{H-M}" stroke="#333"/>')
    parts.append(f'<text x="{(W)/2}" y="{H-18}" font-size="13" text-anchor="middle">{esc(xlabel)}{" (log)" if logx else ""}</text>')
    parts.append(f'<text x="16" y="{H/2}" font-size="13" transform="rotate(-90 16 {H/2})" text-anchor="middle">{esc(ylabel)}</text>')
    for i, r in enumerate(rows):
        if r.get(xkey) is None or r.get(ykey) is None: continue
        col = C["verm"] if r.get("pareto") else C["blue"]
        parts.append(f'<circle cx="{px(r[xkey]):.1f}" cy="{py(r[ykey]):.1f}" r="5" fill="{col}" fill-opacity="0.85"><title>{esc(r["config"])}: {esc(r[xkey])} / {esc(r[ykey])}</title></circle>')
        parts.append(f'<text x="{px(r[xkey]):.1f}" y="{py(r[ykey])-10:.1f}" font-size="9" text-anchor="middle" fill="#333">{esc(str(i+1))}</text>')
    parts.append(f'<text x="{W-24}" y="24" font-size="12" text-anchor="end" fill="{C["verm"]}">Pareto frontier</text>')
    parts.append(f'<circle cx="{W-14}" cy="20" r="4" fill="{C["verm"]}"/>')
    parts.append('</svg>')
    Path(out).write_text("\n".join(parts))

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results/summary.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "results/charts"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    rows = []
    with open(src, newline="") as fh:
        for r in csv.DictReader(fh):
            for k in ("median_tok_s", "ttft_ms", "quality_index", "energy_gpu_wh",
                      "cost_per_task_usd_low", "t_task_s"):
                r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
            rows.append(r)
    # summary.csv is row-per-config; frontier membership is recomputed here from QI/tok
    for r in rows:
        r["pareto"] = True  # recompute below
    def dominated(r):
        return any((o["median_tok_s"] >= r["median_tok_s"] and o["quality_index"] >= r["quality_index"])
                   and (o["median_tok_s"] > r["median_tok_s"] or o["quality_index"] > r["quality_index"]) for o in rows)
    for r in rows:
        r["pareto"] = not dominated(r)
    n = len(rows)
    scatter(rows, "cost_per_task_usd_low", "quality_index", "GPU-energy cost per successful task (USD, low band)", "Quality Index", f"Quality vs cost per successful task (n={n}, GPU-only energy)", f"{outdir}/qi-vs-cost.svg", logx=True)
    scatter(rows, "t_task_s", "quality_index", "Median time per successful task (s)", "Quality Index", f"Quality vs median time per task (n={n})", f"{outdir}/qi-vs-time.svg")
    scatter(rows, "median_tok_s", "quality_index", "Median decode tok/s", "Quality Index", f"Quality vs aggregate throughput (n={n})", f"{outdir}/qi-vs-tokps.svg")
    scatter(rows, "energy_gpu_wh", "quality_index", "GPU energy per successful task (Wh, lower bound)", "Quality Index", f"Quality vs GPU energy efficiency (n={n})", f"{outdir}/qi-vs-energy.svg")

    # Throughput/latency vs concurrency, from raw speed.jsonl (measure.py sweep
    # rows). Optional: chart is skipped when the file is absent.
    speed_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/speed.jsonl")
    conc_rows = []
    if speed_path.exists():
        by_conc = {}
        with open(speed_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c = rec.get("concurrency")
                if c is None or not rec.get("ok", True):
                    continue
                by_conc.setdefault(c, []).append(rec)
        for c, recs in sorted(by_conc.items()):
            agg = [r["tok_per_s_aggregate"] for r in recs if r.get("tok_per_s_aggregate")]
            per = [r["tok_per_s_per_req"] for r in recs if r.get("tok_per_s_per_req")]
            ttfts = [r["ttft_s"] for r in recs if r.get("ttft_s")]
            if not agg:
                continue
            conc_rows.append({
                "concurrency": c, "n": len(recs),
                "agg_med": sorted(agg)[len(agg)//2],
                "per_med": sorted(per)[len(per)//2] if per else None,
                "ttft_med_ms": round(1000*sorted(ttfts)[len(ttfts)//2], 1) if ttfts else None,
            })
    if conc_rows:
        W, H, M = 720, 480, 70
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" role="img" aria-label="throughput vs concurrency">' % (W, H),
                 '<title>Throughput and per-stream rate vs concurrency</title>',
                 '<rect width="100%" height="100%" fill="white"/>']
        xmax = max(r["concurrency"] for r in conc_rows)
        ymax = max(r["agg_med"] for r in conc_rows) * 1.15
        def cx(v): return M + (v - 1) / max(xmax - 1, 1) * (W - M - 40)
        def cy(v): return H - M - v / ymax * (H - M - 40)
        for gv in range(6):
            gy = cy(ymax*gv/5)
            parts.append(f'<line x1="{M}" y1="{gy:.0f}" x2="{W-20}" y2="{gy:.0f}" stroke="#ddd"/>'
                         f'<text x="{M-8}" y="{gy+4:.0f}" font-size="11" text-anchor="end" fill="#444">{ymax*gv/5:.3g}</text>')
        parts.append(f'<line x1="{M}" y1="{H-M}" x2="{W-20}" y2="{H-M}" stroke="#333"/>')
        parts.append(f'<text x="{W/2}" y="{H-18}" font-size="13" text-anchor="middle">concurrency (parallel requests)</text>')
        parts.append(f'<text x="16" y="{H/2}" font-size="13" transform="rotate(-90 16 {H/2})" text-anchor="middle">tok/s</text>')
        agg_pts = " ".join(f"{cx(r['concurrency']):.1f},{cy(r['agg_med']):.1f}" for r in conc_rows)
        parts.append(f'<polyline points="{agg_pts}" fill="none" stroke="{C["blue"]}" stroke-width="2"/>')
        per_pts = " ".join(f"{cx(r['concurrency']):.1f},{cy(r['per_med']):.1f}" for r in conc_rows if r["per_med"])
        if per_pts:
            parts.append(f'<polyline points="{per_pts}" fill="none" stroke="{C["orange"]}" stroke-width="2" stroke-dasharray="5,3"/>')
        for r in conc_rows:
            parts.append(f'<circle cx="{cx(r["concurrency"]):.1f}" cy="{cy(r["agg_med"]):.1f}" r="5" fill="{C["blue"]}"><title>aggregate: {r["agg_med"]:.1f} tok/s (n={r["n"]} runs)</title></circle>')
            if r["per_med"]:
                parts.append(f'<circle cx="{cx(r["concurrency"]):.1f}" cy="{cy(r["per_med"]):.1f}" r="4" fill="{C["orange"]}"><title>per-stream: {r["per_med"]:.1f} tok/s</title></circle>')
        parts.append(f'<text x="{W-150}" y="22" font-size="12" fill="{C["blue"]}">aggregate tok/s</text>')
        parts.append(f'<text x="{W-150}" y="38" font-size="12" fill="{C["orange"]}">per-stream tok/s</text>')
        parts.append('</svg>')
        Path(f"{outdir}/conc-vs-throughput.svg").write_text("\n".join(parts))
        print(f"concurrency chart: n_conc={len(conc_rows)} from {speed_path}")
    else:
        print("concurrency chart: skipped (no results/speed.jsonl)")
    print(f"wrote charts to {outdir} (configs n={n})")

if __name__ == "__main__":
    main()
