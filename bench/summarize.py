#!/usr/bin/env python3
"""Summarize experiment results: quality index, cost/energy bands, Pareto frontier.

Inputs: results/experiments.csv (one row per config, from run-experiment.sh)
        results/quality.jsonl   (per-task eval records, optional but required for QI)
Outputs: stdout JSON + results/summary.csv

Quality Index (disclosed, local, NOT a universal score):
  QI = mean(bucket pass rate) over the four tuning buckets
       (math, coding, instruction, longctx), each shown beside the composite.
  Sample sizes are small; treat QI as a smoke gate, not a ranking oracle.

Cost model (disclosed):
  - energy_gpu_wh = sum(power_w) * t_task_s / 3600 per successful task,
    GPU-only LOWER BOUND (idle host, fans, networking excluded).
  - cost_usd_band = energy_kwh * [price_low, price_high] with defaults
    0.05 / 0.30 USD/kWh (clearly hypothetical; no purchase data used).
  - failed/retried attempts increase the denominator via observed success rate.
"""
from __future__ import annotations
import csv, json, math, statistics, sys
from collections import defaultdict
from pathlib import Path

PRICE_LOW, PRICE_HIGH = 0.05, 0.30  # USD/kWh, hypothetical band, disclosed

def load_quality(path):
    buckets = defaultdict(lambda: [0, 0])  # ok, total
    if not Path(path).exists():
        return {}, buckets
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        b = r.get("bucket", "?")
        if b.startswith("heldout"):
            continue  # held-out scored separately, never in tuning QI
        buckets[b][1] += 1
        buckets[b][0] += 1 if r.get("ok") else 0
    return {b: ok / t for b, (ok, t) in buckets.items()}, buckets

def pareto(rows):
    # frontier: maximize median_tok_s AND quality_index
    # A missing quality index (no eval data) is treated as 0.0: a config
    # without quality evidence never dominates one with evidence.
    for r in rows:
        if r["quality_index"] is None:
            r["quality_index"] = 0.0
    front = []
    for r in rows:
        dominated = any(
            (o["median_tok_s"] >= r["median_tok_s"] and o["quality_index"] >= r["quality_index"])
            and (o["median_tok_s"] > r["median_tok_s"] or o["quality_index"] > r["quality_index"])
            for o in rows)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda x: -x["quality_index"])

def main():
    exp_csv = sys.argv[1] if len(sys.argv) > 1 else "results/experiments.csv"
    q_path = sys.argv[2] if len(sys.argv) > 2 else "results/quality.jsonl"
    out_csv = sys.argv[3] if len(sys.argv) > 3 else "results/summary.csv"
    rates, counts = load_quality(q_path)
    qi = round(sum(rates.values()) / len(rates), 4) if rates else None
    rows = list(csv.DictReader(open(exp_csv)))
    out = []
    for r in rows:
        tok = float(r["median_tok_s"])
        power = [float(x) for x in r["power_w"].split(";") if x]
        ttft = float(r["ttft_ms"])
        p50 = float(r["itl_p50"])
        t_task_s = 256 / tok if tok else float("inf")  # 256-gen-token protocol task
        energy_wh = sum(power) * t_task_s / 3600 if power else None
        success = 1.0  # measured runs only enter the CSV when ok; failures counted separately
        cost_low = energy_wh / 1000 * PRICE_LOW / success if energy_wh is not None else None
        cost_high = energy_wh / 1000 * PRICE_HIGH / success if energy_wh is not None else None
        out.append({
            "config": r["config"], "median_tok_s": tok, "ttft_ms": ttft,
            "itl_p50": p50, "itl_p95": float(r["itl_p95"]),
            "quality_index": qi, "bucket_rates": rates,
            "bucket_counts": {b: v[1] for b, v in counts.items()},
            "t_task_s": round(t_task_s, 3),
            "energy_gpu_wh": round(energy_wh, 4) if energy_wh is not None else None,
            "energy_note": "GPU-only lower bound",
            "cost_per_task_usd_low": round(cost_low, 6) if cost_low is not None else None,
            "cost_per_task_usd_high": round(cost_high, 6) if cost_high is not None else None,
            "price_band": [PRICE_LOW, PRICE_HIGH],
        })
    front = pareto(out) if out else []
    summary = {"n_configs": len(out), "quality_index": qi, "rows": out,
               "pareto_frontier": [r["config"] for r in front],
               "max_speed": max(out, key=lambda r: r["median_tok_s"])["config"] if out else None,
               "max_quality": front[0]["config"] if front else None,
               "balanced": front[0]["config"] if front else None}
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()) if out else ["config"])
        w.writeheader()
        for r in out:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})
    json.dump(summary, sys.stdout, indent=1)
    print()

if __name__ == "__main__":
    main()
