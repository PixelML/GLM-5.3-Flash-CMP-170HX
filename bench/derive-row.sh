#!/usr/bin/env bash
# Derive the experiments.csv row for ONE fresh run directory from its own
# raw logs and end-of-run GPU snapshot. The row is only written if the
# required per-phase outputs exist.
# Usage: bench/derive-row.sh results/run-<id>
set -euo pipefail
cd "$(dirname "$0")/.."
OUTDIR="${1:?usage: derive-row.sh results/run-<id>}"
[ -f "$OUTDIR/speed-c1.jsonl" ] || { echo "missing $OUTDIR/speed-c1.jsonl"; exit 1; }
[ -f "$OUTDIR/quality.jsonl" ] || { echo "missing $OUTDIR/quality.jsonl"; exit 1; }
[ -f "$OUTDIR/gpu-final.csv" ] || { echo "missing $OUTDIR/gpu-final.csv"; exit 1; }
python3 - "$OUTDIR" <<'PY'
import csv, json, statistics, sys
from pathlib import Path
out = Path(sys.argv[1])
speed = [json.loads(l) for l in open(out / 'speed-c1.jsonl') if l.strip().startswith('{') and '"concurrency"' in l]
speed = [r for r in speed if r.get('ok')]
assert speed, 'no ok speed rows'
med = round(statistics.median(r['tok_per_s_per_req'] for r in speed), 2)
def med_or_na(key):
    vals = [r[key] for r in speed if r.get(key) is not None]
    return round(statistics.median(vals), 1) if vals else 'na'
snap = []
for line in open(out / 'gpu-final.csv'):
    parts = [p.strip() for p in line.split(',')]
    if len(parts) >= 6:
        snap.append({'mem': parts[1].split()[0], 'tc': parts[3], 'tm': parts[4], 'pw': parts[5].split()[0]})
assert snap, 'empty gpu snapshot'
cfg = 'phase63-baseline;ts=1;1;1;1;th=8;par=1;ctx=16384;fa=auto;ctk=default;ctv=default;b=2048;ub=512;mmap=yes'
row = [cfg, str(med), med_or_na('itl_ms_p50'), med_or_na('itl_ms_p95'),
       ';'.join(s['mem'] for s in snap), ';'.join(s['tc'] for s in snap),
       ';'.join(s['tm'] for s in snap), ';'.join(s['pw'] for s in snap),
       str(out / 'quality.jsonl'), str(out / 'speed-c1.jsonl'), 'measured']
with open(out / 'experiments.csv', 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['config', 'median_tok_s', 'itl_p50', 'itl_p95', 'mem_per_card_mib', 'temp_core', 'temp_mem', 'power_w', 'quality_log', 'raw_log', 'decision'])
    w.writerow(row)
print('derived row for', out)
PY
