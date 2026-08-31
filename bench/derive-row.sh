#!/usr/bin/env bash
# Derive the experiments.csv row for ONE fresh run directory from its own
# raw logs and end-of-run GPU snapshot. Runtime identity, token-accounting
# endpoint, harness revision, and thermal-watch status are recorded from
# evidence, never assumed.
# Usage: bench/derive-row.sh results/run-<id>
set -euo pipefail
cd "$(dirname "$0")/.."
OUTDIR="${1:?usage: derive-row.sh results/run-<id>}"
[ -f "$OUTDIR/speed-c1.jsonl" ] || { echo "missing $OUTDIR/speed-c1.jsonl"; exit 1; }
[ -f "$OUTDIR/quality.jsonl" ] || { echo "missing $OUTDIR/quality.jsonl"; exit 1; }
[ -f "$OUTDIR/gpu-final.csv" ] || { echo "missing $OUTDIR/gpu-final.csv"; exit 1; }

# Harness identity: prefer the value rebench.sh captured BEFORE any run
# artifact existed; a late local capture is flagged as such.
HARNESS_REVISION="${HARNESS_REVISION:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
if [ -z "${HARNESS_DIRTY:-}" ]; then
  if [ -n "$(git status --porcelain 2>/dev/null | head -1)" ]; then
    HARNESS_DIRTY=dirty-late-capture
  else
    HARNESS_DIRTY=clean-late-capture
  fi
fi

# Runtime fingerprint: hash the exact binary that served the run. A hash is
# identity evidence only; it does NOT by itself verify the pinned source
# revision.
RUNTIME_SHA256=unset
if [ -n "${LLAMA_SERVER:-}" ] && [ -x "$LLAMA_SERVER" ]; then
  RUNTIME_SHA256="$(sha256sum "$LLAMA_SERVER" | cut -d' ' -f1)"
fi

# Per-card configured power limits, queried live rather than assumed.
POWER_LIMITS="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader 2>/dev/null | tr '
' ';' || true)"
POWER_LIMITS="${POWER_LIMITS:-unavailable}"

# Thermal breaches: "none" may only be asserted by the caller (rebench.sh)
# after it confirmed a healthy safety watcher for the whole run window.
THERMAL_BREACHES="${THERMAL_BREACHES:-unverified}"
export HARNESS_REVISION HARNESS_DIRTY RUNTIME_SHA256 POWER_LIMITS THERMAL_BREACHES OUTDIR

python3 - <<'PY'
import csv, json, os, statistics
from pathlib import Path
out = Path(os.environ['OUTDIR'])
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
identity = {
    'harness_revision': os.environ['HARNESS_REVISION'],
    'harness_dirty': os.environ['HARNESS_DIRTY'],
    'runtime_binary_sha256': os.environ['RUNTIME_SHA256'],
    'runtime_binary_sha256_note': 'identity evidence for the serving binary only; does not by itself verify the pinned source revision',
    'pinned_runtime_revision': '00699716c275498ff84d71e329178fe21cba56a6',
    'runtime_revision_verified': False,
    'token_accounting_endpoint': '/v1/chat/completions',
    'power_limits_configured': os.environ['POWER_LIMITS'],
    'thermal_breaches': os.environ['THERMAL_BREACHES'],
    'thermal_breaches_note': 'asserted none only when the caller confirmed a healthy safety watch for the whole run window',
}
with open(out / 'run-identity.json', 'w') as f:
    json.dump(identity, f, indent=2)
'\n'
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
