#!/usr/bin/env bash
# Derive the experiments.csv row AND the full reproducibility run manifest
# for ONE fresh run directory from its own raw logs, the end-of-run GPU
# snapshot, and the actual serve launch environment (same vars/defaults as
# bench/serve.sh). Only written if the required per-phase outputs exist.
# Usage: bench/derive-row.sh results/run-<id>
set -euo pipefail
cd "$(dirname "$0")/.."
OUTDIR="${1:?usage: derive-row.sh results/run-<id>}"
[ -f "$OUTDIR/speed-c1.jsonl" ] || { echo "missing $OUTDIR/speed-c1.jsonl"; exit 1; }
[ -f "$OUTDIR/quality.jsonl" ] || { echo "missing $OUTDIR/quality.jsonl"; exit 1; }
[ -f "$OUTDIR/gpu-final.csv" ] || { echo "missing $OUTDIR/gpu-final.csv"; exit 1; }
python3 - "$OUTDIR" <<'PYHEREDOC'

import csv, json, os, platform, statistics, subprocess, sys
from datetime import datetime, timezone
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
def env(name, default):
    v = os.environ.get(name, '')
    return v if v else default
TSPLIT = env('TSPLIT', '1,1,1,1')
CTX = env('CTX', '16384')
SPLIT = env('SPLIT', 'layer')
PARALLEL = env('PARALLEL', '1')
THREADS = env('THREADS', '8')
FA = env('FLASH_ATTN', 'default(auto)')
CTK = env('CACHE_K', 'default(f16)')
CTV = env('CACHE_V', 'default(f16)')
B = env('BATCH', 'default(2048)')
UB = env('UBATCH', 'default(512)')
MMAP = 'no' if os.environ.get('NO_MMAP', '') else 'yes'
cfg = ('c1;ts=%s;th=%s;par=%s;ctx=%s;split=%s;fa=%s;ctk=%s;ctv=%s;b=%s;ub=%s;mmap=%s'
       % (TSPLIT, THREADS, PARALLEL, CTX, SPLIT, FA, CTK, CTV, B, UB, MMAP))
row = [cfg, str(med), med_or_na('itl_ms_p50'), med_or_na('itl_ms_p95'),
       ';'.join(s['mem'] for s in snap), ';'.join(s['tc'] for s in snap),
       ';'.join(s['tm'] for s in snap), ';'.join(s['pw'] for s in snap),
       str(out / 'quality.jsonl'), str(out / 'speed-c1.jsonl'), 'measured']
with open(out / 'experiments.csv', 'w', newline='') as f:
    wtr = csv.writer(f, lineterminator='\n')
    wtr.writerow(['config', 'median_tok_s', 'itl_p50', 'itl_p95', 'mem_per_card_mib', 'temp_core', 'temp_mem', 'power_w', 'quality_log', 'raw_log', 'decision'])
    wtr.writerow(row)
def sh(cmd, default='not-available'):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return default
gpus = sh(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])
drv = sh(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']).splitlines()[:1]
manifest = {
    'kind': 'rebench-run',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'run_dir': str(out),
    'model': {'file': env('MODEL_FILE', 'GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf'),
              'dir': 'shared model library (path redacted)',
              'revision': 'unsloth/GLM-5.3-Flash-GGUF@2975ab414d30340466d8c51533c6e91f0cca64c1'},
    'runtime': {'server': 'llama-server (unslothai/llama.cpp fork)',
                'revision': '00699716c275498ff84d71e329178fe21cba56a6',
                'build_flags': '-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80'},
    'harness_revision': sh(['git', 'rev-parse', 'HEAD']),
    'harness_dirty': bool(sh(['git', 'status', '--porcelain'], default='')),
    'serve_config': {'ctx': CTX, 'split_mode': SPLIT, 'tensor_split': TSPLIT,
                     'parallel': PARALLEL, 'threads': THREADS,
                     'flash_attn': FA, 'cache_k': CTK, 'cache_v': CTV,
                     'batch': B, 'ubatch': UB, 'mmap': MMAP},
    'host': {'gpus': gpus, 'gpu_count': len(snap),
             'kernel': platform.release(), 'driver': drv,
             'power_limit': 'as-configured; exact per-card limit not queried'},
    'protocol': {'warmups': 3, 'c1_reps': len(speed), 'completion_tokens': 256,
                 'temperature': 0.0, 'seed': 42,
                 'token_accounting': 'final usage object from /v1/completions (bench/measure.py)'},
    'safety': {'core_limit_c': 80, 'mem_limit_c': 85,
               'xid_scan': 'continuous watcher (thermal-guard.sh --watch)',
               'watcher_status': 'active for the entire driver run',
               'breaches': 'none (run fails closed otherwise)'},
    'artifacts': {'warmups': str(out / 'warmups.jsonl'),
                  'speed_c1': str(out / 'speed-c1.jsonl'),
                  'ladder': str(out / 'ladder.jsonl'),
                  'soak': str(out / 'soak.jsonl'),
                  'quality': str(out / 'quality.jsonl'),
                  'experiments_csv': str(out / 'experiments.csv'),
                  'summary_csv': str(out / 'summary.csv'),
                  'summary_json': str(out / 'summary.json'),
                  'charts_dir': str(out / 'charts'),
                  'gpu_snapshot': str(out / 'gpu-final.csv')},
}
with open(out / 'run-manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
    f.write('\n')
print('derived row + run manifest for', out)
PYHEREDOC
