#!/usr/bin/env bash
# Offline gate: verify a run directory's result rows all succeeded and the
# protocol contract was actually met before publication-shaped artifacts
# are derived. Exit 1 on any failed request, short ladder, or short soak.
# Usage: REBENCH_OUTDIR=results/run-<id> bash bench/verify-rows.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${REBENCH_OUTDIR:?set REBENCH_OUTDIR to the run directory}"
[ -f "$OUT/ladder.jsonl" ] || { echo "missing $OUT/ladder.jsonl"; exit 1; }
[ -f "$OUT/soak.jsonl" ] || { echo "missing $OUT/soak.jsonl"; exit 1; }

python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
def rows(name):
    with open(out + "/" + name) as f:
        return [json.loads(l) for l in f if l.strip().startswith("{")]

failed = 0
ladder = [r for r in rows("ladder.jsonl") if r.get("ok") is not None]
levels = sorted({r["concurrency"] for r in ladder})
reps = {}
for r in ladder:
    reps.setdefault(r["concurrency"], []).append(r["ok"])
if levels != [1, 2, 4]:
    print(f"FAIL: ladder levels {levels} != [1, 2, 4]"); failed += 1
for lv, oks in sorted(reps.items()):
    if len(oks) < 2:
        print(f"FAIL: ladder c={lv} has {len(oks)} reps < 2"); failed += 1
    if not all(oks):
        print(f"FAIL: ladder c={lv} contains failed request batches"); failed += 1
soak = rows("soak.jsonl")
if not soak:
    print("FAIL: soak produced no rows"); failed += 1
bad_soak = [r for r in soak if not r.get("ok")]
if bad_soak:
    print(f"FAIL: {len(bad_soak)} soak rows with failed requests"); failed += 1
speed = [r for r in rows("speed-c1.jsonl") if r.get("ok") is not None]
bad_speed = [r for r in speed if not r.get("ok")]
if bad_speed:
    print(f"FAIL: {len(bad_speed)} speed rows with failed requests"); failed += 1
counts = {
    "speed_rows": len(speed),
    "speed_failed": len(bad_speed),
    "ladder_levels": levels,
    "ladder_rows": len(ladder),
    "ladder_failed_rows": sum(1 for r in ladder if not r.get("ok")),
    "soak_rows": len(soak),
    "soak_failed_rows": len(bad_soak),
}
with open(out + "/actual-counts.json", "w") as f:
    json.dump(counts, f, indent=2); f.write("\n")
if failed:
    sys.exit(1)
print("row verification PASS:", counts)
PY
